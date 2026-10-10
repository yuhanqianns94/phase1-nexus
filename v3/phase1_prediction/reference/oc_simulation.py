"""Operating-characteristic simulation for Phase I dose-escalation designs.

The module deliberately separates two sources of randomness:

* a ``TruthScenario`` is one fixed dose--DLT probability vector;
* repeated virtual trials generate Binomial DLT outcomes conditional on that
  fixed truth.

Posterior uncertainty belongs in an outer collection of truth scenarios.  A
truth curve is never redrawn for each patient or cohort.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SUPPORTED_DESIGNS = ("crm", "boin", "three_plus_three")
SUPPORTED_MTD_DEFINITIONS = (
    "acceptable_interval",
    "closest_to_target",
    "highest_not_above_target",
)
MAX_TOTAL_VIRTUAL_TRIALS = 250_000


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _strictly_increasing_positive(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    resolved = tuple(_finite_float(value, label=label) for value in values)
    if len(resolved) < 2:
        raise ValueError(f"{label} must contain at least two values.")
    if any(value <= 0 for value in resolved):
        raise ValueError(f"{label} values must all be positive.")
    if any(right <= left for left, right in zip(resolved, resolved[1:])):
        raise ValueError(f"{label} must be strictly increasing.")
    return resolved


@dataclass(frozen=True)
class OCConfig:
    """Validated settings shared by every simulated design."""

    doses: tuple[float, ...]
    target_dlt: float = 0.25
    acceptable_low: float = 0.20
    acceptable_high: float = 0.35
    start_index: int = 0
    cohort_size: int = 3
    max_sample_size: int = 30
    designs: tuple[str, ...] = SUPPORTED_DESIGNS
    trials_per_scenario: int = 100
    seed: int = 9
    mtd_definition: str = "acceptable_interval"
    crm_skeleton: tuple[float, ...] | None = None
    crm_prior_mean: float = 0.0
    crm_prior_sd: float = 1.34
    crm_max_step: int = 1
    crm_ewoc_threshold: float = 1.0
    crm_safety_stop_probability: float = 0.95
    crm_allow_dose_skipping: bool = False
    boin_lambda_e: float | None = None
    boin_lambda_d: float | None = None
    boin_phi1: float | None = None
    boin_phi2: float | None = None
    boin_elimination_cutoff: float = 0.95
    boin_elimination_min_n: int = 1
    boin_stop_rule_enabled: bool = True
    boin_stop_rule_m: int = 12
    boin_modify_1_3_deesc_to_stay: bool = False
    boin_check_lowest_dose: bool = True
    boin_lowest_dose_delta: float = 0.03
    boin_check_final_mtd_deesc_bound: bool = True
    boin_use_isotonic: bool = True
    three_plus_three_highest_safe_policy: str = "select_highest"

    def validated(self) -> "OCConfig":
        doses = _strictly_increasing_positive(self.doses, label="doses")
        designs = tuple(dict.fromkeys(str(value) for value in self.designs))
        if not designs or any(value not in SUPPORTED_DESIGNS for value in designs):
            raise ValueError(
                "designs must contain one or more of: " + ", ".join(SUPPORTED_DESIGNS)
            )
        if self.boin_elimination_min_n != 1:
            raise ValueError("v3 safety checks start from the first cohort; elimination_min_n must be 1.")
        target = _finite_float(self.target_dlt, label="target_dlt")
        low = _finite_float(self.acceptable_low, label="acceptable_low")
        high = _finite_float(self.acceptable_high, label="acceptable_high")
        if not 0 < target < 1:
            raise ValueError("target_dlt must lie strictly between 0 and 1.")
        if not 0 <= low <= target <= high <= 1:
            raise ValueError(
                "acceptable interval must satisfy 0 <= low <= target <= high <= 1."
            )
        if isinstance(self.start_index, bool) or not isinstance(self.start_index, int):
            raise ValueError("start_index must be an integer.")
        if not 0 <= self.start_index < len(doses):
            raise ValueError("start_index is outside the dose list.")
        if (
            isinstance(self.cohort_size, bool)
            or not isinstance(self.cohort_size, int)
            or self.cohort_size <= 0
        ):
            raise ValueError("cohort_size must be a positive integer.")
        if (
            isinstance(self.max_sample_size, bool)
            or not isinstance(self.max_sample_size, int)
            or self.max_sample_size <= 0
        ):
            raise ValueError("max_sample_size must be a positive integer.")
        if (
            any(design in designs for design in ("crm", "boin"))
            and self.max_sample_size < self.cohort_size
        ):
            raise ValueError(
                "max_sample_size must be at least one CRM/BOIN cohort."
            )
        if (
            isinstance(self.trials_per_scenario, bool)
            or not isinstance(self.trials_per_scenario, int)
            or self.trials_per_scenario <= 0
        ):
            raise ValueError("trials_per_scenario must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if "crm" in designs and self.max_sample_size % self.cohort_size != 0:
            raise ValueError(
                "CRM requires max_sample_size to be exactly divisible by cohort_size."
            )
        if "three_plus_three" in designs and (
            self.max_sample_size < 3 or self.max_sample_size % 3 != 0
        ):
            raise ValueError(
                "3+3 requires max_sample_size to be a positive multiple of 3."
            )
        if self.mtd_definition not in SUPPORTED_MTD_DEFINITIONS:
            raise ValueError(
                "mtd_definition must be one of: "
                + ", ".join(SUPPORTED_MTD_DEFINITIONS)
            )
        skeleton = self.crm_skeleton
        if "crm" in designs:
            if skeleton is not None:
                skeleton = _strictly_increasing_positive(
                    skeleton, label="crm_skeleton"
                )
                if len(skeleton) != len(doses) or any(value >= 1 for value in skeleton):
                    raise ValueError(
                        "crm_skeleton must have one strictly increasing probability "
                        "in (0, 1) for every dose."
                    )
            if _finite_float(self.crm_prior_sd, label="crm_prior_sd") <= 0:
                raise ValueError("crm_prior_sd must be > 0.")
            if (
                isinstance(self.crm_max_step, bool)
                or not isinstance(self.crm_max_step, int)
                or self.crm_max_step <= 0
            ):
                raise ValueError("crm_max_step must be a positive integer.")
            for label, value in (
                ("crm_ewoc_threshold", self.crm_ewoc_threshold),
                ("crm_safety_stop_probability", self.crm_safety_stop_probability),
            ):
                numeric = _finite_float(value, label=label)
                if not 0 < numeric <= 1:
                    raise ValueError(f"{label} must lie in (0, 1].")
        if "boin" in designs:
            if self.max_sample_size % self.cohort_size != 0:
                raise ValueError(
                    "BOIN requires max_sample_size to be exactly divisible by cohort_size."
                )
            cutoff = _finite_float(
                self.boin_elimination_cutoff, label="boin_elimination_cutoff"
            )
            if not 0 < cutoff <= 1:
                raise ValueError("boin_elimination_cutoff must lie in (0, 1].")
            if (
                self.boin_lambda_e is None
                and self.boin_lambda_d is None
                and self.boin_phi1 is None
                and self.boin_phi2 is None
                and 1.4 * target >= 1
            ):
                raise ValueError(
                    "Automatic BOIN boundaries require target_dlt < 1/1.4 "
                    "(approximately 0.714); otherwise enter manual boundaries."
                )
            if self.boin_lambda_e is not None and not 0 < float(self.boin_lambda_e) < target:
                raise ValueError("boin_lambda_e must lie between 0 and target_dlt.")
            if self.boin_lambda_d is not None and not target < float(self.boin_lambda_d) < 1:
                raise ValueError("boin_lambda_d must lie between target_dlt and 1.")
            if (self.boin_lambda_e is None) != (self.boin_lambda_d is None):
                raise ValueError(
                    "Enter both BOIN boundaries or leave both blank for automatic values."
                )
            if (self.boin_phi1 is None) != (self.boin_phi2 is None):
                raise ValueError(
                    "Enter both BOIN phi values or leave both blank for automatic values."
                )
            if self.boin_phi1 is not None:
                phi1 = _finite_float(self.boin_phi1, label="boin_phi1")
                phi2 = _finite_float(self.boin_phi2, label="boin_phi2")
                if not 0 < phi1 < target < phi2 < 1:
                    raise ValueError(
                        "BOIN phi values must satisfy 0 < phi1 < target_dlt < phi2 < 1."
                    )
                if self.boin_lambda_e is not None:
                    raise ValueError(
                        "Enter BOIN phi values or direct lambda boundaries, not both."
                    )
            if self.boin_lambda_e is not None and self.boin_lambda_d is not None:
                if float(self.boin_lambda_e) >= float(self.boin_lambda_d):
                    raise ValueError("boin_lambda_e must be smaller than boin_lambda_d.")
            if (
                isinstance(self.boin_elimination_min_n, bool)
                or not isinstance(self.boin_elimination_min_n, int)
                or self.boin_elimination_min_n <= 0
            ):
                raise ValueError("boin_elimination_min_n must be a positive integer.")
            if (
                isinstance(self.boin_stop_rule_m, bool)
                or not isinstance(self.boin_stop_rule_m, int)
                or self.boin_stop_rule_m < self.cohort_size
            ):
                raise ValueError("boin_stop_rule_m must be at least cohort_size.")
            for label, value in (
                ("boin_stop_rule_enabled", self.boin_stop_rule_enabled),
                (
                    "boin_modify_1_3_deesc_to_stay",
                    self.boin_modify_1_3_deesc_to_stay,
                ),
                ("boin_check_lowest_dose", self.boin_check_lowest_dose),
                (
                    "boin_check_final_mtd_deesc_bound",
                    self.boin_check_final_mtd_deesc_bound,
                ),
                ("boin_use_isotonic", self.boin_use_isotonic),
            ):
                if not isinstance(value, bool):
                    raise ValueError(f"{label} must be a boolean.")
            lowest_delta = _finite_float(
                self.boin_lowest_dose_delta, label="boin_lowest_dose_delta"
            )
            if lowest_delta < 0 or (
                self.boin_check_lowest_dose and cutoff - lowest_delta <= 0
            ):
                raise ValueError(
                    "boin_lowest_dose_delta must be non-negative and smaller than "
                    "the elimination cutoff when the lowest-dose check is enabled."
                )
        if "three_plus_three" in designs and (
            self.three_plus_three_highest_safe_policy
            not in {"select_highest", "expand_to_six"}
        ):
            raise ValueError(
                "three_plus_three_highest_safe_policy must be 'select_highest' "
                "or 'expand_to_six'."
            )
        return OCConfig(
            **{
                **asdict(self),
                "doses": doses,
                "designs": designs,
                "crm_skeleton": skeleton,
            }
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self.validated())
        payload["doses"] = list(payload["doses"])
        payload["designs"] = list(payload["designs"])
        if payload["crm_skeleton"] is not None:
            payload["crm_skeleton"] = list(payload["crm_skeleton"])
        return payload


@dataclass(frozen=True)
class TruthScenario:
    """One fixed data-generating dose--DLT curve."""

    name: str
    probabilities: tuple[float, ...]
    source: str
    raw_probabilities: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validated(self, n_dose: int) -> "TruthScenario":
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Every truth scenario requires a non-empty name.")
        values = tuple(
            _finite_float(value, label=f"scenario {self.name!r} probability")
            for value in self.probabilities
        )
        if len(values) != n_dose:
            raise ValueError(
                f"Scenario {self.name!r} has {len(values)} probabilities; expected {n_dose}."
            )
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("True DLT probabilities must lie between 0 and 1.")
        if any(right < left for left, right in zip(values, values[1:])):
            raise ValueError(
                f"Scenario {self.name!r} is not non-decreasing across doses."
            )
        raw = self.raw_probabilities
        if raw is not None and len(raw) != n_dose:
            raise ValueError("raw_probabilities must match the dose count.")
        return TruthScenario(
            name=self.name.strip(),
            probabilities=values,
            source=str(self.source),
            raw_probabilities=(tuple(float(value) for value in raw) if raw else None),
            metadata=dict(self.metadata),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "probabilities": list(self.probabilities),
            "source": self.source,
            "raw_probabilities": (
                list(self.raw_probabilities)
                if self.raw_probabilities is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


def isotonic_nondecreasing(values: Sequence[float]) -> np.ndarray:
    """Least-squares non-decreasing projection using pooled adjacent violators."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite vector.")
    levels: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, value in enumerate(array):
        levels.append(float(value))
        weights.append(1.0)
        starts.append(index)
        ends.append(index)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            combined_weight = weights[-2] + weights[-1]
            combined_level = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / combined_weight
            levels[-2:] = [combined_level]
            weights[-2:] = [combined_weight]
            ends[-2:] = [ends[-1]]
            starts.pop()
    projected = np.empty_like(array)
    for level, start, end in zip(levels, starts, ends, strict=True):
        projected[start : end + 1] = level
    return np.clip(projected, 0.0, 1.0)


def monotonic_truth_scenario(
    name: str,
    probabilities: Sequence[float],
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> TruthScenario:
    """Project one raw curve and retain an explicit audit of every change."""

    raw = np.asarray(probabilities, dtype=float)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("probabilities must be a non-empty finite vector.")
    if np.any((raw < 0) | (raw > 1)):
        raise ValueError("probabilities must lie between 0 and 1.")
    adjusted = isotonic_nondecreasing(raw)
    difference = adjusted - raw
    audit = {
        **(metadata or {}),
        "monotonicity_method": "least_squares_pava",
        "monotonicity_changed": bool(np.any(np.abs(difference) > 1e-12)),
        "maximum_absolute_adjustment": float(np.max(np.abs(difference))),
        "mean_absolute_adjustment": float(np.mean(np.abs(difference))),
    }
    return TruthScenario(
        name=name,
        probabilities=tuple(float(value) for value in adjusted),
        raw_probabilities=tuple(float(value) for value in raw),
        source=source,
        metadata=audit,
    )


def true_mtd_index(probabilities: Sequence[float], config: OCConfig) -> int | None:
    """Resolve the scenario's correct decision under the configured definition."""

    values = np.asarray(probabilities, dtype=float)
    if config.mtd_definition == "closest_to_target":
        return int(np.argmin(np.abs(values - config.target_dlt)))
    if config.mtd_definition == "highest_not_above_target":
        eligible = np.flatnonzero(values <= config.target_dlt)
        return int(eligible[-1]) if len(eligible) else None
    if np.all(values < config.acceptable_low):
        # The MTD was not reached inside the tested grid; the highest tested
        # dose is the correct finite-grid recommendation, not "no selection".
        return len(values) - 1
    if np.all(values > config.acceptable_high):
        return None
    eligible = np.flatnonzero(
        (values >= config.acceptable_low) & (values <= config.acceptable_high)
    )
    if not len(eligible):
        return None
    distances = np.abs(values[eligible] - config.target_dlt)
    return int(eligible[int(np.argmin(distances))])


def truth_target_status(probabilities: Sequence[float], config: OCConfig) -> str:
    """Classify where the acceptable toxicity target lies relative to the grid."""

    values = np.asarray(probabilities, dtype=float)
    if config.mtd_definition == "closest_to_target":
        return "closest_to_target_dose_available"
    if config.mtd_definition == "highest_not_above_target":
        return (
            "one_or_more_doses_not_above_target"
            if np.any(values <= config.target_dlt)
            else "all_doses_above_target_no_safe_dose"
        )
    if np.all(values < config.acceptable_low):
        return "all_below_acceptable_range_mtd_not_reached"
    if np.all(values > config.acceptable_high):
        return "all_above_acceptable_range_no_safe_dose"
    if np.any(
        (values >= config.acceptable_low) & (values <= config.acceptable_high)
    ):
        return "one_or_more_doses_in_acceptable_range"
    return "target_between_tested_dose_levels"


def build_gpt_wrapper_standard_scenarios(
    doses: Sequence[float],
    *,
    target_dlt: float,
    p_floor: float = 0.02,
    p_cap: float | None = None,
) -> list[TruthScenario]:
    """Build the GPT-wrapper web K-by-K diagonal-target OC matrix."""

    resolved_doses = _strictly_increasing_positive(doses, label="doses")
    dose_count = len(resolved_doses)
    target = _finite_float(target_dlt, label="target_dlt")
    if not 0.0 < target < 1.0:
        raise ValueError("target_dlt must lie in (0, 1).")
    cap = float(np.clip(0.65 if p_cap is None else p_cap, target + 0.03, 0.95))
    floor = float(np.clip(p_floor, 1e-6, target - 0.02))
    epsilon_below = float(
        np.clip(
            0.13 * (target / 0.25),
            0.06,
            min(target - floor - 0.01, 0.20),
        )
    )
    jump_above = float(
        np.clip(
            0.15 * (target / 0.25) ** 0.5,
            0.07,
            min(0.30, (1.0 - target) * 0.85),
        )
    )
    matrix = np.zeros((dose_count, dose_count), dtype=float)
    for row in range(dose_count):
        left_count = row
        right_count = dose_count - 1 - row
        if left_count > 0:
            high_left = target - epsilon_below * (
                1.0 + 0.04 * max(0, left_count - 2)
            )
            high_left = max(high_left, floor + (target - floor) * 0.15)
            span = max(high_left - floor, 0.05)
            low_left = max(
                floor,
                high_left
                - span
                * (
                    0.35
                    + 0.20
                    * (dose_count - 1 - row)
                    / max(1, dose_count - 1)
                ),
            )
            if low_left >= high_left - (left_count - 1) * 1e-4:
                low_left = max(floor, high_left - 0.08 * left_count)
            matrix[row, :left_count] = np.linspace(
                low_left, high_left, left_count
            )
        matrix[row, row] = target
        if right_count > 0:
            first_right = target + jump_above
            stretch = (cap - first_right) * (
                0.55 + 0.35 * (1.0 - right_count / max(1, dose_count - 1))
            )
            stretch += 0.06 * (row / max(1, dose_count - 1))
            last_right = float(
                np.clip(
                    first_right + stretch,
                    first_right + 0.02 * max(1, right_count - 1),
                    cap,
                )
            )
            matrix[row, row + 1 :] = np.linspace(
                first_right, last_right, right_count
            )
    for row in range(dose_count):
        for column in range(1, dose_count):
            if matrix[row, column] <= matrix[row, column - 1]:
                matrix[row, column] = matrix[row, column - 1] + 1e-4
        matrix[row] = np.clip(matrix[row], floor, cap)
        matrix[row, row] = target
    return [
        TruthScenario(
            name=f"gpt_wrapper_target_at_dose_{row + 1}",
            probabilities=tuple(float(value) for value in matrix[row]),
            source="gpt_wrapper_standard_diagonal_target_matrix",
            metadata={"reference_true_mtd_index": row},
        )
        for row in range(dose_count)
    ]


def _expit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def build_stress_scenarios(
    doses: Sequence[float],
    posterior_mean: Sequence[float],
    *,
    target_dlt: float,
) -> list[TruthScenario]:
    """Create transparent safe/toxic, shifted, slope, and shape stress curves."""

    resolved_doses = np.asarray(
        _strictly_increasing_positive(doses, label="doses"), dtype=float
    )
    mean = isotonic_nondecreasing(posterior_mean)
    if len(mean) != len(resolved_doses):
        raise ValueError("posterior_mean must contain one probability per dose.")
    k = len(mean)
    logits = _logit(mean)
    center = float(np.mean(logits))
    pivot = max(1, k // 2)
    all_safe = np.linspace(max(0.01, target_dlt * 0.15), target_dlt * 0.65, k)
    all_toxic = np.linspace(
        min(0.95, target_dlt + 0.15), min(0.98, target_dlt + 0.45), k
    )
    plateau = mean.copy()
    plateau[pivot:] = plateau[pivot]
    sudden = np.concatenate(
        [
            np.linspace(max(0.01, target_dlt * 0.20), target_dlt * 0.65, pivot),
            np.linspace(min(0.95, target_dlt + 0.22), min(0.98, target_dlt + 0.48), k - pivot),
        ]
    )
    raw_scenarios: list[tuple[str, np.ndarray, str]] = [
        ("all_safe", all_safe, "stress_all_doses_below_target"),
        ("all_toxic", all_toxic, "stress_all_doses_above_target"),
        ("historical_safer", _expit(logits - 0.55), "stress_log_odds_shift_minus_0.55"),
        ("historical_more_toxic", _expit(logits + 0.55), "stress_log_odds_shift_plus_0.55"),
        ("historical_flatter", _expit(center + 0.45 * (logits - center)), "stress_slope_multiplier_0.45"),
        ("historical_steeper", _expit(center + 1.75 * (logits - center)), "stress_slope_multiplier_1.75"),
        ("plateau", plateau, "stress_non_logistic_plateau"),
        ("sudden_rise", sudden, "stress_non_logistic_sudden_rise"),
    ]
    for index in sorted({0, k // 2, k - 1}):
        log_doses = np.log(resolved_doses)
        positive_gaps = np.diff(log_doses)
        gap_scale = float(np.median(positive_gaps)) if len(positive_gaps) else 1.0
        offsets = (log_doses - log_doses[index]) / max(gap_scale, 1e-8)
        raw_scenarios.append(
            (
                f"target_at_dose_{index + 1}",
                _expit(float(_logit(np.asarray([target_dlt]))[0]) + 0.75 * offsets),
                "stress_target_location",
            )
        )
    return [
        monotonic_truth_scenario(name, values, source=source)
        for name, values, source in raw_scenarios
    ]


def _simulate_one_trial(
    design: str,
    probabilities: tuple[float, ...],
    config: OCConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if design == "crm":
        from .design_engines.crm import simulate_crm_trial

        return simulate_crm_trial(
            probabilities,
            start_index=config.start_index,
            cohort_size=config.cohort_size,
            max_sample_size=config.max_sample_size,
            target_dlt=config.target_dlt,
            rng=rng,
            skeleton=(list(config.crm_skeleton) if config.crm_skeleton else None),
            prior_mean=config.crm_prior_mean,
            prior_sd=config.crm_prior_sd,
            max_step=config.crm_max_step,
            allow_dose_skipping=config.crm_allow_dose_skipping,
            ewoc_threshold=config.crm_ewoc_threshold,
            safety_stop_probability=config.crm_safety_stop_probability,
        )
    if design == "boin":
        from phase1_simulation.boin import simulate_boin_trial

        return simulate_boin_trial(
            probabilities,
            start_index=config.start_index,
            cohort_size=config.cohort_size,
            max_sample_size=config.max_sample_size,
            target_dlt=config.target_dlt,
            rng=rng,
            lambda_e=config.boin_lambda_e,
            lambda_d=config.boin_lambda_d,
            phi1=config.boin_phi1,
            phi2=config.boin_phi2,
            elimination_cutoff=config.boin_elimination_cutoff,
            stop_rule_enabled=config.boin_stop_rule_enabled,
            stop_rule_m=config.boin_stop_rule_m,
            modify_1_3_deesc_to_stay=config.boin_modify_1_3_deesc_to_stay,
            check_lowest_dose=config.boin_check_lowest_dose,
            lowest_dose_delta=config.boin_lowest_dose_delta,
            check_final_mtd_deesc_bound=(
                config.boin_check_final_mtd_deesc_bound
            ),
            use_isotonic=config.boin_use_isotonic,
        )
    if design == "three_plus_three":
        from .design_engines.three_plus_three import simulate_three_plus_three_trial

        return simulate_three_plus_three_trial(
            probabilities,
            start_index=config.start_index,
            max_sample_size=config.max_sample_size,
            target_dlt=config.target_dlt,
            rng=rng,
            highest_safe_policy=config.three_plus_three_highest_safe_policy,
        )
    raise ValueError(f"Unsupported design: {design!r}")


def _proportion_summary(successes: int, total: int) -> dict[str, float]:
    probability = successes / total if total else float("nan")
    standard_error = math.sqrt(probability * (1 - probability) / total) if total else float("nan")
    if total:
        z = 1.96
        denominator = 1 + z**2 / total
        center = (probability + z**2 / (2 * total)) / denominator
        half_width = (
            z
            * math.sqrt(
                probability * (1 - probability) / total
                + z**2 / (4 * total**2)
            )
            / denominator
        )
        interval_low = max(0.0, center - half_width)
        interval_high = min(1.0, center + half_width)
    else:
        interval_low = float("nan")
        interval_high = float("nan")
    return {
        "probability": float(probability),
        "monte_carlo_standard_error": float(standard_error),
        "confidence_interval_method": "wilson_score",
        "confidence_interval_95_low": float(interval_low),
        "confidence_interval_95_high": float(interval_high),
    }


def _termination_category(outcome: Mapping[str, Any]) -> str:
    reason = str(outcome.get("stop_reason") or "")
    selected = outcome.get("selected_index")
    if any(token in reason for token in ("overtox", "over_toxic", "eliminated")):
        return "safety_stop" if selected is None else "natural_design_completion"
    if reason == "max_sample_size_reached":
        return (
            "maximum_sample_no_selection"
            if selected is None
            else "maximum_sample_reached"
        )
    if selected is None:
        return "no_admissible_dose"
    return "natural_design_completion"


def summarize_trial_outcomes(
    outcomes: Sequence[dict[str, Any]],
    probabilities: Sequence[float] | None,
    config: OCConfig,
    *,
    design: str | None = None,
) -> dict[str, Any]:
    """Summarize selection, safety, allocation, efficiency, and MC error."""

    if not outcomes:
        raise ValueError("At least one trial outcome is required.")
    k = len(config.doses)
    selection_counts = np.zeros(k, dtype=int)
    treated = np.zeros(k, dtype=float)
    dlts = np.zeros(k, dtype=float)
    none_count = 0
    early_stop_count = 0
    total_n: list[float] = []
    allocation_fractions: list[np.ndarray] = []
    total_dlt: list[float] = []
    cohorts: list[float] = []
    action_totals: dict[str, float] = {}
    termination_counts: dict[str, int] = {}
    for outcome in outcomes:
        selected = outcome.get("selected_index")
        if selected is None:
            none_count += 1
        else:
            selection_counts[int(selected)] += 1
        treated += np.asarray(outcome["treated_by_dose"], dtype=float)
        dlts += np.asarray(outcome["dlt_by_dose"], dtype=float)
        total_n.append(float(outcome["total_n"]))
        outcome_total = float(outcome["total_n"])
        allocation_fractions.append(
            np.asarray(outcome["treated_by_dose"], dtype=float)
            / max(outcome_total, 1.0)
        )
        total_dlt.append(float(outcome["total_dlt"]))
        cohorts.append(float(outcome["cohort_count"]))
        for action, count in dict(outcome.get("action_counts") or {}).items():
            action_totals[action] = action_totals.get(action, 0.0) + float(count)
        termination = _termination_category(outcome)
        termination_counts[termination] = termination_counts.get(termination, 0) + 1
        early_stop_count += int(termination == "safety_stop")

    n_trial = len(outcomes)
    total_treated = float(np.sum(treated))
    mean_trial_allocation = np.mean(np.asarray(allocation_fractions), axis=0)
    pooled_allocation = (
        treated / total_treated if total_treated else np.zeros(k, dtype=float)
    )
    allocation_basis = (
        "mean_of_within_trial_patient_fractions"
        if design in {"crm", "three_plus_three"}
        else "pooled_patients_across_trials"
    )
    reference_allocation = (
        mean_trial_allocation
        if allocation_basis == "mean_of_within_trial_patient_fractions"
        else pooled_allocation
    )
    payload: dict[str, Any] = {
        "trial_count": n_trial,
        "selection_probability_by_dose": (selection_counts / n_trial).tolist(),
        "no_selection": _proportion_summary(none_count, n_trial),
        "early_stop": _proportion_summary(early_stop_count, n_trial),
        "mean_patients_by_dose": (treated / n_trial).tolist(),
        "pooled_patient_allocation_fraction_by_dose": pooled_allocation.tolist(),
        "mean_trial_patient_allocation_fraction_by_dose": (
            mean_trial_allocation.tolist()
        ),
        "patient_allocation_basis": allocation_basis,
        "mean_dlt_by_dose": (dlts / n_trial).tolist(),
        "mean_total_sample_size": float(np.mean(total_n)),
        "median_total_sample_size": float(np.median(total_n)),
        "mean_total_dlt": float(np.mean(total_dlt)),
        "pooled_observed_dlt_rate": float(np.sum(dlts) / total_treated) if total_treated else 0.0,
        "mean_cohort_count": float(np.mean(cohorts)),
        "mean_action_counts": {
            action: count / n_trial for action, count in sorted(action_totals.items())
        },
        "termination_probability": {
            category: count / n_trial
            for category, count in sorted(termination_counts.items())
        },
    }
    payload["patient_allocation_fraction_by_dose"] = reference_allocation.tolist()
    payload["mean_observed_dlt_rate"] = payload["pooled_observed_dlt_rate"]
    if probabilities is not None:
        truth = np.asarray(probabilities, dtype=float)
        correct_index = true_mtd_index(truth, config)
        correct = sum(
            outcome.get("selected_index") == correct_index for outcome in outcomes
        )
        if config.mtd_definition == "acceptable_interval":
            acceptable = sum(
                outcome.get("selected_index") is not None
                and config.acceptable_low
                <= truth[int(outcome["selected_index"])]
                <= config.acceptable_high
                for outcome in outcomes
            )
            overtoxic_threshold = config.acceptable_high
            acceptable_metric_definition = "within_configured_acceptable_interval"
        elif config.mtd_definition == "closest_to_target":
            acceptable = correct
            overtoxic_threshold = config.target_dlt
            acceptable_metric_definition = "same_as_closest_to_target_correct_decision"
        else:
            acceptable = sum(
                outcome.get("selected_index") is not None
                and truth[int(outcome["selected_index"])] <= config.target_dlt
                for outcome in outcomes
            )
            overtoxic_threshold = config.target_dlt
            acceptable_metric_definition = "selected_dose_not_above_target"
        overtoxic_selection = sum(
            outcome.get("selected_index") is not None
            and truth[int(outcome["selected_index"])] > overtoxic_threshold
            for outcome in outcomes
        )
        overtoxic_doses = truth > overtoxic_threshold
        payload.update(
            {
                "true_mtd_index": correct_index,
                "truth_target_status": truth_target_status(truth, config),
                "correct_dose_definition": config.mtd_definition,
                "acceptable_metric_definition": acceptable_metric_definition,
                "overtoxic_threshold": float(overtoxic_threshold),
                "correct_decision": _proportion_summary(correct, n_trial),
                "acceptable_selection": _proportion_summary(acceptable, n_trial),
                "overtoxic_selection": _proportion_summary(overtoxic_selection, n_trial),
                "overtoxic_patient_fraction": (
                    float(np.sum(treated[overtoxic_doses]) / total_treated)
                    if total_treated
                    else 0.0
                ),
            }
        )
    return payload


def _quantile_summary(
    scenario_summaries: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    paths = {
        "correct_decision_probability": ("correct_decision", "probability"),
        "acceptable_selection_probability": ("acceptable_selection", "probability"),
        "overtoxic_selection_probability": ("overtoxic_selection", "probability"),
        "overtoxic_patient_fraction": ("overtoxic_patient_fraction",),
        "mean_total_sample_size": ("mean_total_sample_size",),
        "mean_total_dlt": ("mean_total_dlt",),
        "early_stop_probability": ("early_stop", "probability"),
    }
    result: dict[str, dict[str, float]] = {}
    for label, path in paths.items():
        values: list[float] = []
        for summary in scenario_summaries:
            value: Any = summary
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            quantiles = np.quantile(values, [0.05, 0.5, 0.95])
            result[label] = {
                "q05": float(quantiles[0]),
                "median": float(quantiles[1]),
                "q95": float(quantiles[2]),
            }
    return result


def _clustered_probability_summary(
    scenario_summaries: Sequence[dict[str, Any]], source_key: str
) -> dict[str, Any]:
    """Estimate a marginal probability using truth curves as outer clusters."""

    values = np.asarray(
        [float(summary[source_key]["probability"]) for summary in scenario_summaries],
        dtype=float,
    )
    probability = float(np.mean(values))
    if len(values) > 1:
        standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        method = "outer_truth_curve_cluster_standard_error"
    else:
        standard_error = float(
            scenario_summaries[0][source_key]["monte_carlo_standard_error"]
        )
        method = "single_truth_inner_binomial_standard_error"
    return {
        "probability": probability,
        "monte_carlo_standard_error": standard_error,
        "standard_error_method": method,
        "confidence_interval_95_low": max(0.0, probability - 1.96 * standard_error),
        "confidence_interval_95_high": min(1.0, probability + 1.96 * standard_error),
        "outer_truth_curve_count": int(len(values)),
    }


def _aggregate_scenario_summaries(
    scenario_summaries: Sequence[dict[str, Any]], config: OCConfig
) -> dict[str, Any]:
    """Aggregate compact conditional summaries without retaining trial histories."""

    if not scenario_summaries:
        raise ValueError("At least one scenario summary is required.")
    trial_counts = np.asarray(
        [int(summary["trial_count"]) for summary in scenario_summaries], dtype=float
    )
    total_trials = int(np.sum(trial_counts))
    scenario_weights = trial_counts / total_trials

    def weighted_vector(key: str) -> np.ndarray:
        matrix = np.asarray([summary[key] for summary in scenario_summaries], dtype=float)
        return np.sum(matrix * scenario_weights[:, None], axis=0)

    mean_patients = weighted_vector("mean_patients_by_dose")
    mean_dlts = weighted_vector("mean_dlt_by_dose")
    total_patient_exposure = float(np.sum(mean_patients))
    pooled_allocation = (
        (mean_patients / total_patient_exposure).tolist()
        if total_patient_exposure > 0
        else [0.0] * len(config.doses)
    )
    action_names = sorted(
        {
            key
            for summary in scenario_summaries
            for key in summary.get("mean_action_counts", {})
        }
    )
    termination_names = sorted(
        {
            key
            for summary in scenario_summaries
            for key in summary.get("termination_probability", {})
        }
    )
    payload: dict[str, Any] = {
        "trial_count": total_trials,
        "selection_probability_by_dose": weighted_vector(
            "selection_probability_by_dose"
        ).tolist(),
        "no_selection": _clustered_probability_summary(
            scenario_summaries, "no_selection"
        ),
        "early_stop": _clustered_probability_summary(
            scenario_summaries, "early_stop"
        ),
        "mean_patients_by_dose": mean_patients.tolist(),
        "pooled_patient_allocation_fraction_by_dose": pooled_allocation,
        "mean_trial_patient_allocation_fraction_by_dose": weighted_vector(
            "mean_trial_patient_allocation_fraction_by_dose"
        ).tolist(),
        "patient_allocation_fraction_by_dose": weighted_vector(
            "patient_allocation_fraction_by_dose"
        ).tolist(),
        "patient_allocation_basis": scenario_summaries[0].get(
            "patient_allocation_basis"
        ),
        "mean_dlt_by_dose": mean_dlts.tolist(),
        "mean_total_sample_size": float(
            sum(
                weight * float(summary["mean_total_sample_size"])
                for weight, summary in zip(
                    scenario_weights, scenario_summaries, strict=True
                )
            )
        ),
        "median_total_sample_size": float(
            np.median(
                [float(summary["median_total_sample_size"]) for summary in scenario_summaries]
            )
        ),
        "median_total_sample_size_basis": "median_of_conditional_scenario_medians",
        "mean_total_dlt": float(
            sum(
                weight * float(summary["mean_total_dlt"])
                for weight, summary in zip(
                    scenario_weights, scenario_summaries, strict=True
                )
            )
        ),
        "pooled_observed_dlt_rate": (
            float(np.sum(mean_dlts) / total_patient_exposure)
            if total_patient_exposure > 0
            else 0.0
        ),
        "mean_cohort_count": float(
            sum(
                weight * float(summary["mean_cohort_count"])
                for weight, summary in zip(
                    scenario_weights, scenario_summaries, strict=True
                )
            )
        ),
        "mean_action_counts": {
            action: float(
                sum(
                    weight * float(summary.get("mean_action_counts", {}).get(action, 0.0))
                    for weight, summary in zip(
                        scenario_weights, scenario_summaries, strict=True
                    )
                )
            )
            for action in action_names
        },
        "termination_probability": {
            category: float(
                sum(
                    weight
                    * float(summary.get("termination_probability", {}).get(category, 0.0))
                    for weight, summary in zip(
                        scenario_weights, scenario_summaries, strict=True
                    )
                )
            )
            for category in termination_names
        },
        "correct_decision": _clustered_probability_summary(
            scenario_summaries, "correct_decision"
        ),
        "acceptable_selection": _clustered_probability_summary(
            scenario_summaries, "acceptable_selection"
        ),
        "overtoxic_selection": _clustered_probability_summary(
            scenario_summaries, "overtoxic_selection"
        ),
        "true_mtd_index": None,
        "true_mtd_varies_across_scenarios": len(scenario_summaries) > 1,
    }
    payload["mean_observed_dlt_rate"] = payload["pooled_observed_dlt_rate"]
    overtoxic_exposure = sum(
        float(summary["overtoxic_patient_fraction"])
        * float(summary["mean_total_sample_size"])
        * int(summary["trial_count"])
        for summary in scenario_summaries
    )
    total_exposure = sum(
        float(summary["mean_total_sample_size"]) * int(summary["trial_count"])
        for summary in scenario_summaries
    )
    payload["overtoxic_patient_fraction"] = (
        overtoxic_exposure / total_exposure if total_exposure > 0 else 0.0
    )
    return payload


def _compact_trial_outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Drop per-cohort histories immediately after one trial is simulated."""

    keys = (
        "selected_index",
        "treated_by_dose",
        "dlt_by_dose",
        "total_n",
        "total_dlt",
        "cohort_count",
        "early_stopped",
        "stop_reason",
        "action_counts",
    )
    return {key: outcome.get(key) for key in keys}


def simulate_operating_characteristics(
    scenarios: Iterable[TruthScenario],
    config: OCConfig,
    *,
    aggregate_scenarios: bool = True,
) -> dict[str, Any]:
    """Run every requested design against the same fixed truth scenarios.

    Random streams are deterministically separated by design, scenario, and
    replicate.  This makes a result stable when another design or scenario is
    added or removed.  It is intentionally different from GPT-wrapper's one
    sequential RNG stream, but has the same sampling distribution.

    ``aggregate_scenarios`` is only scientifically meaningful when the scenario
    collection carries an explicit probability distribution (for example,
    posterior draws).  Standard target-location and stress scenario sets are
    reported one row at a time and must pass ``False`` here.
    """

    resolved_config = config.validated()
    resolved_scenarios = [
        scenario.validated(len(resolved_config.doses)) for scenario in scenarios
    ]
    if not resolved_scenarios:
        raise ValueError("At least one truth scenario is required.")
    scenario_names = [scenario.name for scenario in resolved_scenarios]
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("Truth scenario names must be unique for reproducible seeding.")
    total_work = (
        len(resolved_scenarios)
        * resolved_config.trials_per_scenario
        * len(resolved_config.designs)
    )
    if total_work > MAX_TOTAL_VIRTUAL_TRIALS:
        raise ValueError(
            f"Requested {total_work:,} virtual trials exceeds the synchronous "
            f"safety limit of {MAX_TOTAL_VIRTUAL_TRIALS:,}. Reduce truth curves, "
            "trials per curve, or selected designs."
        )

    design_results: dict[str, Any] = {}
    for design in resolved_config.designs:
        per_scenario: list[dict[str, Any]] = []
        for scenario in resolved_scenarios:
            outcomes = []
            for replicate in range(resolved_config.trials_per_scenario):
                seed_material = (
                    f"{resolved_config.seed}\0{design}\0{scenario.name}\0{replicate}"
                ).encode("utf-8")
                stream_seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:8], "big"
                )
                rng = np.random.default_rng(stream_seed)
                outcome = _simulate_one_trial(
                    design, scenario.probabilities, resolved_config, rng
                )
                outcomes.append(_compact_trial_outcome(outcome))
            summary = summarize_trial_outcomes(
                outcomes,
                scenario.probabilities,
                resolved_config,
                design=design,
            )
            per_scenario.append(
                {
                    "scenario_name": scenario.name,
                    "source": scenario.source,
                    "truth_probabilities": list(scenario.probabilities),
                    **summary,
                }
            )
        design_payload: dict[str, Any] = {
            "scenario_performance_quantiles": _quantile_summary(per_scenario),
            "per_scenario": per_scenario,
        }
        if aggregate_scenarios:
            design_payload["integrated_summary"] = _aggregate_scenario_summaries(
                per_scenario, resolved_config
            )
        design_results[design] = design_payload

    return {
        "format": "phase1_operating_characteristics_v1",
        "method": {
            "truth_is_fixed_within_each_virtual_trial": True,
            "patient_outcome_model": "independent_binomial_dlt_conditional_on_truth",
            "late_onset_or_dropout_model": None,
            "interpretation": (
                "Conditional OC for one fixed scenario. A scenario average is "
                "Bayesian assurance only when scenarios are sampled from a stated "
                "posterior truth distribution; arbitrary stress scenarios are not "
                "assigned Bayesian model probability."
            ),
            "integrated_standard_error": (
                "Truth curves are outer Monte Carlo clusters; virtual trials within "
                "one curve are not treated as marginally independent."
            ),
            "random_number_stream": (
                "one_sequential_numpy_generator_per_design_reset_to_the_common_seed"
            ),
            "scenario_aggregation_enabled": bool(aggregate_scenarios),
            "synchronous_virtual_trial_limit": MAX_TOTAL_VIRTUAL_TRIALS,
        },
        "config": resolved_config.to_payload(),
        "scenario_count": len(resolved_scenarios),
        "total_trials_per_design": (
            len(resolved_scenarios) * resolved_config.trials_per_scenario
        ),
        "truth_scenarios": [scenario.to_payload() for scenario in resolved_scenarios],
        "design_results": design_results,
    }
