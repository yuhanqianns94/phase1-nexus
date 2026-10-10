"""BOIN operating-characteristic engine aligned with GPT-wrapper web.

The simulator works with zero-based dose indices and cumulative dose-level
Binomial data.  It implements the local BOIN escalation/de-escalation rule,
Beta(1, 1) overdose elimination, and a weighted PAVA final selection.

The engine uses complete cohorts only, with the reference platform's optional
stop-at-``m`` rule, overdose controls, and isotonic final recommendation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import beta


_COMPARISON_TOLERANCE = 1e-12


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite probability.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _probability_closed_right(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite probability.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return result


def _boundaries_from_phi(
    target_dlt: float,
    phi1: float,
    phi2: float,
) -> tuple[float, float, float, float]:
    """Return BOIN decision boundaries from the reference phi calibration."""

    if not (0.0 < phi1 < target_dlt < phi2 < 1.0):
        raise ValueError(
            "BOIN requires 0 < phi1 < target_dlt < phi2 < 1."
        )
    lambda_e = math.log((1.0 - phi1) / (1.0 - target_dlt)) / math.log(
        target_dlt * (1.0 - phi1) / (phi1 * (1.0 - target_dlt))
    )
    lambda_d = math.log((1.0 - target_dlt) / (1.0 - phi2)) / math.log(
        phi2 * (1.0 - target_dlt) / (target_dlt * (1.0 - phi2))
    )
    return float(lambda_e), float(lambda_d), float(phi1), float(phi2)


def _automatic_boundaries(target_dlt: float) -> tuple[float, float, float, float]:
    """Return lambda_e, lambda_d, phi1, phi2 using the standard BOIN defaults."""

    return _boundaries_from_phi(target_dlt, 0.6 * target_dlt, 1.4 * target_dlt)


def _posterior_overtoxicity_probability(
    dlt_count: int,
    patient_count: int,
    target_dlt: float,
) -> float:
    """Pr(p > target_dlt | data) under the BOIN Beta(1, 1) safety prior."""

    posterior_alpha = 1 + dlt_count
    posterior_beta = 1 + patient_count - dlt_count
    return float(1.0 - beta.cdf(target_dlt, posterior_alpha, posterior_beta))


def _weighted_pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted non-decreasing pooled-adjacent-violators algorithm."""

    resolved_values = np.asarray(values, dtype=float)
    resolved_weights = np.asarray(weights, dtype=float)
    if (
        resolved_values.ndim != 1
        or resolved_weights.ndim != 1
        or len(resolved_values) != len(resolved_weights)
        or len(resolved_values) == 0
    ):
        raise ValueError("PAVA values and weights must be non-empty equal 1D arrays.")
    if not np.isfinite(resolved_values).all() or not np.isfinite(resolved_weights).all():
        raise ValueError("PAVA values and weights must be finite.")
    if np.any(resolved_weights <= 0):
        raise ValueError("PAVA weights must be strictly positive.")

    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(
        zip(resolved_values, resolved_weights, strict=True)
    ):
        blocks.append(
            {
                "weight": float(weight),
                "weighted_sum": float(weight * value),
                "start": index,
                "end": index,
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left["weighted_sum"]) / float(left["weight"])
            right_mean = float(right["weighted_sum"]) / float(right["weight"])
            if left_mean <= right_mean + 1e-15:
                break
            blocks[-2:] = [
                {
                    "weight": float(left["weight"]) + float(right["weight"]),
                    "weighted_sum": float(left["weighted_sum"])
                    + float(right["weighted_sum"]),
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                }
            ]

    fitted = np.empty(len(resolved_values), dtype=float)
    for block in blocks:
        block_mean = float(block["weighted_sum"]) / float(block["weight"])
        fitted[int(block["start"]) : int(block["end"]) + 1] = block_mean
    return fitted


def _select_isotonic_mtd(
    dlt_by_dose: np.ndarray,
    treated_by_dose: np.ndarray,
    eliminated: np.ndarray,
    target_dlt: float,
    upper_bound: float | None = None,
) -> int | None:
    """Select the closest tried, non-eliminated dose after weighted PAVA."""

    tried = treated_by_dose > 0
    if not np.any(tried):
        return None
    observed = np.divide(dlt_by_dose, treated_by_dose, out=np.zeros_like(dlt_by_dose, dtype=float), where=tried)
    # Match the reference PAVA's epsilon weight for untried levels, while
    # ensuring those levels remain ineligible for the final recommendation.
    isotonic = _weighted_pava(observed, np.maximum(treated_by_dose, 1e-12))
    admissible_mask = ~eliminated & tried
    if upper_bound is not None:
        admissible_mask &= isotonic <= upper_bound + _COMPARISON_TOLERANCE
    admissible_positions = np.flatnonzero(admissible_mask)
    if len(admissible_positions) == 0:
        return None

    differences = np.abs(isotonic[admissible_positions] - target_dlt)
    minimum_difference = float(np.min(differences))
    tied_positions = admissible_positions[
        np.abs(differences - minimum_difference) <= _COMPARISON_TOLERANCE
    ]
    tied_values = isotonic[tied_positions]
    if len(tied_positions) == 1:
        chosen_position = int(tied_positions[0])
    elif np.all(tied_values < target_dlt):
        chosen_position = int(np.max(tied_positions))
    elif np.all(tied_values > target_dlt):
        chosen_position = int(np.min(tied_positions))
    elif np.any(np.isclose(tied_values, target_dlt)):
        chosen_position = int(np.max(tied_positions[np.isclose(tied_values, target_dlt)]))
    else:
        # Below/equal-target ties favor the highest dose, matching BOIN's rule.
        chosen_position = int(np.max(tied_positions))
    return chosen_position


def _select_raw_mtd(
    dlt_by_dose: np.ndarray,
    treated_by_dose: np.ndarray,
    eliminated: np.ndarray,
    target_dlt: float,
    upper_bound: float | None,
) -> int | None:
    tried = treated_by_dose > 0
    admissible = tried & ~eliminated
    observed = np.zeros_like(treated_by_dose, dtype=float)
    observed[tried] = dlt_by_dose[tried] / treated_by_dose[tried]
    if upper_bound is not None:
        admissible &= observed <= upper_bound + _COMPARISON_TOLERANCE
    indices = np.flatnonzero(admissible)
    if not len(indices):
        return None
    # The reference uses argmin for raw-rate selection: ties retain the lowest.
    return int(indices[np.argmin(np.abs(observed[indices] - target_dlt))])


def simulate_boin_trial(
    true_probabilities: Sequence[float] | np.ndarray,
    *,
    start_index: int,
    cohort_size: int,
    max_sample_size: int,
    target_dlt: float,
    rng: np.random.Generator,
    lambda_e: float | None = None,
    lambda_d: float | None = None,
    phi1: float | None = None,
    phi2: float | None = None,
    elimination_cutoff: float = 0.95,
    stop_rule_enabled: bool = True,
    stop_rule_m: int = 12,
    modify_1_3_deesc_to_stay: bool = False,
    check_lowest_dose: bool = True,
    lowest_dose_delta: float = 0.03,
    check_final_mtd_deesc_bound: bool = True,
    use_isotonic: bool = True,
    allow_nonmonotone_truth: bool = False,
) -> dict[str, Any]:
    """Simulate one standard BOIN dose-escalation trial.

    DLT outcomes are generated from cumulative dose-level Binomial models.  The
    dose decision after each completed cohort is based on the cumulative
    observed rate at the current dose.  A final MTD is selected only from tried,
    non-eliminated doses.
    """

    try:
        probabilities = np.asarray(true_probabilities, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("true_probabilities must be a numeric 1D sequence.") from error
    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("true_probabilities must be a 1D sequence with at least two doses.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("true_probabilities must contain finite values in [0, 1].")
    if not allow_nonmonotone_truth and np.any(np.diff(probabilities) < -_COMPARISON_TOLERANCE):
        raise ValueError("true_probabilities must be non-decreasing by dose.")

    if isinstance(start_index, bool) or not isinstance(start_index, (int, np.integer)):
        raise ValueError("start_index must be an integer.")
    resolved_start = int(start_index)
    if not 0 <= resolved_start < len(probabilities):
        raise ValueError("start_index is outside true_probabilities.")
    resolved_cohort_size = _positive_integer(cohort_size, name="cohort_size")
    resolved_max_sample_size = _positive_integer(
        max_sample_size, name="max_sample_size"
    )
    if (
        resolved_max_sample_size < resolved_cohort_size
        or resolved_max_sample_size % resolved_cohort_size != 0
    ):
        raise ValueError(
            "max_sample_size must allow complete cohorts and be exactly divisible "
            "by cohort_size."
        )
    resolved_target = _probability(target_dlt, name="target_dlt")
    resolved_elimination_cutoff = _probability_closed_right(
        elimination_cutoff, name="elimination_cutoff"
    )
    for name, value in (
        ("stop_rule_enabled", stop_rule_enabled),
        ("modify_1_3_deesc_to_stay", modify_1_3_deesc_to_stay),
        ("check_lowest_dose", check_lowest_dose),
        ("check_final_mtd_deesc_bound", check_final_mtd_deesc_bound),
        ("use_isotonic", use_isotonic),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator.")

    if (lambda_e is None) != (lambda_d is None):
        raise ValueError("Enter both BOIN boundaries or leave both automatic.")
    if (phi1 is None) != (phi2 is None):
        raise ValueError("Enter both BOIN phi values or leave both automatic.")
    if phi1 is not None and lambda_e is not None:
        raise ValueError("Enter phi1/phi2 or lambda_e/lambda_d, not both.")
    if phi1 is not None:
        resolved_phi1 = _probability(phi1, name="phi1")
        resolved_phi2 = _probability(phi2, name="phi2")
        resolved_lambda_e, resolved_lambda_d, phi1, phi2 = _boundaries_from_phi(
            resolved_target, resolved_phi1, resolved_phi2
        )
        boundary_source = "manual_phi"
    elif lambda_e is None:
        resolved_lambda_e, resolved_lambda_d, phi1, phi2 = _automatic_boundaries(
            resolved_target
        )
        boundary_source = "automatic"
    else:
        resolved_lambda_e = _probability(lambda_e, name="lambda_e")
        resolved_lambda_d = _probability(lambda_d, name="lambda_d")
        phi1 = None
        phi2 = None
        boundary_source = "manual_lambda"
    if not (resolved_lambda_e < resolved_target < resolved_lambda_d):
        raise ValueError("BOIN boundaries must satisfy lambda_e < target_dlt < lambda_d.")

    dose_count = len(probabilities)
    resolved_stop_rule_m = _positive_integer(stop_rule_m, name="stop_rule_m")
    if resolved_stop_rule_m < resolved_cohort_size:
        raise ValueError("stop_rule_m must be at least cohort_size.")
    resolved_lowest_delta = float(lowest_dose_delta)
    if not math.isfinite(resolved_lowest_delta) or resolved_lowest_delta < 0.0:
        raise ValueError("lowest_dose_delta must be a finite non-negative number.")
    lowest_cutoff = resolved_elimination_cutoff - resolved_lowest_delta
    if check_lowest_dose and not 0.0 < lowest_cutoff <= 1.0:
        raise ValueError(
            "elimination_cutoff - lowest_dose_delta must lie in (0, 1]."
        )
    max_cohort_count = resolved_max_sample_size // resolved_cohort_size
    treated = np.zeros(dose_count, dtype=int)
    dlts = np.zeros(dose_count, dtype=int)
    eliminated = np.zeros(dose_count, dtype=bool)
    current = resolved_start
    cohort_count = 0
    action_counts: dict[str, int] = {}
    early_stopped = False
    stop_reason = "max_sample_size_reached"

    stopped_by_m = False
    last_decision = ""
    last_eliminated = False

    def count_action(action: str) -> None:
        action_counts[action] = action_counts.get(action, 0) + 1

    def enroll_cohort() -> None:
        nonlocal cohort_count
        observed_dlts = int(rng.binomial(resolved_cohort_size, float(probabilities[current])))
        treated[current] += resolved_cohort_size
        dlts[current] += observed_dlts
        cohort_count += 1

    def apply_overdose_control() -> bool:
        """Apply GPT-wrapper overdose controls at the current dose."""

        nonlocal current, early_stopped, stop_reason, last_eliminated
        last_eliminated = False
        if treated[current] > 0:
            overdose_probability = _posterior_overtoxicity_probability(
                int(dlts[current]), int(treated[current]), resolved_target
            )
            if (
                check_lowest_dose
                and current == 0
                and overdose_probability > lowest_cutoff
            ):
                early_stopped = True
                stop_reason = "lowest_dose_extra_safety_stop"
                count_action("stop_lowest_dose_extra_safety")
                return True
            if overdose_probability > resolved_elimination_cutoff:
                eliminated[current:] = True
                last_eliminated = True
                if current == 0:
                    early_stopped = True
                    stop_reason = "lowest_dose_eliminated_for_overtoxicity"
                    count_action("stop_lowest_dose_eliminated")
                    return True
                current -= 1
                count_action("deescalate_after_elimination")
                return False
        return False

    def apply_standard_decision() -> bool:
        """Update current dose; return True only for an early safety stop."""

        nonlocal current, last_decision
        if apply_overdose_control():
            return True
        if last_eliminated:
            last_decision = "deescalate_after_elimination"
            return False

        observed_rate = float(dlts[current] / treated[current])
        if observed_rate <= resolved_lambda_e:
            if current < dose_count - 1 and not eliminated[current + 1]:
                current += 1
                count_action("escalate")
                last_decision = "escalate"
            else:
                count_action("stay")
                last_decision = "stay"
        # Match BOIN_GPT.py's ordinary cohort decision at the boundary.
        elif observed_rate >= resolved_lambda_d:
            if (
                modify_1_3_deesc_to_stay
                and current > 0
                and treated[current] == 3
                and dlts[current] == 1
            ):
                count_action("stay_modified_from_1_over_3")
                last_decision = "stay_modified_from_1_over_3"
                return False
            if current > 0:
                current -= 1
                count_action("deescalate")
                last_decision = "deescalate"
            else:
                count_action("stay")
                last_decision = "stay"
        else:
            count_action("stay")
            last_decision = "stay"
        return False

    while cohort_count < max_cohort_count and not early_stopped and not stopped_by_m:
        treated_index = current
        enroll_cohort()
        if apply_standard_decision():
            break
        if (
            stop_rule_enabled
            and last_decision.startswith("stay")
            and treated[treated_index] >= resolved_stop_rule_m
        ):
            stopped_by_m = True
            stop_reason = "single_dose_stay_stop_rule_m_reached"

    selected_index = (
        None
        if early_stopped
        else (
            _select_isotonic_mtd(
                dlts,
                treated,
                eliminated,
                resolved_target,
                resolved_lambda_d if check_final_mtd_deesc_bound else None,
            )
            if use_isotonic
            else _select_raw_mtd(
                dlts,
                treated,
                eliminated,
                resolved_target,
                resolved_lambda_d if check_final_mtd_deesc_bound else None,
            )
        )
    )
    return {
        "selected_index": selected_index,
        "treated_by_dose": treated.tolist(),
        "dlt_by_dose": dlts.tolist(),
        "total_n": int(np.sum(treated)),
        "total_dlt": int(np.sum(dlts)),
        "cohort_count": cohort_count,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "action_counts": dict(sorted(action_counts.items())),
        "boundaries": {
            "lambda_e": resolved_lambda_e,
            "lambda_d": resolved_lambda_d,
            "phi1": phi1,
            "phi2": phi2,
            "source": boundary_source,
            "elimination_cutoff": resolved_elimination_cutoff,
            "check_lowest_dose": check_lowest_dose,
            "lowest_dose_delta": resolved_lowest_delta,
            "check_final_mtd_deesc_bound": check_final_mtd_deesc_bound,
        },
        "platform_rules": {
            "stop_rule_enabled": stop_rule_enabled,
            "stop_rule_m": resolved_stop_rule_m,
            "modify_1_3_deesc_to_stay": modify_1_3_deesc_to_stay,
            "use_isotonic": use_isotonic,
        },
    }
