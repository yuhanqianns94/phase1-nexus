"""Grid search for transparent BOIN operating-characteristic tuning.

The legacy CRM/BOIN/3+3 engines remain available, but the active simulation UI
uses this module to compare BOIN parameter sets against exactly the same truth
scenarios.  "Best" therefore always means best under the explicitly recorded
lexicographic metric priority; Pareto-efficient alternatives are retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .oc_service import build_truth_scenarios_from_curve, planned_oc_doses
from .oc_simulation import (
    MAX_TOTAL_VIRTUAL_TRIALS,
    OCConfig,
    simulate_operating_characteristics,
)


MAX_BOIN_PARAMETER_CANDIDATES = 96
SUPPORTED_SEARCH_TRUTH_MODES = (
    "gpt_wrapper_standard",
    "posterior_predictive",
)


def _candidate_tokens(raw: str, *, label: str) -> tuple[str, ...]:
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be entered as text.")
    tokens = tuple(
        token.strip()
        for token in re.split(r"[,;\n]+", raw)
        if token.strip()
    )
    if not tokens:
        raise ValueError(f"Enter at least one {label} candidate.")
    return tokens


def parse_probability_candidates(raw: str, *, label: str) -> tuple[float, ...]:
    """Parse a user-defined comma/newline list of probabilities in ``(0, 1]``."""

    values: list[float] = []
    for token in _candidate_tokens(raw, label=label):
        try:
            value = float(token)
        except ValueError as error:
            raise ValueError(f"{label} contains a non-numeric value: {token!r}.") from error
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"Every {label} candidate must lie in (0, 1].")
        if value not in values:
            values.append(value)
    return tuple(values)


def parse_positive_integer_candidates(
    raw: str,
    *,
    label: str,
    maximum: int | None = None,
) -> tuple[int, ...]:
    """Parse a de-duplicated list of positive integer grid candidates."""

    values: list[int] = []
    for token in _candidate_tokens(raw, label=label):
        try:
            numeric = float(token)
        except ValueError as error:
            raise ValueError(f"{label} contains a non-numeric value: {token!r}.") from error
        if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
            raise ValueError(f"Every {label} candidate must be a positive integer.")
        value = int(numeric)
        if maximum is not None and value > maximum:
            raise ValueError(f"Every {label} candidate must be at most {maximum}.")
        if value not in values:
            values.append(value)
    return tuple(values)


def parse_stop_rule_candidates(
    raw: str,
    *,
    cohort_size: int,
    maximum_sample_size: int,
) -> tuple[int | None, ...]:
    """Parse stop-at-m values; ``off``/``none`` disables the rule."""

    values: list[int | None] = []
    for token in _candidate_tokens(raw, label="stop-at-m"):
        if token.casefold() in {"off", "none", "disabled"}:
            value = None
        else:
            parsed = parse_positive_integer_candidates(
                token, label="stop-at-m", maximum=maximum_sample_size
            )[0]
            if parsed < cohort_size:
                raise ValueError(
                    f"Every stop-at-m candidate must be at least cohort size "
                    f"({cohort_size})."
                )
            value = parsed
        if value not in values:
            values.append(value)
    return tuple(values)


def parse_calibration_candidates(
    raw: str, *, target_dlt: float
) -> tuple["BOINCalibration", ...]:
    """Parse ``automatic`` and user-entered ``phi1:phi2`` calibration pairs."""

    values: list[BOINCalibration] = []
    signatures: set[tuple[float | None, float | None]] = set()
    for position, token in enumerate(
        _candidate_tokens(raw, label="boundary calibration"), start=1
    ):
        if token.casefold() in {"automatic", "auto"}:
            candidate = BOINCalibration("automatic")
        else:
            parts = tuple(part.strip() for part in token.split(":"))
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    "Enter each boundary calibration as phi1:phi2, separated by "
                    "commas or new lines; use 'automatic' for the BOIN default."
                )
            try:
                phi1, phi2 = (float(part) for part in parts)
            except ValueError as error:
                raise ValueError(
                    f"Boundary calibration {token!r} contains a non-numeric value."
                ) from error
            candidate = BOINCalibration(
                f"custom_{position}: phi1={phi1:g}, phi2={phi2:g}", phi1, phi2
            )
        validated = candidate.validated(target_dlt)
        signature = (validated.phi1, validated.phi2)
        if signature not in signatures:
            values.append(validated)
            signatures.add(signature)
    return tuple(values)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number.")
    return numeric


@dataclass(frozen=True)
class BOINCalibration:
    """One BOIN phi calibration; ``None`` requests the engine default."""

    label: str
    phi1: float | None = None
    phi2: float | None = None

    def validated(self, target_dlt: float) -> "BOINCalibration":
        label = str(self.label).strip()
        if not label:
            raise ValueError("Every BOIN calibration requires a label.")
        if (self.phi1 is None) != (self.phi2 is None):
            raise ValueError("Enter both phi1 and phi2 or use automatic calibration.")
        if self.phi1 is None:
            return BOINCalibration(label=label)
        phi1 = _finite(self.phi1, label="phi1")
        phi2 = _finite(self.phi2, label="phi2")
        if not 0 < phi1 < target_dlt < phi2 < 1:
            raise ValueError(
                "BOIN calibration must satisfy 0 < phi1 < target < phi2 < 1."
            )
        return BOINCalibration(label=label, phi1=phi1, phi2=phi2)


@dataclass(frozen=True)
class BOINScoreWeights:
    """Optional legacy utility diagnostic; never used to rank candidates in v3."""

    correct_decision_reward: float = 1.0
    overtoxic_selection_penalty: float = 1.0
    overtoxic_exposure_penalty: float = 1.0
    sample_size_penalty: float = 0.01

    def validated(self) -> "BOINScoreWeights":
        payload = {
            key: _finite(value, label=key) for key, value in asdict(self).items()
        }
        if any(value < 0 for value in payload.values()):
            raise ValueError("BOIN score weights must be non-negative.")
        if not any(value > 0 for value in payload.values()):
            raise ValueError("At least one BOIN score weight must be positive.")
        return BOINScoreWeights(**payload)

    def score(self, metrics: Mapping[str, float]) -> float:
        return float(
            self.correct_decision_reward * metrics["correct_decision"]
            - self.overtoxic_selection_penalty * metrics["overtoxic_selection"]
            - self.overtoxic_exposure_penalty * metrics["overtoxic_patient_fraction"]
            - self.sample_size_penalty * metrics["mean_total_sample_size"]
        )


@dataclass(frozen=True)
class BOINParameterCandidate:
    """One complete set of BOIN parameters varied by the grid search."""

    calibration: BOINCalibration
    elimination_cutoff: float
    elimination_min_n: int
    stop_rule_m: int | None

    def to_config(self, base_config: OCConfig) -> OCConfig:
        calibration = self.calibration.validated(base_config.target_dlt)
        return replace(
            base_config,
            designs=("boin",),
            boin_phi1=calibration.phi1,
            boin_phi2=calibration.phi2,
            boin_lambda_e=None,
            boin_lambda_d=None,
            boin_elimination_cutoff=float(self.elimination_cutoff),
            boin_elimination_min_n=int(self.elimination_min_n),
            boin_stop_rule_enabled=self.stop_rule_m is not None,
            boin_stop_rule_m=(
                int(self.stop_rule_m)
                if self.stop_rule_m is not None
                else int(base_config.boin_stop_rule_m)
            ),
        ).validated()

    def identity_payload(self) -> dict[str, Any]:
        return {
            "calibration_label": self.calibration.label,
            "phi1": self.calibration.phi1,
            "phi2": self.calibration.phi2,
            "elimination_cutoff": float(self.elimination_cutoff),
            "elimination_min_n": int(self.elimination_min_n),
            "stop_rule_enabled": self.stop_rule_m is not None,
            "stop_rule_m": (
                int(self.stop_rule_m) if self.stop_rule_m is not None else None
            ),
        }

    @property
    def candidate_id(self) -> str:
        canonical = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "boin_" + hashlib.sha256(canonical).hexdigest()[:12]


def build_boin_parameter_grid(
    *,
    calibrations: Sequence[BOINCalibration],
    elimination_cutoffs: Sequence[float],
    elimination_min_n: int,
    stop_rule_ms: Sequence[int | None],
    base_config: OCConfig,
) -> tuple[BOINParameterCandidate, ...]:
    """Build, validate, and de-duplicate a bounded Cartesian parameter grid."""

    resolved_base = replace(base_config, designs=("boin",)).validated()
    if not all(
        (
            calibrations,
            elimination_cutoffs,
            stop_rule_ms,
        )
    ):
        raise ValueError(
            "Every BOIN parameter dimension must contain at least one value."
        )
    candidates: list[BOINParameterCandidate] = []
    seen: set[str] = set()
    for calibration in calibrations:
        for cutoff in elimination_cutoffs:
            for stop_m in stop_rule_ms:
                candidate = BOINParameterCandidate(
                    calibration=calibration.validated(resolved_base.target_dlt),
                    elimination_cutoff=_finite(
                        cutoff, label="elimination_cutoff"
                    ),
                    elimination_min_n=int(elimination_min_n),
                    stop_rule_m=(int(stop_m) if stop_m is not None else None),
                )
                candidate.to_config(resolved_base)
                if candidate.candidate_id not in seen:
                    candidates.append(candidate)
                    seen.add(candidate.candidate_id)
    if len(candidates) > MAX_BOIN_PARAMETER_CANDIDATES:
        raise ValueError(
            f"The grid contains {len(candidates)} unique candidates; the limit is "
            f"{MAX_BOIN_PARAMETER_CANDIDATES}. Reduce one or more parameter lists."
        )
    return tuple(candidates)


def _scenario_macro_metrics(
    per_scenario: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    if not per_scenario:
        raise ValueError("BOIN search requires at least one scenario result.")

    def mean_path(*path: str, default: float | None = None) -> float:
        values: list[float] = []
        for row in per_scenario:
            value: Any = row
            for key in path:
                value = value.get(key) if isinstance(value, Mapping) else None
            if value is None and default is not None:
                value = default
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("A BOIN scenario result is missing a required metric.")
            values.append(float(value))
        return float(np.mean(values))

    return {
        "correct_decision": mean_path("correct_decision", "probability"),
        "acceptable_selection": mean_path("acceptable_selection", "probability"),
        "overtoxic_selection": mean_path("overtoxic_selection", "probability"),
        "overtoxic_patient_fraction": mean_path("overtoxic_patient_fraction"),
        "no_selection": mean_path("no_selection", "probability"),
        "safety_stop": mean_path("termination_probability", "safety_stop", default=0.0),
        "mean_total_sample_size": mean_path("mean_total_sample_size"),
        "mean_total_dlt": mean_path("mean_total_dlt"),
    }


def _is_pareto_efficient(index: int, rows: Sequence[Mapping[str, Any]]) -> bool:
    current = rows[index]["macro_metrics"]
    maximize = ("correct_decision",)
    minimize = (
        "overtoxic_selection",
        "overtoxic_patient_fraction",
        "mean_total_sample_size",
    )
    for other_index, other_row in enumerate(rows):
        if other_index == index:
            continue
        other = other_row["macro_metrics"]
        no_worse = all(other[key] >= current[key] for key in maximize) and all(
            other[key] <= current[key] for key in minimize
        )
        strictly_better = any(other[key] > current[key] for key in maximize) or any(
            other[key] < current[key] for key in minimize
        )
        if no_worse and strictly_better:
            return False
    return True


def _decision_boundaries(
    target_dlt: float, calibration: BOINCalibration
) -> tuple[float, float, float, float]:
    resolved = calibration.validated(target_dlt)
    phi1 = 0.6 * target_dlt if resolved.phi1 is None else resolved.phi1
    phi2 = 1.4 * target_dlt if resolved.phi2 is None else resolved.phi2
    lambda_e = math.log((1 - phi1) / (1 - target_dlt)) / math.log(
        target_dlt * (1 - phi1) / (phi1 * (1 - target_dlt))
    )
    lambda_d = math.log((1 - target_dlt) / (1 - phi2)) / math.log(
        phi2 * (1 - target_dlt) / (target_dlt * (1 - phi2))
    )
    return float(lambda_e), float(lambda_d), float(phi1), float(phi2)


def run_boin_parameter_search(
    curve_result: Mapping[str, Any],
    *,
    base_config: OCConfig,
    candidates: Sequence[BOINParameterCandidate],
    score_weights: BOINScoreWeights = BOINScoreWeights(),
    ranking_priority: Sequence[str] = ("correct_decision", "overtoxic_selection", "safety_stop", "mean_total_sample_size"),
    truth_mode: str = "posterior_predictive",
    posterior_curve_count: int = 100,
    curve_method_key: str = "article_line_weighted",
) -> dict[str, Any]:
    """Run every candidate on one shared truth set and rank the resulting OCs."""

    if truth_mode not in SUPPORTED_SEARCH_TRUTH_MODES:
        raise ValueError(
            "truth_mode must be gpt_wrapper_standard or posterior_predictive."
        )
    resolved_base = replace(base_config, designs=("boin",)).validated()
    if tuple(resolved_base.doses) != tuple(planned_oc_doses(curve_result)):
        raise ValueError("BOIN search doses must match the target trial dose grid.")
    weights = score_weights.validated()
    ranking_priority = tuple(ranking_priority)
    if len(ranking_priority) != 4 or set(ranking_priority) != {"correct_decision", "overtoxic_selection", "safety_stop", "mean_total_sample_size"}:
        raise ValueError("Ranking priority must contain each of the four metrics exactly once.")
    if not candidates:
        raise ValueError("At least one BOIN parameter candidate is required.")
    if len(candidates) > MAX_BOIN_PARAMETER_CANDIDATES:
        raise ValueError(
            f"At most {MAX_BOIN_PARAMETER_CANDIDATES} candidates may be compared."
        )
    scenarios, truth_diagnostics = build_truth_scenarios_from_curve(
        curve_result,
        mode=truth_mode,
        posterior_curve_count=posterior_curve_count,
        seed=resolved_base.seed,
        include_stress_scenarios=False,
        curve_method_key=curve_method_key,
        target_dlt=resolved_base.target_dlt,
    )
    total_trials = len(candidates) * len(scenarios) * resolved_base.trials_per_scenario
    if total_trials > MAX_TOTAL_VIRTUAL_TRIALS:
        raise ValueError(
            f"Requested {total_trials:,} virtual trials exceeds the synchronous "
            f"BOIN-search limit of {MAX_TOTAL_VIRTUAL_TRIALS:,}."
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        config = candidate.to_config(resolved_base)
        simulation = simulate_operating_characteristics(
            scenarios,
            config,
            aggregate_scenarios=(truth_mode == "posterior_predictive"),
        )
        design_payload = simulation["design_results"]["boin"]
        macro = _scenario_macro_metrics(design_payload["per_scenario"])
        lambda_e, lambda_d, resolved_phi1, resolved_phi2 = _decision_boundaries(
            config.target_dlt, candidate.calibration
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "parameters": {
                    **candidate.identity_payload(),
                    "resolved_phi1": resolved_phi1,
                    "resolved_phi2": resolved_phi2,
                    "lambda_e": lambda_e,
                    "lambda_d": lambda_d,
                },
                "score": weights.score(macro),
                "macro_metrics": macro,
                "simulation": simulation,
            }
        )

    for index, row in enumerate(rows):
        row["pareto_efficient"] = _is_pareto_efficient(index, rows)
    rows.sort(key=lambda row: tuple(
        (-1 if metric == "correct_decision" else 1) * float(row["macro_metrics"][metric])
        for metric in ranking_priority) + (str(row["candidate_id"]),))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return {
        "format": "phase1_boin_parameter_search_v3_priority",
        "ranking_priority": list(ranking_priority),
        "purpose": "compare_boin_parameter_settings_under_common_truth_scenarios",
        "best_candidate_interpretation": (
            "First under the recorded lexicographic priorities; not a universally optimal "
            "clinical design. Review Pareto-efficient alternatives and each scenario."
        ),
        "truth_mode": truth_mode,
        "curve_method_key": curve_method_key,
        "truth_scenario_count": len(scenarios),
        "truth_scenarios": [scenario.to_payload() for scenario in scenarios],
        "truth_generation": truth_diagnostics,
        "candidate_count": len(rows),
        "total_virtual_trials": total_trials,
        "score_formula": (
            "w_correct*correct_decision - w_overtoxic_selection*overtoxic_selection "
            "- w_overtoxic_exposure*overtoxic_patient_fraction "
            "- w_sample_size*mean_total_sample_size"
        ),
        "score_weights": asdict(weights),
        "score_role": "diagnostic_only_not_used_for_ranking",
        "metric_definitions": {
            "correct_decision": resolved_base.mtd_definition,
            "acceptable_interval_used": (
                resolved_base.mtd_definition == "acceptable_interval"
            ),
            "acceptable_low": (
                resolved_base.acceptable_low
                if resolved_base.mtd_definition == "acceptable_interval"
                else None
            ),
            "acceptable_high": (
                resolved_base.acceptable_high
                if resolved_base.mtd_definition == "acceptable_interval"
                else None
            ),
            "overtoxic_threshold": (
                resolved_base.acceptable_high
                if resolved_base.mtd_definition == "acceptable_interval"
                else resolved_base.target_dlt
            ),
            "mean_total_sample_size": (
                "mean number of enrolled patients; this is the fourth term used by "
                "the utility score and Pareto ranking"
            ),
        },
        "base_config": resolved_base.to_payload(),
        "best_candidate_id": rows[0]["candidate_id"],
        "candidate_results": rows,
    }
