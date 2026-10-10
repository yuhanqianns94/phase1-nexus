from __future__ import annotations

import itertools
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import numpy as np

from .boin import simulate_boin_trial

MAX_SIMULATED_TRIALS = 250_000
SIMULATION_MODEL_VERSION = "phase1_v1_oc_cohort_only_v3"

RANKING_DIRECTIONS = {
    "correct_selection_probability": "maximize",
    "overtoxic_selection_probability": "minimize",
    "early_stop_probability": "minimize",
    "mean_total_n": "minimize",
}
DEFAULT_RANKING_PRIORITY = tuple(RANKING_DIRECTIONS)


def _is_multidrug_prediction(prediction: Mapping[str, Any]) -> bool:
    # Dispatch by the stable payload contract. The multidrug engine separately
    # validates its calculation version before any simulation can run.
    return prediction.get('format') in {
        'multidrug_target_prediction_v1', 'phase1_v3_multidrug_prediction_v1',
    } or str(prediction.get('model_version', '')).startswith('v3_multidrug_bayes_')


def _simulation_version(prediction: Mapping[str, Any]) -> str:
    if _is_multidrug_prediction(prediction):
        from .multidrug import REGIMEN_PATH_SIMULATION_VERSION
        return REGIMEN_PATH_SIMULATION_VERSION
    return SIMULATION_MODEL_VERSION


def _validated_ranking_priority(ranking_priority: Iterable[str] | None) -> tuple[str, ...]:
    values = DEFAULT_RANKING_PRIORITY if ranking_priority is None else tuple(str(value) for value in ranking_priority)
    expected = set(RANKING_DIRECTIONS)
    if len(values) != len(expected) or set(values) != expected:
        raise ValueError("Ranking priority must contain each supported metric exactly once.")
    return values


def _ranking_key(row: Mapping[str, Any], ranking_priority: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(
        -float(row[metric]) if RANKING_DIRECTIONS[metric] == "maximize" else float(row[metric])
        for metric in ranking_priority
    )


@dataclass(frozen=True)
class BoinSettings:
    target_dlt: float = 0.25
    cohort_size: int = 3
    max_sample_size: int = 30
    start_index: int = 0
    elimination_cutoff: float = 0.95
    stop_rule_enabled: bool = True
    stop_rule_m: int = 12
    use_isotonic: bool = True
    mtd_definition: str = "acceptable_interval"
    acceptable_low: float = 0.20
    acceptable_high: float = 0.35
    phi1: float | None = None
    phi2: float | None = None
    check_lowest_dose: bool = True
    lowest_dose_delta: float = 0.03
    check_final_mtd_deesc_bound: bool = True
    modify_1_3_deesc_to_stay: bool = False

    def validated(self, dose_count: int) -> "BoinSettings":
        from .reference_runner import config_from_settings
        config_from_settings(self, tuple(range(1, dose_count + 1)), 100, 9)
        return self


@dataclass(frozen=True)
class BoinDesignCandidate:
    label: str
    settings: BoinSettings


def _posterior_matrix(prediction: Mapping[str, Any], method_key: str) -> tuple[np.ndarray, tuple[float, ...]]:
    if _is_multidrug_prediction(prediction):
        from .multidrug import path_matrix
        matrix, points, _ = path_matrix(prediction, method_key)
        return matrix, tuple(float(i + 1) for i in range(len(points)))
    from phase1_prediction.reference.prediction_contract import ensure_current_prediction
    ensure_current_prediction(prediction)
    if "toxicity_curve" in prediction:
        from phase1_prediction.reference.oc_service import joint_target_posterior_draws, planned_oc_doses
        draws = joint_target_posterior_draws(prediction, curve_method_key=method_key)
        return np.asarray(draws["posterior_draws"], dtype=float), planned_oc_doses(prediction)
    methods = prediction.get("methods")
    method = methods.get(method_key) if isinstance(methods, Mapping) else None
    if not isinstance(method, Mapping) or method.get("status") != "fitted":
        raise ValueError(f"Prediction method {method_key!r} is not fitted.")
    doses = tuple(float(value) for value in method.get("doses", []))
    matrix = np.asarray(method.get("posterior_draws"), dtype=float)
    if len(doses) < 2 or any(value <= 0 for value in doses) or any(b <= a for a, b in zip(doses, doses[1:])):
        raise ValueError("Prediction does not contain valid strictly increasing planned doses.")
    if matrix.ndim != 2 or matrix.shape[1] != len(doses) or len(matrix) == 0:
        raise ValueError("Prediction lacks a complete draw-by-dose posterior matrix.")
    if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Posterior probabilities must be finite values in [0, 1].")
    if np.any(np.diff(matrix, axis=1) < -1e-10):
        raise ValueError("Every posterior draw must be non-decreasing across doses.")
    return matrix, doses


def _true_index(probabilities: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(probabilities - target)))


def run_boin_simulation(
    prediction: Mapping[str, Any], *, method_key: str, settings: BoinSettings,
    posterior_curve_count: int = 100, trials_per_curve: int = 100, seed: int = 9,
) -> dict[str, Any]:
    if _is_multidrug_prediction(prediction):
        from .multidrug import simulate_regimen_path
        return simulate_regimen_path(prediction, method_key=method_key, settings=settings,
                                     posterior_curve_count=posterior_curve_count, trials_per_curve=trials_per_curve, seed=seed)
    from .reference_runner import run_reference_simulation
    return run_reference_simulation(prediction, method_key=method_key, settings=settings, posterior_curve_count=posterior_curve_count, trials_per_curve=trials_per_curve, seed=seed)


def compare_boin_designs(
    prediction: Mapping[str, Any], *, method_key: str, target_dlt: float,
    candidates: Iterable[BoinDesignCandidate], posterior_curve_count: int = 50,
    trials_per_curve: int = 100, seed: int = 9, max_candidates: int = 96,
    ranking_priority: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare designs and rank them using a user-selected metric priority."""
    resolved_priority = _validated_ranking_priority(ranking_priority)
    values = list(candidates)
    if not values or len(values) > max_candidates:
        raise ValueError(f"Design comparison must contain 1 to {max_candidates} candidates.")
    if len(values) * posterior_curve_count * trials_per_curve > MAX_SIMULATED_TRIALS:
        raise ValueError(f"A design comparison is limited to {MAX_SIMULATED_TRIALS:,} virtual trials.")
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(values):
        settings = replace(candidate.settings, target_dlt=float(target_dlt))
        try:
            result = run_boin_simulation(
                prediction, method_key=method_key, settings=settings,
                posterior_curve_count=posterior_curve_count,
                trials_per_curve=trials_per_curve, seed=seed,
            )
            rows.append({
                "candidate_index": index, "candidate_label": candidate.label or f"Design {index + 1}",
                "status": "completed", **asdict(settings),
                "correct_selection_probability": result["correct_selection_probability"],
                "overtoxic_selection_probability": result["overtoxic_selection_probability"],
                "no_selection_probability": result["no_selection_probability"],
                "early_stop_probability": result["early_stop_probability"],
                "mean_total_n": result["mean_total_n"], "result": result,
            })
        except ValueError as exc:
            rows.append({
                "candidate_index": index, "candidate_label": candidate.label or f"Design {index + 1}",
                "status": "invalid", "error": str(exc), **asdict(settings),
            })
    completed = [row for row in rows if row["status"] == "completed"]
    completed.sort(key=lambda row: _ranking_key(row, resolved_priority))
    identity = {
        "simulation_model_version": _simulation_version(prediction),
        "prediction_run_id": prediction.get("run_id") or prediction.get("prediction_id"), "method_key": method_key,
        "target_dlt": target_dlt, "candidates": [asdict(item) for item in values],
        "posterior_curve_count": posterior_curve_count, "trials_per_curve": trials_per_curve,
        "seed": seed, "ranking_priority": list(resolved_priority),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "format": "phase1_v3_boin_design_comparison_v3",
        "simulation_model_version": identity['simulation_model_version'],
        "run_id": "design_comparison_" + fingerprint[:20],
        "input_fingerprint": fingerprint, "prediction_run_id": identity['prediction_run_id'],
        "method_key": method_key, "target_dlt": float(target_dlt),
        "ranking_priority": list(resolved_priority),
        "ranking_directions": dict(RANKING_DIRECTIONS),
        "posterior_curve_count": posterior_curve_count, "trials_per_curve": trials_per_curve,
        "candidate_count": len(rows), "ranked": completed,
        "invalid": [row for row in rows if row["status"] != "completed"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def search_boin_parameters(
    prediction: Mapping[str, Any], *, method_key: str,
    target_dlt: Iterable[float], cohort_size: Iterable[int], max_sample_size: Iterable[int],
    elimination_cutoff: Iterable[float], stop_rule_m: Iterable[int],
    posterior_curve_count: int = 50, trials_per_curve: int = 100, seed: int = 9,
    max_candidates: int = 96,
    ranking_priority: Iterable[str] | None = None,
    base_settings: BoinSettings | None = None,
) -> dict[str, Any]:
    candidates = list(itertools.product(target_dlt, cohort_size, max_sample_size, elimination_cutoff, stop_rule_m))
    if not candidates or len(candidates) > max_candidates:
        raise ValueError(f"Parameter grid must contain 1 to {max_candidates} candidates.")
    requested_trials = len(candidates) * posterior_curve_count * trials_per_curve
    if requested_trials > MAX_SIMULATED_TRIALS:
        raise ValueError(f"A parameter search is limited to {MAX_SIMULATED_TRIALS:,} virtual trials.")
    rows = []
    for index, values in enumerate(candidates):
        settings = replace(base_settings or BoinSettings(), target_dlt=float(values[0]), cohort_size=int(values[1]), max_sample_size=int(values[2]), elimination_cutoff=float(values[3]), stop_rule_m=int(values[4]))
        comparison = compare_boin_designs(
            prediction, method_key=method_key, target_dlt=float(values[0]),
            candidates=[BoinDesignCandidate(label=f"Candidate {index + 1}", settings=settings)],
            posterior_curve_count=posterior_curve_count, trials_per_curve=trials_per_curve,
            seed=seed, max_candidates=max_candidates,
        )
        if comparison["ranked"]:
            row = comparison["ranked"][0]
        else:
            row = comparison["invalid"][0]
        row["candidate_index"] = index
        rows.append(row)
    completed = [row for row in rows if row["status"] == "completed"]
    resolved_priority = _validated_ranking_priority(ranking_priority)
    completed.sort(key=lambda row: _ranking_key(row, resolved_priority))
    identity = {'prediction_run_id':prediction.get('run_id') or prediction.get('prediction_id'),
                'method_key':method_key, 'simulation_model_version':_simulation_version(prediction),
                'path_level_ids':prediction.get('target',{}).get('path_level_ids',[]),
                'settings':[asdict(replace(base_settings or BoinSettings(),target_dlt=float(v[0]),
                    cohort_size=int(v[1]),max_sample_size=int(v[2]),elimination_cutoff=float(v[3]),stop_rule_m=int(v[4]))) for v in candidates],
                'posterior_curve_count':posterior_curve_count,'trials_per_curve':trials_per_curve,
                'seed':seed,'ranking_priority':list(resolved_priority)}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {"format": "phase1_v3_boin_parameter_search_v2", "method_key": method_key,
            'prediction_run_id':identity['prediction_run_id'], 'run_id':'parameter_search_'+fingerprint[:20],
            'input_fingerprint':fingerprint,'seed':seed,'posterior_curve_count':posterior_curve_count,
            'trials_per_curve':trials_per_curve,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            "simulation_model_version": identity['simulation_model_version'], "ranking_priority": list(resolved_priority),
            "candidate_count": len(rows), "ranked": completed, "invalid": [row for row in rows if row["status"] != "completed"]}
