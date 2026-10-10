"""Normalize the v1 OC output for existing v3 CLI and report consumers."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from phase1_prediction.reference.oc_service import run_oc_from_curve
from phase1_prediction.reference.oc_simulation import OCConfig, TruthScenario, simulate_operating_characteristics


def config_from_settings(settings, doses, trials_per_curve, seed):
    return OCConfig(doses=tuple(doses), designs=("boin",), target_dlt=settings.target_dlt,
        acceptable_low=settings.acceptable_low, acceptable_high=settings.acceptable_high,
        mtd_definition=settings.mtd_definition, start_index=settings.start_index,
        cohort_size=settings.cohort_size, max_sample_size=settings.max_sample_size,
        trials_per_scenario=trials_per_curve, seed=seed,
        boin_phi1=settings.phi1, boin_phi2=settings.phi2,
        boin_elimination_cutoff=settings.elimination_cutoff,
        boin_stop_rule_enabled=settings.stop_rule_enabled, boin_stop_rule_m=settings.stop_rule_m,
        boin_use_isotonic=settings.use_isotonic, boin_check_lowest_dose=settings.check_lowest_dose,
        boin_lowest_dose_delta=settings.lowest_dose_delta,
        boin_check_final_mtd_deesc_bound=settings.check_final_mtd_deesc_bound,
        boin_modify_1_3_deesc_to_stay=settings.modify_1_3_deesc_to_stay).validated()


def run_reference_simulation(prediction, *, method_key, settings, posterior_curve_count=100, trials_per_curve=10, seed=9):
    from .service import _posterior_matrix, SIMULATION_MODEL_VERSION
    matrix, doses = _posterior_matrix(prediction, method_key)
    config = config_from_settings(settings, doses, trials_per_curve, seed)
    if isinstance(posterior_curve_count, bool) or not isinstance(posterior_curve_count, int) or not 0 < posterior_curve_count <= len(matrix):
        raise ValueError(f"Posterior curve count must be a positive integer, at most {len(matrix)}.")
    reference_curve = prediction.get("reference_result") or (prediction if "toxicity_curve" in prediction else None)
    prediction_id = prediction.get("run_id") or prediction.get("prediction_run_identity_sha256")
    if reference_curve:
        raw = run_oc_from_curve(reference_curve, config,
            mode="posterior_predictive", posterior_curve_count=posterior_curve_count, curve_method_key=method_key)
    else:
        # Compatibility with external posterior matrices: same scenario engine,
        # evaluation definition and independent random streams, not the old loop.
        indices = np.random.default_rng(seed).choice(len(matrix), size=posterior_curve_count, replace=False)
        scenarios = [TruthScenario(f"posterior_draw_{int(i)}", tuple(matrix[i]), "external_joint_posterior") for i in indices]
        raw = simulate_operating_characteristics(scenarios, config)
    payload = raw["design_results"]["boin"]
    summary = payload["integrated_summary"]
    identity = {"simulation_model_version": SIMULATION_MODEL_VERSION, "prediction_run_id": prediction_id,
                "method_key": method_key, "config": config.to_payload(), "posterior_curve_count": posterior_curve_count}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    true_counts = np.zeros(len(doses))
    for row in payload["per_scenario"]:
        index = row.get("true_mtd_index")
        if index is not None:
            true_counts[index] += 1 / posterior_curve_count
    return {"format": "phase1_v3_boin_simulation_v3", "method_key": method_key,
            "simulation_model_version": SIMULATION_MODEL_VERSION, "run_id": "simulation_" + fingerprint[:20],
            "input_fingerprint": fingerprint, "prediction_run_id": prediction_id,
            "prediction_library_fingerprint": (prediction.get("evidence_contract") or prediction.get("v3_evidence_contract") or {}).get("library_fingerprint"),
            "doses": list(doses), "settings": asdict(settings), "seed": seed,
            "posterior_curve_count": posterior_curve_count, "trials_per_curve": trials_per_curve,
            "total_trials": summary["trial_count"], "selection_probability": summary["selection_probability_by_dose"],
            "no_selection_probability": summary["no_selection"]["probability"],
            "correct_dose_definition": config.mtd_definition, "true_mtd_probability": true_counts.tolist(),
            "correct_selection_probability": summary["correct_decision"]["probability"],
            "overtoxic_selection_probability": summary["overtoxic_selection"]["probability"],
            "pooled_overtoxic_patient_fraction": summary["overtoxic_patient_fraction"],
            "early_stop_probability": summary["termination_probability"].get("safety_stop", 0),
            "mean_patients_by_dose": summary["mean_patients_by_dose"], "mean_dlts_by_dose": summary["mean_dlt_by_dose"],
            "mean_total_n": summary["mean_total_sample_size"], "mean_total_dlt": summary["mean_total_dlt"],
            "reference_result": raw, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
