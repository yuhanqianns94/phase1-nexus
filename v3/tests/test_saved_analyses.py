from pathlib import Path

from phase1_project.analyses import list_analyses, load_analysis, save_analysis
from phase1_web.reporting import analysis_html, prediction_csv, simulation_csv


def _prediction() -> dict:
    return {
        "run_id": "prediction_example", "target": {
            "name": "Example trial", "drug": "Drug", "disease": "Cancer",
            "dose_unit": "mg", "target_dlt": .25,
        },
        "methods": {"power_likelihood_random_intercept_slope": {"status": "fitted", "summary": [
            {"dose": 1, "posterior_mean": .2, "low_95": .1, "high_95": .3},
        ]}},
    }


def _simulation(run_id: str) -> dict:
    return {
        "run_id": run_id, "prediction_run_id": "prediction_example",
        "method_key": "power_likelihood_random_intercept_slope", "doses": [1], "selection_probability": [.8],
        "true_mtd_probability": [.7], "mean_patients_by_dose": [6.0],
        "mean_dlts_by_dose": [1.0], "correct_selection_probability": .8,
        "overtoxic_selection_probability": .1, "no_selection_probability": .1,
        "mean_total_n": 6,
    }


def test_saved_analysis_identity_includes_simulation_and_exports_are_readable(tmp_path: Path) -> None:
    first = {"prediction": _prediction(), "simulation": _simulation("simulation_one"), "active_method": "power_likelihood_random_intercept_slope"}
    second = {"prediction": _prediction(), "simulation": _simulation("simulation_two"), "active_method": "power_likelihood_random_intercept_slope"}
    first_path = save_analysis(tmp_path, first)
    assert save_analysis(tmp_path, first) == first_path
    second_path = save_analysis(tmp_path, second)
    assert first_path != second_path
    records = list_analyses(tmp_path)
    assert len(records) == 2
    assert load_analysis(first_path)["prediction_run_id"] == "prediction_example"
    assert prediction_csv(_prediction()).startswith(b"\xef\xbb\xbf")
    assert simulation_csv(first["simulation"]).startswith(b"\xef\xbb\xbf")
    assert b"Closest-to-target selection" in analysis_html(first)
