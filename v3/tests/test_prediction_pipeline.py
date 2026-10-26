from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phase1_extraction.artifacts import ArticlePaths, write_json
from phase1_extraction.friendly_review import load_or_create_friendly_review, save_friendly_review
from phase1_extraction.models import (
    Cohort, CurveRow, DLTDefinition, DLTWindow, DoseExpression, DoseSeries,
    DosingPhase, Intervention, PopulationDescription, Regimen, RegimenComponent,
    ReportedExtraction, Study, StudyPart,
)
from phase1_prediction.evidence import load_evidence_library
from phase1_prediction.models import TargetTrial
from phase1_prediction.legacy_service import predict_target_curve, preview_target_evidence
from phase1_prediction.curves import _hierarchical_linear_predictor
from phase1_prediction.bayesian import softplus
from phase1_simulation.service import (
    BoinDesignCandidate, BoinSettings, compare_boin_designs, run_boin_simulation,
    search_boin_parameters,
)


def _article(root: Path, document_id: str, *, rows: list[tuple[float, int, int]], series_id: str = "series") -> ArticlePaths:
    paths = ArticlePaths(root, document_id)
    paths.ensure()
    extraction = ReportedExtraction(
        document_id=document_id,
        studies=[Study(study_id="study", label="Example phase I study")],
        study_parts=[StudyPart(part_id="part_i",study_id="study",label="Phase I",phase_raw="Phase I")],
        populations=[PopulationDescription(
            population_id="pop", study_id="study", disease_core="multiple myeloma",
            age_min=18, age_max=75, age_unit="years", sex_eligibility_raw="all",
            inclusion_criteria=["relapsed multiple myeloma"],
        )],
        interventions=[Intervention(
            intervention_id="drug", name_raw="IPH2101", modality="drug",
            mechanism_raw=["anti-KIR monoclonal antibody"],
        )],
        regimens=[Regimen(
            regimen_id="regimen", label="IPH2101 monotherapy", composition="single_component",
            components=[RegimenComponent(
                component_id="component", intervention_id="drug", role="escalated",
                participates_in_dose_finding=True,
                dosing_phases=[DosingPhase(
                    dosing_phase_id="phase", phase_label="treatment", route_raw="intravenous",
                    frequency_raw="every 28 days", dose_form_raw="infusion",
                )],
            )],
        )],
        dlt_windows=[DLTWindow(window_id="window", study_id="study", raw_definition="28 days", normalized_days=28)],
        dlt_definitions=[DLTDefinition(definition_id="definition", study_id="study", criteria_raw=["grade 3 nonhematologic toxicity"])],
        dose_series=[DoseSeries(
            series_id=series_id, study_id="study", part_id="part_i", label=f"{series_id} IPH2101",
            regimen_id="regimen", varying_component_id="component", window_id="window",
            comparable_unit="mg/kg",
        )],
        cohorts=[Cohort(
            cohort_id=f"cohort_{index}", series_id=series_id, label=f"{dose} mg/kg",
            actual_status="actual", regimen_id="regimen",
            dose=DoseExpression(raw_expression=f"{dose} mg/kg", value_kind="scalar", value=dose, unit_raw="mg/kg", unit_normalized="mg/kg"),
            result_scope_ids=["scope"],
        ) for index, (dose, _n, _y) in enumerate(rows)],
    )
    curve_rows = [CurveRow(
        curve_row_id=f"row_{index}", document_id=document_id, study_id="study", part_id="part_i",
        cohort_id=f"cohort_{index}", series_id=series_id, result_scope_id="scope",
        regimen_id="regimen", varying_component_id="component", dose_value=dose,
        dose_unit="mg/kg", dose_raw=f"{dose} mg/kg", window_id="window", n=n, y=y,
        rate=y/n, denominator_role="paired_dlt",
        denominator_basis="reported",
        numerator_basis="reported" if y else "imputed_zero", rule_ids=[] if y else ["R2"],
        included_in_default=True, included_in_reported_only=index == 0,
    ) for index, (dose, n, y) in enumerate(rows)]
    write_json(paths.reported_extraction, extraction)
    write_json(paths.curve_rows, {
        "policy_version": "test", "project_default": [row.model_dump(mode="json") for row in curve_rows],
        "reported_only": [row.model_dump(mode="json") for row in curve_rows if row.included_in_reported_only],
        "all_rows": [row.model_dump(mode="json") for row in curve_rows],
    })
    return paths


def test_native_v3_evidence_preserves_series_scopes_and_resolver_values(tmp_path: Path) -> None:
    _article(tmp_path, "PMC1", rows=[(.1, 3, 0), (.3, 6, 2)], series_id="first")
    _article(tmp_path, "PMC2", rows=[(.2, 4, 1), (.6, 3, 1)], series_id="second")
    library = load_evidence_library([tmp_path])
    assert library.document_count == 2
    assert {item.series_id for item in library.series} == {"first", "second"}
    first = next(item for item in library.series if item.series_id == "first")
    assert [(point.dose, point.n, point.y) for point in first.points] == [(.1, 3, 0), (.3, 6, 2)]
    assert first.points[0].denominator_role == "paired_dlt"
    assert first.points[1].denominator_role == "paired_dlt"
    assert first.points[0].numerator_basis == "imputed_zero"
    assert first.features["drug"] == "IPH2101"
    assert first.study_group_id == "PMC1|study|part_i"


def test_reviewed_curve_is_accuracy_only_without_changing_prediction_or_extraction(tmp_path: Path) -> None:
    paths = _article(tmp_path, "PMC1", rows=[(.1, 3, 0), (.3, 6, 2)])
    raw_before = paths.curve_rows.read_bytes()
    review = load_or_create_friendly_review(paths)
    unit = next(unit for unit in review.units if unit.predicted_point and unit.predicted_point.cohort_id == "cohort_1")
    unit.decision = "incorrect"
    unit.gold_point = unit.predicted_point.model_copy(update={"n": 5, "y": 1})
    review.module_complete["curve_results"] = True
    save_friendly_review(paths, review)

    extracted = load_evidence_library([tmp_path], review_policy="extracted")
    corrected = load_evidence_library([tmp_path], review_policy="reviewed_if_available")
    assert [(p.n, p.y) for p in extracted.series[0].points] == [(3, 0), (6, 2)]
    assert [(p.n, p.y) for p in corrected.series[0].points] == [(3, 0), (6, 2)]
    assert corrected.fingerprint == extracted.fingerprint
    assert corrected.review_policy == "extracted"
    assert corrected.series[0].points[1].source == "extraction"
    assert paths.curve_rows.read_bytes() == raw_before


def test_all_three_methods_and_boin_run_without_writing_article_curve_cache(tmp_path: Path) -> None:
    _article(tmp_path, "PMC1", rows=[(.1, 3, 0), (.3, 6, 2)])
    _article(tmp_path, "PMC2", rows=[(.1, 4, 1), (.3, 3, 1)])
    library = load_evidence_library([tmp_path])
    target = TargetTrial(
        name="New IPH2101 study", drug="IPH2101", disease="multiple myeloma",
        planned_doses=(.05, .1, .2, .4), dose_unit="mg/kg", route="intravenous",
        schedule="every 28 days", dlt_window="28 days",
    )
    files_before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    prediction = predict_target_curve(library, target, threshold=0, draw_count=64, seed=7)
    files_after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    assert files_after == files_before
    assert set(prediction.methods) == {
        "article_line_weighted", "article_posterior_draw_pool",
        "power_likelihood_random_intercept_slope",
    }
    assert prediction.model_version == "phase1_v3_no_uplift_v1"
    for method_key, method in prediction.methods.items():
        assert method["status"] == "fitted", (method_key, method)
        assert len(method["posterior_draws"]) == 64
        assert all(len(row) == 4 for row in method["posterior_draws"])
        assert method["intermediate_series_fits_persisted"] is False
        simulation = run_boin_simulation(
            prediction.to_dict(), method_key=method_key, settings=BoinSettings(max_sample_size=12),
            posterior_curve_count=8, trials_per_curve=2, seed=3,
        )
        assert simulation["total_trials"] == 16
        assert sum(simulation["selection_probability"]) + simulation["no_selection_probability"] == pytest.approx(1)
        assert len(simulation["mean_patients_by_dose"]) == 4


def test_reported_only_remains_but_legacy_review_policy_cannot_filter_library(tmp_path: Path) -> None:
    paths = _article(tmp_path, "PMC1", rows=[(.1, 3, 0), (.3, 6, 2)])
    reported = load_evidence_library([tmp_path], evidence_policy="reported_only")
    assert [(p.dose, p.y) for p in reported.series[0].points] == [(.1, 0)]
    absent = load_evidence_library([tmp_path], review_policy="reviewed_only")
    assert len(absent.series) == 1 and absent.review_policy == "extracted"
    review = load_or_create_friendly_review(paths)
    review.module_complete["curve_results"] = True
    for unit in review.units:
        if unit.unit_kind == "curve_point":
            unit.decision = "correct"
            unit.gold_point = unit.predicted_point
    save_friendly_review(paths, review)
    included = load_evidence_library([tmp_path], review_policy="reviewed_only")
    assert len(included.series) == 1


def test_parameter_search_uses_same_complete_posterior_curves(tmp_path: Path) -> None:
    _article(tmp_path, "PMC1", rows=[(.1, 3, 0), (.3, 6, 2)])
    library = load_evidence_library([tmp_path])
    target = TargetTrial(name="Target", drug="IPH2101", disease="multiple myeloma", planned_doses=(.05, .1, .2), dose_unit="mg/kg")
    prediction = predict_target_curve(library, target, threshold=0, draw_count=32, seed=4).to_dict()
    search = search_boin_parameters(
        prediction, method_key="power_likelihood_random_intercept_slope", target_dlt=[.2, .25], cohort_size=[3],
        max_sample_size=[12], elimination_cutoff=[.9], stop_rule_m=[9],
        posterior_curve_count=4, trials_per_curve=1,
    )
    assert search["candidate_count"] == 2
    assert len(search["ranked"]) == 2


def test_evidence_choices_and_target_create_bound_prediction_and_design_runs(tmp_path: Path) -> None:
    _article(tmp_path, "PMC1", rows=[(.1, 3, 0), (.3, 6, 2)], series_id="first")
    _article(tmp_path, "PMC2", rows=[(.1, 4, 1), (.3, 3, 1)], series_id="second")
    library = load_evidence_library([tmp_path])
    target = TargetTrial(
        name="Target", drug="IPH2101", disease="multiple myeloma",
        planned_doses=(.05, .1, .2), dose_unit="mg/kg", target_dlt=.30,
    )
    preview = preview_target_evidence(library, target, threshold=0)
    excluded_key = next(row["series_key"] for row in preview["selected_series"] if row["document_id"] == "PMC2")
    prediction = predict_target_curve(
        library, target, threshold=0, draw_count=32, seed=4,
        exclude_series_keys=[excluded_key],
    ).to_dict()
    repeated = predict_target_curve(
        library, target, threshold=0, draw_count=32, seed=4,
        exclude_series_keys=[excluded_key],
    ).to_dict()
    assert prediction["run_id"] == repeated["run_id"]
    assert {row["document_id"] for row in prediction["selected_series"]} == {"PMC1"}
    assert any(row["reason"] == "manually_excluded" for row in prediction["excluded_series"])

    comparison = compare_boin_designs(
        prediction, method_key="power_likelihood_random_intercept_slope", target_dlt=target.target_dlt,
        candidates=[
            BoinDesignCandidate("Compact", BoinSettings(target_dlt=.20, max_sample_size=12)),
            BoinDesignCandidate("Invalid", BoinSettings(target_dlt=.20, cohort_size=3, max_sample_size=10)),
        ],
        posterior_curve_count=4, trials_per_curve=1,
    )
    assert comparison["prediction_run_id"] == prediction["run_id"]
    assert comparison["target_dlt"] == .30
    assert comparison["ranked"][0]["target_dlt"] == .30
    assert comparison["invalid"][0]["candidate_label"] == "Invalid"


def test_design_comparison_uses_user_selected_metric_importance(monkeypatch) -> None:
    import phase1_simulation.service as service

    prediction = {"run_id": "prediction", "methods": {"method": {
        "status": "fitted", "doses": [1, 2], "posterior_draws": [[.1, .4]],
    }}}

    def simulated(_prediction, *, settings, **_kwargs):
        if settings.max_sample_size == 12:
            return {
                "correct_selection_probability": .90, "overtoxic_selection_probability": .40,
                "no_selection_probability": 0, "early_stop_probability": .05, "mean_total_n": 12,
            }
        return {
            "correct_selection_probability": .80, "overtoxic_selection_probability": .10,
            "no_selection_probability": 0, "early_stop_probability": .10, "mean_total_n": 18,
        }

    monkeypatch.setattr(service, "run_boin_simulation", simulated)
    candidates = [
        BoinDesignCandidate("Selection-first", BoinSettings(max_sample_size=12)),
        BoinDesignCandidate("Safety-first", BoinSettings(max_sample_size=18)),
    ]
    default = compare_boin_designs(
        prediction, method_key="method", target_dlt=.25, candidates=candidates,
        posterior_curve_count=1, trials_per_curve=1,
    )
    custom_priority = [
        "overtoxic_selection_probability", "correct_selection_probability",
        "early_stop_probability", "mean_total_n",
    ]
    custom = compare_boin_designs(
        prediction, method_key="method", target_dlt=.25, candidates=candidates,
        posterior_curve_count=1, trials_per_curve=1, ranking_priority=custom_priority,
    )
    assert default["ranked"][0]["candidate_label"] == "Selection-first"
    assert custom["ranked"][0]["candidate_label"] == "Safety-first"
    assert custom["ranking_priority"] == custom_priority
    assert custom["run_id"] != default["run_id"]
    with pytest.raises(ValueError, match="exactly once"):
        compare_boin_designs(
            prediction, method_key="method", target_dlt=.25, candidates=candidates,
            posterior_curve_count=1, trials_per_curve=1,
            ranking_priority=["mean_total_n"] * 4,
        )


def test_invalid_target_and_excessive_simulation_are_blocked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        TargetTrial(
            name="Target", drug="IPH2101", disease="multiple myeloma",
            planned_doses=(.1, float("nan")), dose_unit="mg/kg",
        ).validated()

    _article(tmp_path, "PMC1", rows=[(.1, 3, 0), (.3, 6, 2)])
    library = load_evidence_library([tmp_path])
    target = TargetTrial(
        name="Target", drug="IPH2101", disease="multiple myeloma",
        planned_doses=(.05, .1, .2), dose_unit="mg/kg",
    )
    prediction = predict_target_curve(library, target, threshold=0, draw_count=32, seed=4).to_dict()
    with pytest.raises(ValueError, match="250,000"):
        run_boin_simulation(
            prediction, method_key="power_likelihood_random_intercept_slope", settings=BoinSettings(max_sample_size=12),
            posterior_curve_count=32, trials_per_curve=8_000,
        )


def test_hierarchical_predictor_reduces_to_random_intercept_and_shared_curve() -> None:
    offsets = np.asarray([-1.0, 0.0, 1.0])
    groups = np.asarray([0, 0, 1])
    intercepts = np.asarray([.2, -.1])
    no_slope = _hierarchical_linear_predictor(
        -1.0, .3, offsets, groups, intercept_effects=intercepts,
        slope_effects=np.zeros(2),
    )
    assert no_slope == pytest.approx(-1.0 + intercepts[groups] + softplus(.3) * offsets)
    shared = _hierarchical_linear_predictor(-1.0, .3, offsets, groups)
    assert shared == pytest.approx(-1.0 + softplus(.3) * offsets)


@pytest.mark.parametrize("rows", [
    [(.1, 6, 0), (.2, 6, 0), (.4, 6, 0)],
    [(.1, 6, 6), (.2, 6, 6), (.4, 6, 6)],
    [(.1, 3, 0)],
])
def test_random_intercept_slope_method_handles_extreme_and_sparse_series(tmp_path: Path, rows) -> None:
    _article(tmp_path, "PMC1", rows=rows)
    library = load_evidence_library([tmp_path])
    target = TargetTrial(
        name="Target", drug="IPH2101", disease="multiple myeloma",
        planned_doses=(.05, .1, .2, .4), dose_unit="mg/kg",
    )
    prediction = predict_target_curve(library, target, threshold=0, draw_count=32, seed=14)
    method = prediction.methods["power_likelihood_random_intercept_slope"]
    assert method["status"] == "fitted", method
    draws = np.asarray(method["posterior_draws"])
    assert np.isfinite(draws).all()
    assert np.all((draws >= 0) & (draws <= 1))
    assert np.all(np.diff(draws, axis=1) >= -1e-12)
    assert method["new_target_study_random_intercept"] is True
    assert method["new_target_study_random_slope"] is True
