"""Numerical contract tests against the actual v1 working-tree implementation."""
import importlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from test_prediction_pipeline import _article
from phase1_prediction.evidence import load_evidence_library
from phase1_prediction.compact_adapter import library_records, target_query
from phase1_prediction.models import TargetTrial
from phase1_prediction.reference_bridge import build_workspace, predict
from phase1_prediction.reference import toxicity_similarity as similarity
from phase1_prediction.reference import prediction_service as prediction
from phase1_prediction.reference import oc_simulation as oc
from phase1_prediction.reference.oc_service import joint_target_posterior_draws
from phase1_prediction.reference.boin_parameter_search import BOINCalibration, build_boin_parameter_grid, run_boin_parameter_search
from phase1_extraction.artifacts import write_json
from phase1_prediction.legacy_service import predict_target_curve, preview_target_evidence


@pytest.fixture(scope="module")
def oracle():
    path = Path(__file__).resolve().parents[2] / "v1/mtd_similarity"
    if not path.is_dir():
        pytest.skip("Source-checkout oracle only; packaged tests do not require sibling v1.")
    spec = importlib.util.spec_from_file_location("v1_parity_oracle", path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return lambda name: importlib.import_module("v1_parity_oracle." + name)


def _library(tmp_path):
    _article(tmp_path, "PMC1", rows=[(.1, 3, 0), (.3, 6, 2)])
    _article(tmp_path, "PMC2", rows=[(.1, 5, 0), (.4, 6, 3)])
    return load_evidence_library([tmp_path])


def _query():
    return target_query(TargetTrial("Trial", "IPH2101", "multiple myeloma", (.05, .1, .2), "mg/kg",
        route="intravenous", dlt_window="21 days", trial_type="single", age="18-65 years",
        mechanism="anti-KIR monoclonal antibody", eligibility="relapsed multiple myeloma",
        dlt_definition="grade 3 nonhematologic toxicity", dose_form="infusion", sex="all",
        regimen="IPH2101 monotherapy", schedule="every 28 days"))


def test_seven_similarity_groups_use_native_profiles_and_real_days(tmp_path):
    from phase1_prediction.profile_similarity import LABELS,compare_profiles,normalize_weights
    from phase1_prediction.profiles import query_profile
    library = _library(tmp_path)
    records,query=library_records(library),_query()
    assert records[0]['data']['_comparison_profile']['windows'][0]['days']==28
    weights={k:1 for k in LABELS}
    scored,backend=similarity.score_library(query,records,embedding_backend='local',field_weights=weights)
    assert backend=='local_token_cosine'
    for row in scored:
        assert set(row['groups'])==set(LABELS)
        assert row['design_compared'] is False
        expected=[compare_profiles(query_profile(query),p,weights=weights)['similarity'] for p in row['data']['_comparison_scopes'].values()]
        assert row['similarity']==max(expected)
        assert row['groups']['window']['score']<1
    defaults,threshold,_=similarity.resolve_similarity_parameters(use_manual_default=True)
    assert threshold==.3 and sum(normalize_weights(defaults).values())==pytest.approx(1)


@pytest.mark.parametrize("trial_type", ["single", "combination"])
def test_three_complete_curve_methods_match_unadjusted_oracle(tmp_path, oracle, trial_type, monkeypatch):
    # Scoring intentionally changed. Hold scores fixed to test the downstream
    # numerical methods against the independent v1 implementation.
    monkeypatch.setattr(oracle("prediction_service"), "score_library", similarity.score_library)
    monkeypatch.setattr(oracle("estimate_toxicity_curve"), "score_library", similarity.score_library)
    library = _library(tmp_path)
    workspace = build_workspace(library, draw_count=64, seed=17)
    query = _query()
    query["Trial_Type"] = {"value": trial_type}
    actual = predict(library, query, workspace)
    old_workspace = oracle("prediction_service").PredictionWorkspace(
        **workspace.__dict__,
        combination_alpha_uplift_prior=oracle("target_trial_type_prior").load_combination_alpha_uplift_prior(),
    )
    expected = oracle("prediction_service").predict_target_curve(
        query, old_workspace, apply_combination_prior_adjustment=False,
    )
    assert set(actual["toxicity_curve"]) == {"article_line_weighted", "article_posterior_draw_pool", "power_likelihood_random_intercept_slope"}
    assert actual["embedding_backend_used"] == expected["embedding_backend_used"]
    for key, method in actual["toxicity_curve"].items():
        assert method["status"] == "fitted"
        expected_points = [{k: v for k, v in p.items() if not k.startswith("unadjusted_")
                            and k != "target_trial_type_prior_adjustment"}
                           for p in expected["toxicity_curve"][key]["curve_points"]]
        assert method["curve_points"] == expected_points
        a = joint_target_posterior_draws(actual, curve_method_key=key, requested_draw_count=32, seed=11)
        e = oracle("oc_service").joint_target_posterior_draws(expected, curve_method_key=key, requested_draw_count=32, seed=11)
        np.testing.assert_array_equal(a["posterior_draws"], e["posterior_draws"])


def test_scopes_do_not_merge_and_adapter_does_not_impute_points(tmp_path, oracle):
    library = _library(tmp_path)
    first = library.series[0]
    added = replace(first, series_key=first.series_key + "_second_window", window_id="other_window")
    library = replace(library, series=(*library.series, added))
    records = library_records(library)
    assert len(records) == 2
    data = records[0]["data"]
    assert len(data["dose_levels"]) == len(data["dlt_count_at_each_dose_level"]) == 4
    assert len({r["arm_or_cohort"] for r in data["dose_levels"]}) == 2
    rows, skipped = oracle("dlt_evidence").build_candidate_rows(data)
    assert len(rows) == 4
    assert sum(r["n_patients_with_dlt"] for r in rows) == 4


def test_boin_reference_oc_matches_v1_when_approved_exceptions_are_applied(oracle):
    scenarios = [oc.TruthScenario("low", (.05, .15, .25), "test"), oc.TruthScenario("high", (.35, .55, .8), "test")]
    config = oc.OCConfig(doses=(1., 2., 3.), designs=("boin",), max_sample_size=12, trials_per_scenario=50, seed=9)
    old = oracle("oc_simulation")
    old_config = old.OCConfig(**config.__dict__, boin_accelerated_titration=False)
    actual = oc.simulate_operating_characteristics(scenarios, config)
    expected = old.simulate_operating_characteristics(scenarios, old_config)
    assert actual["design_results"] == expected["design_results"]


def test_boin_grid_has_no_titration_and_ranking_priority_is_recorded(tmp_path):
    library = _library(tmp_path)
    raw = predict(library, _query(), build_workspace(library, draw_count=32))
    config = oc.OCConfig(doses=(.05, .1, .2), designs=("boin",), max_sample_size=12, trials_per_scenario=2)
    candidates = build_boin_parameter_grid(calibrations=[BOINCalibration("automatic")], elimination_cutoffs=[.9, .95], elimination_min_n=1,
                                           stop_rule_ms=[None, 9], base_config=config)
    priority = ("mean_total_sample_size", "safety_stop", "overtoxic_selection", "correct_decision")
    result = run_boin_parameter_search(raw, base_config=config, candidates=candidates, posterior_curve_count=4, ranking_priority=priority)
    assert len(result["candidate_results"]) == 4
    assert result["ranking_priority"] == list(priority)
    assert "accelerated" not in json.dumps(result)
    rows = result["candidate_results"]
    keys = [tuple((-1 if metric == "correct_decision" else 1) * row["macro_metrics"][metric] for metric in priority) for row in rows]
    assert keys == sorted(keys)


def test_metadata_only_articles_remain_in_similarity_corpus_not_likelihood(tmp_path):
    _library(tmp_path)
    paths = _article(tmp_path, "PMC3", rows=[(.1, 3, 0)])
    write_json(paths.curve_rows, {"project_default": [], "reported_only": []})
    library = load_evidence_library([tmp_path])
    records = library_records(library)
    assert len(records) == 3
    third = next(r for r in records if r["pmcid"] == "PMC3")
    assert third["data"]["Trial_Inclusion"]
    assert third["data"]["dose_levels"] == third["data"]["dlt_count_at_each_dose_level"] == []
    preview = prediction.preview_historical_articles(_query(), build_workspace(library, draw_count=32))
    assert "PMC3" not in {c["document_id"] for c in preview["candidates"]}


def test_unit_conversion_scope_audit_and_v1_curve_summary(tmp_path):
    library = _library(tmp_path)
    first = library.series[0]
    incompatible = replace(first, series_key=first.series_key + "_kg", unit="mg",
                           points=tuple(replace(p, unit="mg", source_unit="mg") for p in first.points))
    library = replace(library, series=(*library.series, incompatible))
    target = TargetTrial("Trial", "IPH2101", "multiple myeloma", (.00005, .0001, .0002), "g/kg")
    result = predict_target_curve(library, target, draw_count=32)
    assert result.target["dose_unit"] == "mg/kg"
    np.testing.assert_allclose(result.target["planned_doses"], [.05, .1, .2])
    assert incompatible.series_key not in {r["series_key"] for r in result.selected_series}
    assert any(r["series_key"] == incompatible.series_key for r in result.excluded_series)
    for key, method in result.methods.items():
        assert [p["posterior_mean"] for p in method["summary"]] == [p["dlt_probability"] for p in result.reference_result["toxicity_curve"][key]["curve_points"]]


def test_optional_grid_is_for_display_only_and_boins_require_explicit_doses(tmp_path):
    from phase1_prediction.reference.oc_service import planned_oc_doses
    library = _library(tmp_path)
    result = predict_target_curve(library, TargetTrial("", "IPH2101", "multiple myeloma", (), "mg/kg"), draw_count=32)
    # v1 combines a 101-point display grid with observed historical doses.
    assert len(result.methods["article_line_weighted"]["summary"]) >= 101
    with pytest.raises(ValueError, match="explicit|automatic|user-entered"):
        planned_oc_doses(result.reference_result)


def test_cli_defaults_use_new_independent_source_contract():
    from phase1_project.cli import parser
    args = parser().parse_args(["predict", "--extraction-root", "data", "--target", "target.json", "--output", "out.json"])
    assert args.threshold == 0. and args.top_articles is None
    assert args.draw_count == 1000 and args.seed == 20260919
    assert args.backend == 'logistic'
    assert args.synthesis == 'same_article_joint_posterior_mixture'
    assert not hasattr(args, 'historical_ess_cap')


def test_web_prediction_export_and_cli_prediction_use_identical_boin_path(tmp_path):
    from phase1_simulation.service import BoinSettings, run_boin_simulation
    library = _library(tmp_path)
    result = predict_target_curve(library, TargetTrial("Test", "IPH2101", "multiple myeloma", (.05, .1, .2), "mg/kg"), draw_count=32)
    options = dict(method_key="article_line_weighted", settings=BoinSettings(max_sample_size=12), posterior_curve_count=4, trials_per_curve=2, seed=9)
    native = run_boin_simulation(result.to_dict(), **options)
    web_export = run_boin_simulation(result.reference_result, **options)
    assert native["reference_result"]["design_results"] == web_export["reference_result"]["design_results"]
