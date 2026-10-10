"""Outcome and simulation regressions verified independently of target fitting."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from phase1_extraction.models import DoseExpression, DoseSubgroup
from phase1_prediction.multidrug.observations import build_article_scopes
from phase1_prediction.multidrug.workflow import validate_boin_target
from phase1_simulation.multidrug import simulate_regimen_path
from phase1_simulation.service import BoinSettings, BoinDesignCandidate, compare_boin_designs
from test_cohort_exposures import extraction, write_article
from test_multidrug_transport import fixture


def scalar(value, unit):
    return DoseExpression(raw_expression=f'{value} {unit}', value_kind='scalar', value=value,
                          unit_raw=unit, unit_normalized=unit)


def prediction(truth):
    from phase1_prediction.multidrug import TARGET_PREDICTION_VERSION
    *_, target = fixture(1)
    return {'format': 'multidrug_target_prediction_v1', 'status': 'fitted',
            'prediction_engine_version': TARGET_PREDICTION_VERSION,
            'prediction_id': 'audit_prediction', 'backend': 'logistic',
            'synthesis': 'same_article_joint_posterior_mixture',
            'target': target.model_dump(mode='json'), 'posterior_draws': truth}


def test_subgroup_reference_converts_each_reported_unit_before_averaging(tmp_path):
    e = extraction(); e.cohorts = e.cohorts[:1]
    exposure = e.cohorts[0].exposures[0]
    exposure.dose = DoseExpression(raw_expression='1 or 2 mg', value_kind='discrete',
                                   discrete_values=[1, 2], unit_raw='mg', unit_normalized='mg')
    exposure.dose_semantics = 'subgroups'
    exposure.dose_subgroups = [
        DoseSubgroup(dose=scalar(1000, 'ug'), n=1, population_basis='dlt_evaluable'),
        DoseSubgroup(dose=scalar(2, 'mg'), n=2, population_basis='dlt_evaluable')]
    scopes, _ = build_article_scopes(write_article(tmp_path, e))
    assert scopes[0].drugs[0].unit == 'mg'
    assert scopes[0].drugs[0].reference_dose == pytest.approx(math.sqrt(2))
    assert scopes[0].observations[0].exposures[0].dose_subgroups == [(1., 1), (2., 2)]


def test_explicit_outcome_rejection_cannot_be_overridden_by_valid_counts(tmp_path):
    paths = write_article(tmp_path, extraction())
    payload = json.loads(paths.curve_rows.read_text())
    payload['all_rows'][0]['outcome_exclusion_reasons'] = ['Patients overlap another result']
    payload['all_rows'][0]['outcome_usable'] = False
    payload['all_rows'][0]['exclusion_reasons'] = []
    paths.curve_rows.write_text(json.dumps(payload))
    scopes, _ = build_article_scopes(paths)
    assert sum(o.n for s in scopes for o in s.observations) == 6
    assert any('Patients overlap another result' in r['reasons'] for s in scopes for r in s.exclusions)


def test_unknown_unit_cannot_be_arbitrarily_attached_to_one_incompatible_scope(tmp_path):
    e = extraction()
    for cohort, unit in zip(e.cohorts, ('mg', 'mg/kg', None)):
        cohort.exposures = cohort.exposures[:1]
        cohort.exposures[0].dose.unit_raw = unit
        cohort.exposures[0].dose.unit_normalized = unit
        cohort.dose.unit_raw = unit
        cohort.dose.unit_normalized = unit
    scopes, _ = build_article_scopes(write_article(tmp_path, e))
    assert len(scopes) == 3
    assert all(len(scope.observations) == 1 for scope in scopes)


@pytest.mark.parametrize('definition,expected_threshold', [
    ('acceptable_interval', .35), ('closest_to_target', .25), ('highest_not_above_target', .25)])
def test_overtoxicity_metrics_follow_selected_evaluation_definition(definition, expected_threshold):
    result = simulate_regimen_path(prediction([[.30, .30, .30]] * 3),
        settings=BoinSettings(mtd_definition=definition, check_lowest_dose=False,
                              elimination_cutoff=1., max_sample_size=3,
                              check_final_mtd_deesc_bound=False),
        posterior_curve_count=2, trials_per_curve=30, seed=17)
    expected_over = sum(t['selected_index'] is not None and .30 > expected_threshold for t in result['trials'])
    assert result['overtoxic_selection_probability'] == expected_over / result['total_trials']
    assert result['pooled_overtoxic_patient_fraction'] == float(.30 > expected_threshold)
    assert result['overtoxic_threshold'] == expected_threshold


def test_duplicate_regimen_labels_do_not_create_distinct_boin_levels():
    *_, target = fixture(1)
    target.points[1].doses = dict(target.points[0].doses)
    with pytest.raises(ValueError, match='distinct regimens'):
        validate_boin_target(target)


def test_design_candidates_share_unsorted_truth_scenarios_and_trial_streams():
    p = prediction([[.1, .4, .2], [.2, .6, .3], [.15, .3, .1], [.2, .1, .5]])
    settings = BoinSettings(max_sample_size=6)
    result = compare_boin_designs(p, method_key=None, target_dlt=.25,
        candidates=[BoinDesignCandidate('one', settings), BoinDesignCandidate('copy', replace(settings))],
        posterior_curve_count=3, trials_per_curve=4, seed=11)
    a, b = [r['result'] for r in result['ranked']]
    assert a['per_scenario'] == b['per_scenario']
    assert a['trials'] == b['trials']
    assert not a['truth_was_projected_or_sorted']
    assert a['nonmonotone_truth_fraction'] == 1.
    for row in a['per_scenario']:
        np.testing.assert_array_equal(row['truth'], p['posterior_draws'][row['posterior_draw_index']])


def test_direct_prediction_identity_is_preserved_in_design_comparison():
    p = prediction([[.1, .2, .3]] * 2)
    settings = BoinSettings(max_sample_size=3)
    options = dict(method_key=None, target_dlt=.25,
                   candidates=[BoinDesignCandidate('one', settings)],
                   posterior_curve_count=1, trials_per_curve=1)
    first = compare_boin_designs(p, **options)
    p['prediction_id'] = 'another_prediction'
    second = compare_boin_designs(p, **options)
    assert first['prediction_run_id'] == 'audit_prediction'
    assert second['prediction_run_id'] == 'another_prediction'
    assert first['run_id'] != second['run_id']
    assert first['simulation_model_version'] == first['ranked'][0]['result']['simulation_model_version']
