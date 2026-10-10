"""No hidden external evidence, source reuse, or false precision in the pilot."""
import numpy as np
import pytest
from scipy.special import logit

from phase1_prediction.multidrug.contracts import Exposure, Observation
from phase1_prediction.multidrug.interaction_pilot import (
    PairPriorScenario, interaction_evidence_gate, predict_primary_first_pilot,
)
from test_component_assembly_methods import mono_library


def library(extra=False):
    entries, target = mono_library(extra_a=extra)
    for scope, _, _ in entries:
        sid = scope.drugs[0].model_drug_id
        scope.observations = [Observation(observation_id=scope.scope_id, cohort_id=scope.scope_id,
            n=10, y=1, exposures=[Exposure(drug_id=sid, presence='present', dose=100.)])]
        target.mapping_overrides[scope.scope_id] = {sid: sid}
    return entries, target


@pytest.mark.parametrize('aggregation', ['average', 'mixture'])
def test_interaction_scenarios_preserve_draws_and_shift_each_complete_curve(aggregation):
    entries, target = library()
    baseline = predict_primary_first_pilot(entries, target, aggregation=aggregation, draw_count=100)
    shifted = predict_primary_first_pilot(entries, target, aggregation=aggregation, draw_count=100,
        scenario=PairPriorScenario(mean=.6, basis='sensitivity_scenario'))
    assert baseline['status'] == shifted['status'] == 'fitted'
    assert baseline['allocation'] == shifted['allocation']
    assert baseline['component_pools'] == shifted['component_pools']
    r = np.array([.5, 1., 2.])
    np.testing.assert_allclose(logit(shifted['posterior_draws']) - logit(baseline['posterior_draws']),
        np.broadcast_to(.6 * 2*r/(1+r), (100, 3)), atol=1e-10)
    assert not shifted['external_interaction_data_used']
    assert not shifted['target_outcomes_used']
    assert shifted['prediction_id'] != baseline['prediction_id']


def test_scenario_average_is_after_assembly_and_same_eta_shared_across_recipes():
    entries, target = library(extra=True)
    base = predict_primary_first_pilot(entries, target, aggregation='average', draw_count=200,
        scenario=PairPriorScenario(mean=.5, sd=.3, basis='sensitivity_scenario'))
    # Full eta uncertainty persists across the two alternative A sources.
    draws = [row['prior_interactions'][0]['value'] for row in base['provenance']]
    assert .25 < np.std(draws) < .35
    assert base['assembly_computation']['evaluated_recipe_count'] == 2
    assert base['pair_prior_scenario']['basis'] == 'sensitivity_scenario'


def test_source_article_identity_conflict_is_rejected_across_fits():
    entries, target = library()
    s0, s1 = entries[0][0], entries[1][0]
    s1.document_id = s0.document_id
    s1.drugs[0].name = s0.drugs[0].name
    s1.drugs[0].profile['name'] = s0.drugs[0].name
    with pytest.raises(ValueError, match='one article drug entity'):
        predict_primary_first_pilot(entries, target)


def test_no_curve_when_secondary_has_no_support_and_no_hidden_prior_component():
    entries, target = library()
    target.mapping_overrides = {entries[0][0].scope_id: {'A': 'A'}}
    result = predict_primary_first_pilot(entries[:1], target)
    assert result['status'] == 'no_similarity_support'
    assert result['posterior_draws'] == []
    assert result['allocation']['unsupported_target_ids'] == ['B']


@pytest.mark.parametrize('kind', ['clinical_dlt_counts', 'pk_exposure_ratio',
    'ddi_description', 'faers_signal', 'cell_efficacy_synergy'])
def test_no_website_score_silently_becomes_numeric_prior(kind):
    assert interaction_evidence_gate(kind)['automatic_numeric_prior'] is False


def test_invalid_or_mislabelled_priors_rejected():
    for value in [dict(mean=1), dict(sd=0), dict(mean=float('nan')),
                  dict(basis='drugcombo_estimate'), dict(eta_from_zip=4)]:
        with pytest.raises(ValueError):
            PairPriorScenario.model_validate(value)


def test_joint_sources_cannot_have_their_posterior_eta_replaced():
    from test_multidrug_transport import fixture
    from phase1_prediction.multidrug.component_assembly import assemble_posteriors
    scope, arrays, manifest, target = fixture()
    with pytest.raises(ValueError, match='cannot replace'):
        assemble_posteriors([(scope, arrays, manifest)], target,
            pair_prior_scenario=PairPriorScenario())
