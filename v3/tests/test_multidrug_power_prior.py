"""Mathematical and workflow boundaries of target-only historical borrowing."""
import copy

import jax
import numpy as np
import pytest
from numpyro import handlers
from scipy.special import expit, logit
from scipy.stats import binom

from phase1_prediction.multidrug.contracts import Exposure, Observation, PriorSpec, SamplerSpec, RegimenPoint
from phase1_prediction.multidrug.mapping import match_target
from phase1_prediction.multidrug import power_prior as pp, service
from phase1_simulation.multidrug import simulate_regimen_path
from phase1_simulation.service import BoinSettings
from test_multidrug_transport import fixture


def article(k=2, doc='paper', strength=.8):
    scope, arrays, manifest, target = fixture(k)
    scope.document_id = doc
    scope.scope_id = doc + '_scope'
    scope.observations = [Observation(observation_id=f'obs_{i}', cohort_id=f'cohort_{i}', y=y, n=10,
        source_row_ids=[f'row_{i}'], exposures=[Exposure(drug_id=d.model_drug_id, presence='present',
        dose=dose) for d in scope.drugs]) for i, (dose, y) in enumerate(((50., 1), (100., 2), (200., 4)))]
    mapping = match_target(target, scope)
    mapping['score'].update(similarity=strength)
    return (scope, mapping, strength), target


def test_absolute_powers_budget_and_review_are_not_renormalized():
    first, target = article(doc='one')
    second, _ = article(doc='two', strength=.4)
    spec = pp.PowerPriorSpec(patient_budget=18.)
    included = pp.prepare_power_prior([first, second], target, spec)
    uncertain = pp.prepare_power_prior([first, second], target, spec, {'one': 'Uncertain', 'two': 'Uncertain'})
    excluded = pp.prepare_power_prior([first, second], target, spec, {'one': 'Include', 'two': 'Exclude'})
    assert included['cap_factor'] == .5
    assert included['discounted_patient_count'] == pytest.approx(18.)
    assert uncertain['discounted_patient_count'] == pytest.approx(9.)
    assert [x['power'] for x in excluded['sources']] == pytest.approx([.4, 0.])
    assert sum(x['power'] for x in included['sources']) == pytest.approx(.6)


def test_one_scope_per_article_preserves_source_and_avoids_duplicate_patients():
    first, target = article()
    snapshot = first[0].model_dump()
    weaker = copy.deepcopy(first)
    weaker[0].scope_id = 'other_context'
    prepared = pp.prepare_power_prior([first, (weaker[0], weaker[1], .4)], target, pp.PowerPriorSpec())
    assert len(prepared['sources']) == 1
    assert prepared['sources'][0]['patient_count'] == 30
    assert first[0].model_dump() == snapshot
    scope = first[0].model_copy(deep=True)
    scope.observations.append(scope.observations[0].model_copy(deep=True))
    rejected = pp.prepare_power_prior([(scope, first[1], first[2])], target, pp.PowerPriorSpec())
    assert not rejected['sources']
    assert 'twice' in rejected['excluded'][0]['reason']


def test_manual_permutation_and_source_relative_coordinates_are_used():
    row, target = article()
    source = row[0]
    source.drugs[1].reference_dose = 200.
    target.mapping_overrides = {source.scope_id: {'A': 'B', 'B': 'A'}}
    mapped = match_target(target, source)
    prepared = pp.prepare_power_prior([(source, mapped, .8)], target, pp.PowerPriorSpec())
    observation = prepared['sources'][0]['scope']['observations'][1]
    assert [(e['drug_id'], e['dose']) for e in observation['exposures']] == [('A', .5), ('B', 1.)]
    assert prepared['sources'][0]['mapping']['pairs'] == {'A': 'B', 'B': 'A'}


def test_same_drug_unit_conversion_uses_target_reference_in_source_units():
    row, target = article(k=1)
    target.drugs[0].transport_mode = 'same_entity_unit_conversion'
    target.drugs[0].unit = 'g'
    target.drugs[0].reference_dose = .2
    prepared = pp.prepare_power_prior([row], target, pp.PowerPriorSpec())
    assert prepared['sources'][0]['scope']['observations'][1]['exposures'][0]['dose'] == pytest.approx(.5)


def test_no_silent_component_removal_or_background_numeric_anchor():
    row, target = article()
    single = target.model_copy(deep=True)
    single.drugs = single.drugs[:1]
    single.points = [RegimenPoint(level_id='one', doses={'A': 50.})]
    mapping = match_target(single, row[0])
    result = pp.prepare_power_prior([(row[0], mapping, .8)], single, pp.PowerPriorSpec())
    assert not result['sources'] and 'removed' in result['excluded'][0]['reason']
    row[0].drugs[1].mode = 'background'
    target.drugs[1].allow_background_anchor = True
    result = pp.prepare_power_prior([row], target, pp.PowerPriorSpec())
    assert not result['sources'] and 'anchoring' in result['excluded'][0]['reason']


@pytest.mark.parametrize('backend', ['logistic'])
@pytest.mark.parametrize('k', [1, 2, 3])
def test_power_applies_to_outcome_once_and_not_parameter_or_latent_priors(backend, k):
    row, target = article(k=k)
    prepared = pp.prepare_power_prior([row], target, pp.PowerPriorSpec())
    scope = pp.ModelScope.model_validate(prepared['sources'][0]['scope'])
    scope.observations[0].exposures[0].dose = None
    scope.observations[1].exposures[0].presence = 'unknown'
    args = (backend, PriorSpec(), pp.PowerPriorSpec())
    traces = [handlers.trace(handlers.seed(pp.power_prior_model, rng_seed=3)).get_trace([scope], [power], *args)
              for power in (1., .4)]
    for name in traces[0]:
        if traces[0][name]['type'] != 'sample':
            continue
        a, b = traces[0][name], traces[1][name]
        if name.startswith('article_0/cohort_'):
            assert float(b['fn'].log_factor) == pytest.approx(.4 * float(a['fn'].log_factor))
        else:
            np.testing.assert_allclose(a['value'], b['value'])
            np.testing.assert_allclose(a['fn'].log_prob(a['value']), b['fn'].log_prob(b['value']))
    assert 'article_0/missing_dose_0_0' in traces[0]
    assert sum(name == 'a' for name in traces[0]) == 1


def test_simple_binary_likelihood_matches_hand_calculation_and_zero_power_omits_it():
    row, target = article(k=1)
    prepared = pp.prepare_power_prior([row], target, pp.PowerPriorSpec())
    scope = pp.ModelScope.model_validate(prepared['sources'][0]['scope'])
    fixed = {'a': np.array([logit(.2)]), 'log_b_numeric': np.zeros(1), 'tau': .5, 'study_z': np.array([0.])}
    traced = handlers.trace(handlers.seed(handlers.substitute(pp.power_prior_model, data=fixed), rng_seed=4))
    trace = traced.get_trace([scope], [.5], 'logistic', PriorSpec(), pp.PowerPriorSpec())
    assert float(trace['article_0/cohort_1']['fn'].log_factor) == pytest.approx(.5 * binom.logpmf(2, 10, .2))
    zero = traced.get_trace([scope], [0.], 'logistic', PriorSpec(), pp.PowerPriorSpec())
    assert not any('cohort_' in name for name in zero)


def fake_fit(prepared, backend, prior, sampler):
    k = len(prepared['sources'][0]['scope']['drugs'])
    _, arrays, _, _ = fixture(k, backend)
    arrays['tau'] = np.full(arrays['a'].shape[:2], .4)
    return arrays, {'status': 'fitted', 'attempts': [], 'inference': 'historical_power_prior_predictive'}


@pytest.mark.parametrize('backend', ['logistic'])
def test_target_cache_frozen_curves_and_boin_do_not_refit_sources(tmp_path, monkeypatch, backend):
    row, target = article()
    calls = []
    def fit(*args):
        calls.append(1)
        return fake_fit(*args)
    monkeypatch.setattr(pp, 'fit_power_prior', fit)
    monkeypatch.setattr(service, '_select', lambda *a, **kw: ([row], {'notices': []}))
    monkeypatch.setattr(service, 'fit_library', lambda *a, **kw: pytest.fail('Source fitting is forbidden here'))
    kwargs = dict(backend=backend, synthesis=pp.METHOD, store_root=tmp_path, draw_count=8)
    result = service.predict(None, target, **kwargs).to_dict()
    again = service.predict(None, target, **kwargs).to_dict()
    method = result['methods'][pp.METHOD]
    assert len(calls) == 1 and method == again['methods'][pp.METHOD]
    assert not result['evidence_contract']['source_fit_ids']
    changed = target.model_copy(deep=True)
    changed.points[1].doses = {'A': 75., 'B': 50.}
    redraw = service.evaluate_frozen_prediction(result, changed)['methods'][pp.METHOD]
    assert redraw['provenance'] == method['provenance']
    np.testing.assert_allclose(np.asarray(redraw['posterior_draws'])[:, [0, 2]], np.asarray(method['posterior_draws'])[:, [0, 2]])
    assert not np.allclose(np.asarray(redraw['posterior_draws'])[:, 1], np.asarray(method['posterior_draws'])[:, 1])
    assert len(calls) == 1
    changed.drugs[0].name = 'different'
    with pytest.raises(ValueError, match='frozen'):
        service.evaluate_frozen_prediction(result, changed)
    sim = simulate_regimen_path(result, method_key=pp.METHOD, settings=BoinSettings(), posterior_curve_count=2, trials_per_curve=2)
    assert sim['total_trials'] == 4 and sim['synthesis'] == pp.METHOD
    assert all('target_random_intercept' in x for x in sim['draw_provenance'])
    fit_id = result['evidence_contract']['target_fit_id']
    with (tmp_path / 'target_power_priors' / fit_id / 'draws.npz').open('ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        pp.PowerPriorStore(tmp_path).load(fit_id)


def test_new_study_intercept_is_shared_across_whole_curve():
    row, target = article(k=1)
    prepared = pp.prepare_power_prior([row], target, pp.PowerPriorSpec())
    arrays, diagnostic = fake_fit(prepared, 'logistic', PriorSpec(), SamplerSpec())
    manifest = {**diagnostic, 'target_fit_id': 'power_test', 'backend': 'logistic', 'sampler': {}}
    result = pp.evaluate_power_prior(prepared, arrays, manifest, target, draw_count=20, seed=4)
    for curve, provenance in zip(result['posterior_draws'], result['provenance']):
        expected = expit(-2 + provenance['target_random_intercept'] + np.log([.5, 1, 2]))
        np.testing.assert_allclose(curve, expected)


def test_zero_strength_generates_explicit_prior_scenarios_and_nuts_initializes():
    from numpyro.infer.util import initialize_model
    row, target = article()
    prepared = pp.prepare_power_prior([row], target, pp.PowerPriorSpec(strength=0))
    arrays, diagnostics = pp.fit_power_prior(prepared, 'logistic', PriorSpec(), SamplerSpec(samples=4))
    assert diagnostics['inference'] == 'explicit_prior_predictive'
    assert arrays['a'].shape == (2, 4, 2)
    scope = pp.ModelScope.model_validate(prepared['sources'][0]['scope'])
    initialized = initialize_model(jax.random.PRNGKey(5), pp.power_prior_model,
        model_args=([scope], [.5], 'logistic', PriorSpec(), pp.PowerPriorSpec()))
    assert np.isfinite(initialized.param_info.potential_energy)


def test_relative_redraw_can_acquire_actual_units_but_cannot_change_same_entity_reference():
    row, target = article(k=1)
    signature = pp.target_signature(target)
    target.drugs[0].reference_dose = 800.
    assert pp.target_signature(target) == signature
    target.drugs[0].transport_mode = 'same_entity_unit_conversion'
    signature = pp.target_signature(target)
    target.drugs[0].reference_dose = 900.
    assert pp.target_signature(target) != signature


def test_cache_identity_changes_for_data_discount_prior_and_not_grid():
    row, target = article()
    settings = pp.PowerPriorSpec()
    prepared = pp.prepare_power_prior([row], target, settings)
    fid = pp.PowerPriorStore.identity(prepared, 'logistic', PriorSpec(), SamplerSpec())
    target.points[0].doses['A'] = 80.
    assert pp.PowerPriorStore.identity(pp.prepare_power_prior([row], target, settings), 'logistic', PriorSpec(), SamplerSpec()) == fid
    for variant in (pp.PowerPriorSpec(strength=.5), pp.PowerPriorSpec(heterogeneity_scale=1.)):
        assert pp.PowerPriorStore.identity(pp.prepare_power_prior([row], target, variant), 'logistic', PriorSpec(), SamplerSpec()) != fid
    row[0].observations[0].y += 1
    assert pp.PowerPriorStore.identity(pp.prepare_power_prior([row], target, settings), 'logistic', PriorSpec(), SamplerSpec()) != fid


@pytest.mark.parametrize('backend', ['logistic'])
def test_three_drug_multiple_article_nuts_has_finite_gradient(backend):
    from numpyro.infer.util import initialize_model
    first, target = article(k=3, doc='one')
    second, _ = article(k=3, doc='two')
    second[0].observations[0].exposures[1].presence = 'absent'
    second[0].observations[1].exposures[2].presence = 'mixed'
    second[0].observations[1].exposures[2].exposed_n = 4
    prepared = pp.prepare_power_prior([first, second], target, pp.PowerPriorSpec())
    scopes = [pp.ModelScope.model_validate(c['scope']) for c in prepared['sources']]
    initialized = initialize_model(jax.random.PRNGKey(5), pp.power_prior_model,
        model_args=(scopes, [.5, .3], backend, PriorSpec(), pp.PowerPriorSpec()))
    assert np.isfinite(initialized.param_info.potential_energy)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(initialized.param_info.z_grad))
