"""Ordinary Bayesian product interactions: likelihood, priors, and predictions."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from numpyro.infer.util import initialize_model, log_density
from scipy.special import expit
from scipy.stats import binom, norm

from phase1_prediction.multidrug import power_prior as pp
from phase1_prediction.multidrug.contracts import Exposure, Observation, PriorSpec, SamplerSpec
from phase1_prediction.multidrug.fitting import fit_source, source_model
from phase1_prediction.multidrug.math import product_interaction_basis, product_logistic_risk
from phase1_prediction.multidrug.prior_presets import canonical_prior, risk_prior_preset
from phase1_prediction.multidrug.store import source_fit_id
from test_multidrug_power_prior import article
from test_multidrug_transport import fixture


def test_product_formula_first_three_interactions_zero_baseline_and_negative_derivative():
    r = jnp.array([[0., 0., 0., 0.], [2., 3., 4., 5.]])
    basis = np.asarray(product_interaction_basis(r))
    np.testing.assert_allclose(basis, [[0., 0., 0., 0.], [6., 8., 12., 24.]])
    alpha, slopes, eta = -3., jnp.log(jnp.array([.5, .6, .7, .8])), jnp.array([.1, -.2, .3, -.1])
    actual = product_logistic_risk(alpha, slopes, r, eta)
    np.testing.assert_allclose(actual, expit(alpha + np.asarray(r) @ np.exp(slopes) + basis @ eta))
    assert float(actual[0]) == pytest.approx(expit(alpha))
    # Positive main effects cannot offset every possible negative interaction.
    derivative = jax.grad(lambda x: product_logistic_risk(-2., jnp.zeros(2),
        jnp.array([x, 2.]), jnp.array([-1.])))(1.)
    assert float(derivative) < 0


def test_source_log_posterior_equals_normal_priors_plus_joint_binomial_likelihood():
    row, _ = article()
    scope = row[0]
    prior = PriorSpec(intercept_mean=-3., intercept_sd=1.2, log_slope_mean=-.2,
                      log_slope_sd=.4, pair_sd=.7)
    fixed = {'alpha': jnp.array(-2.5), 'log_b_effects': jnp.array([-.3, -.6]), 'eta': jnp.array([.15])}
    actual, trace = log_density(source_model, (scope, 'logistic_product', prior), {}, fixed)
    expected = (norm.logpdf(fixed['alpha'], prior.intercept_mean, prior.intercept_sd)
                + norm.logpdf(fixed['log_b_effects'], prior.log_slope_mean, prior.log_slope_sd).sum()
                + norm.logpdf(fixed['eta'], 0, prior.pair_sd).sum())
    for observation in scope.observations:
        r = np.array([e.dose / d.reference_dose for d, e in zip(scope.drugs, observation.exposures)])
        probability = expit(float(fixed['alpha']) + np.exp(fixed['log_b_effects']) @ r
                            + float(fixed['eta'][0]) * np.prod(r))
        expected += binom.logpmf(observation.y, observation.n, probability)
    assert float(actual) == pytest.approx(expected)
    assert 'a' not in trace and 'alpha' in trace
    assert all(trace[name]['type'] == 'sample' for name in ('alpha', 'log_b_effects', 'eta'))


def test_background_has_estimated_main_effect_and_absent_state_retains_shared_intercept():
    row, _ = article(k=1)
    scope = row[0]
    scope.drugs[0].mode = 'background'
    scope.drugs[0].reference_dose = None
    scope.drugs[0].unit = None
    scope.observations = scope.observations[:2]
    for observation, presence in zip(scope.observations, ['present', 'absent']):
        observation.exposures[0].presence = presence
        observation.exposures[0].dose = None
    fixed = {'alpha': jnp.array(-2.), 'log_b_effects': jnp.array([np.log(.6)])}
    trace = handlers.trace(handlers.substitute(source_model, data=fixed)).get_trace(
        scope, 'logistic_product', PriorSpec())
    for i, probability in enumerate([expit(-2. + .6), expit(-2.)]):
        factor = trace[f'cohort_{i}']
        actual = float(factor['fn'].log_prob(factor['value']))
        assert actual == pytest.approx(binom.logpmf(scope.observations[i].y, 10, probability))


def test_source_prior_arrays_and_cache_identity_separate_probability_models():
    scope, _, _, _ = fixture(3)
    prior = canonical_prior(risk_prior_preset('regularized', 'logistic_product'), 'logistic_product')
    sampler = SamplerSpec(samples=4, warmup=1)
    arrays, diagnostics = fit_source(scope, 'logistic_product', prior, sampler)
    assert diagnostics['status'] == 'prior_only'
    assert arrays['alpha'].shape == (2, 4)
    assert arrays['log_b'].shape == (2, 4, 3)
    assert arrays['eta'].shape == (2, 4, 4)
    assert 'a' not in arrays
    assert source_fit_id(scope, 'logistic_product', prior, sampler) != source_fit_id(scope, 'logistic', prior, sampler)


def test_power_tempering_applies_only_to_outcomes_and_shifts_one_alpha():
    row, target = article()
    spec, prior = pp.PowerPriorSpec(), PriorSpec()
    prepared = pp.prepare_power_prior([row], target, spec)
    scope = pp.ModelScope.model_validate(prepared['sources'][0]['scope'])
    fixed = {'alpha': jnp.array(-3.), 'log_b_effects': jnp.array([-.4, -.7]),
             'eta': jnp.array([.2]), 'tau': jnp.array(.5), 'study_z': jnp.array([.6])}
    traced = handlers.trace(handlers.substitute(pp.power_prior_model, data=fixed))
    traces = [traced.get_trace([scope], [power], 'logistic_product', prior, spec) for power in (.25, 1.)]
    for name in ('alpha', 'log_b_effects', 'eta', 'tau', 'study_z'):
        a, b = [trace[name] for trace in traces]
        np.testing.assert_allclose(a['fn'].log_prob(a['value']), b['fn'].log_prob(b['value']))
    observation = scope.observations[0]
    r = np.array([e.dose for e in observation.exposures])
    p = expit(-3. + .5 * .6 + np.exp(fixed['log_b_effects']) @ r + .2 * np.prod(r))
    for trace, power in zip(traces, [.25, 1.]):
        factor = trace['article_0/cohort_0']
        assert float(factor['fn'].log_prob(factor['value'])) == pytest.approx(power * binom.logpmf(observation.y, observation.n, p))


def test_power_zero_information_prediction_retains_scalar_alpha_and_whole_curve_draws():
    row, target = article()
    spec = pp.PowerPriorSpec(strength=0)
    prepared = pp.prepare_power_prior([row], target, spec)
    arrays, diagnostics = pp.fit_power_prior(prepared, 'logistic_product', PriorSpec(), SamplerSpec(samples=4))
    assert arrays['alpha'].shape == (2, 4)
    manifest = {**diagnostics, 'backend': 'logistic_product', 'target_fit_id': 'power_fixture'}
    result = pp.evaluate_power_prior(prepared, arrays, manifest, target, draw_count=10, seed=7)
    assert result['status'] == 'fitted'
    r = pp._target_coordinates(target)
    alpha, slopes, eta = arrays['alpha'].reshape(-1), arrays['log_b'].reshape(-1, 2), arrays['eta'].reshape(-1, 1)
    for prediction, trace in zip(result['posterior_draws'], result['provenance']):
        index = trace['flat_draw']
        expected = product_logistic_risk(alpha[index] + trace['target_random_intercept'], slopes[index], r, eta[index])
        np.testing.assert_allclose(prediction, expected)
    assert result == pp.evaluate_power_prior(prepared, arrays, manifest, target, draw_count=10, seed=7)


def test_three_drug_power_model_has_finite_nuts_gradient():
    first, target = article(k=3, doc='first')
    second, _ = article(k=3, doc='second')
    second[0].observations[0].exposures[1].presence = 'absent'
    second[0].observations[1].exposures[2].presence = 'mixed'
    second[0].observations[1].exposures[2].exposed_n = 4
    prepared = pp.prepare_power_prior([first, second], target, pp.PowerPriorSpec())
    scopes = [pp.ModelScope.model_validate(c['scope']) for c in prepared['sources']]
    initialized = initialize_model(jax.random.PRNGKey(8), pp.power_prior_model,
        model_args=(scopes, [.5, .3], 'logistic_product', PriorSpec(), pp.PowerPriorSpec()))
    assert np.isfinite(initialized.param_info.potential_energy)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(initialized.param_info.z_grad))


def test_joint_source_nuts_updates_alpha_slopes_and_interaction():
    scope, _, _, _ = fixture(2)
    scope.observations = []
    for i, (a, b) in enumerate(itertools.product([0., .5, 1., 2.], repeat=2)):
        p = expit(-3. + .6 * a + .5 * b + .2 * a * b)
        scope.observations.append(Observation(observation_id=str(i), cohort_id=str(i), y=round(60 * p), n=60,
            exposures=[Exposure(drug_id=d.model_drug_id, presence='present' if r else 'absent',
                                dose=100 * r if r else None) for d, r in zip(scope.drugs, (a, b))]))
    prior = PriorSpec(intercept_mean=-2., intercept_sd=1., log_slope_sd=.5, pair_sd=.5)
    arrays, diagnostics = fit_source(scope, 'logistic_product', prior,
        SamplerSpec(warmup=150, samples=200, chains=2, retry_count=0, min_ess=20, max_rhat=1.15, seed=81))
    assert diagnostics['status'] == 'fitted', diagnostics
    assert arrays['alpha'].mean() < -2.5
    assert arrays['log_b'].shape == (2, 200, 2)
    assert arrays['eta'].shape == (2, 200, 1)
    assert np.std(arrays['eta']) > 0
