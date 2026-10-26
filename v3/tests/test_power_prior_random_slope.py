"""Oracle checks for article slopes and whole-curve new-study prediction."""
import numpy as np
import pytest
from numpyro import handlers
from scipy.special import expit
from scipy.stats import binom

from phase1_prediction.multidrug import power_prior as pp, service
from phase1_prediction.multidrug.contracts import PriorSpec, SamplerSpec
from test_single_drug_pipeline import example, arrays, BACKEND


def prepared_example(**settings):
    scored, target = example()
    second = scored[0][0].model_copy(deep=True)
    second.document_id = 'second'
    second.scope_id = 'second_scope'
    scored.append((second, scored[0][1], scored[0][2]))
    projected, target, _ = service.prepare_single_drug_evidence(scored, target)
    prepared = pp.prepare_power_prior(projected, target, pp.PowerPriorSpec(**settings))
    return prepared, target


def test_article_random_slopes_enter_each_cohort_likelihood_once():
    prepared, _ = prepared_example()
    scopes = [pp.ModelScope.model_validate(item['scope']) for item in prepared['sources']]
    fixed = {'alpha': -2., 'ell': .1, 'delta': .4, 'tau': .3, 'tau_slope': .5,
             'study_z': np.array([1., -1.]), 'study_slope_z': np.array([-.8, .8])}
    traces = [handlers.trace(handlers.substitute(handlers.seed(pp.power_prior_model, 7), data=fixed)).get_trace(
        scopes, powers, BACKEND, PriorSpec(), pp.PowerPriorSpec()) for powers in ([1., 1.], [.25, .7])]
    for article, power in enumerate((.25, .7)):
        factors = [name for name in traces[0] if name.startswith(f'article_{article}/cohort_')]
        assert len(factors) == 3
        for dose_index, name in enumerate(factors):
            probability = expit(-2 + .3 * fixed['study_z'][article]
                + np.exp(.1 + .5 * fixed['study_slope_z'][article]) * (.5, 1., 2.)[dose_index] + .4)
            expected = binom.logpmf(dose_index + 1, 10, probability)
            for trace, multiplier in zip(traces, (1., power)):
                factor = trace[name]
                np.testing.assert_allclose(factor['fn'].log_prob(factor['value']), multiplier * expected)
    # Discounting applies only to outcomes, not to either article-effect prior.
    for name in ('tau', 'tau_slope', 'study_z', 'study_slope_z'):
        first, second = (trace[name] for trace in traces)
        assert first['scale'] is None and second['scale'] is None
        np.testing.assert_allclose(first['fn'].log_prob(first['value']), second['fn'].log_prob(second['value']))


def test_new_study_has_one_positive_slope_per_whole_frozen_curve():
    prepared, target = prepared_example()
    samples = arrays()
    samples['tau'][:] = .25
    samples['tau_slope'][:] = .6
    manifest = {'target_fit_id': 'power_test', 'backend': BACKEND, 'status': 'fitted'}
    result = pp.evaluate_power_prior(prepared, samples, manifest, target, draw_count=50, seed=19)
    for curve, origin in zip(result['posterior_draws'], result['provenance']):
        slope = np.exp(origin['target_random_log_slope'])
        expected = expit(-2 + origin['target_random_intercept'] + slope * np.array([.5, 1., 2.]) + .4)
        np.testing.assert_allclose(curve, expected)
        assert np.all(np.diff(curve) > 0)
    assert np.std([row['target_random_log_slope'] for row in result['provenance']]) > 0
    assert result['power_prior_version'] == pp.POWER_PRIOR_VERSION
    revised = target.model_copy(deep=True)
    revised.points[0].doses[revised.drugs[0].drug_id] = .25 * revised.drugs[0].reference_dose
    redrawn = pp.evaluate_power_prior(prepared, samples, manifest, revised, draw_count=50, seed=19)
    assert result['provenance'] == redrawn['provenance']
    np.testing.assert_array_equal(np.asarray(result['posterior_draws'])[:, 1:], np.asarray(redrawn['posterior_draws'])[:, 1:])
    samples['tau_slope'][:] = 0.
    fixed = pp.evaluate_power_prior(prepared, samples, manifest, target, draw_count=50, seed=19)
    assert all(row['target_random_log_slope'] == 0. for row in fixed['provenance'])


def test_prior_only_and_nuts_initialization_include_slope_heterogeneity():
    import jax
    from numpyro.infer.util import initialize_model
    prepared, _ = prepared_example(strength=0.)
    samples, diagnostics = pp.fit_power_prior(prepared, BACKEND, PriorSpec(), SamplerSpec(samples=4))
    assert diagnostics['inference'] == 'explicit_prior_predictive'
    assert samples['tau_slope'].shape == (2, 4)
    assert np.all(samples['tau_slope'] > 0)
    assert 'study_slope_z' not in samples
    scopes = [pp.ModelScope.model_validate(item['scope']) for item in prepared['sources']]
    initialized = initialize_model(jax.random.PRNGKey(5), pp.power_prior_model,
        model_args=(scopes, [.5, .3], BACKEND, PriorSpec(), pp.PowerPriorSpec()))
    assert np.isfinite(initialized.param_info.potential_energy)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(initialized.param_info.z_grad))


def test_slope_prior_separates_cache_and_old_draws_cannot_silently_reuse():
    prepared, target = prepared_example()
    changed, _ = prepared_example(slope_heterogeneity_scale=.8)
    identity = lambda value: pp.PowerPriorStore.identity(value, BACKEND, PriorSpec(), SamplerSpec())
    assert identity(prepared) != identity(changed)
    old = arrays()
    del old['tau_slope']
    with pytest.raises(ValueError, match='predates random slopes'):
        pp.evaluate_power_prior(prepared, old, {'target_fit_id': 'old', 'backend': BACKEND, 'status': 'fitted'},
                                target, draw_count=10, seed=1)
