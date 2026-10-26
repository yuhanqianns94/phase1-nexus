import jax.numpy as jnp
import pytest

from phase1_prediction.multidrug.math import (baseline_risk, component_risks, interaction_sets,
    logistic_risk)


def test_logistic_pairwise_interactions_and_exact_absence():
    assert interaction_sets(5) == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
                                  (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    q = jnp.array([.1, .08, .05])
    p0 = float(baseline_risk(q))
    expected = 1 / (1 + (1-p0)/p0 * jnp.exp(-.4))
    assert float(logistic_risk(q, jnp.ones(3), jnp.array([.3, .2, -.1]))) == pytest.approx(float(expected))
    for k in (1, 2, 3, 4):
        qs = jnp.arange(1, k + 1) / 100
        assert float(logistic_risk(qs, jnp.ones(k), jnp.zeros(len(interaction_sets(k))))) == pytest.approx(float(baseline_risk(qs)))
    assert float(logistic_risk(jnp.array([.1, .08, 0]), jnp.array([1., 1., 0.]), jnp.array([.3, 100., 100.]))) == pytest.approx(
        float(logistic_risk(q[:2], jnp.ones(2), jnp.array([.3]))))
    assert float(logistic_risk(jnp.zeros(3), jnp.zeros(3), jnp.ones(3))) == 0


def test_pairwise_model_is_invariant_to_drug_order_and_includes_later_pairs():
    import itertools
    import numpy as np
    from scipy.special import expit, logit
    from phase1_prediction.multidrug.math import interaction_basis
    q = np.array([.1, .2, .3, .15])
    r = np.array([.5, 2., 1.5, 3.])
    pairs = interaction_sets(4)
    eta = np.array([.1, -.3, .7, -.2, .05, .4])
    expected_shift = sum(v * 2*r[i]*r[j]/(1+r[i]*r[j])
                         for (i, j), v in zip(pairs, eta))
    expected = expit(logit(1-np.prod(1-q)) + expected_shift)
    assert float(logistic_risk(q, r, eta)) == pytest.approx(expected)
    for permutation in itertools.permutations(range(4)):
        ordered_eta = np.array([eta[pairs.index(tuple(sorted((permutation[i], permutation[j]))))]
                                for i, j in pairs])
        actual = logistic_risk(q[list(permutation)], r[list(permutation)], ordered_eta)
        assert float(actual) == pytest.approx(expected)
    assert np.asarray(interaction_basis(r)).shape == (6,)


def test_retired_three_way_prior_is_removed_without_reusing_legacy_posteriors(tmp_path):
    import json
    from phase1_prediction.multidrug.contracts import PriorSpec
    from phase1_prediction.multidrug.prior_presets import canonical_prior
    from phase1_prediction.multidrug.store import PosteriorStore
    from phase1_prediction.multidrug.power_prior import PowerPriorStore
    legacy = {'prior_id': 'multidrug_weak_v1', 'triple_sd': .75, 'pair_sd': .4}
    prior = PriorSpec.model_validate(legacy)
    assert 'triple_sd' not in prior.model_dump()
    assert prior.pair_sd == .4
    assert canonical_prior(legacy, 'logistic') == canonical_prior({'pair_sd': .4}, 'logistic')
    assert 'triple_sd' in legacy
    for identity, store, manifest in [
        ('fit_old', PosteriorStore(tmp_path), {'source_fit_id':'fit_old', 'model_version':'v3_multidrug_bayes_2'}),
        ('power_old', PowerPriorStore(tmp_path), {'target_fit_id':'power_old', 'version':'fixed_partial_power_prior_v3'})
    ]:
        directory = store.root / identity
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match='version|Incompatible'):
            store.load(identity)
