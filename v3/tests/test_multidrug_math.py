import jax.numpy as jnp
import pytest

from phase1_prediction.multidrug.math import (baseline_risk, component_risks, interaction_sets,
    logistic_risk)


def test_logistic_four_interactions_and_exact_absence():
    assert interaction_sets(5) == [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
    q = jnp.array([.1, .08, .05])
    assert float(logistic_risk(q, jnp.ones(3), jnp.array([.3, .2, -.1, .1]))) == pytest.approx(.3090527471773955)
    for k in (1, 2, 3, 4):
        qs = jnp.arange(1, k + 1) / 100
        assert float(logistic_risk(qs, jnp.ones(k), jnp.zeros(len(interaction_sets(k))))) == pytest.approx(float(baseline_risk(qs)))
    assert float(logistic_risk(jnp.array([.1, .08, 0]), jnp.array([1., 1., 0.]), jnp.array([.3, 100., 100., 100.]))) == pytest.approx(
        float(logistic_risk(q[:2], jnp.ones(2), jnp.array([.3]))))
    assert float(logistic_risk(jnp.zeros(3), jnp.zeros(3), jnp.ones(4))) == 0
