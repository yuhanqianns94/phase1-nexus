"""Pure, differentiable dose-risk functions; no fitting or data borrowing."""
from __future__ import annotations

from itertools import combinations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import expit


def interaction_sets(drug_count):
    h = range(min(drug_count, 3))
    return [s for n in (2, 3) for s in combinations(h, n)]


def component_risks(a, log_b, relative_doses, numeric_mask, present):
    relative_doses = jnp.asarray(relative_doses)
    logits = a + jnp.where(numeric_mask, jnp.exp(log_b) * jnp.log(jnp.maximum(relative_doses, 1e-300)), 0.)
    return jnp.where(present, expit(logits), 0.)


def baseline_risk(q):
    return -jnp.expm1(jnp.sum(jnp.log1p(-jnp.clip(q, 0, 1 - 1e-15)), axis=-1))


def interaction_basis(relative_doses):
    r = jnp.asarray(relative_doses)
    terms = []
    for group in interaction_sets(r.shape[-1]):
        # Stable saturating form, including exact absence.
        # Scalar column slices avoid a constant-folded advanced-index gather
        # that produces incorrect multi-level bases in the supported JAX runtime.
        logs = sum(jnp.log(jnp.maximum(r[..., j], 1e-300)) for j in group)
        present = jnp.ones(r.shape[:-1], dtype=bool)
        for j in group:
            present = present & (r[..., j] > 0)
        terms.append(jnp.where(present, 2 * expit(logs), 0.))
    return jnp.stack(terms, axis=-1) if terms else jnp.zeros((*r.shape[:-1], 0))


def logistic_risk(q, relative_doses, eta):
    p0 = baseline_risk(q)
    safe = jnp.clip(p0, 1e-300, 1 - 1e-15)
    shift = jnp.sum(interaction_basis(relative_doses) * eta, axis=-1)
    p = expit(jnp.log(safe) - jnp.log1p(-safe) + shift)
    return jnp.where(jnp.any(jnp.asarray(relative_doses) > 0, axis=-1), p, 0.)


def product_interaction_basis(relative_doses):
    """Ordinary products, restricted to pair/triple terms among the first three drugs."""
    r = jnp.asarray(relative_doses)
    terms = []
    for group in interaction_sets(r.shape[-1]):
        term = jnp.ones(r.shape[:-1])
        for j in group:
            term = term * r[..., j]
        terms.append(term)
    return jnp.stack(terms, axis=-1) if terms else jnp.zeros((*r.shape[:-1], 0))


def product_logistic_risk(alpha, log_b, relative_doses, eta):
    """One shared intercept, positive main effects, and signed product interactions.

    Generic dose-unknown exposure is r=1 when present and r=0 when absent.
    Unlike the component model, zero exposure retains sigmoid(alpha).
    Signed interactions do not guarantee coordinatewise dose monotonicity.
    """
    r = jnp.asarray(relative_doses)
    predictor = (alpha + jnp.sum(jnp.exp(log_b) * r, axis=-1)
                 + jnp.sum(product_interaction_basis(r) * eta, axis=-1))
    return expit(predictor)
