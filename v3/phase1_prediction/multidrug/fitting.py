"""Independent-source NUTS; no external data or target inputs accepted."""
from __future__ import annotations

from itertools import product
import time

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary

from .contracts import ModelScope, PriorSpec, SamplerSpec
from .math import component_risks, logistic_risk, product_logistic_risk, interaction_sets
from . import validate_backend


def _binomial_logprob(y, n, p):
    return dist.Binomial(total_count=n, probs=jnp.clip(p, 1e-15, 1 - 1e-15)).log_prob(y)


def _split_logprob(y, n, exposed_n, p_on, p_off):
    """Sum over unobserved subgroup DLT counts, without assigning invented y."""
    lo, hi = max(0, y - (n - exposed_n)), min(y, exposed_n)
    k = jnp.arange(lo, hi + 1)
    return logsumexp(_binomial_logprob(k, exposed_n, p_on)
                     + _binomial_logprob(y - k, n - exposed_n, p_off))


def _allocation_logprob(y, groups):
    """Exact convolution for aggregate y across known dose-allocation groups."""
    log_probs = jnp.zeros(1)
    patients = 0
    for n, p in groups:
        values = []
        for total in range(min(y, patients + n) + 1):
            counts = jnp.arange(max(0, total - patients), min(n, total) + 1)
            values.append(logsumexp(log_probs[total - counts] + _binomial_logprob(counts, n, p)))
        log_probs = jnp.stack(values)
        patients += n
    return log_probs[y] if y <= patients else jnp.array(-jnp.inf)


def sample_curve_parameters(drugs, backend, prior):
    """One joint proper prior, reused by article-local and target models."""
    validate_backend(backend)
    k = len(drugs)
    if backend == 'logistic_product':
        a = numpyro.sample('alpha', dist.Normal(prior.intercept_mean, prior.intercept_sd))
        # A present background drug has an estimated main effect at r=1.
        # Its effect is not fixed merely because its physical dose is unknown.
        log_b = numpyro.sample('log_b_effects', dist.Normal(
            prior.log_slope_mean, prior.log_slope_sd).expand((k,)).to_event(1))
    else:
        numeric_indices = [i for i, d in enumerate(drugs) if d.mode == "numeric"]
        a = numpyro.sample("a", dist.Normal(prior.intercept_mean, prior.intercept_sd).expand((k,)).to_event(1))
        log_b = jnp.zeros(k)
        if numeric_indices:
            sampled = numpyro.sample("log_b_numeric", dist.Normal(prior.log_slope_mean, prior.log_slope_sd).expand((len(numeric_indices),)).to_event(1))
            log_b = log_b.at[jnp.array(numeric_indices)].set(sampled)
    numpyro.deterministic("log_b", log_b)
    groups = interaction_sets(k)
    eta = (numpyro.sample("eta", dist.Normal(0., jnp.array([
        prior.pair_sd if len(g) == 2 else prior.triple_sd for g in groups])).to_event(1))
        if groups else jnp.zeros(0))
    if not groups:
        numpyro.deterministic("eta", eta)
    return a, log_b, eta


def curve_risk_function(drugs, backend, parameters):
    validate_backend(backend)
    a, log_b, eta = parameters
    numeric_mask = jnp.array([d.mode == "numeric" for d in drugs])

    def risk(r, present):
        r = jnp.where(jnp.array(present), jnp.array(r), 0.)
        if backend == 'logistic_product':
            return product_logistic_risk(a, log_b, r, eta)
        q = component_risks(a, log_b, r, numeric_mask, jnp.array(present))
        return logistic_risk(q, r, eta)

    return risk


def observe_scope(scope, risk, prior, likelihood_power=1.):
    """Temper the observed outcome likelihood, never the latent-dose priors.

    Finite cohort-state mixtures are marginalized before applying the power;
    continuous dose/fraction priors remain untempered (partial discounting).
    """
    for oi, observation in enumerate(scope.observations):
        allocated = [(j,e) for j,e in enumerate(observation.exposures) if e.dose_subgroups]
        if len(allocated) > 1:
            raise ValueError('Multiple dose allocations require joint patient subgroup mapping')
        options = []
        mixed_known, mixed_unknown = [], []
        for j, (slot, e) in enumerate(zip(scope.drugs, observation.exposures)):
            if e.presence == "absent":
                options.append([(1., False, 0.)]); continue
            if slot.mode == "background":
                doses = [1.]
            elif e.dose is not None:
                doses = [e.dose / slot.reference_dose]
            elif e.dose_subgroups:
                doses = [1.]  # Replaced by the exact allocation likelihood below.
            elif e.candidates:
                doses = [value / slot.reference_dose for value in e.candidates]
            else:
                lo = np.log(e.lower / slot.reference_dose) if e.lower else -np.inf
                hi = np.log(e.upper / slot.reference_dose) if e.upper else np.inf
                if lo == hi:
                    doses = [jnp.exp(lo)]
                else:
                    bounds={}
                    if np.isfinite(lo):bounds['low']=lo
                    if np.isfinite(hi):bounds['high']=hi
                    location = e.log_relative_dose_prior_mean
                    distribution = (dist.TruncatedNormal(location, prior.missing_log_dose_sd, **bounds) if bounds
                                    else dist.Normal(location, prior.missing_log_dose_sd))
                    latent = numpyro.sample(f"missing_dose_{oi}_{j}", distribution)
                    doses = [jnp.exp(latent)]
            if e.presence == "unknown":
                p = prior.unknown_presence_probability
                options.append([(1 - p, False, 0.)] + [(p / len(doses), True, r) for r in doses])
            else:
                options.append([(1 / len(doses), True, r) for r in doses])
                if e.presence == "mixed" and not e.dose_subgroups:
                    if e.exposed_n is not None:
                        mixed_known.append((j, e.exposed_n))
                    else:
                        mixed_unknown.append(j)
        if len(mixed_known) > 1 or (mixed_known and mixed_unknown):
            raise ValueError("Multiple mixed exposures need joint subgroup allocation, not invented independent groups")
        fractions = {j: numpyro.sample(f"mixed_fraction_{oi}_{j}", dist.Beta(1., 1.)) for j in mixed_unknown}
        log_likelihoods = []
        for states in product(*options):
            weight = float(np.prod([x[0] for x in states]))
            present = [x[1] for x in states]
            r = [x[2] for x in states]
            p = risk(r, present)
            if allocated:
                if mixed_known or mixed_unknown:
                    raise ValueError('Dose allocations plus mixed concomitant use require joint subgroup mapping')
                j,e = allocated[0]
                subgroups=[]
                for dose,n in e.dose_subgroups:
                    rr=list(r);rr[j]=dose/scope.drugs[j].reference_dose
                    subgroups.append((n,risk(rr,present)))
                ll=_allocation_logprob(observation.y,subgroups)
            elif mixed_known:
                j, n_on = mixed_known[0]
                absent = list(present); absent[j] = False
                ll = _split_logprob(observation.y, observation.n, n_on, p, risk(r, absent))
            elif mixed_unknown:
                # Explicit patient-level latent exposure mixture, unlike cohort-unknown presence.
                p_mixture = 0.
                for bits in product((False, True), repeat=len(mixed_unknown)):
                    state_present, state_weight = list(present), 1.
                    for j, on in zip(mixed_unknown, bits):
                        state_present[j] = on
                        state_weight *= fractions[j] if on else 1 - fractions[j]
                    p_mixture += state_weight * risk(r, state_present)
                ll = _binomial_logprob(observation.y, observation.n, p_mixture)
            else:
                ll = _binomial_logprob(observation.y, observation.n, p)
            log_likelihoods.append(jnp.log(weight) + ll)
        # Cohort-level states are integrated OUTSIDE the binomial likelihood.
        numpyro.factor(f"cohort_{oi}", likelihood_power * logsumexp(jnp.stack(log_likelihoods)))


def source_model(scope: ModelScope, backend: str, prior: PriorSpec):
    parameters = sample_curve_parameters(scope.drugs, backend, prior)
    risk = curve_risk_function(scope.drugs, backend, parameters)
    observe_scope(scope, risk, prior)


def fit_source(scope, backend, prior=None, sampler=None):
    prior, sampler = prior or PriorSpec(), sampler or SamplerSpec()
    started = time.monotonic()
    if not scope.observations:
        samples = Predictive(source_model, num_samples=sampler.chains * sampler.samples)(
            jax.random.PRNGKey(sampler.seed), scope, backend, prior)
        arrays = {k: np.asarray(v).reshape((sampler.chains, sampler.samples, *v.shape[1:]))
                  for k, v in samples.items() if not k.startswith("cohort_")}
        return arrays, {"status": "prior_only", "elapsed_seconds": time.monotonic() - started,
                        "reason": "No legal binomial observations; not independent historical outcome evidence"}
    return run_nuts(source_model, (scope, backend, prior), sampler)


def run_nuts(model, model_args, sampler):
    """Shared NUTS diagnostics; no fallback estimator on a failed fit."""
    started = time.monotonic()
    attempts = []
    for attempt in range(sampler.retry_count + 1):
        factor = 2**attempt
        kernel = NUTS(model, target_accept_prob=min(.99, sampler.target_accept + .04 * attempt),
                      max_tree_depth=min(15, sampler.max_tree_depth + attempt))
        mcmc = MCMC(kernel, num_warmup=sampler.warmup * factor, num_samples=sampler.samples * factor,
                    num_chains=sampler.chains, chain_method="sequential", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(sampler.seed + attempt), *model_args,
                 extra_fields=("diverging", "num_steps"))
        arrays = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        variables = {k: v for k, v in arrays.items() if k != "log_b" and v.size}
        stats = summary(variables, group_by_chain=True)
        rhats = np.concatenate([np.asarray(s["r_hat"]).reshape(-1) for s in stats.values()])
        effective_sizes = np.concatenate([np.asarray(s["n_eff"]).reshape(-1) for s in stats.values()])
        diagnostics_finite = np.isfinite(rhats).all() and np.isfinite(effective_sizes).all()
        rhat = float(np.max(rhats))
        ess = float(np.min(effective_sizes))
        extra = mcmc.get_extra_fields()
        divergent = int(np.asarray(extra["diverging"]).sum())
        finite = all(np.isfinite(a).all() for a in arrays.values())
        ok = finite and diagnostics_finite and rhat <= sampler.max_rhat and ess >= sampler.min_ess and divergent == 0
        attempts.append({"attempt": attempt + 1, "max_rhat": rhat if np.isfinite(rhat) else None, "min_ess": ess if np.isfinite(ess) else None,
                         "divergences": divergent, "draws_per_chain": sampler.samples * factor,
                         "passed": bool(ok)})
        if ok:
            break
    return arrays, {"status": "fitted" if ok else "numerical_failure", "attempts": attempts,
                    "elapsed_seconds": time.monotonic() - started,
                    "reason": None if ok else "Posterior sampling diagnostics did not pass; no fallback used"}
