"""One-parameter power-model CRM trial simulator.

The implementation follows the decision model used by ``GPT-wrapper web``:

``p_j(a) = skeleton_j ** exp(a)``,  ``a ~ Normal(prior_mean, prior_sd**2)``,

The one-dimensional posterior is evaluated with mode-scaled adaptive
quadrature, which preserves the reference decisions while avoiding numerical
underflow in sparse or separated data.
"""

from __future__ import annotations

from functools import lru_cache
import math
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar


_PRIOR_SD_INTEGRATION_LIMIT = 10.0


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number.")
    return numeric


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer.")
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return numeric


def _probability_vector(
    values: Sequence[float] | np.ndarray,
    *,
    label: str,
    minimum_length: int,
    allow_boundary: bool,
    strictly_increasing: bool,
) -> np.ndarray:
    try:
        raw_values = list(values)
    except TypeError as error:
        raise ValueError(
            f"{label} must be a one-dimensional numeric sequence."
        ) from error
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in raw_values
    ):
        raise ValueError(f"{label} must contain only finite numeric probabilities.")
    array = np.asarray(raw_values, dtype=float)
    if array.ndim != 1 or len(array) < minimum_length:
        raise ValueError(
            f"{label} must be one-dimensional with at least {minimum_length} values."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite numeric probabilities.")
    if allow_boundary:
        valid_range = np.all((array >= 0.0) & (array <= 1.0))
        range_description = "[0, 1]"
    else:
        valid_range = np.all((array > 0.0) & (array < 1.0))
        range_description = "(0, 1)"
    if not valid_range:
        raise ValueError(f"{label} probabilities must lie in {range_description}.")
    differences = np.diff(array)
    if strictly_increasing and np.any(differences <= 0.0):
        raise ValueError(f"{label} must be strictly increasing by dose level.")
    if not strictly_increasing and np.any(differences < 0.0):
        raise ValueError(f"{label} must be non-decreasing by dose level.")
    return array


def _default_skeleton(n_doses: int, target_dlt: float) -> np.ndarray:
    """Return this project's target-centred automatic CRM skeleton.

    This is deliberately not the GPT-wrapper reference skeleton.  The target
    occupies the centre dose (for an odd number of levels), with neighbouring
    probabilities spaced evenly on the logit scale.
    """

    centre = (n_doses - 1) / 2.0
    offsets = (np.arange(n_doses, dtype=float) - centre) * 0.55
    target_logit = math.log(target_dlt / (1.0 - target_dlt))
    logits = target_logit + offsets
    return 1.0 / (1.0 + np.exp(-logits))


def _mapped_quadrature(
    lower: float,
    upper: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    half_width = (upper - lower) / 2.0
    midpoint = (upper + lower) / 2.0
    return midpoint + half_width * nodes, half_width * weights


def _log_posterior_density(
    parameter_values: np.ndarray,
    *,
    skeleton: np.ndarray,
    dlt_counts: np.ndarray,
    treated_counts: np.ndarray,
    prior_mean: float,
    prior_sd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log posterior density (up to a constant) and model probabilities."""

    exponent = np.exp(np.clip(parameter_values, -745.0, 709.0))
    log_probabilities = exponent[:, None] * np.log(skeleton)[None, :]
    probabilities = np.exp(log_probabilities)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_one_minus = np.log(-np.expm1(log_probabilities))

    event_terms = np.zeros_like(log_probabilities)
    np.multiply(
        log_probabilities,
        dlt_counts[None, :],
        out=event_terms,
        where=dlt_counts[None, :] > 0,
    )
    non_event_counts = treated_counts - dlt_counts
    non_event_terms = np.zeros_like(log_probabilities)
    np.multiply(
        log_one_minus,
        non_event_counts[None, :],
        out=non_event_terms,
        where=non_event_counts[None, :] > 0,
    )
    log_likelihood = np.sum(event_terms + non_event_terms, axis=1)
    standardized = (parameter_values - prior_mean) / prior_sd
    log_prior = -0.5 * standardized**2
    return log_prior + log_likelihood, probabilities


def _log_weighted_sum(log_values: np.ndarray, weights: np.ndarray) -> float:
    log_terms = log_values + np.log(weights)
    maximum = float(np.max(log_terms))
    if not math.isfinite(maximum):
        raise RuntimeError("CRM posterior normalization failed.")
    return maximum + math.log(float(np.sum(np.exp(log_terms - maximum))))


@lru_cache(maxsize=16_384)
def _posterior_summary_cached(
    skeleton_values: tuple[float, ...],
    dlt_values: tuple[int, ...],
    treated_values: tuple[int, ...],
    prior_mean: float,
    prior_sd: float,
    target_dlt: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    skeleton = np.asarray(skeleton_values, dtype=float)
    dlt_counts = np.asarray(dlt_values, dtype=float)
    treated_counts = np.asarray(treated_values, dtype=float)
    lower = prior_mean - _PRIOR_SD_INTEGRATION_LIMIT * prior_sd
    upper = prior_mean + _PRIOR_SD_INTEGRATION_LIMIT * prior_sd
    def scalar_log_posterior(parameter: float) -> float:
        log_density, _probabilities = _log_posterior_density(
            np.asarray([parameter], dtype=float),
            skeleton=skeleton,
            dlt_counts=dlt_counts,
            treated_counts=treated_counts,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
        )
        return float(log_density[0])

    # A fixed grid across +/-10 prior SD can miss a narrow likelihood peak and
    # can even reverse the selected CRM dose.  The bounded mode is used only as
    # a log-scaling constant; adaptive integration then evaluates the same
    # posterior model as GPT-wrapper web without copying its underflow risk.
    optimized = minimize_scalar(
        lambda parameter: -scalar_log_posterior(float(parameter)),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not optimized.success or not math.isfinite(float(optimized.fun)):
        raise RuntimeError("CRM posterior mode search failed.")
    log_scale = -float(optimized.fun)
    cutoffs = np.asarray(
        [
            math.log(math.log(target_dlt) / math.log(float(probability)))
            for probability in skeleton
        ],
        dtype=float,
    )

    def vector_integrand(parameter: float) -> np.ndarray:
        log_density, probabilities = _log_posterior_density(
            np.asarray([parameter], dtype=float),
            skeleton=skeleton,
            dlt_counts=dlt_counts,
            treated_counts=treated_counts,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
        )
        weight = math.exp(float(log_density[0]) - log_scale)
        tail_indicators = (parameter <= cutoffs).astype(float)
        return np.concatenate(
            ([weight], weight * probabilities[0], weight * tail_indicators)
        )

    interior_points = sorted(
        {float(value) for value in cutoffs if lower < float(value) < upper}
    )
    integrals, _error = quad_vec(
        vector_integrand,
        lower,
        upper,
        epsabs=1e-11,
        epsrel=1e-10,
        points=interior_points or None,
        limit=300,
    )
    normalizer = float(integrals[0])
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise RuntimeError("CRM posterior normalization failed.")
    dose_count = len(skeleton)
    posterior_means = np.asarray(integrals[1 : 1 + dose_count]) / normalizer
    overdose_probabilities = np.clip(
        np.asarray(integrals[1 + dose_count :]) / normalizer, 0.0, 1.0
    )

    return (
        tuple(float(value) for value in posterior_means),
        tuple(float(value) for value in overdose_probabilities),
    )


def _posterior_summary(
    skeleton: np.ndarray,
    dlt_counts: np.ndarray,
    treated_counts: np.ndarray,
    *,
    prior_mean: float,
    prior_sd: float,
    target_dlt: float,
) -> tuple[np.ndarray, np.ndarray]:
    means, overdose = _posterior_summary_cached(
        tuple(float(value) for value in skeleton),
        tuple(int(value) for value in dlt_counts),
        tuple(int(value) for value in treated_counts),
        float(prior_mean),
        float(prior_sd),
        float(target_dlt),
    )
    return np.asarray(means, dtype=float), np.asarray(overdose, dtype=float)


def _closest_admissible_dose(
    estimated_probabilities: np.ndarray,
    overdose_probabilities: np.ndarray,
    *,
    target_dlt: float,
    ewoc_threshold: float,
) -> int | None:
    admissible = np.flatnonzero(overdose_probabilities <= ewoc_threshold)
    if not len(admissible):
        return None
    distances = np.abs(estimated_probabilities[admissible] - target_dlt)
    return int(admissible[int(np.argmin(distances))])


def simulate_crm_trial(
    true_probabilities: Sequence[float] | np.ndarray,
    *,
    start_index: int,
    cohort_size: int,
    max_sample_size: int,
    target_dlt: float,
    rng: np.random.Generator,
    skeleton: Sequence[float] | np.ndarray | None = None,
    prior_mean: float = 0.0,
    prior_sd: float = 1.34,
    max_step: int = 1,
    allow_dose_skipping: bool = False,
    ewoc_threshold: float = 1.0,
    safety_stop_probability: float = 0.95,
) -> dict[str, Any]:
    """Simulate one cohort-by-cohort CRM trial.

    Dose indices are zero based.  The reference-equivalent default sets EWOC
    to 1, so it does not filter doses.  If a user explicitly lowers it, EWOC
    uses the event
    ``Pr(p_j > target_dlt | data)``; ``ewoc_threshold`` is the largest allowed
    posterior probability of that event.  The distinct
    ``safety_stop_probability`` controls early termination when even the
    lowest dose is probably above target.

    By default, both upward and downward moves are limited by ``max_step``,
    matching GPT-wrapper web.  ``allow_dose_skipping`` is retained as an
    explicit extension that removes the upward limit only.
    If no dose meets an explicitly enabled EWOC threshold but the lowest-dose
    safety rule has not fired, the next cohort remains at the lowest dose.
    """

    true_dlt = _probability_vector(
        true_probabilities,
        label="true_probabilities",
        minimum_length=2,
        allow_boundary=True,
        strictly_increasing=False,
    )
    n_doses = len(true_dlt)
    if isinstance(start_index, (bool, np.bool_)) or not isinstance(
        start_index, Integral
    ):
        raise ValueError("start_index must be a zero-based integer dose index.")
    start_index = int(start_index)
    if not 0 <= start_index < n_doses:
        raise ValueError("start_index is outside the available dose levels.")
    cohort_size = _positive_integer(cohort_size, label="cohort_size")
    max_sample_size = _positive_integer(max_sample_size, label="max_sample_size")
    if max_sample_size < cohort_size or max_sample_size % cohort_size != 0:
        raise ValueError(
            "max_sample_size must be at least one cohort and exactly divisible "
            "by cohort_size."
        )
    target_dlt = _finite_float(target_dlt, label="target_dlt")
    if not 0.0 < target_dlt < 1.0:
        raise ValueError("target_dlt must lie in (0, 1).")
    prior_mean = _finite_float(prior_mean, label="prior_mean")
    prior_sd = _finite_float(prior_sd, label="prior_sd")
    if prior_sd <= 0.0:
        raise ValueError("prior_sd must be greater than zero.")
    integration_lower = prior_mean - _PRIOR_SD_INTEGRATION_LIMIT * prior_sd
    integration_upper = prior_mean + _PRIOR_SD_INTEGRATION_LIMIT * prior_sd
    if (
        not math.isfinite(integration_lower)
        or not math.isfinite(integration_upper)
        or integration_lower >= integration_upper
    ):
        raise ValueError(
            "prior_mean and prior_sd do not define numerically stable CRM "
            "integration bounds."
        )
    ewoc_threshold = _finite_float(ewoc_threshold, label="ewoc_threshold")
    if not 0.0 < ewoc_threshold <= 1.0:
        raise ValueError("ewoc_threshold must lie in (0, 1].")
    safety_stop_probability = _finite_float(
        safety_stop_probability, label="safety_stop_probability"
    )
    if not 0.0 < safety_stop_probability <= 1.0:
        raise ValueError("safety_stop_probability must lie in (0, 1].")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an instance of numpy.random.Generator.")
    if not isinstance(allow_dose_skipping, (bool, np.bool_)):
        raise ValueError("allow_dose_skipping must be boolean.")
    allow_dose_skipping = bool(allow_dose_skipping)
    if isinstance(max_step, (bool, np.bool_)) or not isinstance(max_step, Integral):
        raise ValueError("max_step must be a positive integer.")
    max_step = int(max_step)
    if max_step <= 0:
        raise ValueError("max_step must be a positive integer.")

    if skeleton is None:
        model_skeleton = _default_skeleton(n_doses, target_dlt)
        skeleton_source = "project_target_centered_logit_spacing"
    else:
        model_skeleton = _probability_vector(
            skeleton,
            label="skeleton",
            minimum_length=n_doses,
            allow_boundary=False,
            strictly_increasing=True,
        )
        if len(model_skeleton) != n_doses:
            raise ValueError(
                "skeleton length must equal the number of true probabilities."
            )
        skeleton_source = "user_supplied"

    treated_counts = np.zeros(n_doses, dtype=int)
    dlt_counts = np.zeros(n_doses, dtype=int)
    current_index = start_index
    cohort_count = 0
    early_stopped = False
    stop_reason: str | None = None
    action_counts = {
        "escalate": 0,
        "deescalate": 0,
        "stay": 0,
        "safety_stop": 0,
    }
    history: list[dict[str, Any]] = []
    estimated_probabilities: np.ndarray | None = None
    overdose_probabilities: np.ndarray | None = None

    while int(np.sum(treated_counts)) < max_sample_size:
        treated_index = current_index
        cohort_dlt = int(rng.binomial(cohort_size, true_dlt[treated_index]))
        treated_counts[treated_index] += cohort_size
        dlt_counts[treated_index] += cohort_dlt
        cohort_count += 1

        estimated_probabilities, overdose_probabilities = _posterior_summary(
            model_skeleton,
            dlt_counts,
            treated_counts,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
            target_dlt=target_dlt,
        )

        history_row: dict[str, Any] = {
            "cohort_number": cohort_count,
            "treated_index": treated_index,
            "cohort_size": cohort_size,
            "cohort_dlt": cohort_dlt,
            "cumulative_treated_by_dose": treated_counts.tolist(),
            "cumulative_dlt_by_dose": dlt_counts.tolist(),
            "estimated_probabilities": estimated_probabilities.tolist(),
            "posterior_overdose_probabilities": overdose_probabilities.tolist(),
            "recommended_next_index": None,
            "action": None,
        }

        if overdose_probabilities[0] > safety_stop_probability:
            early_stopped = True
            stop_reason = "lowest_dose_overtoxicity"
            action_counts["safety_stop"] += 1
            history_row["action"] = "safety_stop"
            history.append(history_row)
            break

        if int(np.sum(treated_counts)) >= max_sample_size:
            history_row["action"] = "complete"
            history.append(history_row)
            break

        best_index = _closest_admissible_dose(
            estimated_probabilities,
            overdose_probabilities,
            target_dlt=target_dlt,
            ewoc_threshold=ewoc_threshold,
        )
        if best_index is None:
            next_index = 0
        elif best_index > current_index:
            upward_step = best_index - current_index
            if not allow_dose_skipping:
                upward_step = min(max_step, upward_step)
            next_index = current_index + upward_step
        elif best_index < current_index:
            next_index = current_index - min(max_step, current_index - best_index)
        else:
            next_index = best_index

        if next_index > current_index:
            action = "escalate"
        elif next_index < current_index:
            action = "deescalate"
        else:
            action = "stay"
        action_counts[action] += 1
        history_row["recommended_next_index"] = int(next_index)
        history_row["action"] = action
        history.append(history_row)
        current_index = int(next_index)

    if estimated_probabilities is None or overdose_probabilities is None:
        raise RuntimeError("CRM trial ended without processing a cohort.")

    if early_stopped:
        selected_index: int | None = None
    else:
        selected_index = _closest_admissible_dose(
            estimated_probabilities,
            overdose_probabilities,
            target_dlt=target_dlt,
            ewoc_threshold=ewoc_threshold,
        )
        if selected_index is None:
            stop_reason = "max_sample_size_reached_no_ewoc_admissible_dose"
        else:
            stop_reason = "max_sample_size_reached"

    return {
        "selected_index": selected_index,
        "treated_by_dose": treated_counts.tolist(),
        "dlt_by_dose": dlt_counts.tolist(),
        "total_n": int(np.sum(treated_counts)),
        "total_dlt": int(np.sum(dlt_counts)),
        "cohort_count": cohort_count,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "action_counts": action_counts,
        "estimated_probabilities": estimated_probabilities.tolist(),
        "posterior_overdose_probabilities": overdose_probabilities.tolist(),
        "history": history,
        "metadata": {
            "design": "CRM",
            "model": "one_parameter_power_crm",
            "dose_indexing": "zero_based",
            "skeleton": model_skeleton.tolist(),
            "skeleton_source": skeleton_source,
            "target_dlt": target_dlt,
            "prior": {
                "parameter": "a",
                "distribution": "normal",
                "mean": prior_mean,
                "sd": prior_sd,
            },
            "ewoc": {
                "overdose_event": "dose_probability_above_target_dlt",
                "maximum_allowed_posterior_overdose_probability": ewoc_threshold,
            },
            "lowest_dose_safety_stop": {
                "event": "lowest_dose_probability_above_target_dlt",
                "posterior_probability_cutoff": safety_stop_probability,
                "comparison": "strictly_greater_than",
            },
            "dose_skipping": {
                "allowed": allow_dose_skipping,
                "max_step": max_step,
                "policy_when_disabled": "both_directions_limited_by_max_step",
            },
            "posterior_computation": {
                "method": "mode_scaled_adaptive_vector_quadrature",
                "integration_limits_in_prior_sd": _PRIOR_SD_INTEGRATION_LIMIT,
            },
            "action_count_policy": (
                "between_cohort_allocation_decisions_plus_safety_stop"
            ),
        },
    }
