"""Monotone Bayesian Binomial-logistic dose--DLT curve fitting.

The active estimator uses a two-parameter logistic curve for every historical
arm.  It obtains a fast, reproducible posterior approximation by finding the
MAP under explicit Normal priors and applying a Laplace approximation in the
unconstrained ``(alpha, eta)`` space:

``logit(p) = alpha + softplus(eta) * log(dose / geometric_mean_dose)``.

``softplus(eta)`` makes the dose slope non-negative without treating an
assumed zero-DLT anchor as a real Binomial observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


MODEL_NAME = "monotone_bayesian_binomial_logistic_log_dose_laplace_v2"
POSTERIOR_APPROXIMATION = "laplace_normal_in_alpha_eta_space"
DEFAULT_PRIOR_CATALOG = Path(__file__).with_name("bayesian_prior_candidates.json")
# 1,024 leaves roughly 25 draws in each 2.5% posterior tail.  This is a more
# stable default for credible intervals and posterior-predictive OC than the
# original 256, while remaining practical for the local multi-prior cache.
DEFAULT_POSTERIOR_DRAW_COUNT = 1_024
DEFAULT_PRIOR_PREDICTIVE_DRAW_COUNT = 4_096


@dataclass(frozen=True)
class BayesianPrior:
    """Independent Normal priors for the logistic intercept and slope latent."""

    name: str
    description: str
    alpha_mean: float
    alpha_sd: float
    eta_mean: float
    eta_sd: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "alpha": {
                "distribution": "normal",
                "mean": self.alpha_mean,
                "sd": self.alpha_sd,
            },
            "eta": {
                "distribution": "normal",
                "mean": self.eta_mean,
                "sd": self.eta_sd,
            },
        }


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _canonical_json_hash(payload: Any) -> str:
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def load_prior_catalog(
    path: Path = DEFAULT_PRIOR_CATALOG,
) -> tuple[dict[str, BayesianPrior], dict[str, Any], str]:
    """Load, validate, and fingerprint the editable named-prior catalog."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Bayesian prior catalog does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Bayesian prior catalog is invalid JSON: {path}") from error

    if not isinstance(payload, dict):
        raise ValueError("Bayesian prior catalog must be a JSON object.")
    if payload.get("format") != "mtd_similarity_bayesian_prior_catalog_v1":
        raise ValueError(
            "Bayesian prior catalog format must be "
            "'mtd_similarity_bayesian_prior_catalog_v1'."
        )
    if payload.get("model") != MODEL_NAME:
        raise ValueError(
            "Bayesian prior catalog model must be "
            f"{MODEL_NAME!r}."
        )
    raw_priors = payload.get("priors")
    if not isinstance(raw_priors, dict) or not raw_priors:
        raise ValueError("Bayesian prior catalog must contain a non-empty priors object.")

    default_prior = payload.get("default_prior")
    if not isinstance(default_prior, str) or default_prior not in raw_priors:
        raise ValueError(
            "Bayesian prior catalog default_prior must name one configured prior."
        )
    raw_policy = payload.get("prior_policy")
    if not isinstance(raw_policy, dict):
        raise ValueError("Bayesian prior catalog requires a prior_policy object.")
    minimum_alpha_sd = _finite_number(
        raw_policy.get("minimum_alpha_sd"),
        label="prior_policy.minimum_alpha_sd",
    )
    minimum_eta_sd = _finite_number(
        raw_policy.get("minimum_eta_sd"),
        label="prior_policy.minimum_eta_sd",
    )
    if minimum_alpha_sd <= 0 or minimum_eta_sd <= 0:
        raise ValueError("Bayesian prior policy minimum standard deviations must be > 0.")

    catalog: dict[str, BayesianPrior] = {}
    for name, raw_prior in raw_priors.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every Bayesian prior name must be non-empty.")
        if not isinstance(raw_prior, dict):
            raise ValueError(f"Bayesian prior {name!r} must be an object.")
        alpha = raw_prior.get("alpha")
        eta = raw_prior.get("eta")
        if not isinstance(alpha, dict) or not isinstance(eta, dict):
            raise ValueError(f"Bayesian prior {name!r} requires alpha and eta objects.")
        if alpha.get("distribution") != "normal" or eta.get("distribution") != "normal":
            raise ValueError(
                f"Bayesian prior {name!r} supports only Normal alpha and eta priors."
            )
        alpha_mean = _finite_number(alpha.get("mean"), label=f"{name}.alpha.mean")
        alpha_sd = _finite_number(alpha.get("sd"), label=f"{name}.alpha.sd")
        eta_mean = _finite_number(eta.get("mean"), label=f"{name}.eta.mean")
        eta_sd = _finite_number(eta.get("sd"), label=f"{name}.eta.sd")
        if alpha_sd <= 0 or eta_sd <= 0:
            raise ValueError(f"Bayesian prior {name!r} standard deviations must be > 0.")
        if alpha_sd < minimum_alpha_sd or eta_sd < minimum_eta_sd:
            raise ValueError(
                f"Bayesian prior {name!r} is narrower than the catalog's "
                "weak-information policy."
            )
        description = raw_prior.get("description")
        if not isinstance(description, str):
            description = ""
        catalog[name] = BayesianPrior(
            name=name,
            description=description,
            alpha_mean=alpha_mean,
            alpha_sd=alpha_sd,
            eta_mean=eta_mean,
            eta_sd=eta_sd,
        )
    return catalog, payload, _canonical_json_hash(payload)


def resolve_prior(
    prior_name: str, catalog: dict[str, BayesianPrior]
) -> BayesianPrior:
    if prior_name not in catalog:
        available = ", ".join(sorted(catalog))
        raise ValueError(
            f"Unknown Bayesian prior {prior_name!r}. Available priors: {available}."
        )
    return catalog[prior_name]


def format_prior_distribution(prior: BayesianPrior) -> str:
    """Return the Normal alpha/eta prior as a compact human-readable line pair."""

    return (
        f"  alpha ~ Normal(mean={prior.alpha_mean:g}, sd={prior.alpha_sd:g})\n"
        f"  eta   ~ Normal(mean={prior.eta_mean:g}, sd={prior.eta_sd:g})\n"
        "  slope = softplus(eta) >= 0"
    )


def format_prior_panel_text(
    selected_prior_name: str | None,
    catalog: dict[str, BayesianPrior] | None = None,
    *,
    catalog_path: Path = DEFAULT_PRIOR_CATALOG,
) -> str:
    """Build a right-margin panel describing available and selected priors."""

    if catalog is None:
        catalog, _payload, _fingerprint = load_prior_catalog(catalog_path)
    lines = [
        "Bayesian priors",
        "alpha: log-odds at geometric-mean dose",
        "eta: softplus -> non-negative slope",
        "",
    ]
    if selected_prior_name:
        if selected_prior_name in catalog:
            selected = catalog[selected_prior_name]
            lines.extend(
                [
                    f"Selected: {selected_prior_name}",
                    format_prior_distribution(selected),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"Selected: {selected_prior_name}",
                    "  (not found in current prior catalog)",
                    "",
                ]
            )
    lines.append("Available priors:")
    for name in sorted(catalog):
        prior = catalog[name]
        marker = " *" if name == selected_prior_name else ""
        lines.append(f"- {name}{marker}")
        lines.append(
            f"  alpha ~ N({prior.alpha_mean:g}, {prior.alpha_sd:g}^2)"
        )
        lines.append(
            f"  eta   ~ N({prior.eta_mean:g}, {prior.eta_sd:g}^2)"
        )
    return "\n".join(lines)


def annotate_figure_with_prior_panel(
    figure: Any,
    selected_prior_name: str | None,
    *,
    catalog: dict[str, BayesianPrior] | None = None,
    catalog_path: Path = DEFAULT_PRIOR_CATALOG,
) -> None:
    """Reserve the figure's right margin for available/selected prior details."""

    if not selected_prior_name and catalog is None:
        try:
            catalog, _payload, _fingerprint = load_prior_catalog(catalog_path)
        except ValueError:
            return
    text = format_prior_panel_text(
        selected_prior_name,
        catalog,
        catalog_path=catalog_path,
    )
    figure.subplots_adjust(right=0.62)
    figure.text(
        0.64,
        0.5,
        text,
        transform=figure.transFigure,
        va="center",
        ha="left",
        fontsize=7,
        family="monospace",
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.65",
            "alpha": 0.96,
        },
    )


def softplus(value: float | np.ndarray) -> float | np.ndarray:
    """Stable softplus transform used to make the dose slope non-negative."""

    return np.logaddexp(0.0, value)


def _expit(value: np.ndarray) -> np.ndarray:
    """Stable logistic inverse link without an extra dependency surface."""

    positive = value >= 0
    result = np.empty_like(value, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _dose_dlt_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    doses: list[float] = []
    events: list[float] = []
    totals: list[float] = []
    for index, row in enumerate(rows):
        raw_dose = row.get("canonical_dose", row.get("dose"))
        dose = _finite_number(raw_dose, label=f"rows[{index}].dose")
        event = _finite_number(
            row.get("n_patients_with_dlt"), label=f"rows[{index}].n_patients_with_dlt"
        )
        total = _finite_number(row.get("n_patients"), label=f"rows[{index}].n_patients")
        if dose <= 0:
            raise ValueError(f"rows[{index}].dose must be > 0.")
        if not event.is_integer() or not total.is_integer():
            raise ValueError(
                f"rows[{index}] Binomial DLT counts must be integer-valued."
            )
        if total <= 0 or event < 0 or event > total:
            raise ValueError(f"rows[{index}] has invalid Binomial DLT counts.")
        doses.append(dose)
        events.append(float(int(event)))
        totals.append(float(int(total)))
    if not doses:
        raise ValueError("At least one observed dose--DLT point is required.")
    return (
        np.asarray(doses, dtype=float),
        np.asarray(events, dtype=float),
        np.asarray(totals, dtype=float),
    )


def _negative_log_posterior(
    theta: np.ndarray,
    *,
    log_dose_offsets: np.ndarray,
    events: np.ndarray,
    totals: np.ndarray,
    prior: BayesianPrior,
) -> float:
    alpha, eta = theta
    slope = float(softplus(float(eta)))
    linear_predictor = alpha + slope * log_dose_offsets
    log_probability = -np.logaddexp(0.0, -linear_predictor)
    log_one_minus_probability = -np.logaddexp(0.0, linear_predictor)
    log_likelihood = float(
        np.sum(events * log_probability + (totals - events) * log_one_minus_probability)
    )
    log_prior = -0.5 * (
        ((alpha - prior.alpha_mean) / prior.alpha_sd) ** 2
        + ((eta - prior.eta_mean) / prior.eta_sd) ** 2
    )
    value = -(log_likelihood + log_prior)
    return value if math.isfinite(value) else math.inf


def _negative_log_posterior_gradient(
    theta: np.ndarray,
    *,
    log_dose_offsets: np.ndarray,
    events: np.ndarray,
    totals: np.ndarray,
    prior: BayesianPrior,
) -> np.ndarray:
    alpha, eta = theta
    eta_sigmoid = float(_expit(np.asarray([eta]))[0])
    slope = float(softplus(float(eta)))
    linear_predictor = alpha + slope * log_dose_offsets
    probabilities = _expit(linear_predictor)
    residual = totals * probabilities - events
    return np.asarray(
        [
            np.sum(residual) + (alpha - prior.alpha_mean) / prior.alpha_sd**2,
            np.sum(residual * log_dose_offsets * eta_sigmoid)
            + (eta - prior.eta_mean) / prior.eta_sd**2,
        ],
        dtype=float,
    )


def _negative_log_posterior_hessian(
    theta: np.ndarray,
    *,
    log_dose_offsets: np.ndarray,
    events: np.ndarray,
    totals: np.ndarray,
    prior: BayesianPrior,
) -> np.ndarray:
    """Analytic Hessian used for a stable Laplace covariance."""

    alpha, eta = theta
    eta_sigmoid = float(_expit(np.asarray([eta]))[0])
    eta_second_derivative = eta_sigmoid * (1.0 - eta_sigmoid)
    slope = float(softplus(float(eta)))
    linear_predictor = alpha + slope * log_dose_offsets
    probabilities = _expit(linear_predictor)
    curvature = totals * probabilities * (1.0 - probabilities)
    residual = totals * probabilities - events
    d_eta = log_dose_offsets * eta_sigmoid
    d2_eta = log_dose_offsets * eta_second_derivative
    return np.asarray(
        [
            [
                np.sum(curvature) + 1.0 / prior.alpha_sd**2,
                np.sum(curvature * d_eta),
            ],
            [
                np.sum(curvature * d_eta),
                np.sum(curvature * d_eta**2 + residual * d2_eta)
                + 1.0 / prior.eta_sd**2,
            ],
        ],
        dtype=float,
    )


def _credible_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
    }


def _representable_positive_exp(log_value: float) -> float:
    """Exponentiate a finite log-dose while avoiding implementation overflow."""

    maximum = float(np.finfo(float).max)
    minimum = float(np.nextafter(0.0, 1.0))
    if log_value >= math.log(maximum):
        return maximum
    if log_value <= math.log(minimum):
        return minimum
    return math.exp(log_value)


def _display_dose_ratio(log_offset: float) -> float | None:
    """Return a display-only dose ratio, or None when it is not representable."""

    maximum = float(np.finfo(float).max)
    minimum = float(np.nextafter(0.0, 1.0))
    if log_offset >= math.log(maximum) or log_offset <= math.log(minimum):
        return None
    return math.exp(log_offset)


def fit_bayesian_binomial_logistic(
    rows: list[dict[str, Any]],
    prior: BayesianPrior,
    *,
    draw_count: int = DEFAULT_POSTERIOR_DRAW_COUNT,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Fit one monotone dose--DLT curve with a Laplace posterior approximation."""

    if draw_count < 32:
        raise ValueError("draw_count must be at least 32.")
    doses, events, totals = _dose_dlt_arrays(rows)
    log_doses = np.log(doses)
    log_dose_reference = float(np.mean(log_doses))
    dose_reference = _representable_positive_exp(log_dose_reference)
    log_dose_offsets = log_doses - log_dose_reference
    initial = np.asarray([prior.alpha_mean, prior.eta_mean], dtype=float)
    objective = partial(
        _negative_log_posterior,
        log_dose_offsets=log_dose_offsets,
        events=events,
        totals=totals,
        prior=prior,
    )
    gradient_function = partial(
        _negative_log_posterior_gradient,
        log_dose_offsets=log_dose_offsets,
        events=events,
        totals=totals,
        prior=prior,
    )

    result = minimize(
        objective,
        initial,
        jac=gradient_function,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1_000},
    )
    gradient = gradient_function(result.x)
    gradient_norm = float(np.linalg.norm(gradient))
    if (
        not np.isfinite(result.x).all()
        or not math.isfinite(float(result.fun))
        or gradient_norm > 1e-4
    ):
        return {
            "status": "not_fit_eligible",
            "skip_reasons": ["bayesian_map_optimization_failed"],
            "fit_diagnostics": {
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "gradient_norm": gradient_norm,
            },
        }

    hessian = _negative_log_posterior_hessian(
        result.x,
        log_dose_offsets=log_dose_offsets,
        events=events,
        totals=totals,
        prior=prior,
    )
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
        if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 1e-10:
            raise np.linalg.LinAlgError("Hessian is not positive definite.")
        covariance = np.linalg.inv(hessian)
        covariance = (covariance + covariance.T) / 2.0
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        return {
            "status": "not_fit_eligible",
            "skip_reasons": ["bayesian_laplace_hessian_failed"],
            "fit_diagnostics": {
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "gradient_norm": gradient_norm,
                "hessian_error": str(error),
            },
        }

    random_generator = np.random.default_rng(random_seed)
    draws = random_generator.multivariate_normal(result.x, covariance, size=draw_count)
    alpha_draws = draws[:, 0]
    eta_draws = draws[:, 1]
    slope_draws = np.asarray(softplus(eta_draws), dtype=float)
    fitted_draws = _expit(
        alpha_draws[:, None] + slope_draws[:, None] * log_dose_offsets[None, :]
    )
    fitted_means = np.mean(fitted_draws, axis=0)
    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    return {
        "status": "fitted",
        "model": MODEL_NAME,
        "posterior_approximation": POSTERIOR_APPROXIMATION,
        "prior_id": prior.name,
        "prior": prior.to_payload(),
        "dose_standardization": {
            "type": "natural_log_ratio_to_geometric_mean_observed_dose",
            "reference_dose": dose_reference,
            "reference_log_dose": log_dose_reference,
            "reference_definition": "geometric_mean_of_observed_doses",
            "log_base": "e",
        },
        "posterior": {
            "mode": {
                "alpha": float(result.x[0]),
                "eta": float(result.x[1]),
                "monotone_slope": float(softplus(float(result.x[1]))),
            },
            "covariance_alpha_eta": covariance.tolist(),
            "draw_count": draw_count,
            "draw_seed": int(random_seed),
            "draws_alpha_eta": draws.tolist(),
            "parameter_credible_interval_95": {
                "alpha": _credible_interval(alpha_draws),
                "eta": _credible_interval(eta_draws),
                "monotone_slope": _credible_interval(slope_draws),
            },
        },
        "fit_diagnostics": {
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "negative_log_posterior_at_mode": float(result.fun),
            "gradient_norm": gradient_norm,
            "hessian_eigenvalues": [float(value) for value in eigenvalues],
            "hessian_condition_number": condition_number,
            "laplace_quality": (
                "ill_conditioned" if condition_number > 1e8 else "acceptable"
            ),
        },
        "fitted_dlt_probability": [
            {
                "dose": float(dose),
                "posterior_mean": float(mean),
                "posterior_credible_interval_95": _credible_interval(draws_at_dose),
            }
            for dose, mean, draws_at_dose in zip(
                doses, fitted_means, fitted_draws.T, strict=True
            )
        ],
    }


def fit_mle_binomial_logistic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit the same monotone logistic form by classical Binomial MLE (no prior).

    ``logit(p) = alpha + softplus(eta) * log(dose / geometric_mean_dose)``.

    This is the non-Bayesian counterpart of ``fit_bayesian_binomial_logistic`` and
    is intended for visual reference overlays only.
    """

    doses, events, totals = _dose_dlt_arrays(rows)
    eligibility_reasons: list[str] = []
    if len(np.unique(doses)) < 2:
        eligibility_reasons.append("mle_requires_two_distinct_doses")
    total_events = float(np.sum(events))
    total_observations = float(np.sum(totals))
    if total_events <= 0:
        eligibility_reasons.append("mle_requires_at_least_one_event")
    if total_events >= total_observations:
        eligibility_reasons.append("mle_requires_at_least_one_non_event")
    if eligibility_reasons:
        return {
            "status": "not_fit_eligible",
            "skip_reasons": eligibility_reasons,
            "fit_diagnostics": {
                "distinct_dose_count": int(len(np.unique(doses))),
                "event_count": int(total_events),
                "non_event_count": int(total_observations - total_events),
            },
        }
    log_doses = np.log(doses)
    log_dose_reference = float(np.mean(log_doses))
    dose_reference = _representable_positive_exp(log_dose_reference)
    log_dose_offsets = log_doses - log_dose_reference
    overall_rate = float(np.sum(events) / np.sum(totals))
    clipped_rate = min(max(overall_rate, 1e-6), 1.0 - 1e-6)
    initial_alpha = float(math.log(clipped_rate / (1.0 - clipped_rate)))
    initial = np.asarray([initial_alpha, 0.0], dtype=float)

    def objective(theta: np.ndarray) -> float:
        alpha, eta = theta
        slope = float(softplus(float(eta)))
        linear_predictor = alpha + slope * log_dose_offsets
        log_probability = -np.logaddexp(0.0, -linear_predictor)
        log_one_minus_probability = -np.logaddexp(0.0, linear_predictor)
        log_likelihood = float(
            np.sum(
                events * log_probability
                + (totals - events) * log_one_minus_probability
            )
        )
        value = -log_likelihood
        return value if math.isfinite(value) else math.inf

    def gradient(theta: np.ndarray) -> np.ndarray:
        alpha, eta = theta
        eta_sigmoid = float(_expit(np.asarray([eta]))[0])
        slope = float(softplus(float(eta)))
        linear_predictor = alpha + slope * log_dose_offsets
        probabilities = _expit(linear_predictor)
        residual = totals * probabilities - events
        return np.asarray(
            [
                np.sum(residual),
                np.sum(residual * log_dose_offsets * eta_sigmoid),
            ],
            dtype=float,
        )

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1_000},
    )
    gradient_norm = float(np.linalg.norm(gradient(result.x)))
    if (
        not np.isfinite(result.x).all()
        or not math.isfinite(float(result.fun))
        or gradient_norm > 1e-3
    ):
        return {
            "status": "not_fit_eligible",
            "skip_reasons": ["mle_logistic_optimization_failed"],
            "fit_diagnostics": {
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "gradient_norm": gradient_norm,
            },
        }

    alpha_hat = float(result.x[0])
    eta_hat = float(result.x[1])
    slope_hat = float(softplus(eta_hat))
    fitted_means = _expit(alpha_hat + slope_hat * log_dose_offsets)
    return {
        "status": "fitted",
        "model": "monotone_mle_binomial_logistic_log_dose_v1",
        "estimation": "binomial_maximum_likelihood",
        "dose_standardization": {
            "type": "natural_log_ratio_to_geometric_mean_observed_dose",
            "reference_dose": dose_reference,
            "reference_log_dose": log_dose_reference,
            "reference_definition": "geometric_mean_of_observed_doses",
            "log_base": "e",
        },
        "parameters": {
            "alpha": alpha_hat,
            "eta": eta_hat,
            "monotone_slope": slope_hat,
        },
        "fit_diagnostics": {
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "negative_log_likelihood": float(result.fun),
            "gradient_norm": gradient_norm,
        },
        "fitted_dlt_probability": [
            {
                "dose": float(dose),
                "dlt_probability": float(mean),
            }
            for dose, mean in zip(doses, fitted_means, strict=True)
        ],
    }


def mle_probability(fit: dict[str, Any], dose: float) -> float:
    """Evaluate a fitted classical MLE logistic curve at one dose."""

    resolved_dose = _finite_number(dose, label="dose")
    if resolved_dose <= 0:
        raise ValueError("dose must be > 0.")
    if fit.get("status") != "fitted":
        raise ValueError("Cannot predict from a non-fitted MLE logistic curve.")
    standardization = fit.get("dose_standardization")
    parameters = fit.get("parameters")
    if not isinstance(standardization, dict) or not isinstance(parameters, dict):
        raise ValueError("MLE logistic fit lacks parameter metadata.")
    reference_log_dose = _finite_number(
        standardization.get("reference_log_dose"),
        label="dose_standardization.reference_log_dose",
    )
    alpha = _finite_number(parameters.get("alpha"), label="parameters.alpha")
    eta = _finite_number(parameters.get("eta"), label="parameters.eta")
    log_dose_offset = math.log(resolved_dose) - reference_log_dose
    return float(_expit(np.asarray([alpha + softplus(eta) * log_dose_offset]))[0])


def posterior_probability_draws(
    fit: dict[str, Any], dose: float
) -> np.ndarray:
    """Evaluate stored posterior draws at one raw canonical-unit dose."""

    resolved_dose = _finite_number(dose, label="dose")
    if resolved_dose <= 0:
        raise ValueError("dose must be > 0.")
    if fit.get("status") != "fitted":
        raise ValueError("Cannot predict from a non-fitted Bayesian curve.")
    standardization = fit.get("dose_standardization")
    posterior = fit.get("posterior")
    if not isinstance(standardization, dict) or not isinstance(posterior, dict):
        raise ValueError("Bayesian curve cache entry lacks posterior metadata.")
    if (
        standardization.get("type")
        != "natural_log_ratio_to_geometric_mean_observed_dose"
    ):
        raise ValueError(
            "Bayesian curve cache uses an unsupported dose transformation. "
            "Rebuild the cache."
        )
    reference_dose = _finite_number(
        standardization.get("reference_dose"), label="dose_standardization.reference_dose"
    )
    if reference_dose <= 0:
        raise ValueError("Bayesian curve reference dose must be > 0.")
    draws = np.asarray(posterior.get("draws_alpha_eta"), dtype=float)
    if (
        draws.ndim != 2
        or draws.shape[1] != 2
        or len(draws) == 0
        or not np.isfinite(draws).all()
    ):
        raise ValueError("Bayesian curve cache entry lacks valid posterior draws.")
    raw_reference_log_dose = standardization.get("reference_log_dose")
    reference_log_dose = (
        _finite_number(
            raw_reference_log_dose,
            label="dose_standardization.reference_log_dose",
        )
        if raw_reference_log_dose is not None
        else math.log(reference_dose)
    )
    log_dose_offset = math.log(resolved_dose) - reference_log_dose
    return _expit(draws[:, 0] + softplus(draws[:, 1]) * log_dose_offset)


def posterior_probability_summary(fit: dict[str, Any], dose: float) -> dict[str, Any]:
    draws = posterior_probability_draws(fit, dose)
    return {
        "posterior_mean": float(np.mean(draws)),
        "posterior_credible_interval_95": _credible_interval(draws),
        "draw_count": int(len(draws)),
    }


def prior_predictive_summary(
    prior: BayesianPrior,
    *,
    log_dose_offsets: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    draw_count: int = DEFAULT_PRIOR_PREDICTIVE_DRAW_COUNT,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Summarize a candidate prior before it is allowed to dominate sparse arms."""

    if draw_count < 32:
        raise ValueError("draw_count must be at least 32.")
    offsets = np.asarray(log_dose_offsets, dtype=float)
    if not np.isfinite(offsets).all():
        raise ValueError("log_dose_offsets must be finite.")
    generator = np.random.default_rng(random_seed)
    alpha = generator.normal(prior.alpha_mean, prior.alpha_sd, draw_count)
    eta = generator.normal(prior.eta_mean, prior.eta_sd, draw_count)
    probabilities = _expit(
        alpha[:, None] + softplus(eta)[:, None] * offsets[None, :]
    )
    return {
        "prior_id": prior.name,
        "prior": prior.to_payload(),
        "draw_count": draw_count,
        "random_seed": random_seed,
        "log_dose_offset_grid": [float(value) for value in offsets],
        "dlt_probability": [
            {
                "log_dose_offset": float(offset),
                "dose_ratio_to_geometric_mean": _display_dose_ratio(float(offset)),
                "mean": float(np.mean(draws)),
                "credible_interval_95": _credible_interval(draws),
            }
            for offset, draws in zip(offsets, probabilities.T, strict=True)
        ],
    }
