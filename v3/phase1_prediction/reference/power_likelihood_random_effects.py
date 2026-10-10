"""Study-random-intercept extension of the powered historical likelihood.

The fixed-effect challenger in :mod:`power_likelihood_curve` assumes that all
accepted historical study units share one curve.  This module keeps the same
audited source rows, likelihood powers, prior, target dose coordinate, and ESS
cap, but lets each independent historical study unit shift the curve on the
log-odds scale::

    logit(p_gj) = alpha + b_g + softplus(eta) * log(d_gj / d_ref)
    b_g | tau ~ Normal(0, tau**2)

The study effects are integrated out with deterministic Gauss--Hermite
quadrature.  The posterior over ``alpha, eta, rho`` (where
``tau = softplus(rho)``) is approximated by a Laplace Normal at the MAP.  A new
target-study random intercept is then sampled once per posterior draw and held
fixed across every dose on that complete draw-level curve.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .bayesian_toxicity import BayesianPrior, softplus
from .power_likelihood_curve import (
    DEFAULT_BORROWING_STRENGTH,
    DEFAULT_HISTORICAL_ESS_CAP,
    _canonical_hash,
    _finite_number,
    _prepare_powered_lines,
    _prior_from_lines,
    PowerLikelihoodSettings,
)
from .prediction_contract import ensure_current_prediction
from .toxicity_curve_common import ToxicityQuery


MODEL_NAME = "similarity_weighted_power_likelihood_random_intercept_logistic_laplace_v1"
POSTERIOR_APPROXIMATION = (
    "gauss_hermite_marginal_likelihood_plus_laplace_normal_in_alpha_eta_rho"
)
DEFAULT_TAU_LATENT_MEAN = -1.1
DEFAULT_TAU_LATENT_SD = 1.0
DEFAULT_QUADRATURE_NODES = 32


@dataclass(frozen=True)
class RandomEffectsSettings:
    """Controls for the study-level random-intercept extension."""

    historical_ess_cap: float = DEFAULT_HISTORICAL_ESS_CAP
    borrowing_strength: float = DEFAULT_BORROWING_STRENGTH
    tau_latent_mean: float = DEFAULT_TAU_LATENT_MEAN
    tau_latent_sd: float = DEFAULT_TAU_LATENT_SD
    quadrature_nodes: int = DEFAULT_QUADRATURE_NODES
    random_seed: int = 20260822

    def validated(self) -> "RandomEffectsSettings":
        base = PowerLikelihoodSettings(
            historical_ess_cap=self.historical_ess_cap,
            borrowing_strength=self.borrowing_strength,
            random_seed=self.random_seed,
        ).validated()
        rho_mean = _finite_number(self.tau_latent_mean, "tau_latent_mean")
        rho_sd = _finite_number(self.tau_latent_sd, "tau_latent_sd")
        if rho_sd <= 0:
            raise ValueError("tau_latent_sd must be > 0.")
        if (
            isinstance(self.quadrature_nodes, bool)
            or not isinstance(self.quadrature_nodes, int)
            or self.quadrature_nodes < 16
            or self.quadrature_nodes > 128
        ):
            raise ValueError("quadrature_nodes must be an integer in [16, 128].")
        return RandomEffectsSettings(
            historical_ess_cap=base.historical_ess_cap,
            borrowing_strength=base.borrowing_strength,
            tau_latent_mean=rho_mean,
            tau_latent_sd=rho_sd,
            quadrature_nodes=self.quadrature_nodes,
            random_seed=base.random_seed,
        )


def _study_group_key(line: Mapping[str, Any]) -> tuple[str, str]:
    document_id = str(line.get("document_id") or line.get("pmcid") or "")
    study_unit_id = str(line.get("study_unit_id") or "document_level")
    return document_id, study_unit_id


def _group_rows(
    accepted: Sequence[Mapping[str, Any]], reference_log_dose: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    line_indexes: dict[tuple[str, str], set[int]] = defaultdict(set)
    for line_index, item in enumerate(accepted):
        key = _study_group_key(item["line"])
        power = float(item["likelihood_power"])
        for row in item["rows"]:
            grouped[key].append(
                {
                    "log_dose_offset": math.log(float(row["canonical_dose"]))
                    - reference_log_dose,
                    "events": int(row["n_patients_with_dlt"]),
                    "total": int(row["n_patients"]),
                    "power": power,
                    "cohort_id": row.get("cohort_id"),
                    "denominator_role": row.get("denominator_role"),
                }
            )
            line_indexes[key].add(line_index)
    return [
        {
            "document_id": key[0],
            "study_unit_id": key[1],
            "group_id": f"{key[0]}::{key[1]}",
            "rows": rows,
            "line_indexes": sorted(line_indexes[key]),
        }
        for key, rows in sorted(grouped.items())
    ]


def _quadrature(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(nodes)
    standard_normal_nodes = math.sqrt(2.0) * raw_nodes
    log_standard_normal_weights = np.log(raw_weights) - 0.5 * math.log(math.pi)
    return standard_normal_nodes, log_standard_normal_weights


def _group_log_marginal_and_gradient(
    theta: np.ndarray,
    group: Mapping[str, Any],
    quadrature_nodes: np.ndarray,
    log_quadrature_weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    alpha, eta, rho = theta
    slope = float(softplus(float(eta)))
    slope_derivative = float(expit(eta))
    tau = float(softplus(float(rho)))
    tau_derivative = float(expit(rho))

    rows = group["rows"]
    offsets = np.asarray([row["log_dose_offset"] for row in rows], dtype=float)
    events = np.asarray([row["events"] for row in rows], dtype=float)
    totals = np.asarray([row["total"] for row in rows], dtype=float)
    powers = np.asarray([row["power"] for row in rows], dtype=float)

    random_intercepts = tau * quadrature_nodes
    linear_predictor = alpha + random_intercepts[:, None] + slope * offsets[None, :]
    log_probability = -np.logaddexp(0.0, -linear_predictor)
    log_one_minus_probability = -np.logaddexp(0.0, linear_predictor)
    node_log_likelihood = np.sum(
        powers[None, :]
        * (
            events[None, :] * log_probability
            + (totals - events)[None, :] * log_one_minus_probability
        ),
        axis=1,
    )
    combined_log_weights = log_quadrature_weights + node_log_likelihood
    log_marginal = float(logsumexp(combined_log_weights))
    posterior_node_weights = np.exp(combined_log_weights - log_marginal)

    probabilities = expit(linear_predictor)
    residual = powers[None, :] * (totals[None, :] * probabilities - events[None, :])
    conditional_gradient = np.column_stack(
        [
            np.sum(residual, axis=1),
            np.sum(residual * offsets[None, :] * slope_derivative, axis=1),
            np.sum(
                residual * quadrature_nodes[:, None] * tau_derivative,
                axis=1,
            ),
        ]
    )
    marginal_negative_gradient = np.sum(
        posterior_node_weights[:, None] * conditional_gradient,
        axis=0,
    )
    return log_marginal, marginal_negative_gradient, posterior_node_weights


def _objective_and_gradient(
    theta: np.ndarray,
    *,
    groups: Sequence[Mapping[str, Any]],
    quadrature_nodes: np.ndarray,
    log_quadrature_weights: np.ndarray,
    prior: BayesianPrior,
    settings: RandomEffectsSettings,
) -> tuple[float, np.ndarray]:
    alpha, eta, rho = theta
    negative_log_posterior = 0.5 * (
        ((alpha - prior.alpha_mean) / prior.alpha_sd) ** 2
        + ((eta - prior.eta_mean) / prior.eta_sd) ** 2
        + ((rho - settings.tau_latent_mean) / settings.tau_latent_sd) ** 2
    )
    gradient = np.asarray(
        [
            (alpha - prior.alpha_mean) / prior.alpha_sd**2,
            (eta - prior.eta_mean) / prior.eta_sd**2,
            (rho - settings.tau_latent_mean) / settings.tau_latent_sd**2,
        ],
        dtype=float,
    )
    for group in groups:
        log_marginal, group_gradient, _weights = _group_log_marginal_and_gradient(
            theta,
            group,
            quadrature_nodes,
            log_quadrature_weights,
        )
        negative_log_posterior -= log_marginal
        gradient += group_gradient
    result = float(negative_log_posterior)
    if not math.isfinite(result) or not np.isfinite(gradient).all():
        return math.inf, np.full(3, math.nan)
    return result, gradient


def _numerical_hessian(
    theta: np.ndarray,
    gradient_function: Any,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for index in range(len(theta)):
        step = 1e-4 * max(1.0, abs(float(theta[index])))
        upper = theta.copy()
        lower = theta.copy()
        upper[index] += step
        lower[index] -= step
        columns.append(
            (gradient_function(upper) - gradient_function(lower)) / (2 * step)
        )
    hessian = np.column_stack(columns)
    return (hessian + hessian.T) / 2.0


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    return float(ordered_values[np.searchsorted(cumulative, probability, side="left")])


def _historical_group_diagnostics(
    theta: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
    quadrature_nodes: np.ndarray,
    log_quadrature_weights: np.ndarray,
) -> list[dict[str, Any]]:
    tau = float(softplus(float(theta[2])))
    random_intercepts = tau * quadrature_nodes
    output: list[dict[str, Any]] = []
    for group in groups:
        _log_marginal, _gradient, weights = _group_log_marginal_and_gradient(
            theta,
            group,
            quadrature_nodes,
            log_quadrature_weights,
        )
        mean = float(np.sum(weights * random_intercepts))
        variance = float(np.sum(weights * (random_intercepts - mean) ** 2))
        output.append(
            {
                "group_id": group["group_id"],
                "document_id": group["document_id"],
                "study_unit_id": group["study_unit_id"],
                "source_row_count": len(group["rows"]),
                "random_intercept_posterior_mean": mean,
                "random_intercept_posterior_sd": math.sqrt(max(0.0, variance)),
                "random_intercept_credible_interval_95": {
                    "low": _weighted_quantile(random_intercepts, weights, 0.025),
                    "high": _weighted_quantile(random_intercepts, weights, 0.975),
                },
            }
        )
    return output


def _contributor_audit(
    accepted: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in accepted:
        line = item["line"]
        output.append(
            {
                "document_id": line.get("document_id") or line.get("pmcid"),
                "pmcid": line.get("pmcid") or line.get("document_id"),
                "study_unit_id": line.get("study_unit_id"),
                "random_effect_group_id": "::".join(_study_group_key(line)),
                "escalation_series_id": line.get("escalation_series_id"),
                "segment_index": line.get("segment_index"),
                "drug_name": line.get("drug_name"),
                "similarity": line.get("similarity"),
                "segment_similarity_weight": line.get("segment_similarity_weight"),
                "reliability_weight": line.get("reliability_weight"),
                "raw_likelihood_power": float(item["raw_power"]),
                "likelihood_power": float(item["likelihood_power"]),
                "source_patient_count": int(item["patient_count"]),
                "effective_patient_count": float(item["effective_patient_count"]),
                "source_point_count": len(item["rows"]),
                "denominator_roles": sorted(
                    {str(row["denominator_role"]) for row in item["rows"]}
                ),
                "observed_dose_min": line.get("observed_dose_min"),
                "observed_dose_max": line.get("observed_dose_max"),
            }
        )
    return output


def estimate_power_likelihood_random_effects_curve(
    historical_lines: Sequence[Mapping[str, Any]],
    spec: ToxicityQuery,
    *,
    bayesian_prior: str,
    settings: RandomEffectsSettings | None = None,
) -> dict[str, Any]:
    """Fit a powered shared curve with one random intercept per study unit."""

    resolved = (settings or RandomEffectsSettings()).validated()
    if not spec.planned_doses:
        raise ValueError("Random-effects fitting requires a target dose grid.")
    power_settings = PowerLikelihoodSettings(
        historical_ess_cap=resolved.historical_ess_cap,
        borrowing_strength=resolved.borrowing_strength,
        random_seed=resolved.random_seed,
    )
    accepted, skipped, weight_diagnostics = _prepare_powered_lines(
        historical_lines,
        power_settings,
    )
    if not accepted:
        return {
            "status": "no_eligible_original_binomial_evidence",
            "model": MODEL_NAME,
            "model_assumption": "study_random_intercept_shared_slope",
            "curve_points": [],
            "historical_lines": [],
            "weight_diagnostics": weight_diagnostics,
            "skipped_reasons": skipped,
        }
    try:
        prior = _prior_from_lines([item["line"] for item in accepted], bayesian_prior)
    except ValueError as error:
        return {
            "status": "prior_contract_failed",
            "model": MODEL_NAME,
            "model_assumption": "study_random_intercept_shared_slope",
            "curve_points": [],
            "historical_lines": [],
            "weight_diagnostics": weight_diagnostics,
            "skipped_reasons": {str(error): 1, **skipped},
        }

    reference_log_dose = float(np.mean(np.log(np.asarray(spec.planned_doses))))
    reference_dose = float(math.exp(reference_log_dose))
    groups = _group_rows(accepted, reference_log_dose)
    if len(groups) < 2:
        return {
            "status": "insufficient_independent_studies_for_random_effects",
            "model": MODEL_NAME,
            "model_assumption": "study_random_intercept_shared_slope",
            "curve_points": [],
            "historical_lines": _contributor_audit(accepted),
            "weight_diagnostics": {
                **weight_diagnostics,
                "independent_study_group_count": len(groups),
            },
            "skipped_reasons": skipped,
        }

    quadrature_nodes, log_quadrature_weights = _quadrature(resolved.quadrature_nodes)
    arguments = {
        "groups": groups,
        "quadrature_nodes": quadrature_nodes,
        "log_quadrature_weights": log_quadrature_weights,
        "prior": prior,
        "settings": resolved,
    }

    def objective(theta: np.ndarray) -> float:
        return _objective_and_gradient(theta, **arguments)[0]

    def gradient(theta: np.ndarray) -> np.ndarray:
        return _objective_and_gradient(theta, **arguments)[1]

    starts = [
        np.asarray([prior.alpha_mean, prior.eta_mean, rho], dtype=float)
        for rho in (resolved.tau_latent_mean, -2.0, 0.0)
    ]
    fits = [
        minimize(
            objective,
            start,
            jac=gradient,
            method="BFGS",
            options={"gtol": 1e-7, "maxiter": 1_000},
        )
        for start in starts
    ]
    finite_fits = [
        fit
        for fit in fits
        if np.isfinite(fit.x).all() and math.isfinite(float(fit.fun))
    ]
    if not finite_fits:
        return {
            "status": "map_optimization_failed",
            "model": MODEL_NAME,
            "model_assumption": "study_random_intercept_shared_slope",
            "curve_points": [],
            "historical_lines": _contributor_audit(accepted),
            "weight_diagnostics": weight_diagnostics,
            "skipped_reasons": skipped,
        }
    fit = min(finite_fits, key=lambda item: float(item.fun))
    gradient_at_mode = gradient(fit.x)
    gradient_norm = float(np.linalg.norm(gradient_at_mode))
    if not np.isfinite(gradient_at_mode).all() or gradient_norm > 5e-4:
        return {
            "status": "map_optimization_failed",
            "model": MODEL_NAME,
            "model_assumption": "study_random_intercept_shared_slope",
            "curve_points": [],
            "historical_lines": _contributor_audit(accepted),
            "weight_diagnostics": weight_diagnostics,
            "skipped_reasons": skipped,
            "fit_diagnostics": {
                "optimizer_success": bool(fit.success),
                "optimizer_message": str(fit.message),
                "gradient_norm": gradient_norm,
            },
        }

    hessian = _numerical_hessian(fit.x, gradient)
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
        if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 1e-9:
            raise np.linalg.LinAlgError("Hessian is not positive definite.")
        covariance = np.linalg.inv(hessian)
        covariance = (covariance + covariance.T) / 2.0
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        return {
            "status": "laplace_hessian_failed",
            "model": MODEL_NAME,
            "model_assumption": "study_random_intercept_shared_slope",
            "curve_points": [],
            "historical_lines": _contributor_audit(accepted),
            "weight_diagnostics": weight_diagnostics,
            "skipped_reasons": skipped,
            "fit_diagnostics": {
                "optimizer_success": bool(fit.success),
                "optimizer_message": str(fit.message),
                "gradient_norm": gradient_norm,
                "hessian_error": str(error),
            },
        }

    existing_draw_counts = [
        int(item["line"]["bayesian_fit"]["posterior"]["draw_count"])
        for item in accepted
        if isinstance(item["line"].get("bayesian_fit"), Mapping)
        and isinstance(item["line"]["bayesian_fit"].get("posterior"), Mapping)
        and isinstance(item["line"]["bayesian_fit"]["posterior"].get("draw_count"), int)
    ]
    draw_count = max(32, min(existing_draw_counts) if existing_draw_counts else 1_024)
    seed_payload = {
        "settings": resolved.__dict__,
        "prior": prior.to_payload(),
        "reference_dose": reference_dose,
        "groups": groups,
    }
    derived_seed = int.from_bytes(
        hashlib.sha256(
            f"{resolved.random_seed}|{_canonical_hash(seed_payload)}".encode("utf-8")
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )
    rng = np.random.default_rng(derived_seed)
    parameter_draws = rng.multivariate_normal(fit.x, covariance, size=draw_count)
    tau_draws = softplus(parameter_draws[:, 2])
    target_standard_normal_draws = rng.standard_normal(draw_count)
    target_random_intercept_draws = tau_draws * target_standard_normal_draws
    doses = np.asarray(spec.planned_doses, dtype=float)
    target_offsets = np.log(doses) - reference_log_dose
    slopes = softplus(parameter_draws[:, 1])
    population_mean = expit(
        parameter_draws[:, 0, None] + slopes[:, None] * target_offsets[None, :]
    )
    target_predictive = expit(
        parameter_draws[:, 0, None]
        + target_random_intercept_draws[:, None]
        + slopes[:, None] * target_offsets[None, :]
    )

    curve_points: list[dict[str, Any]] = []
    population_mean_points: list[dict[str, Any]] = []
    powered_weight_sum = sum(float(item["likelihood_power"]) for item in accepted)
    for index, dose in enumerate(doses):
        predictive_draws = target_predictive[:, index]
        mean_draws = population_mean[:, index]
        extrapolation_power = sum(
            float(item["likelihood_power"])
            for item in accepted
            if float(dose) < float(item["line"].get("observed_dose_min", dose))
            or float(dose) > float(item["line"].get("observed_dose_max", dose))
        )
        common = {
            "dose": float(dose),
            "unit": spec.target_unit,
            "contributor_count": len(accepted),
            "independent_study_group_count": len(groups),
            "effective_historical_sample_size": weight_diagnostics[
                "effective_historical_sample_size"
            ],
            "extrapolation_power_fraction": (
                float(extrapolation_power / powered_weight_sum)
                if powered_weight_sum > 0
                else 0.0
            ),
        }
        curve_points.append(
            {
                **common,
                "prediction_target": "new_target_study_including_between_study_variation",
                "dlt_probability": float(np.mean(predictive_draws)),
                "posterior_credible_interval_95": {
                    "low": float(np.quantile(predictive_draws, 0.025)),
                    "high": float(np.quantile(predictive_draws, 0.975)),
                },
            }
        )
        population_mean_points.append(
            {
                **common,
                "prediction_target": "population_mean_study_random_intercept_zero",
                "dlt_probability": float(np.mean(mean_draws)),
                "posterior_credible_interval_95": {
                    "low": float(np.quantile(mean_draws, 0.025)),
                    "high": float(np.quantile(mean_draws, 0.975)),
                },
            }
        )

    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    tau_summary = {
        "posterior_mean": float(np.mean(tau_draws)),
        "posterior_median": float(np.median(tau_draws)),
        "posterior_credible_interval_95": {
            "low": float(np.quantile(tau_draws, 0.025)),
            "high": float(np.quantile(tau_draws, 0.975)),
        },
        "unit": "log_odds_standard_deviation",
    }
    document_ids = {
        str(item["line"].get("document_id") or item["line"].get("pmcid"))
        for item in accepted
    }
    return {
        "status": "fitted",
        "model": MODEL_NAME,
        "model_assumption": "study_random_intercept_shared_slope",
        "method_label": "Power likelihood with study random intercepts (experimental)",
        "bayesian_prior": bayesian_prior,
        "prior": prior.to_payload(),
        "random_effect_prior": {
            "parameterization": "tau=softplus(rho)",
            "rho_distribution": "normal",
            "rho_mean": resolved.tau_latent_mean,
            "rho_sd": resolved.tau_latent_sd,
            "tau_interpretation": "between-study log-odds standard deviation",
        },
        "posterior_approximation": POSTERIOR_APPROXIMATION,
        "marginal_likelihood_integration": {
            "method": "gauss_hermite_quadrature",
            "node_count": resolved.quadrature_nodes,
            "random_effect_distribution": "Normal(0, tau^2)",
            "likelihood_power_location": "conditional_binomial_likelihood_only",
        },
        "dose_standardization": {
            "type": "natural_log_ratio_to_target_dose_geometric_mean",
            "reference_dose": reference_dose,
            "reference_log_dose": reference_log_dose,
            "reference_definition": "geometric_mean_of_target_planned_or_display_doses",
            "log_base": "e",
        },
        "posterior": {
            "mode": {
                "alpha": float(fit.x[0]),
                "eta": float(fit.x[1]),
                "rho": float(fit.x[2]),
                "monotone_slope": float(softplus(float(fit.x[1]))),
                "tau": float(softplus(float(fit.x[2]))),
            },
            "covariance_alpha_eta_rho": covariance.tolist(),
            "draw_count": int(draw_count),
            "draw_seed": int(derived_seed),
            "draws_alpha_eta_rho": parameter_draws.tolist(),
            "target_standard_normal_draws": target_standard_normal_draws.tolist(),
            "target_random_intercept_draws": target_random_intercept_draws.tolist(),
        },
        "curve_points": curve_points,
        "population_mean_curve_points": population_mean_points,
        "historical_lines": _contributor_audit(accepted),
        "weight_diagnostics": {
            **weight_diagnostics,
            "contributor_line_count": len(accepted),
            "contributor_document_count": len(document_ids),
            "independent_study_group_count": len(groups),
            "random_effect_group_definition": "document_id_plus_study_unit_id",
            "duplicate_publications_of_one_trial_require_external_trial_id_review": True,
        },
        "heterogeneity_diagnostics": {
            "tau": tau_summary,
            "historical_group_random_effects_at_posterior_mode": (
                _historical_group_diagnostics(
                    fit.x,
                    groups,
                    quadrature_nodes,
                    log_quadrature_weights,
                )
            ),
            "interpretation": (
                "Tau is the between-study log-odds standard deviation. The main curve "
                "includes a new target-study random intercept; population_mean_curve_points "
                "sets that intercept to zero."
            ),
        },
        "fit_diagnostics": {
            "optimizer_success": bool(fit.success),
            "optimizer_message": str(fit.message),
            "gradient_norm": gradient_norm,
            "hessian_condition_number": condition_number,
            "multi_start_count": len(starts),
        },
        "skipped_reasons": skipped,
        "interpretation": (
            "Historical study units share one monotone slope but may shift up or down "
            "through Normal random intercepts. The main curve predicts a new target "
            "study and therefore includes between-study variation. Likelihood powers "
            "remain fixed information discounts and the prior is applied once."
        ),
    }


def random_effects_draws_at_doses(
    method: Mapping[str, Any], doses: Sequence[float]
) -> np.ndarray:
    """Evaluate the stored target-study predictive draws at positive doses."""

    ensure_current_prediction(method)
    if method.get("status") != "fitted" or method.get("model") != MODEL_NAME:
        raise ValueError("A fitted random-effects power-likelihood curve is required.")
    resolved_doses = np.asarray(doses, dtype=float)
    if (
        resolved_doses.ndim != 1
        or not len(resolved_doses)
        or not np.isfinite(resolved_doses).all()
        or np.any(resolved_doses <= 0)
    ):
        raise ValueError("doses must be positive finite values.")
    standardization = method.get("dose_standardization")
    posterior = method.get("posterior")
    if not isinstance(standardization, Mapping) or not isinstance(posterior, Mapping):
        raise ValueError("Random-effects method lacks posterior metadata.")
    reference_log_dose = _finite_number(
        standardization.get("reference_log_dose"), "reference_log_dose"
    )
    parameter_draws = np.asarray(posterior.get("draws_alpha_eta_rho"), dtype=float)
    random_intercepts = np.asarray(
        posterior.get("target_random_intercept_draws"), dtype=float
    )
    if (
        parameter_draws.ndim != 2
        or parameter_draws.shape[1] != 3
        or not len(parameter_draws)
        or not np.isfinite(parameter_draws).all()
        or random_intercepts.shape != (len(parameter_draws),)
        or not np.isfinite(random_intercepts).all()
    ):
        raise ValueError("Random-effects posterior draws are invalid.")
    offsets = np.log(resolved_doses) - reference_log_dose
    raw = expit(
        parameter_draws[:, 0, None]
        + random_intercepts[:, None]
        + softplus(parameter_draws[:, 1, None]) * offsets[None, :]
    )
    return raw


def random_effects_probability_summary(
    method: Mapping[str, Any], dose: float
) -> dict[str, Any]:
    draws = random_effects_draws_at_doses(method, [dose])[:, 0]
    return {
        "model": method.get("model"),
        "method": "similarity_weighted_power_likelihood_random_effects",
        "prediction_target": "new_target_study_including_between_study_variation",
        "dose": float(dose),
        "dlt_probability": float(np.mean(draws)),
        "posterior_credible_interval_95": {
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
        },
        "posterior_draw_count": int(len(draws)),
        "effective_historical_sample_size": method.get("weight_diagnostics", {}).get(
            "effective_historical_sample_size"
        ),
        "between_study_tau": method.get("heterogeneity_diagnostics", {})
        .get("tau", {})
        .get("posterior_mean"),
    }
