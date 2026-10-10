"""Powered historical likelihood with study random intercepts and slopes.

The independent study effects are marginalized with two-dimensional
Gauss--Hermite quadrature.  A Laplace approximation is then formed for the
four population parameters ``alpha``, ``eta``, ``rho_alpha`` and ``rho_eta``.
For a new target study, one intercept effect and one latent-slope effect are
drawn per posterior curve and held fixed across every evaluated dose.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .bayesian_toxicity import BayesianPrior, softplus
from .power_likelihood_curve import (
    DEFAULT_BORROWING_STRENGTH,
    DEFAULT_HISTORICAL_ESS_CAP,
    PowerLikelihoodSettings,
    _canonical_hash,
    _finite_number,
    _prepare_powered_lines,
    _prior_from_lines,
)
from .power_likelihood_random_effects import _contributor_audit, _group_rows
from .prediction_contract import ensure_current_prediction
from .toxicity_curve_common import ToxicityQuery


MODEL_NAME = "similarity_weighted_power_likelihood_random_intercept_slope_logistic_laplace_v1"
MODEL_VERSION = "three_method_random_intercept_slope_no_uplift_v1"
# Keep posterior random streams stable when only output/replay metadata changes.
POSTERIOR_SEED_VERSION = "three_method_random_intercept_slope_v1"
POSTERIOR_APPROXIMATION = (
    "two_dimensional_gauss_hermite_marginal_likelihood_plus_laplace_normal"
)
DEFAULT_INTERCEPT_SCALE_LATENT_MEAN = -1.1
DEFAULT_INTERCEPT_SCALE_LATENT_SD = 1.0
DEFAULT_SLOPE_SCALE_LATENT_MEAN = -1.5
DEFAULT_SLOPE_SCALE_LATENT_SD = 0.8
DEFAULT_QUADRATURE_NODES = 16
MIN_SCALE = 1e-8


def hierarchical_linear_predictor(
    alpha: float | np.ndarray,
    eta: float | np.ndarray,
    log_dose_offset: float | np.ndarray,
    *,
    random_intercept: float | np.ndarray = 0.0,
    random_slope_latent: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Evaluate the model and make its two exact degeneration checks explicit."""

    return (
        np.asarray(alpha)
        + np.asarray(random_intercept)
        + np.asarray(softplus(np.asarray(eta) + np.asarray(random_slope_latent)))
        * np.asarray(log_dose_offset)
    )


@dataclass(frozen=True)
class RandomInterceptSlopeSettings:
    historical_ess_cap: float = DEFAULT_HISTORICAL_ESS_CAP
    borrowing_strength: float = DEFAULT_BORROWING_STRENGTH
    intercept_scale_latent_mean: float = DEFAULT_INTERCEPT_SCALE_LATENT_MEAN
    intercept_scale_latent_sd: float = DEFAULT_INTERCEPT_SCALE_LATENT_SD
    slope_scale_latent_mean: float = DEFAULT_SLOPE_SCALE_LATENT_MEAN
    slope_scale_latent_sd: float = DEFAULT_SLOPE_SCALE_LATENT_SD
    quadrature_nodes: int = DEFAULT_QUADRATURE_NODES
    random_seed: int = 20260908

    def validated(self) -> "RandomInterceptSlopeSettings":
        base = PowerLikelihoodSettings(
            historical_ess_cap=self.historical_ess_cap,
            borrowing_strength=self.borrowing_strength,
            random_seed=self.random_seed,
        ).validated()
        values = (
            self.intercept_scale_latent_mean,
            self.intercept_scale_latent_sd,
            self.slope_scale_latent_mean,
            self.slope_scale_latent_sd,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("Random-effect scale-prior parameters must be finite.")
        if self.intercept_scale_latent_sd <= 0 or self.slope_scale_latent_sd <= 0:
            raise ValueError("Random-effect scale-prior standard deviations must be positive.")
        if (
            isinstance(self.quadrature_nodes, bool)
            or not isinstance(self.quadrature_nodes, int)
            or not 8 <= self.quadrature_nodes <= 32
        ):
            raise ValueError("quadrature_nodes must be an integer in [8, 32].")
        return RandomInterceptSlopeSettings(
            historical_ess_cap=base.historical_ess_cap,
            borrowing_strength=base.borrowing_strength,
            intercept_scale_latent_mean=float(self.intercept_scale_latent_mean),
            intercept_scale_latent_sd=float(self.intercept_scale_latent_sd),
            slope_scale_latent_mean=float(self.slope_scale_latent_mean),
            slope_scale_latent_sd=float(self.slope_scale_latent_sd),
            quadrature_nodes=self.quadrature_nodes,
            random_seed=base.random_seed,
        )


def _quadrature_grid(nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(nodes)
    standard = math.sqrt(2.0) * raw_nodes
    log_weights = np.log(raw_weights) - 0.5 * math.log(math.pi)
    intercept_nodes, slope_nodes = np.meshgrid(standard, standard, indexing="ij")
    intercept_log_weights, slope_log_weights = np.meshgrid(
        log_weights, log_weights, indexing="ij"
    )
    return (
        intercept_nodes.ravel(),
        slope_nodes.ravel(),
        (intercept_log_weights + slope_log_weights).ravel(),
    )


def _group_log_marginal_and_gradient(
    theta: np.ndarray,
    group: Mapping[str, Any],
    intercept_nodes: np.ndarray,
    slope_nodes: np.ndarray,
    log_weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    alpha, eta, rho_alpha, rho_eta = theta
    tau_alpha = float(softplus(float(rho_alpha))) + MIN_SCALE
    tau_eta = float(softplus(float(rho_eta))) + MIN_SCALE
    tau_alpha_derivative = float(expit(rho_alpha))
    tau_eta_derivative = float(expit(rho_eta))
    rows = group["rows"]
    offsets = np.asarray([row["log_dose_offset"] for row in rows], dtype=float)
    events = np.asarray([row["events"] for row in rows], dtype=float)
    totals = np.asarray([row["total"] for row in rows], dtype=float)
    powers = np.asarray([row["power"] for row in rows], dtype=float)

    latent_slopes = eta + tau_eta * slope_nodes
    slopes = np.asarray(softplus(latent_slopes), dtype=float)
    slope_derivatives = expit(latent_slopes)
    linear = (
        alpha
        + tau_alpha * intercept_nodes[:, None]
        + slopes[:, None] * offsets[None, :]
    )
    log_probability = -np.logaddexp(0.0, -linear)
    log_one_minus_probability = -np.logaddexp(0.0, linear)
    node_log_likelihood = np.sum(
        powers[None, :]
        * (
            events[None, :] * log_probability
            + (totals - events)[None, :] * log_one_minus_probability
        ),
        axis=1,
    )
    combined = log_weights + node_log_likelihood
    log_marginal = float(logsumexp(combined))
    posterior_weights = np.exp(combined - log_marginal)
    residual = powers[None, :] * (totals[None, :] * expit(linear) - events[None, :])
    conditional_negative_gradient = np.column_stack(
        [
            np.sum(residual, axis=1),
            np.sum(residual * offsets[None, :] * slope_derivatives[:, None], axis=1),
            np.sum(
                residual * intercept_nodes[:, None] * tau_alpha_derivative,
                axis=1,
            ),
            np.sum(
                residual
                * offsets[None, :]
                * slope_derivatives[:, None]
                * slope_nodes[:, None]
                * tau_eta_derivative,
                axis=1,
            ),
        ]
    )
    gradient = np.sum(
        posterior_weights[:, None] * conditional_negative_gradient, axis=0
    )
    return log_marginal, gradient, posterior_weights


def _objective_and_gradient(
    theta: np.ndarray,
    *,
    groups: Sequence[Mapping[str, Any]],
    intercept_nodes: np.ndarray,
    slope_nodes: np.ndarray,
    log_weights: np.ndarray,
    prior: BayesianPrior,
    settings: RandomInterceptSlopeSettings,
) -> tuple[float, np.ndarray]:
    alpha, eta, rho_alpha, rho_eta = theta
    value = 0.5 * (
        ((alpha - prior.alpha_mean) / prior.alpha_sd) ** 2
        + ((eta - prior.eta_mean) / prior.eta_sd) ** 2
        + (
            (rho_alpha - settings.intercept_scale_latent_mean)
            / settings.intercept_scale_latent_sd
        )
        ** 2
        + (
            (rho_eta - settings.slope_scale_latent_mean)
            / settings.slope_scale_latent_sd
        )
        ** 2
    )
    gradient = np.asarray(
        [
            (alpha - prior.alpha_mean) / prior.alpha_sd**2,
            (eta - prior.eta_mean) / prior.eta_sd**2,
            (rho_alpha - settings.intercept_scale_latent_mean)
            / settings.intercept_scale_latent_sd**2,
            (rho_eta - settings.slope_scale_latent_mean)
            / settings.slope_scale_latent_sd**2,
        ],
        dtype=float,
    )
    for group in groups:
        log_marginal, group_gradient, _ = _group_log_marginal_and_gradient(
            theta, group, intercept_nodes, slope_nodes, log_weights
        )
        value -= log_marginal
        gradient += group_gradient
    if not math.isfinite(float(value)) or not np.isfinite(gradient).all():
        return math.inf, np.full(4, math.nan)
    return float(value), gradient


def _numerical_hessian(theta: np.ndarray, gradient_function: Any) -> np.ndarray:
    columns: list[np.ndarray] = []
    for index in range(len(theta)):
        step = 1e-4 * max(1.0, abs(float(theta[index])))
        upper, lower = theta.copy(), theta.copy()
        upper[index] += step
        lower[index] -= step
        columns.append((gradient_function(upper) - gradient_function(lower)) / (2 * step))
    matrix = np.column_stack(columns)
    return (matrix + matrix.T) / 2.0


def _scale_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "posterior_mean": float(np.mean(values)),
        "posterior_median": float(np.median(values)),
        "posterior_credible_interval_95": {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        },
    }


def _group_dose_coverage(groups: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for group in groups:
        offsets = sorted({float(row["log_dose_offset"]) for row in group["rows"]})
        log_range = offsets[-1] - offsets[0] if len(offsets) > 1 else 0.0
        limited = len(offsets) < 2 or log_range < math.log(1.5)
        rows.append(
            {
                "group_id": group["group_id"],
                "source_row_count": len(group["rows"]),
                "unique_dose_count": len(offsets),
                "log_dose_range": log_range,
                "slope_information": "limited" if limited else "informative",
            }
        )
        if limited:
            warnings.append(
                f"Study group {group['group_id']} has limited within-group dose coverage for estimating a random slope."
            )
    return rows, warnings


def _failure(
    status: str,
    accepted: Sequence[Mapping[str, Any]],
    weight_diagnostics: Mapping[str, Any],
    skipped: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_assumption": "independent_study_random_intercept_and_latent_slope",
        "curve_points": [],
        "historical_lines": _contributor_audit(accepted),
        "weight_diagnostics": dict(weight_diagnostics),
        "skipped_reasons": dict(skipped),
        **extra,
    }


def estimate_power_likelihood_random_intercept_slope_curve(
    historical_lines: Sequence[Mapping[str, Any]],
    spec: ToxicityQuery,
    *,
    bayesian_prior: str,
    settings: RandomInterceptSlopeSettings | None = None,
) -> dict[str, Any]:
    """Fit the new three-method power-likelihood model."""

    resolved = (settings or RandomInterceptSlopeSettings()).validated()
    if not spec.planned_doses:
        raise ValueError("Random-intercept/slope fitting requires a target dose grid.")
    power_settings = PowerLikelihoodSettings(
        historical_ess_cap=resolved.historical_ess_cap,
        borrowing_strength=resolved.borrowing_strength,
        random_seed=resolved.random_seed,
    )
    accepted, skipped, weight_diagnostics = _prepare_powered_lines(
        historical_lines, power_settings
    )
    if not accepted:
        return _failure(
            "no_eligible_original_binomial_evidence",
            accepted,
            weight_diagnostics,
            skipped,
        )
    try:
        prior = _prior_from_lines([item["line"] for item in accepted], bayesian_prior)
    except ValueError as error:
        return _failure(
            "prior_contract_failed",
            accepted,
            weight_diagnostics,
            {str(error): 1, **skipped},
        )

    reference_log_dose = float(np.mean(np.log(np.asarray(spec.planned_doses))))
    reference_dose = float(math.exp(reference_log_dose))
    groups = _group_rows(accepted, reference_log_dose)
    intercept_nodes, slope_nodes, log_weights = _quadrature_grid(
        resolved.quadrature_nodes
    )
    arguments = {
        "groups": groups,
        "intercept_nodes": intercept_nodes,
        "slope_nodes": slope_nodes,
        "log_weights": log_weights,
        "prior": prior,
        "settings": resolved,
    }

    def objective(theta: np.ndarray) -> float:
        return _objective_and_gradient(theta, **arguments)[0]

    def gradient(theta: np.ndarray) -> np.ndarray:
        return _objective_and_gradient(theta, **arguments)[1]

    base = np.asarray(
        [
            prior.alpha_mean,
            prior.eta_mean,
            resolved.intercept_scale_latent_mean,
            resolved.slope_scale_latent_mean,
        ],
        dtype=float,
    )
    starts = [
        base,
        base + np.asarray([0.5, 0.0, 0.0, 0.0]),
        base + np.asarray([-0.5, 0.0, 0.0, 0.0]),
        base + np.asarray([0.0, 0.5, -0.5, 0.5]),
        base + np.asarray([0.0, -0.5, 0.5, -0.5]),
    ]
    fits = [
        minimize(
            objective,
            start,
            jac=gradient,
            method="BFGS",
            options={"gtol": 1e-7, "maxiter": 1_500},
        )
        for start in starts
    ]
    diagnostic_rows: list[dict[str, Any]] = []
    acceptable = []
    for index, fit in enumerate(fits):
        finite = bool(np.isfinite(fit.x).all() and math.isfinite(float(fit.fun)))
        gradient_norm = (
            float(np.max(np.abs(gradient(fit.x)))) if finite else math.inf
        )
        diagnostic_rows.append(
            {
                "initialization": index + 1,
                "optimizer_success": bool(fit.success),
                "objective": float(fit.fun) if finite else None,
                "gradient_infinity_norm": gradient_norm,
                "message": str(fit.message),
            }
        )
        if finite and gradient_norm <= 5e-4:
            acceptable.append(fit)
    if not acceptable:
        return _failure(
            "map_optimization_failed",
            accepted,
            weight_diagnostics,
            skipped,
            fit_diagnostics={
                "optimizer_runs": diagnostic_rows,
                "gradient_failure_threshold": 5e-4,
            },
        )
    fit = min(acceptable, key=lambda item: float(item.fun))
    gradient_at_mode = gradient(fit.x)
    gradient_norm = float(np.max(np.abs(gradient_at_mode)))
    hessian = _numerical_hessian(fit.x, gradient)
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
        if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 1e-9:
            raise np.linalg.LinAlgError("Hessian is not positive definite.")
        covariance = np.linalg.inv(hessian)
        covariance = (covariance + covariance.T) / 2.0
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        return _failure(
            "laplace_hessian_failed",
            accepted,
            weight_diagnostics,
            skipped,
            fit_diagnostics={
                "optimizer_runs": diagnostic_rows,
                "selected_gradient_infinity_norm": gradient_norm,
                "hessian_error": str(error),
            },
        )

    existing_draw_counts = [
        int(item["line"]["bayesian_fit"]["posterior"]["draw_count"])
        for item in accepted
        if isinstance(item["line"].get("bayesian_fit"), Mapping)
        and isinstance(item["line"]["bayesian_fit"].get("posterior"), Mapping)
        and isinstance(item["line"]["bayesian_fit"]["posterior"].get("draw_count"), int)
    ]
    draw_count = max(32, min(existing_draw_counts) if existing_draw_counts else 1_024)
    seed_payload = {
        "model_version": POSTERIOR_SEED_VERSION,
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
    tau_alpha_draws = np.asarray(softplus(parameter_draws[:, 2])) + MIN_SCALE
    tau_eta_draws = np.asarray(softplus(parameter_draws[:, 3])) + MIN_SCALE
    target_standard_intercepts = rng.standard_normal(draw_count)
    target_standard_slopes = rng.standard_normal(draw_count)
    target_intercepts = tau_alpha_draws * target_standard_intercepts
    target_slope_shifts = tau_eta_draws * target_standard_slopes
    doses = np.asarray(spec.planned_doses, dtype=float)
    offsets = np.log(doses) - reference_log_dose
    target_slopes = np.asarray(softplus(parameter_draws[:, 1] + target_slope_shifts))
    target_draws = expit(
        parameter_draws[:, 0, None]
        + target_intercepts[:, None]
        + target_slopes[:, None] * offsets[None, :]
    )
    population_draws = expit(
        parameter_draws[:, 0, None]
        + np.asarray(softplus(parameter_draws[:, 1]))[:, None] * offsets[None, :]
    )
    if (
        not np.isfinite(target_draws).all()
        or np.any((target_draws < 0) | (target_draws > 1))
        or np.any(np.diff(target_draws, axis=1) < -1e-12)
    ):
        return _failure(
            "posterior_curve_validation_failed",
            accepted,
            weight_diagnostics,
            skipped,
        )

    curve_points: list[dict[str, Any]] = []
    population_points: list[dict[str, Any]] = []
    powered_weight_sum = sum(float(item["likelihood_power"]) for item in accepted)
    for index, dose in enumerate(doses):
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
        values = target_draws[:, index]
        curve_points.append(
            {
                **common,
                "prediction_target": "new_target_study_including_intercept_and_slope_variation",
                "dlt_probability": float(np.mean(values)),
                "posterior_credible_interval_95": {
                    "low": float(np.quantile(values, 0.025)),
                    "high": float(np.quantile(values, 0.975)),
                },
            }
        )
        population_values = population_draws[:, index]
        population_points.append(
            {
                **common,
                "prediction_target": "population_mean_study_effects_zero",
                "dlt_probability": float(np.mean(population_values)),
                "posterior_credible_interval_95": {
                    "low": float(np.quantile(population_values, 0.025)),
                    "high": float(np.quantile(population_values, 0.975)),
                },
            }
        )

    coverage, warnings = _group_dose_coverage(groups)
    if len(groups) < 2:
        warnings.append(
            "Only one study group is available; both random-effect scales are strongly prior driven."
        )
    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    if condition_number > 1e10:
        warnings.append(
            "The Laplace Hessian is ill-conditioned; interpret heterogeneity estimates cautiously."
        )
    objectives = [float(item.fun) for item in acceptable]
    objective_range = max(objectives) - min(objectives)
    if objective_range > 0.05:
        warnings.append(
            "Multiple optimizer starts reached materially different objective values."
        )
    document_ids = {
        str(item["line"].get("document_id") or item["line"].get("pmcid"))
        for item in accepted
    }
    return {
        "status": "fitted",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_assumption": "independent_study_random_intercept_and_latent_slope",
        "method_label": "Power likelihood with study random intercepts and slopes",
        "bayesian_prior": bayesian_prior,
        "prior": prior.to_payload(),
        "random_effect_prior": {
            "parameterization": "marginal_normal_effects",
            "independent_effects": True,
            "intercept_scale": {
                "transform": "tau_alpha=softplus(rho_alpha)",
                "rho_mean": resolved.intercept_scale_latent_mean,
                "rho_sd": resolved.intercept_scale_latent_sd,
            },
            "latent_slope_scale": {
                "transform": "tau_eta=softplus(rho_eta)",
                "rho_mean": resolved.slope_scale_latent_mean,
                "rho_sd": resolved.slope_scale_latent_sd,
            },
        },
        "posterior_approximation": POSTERIOR_APPROXIMATION,
        "marginal_likelihood_integration": {
            "method": "two_dimensional_gauss_hermite_quadrature",
            "nodes_per_dimension": resolved.quadrature_nodes,
            "node_pair_count": resolved.quadrature_nodes**2,
            "likelihood_power_location": "conditional_binomial_likelihood_only",
        },
        "dose_standardization": {
            "type": "natural_log_ratio_to_target_dose_geometric_mean",
            "reference_dose": reference_dose,
            "reference_log_dose": reference_log_dose,
            "log_base": "e",
        },
        "posterior": {
            "mode": {
                "alpha": float(fit.x[0]),
                "eta": float(fit.x[1]),
                "rho_alpha": float(fit.x[2]),
                "rho_eta": float(fit.x[3]),
                "tau_alpha": float(softplus(float(fit.x[2]))),
                "tau_eta": float(softplus(float(fit.x[3]))),
            },
            "covariance_alpha_eta_rho_alpha_rho_eta": covariance.tolist(),
            "draw_count": int(draw_count),
            "draw_seed": int(derived_seed),
            "draws_alpha_eta_rho_alpha_rho_eta": parameter_draws.tolist(),
            "target_standard_normal_intercept_draws": target_standard_intercepts.tolist(),
            "target_standard_normal_slope_draws": target_standard_slopes.tolist(),
            "target_random_intercept_draws": target_intercepts.tolist(),
            "target_random_slope_latent_draws": target_slope_shifts.tolist(),
        },
        "curve_points": curve_points,
        "population_mean_curve_points": population_points,
        "historical_lines": _contributor_audit(accepted),
        "weight_diagnostics": {
            **weight_diagnostics,
            "contributor_line_count": len(accepted),
            "contributor_document_count": len(document_ids),
            "independent_study_group_count": len(groups),
            "random_effect_group_definition": "document_id_plus_study_unit_id",
        },
        "heterogeneity_diagnostics": {
            "random_intercept_scale": _scale_summary(tau_alpha_draws),
            "random_slope_latent_scale": _scale_summary(tau_eta_draws),
            "study_group_dose_coverage": coverage,
        },
        "new_target_study_random_intercept": True,
        "new_target_study_random_slope": True,
        "target_curve_effect_sharing": (
            "One random intercept and one latent-slope effect are shared across all doses in each complete target curve."
        ),
        "fit_warnings": warnings,
        "fit_diagnostics": {
            "optimizer_runs": diagnostic_rows,
            "selected_gradient_infinity_norm": gradient_norm,
            "gradient_failure_threshold": 5e-4,
            "hessian_min_eigenvalue": float(np.min(eigenvalues)),
            "hessian_condition_number": condition_number,
            "multi_start_objective_range": objective_range,
            "independent_sampling_validation": "not_performed_in_runtime",
        },
        "skipped_reasons": skipped,
    }


def random_intercept_slope_draws_at_doses(
    method: Mapping[str, Any], doses: Sequence[float]
) -> np.ndarray:
    ensure_current_prediction(method)
    if method.get("status") != "fitted" or method.get("model") != MODEL_NAME:
        raise ValueError("A fitted random-intercept/slope curve is required.")
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
        raise ValueError("Random-intercept/slope method lacks posterior metadata.")
    reference_log = _finite_number(
        standardization.get("reference_log_dose"), "reference_log_dose"
    )
    parameter_draws = np.asarray(
        posterior.get("draws_alpha_eta_rho_alpha_rho_eta"), dtype=float
    )
    target_intercepts = np.asarray(
        posterior.get("target_random_intercept_draws"), dtype=float
    )
    target_slope_shifts = np.asarray(
        posterior.get("target_random_slope_latent_draws"), dtype=float
    )
    if (
        parameter_draws.ndim != 2
        or parameter_draws.shape[1] != 4
        or not len(parameter_draws)
        or not np.isfinite(parameter_draws).all()
        or target_intercepts.shape != (len(parameter_draws),)
        or target_slope_shifts.shape != (len(parameter_draws),)
        or not np.isfinite(target_intercepts).all()
        or not np.isfinite(target_slope_shifts).all()
    ):
        raise ValueError("Random-intercept/slope posterior draws are invalid.")
    offsets = np.log(resolved_doses) - reference_log
    raw = expit(
        parameter_draws[:, 0, None]
        + target_intercepts[:, None]
        + np.asarray(softplus(parameter_draws[:, 1] + target_slope_shifts))[:, None]
        * offsets[None, :]
    )
    return raw


def random_intercept_slope_probability_summary(
    method: Mapping[str, Any], dose: float
) -> dict[str, Any]:
    draws = random_intercept_slope_draws_at_doses(method, [dose])[:, 0]
    heterogeneity = method.get("heterogeneity_diagnostics", {})
    return {
        "model": method.get("model"),
        "method": "similarity_weighted_power_likelihood_random_intercept_slope",
        "prediction_target": "new_target_study_including_intercept_and_slope_variation",
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
        "between_study_intercept_scale": heterogeneity.get(
            "random_intercept_scale", {}
        ).get("posterior_mean"),
        "between_study_slope_latent_scale": heterogeneity.get(
            "random_slope_latent_scale", {}
        ).get("posterior_mean"),
    }
