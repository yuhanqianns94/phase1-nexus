"""Similarity-weighted power-likelihood target toxicity curves.

This module deliberately coexists with ``similarity_weighted_curve``.  The
existing estimator fits every historical line separately and averages its
posterior probabilities.  This estimator instead fits one shared monotone
Binomial-logistic curve from the original dose/DLT counts:

    posterior(theta) proportional to prior(theta) * product_i L_i(theta)^a_i

The powers are fixed, outcome-free borrowing controls.  They are formed from
the already-audited similarity/reliability weights and are scaled down, never
up, to respect an explicit historical effective-sample-size cap.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .bayesian_toxicity import BayesianPrior, softplus
from .prediction_contract import ensure_current_prediction
from .toxicity_curve_common import ToxicityQuery


MODEL_NAME = "similarity_weighted_power_likelihood_monotone_logistic_laplace_v1"
POSTERIOR_APPROXIMATION = "laplace_normal_in_shared_alpha_eta_space"
DEFAULT_HISTORICAL_ESS_CAP = 30.0
DEFAULT_BORROWING_STRENGTH = 1.0


@dataclass(frozen=True)
class PowerLikelihoodSettings:
    """Validated controls for fixed, outcome-free historical borrowing."""

    historical_ess_cap: float = DEFAULT_HISTORICAL_ESS_CAP
    borrowing_strength: float = DEFAULT_BORROWING_STRENGTH
    random_seed: int = 20260822

    def validated(self) -> "PowerLikelihoodSettings":
        cap = _finite_number(self.historical_ess_cap, "historical_ess_cap")
        strength = _finite_number(self.borrowing_strength, "borrowing_strength")
        if cap <= 0:
            raise ValueError("historical_ess_cap must be > 0.")
        if not 0 < strength <= 1:
            raise ValueError("borrowing_strength must be in (0, 1].")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        return PowerLikelihoodSettings(cap, strength, self.random_seed)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _positive_integer(value: Any, label: str) -> int:
    numeric = _finite_number(value, label)
    if not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(numeric)


def _nonnegative_integer(value: Any, label: str) -> int:
    numeric = _finite_number(value, label)
    if not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return int(numeric)


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _prior_from_lines(
    lines: Sequence[Mapping[str, Any]], bayesian_prior: str
) -> BayesianPrior:
    resolved: BayesianPrior | None = None
    signature: tuple[float, float, float, float] | None = None
    for line in lines:
        fit = line.get("bayesian_fit")
        payload = fit.get("prior") if isinstance(fit, Mapping) else None
        if not isinstance(payload, Mapping):
            continue
        alpha = payload.get("alpha")
        eta = payload.get("eta")
        if not isinstance(alpha, Mapping) or not isinstance(eta, Mapping):
            continue
        candidate_signature = (
            _finite_number(alpha.get("mean"), "prior.alpha.mean"),
            _finite_number(alpha.get("sd"), "prior.alpha.sd"),
            _finite_number(eta.get("mean"), "prior.eta.mean"),
            _finite_number(eta.get("sd"), "prior.eta.sd"),
        )
        if candidate_signature[1] <= 0 or candidate_signature[3] <= 0:
            raise ValueError("Power-likelihood prior standard deviations must be > 0.")
        if signature is not None and candidate_signature != signature:
            raise ValueError("Historical lines contain inconsistent Bayesian priors.")
        signature = candidate_signature
        resolved = BayesianPrior(
            name=bayesian_prior,
            description=str(payload.get("description") or ""),
            alpha_mean=candidate_signature[0],
            alpha_sd=candidate_signature[1],
            eta_mean=candidate_signature[2],
            eta_sd=candidate_signature[3],
        )
    if resolved is None:
        raise ValueError("Historical lines do not expose a compatible prior payload.")
    return resolved


def _validated_source_rows(
    line: Mapping[str, Any], line_index: int
) -> list[dict[str, Any]]:
    source_points = line.get("source_points")
    if not isinstance(source_points, list) or not source_points:
        raise ValueError("missing_original_binomial_source_points")
    rows: list[dict[str, Any]] = []
    for point_index, point in enumerate(source_points):
        if not isinstance(point, Mapping):
            raise ValueError("invalid_original_binomial_source_point")
        dose = _finite_number(
            point.get("canonical_dose"),
            f"lines[{line_index}].source_points[{point_index}].canonical_dose",
        )
        if dose <= 0:
            raise ValueError("source_point_dose_must_be_positive")
        events = _nonnegative_integer(
            point.get("n_patients_with_dlt"),
            f"lines[{line_index}].source_points[{point_index}].n_patients_with_dlt",
        )
        total = _positive_integer(
            point.get("n_patients"),
            f"lines[{line_index}].source_points[{point_index}].n_patients",
        )
        if events > total:
            raise ValueError("source_point_dlt_count_exceeds_denominator")
        denominator_role = point.get("denominator_role")
        if denominator_role not in {None, "dlt_evaluable", "treated"}:
            raise ValueError("unsupported_dlt_denominator_role")
        rows.append(
            {
                "canonical_dose": dose,
                "n_patients_with_dlt": events,
                "n_patients": total,
                "denominator_role": denominator_role or "legacy_unspecified",
                "cohort_id": point.get("cohort_id"),
            }
        )
    return rows


def _prepare_powered_lines(
    lines: Sequence[Mapping[str, Any]],
    settings: PowerLikelihoodSettings,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for index, line in enumerate(lines):
        try:
            similarity_weight = _finite_number(
                line.get("segment_similarity_weight"),
                f"lines[{index}].segment_similarity_weight",
            )
            reliability_weight = _finite_number(
                line.get("reliability_weight"),
                f"lines[{index}].reliability_weight",
            )
            raw_power = (
                settings.borrowing_strength * similarity_weight * reliability_weight
            )
            if raw_power <= 0:
                raise ValueError("non_positive_similarity_reliability_weight")
            if raw_power > 1 + 1e-12:
                raise ValueError("likelihood_power_exceeds_one")
            rows = _validated_source_rows(line, index)
        except ValueError as error:
            reason = str(error)
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        patient_count = sum(int(row["n_patients"]) for row in rows)
        accepted.append(
            {
                "line": line,
                "rows": rows,
                "raw_power": float(raw_power),
                "patient_count": patient_count,
            }
        )

    pre_cap_ess = sum(item["raw_power"] * item["patient_count"] for item in accepted)
    cap_scale = (
        min(1.0, settings.historical_ess_cap / pre_cap_ess) if pre_cap_ess > 0 else 0.0
    )
    for item in accepted:
        item["likelihood_power"] = float(item["raw_power"] * cap_scale)
        item["effective_patient_count"] = float(
            item["likelihood_power"] * item["patient_count"]
        )

    effective_ess = sum(item["effective_patient_count"] for item in accepted)
    diagnostics = {
        "weight_policy": "similarity_times_reliability_scaled_down_to_historical_ess_cap",
        "borrowing_strength": settings.borrowing_strength,
        "historical_ess_cap": settings.historical_ess_cap,
        "pre_cap_effective_sample_size": float(pre_cap_ess),
        "ess_cap_scale_factor": float(cap_scale),
        "effective_historical_sample_size": float(effective_ess),
        "power_is_information_fraction_not_probability_of_correctness": True,
        "powers_are_fixed_before_target_outcomes": True,
    }
    return accepted, dict(sorted(skipped.items())), diagnostics


def _negative_log_posterior(
    theta: np.ndarray,
    *,
    log_dose_offsets: np.ndarray,
    effective_events: np.ndarray,
    effective_totals: np.ndarray,
    prior: BayesianPrior,
) -> float:
    alpha, eta = theta
    slope = float(softplus(float(eta)))
    linear_predictor = alpha + slope * log_dose_offsets
    log_probability = -np.logaddexp(0.0, -linear_predictor)
    log_one_minus_probability = -np.logaddexp(0.0, linear_predictor)
    log_likelihood = float(
        np.sum(
            effective_events * log_probability
            + (effective_totals - effective_events) * log_one_minus_probability
        )
    )
    log_prior = -0.5 * (
        ((alpha - prior.alpha_mean) / prior.alpha_sd) ** 2
        + ((eta - prior.eta_mean) / prior.eta_sd) ** 2
    )
    result = -(log_likelihood + log_prior)
    return result if math.isfinite(result) else math.inf


def _gradient(
    theta: np.ndarray,
    *,
    log_dose_offsets: np.ndarray,
    effective_events: np.ndarray,
    effective_totals: np.ndarray,
    prior: BayesianPrior,
) -> np.ndarray:
    alpha, eta = theta
    eta_derivative = float(expit(eta))
    linear_predictor = alpha + float(softplus(float(eta))) * log_dose_offsets
    probabilities = expit(linear_predictor)
    residual = effective_totals * probabilities - effective_events
    return np.asarray(
        [
            np.sum(residual) + (alpha - prior.alpha_mean) / prior.alpha_sd**2,
            np.sum(residual * log_dose_offsets * eta_derivative)
            + (eta - prior.eta_mean) / prior.eta_sd**2,
        ],
        dtype=float,
    )


def _hessian(
    theta: np.ndarray,
    *,
    log_dose_offsets: np.ndarray,
    effective_events: np.ndarray,
    effective_totals: np.ndarray,
    prior: BayesianPrior,
) -> np.ndarray:
    alpha, eta = theta
    eta_derivative = float(expit(eta))
    eta_second = eta_derivative * (1.0 - eta_derivative)
    linear_predictor = alpha + float(softplus(float(eta))) * log_dose_offsets
    probabilities = expit(linear_predictor)
    curvature = effective_totals * probabilities * (1.0 - probabilities)
    residual = effective_totals * probabilities - effective_events
    d_eta = log_dose_offsets * eta_derivative
    d2_eta = log_dose_offsets * eta_second
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


def _binomial_deviance(events: int, total: int, probability: float) -> float:
    probability = min(max(float(probability), 1e-12), 1 - 1e-12)
    result = 0.0
    if events:
        result += events * math.log(events / (total * probability))
    failures = total - events
    if failures:
        result += failures * math.log(failures / (total * (1 - probability)))
    return 2.0 * result


def _heterogeneity_diagnostics(
    accepted: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    eta: float,
    reference_dose: float,
) -> dict[str, Any]:
    line_rows: list[dict[str, Any]] = []
    weighted_deviance = 0.0
    maximum_residual = 0.0
    for item in accepted:
        line = item["line"]
        expected = 0.0
        observed = 0
        variance = 0.0
        deviance = 0.0
        for row in item["rows"]:
            probability = float(
                expit(
                    alpha
                    + float(softplus(eta))
                    * math.log(float(row["canonical_dose"]) / reference_dose)
                )
            )
            total = int(row["n_patients"])
            events = int(row["n_patients_with_dlt"])
            observed += events
            expected += total * probability
            variance += total * probability * (1 - probability)
            deviance += _binomial_deviance(events, total, probability)
        standardized_residual = (
            (observed - expected) / math.sqrt(variance) if variance > 0 else 0.0
        )
        maximum_residual = max(maximum_residual, abs(standardized_residual))
        weighted_deviance += float(item["likelihood_power"]) * deviance
        line_rows.append(
            {
                "document_id": line.get("document_id") or line.get("pmcid"),
                "study_unit_id": line.get("study_unit_id"),
                "segment_index": line.get("segment_index"),
                "observed_dlt_count": observed,
                "expected_dlt_count_at_shared_mode": float(expected),
                "standardized_residual": float(standardized_residual),
                "binomial_deviance": float(deviance),
                "likelihood_power": float(item["likelihood_power"]),
            }
        )
    return {
        "interpretation": (
            "Descriptive conflict diagnostic only; it is not a calibrated "
            "heterogeneity test or a random-effects model."
        ),
        "weighted_binomial_deviance": float(weighted_deviance),
        "maximum_absolute_standardized_line_residual": float(maximum_residual),
        "large_residual_warning": bool(maximum_residual > 2.0),
        "shared_curve_assumption": (
            "All powered historical lines share one alpha and one monotone slope."
        ),
        "line_diagnostics": line_rows,
    }


def estimate_power_likelihood_curve(
    historical_lines: Sequence[Mapping[str, Any]],
    spec: ToxicityQuery,
    *,
    bayesian_prior: str,
    settings: PowerLikelihoodSettings | None = None,
) -> dict[str, Any]:
    """Fit one shared curve from fixed powered historical Binomial likelihoods."""

    resolved_settings = (settings or PowerLikelihoodSettings()).validated()
    if not spec.planned_doses:
        raise ValueError("Power-likelihood fitting requires a target dose grid.")
    accepted, skipped, weight_diagnostics = _prepare_powered_lines(
        historical_lines, resolved_settings
    )
    if not accepted:
        return {
            "status": "no_eligible_original_binomial_evidence",
            "model": MODEL_NAME,
            "model_assumption": "shared_fixed_effect_curve",
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
            "model_assumption": "shared_fixed_effect_curve",
            "curve_points": [],
            "historical_lines": [],
            "weight_diagnostics": weight_diagnostics,
            "skipped_reasons": {str(error): 1, **skipped},
        }

    reference_log_dose = float(np.mean(np.log(np.asarray(spec.planned_doses))))
    reference_dose = float(math.exp(reference_log_dose))
    log_offsets: list[float] = []
    effective_events: list[float] = []
    effective_totals: list[float] = []
    for item in accepted:
        power = float(item["likelihood_power"])
        for row in item["rows"]:
            log_offsets.append(
                math.log(float(row["canonical_dose"])) - reference_log_dose
            )
            effective_events.append(power * int(row["n_patients_with_dlt"]))
            effective_totals.append(power * int(row["n_patients"]))
    offsets = np.asarray(log_offsets, dtype=float)
    powered_events = np.asarray(effective_events, dtype=float)
    powered_totals = np.asarray(effective_totals, dtype=float)
    objective_arguments = {
        "log_dose_offsets": offsets,
        "effective_events": powered_events,
        "effective_totals": powered_totals,
        "prior": prior,
    }
    initial = np.asarray([prior.alpha_mean, prior.eta_mean], dtype=float)
    result = minimize(
        lambda theta: _negative_log_posterior(theta, **objective_arguments),
        initial,
        jac=lambda theta: _gradient(theta, **objective_arguments),
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1_000},
    )
    gradient = _gradient(result.x, **objective_arguments)
    gradient_norm = float(np.linalg.norm(gradient))
    if (
        not np.isfinite(result.x).all()
        or not math.isfinite(float(result.fun))
        or gradient_norm > 1e-4
    ):
        return {
            "status": "map_optimization_failed",
            "model": MODEL_NAME,
            "model_assumption": "shared_fixed_effect_curve",
            "curve_points": [],
            "historical_lines": [],
            "weight_diagnostics": weight_diagnostics,
            "skipped_reasons": skipped,
            "fit_diagnostics": {
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "gradient_norm": gradient_norm,
            },
        }

    hessian = _hessian(result.x, **objective_arguments)
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
        if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 1e-10:
            raise np.linalg.LinAlgError("Hessian is not positive definite.")
        covariance = np.linalg.inv(hessian)
        covariance = (covariance + covariance.T) / 2.0
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        return {
            "status": "laplace_hessian_failed",
            "model": MODEL_NAME,
            "model_assumption": "shared_fixed_effect_curve",
            "curve_points": [],
            "historical_lines": [],
            "weight_diagnostics": weight_diagnostics,
            "skipped_reasons": skipped,
            "fit_diagnostics": {
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
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
    draw_count = min(existing_draw_counts) if existing_draw_counts else 1_024
    draw_count = max(32, draw_count)
    seed_material = {
        "settings": resolved_settings.__dict__,
        "prior": prior.to_payload(),
        "reference_dose": reference_dose,
        "contributors": [
            {
                "document_id": item["line"].get("document_id")
                or item["line"].get("pmcid"),
                "segment_index": item["line"].get("segment_index"),
                "likelihood_power": item["likelihood_power"],
                "rows": item["rows"],
            }
            for item in accepted
        ],
    }
    derived_seed = int.from_bytes(
        hashlib.sha256(
            f"{resolved_settings.random_seed}|{_canonical_hash(seed_material)}".encode(
                "utf-8"
            )
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )
    rng = np.random.default_rng(derived_seed)
    parameter_draws = rng.multivariate_normal(result.x, covariance, size=draw_count)
    doses = np.asarray(spec.planned_doses, dtype=float)
    target_offsets = np.log(doses) - reference_log_dose
    raw_probability_draws = expit(
        parameter_draws[:, 0, None]
        + softplus(parameter_draws[:, 1, None]) * target_offsets[None, :]
    )

    powered_weight_sum = sum(float(item["likelihood_power"]) for item in accepted)
    curve_points: list[dict[str, Any]] = []
    for index, dose in enumerate(doses):
        draws = raw_probability_draws[:, index]
        extrapolation_power = sum(
            float(item["likelihood_power"])
            for item in accepted
            if float(dose) < float(item["line"].get("observed_dose_min", dose))
            or float(dose) > float(item["line"].get("observed_dose_max", dose))
        )
        curve_points.append(
            {
                "dose": float(dose),
                "unit": spec.target_unit,
                "dlt_probability": float(np.mean(draws)),
                "posterior_credible_interval_95": {
                    "low": float(np.quantile(draws, 0.025)),
                    "high": float(np.quantile(draws, 0.975)),
                },
                "contributor_count": len(accepted),
                "effective_historical_sample_size": weight_diagnostics[
                    "effective_historical_sample_size"
                ],
                "extrapolation_power_fraction": (
                    float(extrapolation_power / powered_weight_sum)
                    if powered_weight_sum > 0
                    else 0.0
                ),
            }
        )

    contributor_audit = []
    for item in accepted:
        line = item["line"]
        contributor_audit.append(
            {
                "document_id": line.get("document_id") or line.get("pmcid"),
                "pmcid": line.get("pmcid") or line.get("document_id"),
                "study_unit_id": line.get("study_unit_id"),
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
                "evidence_support_max": line.get("evidence_support_max"),
            }
        )

    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    heterogeneity = _heterogeneity_diagnostics(
        accepted,
        alpha=float(result.x[0]),
        eta=float(result.x[1]),
        reference_dose=reference_dose,
    )
    document_ids = {
        str(item["line"].get("document_id") or item["line"].get("pmcid"))
        for item in accepted
    }
    return {
        "status": "fitted",
        "model": MODEL_NAME,
        "model_assumption": "shared_fixed_effect_curve",
        "method_label": "Similarity-weighted power likelihood (experimental)",
        "bayesian_prior": bayesian_prior,
        "prior": prior.to_payload(),
        "posterior_approximation": POSTERIOR_APPROXIMATION,
        "dose_standardization": {
            "type": "natural_log_ratio_to_target_dose_geometric_mean",
            "reference_dose": reference_dose,
            "reference_log_dose": reference_log_dose,
            "reference_definition": "geometric_mean_of_target_planned_or_display_doses",
            "log_base": "e",
        },
        "posterior": {
            "mode": {
                "alpha": float(result.x[0]),
                "eta": float(result.x[1]),
                "monotone_slope": float(softplus(float(result.x[1]))),
            },
            "covariance_alpha_eta": covariance.tolist(),
            "draw_count": int(draw_count),
            "draw_seed": int(derived_seed),
            "draws_alpha_eta": parameter_draws.tolist(),
        },
        "curve_points": curve_points,
        "historical_lines": contributor_audit,
        "weight_diagnostics": {
            **weight_diagnostics,
            "contributor_line_count": len(accepted),
            "contributor_document_count": len(document_ids),
            "deduplication_unit_currently_available": "document_id",
            "duplicate_publications_of_one_trial_require_external_trial_id_review": True,
        },
        "heterogeneity_diagnostics": heterogeneity,
        "fit_diagnostics": {
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "gradient_norm": gradient_norm,
            "hessian_condition_number": condition_number,
        },
        "skipped_reasons": skipped,
        "interpretation": (
            "All accepted historical Binomial likelihoods update one shared curve. "
            "Likelihood powers are information discounts, not probabilities that a "
            "publication is correct. This fixed-effect candidate does not itself "
            "model between-study random effects."
        ),
    }


def power_likelihood_draws_at_doses(
    method: Mapping[str, Any], doses: Sequence[float]
) -> np.ndarray:
    """Evaluate one stored power-likelihood posterior at positive target doses."""

    ensure_current_prediction(method)
    if method.get("status") != "fitted" or method.get("model") != MODEL_NAME:
        raise ValueError("A fitted power-likelihood curve is required.")
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
        raise ValueError("Power-likelihood method lacks posterior metadata.")
    reference_log_dose = _finite_number(
        standardization.get("reference_log_dose"), "reference_log_dose"
    )
    parameter_draws = np.asarray(posterior.get("draws_alpha_eta"), dtype=float)
    if (
        parameter_draws.ndim != 2
        or parameter_draws.shape[1] != 2
        or not len(parameter_draws)
        or not np.isfinite(parameter_draws).all()
    ):
        raise ValueError("Power-likelihood posterior draws are invalid.")
    offsets = np.log(resolved_doses) - reference_log_dose
    raw = expit(
        parameter_draws[:, 0, None]
        + softplus(parameter_draws[:, 1, None]) * offsets[None, :]
    )
    return raw


def power_likelihood_probability_summary(
    method: Mapping[str, Any], dose: float
) -> dict[str, Any]:
    draws = power_likelihood_draws_at_doses(method, [dose])[:, 0]
    return {
        "model": method.get("model"),
        "method": "similarity_weighted_power_likelihood",
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
    }
