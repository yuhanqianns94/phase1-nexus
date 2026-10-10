"""Similarity-weighted mixture of historical article-level toxicity lines."""

from __future__ import annotations

from typing import Any

import numpy as np

from .bayesian_toxicity import (
    fit_mle_binomial_logistic,
    mle_probability,
    posterior_probability_draws,
)
from .dlt_toxicity_regression import (
    DEFAULT_SINGLE_OBSERVATION_RELIABILITY_WEIGHT,
    build_similarity_historical_lines,
)
from .toxicity_curve_common import ToxicityQuery
from .toxicity_similarity import effective_sample_size


def _weighted_quantile(
    values: list[float], weights: list[float], quantile: float
) -> float | None:
    if not values or not weights or sum(weights) <= 0:
        return None
    pairs = sorted(zip(values, weights, strict=True), key=lambda pair: pair[0])
    cutoff = quantile * sum(weight for _value, weight in pairs)
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= cutoff:
            return value
    return pairs[-1][0]


def historical_line_support_max(line: dict[str, Any]) -> float | None:
    """Return the v2 evidence boundary or the legacy MAD cap.

    New evidence-first records never carry MAD. Their strict support boundary
    is the highest actual cohort with explicit DLT-evaluable evidence. Legacy
    v1 records retain their existing MAD behavior during the side-by-side
    migration.
    """

    for field_name in ("evidence_support_max", "mad_cap"):
        value = line.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _historical_support_audit_fields(
    line: dict[str, Any],
) -> dict[str, Any]:
    evidence_support = line.get("evidence_support_max")
    if evidence_support is not None:
        return {"evidence_support_max": evidence_support}
    return {"mad_cap": line.get("mad_cap")}


def predict_article_line_weighted_at_doses(
    lines: list[dict[str, Any]],
    doses: list[float] | tuple[float, ...],
) -> dict[float, float | None]:
    """Predict at selected doses without constructing contributor diagnostics.

    This cached-line path applies the same pointwise raw-similarity/reliability
    mixture as ``estimate_article_line_weighted``.
    """

    predictions: dict[float, float | None] = {}
    for dose in doses:
        weighted_draws: list[tuple[float, np.ndarray]] = []
        total_weight = 0.0
        for line in lines:
            support_max = historical_line_support_max(line)
            if support_max is not None and dose > support_max:
                continue
            weight = (
                line["segment_similarity_weight"] * line["reliability_weight"]
            )
            if weight <= 0:
                continue
            posterior_draws = posterior_probability_draws(
                line["bayesian_fit"], dose
            )
            weighted_draws.append((weight, posterior_draws))
            total_weight += weight
        if total_weight <= 0:
            predictions[dose] = None
            continue
        draw_count = min(len(draws) for _weight, draws in weighted_draws)
        mixture_draws = np.zeros(draw_count, dtype=float)
        for weight, posterior_draws in weighted_draws:
            mixture_draws += (weight / total_weight) * posterior_draws[:draw_count]
        predictions[dose] = float(np.mean(mixture_draws))
    return predictions


def posterior_mixture_draws_at_doses(
    lines: list[dict[str, Any]],
    doses: list[float] | tuple[float, ...],
    *,
    contributor_policy: str = "fixed_intersection",
) -> dict[str, Any]:
    """Build coherent target-level posterior draws across several doses.

    Rows are posterior draws and columns are requested dose levels.  Within a
    historical line, one stored ``(alpha, eta)`` draw is therefore used for all
    dose columns.  ``fixed_intersection`` (the OC default) retains only lines
    supported at every requested dose and keeps their normalized weights fixed;
    this avoids silently changing the data-generating ensemble between doses.

    ``pointwise`` reproduces the prediction display policy, including MAD-cap
    removal and re-normalization at each dose.  It is useful for diagnostics but
    can produce a non-monotone target mixture and should not be used as an OC
    truth without an explicit monotonicity step.
    """

    resolved_doses = np.asarray(doses, dtype=float)
    if (
        resolved_doses.ndim != 1
        or not len(resolved_doses)
        or not np.isfinite(resolved_doses).all()
        or np.any(resolved_doses <= 0)
    ):
        raise ValueError("doses must be a non-empty vector of positive finite values.")
    if np.any(np.diff(resolved_doses) <= 0):
        raise ValueError("doses must be strictly increasing for joint curve draws.")
    if contributor_policy not in {"fixed_intersection", "pointwise"}:
        raise ValueError(
            "contributor_policy must be 'fixed_intersection' or 'pointwise'."
        )

    candidate_lines: list[tuple[dict[str, Any], float]] = []
    skipped: dict[str, int] = {}
    fitted_positive_weight_before_policy = 0.0
    excluded_fixed_lines: list[dict[str, Any]] = []
    maximum_dose = float(np.max(resolved_doses))
    for line in lines:
        weight = float(line.get("segment_similarity_weight", 0.0)) * float(
            line.get("reliability_weight", 0.0)
        )
        if not np.isfinite(weight) or weight <= 0:
            skipped["non_positive_similarity_reliability_weight"] = (
                skipped.get("non_positive_similarity_reliability_weight", 0) + 1
            )
            continue
        fit = line.get("bayesian_fit")
        if not isinstance(fit, dict) or fit.get("status") != "fitted":
            skipped["missing_fitted_bayesian_curve"] = (
                skipped.get("missing_fitted_bayesian_curve", 0) + 1
            )
            continue
        fitted_positive_weight_before_policy += weight
        support_max = historical_line_support_max(line)
        if (
            contributor_policy == "fixed_intersection"
            and support_max is not None
            and float(support_max) < maximum_dose
        ):
            reason = (
                "evidence_support_below_maximum_planned_dose"
                if line.get("evidence_support_max") is not None
                else "mad_cap_below_maximum_planned_dose"
            )
            skipped[reason] = (
                skipped.get(reason, 0) + 1
            )
            excluded_fixed_lines.append(
                {
                    "pmcid": line.get("pmcid"),
                    "segment_index": line.get("segment_index"),
                    "arm_or_cohort": line.get("arm_or_cohort"),
                    "unnormalized_weight": weight,
                    **_historical_support_audit_fields(line),
                }
            )
            continue
        candidate_lines.append((line, weight))

    if not candidate_lines:
        raise ValueError(
            "No historical posterior line supports the requested dose range under "
            f"the {contributor_policy!r} contributor policy."
        )

    draw_count = min(
        len(
            np.asarray(
                line["bayesian_fit"]["posterior"]["draws_alpha_eta"], dtype=float
            )
        )
        for line, _weight in candidate_lines
    )
    if draw_count <= 0:
        raise ValueError("Historical posterior lines contain no parameter draws.")

    raw_draws = np.zeros((draw_count, len(resolved_doses)), dtype=float)
    contributor_counts: list[int] = []
    weight_sums: list[float] = []
    normalized_weight_audit: list[dict[str, Any]] = []
    extrapolation_weight_fractions: list[float] = []
    fixed_contributor_effective_sample_size: float | None = None
    if contributor_policy == "fixed_intersection":
        total_weight = sum(weight for _line, weight in candidate_lines)
        normalized = [
            (line, weight / total_weight) for line, weight in candidate_lines
        ]
        fixed_contributor_effective_sample_size = float(
            1.0 / sum(weight**2 for _line, weight in normalized)
        )
        for line, weight in normalized:
            line_matrix = np.column_stack(
                [
                    posterior_probability_draws(line["bayesian_fit"], float(dose))[
                        :draw_count
                    ]
                    for dose in resolved_doses
                ]
            )
            raw_draws += weight * line_matrix
            normalized_weight_audit.append(
                {
                    "pmcid": line.get("pmcid"),
                    "segment_index": line.get("segment_index"),
                    "arm_or_cohort": line.get("arm_or_cohort"),
                    "normalized_weight": float(weight),
                    **_historical_support_audit_fields(line),
                }
            )
        contributor_counts = [len(normalized)] * len(resolved_doses)
        weight_sums = [float(total_weight)] * len(resolved_doses)
        extrapolation_weight_fractions = [
            float(
                sum(
                    weight
                    for line, weight in normalized
                    if float(dose) < float(line["observed_dose_min"])
                    or float(dose) > float(line["observed_dose_max"])
                )
            )
            for dose in resolved_doses
        ]
    else:
        for column, dose in enumerate(resolved_doses):
            active = [
                (line, weight)
                for line, weight in candidate_lines
                if historical_line_support_max(line) is None
                or float(dose) <= float(historical_line_support_max(line))
            ]
            total_weight = sum(weight for _line, weight in active)
            if total_weight <= 0:
                raise ValueError(
                    f"No historical posterior line supports dose {float(dose):g}."
                )
            for line, weight in active:
                raw_draws[:, column] += (weight / total_weight) * (
                    posterior_probability_draws(
                        line["bayesian_fit"], float(dose)
                    )[:draw_count]
                )
            contributor_counts.append(len(active))
            weight_sums.append(float(total_weight))
            extrapolation_weight_fractions.append(
                float(
                    sum(
                        weight
                        for line, weight in active
                        if float(dose) < float(line["observed_dose_min"])
                        or float(dose) > float(line["observed_dose_max"])
                    )
                    / total_weight
                )
            )


    return {
        "status": "fitted",
        "doses": resolved_doses.copy(),
        "posterior_draws": raw_draws,
        "posterior_draw_count": int(draw_count),
        "contributor_policy": contributor_policy,
        "contributor_counts": contributor_counts,
        "unnormalized_weight_sums": weight_sums,
        "fixed_contributor_weights": normalized_weight_audit,
        "fixed_contributor_effective_sample_size": (
            fixed_contributor_effective_sample_size
        ),
        "fixed_policy_fitted_positive_weight_before_exclusion": float(
            fitted_positive_weight_before_policy
        ),
        "fixed_policy_excluded_lines": excluded_fixed_lines,
        "fixed_policy_excluded_weight_fraction": (
            float(
                sum(row["unnormalized_weight"] for row in excluded_fixed_lines)
                / fitted_positive_weight_before_policy
            )
            if fitted_positive_weight_before_policy > 0
            else 0.0
        ),
        "weighted_extrapolation_fraction_by_dose": extrapolation_weight_fractions,
        "skipped_lines": dict(sorted(skipped.items())),
    }


def similarity_weighted_mle_logistic_reference(
    lines: list[dict[str, Any]],
    doses: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Build a display-only MLE-logistic mixture from historical evidence.

    Each eligible historical arm is fit independently with the classical
    Binomial maximum-likelihood logistic model. Predictions are then mixed at
    every dose using the exact same similarity, reliability, and MAD-cap rules
    as the active Bayesian historical mixture. Held-out outcomes are never used.
    This function does not affect the predicted curve or validation metrics.
    The dashed overlay is a deterministic classical-MLE reference.
    """

    fitted_lines: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped_reasons: dict[str, int] = {}
    for line in lines:
        source_points = list(line.get("source_points") or [])
        if not source_points:
            skipped_reasons["missing_source_points"] = (
                skipped_reasons.get("missing_source_points", 0) + 1
            )
            continue
        fit = fit_mle_binomial_logistic(source_points)
        if fit.get("status") == "fitted":
            fitted_lines.append((line, fit))
            continue
        for reason in fit.get("skip_reasons", ["mle_logistic_not_fit_eligible"]):
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    curve_points: list[dict[str, Any]] = []
    for dose in doses:
        if not isinstance(dose, (int, float)) or isinstance(dose, bool) or dose <= 0:
            continue
        contributions: list[tuple[float, float]] = []
        for line, fit in fitted_lines:
            support_max = historical_line_support_max(line)
            if support_max is not None and dose > support_max:
                continue
            weight = float(line["segment_similarity_weight"]) * float(
                line["reliability_weight"]
            )
            if weight <= 0:
                continue
            contributions.append((weight, mle_probability(fit, float(dose))))
        total_weight = sum(weight for weight, _probability in contributions)
        if total_weight <= 0:
            curve_points.append(
                {
                    "dose": float(dose),
                    "dlt_probability": None,
                    "contributor_count": 0,
                    "effective_similarity_reliability_weight": 0.0,
                }
            )
            continue
        mixed = sum(
            weight * probability for weight, probability in contributions
        ) / total_weight
        curve_points.append(
            {
                "dose": float(dose),
                "dlt_probability": mixed,
                "contributor_count": len(contributions),
                "effective_similarity_reliability_weight": total_weight,
            }
        )

    return {
        "type": "similarity_weighted_historical_mle_logistic_reference_curve",
        "purpose": "visual_reference_only",
        "uses_heldout_outcomes": False,
        "not_used_for": [
            "historical_similarity_prediction",
            "leave_one_out_method_comparisons",
        ],
        "status": (
            "fitted"
            if any(point["dlt_probability"] is not None for point in curve_points)
            else "no_supported_historical_doses"
        ),
        "model": "similarity_weighted_mle_binomial_logistic_historical_mixture_v1",
        "curve_points": curve_points,
        "aggregation_stats": {
            "historical_line_count": len(lines),
            "mle_fitted_line_count": len(fitted_lines),
            "skipped_reasons": dict(sorted(skipped_reasons.items())),
        },
    }


def posterior_mixture_summary_at_dose(
    lines: list[dict[str, Any]],
    dose: float,
) -> dict[str, Any]:
    """Return a target-level posterior summary at one dose without refitting."""

    weighted_draws: list[tuple[float, np.ndarray]] = []
    for line in lines:
        support_max = historical_line_support_max(line)
        if support_max is not None and dose > support_max:
            continue
        weight = line["segment_similarity_weight"] * line["reliability_weight"]
        if weight <= 0:
            continue
        weighted_draws.append(
            (weight, posterior_probability_draws(line["bayesian_fit"], dose))
        )
    total_weight = sum(weight for weight, _draws in weighted_draws)
    if total_weight <= 0:
        return {
            "status": "no_supported_historical_lines_at_dose",
            "dose": dose,
            "dlt_probability": None,
            "posterior_credible_interval_95": None,
            "contributor_count": 0,
        }
    draw_count = min(len(draws) for _weight, draws in weighted_draws)
    draws = np.zeros(draw_count, dtype=float)
    for weight, line_draws in weighted_draws:
        draws += (weight / total_weight) * line_draws[:draw_count]
    return {
        "status": "fitted",
        "dose": dose,
        "dlt_probability": float(np.mean(draws)),
        "posterior_credible_interval_95": {
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
        },
        "contributor_count": len(weighted_draws),
        "posterior_draw_count": draw_count,
    }


def estimate_article_line_weighted(
    scored_articles: list[dict[str, Any]],
    spec: ToxicityQuery,
    *,
    bayesian_cache: dict[str, Any],
    bayesian_prior: str,
    single_observation_reliability_weight: float = (
        DEFAULT_SINGLE_OBSERVATION_RELIABILITY_WEIGHT
    ),
    historical_lines: list[dict[str, Any]] | None = None,
    historical_line_skipped: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate cached posterior curves and aggregate them by raw similarity."""

    if historical_lines is None:
        lines, skipped = build_similarity_historical_lines(
            scored_articles,
            spec,
            bayesian_cache=bayesian_cache,
            bayesian_prior=bayesian_prior,
            single_observation_reliability_weight=single_observation_reliability_weight,
        )
    else:
        lines = historical_lines
        skipped = historical_line_skipped or {}
    if not lines:
        return {
            "status": "no_eligible_historical_lines",
            "model": "similarity_weighted_bayesian_historical_curve_mixture",
            "curve_points": [],
            "aggregation_stats": {"skipped_reasons": dict(sorted(skipped.items()))},
        }

    curve_points = []
    for dose in spec.planned_doses:
        contributions: list[dict[str, Any]] = []
        skipped_at_dose: dict[str, int] = {}
        for line in lines:
            support_max = historical_line_support_max(line)
            if support_max is not None and dose > support_max:
                reason = (
                    "evidence_line_beyond_observed_support"
                    if line.get("evidence_support_max") is not None
                    else "not_reached_bayesian_line_beyond_mad"
                )
                skipped_at_dose[reason] = (
                    skipped_at_dose.get(reason, 0)
                    + 1
                )
                continue
            weight = (
                line["segment_similarity_weight"] * line["reliability_weight"]
            )
            if weight <= 0:
                continue
            posterior_draws = posterior_probability_draws(
                line["bayesian_fit"], dose
            )
            probability = float(np.mean(posterior_draws))
            contributions.append(
                {
                    "pmcid": line["pmcid"],
                    "segment_index": line["segment_index"],
                    "arm_or_cohort": line["arm_or_cohort"],
                    "drug_name": line["drug_name"],
                    "line_type": line["line_type"],
                    "similarity": line["similarity"],
                    "article_similarity_weight": line[
                        "article_similarity_weight"
                    ],
                    "segment_similarity_weight": line[
                        "segment_similarity_weight"
                    ],
                    "reliability_weight": line["reliability_weight"],
                    "reliability_policy": line["reliability_policy"],
                    "unnormalized_weight": weight,
                    "source_patient_count": line["source_patient_count"],
                    "posterior_mean_dlt_probability": probability,
                    "predicted_dlt_probability": probability,
                    "posterior_credible_interval_95": {
                        "low": float(np.quantile(posterior_draws, 0.025)),
                        "high": float(np.quantile(posterior_draws, 0.975)),
                    },
                    "posterior_draw_count": int(len(posterior_draws)),
                    "outside_observed_dose_range": (
                        dose < line["observed_dose_min"]
                        or dose > line["observed_dose_max"]
                    ),
                    **_historical_support_audit_fields(line),
                    "_posterior_draws": posterior_draws,
                }
            )

        total_weight = sum(
            contribution["unnormalized_weight"] for contribution in contributions
        )
        if total_weight <= 0:
            curve_points.append(
                {
                    "dose": dose,
                    "unit": spec.target_unit,
                    "dlt_probability": None,
                    "contributors": [],
                    "skipped_at_dose": skipped_at_dose,
                }
            )
            continue

        for contribution in contributions:
            contribution["normalized_weight"] = (
                contribution["unnormalized_weight"] / total_weight
            )
        values = [
            contribution["posterior_mean_dlt_probability"]
            for contribution in contributions
        ]
        weights = [contribution["normalized_weight"] for contribution in contributions]
        draw_count = min(
            contribution["posterior_draw_count"] for contribution in contributions
        )
        mixture_draws = np.zeros(draw_count, dtype=float)
        for contribution in contributions:
            mixture_draws += (
                contribution["normalized_weight"]
                * contribution["_posterior_draws"][:draw_count]
            )
            del contribution["_posterior_draws"]
        curve_points.append(
            {
                "dose": dose,
                "unit": spec.target_unit,
                "dlt_probability": float(np.mean(mixture_draws)),
                "posterior_credible_interval_95": {
                    "low": float(np.quantile(mixture_draws, 0.025)),
                    "high": float(np.quantile(mixture_draws, 0.975)),
                },
                "contributor_distribution_interval_95": {
                    "low": _weighted_quantile(values, weights, 0.025),
                    "high": _weighted_quantile(values, weights, 0.975),
                },
                "posterior_draw_count": draw_count,
                "contributor_count": len(contributions),
                "effective_similarity_reliability_weight": total_weight,
                "similarity_reliability_weighted_source_patients": sum(
                    contribution["normalized_weight"]
                    * contribution["source_patient_count"]
                    for contribution in contributions
                ),
                "single_observation_contributor_count": sum(
                    contribution["line_type"]
                    == "single_observation_bayesian_binomial_logistic"
                    for contribution in contributions
                ),
                "mad_capped_contributor_count": sum(
                    contribution.get("mad_cap") is not None
                    for contribution in contributions
                ),
                "extrapolation_weight_fraction": sum(
                    contribution["normalized_weight"]
                    for contribution in contributions
                    if contribution["outside_observed_dose_range"]
                ),
                "contributors": contributions,
                "skipped_at_dose": skipped_at_dose,
            }
        )

    line_weights = [
        line["segment_similarity_weight"] * line["reliability_weight"]
        for line in lines
    ]
    total_line_weight = sum(line_weights)
    normalized_line_weights = (
        [weight / total_line_weight for weight in line_weights]
        if total_line_weight > 0
        else []
    )
    supported_dose_count = sum(
        point["dlt_probability"] is not None for point in curve_points
    )
    return {
        "status": (
            "fitted" if supported_dose_count else "no_supported_historical_doses"
        ),
        "model": "similarity_weighted_bayesian_historical_curve_mixture",
        "bayesian_prior": bayesian_prior,
        "curve_points": curve_points,
        "historical_lines": lines,
        "aggregation_stats": {
            "line_count": len(lines),
            "article_count": len({line["pmcid"] for line in lines}),
            "single_observation_line_count": sum(
                line["line_type"]
                == "single_observation_bayesian_binomial_logistic"
                for line in lines
            ),
            "mad_capped_line_count": sum(
                line.get("mad_cap") is not None for line in lines
            ),
            "evidence_support_capped_line_count": sum(
                line.get("evidence_support_max") is not None for line in lines
            ),
            "requested_dose_count": len(curve_points),
            "supported_dose_count": supported_dose_count,
            # This global segment-weight diagnostic is not used by the
            # pointwise mixture, which directly starts from raw article
            # similarities and then removes non-contributing segments.
            "segment_weight_ess": effective_sample_size(
                normalized_line_weights
            ),
            "skipped_reasons": dict(sorted(skipped.items())),
        },
    }
