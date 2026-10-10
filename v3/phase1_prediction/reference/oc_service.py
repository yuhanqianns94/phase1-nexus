"""Service layer connecting a predicted toxicity curve to OC simulation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .oc_simulation import (
    MAX_TOTAL_VIRTUAL_TRIALS,
    OCConfig,
    TruthScenario,
    build_gpt_wrapper_standard_scenarios,
    build_stress_scenarios,
    monotonic_truth_scenario,
    simulate_operating_characteristics,
)
from .power_likelihood_curve import power_likelihood_draws_at_doses
from .power_likelihood_random_effects import random_effects_draws_at_doses
from .power_likelihood_random_intercept_slope import (
    random_intercept_slope_draws_at_doses,
)
from .posterior_curve_source_mixture import (
    METHOD_KEY as ARTICLE_POSTERIOR_DRAW_POOL_METHOD_KEY,
    source_mixture_draws_at_doses,
)
from .similarity_weighted_curve import posterior_mixture_draws_at_doses
from .prediction_contract import ensure_current_prediction


SUPPORTED_OC_CURVE_METHODS = (
    "article_line_weighted",
    ARTICLE_POSTERIOR_DRAW_POOL_METHOD_KEY,
    "power_likelihood",
    "power_likelihood_random_effects",
    "power_likelihood_random_intercept_slope",
)


def _curve_method(
    curve_result: Mapping[str, Any], method_key: str = "article_line_weighted"
) -> Mapping[str, Any]:
    if method_key not in SUPPORTED_OC_CURVE_METHODS:
        raise ValueError(
            "curve_method_key must be one of: " + ", ".join(SUPPORTED_OC_CURVE_METHODS)
        )
    methods = curve_result.get("toxicity_curve", {})
    method = methods.get(method_key, {}) if isinstance(methods, Mapping) else {}
    if not isinstance(method, Mapping) or method.get("status") != "fitted":
        raise ValueError(f"A fitted {method_key!r} toxicity curve is required.")
    return method


def planned_oc_doses(curve_result: Mapping[str, Any]) -> tuple[float, ...]:
    """Return user-entered discrete doses and reject an automatic display grid."""

    query = curve_result.get("query", {})
    raw_doses = query.get("planned_doses") if isinstance(query, Mapping) else None
    if not isinstance(raw_doses, list) or len(raw_doses) < 2:
        raise ValueError(
            "OC simulation requires at least two user-entered planned dose levels."
        )
    doses = tuple(float(value) for value in raw_doses)
    if not np.isfinite(doses).all() or any(value <= 0 for value in doses):
        raise ValueError("Planned OC doses must be positive finite values.")
    if any(right <= left for left, right in zip(doses, doses[1:])):
        raise ValueError("Planned OC doses must be strictly increasing.")
    dose_grid = query.get("dose_grid")
    if not isinstance(dose_grid, Mapping) or (
        dose_grid.get("mode") != "explicit_query_planned_doses"
    ):
        raise ValueError(
            "OC requires dose_grid.mode='explicit_query_planned_doses'. This "
            "prediction may use an automatic display grid or lacks proof that doses "
            "were entered by the user. Return to Target Trial and enter the actual "
            "discrete clinical dose levels before running OC."
        )
    return doses


def joint_target_posterior_draws(
    curve_result: Mapping[str, Any],
    *,
    curve_method_key: str = "article_line_weighted",
    contributor_policy: str = "fixed_intersection",
    requested_draw_count: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return the coherent posterior truth matrix used by OC."""

    if contributor_policy != "fixed_intersection":
        raise ValueError(
            "Formal OC truth generation only supports contributor_policy="
            "'fixed_intersection'. Pointwise weights may change between doses."
        )

    ensure_current_prediction(curve_result)
    method = _curve_method(curve_result, curve_method_key)
    doses = planned_oc_doses(curve_result)
    if curve_method_key == ARTICLE_POSTERIOR_DRAW_POOL_METHOD_KEY:
        sampled = source_mixture_draws_at_doses(
            method,
            doses,
            curve_count=requested_draw_count,
            seed=seed,
        )
        matrix = np.asarray(sampled["posterior_draws"], dtype=float)
        mixture_diagnostics = method.get("mixture_diagnostics", {})
        stored_skipped = (
            mixture_diagnostics.get("skipped_lines", {})
            if isinstance(mixture_diagnostics, Mapping)
            else {}
        )
        stored_skipped_details = (
            list(mixture_diagnostics.get("skipped_source_details") or [])
            if isinstance(mixture_diagnostics, Mapping)
            else []
        )
        source_allocations = list(sampled.get("source_allocations") or [])
        contributing_source_count = sum(
            int(int(row.get("allocated_curve_count", 0)) > 0)
            for row in source_allocations
            if isinstance(row, Mapping)
        )
        stored_lines = list(method.get("historical_lines") or [])
        realized_extrapolation_by_dose: list[float] = []
        for dose in doses:
            extrapolation_fraction = 0.0
            for row in source_allocations:
                if not isinstance(row, Mapping):
                    continue
                source_index = row.get("source_index")
                if not isinstance(source_index, int) or not (
                    0 <= source_index < len(stored_lines)
                ):
                    continue
                line = stored_lines[source_index]
                if not isinstance(line, Mapping):
                    continue
                observed_min = line.get("observed_dose_min")
                observed_max = line.get("observed_dose_max")
                below = isinstance(observed_min, (int, float)) and float(dose) < float(
                    observed_min
                )
                above = isinstance(observed_max, (int, float)) and float(dose) > float(
                    observed_max
                )
                if below or above:
                    extrapolation_fraction += float(
                        row.get("realized_curve_fraction", 0.0)
                    )
            realized_extrapolation_by_dose.append(extrapolation_fraction)
        return {
            "posterior_draws": matrix.tolist(),
            "curve_method_key": curve_method_key,
            "truth_construction": (
                "similarity_weighted_finite_mixture_of_complete_article_"
                "posterior_curve_draws"
            ),
            "prediction_target": (
                "one_similarity_weighted_historical_article_posterior_component"
            ),
            "contributor_policy": "fixed_intersection",
            "contributor_counts": [contributing_source_count] * len(doses),
            "skipped_lines": stored_skipped or sampled.get("skipped_lines", {}),
            "fixed_policy_excluded_lines": stored_skipped_details,
            "fixed_policy_excluded_weight_fraction": float(
                mixture_diagnostics.get("excluded_document_similarity_fraction", 0.0)
                if isinstance(mixture_diagnostics, Mapping)
                else 0.0
            ),
            "fixed_contributor_effective_sample_size": None,
            "mixture_source_effective_count": sampled.get("source_effective_count"),
            "historical_ess_cap": None,
            "pre_cap_effective_sample_size": None,
            "ess_cap_scale_factor": None,
            "weighted_extrapolation_fraction_by_dose": (realized_extrapolation_by_dose),
            "allocation_strategy": sampled.get("allocation_strategy"),
            "allocation_tie_breaker": sampled.get("allocation_tie_breaker"),
            "document_allocation_precedes_within_document_source_allocation": (
                sampled.get(
                    "document_allocation_precedes_within_document_source_allocation"
                )
            ),
            "article_allocations": list(sampled.get("article_allocations") or []),
            "source_allocations": source_allocations,
            "row_sources": list(sampled.get("row_sources") or []),
            "maximum_unique_curve_count": int(
                sampled.get("maximum_unique_curve_count", matrix.shape[0])
            ),
            "posterior_truth_rows_preallocated": requested_draw_count is not None,
            "posterior_truth_row_count": int(matrix.shape[0]),
            "second_stage_resampling": False,
            "second_stage_weighting": False,
        }
    if curve_method_key == "power_likelihood":
        matrix = power_likelihood_draws_at_doses(method, doses)
        weight_diagnostics = method.get("weight_diagnostics", {})
        return {
            "posterior_draws": matrix.tolist(),
            "curve_method_key": curve_method_key,
            "truth_construction": (
                "joint_shared_alpha_eta_power_likelihood_posterior_draw"
            ),
            "contributor_policy": "shared_parameter_posterior",
            "contributor_counts": [
                int(
                    method.get("weight_diagnostics", {}).get(
                        "contributor_line_count", 0
                    )
                )
            ]
            * len(doses),
            "skipped_lines": method.get("skipped_reasons", {}),
            "fixed_policy_excluded_lines": [],
            "fixed_policy_excluded_weight_fraction": 0.0,
            "fixed_contributor_effective_sample_size": method.get(
                "weight_diagnostics", {}
            ).get("effective_historical_sample_size"),
            "historical_ess_cap": weight_diagnostics.get("historical_ess_cap"),
            "pre_cap_effective_sample_size": weight_diagnostics.get(
                "pre_cap_effective_sample_size"
            ),
            "ess_cap_scale_factor": weight_diagnostics.get("ess_cap_scale_factor"),
            "weighted_extrapolation_fraction_by_dose": [
                point.get("extrapolation_power_fraction", 0.0)
                for point in method.get("curve_points", [])
            ],
        }
    if curve_method_key == "power_likelihood_random_effects":
        matrix = random_effects_draws_at_doses(method, doses)
        weight_diagnostics = method.get("weight_diagnostics", {})
        return {
            "posterior_draws": matrix.tolist(),
            "curve_method_key": curve_method_key,
            "truth_construction": (
                "joint_new_target_study_random_intercept_posterior_predictive_draw"
            ),
            "prediction_target": ("new_target_study_including_between_study_variation"),
            "contributor_policy": "shared_parameter_plus_new_study_random_intercept",
            "contributor_counts": [
                int(
                    method.get("weight_diagnostics", {}).get(
                        "contributor_line_count", 0
                    )
                )
            ]
            * len(doses),
            "skipped_lines": method.get("skipped_reasons", {}),
            "fixed_policy_excluded_lines": [],
            "fixed_policy_excluded_weight_fraction": 0.0,
            "fixed_contributor_effective_sample_size": method.get(
                "weight_diagnostics", {}
            ).get("effective_historical_sample_size"),
            "historical_ess_cap": weight_diagnostics.get("historical_ess_cap"),
            "pre_cap_effective_sample_size": weight_diagnostics.get(
                "pre_cap_effective_sample_size"
            ),
            "ess_cap_scale_factor": weight_diagnostics.get("ess_cap_scale_factor"),
            "weighted_extrapolation_fraction_by_dose": [
                point.get("extrapolation_power_fraction", 0.0)
                for point in method.get("curve_points", [])
            ],
            "between_study_tau": method.get("heterogeneity_diagnostics", {})
            .get("tau", {})
            .get("posterior_mean"),
        }
    if curve_method_key == "power_likelihood_random_intercept_slope":
        matrix = random_intercept_slope_draws_at_doses(method, doses)
        weight_diagnostics = method.get("weight_diagnostics", {})
        heterogeneity = method.get("heterogeneity_diagnostics", {})
        return {
            "posterior_draws": matrix.tolist(),
            "curve_method_key": curve_method_key,
            "truth_construction": (
                "joint_new_target_study_random_intercept_and_slope_posterior_predictive_draw"
            ),
            "prediction_target": (
                "new_target_study_including_intercept_and_slope_variation"
            ),
            "contributor_policy": (
                "shared_population_parameters_plus_new_study_intercept_and_slope"
            ),
            "contributor_counts": [
                int(weight_diagnostics.get("contributor_line_count", 0))
            ]
            * len(doses),
            "skipped_lines": method.get("skipped_reasons", {}),
            "fixed_policy_excluded_lines": [],
            "fixed_policy_excluded_weight_fraction": 0.0,
            "fixed_contributor_effective_sample_size": weight_diagnostics.get(
                "effective_historical_sample_size"
            ),
            "historical_ess_cap": weight_diagnostics.get("historical_ess_cap"),
            "pre_cap_effective_sample_size": weight_diagnostics.get(
                "pre_cap_effective_sample_size"
            ),
            "ess_cap_scale_factor": weight_diagnostics.get("ess_cap_scale_factor"),
            "weighted_extrapolation_fraction_by_dose": [
                point.get("extrapolation_power_fraction", 0.0)
                for point in method.get("curve_points", [])
            ],
            "between_study_intercept_scale": heterogeneity.get(
                "random_intercept_scale", {}
            ).get("posterior_mean"),
            "between_study_slope_latent_scale": heterogeneity.get(
                "random_slope_latent_scale", {}
            ).get("posterior_mean"),
            "target_curve_effect_sharing": method.get(
                "target_curve_effect_sharing"
            ),
        }
    payload = posterior_mixture_draws_at_doses(
        list(method.get("historical_lines") or []),
        doses,
        contributor_policy=contributor_policy,
    )
    payload["curve_method_key"] = curve_method_key
    payload["truth_construction"] = (
        "fixed_contributor_weighted_average_of_article_posterior_curve_draws"
    )
    return payload


def _display_curve_probabilities(
    curve_result: Mapping[str, Any],
    doses: Sequence[float],
    *,
    curve_method_key: str = "article_line_weighted",
) -> list[float | None]:
    method = _curve_method(curve_result, curve_method_key)
    by_dose = {
        float(point["dose"]): point.get("dlt_probability")
        for point in method.get("curve_points", [])
        if point.get("dose") is not None
    }
    return [
        (float(by_dose[dose]) if by_dose.get(dose) is not None else None)
        for dose in doses
    ]


def build_truth_scenarios_from_curve(
    curve_result: Mapping[str, Any],
    *,
    mode: str,
    posterior_curve_count: int = 100,
    seed: int = 9,
    custom_probabilities: Sequence[float] | None = None,
    include_stress_scenarios: bool = False,
    contributor_policy: str = "fixed_intersection",
    curve_method_key: str = "article_line_weighted",
    target_dlt: float = 0.25,
) -> tuple[list[TruthScenario], dict[str, Any]]:
    """Generate reference-standard, posterior, custom, or stress OC truths."""

    if mode not in {
        "gpt_wrapper_standard",
        "posterior_mean",
        "posterior_predictive",
        "stress",
        "custom",
    }:
        raise ValueError(
            "mode must be gpt_wrapper_standard, posterior_mean, "
            "posterior_predictive, stress, or custom."
        )
    if (
        isinstance(posterior_curve_count, bool)
        or not isinstance(posterior_curve_count, int)
        or posterior_curve_count <= 0
    ):
        raise ValueError("posterior_curve_count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    doses = planned_oc_doses(curve_result)
    if mode == "gpt_wrapper_standard":
        scenarios = build_gpt_wrapper_standard_scenarios(
            doses, target_dlt=float(target_dlt)
        )
        diagnostics = {
            "mode": mode,
            "truth_construction": "gpt_wrapper_k_by_k_diagonal_target_matrix",
            "reference_true_mtd_level_by_scenario": list(range(1, len(doses) + 1)),
            "posterior_truth_not_used": True,
            "stress_scenarios_requested": False,
        }
        if include_stress_scenarios:
            raise ValueError(
                "Standard GPT-wrapper scenarios and stress scenarios must be run "
                "as separate analyses; they cannot be combined into one set."
            )
        return scenarios, diagnostics

    direct_source_mixture_pool = (
        mode == "posterior_predictive"
        and curve_method_key == ARTICLE_POSTERIOR_DRAW_POOL_METHOD_KEY
    )
    joint = joint_target_posterior_draws(
        curve_result,
        curve_method_key=curve_method_key,
        contributor_policy=contributor_policy,
        requested_draw_count=(
            posterior_curve_count if direct_source_mixture_pool else None
        ),
        seed=seed if direct_source_mixture_pool else None,
    )
    matrix = np.asarray(joint["posterior_draws"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(doses):
        raise ValueError("Joint posterior draw matrix has an invalid shape.")
    joint_mean = np.mean(matrix, axis=0)
    if np.any(np.diff(matrix, axis=1) < -1e-10):
        raise ValueError(
            "A fixed-contributor posterior truth draw decreased with dose. Refusing "
            "to hide a joint-curve construction error with isotonic projection."
        )
    display_mean = _display_curve_probabilities(
        curve_result, doses, curve_method_key=curve_method_key
    )
    mean_differences = [
        (
            float(joint_mean[index] - float(display_value))
            if display_value is not None
            else None
        )
        for index, display_value in enumerate(display_mean)
    ]
    finite_differences = [value for value in mean_differences if value is not None]
    maximum_mean_difference = (
        max(abs(value) for value in finite_differences) if finite_differences else None
    )
    diagnostics = {
        "mode": mode,
        "curve_method_key": curve_method_key,
        "truth_construction": joint.get("truth_construction"),
        "prediction_target": joint.get("prediction_target"),
        "joint_posterior_draw_count_available": int(
            joint.get("maximum_unique_curve_count", matrix.shape[0])
        ),
        "joint_posterior_draw_count_materialized": int(matrix.shape[0]),
        "contributor_policy": contributor_policy,
        "fixed_contributor_count": (
            joint["contributor_counts"][0] if joint.get("contributor_counts") else 0
        ),
        "skipped_historical_lines": joint.get("skipped_lines", {}),
        "fixed_policy_excluded_lines": joint.get("fixed_policy_excluded_lines", []),
        "fixed_policy_excluded_weight_fraction": joint.get(
            "fixed_policy_excluded_weight_fraction", 0.0
        ),
        "fixed_contributor_effective_sample_size": joint.get(
            "fixed_contributor_effective_sample_size"
        ),
        "historical_ess_cap": joint.get("historical_ess_cap"),
        "pre_cap_effective_sample_size": joint.get("pre_cap_effective_sample_size"),
        "ess_cap_scale_factor": joint.get("ess_cap_scale_factor"),
        "weighted_extrapolation_fraction_by_dose": joint.get(
            "weighted_extrapolation_fraction_by_dose", []
        ),
        "display_pointwise_posterior_mean": display_mean,
        "oc_fixed_contributor_posterior_mean": joint_mean.tolist(),
        "oc_minus_display_mean_by_dose": mean_differences,
        "maximum_absolute_oc_vs_display_mean_difference": maximum_mean_difference,
        "material_oc_vs_display_difference_warning": bool(
            maximum_mean_difference is not None and maximum_mean_difference > 0.05
        ),
        "mixture_source_effective_count": joint.get("mixture_source_effective_count"),
        "allocation_strategy": joint.get("allocation_strategy"),
        "allocation_tie_breaker": joint.get("allocation_tie_breaker"),
        "document_allocation_precedes_within_document_source_allocation": (
            joint.get("document_allocation_precedes_within_document_source_allocation")
        ),
        "article_allocations": joint.get("article_allocations", []),
        "source_allocations": joint.get("source_allocations", []),
        "row_sources": joint.get("row_sources", []),
        "posterior_truth_rows_preallocated": bool(
            joint.get("posterior_truth_rows_preallocated", False)
        ),
        "second_stage_resampling": joint.get("second_stage_resampling"),
        "second_stage_weighting": joint.get("second_stage_weighting"),
    }

    scenarios: list[TruthScenario]
    if mode == "posterior_mean":
        scenarios = [
            monotonic_truth_scenario(
                f"{curve_method_key}_posterior_mean",
                joint_mean,
                source=f"{curve_method_key}_joint_posterior_mean",
                metadata={"curve_method_key": curve_method_key},
            )
        ]
    elif mode == "posterior_predictive":
        preallocated = bool(joint.get("posterior_truth_rows_preallocated", False))
        if preallocated and posterior_curve_count != matrix.shape[0]:
            raise RuntimeError(
                "The directly allocated article-posterior mixture did not return "
                "exactly the requested number of truth curves."
            )
        if not preallocated and posterior_curve_count > matrix.shape[0]:
            raise ValueError(
                f"Requested {posterior_curve_count} posterior truth curves, but the "
                f"cache contains only {matrix.shape[0]} joint draws. Rebuild the "
                "Bayesian cache with more posterior draws or request fewer curves; "
                "duplicating cached curves would not add precision."
            )
        if preallocated:
            indices = np.arange(matrix.shape[0], dtype=int)
        else:
            rng = np.random.default_rng(seed)
            indices = rng.choice(
                matrix.shape[0], size=posterior_curve_count, replace=False
            )

        row_sources_by_index = {
            int(row["row_index"]): row
            for row in joint.get("row_sources", [])
            if isinstance(row, Mapping) and row.get("row_index") is not None
        }
        source_allocations_by_id = {
            str(row["source_id"]): row
            for row in joint.get("source_allocations", [])
            if isinstance(row, Mapping) and row.get("source_id") is not None
        }
        scenarios = []
        for position, draw_index in enumerate(indices):
            resolved_draw_index = int(draw_index)
            metadata: dict[str, Any] = {
                "posterior_draw_index": resolved_draw_index,
                "curve_method_key": curve_method_key,
                "truth_construction": joint.get("truth_construction"),
                "prediction_target": joint.get("prediction_target"),
            }
            if preallocated:
                row_source = row_sources_by_index.get(resolved_draw_index, {})
                source_id = row_source.get("source_id")
                source_allocation = source_allocations_by_id.get(str(source_id), {})
                metadata.update(
                    {
                        "posterior_pool_row_index": resolved_draw_index,
                        "posterior_draw_index": row_source.get("posterior_draw_index"),
                        "posterior_component_source": {
                            "source_id": source_id,
                            "document_id": source_allocation.get("document_id"),
                            "pmcid": source_allocation.get("pmcid"),
                            "study_unit_id": source_allocation.get("study_unit_id"),
                            "escalation_series_id": source_allocation.get(
                                "escalation_series_id"
                            ),
                            "segment_index": source_allocation.get("segment_index"),
                            "arm_or_cohort": source_allocation.get("arm_or_cohort"),
                        },
                    }
                )
            scenarios.append(
                monotonic_truth_scenario(
                    f"posterior_curve_{position + 1:04d}",
                    matrix[resolved_draw_index],
                    source=f"{curve_method_key}_joint_posterior_draw",
                    metadata=metadata,
                )
            )

        source_draw_pairs = {
            (str(row.get("source_id")), int(row.get("posterior_draw_index")))
            for row in joint.get("row_sources", [])
            if isinstance(row, Mapping)
            and row.get("source_id") is not None
            and row.get("posterior_draw_index") is not None
        }
        diagnostics.update(
            {
                "posterior_curves_requested": posterior_curve_count,
                "sampling_with_replacement": False,
                "unique_posterior_draws_used": (
                    len(source_draw_pairs)
                    if preallocated
                    else int(len(set(map(int, indices))))
                ),
                "unique_source_posterior_draw_pairs_used": (
                    len(source_draw_pairs) if preallocated else None
                ),
                "posterior_pool_generated_directly_for_requested_count": preallocated,
                "second_stage_resampling": False if preallocated else None,
                "second_stage_weighting": False if preallocated else None,
            }
        )
    elif mode == "custom":
        if custom_probabilities is None:
            raise ValueError("custom mode requires custom_probabilities.")
        if len(custom_probabilities) != len(doses):
            raise ValueError(
                "custom_probabilities must contain one value for every planned dose."
            )
        scenarios = [
            monotonic_truth_scenario(
                "user_defined_truth",
                custom_probabilities,
                source="user_defined_true_dlt_probabilities",
            )
        ]
    else:
        scenarios = build_stress_scenarios(
            doses, joint_mean, target_dlt=float(target_dlt)
        )

    if include_stress_scenarios and mode != "stress":
        scenarios.extend(
            build_stress_scenarios(doses, joint_mean, target_dlt=float(target_dlt))
        )
        diagnostics["stress_scenarios_requested"] = True
    return scenarios, diagnostics


def run_oc_from_curve(
    curve_result: Mapping[str, Any],
    config: OCConfig,
    *,
    mode: str = "posterior_predictive",
    posterior_curve_count: int = 100,
    custom_probabilities: Sequence[float] | None = None,
    include_stress_scenarios: bool = False,
    contributor_policy: str = "fixed_intersection",
    curve_method_key: str = "article_line_weighted",
) -> dict[str, Any]:
    """Build truth curves from one prediction and run selected trial designs."""

    resolved = config.validated()
    expected_doses = planned_oc_doses(curve_result)
    if tuple(resolved.doses) != tuple(expected_doses):
        raise ValueError(
            "OCConfig doses must exactly match the prediction's planned doses."
        )
    scenarios, diagnostics = build_truth_scenarios_from_curve(
        curve_result,
        mode=mode,
        posterior_curve_count=posterior_curve_count,
        seed=resolved.seed,
        custom_probabilities=custom_probabilities,
        include_stress_scenarios=False,
        contributor_policy=contributor_policy,
        curve_method_key=curve_method_key,
        target_dlt=resolved.target_dlt,
    )
    stress_scenarios: list[TruthScenario] = []
    if include_stress_scenarios and mode != "stress":
        stress_scenarios = build_stress_scenarios(
            resolved.doses,
            diagnostics["oc_fixed_contributor_posterior_mean"],
            target_dlt=resolved.target_dlt,
        )
    requested_work = (
        (len(scenarios) + len(stress_scenarios))
        * resolved.trials_per_scenario
        * len(resolved.designs)
    )
    if requested_work > MAX_TOTAL_VIRTUAL_TRIALS:
        raise ValueError(
            f"Requested {requested_work:,} total virtual trials exceeds the "
            f"synchronous limit of {MAX_TOTAL_VIRTUAL_TRIALS:,}."
        )
    aggregate_primary = mode in {
        "posterior_mean",
        "posterior_predictive",
        "custom",
    }
    result = simulate_operating_characteristics(
        scenarios,
        resolved,
        aggregate_scenarios=aggregate_primary,
    )
    if stress_scenarios:
        result["stress_test"] = simulate_operating_characteristics(
            stress_scenarios,
            resolved,
            aggregate_scenarios=False,
        )
        result["stress_test"]["aggregation_interpretation"] = (
            "Stress scenarios have no posterior model probabilities. Their "
            "per-scenario results and ranges must not be averaged into Bayesian "
            "assurance."
        )
    result["analysis_interpretation"] = (
        "conditional_assurance_under_fixed_historical_ensemble"
        if mode == "posterior_predictive"
        else (
            "unweighted_stress_scenario_set"
            if mode == "stress"
            else (
                "gpt_wrapper_standard_conditional_operating_characteristics"
                if mode == "gpt_wrapper_standard"
                else "conditional_operating_characteristics"
            )
        )
    )
    result["truth_generation"] = {
        **diagnostics,
        "mode": mode,
        "posterior_curve_count": (
            posterior_curve_count if mode == "posterior_predictive" else None
        ),
        "stress_scenarios_included": bool(mode == "stress" or stress_scenarios),
        "stress_scenarios_separate_from_assurance": bool(stress_scenarios),
        "scientific_warning": (
            "The standard GPT-wrapper mode reports each target-location scenario "
            "separately; these scenarios do not have probabilities and must not be "
            "averaged."
            if mode == "gpt_wrapper_standard"
            else (
                "Historical posterior truths and a design prior derived from the "
                "same curve can be optimistically aligned; report stress scenarios "
                "and an independent decision prior."
            )
        ),
    }
    return result
