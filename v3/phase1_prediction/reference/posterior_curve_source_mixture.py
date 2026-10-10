"""Sample complete posterior curves from similarity-weighted historical sources.

This module implements a *mixture of posteriors*, not the existing draw-wise
weighted average of probabilities.  Every returned row is copied from one
article/segment posterior, so a row can never splice dose columns from
different historical sources.

The function is deliberately independent from the UI and OC service.  Its
return payload contains enough allocation and posterior-index metadata for a
caller to audit exactly where every generated curve came from.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .bayesian_toxicity import (
    DEFAULT_POSTERIOR_DRAW_COUNT,
    posterior_probability_draws,
)
from .similarity_weighted_curve import historical_line_support_max
from .prediction_contract import ensure_current_prediction
from .toxicity_curve_common import ToxicityQuery


AllocationStrategy = Literal["largest_remainder", "multinomial"]
METHOD_KEY = "article_posterior_draw_pool"
MODEL_NAME = "similarity_weighted_article_component_posterior_curve_mixture_v1"
DEFAULT_MIXTURE_SEED = 20_260_823


def _validate_doses(doses: Sequence[float]) -> np.ndarray:
    resolved = np.asarray(doses, dtype=float)
    if (
        resolved.ndim != 1
        or len(resolved) == 0
        or not np.isfinite(resolved).all()
        or np.any(resolved <= 0)
    ):
        raise ValueError("doses must be a non-empty vector of positive finite values.")
    if np.any(np.diff(resolved) <= 0):
        raise ValueError("doses must be strictly increasing for joint curve draws.")
    return resolved


def _positive_finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved) or resolved <= 0:
        return None
    return resolved


def _line_weight(line: Mapping[str, Any]) -> tuple[float | None, str, float | None]:
    """Resolve one component's within-document allocation signal.

    An explicit zero ``segment_similarity_weight`` means that the component was
    deliberately assigned no weight.  It must not fall back to the article
    weight.  Fallback is reserved for legacy rows where that field is absent or
    null.
    """

    similarity_field = "segment_similarity_weight"
    if similarity_field in line and line.get(similarity_field) is not None:
        similarity_weight = _positive_finite_number(line.get(similarity_field))
    else:
        similarity_field = "article_similarity_weight"
        similarity_weight = _positive_finite_number(line.get(similarity_field))
        if similarity_weight is None:
            similarity_field = "similarity"
            similarity_weight = _positive_finite_number(line.get(similarity_field))
    reliability_weight = _positive_finite_number(line.get("reliability_weight", 1.0))
    if similarity_weight is None or reliability_weight is None:
        return None, similarity_field, reliability_weight
    return similarity_weight * reliability_weight, similarity_field, reliability_weight


def _article_similarity_weight(line: Mapping[str, Any]) -> float | None:
    """Return the document-level similarity mass, independent of component count."""

    if (
        "article_similarity_weight" in line
        and line.get("article_similarity_weight") is not None
    ):
        return _positive_finite_number(line.get("article_similarity_weight"))
    similarity = _positive_finite_number(line.get("similarity"))
    if similarity is not None:
        return similarity
    if (
        "segment_similarity_weight" in line
        and line.get("segment_similarity_weight") is not None
    ):
        return _positive_finite_number(line.get("segment_similarity_weight"))
    return None


def _posterior_draw_count(line: Mapping[str, Any]) -> int:
    fit = line.get("bayesian_fit")
    if not isinstance(fit, Mapping) or fit.get("status") != "fitted":
        return 0
    posterior = fit.get("posterior")
    if not isinstance(posterior, Mapping):
        return 0
    draws = np.asarray(posterior.get("draws_alpha_eta"), dtype=float)
    if (
        draws.ndim != 2
        or draws.shape[1] != 2
        or len(draws) == 0
        or not np.isfinite(draws).all()
    ):
        return 0
    return int(len(draws))


def _source_identity(line: Mapping[str, Any]) -> dict[str, Any]:
    """Build a stable component identity that does not depend on input order."""

    document_id = line.get("document_id") or line.get("pmcid")
    study_unit_id = line.get("study_unit_id")
    escalation_series_id = line.get("escalation_series_id")
    segment_index = line.get("segment_index")
    arm_or_cohort = line.get("arm_or_cohort")
    readable_parts = [
        str(value)
        for value in (
            document_id,
            study_unit_id,
            escalation_series_id,
            segment_index,
        )
        if value not in (None, "")
    ]
    stable_payload = {
        "document_id": document_id,
        "pmcid": line.get("pmcid"),
        "study_unit_id": study_unit_id,
        "escalation_series_id": escalation_series_id,
        "segment_index": segment_index,
        "arm_or_cohort": arm_or_cohort,
        "drug_name": line.get("drug_name"),
        "dose_unit": line.get("dose_unit"),
        "observed_dose_min": line.get("observed_dose_min"),
        "observed_dose_max": line.get("observed_dose_max"),
        "source_points": line.get("source_points"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            stable_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    readable = "::".join(readable_parts) or "historical-line"
    return {
        "source_id": f"{readable}::{fingerprint}",
        "document_id": document_id,
        "pmcid": line.get("pmcid") or document_id,
        "study_unit_id": study_unit_id,
        "escalation_series_id": escalation_series_id,
        "segment_index": segment_index,
        "arm_or_cohort": arm_or_cohort,
    }


def _largest_remainder_allocation(
    normalized_weights: np.ndarray, curve_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ideal = normalized_weights * curve_count
    floors = np.floor(ideal).astype(int)
    remainders = ideal - floors
    remaining = int(curve_count - int(np.sum(floors)))
    # Input order is the explicit, auditable tie-breaker when fractional
    # remainders are identical.
    priority = sorted(
        range(len(remainders)), key=lambda index: (-remainders[index], index)
    )
    allocated = floors.copy()
    for index in priority[:remaining]:
        allocated[index] += 1
    return allocated, ideal, remainders


def _document_key(source: Mapping[str, Any]) -> str:
    identity = source["identity"]
    return str(
        identity.get("document_id")
        or identity.get("pmcid")
        or identity.get("source_id")
    )


def _prepare_eligible_sources(
    historical_lines: Sequence[Mapping[str, Any]],
    resolved_doses: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    maximum_dose = float(np.max(resolved_doses))
    eligible: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    skipped_details: list[dict[str, Any]] = []
    document_similarity: dict[str, float] = {}
    identities = [_source_identity(line) for line in historical_lines]
    for line, identity in zip(historical_lines, identities, strict=True):
        document_key = str(
            identity.get("document_id")
            or identity.get("pmcid")
            or identity.get("source_id")
        )
        article_weight = _article_similarity_weight(line)
        if article_weight is not None:
            document_similarity[document_key] = max(
                document_similarity.get(document_key, 0.0), article_weight
            )

    seen_source_ids: set[str] = set()
    for source_index, line in enumerate(historical_lines):
        identity = identities[source_index]
        document_key = str(
            identity.get("document_id")
            or identity.get("pmcid")
            or identity.get("source_id")
        )

        def skip(reason: str) -> None:
            skipped[reason] += 1
            skipped_details.append(
                {
                    **identity,
                    "original_source_index": int(source_index),
                    "reason": reason,
                    "article_similarity_weight": document_similarity.get(document_key),
                    "segment_similarity_weight": line.get("segment_similarity_weight"),
                    "reliability_weight": line.get("reliability_weight"),
                    "support_max": historical_line_support_max(dict(line)),
                }
            )

        if identity["source_id"] in seen_source_ids:
            skip("duplicate_historical_component_identity")
            continue
        weight, similarity_field, reliability_weight = _line_weight(line)
        if weight is None:
            skip("non_positive_similarity_reliability_weight")
            continue
        article_weight = document_similarity.get(document_key)
        if article_weight is None or article_weight <= 0:
            skip("non_positive_article_similarity_weight")
            continue
        available_draw_count = _posterior_draw_count(line)
        if available_draw_count <= 0:
            skip("missing_fitted_bayesian_curve")
            continue
        support_max = historical_line_support_max(dict(line))
        if support_max is not None and support_max < maximum_dose:
            reason = (
                "evidence_support_below_maximum_planned_dose"
                if line.get("evidence_support_max") is not None
                else "mad_cap_below_maximum_planned_dose"
            )
            skip(reason)
            continue
        similarity_weight = weight / float(reliability_weight)
        seen_source_ids.add(str(identity["source_id"]))
        eligible.append(
            {
                "line": line,
                "identity": identity,
                "original_source_index": int(source_index),
                "document_similarity_weight": float(article_weight),
                "unnormalized_weight": float(weight),
                "similarity_weight": float(similarity_weight),
                "similarity_weight_field": similarity_field,
                "reliability_weight": float(reliability_weight),
                "available_posterior_draw_count": available_draw_count,
                "support_max": support_max,
            }
        )
    eligible.sort(key=lambda source: str(source["identity"]["source_id"]))
    for stored_source_index, source in enumerate(eligible):
        source["identity"] = {
            **source["identity"],
            "source_index": int(stored_source_index),
            "original_source_index": int(source["original_source_index"]),
        }
    return (
        eligible,
        dict(sorted(skipped.items())),
        sorted(
            skipped_details,
            key=lambda row: (str(row.get("source_id")), str(row.get("reason"))),
        ),
    )


def _document_allocation_inputs(
    eligible: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[int]], list[str], np.ndarray, np.ndarray, np.ndarray]:
    positions_by_document: dict[str, list[int]] = defaultdict(list)
    for source_position, source in enumerate(eligible):
        positions_by_document[_document_key(source)].append(source_position)
    document_keys = sorted(positions_by_document)
    document_raw_weights = np.asarray(
        [
            max(
                float(eligible[position]["document_similarity_weight"])
                for position in positions_by_document[key]
            )
            for key in document_keys
        ],
        dtype=float,
    )
    document_normalized_weights = document_raw_weights / float(
        np.sum(document_raw_weights)
    )
    document_capacities = np.asarray(
        [
            sum(
                int(eligible[position]["available_posterior_draw_count"])
                for position in positions_by_document[key]
            )
            for key in document_keys
        ],
        dtype=int,
    )
    return (
        positions_by_document,
        document_keys,
        document_raw_weights,
        document_normalized_weights,
        document_capacities,
    )


def _hierarchical_largest_remainder_plan(
    eligible: Sequence[Mapping[str, Any]],
    positions_by_document: Mapping[str, Sequence[int]],
    document_keys: Sequence[str],
    document_normalized_weights: np.ndarray,
    total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate exact quotas without changing weights to accommodate caches.

    Capacity is deliberately *not* considered here.  The statistical contract
    is article-level Hamilton allocation followed by Hamilton allocation among
    that article's eligible components.  Callers validate the resulting target
    quotas against posterior-cache capacity and fail rather than silently move
    curves to a less similar component.
    """

    document_allocation, document_ideal, document_remainders = (
        _largest_remainder_allocation(document_normalized_weights, total)
    )
    source_allocation = np.zeros(len(eligible), dtype=int)
    source_ideal = np.zeros(len(eligible), dtype=float)
    source_remainders = np.zeros(len(eligible), dtype=float)
    for document_position, document_key in enumerate(document_keys):
        source_positions = list(positions_by_document[document_key])
        within_raw = np.asarray(
            [
                eligible[position]["unnormalized_weight"]
                for position in source_positions
            ],
            dtype=float,
        )
        within_weights = within_raw / float(np.sum(within_raw))
        allocated, ideal, remainders = _largest_remainder_allocation(
            within_weights, int(document_allocation[document_position])
        )
        for local_position, source_position in enumerate(source_positions):
            source_allocation[source_position] = int(allocated[local_position])
            source_ideal[source_position] = float(ideal[local_position])
            source_remainders[source_position] = float(remainders[local_position])
    return (
        document_allocation,
        document_ideal,
        document_remainders,
        source_allocation,
        source_ideal,
        source_remainders,
    )


def _maximum_contiguous_hierarchical_capacity(
    eligible: Sequence[Mapping[str, Any]],
    positions_by_document: Mapping[str, Sequence[int]],
    document_keys: Sequence[str],
    document_normalized_weights: np.ndarray,
) -> int:
    """Return the exact largest-remainder capacity from one through ``R``.

    Hamilton quotas can cross integer boundaries in ways that a simple
    ``floor(capacity / weight)`` ratio does not capture.  We therefore replay
    the actual two-level allocation for increasing pool sizes and stop at the
    first size whose component quota cannot be sampled without replacement.
    """

    source_capacities = np.asarray(
        [int(source["available_posterior_draw_count"]) for source in eligible],
        dtype=int,
    )
    total_capacity = int(np.sum(source_capacities))
    guaranteed_bounds: list[int] = []
    for document_position, document_key in enumerate(document_keys):
        source_positions = list(positions_by_document[document_key])
        within_raw = np.asarray(
            [
                eligible[position]["unnormalized_weight"]
                for position in source_positions
            ],
            dtype=float,
        )
        within_weights = within_raw / float(np.sum(within_raw))
        document_weight = float(document_normalized_weights[document_position])
        for local_position, source_position in enumerate(source_positions):
            # Hamilton gives a document at most ceil(R * q_document) curves,
            # and gives a component at most ceil(Q_document * q_component).
            # The bound below therefore proves that every pool size through R
            # fits, without enumerating those obviously safe sizes one by one.
            maximum_safe_document_quota = math.floor(
                float(source_capacities[source_position])
                / float(within_weights[local_position])
                + 1e-12
            )
            guaranteed_bounds.append(
                math.floor(maximum_safe_document_quota / document_weight + 1e-12)
            )
    guaranteed_capacity = min(
        total_capacity,
        min(guaranteed_bounds, default=0),
    )
    for total in range(guaranteed_capacity + 1, total_capacity + 1):
        plan = _hierarchical_largest_remainder_plan(
            eligible,
            positions_by_document,
            document_keys,
            document_normalized_weights,
            total,
        )
        if np.any(plan[3] > source_capacities):
            return total - 1
    return total_capacity


def _capacity_constrained_multinomial(
    normalized_weights: np.ndarray,
    total: int,
    capacities: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if total > int(np.sum(capacities)):
        raise ValueError(
            f"Requested quota {total} exceeds the available no-replacement "
            f"component capacity {int(np.sum(capacities))}."
        )
    allocated = np.zeros(len(normalized_weights), dtype=int)
    for _ in range(total):
        available = allocated < capacities
        probabilities = np.where(available, normalized_weights, 0.0)
        probabilities = probabilities / float(np.sum(probabilities))
        allocated[int(rng.choice(len(probabilities), p=probabilities))] += 1
    ideal = normalized_weights * total
    return allocated, ideal, ideal - np.floor(ideal)


def _component_variance_decomposition(
    draws: np.ndarray,
    provenance: Sequence[tuple[str, int]],
    doses: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    source_ids = sorted({source_id for source_id, _draw_index in provenance})
    for dose_position, dose in enumerate(doses):
        dose_values = draws[:, dose_position]
        overall_mean = float(np.mean(dose_values))
        within = 0.0
        between = 0.0
        for source_id in source_ids:
            positions = [
                index
                for index, (row_source_id, _draw_index) in enumerate(provenance)
                if row_source_id == source_id
            ]
            source_values = dose_values[positions]
            realized_weight = len(positions) / len(draws)
            source_mean = float(np.mean(source_values))
            within += realized_weight * float(np.var(source_values))
            between += realized_weight * (source_mean - overall_mean) ** 2
        rows.append(
            {
                "dose": float(dose),
                "within_component_variance": float(within),
                "between_component_variance": float(between),
                "total_mixture_variance": float(np.var(dose_values)),
            }
        )
    return rows


def maximum_unique_source_mixture_draw_count(
    historical_lines: Sequence[Mapping[str, Any]],
    doses: Sequence[float],
) -> int:
    """Return a quota-aware, contiguous no-replacement capacity."""

    eligible, _skipped, _skipped_details = _prepare_eligible_sources(
        historical_lines, _validate_doses(doses)
    )
    if not eligible:
        return 0
    (
        positions_by_document,
        document_keys,
        _document_raw_weights,
        document_normalized_weights,
        _document_capacities,
    ) = _document_allocation_inputs(eligible)
    return _maximum_contiguous_hierarchical_capacity(
        eligible,
        positions_by_document,
        document_keys,
        document_normalized_weights,
    )


def sample_historical_posterior_curve_mixture(
    historical_lines: Sequence[Mapping[str, Any]],
    doses: Sequence[float],
    *,
    curve_count: int,
    seed: int = 0,
    allocation_strategy: AllocationStrategy = "largest_remainder",
) -> dict[str, Any]:
    """Return a similarity-weighted mixture of complete source posterior curves.

    Documents first receive quotas from their article similarity alone.  Each
    document then divides its fixed quota among eligible components according
    to component similarity times reliability.  Component count, filtering,
    and reliability therefore cannot inflate or reduce the document's total
    mixture share.  Only sources supporting the maximum requested dose are
    admitted; this fixed-intersection policy guarantees that every selected
    source supplies every column of a returned curve.

    ``largest_remainder`` turns normalized weights into exact integer quotas.
    For example, weights ``(0.2, 0.3, 0.5)`` and 100 curves yield
    ``(20, 30, 50)``.  ``multinomial`` instead samples the source counts from a
    multinomial distribution.  Both strategies use ``seed`` for reproducible
    posterior-index sampling, and the latter also uses it for allocation.
    """

    resolved_doses = _validate_doses(doses)
    if isinstance(curve_count, bool) or not isinstance(curve_count, int):
        raise ValueError("curve_count must be a positive integer.")
    if curve_count <= 0:
        raise ValueError("curve_count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if allocation_strategy not in {"largest_remainder", "multinomial"}:
        raise ValueError(
            "allocation_strategy must be 'largest_remainder' or 'multinomial'."
        )

    eligible, skipped, skipped_details = _prepare_eligible_sources(
        historical_lines, resolved_doses
    )

    if not eligible:
        raise ValueError(
            "No fitted positive-weight historical posterior source supports the "
            "complete requested dose range."
        )

    (
        positions_by_document,
        document_keys,
        document_raw_weights,
        document_normalized_weights,
        document_capacities,
    ) = _document_allocation_inputs(eligible)
    eligible_document_keys = set(document_keys)
    excluded_document_weights: dict[str, float] = {}
    for detail in skipped_details:
        document_key = str(
            detail.get("document_id") or detail.get("pmcid") or detail.get("source_id")
        )
        article_weight = _positive_finite_number(
            detail.get("article_similarity_weight")
        )
        if document_key not in eligible_document_keys and article_weight is not None:
            excluded_document_weights[document_key] = max(
                excluded_document_weights.get(document_key, 0.0), article_weight
            )
    excluded_document_weight = float(sum(excluded_document_weights.values()))
    total_document_weight_before_eligibility = float(
        np.sum(document_raw_weights) + excluded_document_weight
    )
    excluded_document_similarity_fraction = (
        excluded_document_weight / total_document_weight_before_eligibility
        if total_document_weight_before_eligibility > 0
        else 0.0
    )
    maximum_unique_draw_count = _maximum_contiguous_hierarchical_capacity(
        eligible,
        positions_by_document,
        document_keys,
        document_normalized_weights,
    )
    if (
        allocation_strategy == "largest_remainder"
        and curve_count > maximum_unique_draw_count
    ):
        raise ValueError(
            f"Requested {curve_count} unique mixture curves, but the quota-aware "
            f"no-replacement capacity is {maximum_unique_draw_count}. Request fewer "
            "curves or rebuild the limiting historical posterior caches with more draws."
        )

    rng = np.random.default_rng(seed)
    if allocation_strategy == "largest_remainder":
        (
            document_allocation,
            document_ideal,
            document_remainders,
            allocation,
            source_ideal_within_document,
            source_fractional_remainders,
        ) = _hierarchical_largest_remainder_plan(
            eligible,
            positions_by_document,
            document_keys,
            document_normalized_weights,
            curve_count,
        )
    else:
        document_allocation, document_ideal, document_remainders = (
            _capacity_constrained_multinomial(
                document_normalized_weights,
                curve_count,
                document_capacities,
                rng,
            )
        )

        allocation = np.zeros(len(eligible), dtype=int)
        source_ideal_within_document = np.zeros(len(eligible), dtype=float)
        source_fractional_remainders = np.zeros(len(eligible), dtype=float)
        for document_position, document_key in enumerate(document_keys):
            source_positions = positions_by_document[document_key]
            within_raw = np.asarray(
                [
                    eligible[position]["unnormalized_weight"]
                    for position in source_positions
                ],
                dtype=float,
            )
            within_weights = within_raw / float(np.sum(within_raw))
            within_capacities = np.asarray(
                [
                    int(eligible[position]["available_posterior_draw_count"])
                    for position in source_positions
                ],
                dtype=int,
            )
            document_quota = int(document_allocation[document_position])
            within_allocation, within_ideal, within_remainders = (
                _capacity_constrained_multinomial(
                    within_weights,
                    document_quota,
                    within_capacities,
                    rng,
                )
            )
            for local_position, source_position in enumerate(source_positions):
                allocation[source_position] = int(within_allocation[local_position])
                source_ideal_within_document[source_position] = float(
                    within_ideal[local_position]
                )
                source_fractional_remainders[source_position] = float(
                    within_remainders[local_position]
                )

    source_capacities = np.asarray(
        [int(source["available_posterior_draw_count"]) for source in eligible],
        dtype=int,
    )
    over_capacity = np.flatnonzero(allocation > source_capacities)
    if len(over_capacity):
        source_position = int(over_capacity[0])
        source = eligible[source_position]
        raise ValueError(
            "The exact similarity-weighted quota requires "
            f"{int(allocation[source_position])} unique curves from component "
            f"{source['identity']['source_id']}, but only "
            f"{int(source_capacities[source_position])} posterior draws are available. "
            "Reduce the requested curve count or rebuild that source cache; quotas "
            "are never reassigned to a less similar component."
        )

    normalized_weights = np.zeros(len(eligible), dtype=float)
    for document_position, document_key in enumerate(document_keys):
        positions = positions_by_document[document_key]
        within_raw = np.asarray(
            [eligible[position]["unnormalized_weight"] for position in positions],
            dtype=float,
        )
        within_normalized = within_raw / float(np.sum(within_raw))
        for local_position, source_position in enumerate(positions):
            normalized_weights[source_position] = float(
                document_normalized_weights[document_position]
                * within_normalized[local_position]
            )

    curve_blocks: list[np.ndarray] = []
    provenance_blocks: list[tuple[str, int]] = []
    source_allocations: list[dict[str, Any]] = []
    for source_position, (source, allocated_count) in enumerate(
        zip(eligible, allocation, strict=True)
    ):
        available = int(source["available_posterior_draw_count"])
        selected_indices = np.asarray([], dtype=int)
        uses_replacement = False
        if allocated_count > 0:
            selected_indices = rng.choice(
                available,
                size=int(allocated_count),
                replace=False,
            )
            line = source["line"]
            complete_source_draws = np.column_stack(
                [
                    posterior_probability_draws(dict(line["bayesian_fit"]), float(dose))
                    for dose in resolved_doses
                ]
            )
            curve_blocks.append(complete_source_draws[selected_indices, :])
            provenance_blocks.extend(
                (source["identity"]["source_id"], int(draw_index))
                for draw_index in selected_indices
            )
        source_allocations.append(
            {
                **source["identity"],
                "source_position": int(source_position),
                "similarity_weight_field": source["similarity_weight_field"],
                "similarity_weight": source["similarity_weight"],
                "reliability_weight": source["reliability_weight"],
                "document_similarity_weight": source["document_similarity_weight"],
                "unnormalized_weight": source["unnormalized_weight"],
                "normalized_weight": float(normalized_weights[source_position]),
                "ideal_curve_count_from_global_weight": float(
                    normalized_weights[source_position] * curve_count
                ),
                "ideal_curve_count_within_document_quota": float(
                    source_ideal_within_document[source_position]
                ),
                "fractional_remainder_within_document": float(
                    source_fractional_remainders[source_position]
                ),
                "allocated_curve_count": int(allocated_count),
                "realized_curve_fraction": float(allocated_count / curve_count),
                "allocation_rounding_error_from_global_ideal": float(
                    allocated_count - normalized_weights[source_position] * curve_count
                ),
                "available_posterior_draw_count": available,
                "sampled_with_replacement": uses_replacement,
                "sampled_posterior_draw_indices": selected_indices.tolist(),
                "support_max": source["support_max"],
            }
        )

    merged = np.vstack(curve_blocks)
    if len(merged) != curve_count or len(provenance_blocks) != curve_count:
        raise RuntimeError(
            "Posterior source allocation did not produce curve_count rows."
        )
    permutation = rng.permutation(curve_count)
    merged = merged[permutation, :]
    permuted_provenance = [provenance_blocks[index] for index in permutation]
    row_sources = [
        {
            "row_index": int(row_index),
            "source_id": source_id,
            "posterior_draw_index": int(draw_index),
        }
        for row_index, (source_id, draw_index) in enumerate(permuted_provenance)
    ]

    article_allocations: list[dict[str, Any]] = []
    for document_position, document_key in enumerate(document_keys):
        positions = positions_by_document[document_key]
        sources = [source_allocations[position] for position in positions]
        article_allocations.append(
            {
                "document_id": sources[0].get("document_id") or sources[0].get("pmcid"),
                "pmcid": sources[0].get("pmcid"),
                "eligible_source_count": len(sources),
                "allocated_source_count": sum(
                    int(source["allocated_curve_count"] > 0) for source in sources
                ),
                "unnormalized_weight": float(document_raw_weights[document_position]),
                "normalized_weight": float(
                    document_normalized_weights[document_position]
                ),
                "ideal_curve_count": float(document_ideal[document_position]),
                "fractional_remainder": float(document_remainders[document_position]),
                "allocated_curve_count": int(document_allocation[document_position]),
                "realized_curve_fraction": float(
                    document_allocation[document_position] / curve_count
                ),
                "allocation_rounding_error": float(
                    document_allocation[document_position]
                    - document_ideal[document_position]
                ),
            }
        )


    component_variance_by_dose = _component_variance_decomposition(
        merged, permuted_provenance, resolved_doses
    )

    return {
        "status": "fitted",
        "method": "similarity_weighted_article_component_posterior_curve_mixture",
        "model": MODEL_NAME,
        "doses": resolved_doses.copy(),
        "posterior_draws": merged,
        "posterior_draw_count": int(curve_count),
        "allocation_strategy": allocation_strategy,
        "allocation_tie_breaker": (
            "fractional_remainder_descending_then_stable_lexical_id"
            if allocation_strategy == "largest_remainder"
            else None
        ),
        "random_seed": int(seed),
        "weight_formula": (
            "document quota from article similarity; within-document component "
            "quota from component similarity * reliability"
        ),
        "contributor_policy": "fixed_intersection",
        "document_posterior_definition": (
            "mixture_over_eligible_study_unit_series_posteriors"
        ),
        "document_allocation_precedes_within_document_source_allocation": True,
        "maximum_unique_curve_count": int(maximum_unique_draw_count),
        "normalized_weight_sum": float(np.sum(normalized_weights)),
        "source_effective_count": float(1.0 / float(np.sum(normalized_weights**2))),
        "document_effective_count": float(
            1.0 / float(np.sum(document_normalized_weights**2))
        ),
        "maximum_document_weight": float(np.max(document_normalized_weights)),
        "document_weight_entropy": float(
            -np.sum(
                document_normalized_weights
                * np.log(np.clip(document_normalized_weights, 1e-300, None))
            )
        ),
        "zero_quota_source_count": int(np.sum(allocation == 0)),
        "component_variance_decomposition_by_dose": component_variance_by_dose,
        "source_allocations": source_allocations,
        "article_allocations": article_allocations,
        "row_sources": row_sources,
        "skipped_lines": skipped,
        "skipped_source_details": skipped_details,
        "excluded_document_similarity_fraction": float(
            excluded_document_similarity_fraction
        ),
    }


def source_mixture_draws_at_doses(
    method: Mapping[str, Any],
    doses: Sequence[float],
    *,
    curve_count: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Regenerate one auditable source-mixture pool from a fitted method."""

    ensure_current_prediction(method)
    if method.get("status") != "fitted":
        raise ValueError("A fitted article posterior draw-pool method is required.")
    posterior = method.get("posterior", {})
    resolved_count = int(
        curve_count
        if curve_count is not None
        else posterior.get("draw_count", DEFAULT_POSTERIOR_DRAW_COUNT)
    )
    resolved_seed = int(
        seed if seed is not None else posterior.get("draw_seed", DEFAULT_MIXTURE_SEED)
    )
    return sample_historical_posterior_curve_mixture(
        list(method.get("historical_lines") or []),
        doses,
        curve_count=resolved_count,
        seed=resolved_seed,
    )


def estimate_article_posterior_draw_pool(
    historical_lines: Sequence[Mapping[str, Any]],
    spec: ToxicityQuery,
    *,
    bayesian_prior: str,
    draw_count: int = DEFAULT_POSTERIOR_DRAW_COUNT,
    seed: int = DEFAULT_MIXTURE_SEED,
) -> dict[str, Any]:
    """Build the displayable posterior curve induced by source mixture sampling."""

    available = maximum_unique_source_mixture_draw_count(
        historical_lines, spec.planned_doses
    )
    if available <= 0:
        return {
            "status": "no_eligible_historical_posterior_sources",
            "model": MODEL_NAME,
            "curve_points": [],
            "historical_lines": [],
            "skipped_reasons": {"no_source_supports_complete_planned_dose_range": 1},
        }
    resolved_count = min(int(draw_count), available)
    try:
        sampled = sample_historical_posterior_curve_mixture(
            historical_lines,
            spec.planned_doses,
            curve_count=resolved_count,
            seed=seed,
        )
    except ValueError as error:
        return {
            "status": "article_posterior_draw_pool_not_fit_eligible",
            "model": MODEL_NAME,
            "curve_points": [],
            "historical_lines": [],
            "skipped_reasons": {str(error): 1},
        }
    draws = np.asarray(sampled["posterior_draws"], dtype=float)
    eligible_lines = [
        dict(historical_lines[int(row["original_source_index"])])
        for row in sampled["source_allocations"]
    ]
    line_by_source_id = {
        str(row["source_id"]): line
        for row, line in zip(sampled["source_allocations"], eligible_lines, strict=True)
    }
    curve_points = []
    for column, dose in enumerate(spec.planned_doses):
        column_draws = draws[:, column]
        extrapolated_rows: list[Mapping[str, Any]] = []
        for row in sampled["source_allocations"]:
            line = line_by_source_id[str(row["source_id"])]
            observed_min = _positive_finite_number(line.get("observed_dose_min"))
            observed_max = _positive_finite_number(line.get("observed_dose_max"))
            if (observed_min is not None and float(dose) < observed_min) or (
                observed_max is not None and float(dose) > observed_max
            ):
                extrapolated_rows.append(row)
        realized_extrapolation_fraction = sum(
            float(row["realized_curve_fraction"]) for row in extrapolated_rows
        )
        theoretical_extrapolation_fraction = sum(
            float(row["normalized_weight"]) for row in extrapolated_rows
        )
        curve_points.append(
            {
                "dose": float(dose),
                "unit": spec.target_unit,
                "dlt_probability": float(np.mean(column_draws)),
                "posterior_credible_interval_95": {
                    "low": float(np.quantile(column_draws, 0.025)),
                    "high": float(np.quantile(column_draws, 0.975)),
                },
                "extrapolation_weight_fraction": float(realized_extrapolation_fraction),
                "theoretical_extrapolation_weight_fraction": float(
                    theoretical_extrapolation_fraction
                ),
                "contributor_count": sum(
                    int(row["allocated_curve_count"] > 0)
                    for row in sampled["source_allocations"]
                ),
            }
        )
    return {
        "status": "fitted",
        "model": MODEL_NAME,
        "model_assumption": (
            "one_complete_historical_component_curve_per_posterior_draw"
        ),
        "interpretation": (
            "relevance_weighted_finite_mixture_not_joint_likelihood_borrowing"
        ),
        "bayesian_prior": bayesian_prior,
        "trial_type": spec.trial_type,
        "curve_points": curve_points,
        "historical_lines": eligible_lines,
        "posterior": {
            "draw_count": int(resolved_count),
            "maximum_unique_draw_count": int(available),
            "draw_seed": int(seed),
            "allocation_strategy": "largest_remainder",
            "allocation_tie_breaker": sampled["allocation_tie_breaker"],
        },
        "mixture_diagnostics": {
            "article_allocations": sampled["article_allocations"],
            "source_allocations": sampled["source_allocations"],
            "row_sources": sampled["row_sources"],
            "source_effective_count": sampled["source_effective_count"],
            "document_effective_count": sampled["document_effective_count"],
            "maximum_document_weight": sampled["maximum_document_weight"],
            "document_weight_entropy": sampled["document_weight_entropy"],
            "zero_quota_source_count": sampled["zero_quota_source_count"],
            "excluded_document_similarity_fraction": sampled[
                "excluded_document_similarity_fraction"
            ],
            "component_variance_decomposition_by_dose": sampled[
                "component_variance_decomposition_by_dose"
            ],
            "weight_formula": sampled["weight_formula"],
            "document_posterior_definition": sampled["document_posterior_definition"],
            "skipped_lines": sampled["skipped_lines"],
            "skipped_source_details": sampled["skipped_source_details"],
        },
    }


def source_mixture_probability_summary(
    method: Mapping[str, Any], dose: float
) -> dict[str, Any]:
    """Evaluate the fitted source-mixture posterior at one dose."""

    sampled = source_mixture_draws_at_doses(method, [float(dose)])
    draws = np.asarray(sampled["posterior_draws"], dtype=float)[:, 0]
    return {
        "status": "fitted",
        "method": "similarity_weighted_article_component_posterior_curve_mixture",
        "dose": float(dose),
        "dlt_probability": float(np.mean(draws)),
        "posterior_credible_interval_95": {
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
        },
        "posterior_draw_count": int(len(draws)),
        "article_allocations": sampled["article_allocations"],
        "source_allocations": sampled["source_allocations"],
    }
