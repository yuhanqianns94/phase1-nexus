"""Pure helpers for automatic or human-reviewed historical-article inclusion.

Similarity remains the source of the numerical borrowing weights.  Manual
review only supplies an article-level inclusion mask; downstream curve methods
continue to apply their existing, method-specific normalization and ESS rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


AUTOMATIC_SELECTION_MODE = "automatic"
MANUAL_SELECTION_MODE = "manual"
THRESHOLD_CANDIDATE_BASIS = "threshold"
TOP_N_CANDIDATE_BASIS = "top_n"
DEFAULT_HISTORICAL_ARTICLE_TOP_N = 40

_VALID_SELECTION_MODES = {
    AUTOMATIC_SELECTION_MODE,
    MANUAL_SELECTION_MODE,
}
_VALID_CANDIDATE_BASES = {
    THRESHOLD_CANDIDATE_BASIS,
    TOP_N_CANDIDATE_BASIS,
}


def _document_id(row: dict[str, Any]) -> str:
    value = row.get("document_id") or row.get("pmcid")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A scored historical row is missing its document identifier.")
    return value.split("::", 1)[0].strip()


def _similarity(row: dict[str, Any]) -> float:
    value = row.get("similarity")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A scored historical row has a non-numeric similarity.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("A scored historical row has a non-finite similarity.")
    return result


def _path_string(value: Any) -> str | None:
    if isinstance(value, (str, Path)):
        result = str(value)
        return result or None
    return None


def _validate_threshold(value: float | int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("similarity_threshold must be a number between 0 and 1.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("similarity_threshold must be a number between 0 and 1.")
    return result


def _validate_top_n(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("historical_article_top_n must be a positive integer.")
    return value


def _validate_basis(value: str) -> str:
    if value not in _VALID_CANDIDATE_BASES:
        raise ValueError(
            "historical_article_candidate_basis must be 'threshold' or 'top_n'."
        )
    return value


def _validate_mode(value: str) -> str:
    if value not in _VALID_SELECTION_MODES:
        raise ValueError(
            "historical_article_selection_mode must be 'automatic' or 'manual'."
        )
    return value


def rank_historical_article_candidates(
    scored_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group study-unit rows by article and rank articles by maximum similarity.

    Legacy libraries have one row per article.  Strict-v2 libraries can have
    multiple study-unit rows per article, so the most similar study unit defines
    the article's review rank while every unit remains visible in the payload.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in scored_rows:
        groups.setdefault(_document_id(row), []).append(row)

    ranked_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for document_id, rows in groups.items():
        rows.sort(
            key=lambda item: (
                -_similarity(item),
                str(item.get("study_unit_id") or ""),
                str(item.get("scoring_id") or ""),
            )
        )
        ranked_groups.append((document_id, rows))
    ranked_groups.sort(key=lambda item: (-_similarity(item[1][0]), item[0]))

    result: list[dict[str, Any]] = []
    for rank, (document_id, rows) in enumerate(ranked_groups, 1):
        representative = rows[0]
        unit_rows = []
        for row in rows:
            field_scores = row.get("field_scores")
            unit_rows.append(
                {
                    "study_unit_id": row.get("study_unit_id"),
                    "scoring_id": row.get("scoring_id"),
                    "similarity": _similarity(row),
                    "available_weight": float(row.get("available_weight", 0.0)),
                    "path": _path_string(row.get("path")),
                    "field_scores": (
                        dict(field_scores) if isinstance(field_scores, dict) else {}
                    ),
                }
            )
        field_scores = representative.get("field_scores")
        result.append(
            {
                "rank": rank,
                "pmcid": document_id,
                "document_id": document_id,
                "similarity": _similarity(representative),
                "system_similarity_weight": _similarity(representative),
                "comparison_groups": representative.get("groups", {}),
                "configuration_scores": representative.get("scope_scores", {}),
                "configuration_keys": representative.get("data", {}).get("_v3_scope_map", {}),
                "available_weight": float(
                    representative.get("available_weight", 0.0)
                ),
                "path": _path_string(representative.get("path")),
                "extraction_paths": sorted(
                    {
                        path
                        for path in (_path_string(row.get("path")) for row in rows)
                        if path is not None
                    }
                ),
                "field_scores": (
                    dict(field_scores) if isinstance(field_scores, dict) else {}
                ),
                "study_unit_count": len(rows),
                "study_units": unit_rows,
            }
        )
    return result


def historical_article_candidates(
    scored_rows: Sequence[dict[str, Any]],
    *,
    similarity_threshold: float = 0.0,
    candidate_basis: str = THRESHOLD_CANDIDATE_BASIS,
    top_n: int = DEFAULT_HISTORICAL_ARTICLE_TOP_N,
) -> list[dict[str, Any]]:
    """Return the bounded, document-level universe offered for human review."""

    threshold = _validate_threshold(similarity_threshold)
    basis = _validate_basis(candidate_basis)
    resolved_top_n = _validate_top_n(top_n)
    ranked = rank_historical_article_candidates(scored_rows)
    if basis == THRESHOLD_CANDIDATE_BASIS:
        candidates = [row for row in ranked if row["similarity"] >= threshold]
    else:
        candidates = ranked[:resolved_top_n]

    positive_total = sum(max(0.0, row["similarity"]) for row in candidates)
    output: list[dict[str, Any]] = []
    for row in candidates:
        normalized_weight = (
            max(0.0, row["similarity"]) / positive_total
            if positive_total > 0
            else 0.0
        )
        output.append(
            {
                **row,
                "normalized_candidate_weight": normalized_weight,
                "candidate_normalized_similarity_weight": normalized_weight,
            }
        )
    return output


def _candidate_snapshot_sha256(
    candidates: Sequence[dict[str, Any]],
    *,
    candidate_basis: str,
    similarity_threshold: float,
    top_n: int,
) -> str:
    snapshot_payload = {
        "candidate_basis": candidate_basis,
        "similarity_threshold": similarity_threshold,
        "top_n": top_n,
        "candidates": [
            {
                "document_id": candidate["document_id"],
                "similarity": candidate["similarity"],
                "study_units": [
                    {
                        "study_unit_id": unit["study_unit_id"],
                        "similarity": unit["similarity"],
                    }
                    for unit in candidate["study_units"]
                ],
            }
            for candidate in candidates
        ],
    }
    return hashlib.sha256(
        json.dumps(
            snapshot_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def preview_historical_article_selection(
    scored_rows: Sequence[dict[str, Any]],
    *,
    similarity_threshold: float = 0.0,
    candidate_basis: str = THRESHOLD_CANDIDATE_BASIS,
    top_n: int = DEFAULT_HISTORICAL_ARTICLE_TOP_N,
) -> dict[str, Any]:
    """Build a serializable candidate payload without fitting a curve."""

    threshold = _validate_threshold(similarity_threshold)
    basis = _validate_basis(candidate_basis)
    resolved_top_n = _validate_top_n(top_n)
    candidates = historical_article_candidates(
        scored_rows,
        similarity_threshold=threshold,
        candidate_basis=basis,
        top_n=resolved_top_n,
    )
    snapshot_sha256 = _candidate_snapshot_sha256(
        candidates,
        candidate_basis=basis,
        similarity_threshold=threshold,
        top_n=resolved_top_n,
    )
    return {
        "candidate_basis": basis,
        "similarity_threshold": threshold,
        "threshold": threshold,
        "top_n": resolved_top_n,
        "candidate_snapshot_sha256": snapshot_sha256,
        "scored_row_count": len(scored_rows),
        "scored_document_count": len(rank_historical_article_candidates(scored_rows)),
        "candidate_document_count": len(candidates),
        "candidate_study_unit_count": sum(
            int(candidate["study_unit_count"]) for candidate in candidates
        ),
        "candidates": candidates,
    }


def retain_curve_eligible_preview_candidates(
    preview: dict[str, Any],
    historical_lines: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Annotate and retain candidates with at least one usable posterior line.

    The ranked snapshot is intentionally left unchanged: it represents the
    original threshold/top-N similarity shortlist that the final estimator
    recomputes for stale-state validation.  The visible candidates are the
    subset that can actually contribute to the requested drug and dose unit.
    """

    line_count_by_document: dict[str, int] = {}
    point_count_by_document: dict[str, int] = {}
    dose_min_by_document: dict[str, float] = {}
    dose_max_by_document: dict[str, float] = {}
    for line in historical_lines:
        document_id = _document_id(line)
        line_count_by_document[document_id] = (
            line_count_by_document.get(document_id, 0) + 1
        )
        points = line.get("source_points")
        if isinstance(points, (list, tuple)):
            point_count_by_document[document_id] = (
                point_count_by_document.get(document_id, 0) + len(points)
            )
        for field, target, reducer in (
            ("observed_dose_min", dose_min_by_document, min),
            ("observed_dose_max", dose_max_by_document, max),
        ):
            value = line.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            previous = target.get(document_id)
            target[document_id] = numeric if previous is None else reducer(
                previous, numeric
            )

    ranked = preview.get("candidates")
    if not isinstance(ranked, list):
        ranked = []
    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for raw_candidate in ranked:
        if not isinstance(raw_candidate, dict):
            continue
        candidate = dict(raw_candidate)
        document_id = _document_id(candidate)
        line_count = line_count_by_document.get(document_id, 0)
        candidate.update(
            {
                "curve_eligible": line_count > 0,
                "curve_line_count": line_count,
                "curve_source_point_count": point_count_by_document.get(
                    document_id, 0
                ),
                "curve_observed_dose_min": dose_min_by_document.get(document_id),
                "curve_observed_dose_max": dose_max_by_document.get(document_id),
                "curve_ineligibility_reason": (
                    None
                    if line_count > 0
                    else (
                        "No fitted DLT posterior line matches the target drug and "
                        "canonical dose unit."
                    )
                ),
            }
        )
        if line_count > 0:
            eligible.append(candidate)
        else:
            ineligible.append(
                {
                    "document_id": document_id,
                    "rank": candidate.get("rank"),
                    "similarity": candidate.get("similarity"),
                    "reason": candidate["curve_ineligibility_reason"],
                }
            )

    positive_total = sum(
        max(0.0, float(candidate.get("similarity") or 0.0))
        for candidate in eligible
    )
    for candidate in eligible:
        weight = max(0.0, float(candidate.get("similarity") or 0.0))
        normalized = weight / positive_total if positive_total > 0 else 0.0
        candidate["normalized_candidate_weight"] = normalized
        candidate["candidate_normalized_similarity_weight"] = normalized

    return {
        **preview,
        "candidate_document_count_before_curve_eligibility": len(ranked),
        "candidate_study_unit_count_before_curve_eligibility": preview.get(
            "candidate_study_unit_count", 0
        ),
        "candidate_document_count": len(eligible),
        "candidate_study_unit_count": sum(
            int(candidate.get("study_unit_count") or 0)
            for candidate in eligible
        ),
        "curve_ineligible_candidate_count": len(ineligible),
        "curve_ineligible_candidates": ineligible,
        "candidate_curve_eligibility_policy": (
            "at_least_one_fitted_dlt_posterior_line_matching_target_drug_and_unit"
        ),
        "candidates": eligible,
    }


def _unique_selection_ids(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Selected historical article IDs must be non-empty strings.")
        document_id = value.split("::", 1)[0].strip()
        if document_id not in seen:
            seen.add(document_id)
            result.append(document_id)
    return result


def apply_historical_article_selection(
    scored_rows: Sequence[dict[str, Any]],
    *,
    similarity_threshold: float = 0.0,
    selection_mode: str = AUTOMATIC_SELECTION_MODE,
    candidate_basis: str = THRESHOLD_CANDIDATE_BASIS,
    top_n: int = DEFAULT_HISTORICAL_ARTICLE_TOP_N,
    selected_document_ids: Iterable[str] | None = None,
    candidate_snapshot_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply automatic thresholding or a validated human inclusion mask.

    In threshold-review mode, a selected v2 article contributes only study-unit
    rows that independently meet the threshold.  In top-N review mode, all
    study-unit rows from a selected candidate article are retained because top-N
    is explicitly an alternative to threshold eligibility.
    """

    mode = _validate_mode(selection_mode)
    threshold = _validate_threshold(similarity_threshold)
    basis = _validate_basis(candidate_basis)
    resolved_top_n = _validate_top_n(top_n)
    if mode == AUTOMATIC_SELECTION_MODE:
        effective_basis = THRESHOLD_CANDIDATE_BASIS
    else:
        effective_basis = basis

    candidates = historical_article_candidates(
        scored_rows,
        similarity_threshold=threshold,
        candidate_basis=effective_basis,
        top_n=resolved_top_n,
    )
    candidate_ids = [str(candidate["document_id"]) for candidate in candidates]
    candidate_set = set(candidate_ids)
    current_snapshot_sha256 = _candidate_snapshot_sha256(
        candidates,
        candidate_basis=effective_basis,
        similarity_threshold=threshold,
        top_n=resolved_top_n,
    )

    if mode == AUTOMATIC_SELECTION_MODE:
        selected_ids = candidate_ids
    else:
        if candidate_snapshot_sha256 is not None:
            if (
                not isinstance(candidate_snapshot_sha256, str)
                or candidate_snapshot_sha256 != current_snapshot_sha256
            ):
                raise ValueError(
                    "The historical-article candidate snapshot is stale. Refresh "
                    "the ranked candidates before generating the curve."
                )
        requested = _unique_selection_ids(selected_document_ids)
        if not requested:
            raise ValueError(
                "Manual historical-article selection requires at least one article."
            )
        unknown = sorted(set(requested) - candidate_set)
        if unknown:
            raise ValueError(
                "Selected historical articles are outside the current candidate "
                f"universe: {', '.join(unknown)}"
            )
        requested_set = set(requested)
        selected_ids = [item for item in candidate_ids if item in requested_set]

    selected_set = set(selected_ids)
    if effective_basis == THRESHOLD_CANDIDATE_BASIS:
        retained = [
            row
            for row in scored_rows
            if _document_id(row) in selected_set and _similarity(row) >= threshold
        ]
    else:
        retained = [row for row in scored_rows if _document_id(row) in selected_set]
    excluded_ids = [item for item in candidate_ids if item not in selected_set]
    audit = {
        "mode": mode,
        "candidate_basis": effective_basis,
        "similarity_threshold": threshold,
        "top_n": resolved_top_n,
        "scored_row_count_before_selection": len(scored_rows),
        "scored_document_count_before_selection": len(
            rank_historical_article_candidates(scored_rows)
        ),
        "candidate_document_count": len(candidate_ids),
        "selected_document_count": len(selected_ids),
        "excluded_candidate_document_count": len(excluded_ids),
        "retained_row_count": len(retained),
        "candidate_document_ids": candidate_ids,
        "candidate_snapshot_sha256": current_snapshot_sha256,
        "candidates": [
            {
                "document_id": candidate["document_id"],
                "rank": candidate["rank"],
                "similarity": candidate["similarity"],
                "available_weight": candidate["available_weight"],
            }
            for candidate in candidates
        ],
        "selected_document_ids": selected_ids,
        "excluded_candidate_document_ids": excluded_ids,
        "selection_is_article_level_inclusion_mask": True,
        "system_similarity_weights_preserved": True,
        "method_specific_weight_renormalization_after_selection": True,
    }
    return retained, audit
