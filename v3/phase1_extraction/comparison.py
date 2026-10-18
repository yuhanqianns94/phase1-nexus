"""Extraction-only comparison for synchronous versus Batch pilot runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .artifacts import ArticlePaths, read_json, write_json
from .friendly_review import _build_text_units
from .models import ReportedExtraction
from .exposures import exposure_identity


_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*%|\s*/\s*\d+)?")


def _clean(text: str | None) -> str:
    value = " ".join((text or "").casefold().split())
    return (
        value.replace("mg/m²", "mg/m2")
        .replace("mg/m^2", "mg/m2")
        .replace("every 2 weeks", "q2w")
        .replace("every two weeks", "q2w")
        .replace("every 4 weeks", "q4w")
        .replace("every four weeks", "q4w")
        .replace("≥", ">=")
        .replace("≤", "<=")
    )


def _numbers(text: str | None) -> list[str]:
    return sorted(" ".join(match.group(0).split()) for match in _NUMBER.finditer(text or ""))


def _text_by_module(extraction: ReportedExtraction) -> dict[str, str]:
    return {unit.module_id: unit.predicted_answer or "" for unit in _build_text_units(extraction)}


def _dose_signature(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return (
        value.value_kind,
        value.value,
        value.range_min,
        value.range_max,
        tuple(value.discrete_values),
        _clean(value.unit_normalized or value.unit_raw),
    )


def _semantic_maps(extraction: ReportedExtraction) -> tuple[dict[str, tuple[Any, ...]], dict[str, tuple[Any, ...]]]:
    interventions = {item.intervention_id: _clean(item.name_raw) for item in extraction.interventions}
    regimens: dict[str, tuple[Any, ...]] = {}
    for regimen in extraction.regimens:
        components = []
        for component in regimen.components:
            components.append(
                (
                    interventions.get(component.intervention_id, f"<missing:{component.intervention_id}>"),
                    component.role,
                    tuple(
                        sorted(
                            [
                                (
                                _dose_signature(phase.dose),
                                _clean(phase.frequency_raw),
                                _clean(phase.cycle_length_raw),
                                )
                                for phase in component.dosing_phases
                            ],
                            key=repr,
                        )
                    ),
                )
            )
        regimens[regimen.regimen_id] = (regimen.composition, tuple(sorted(components, key=repr)))
    series = {
        item.series_id: (
            regimens.get(item.regimen_id, (f"<missing:{item.regimen_id}>",)),
            _clean(item.comparable_unit),
        )
        for item in extraction.dose_series
    }
    cohorts = {
        item.cohort_id: (
            series.get(item.series_id, (f"<missing:{item.series_id}>",)),
            _dose_signature(item.dose),
            item.actual_status,
            exposure_identity(item.exposures),
        )
        for item in extraction.cohorts
    }
    return series, cohorts


def _curve_signature(paths: ArticlePaths, extraction: ReportedExtraction) -> list[dict[str, Any]]:
    if not paths.curve_rows.is_file():
        return []
    payload = read_json(paths.curve_rows)
    rows = payload.get("all_rows", []) if isinstance(payload, dict) else payload
    return sorted(
        [
            {
                "dose_value": row.get("dose_value"),
                "dose_unit": _clean(row.get("dose_unit")),
                "n": row.get("n"),
                "y": row.get("y"),
                "included": row.get("included_in_default"),
                "numerator_basis": row.get("numerator_basis"),
                "denominator_basis": row.get("denominator_basis"),
                "exposures": exposure_identity(row.get("exposures", [])),
            }
            for row in rows
        ],
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
    )


def _reported_numeric_signature(extraction: ReportedExtraction) -> dict[str, list[Any]]:
    series, cohorts = _semantic_maps(extraction)
    scopes = {
        item.result_scope_id: (
            series.get(item.series_id, (f"<missing:{item.series_id}>",)),
            tuple(sorted(
                (cohorts.get(cohort_id, (f"<missing:{cohort_id}>",)) for cohort_id in item.cohort_ids),
                key=repr,
            )),
        )
        for item in extraction.result_scopes
    }
    return {
        "population_age": sorted(
            [(item.age_min, item.age_min_comparator, item.age_max, item.age_max_comparator, item.age_unit)
             for item in extraction.populations], key=repr,
        ),
        "cohort_doses": sorted(
            [(
                item.dose.value_kind,
                item.dose.value,
                item.dose.range_min,
                item.dose.range_max,
                tuple(item.dose.discrete_values),
                item.dose.unit_normalized or item.dose.unit_raw,
            )
            for item in extraction.cohorts
            if item.actual_status == "actual"], key=repr,
        ),
        "population_counts": sorted(
            [
                (
                    item.role,
                    item.reported_value,
                    item.reporting_status,
                    cohorts.get(item.cohort_id) if item.cohort_id else None,
                    scopes.get(item.result_scope_id) if item.result_scope_id else None,
                )
                for item in extraction.population_counts
            ],
            key=repr,
        ),
        "dlt_counts": sorted(
            [
                (
                    item.count_kind,
                    item.reported_count,
                    item.explicit_patient_count,
                    item.reporting_status,
                    cohorts.get(item.cohort_id) if item.cohort_id else None,
                    scopes.get(item.result_scope_id) if item.result_scope_id else None,
                )
                for item in extraction.dlt_observations
            ],
            key=repr,
        ),
        "design_numbers": sorted(
            [
                (
                    item.normalized_type,
                    item.initial_cohort_n,
                    item.added_cohort_n,
                    item.expanded_total_n,
                    item.replacement_count,
                    item.nonevaluable_count,
                    item.cohort_addition_count,
                    tuple(item.expand_when_y_equals),
                    item.stop_when_y_at_least,
                )
                for item in extraction.designs
            ],
            key=repr,
        ),
        "dlt_window_days": sorted((item.normalized_days for item in extraction.dlt_windows), key=repr),
        "result_scope_structure": sorted(scopes.values(), key=repr),
    }


def compare_transport_outputs(sync_paths: ArticlePaths, batch_paths: ArticlePaths) -> dict[str, Any]:
    sync = ReportedExtraction.model_validate(read_json(sync_paths.reported_extraction))
    batch = ReportedExtraction.model_validate(read_json(batch_paths.reported_extraction))
    if sync.document_id != batch.document_id:
        raise ValueError("Cannot compare different documents")
    differences: list[dict[str, Any]] = []
    sync_numeric = _reported_numeric_signature(sync)
    batch_numeric = _reported_numeric_signature(batch)
    for category in sync_numeric:
        if sync_numeric[category] != batch_numeric[category]:
            differences.append(
                {
                    "kind": "numeric_or_relational",
                    "category": category,
                    "synchronous": sync_numeric[category],
                    "batch": batch_numeric[category],
                }
            )
    sync_curve = _curve_signature(sync_paths, sync)
    batch_curve = _curve_signature(batch_paths, batch)
    if sync_curve != batch_curve:
        differences.append(
            {
                "kind": "resolved_curve",
                "category": "dose_n_dlt_curve_rows",
                "synchronous": sync_curve,
                "batch": batch_curve,
            }
        )
    sync_text = _text_by_module(sync)
    batch_text = _text_by_module(batch)
    for module_id in sorted(set(sync_text) | set(batch_text)):
        left, right = sync_text.get(module_id, ""), batch_text.get(module_id, "")
        if _clean(left) == _clean(right):
            continue
        ratio = SequenceMatcher(None, _clean(left), _clean(right)).ratio()
        left_numbers, right_numbers = _numbers(left), _numbers(right)
        if left_numbers != right_numbers or ratio < 0.65:
            differences.append(
                {
                    "kind": "major_metric_text",
                    "category": module_id,
                    "similarity": ratio,
                    "numbers_synchronous": left_numbers,
                    "numbers_batch": right_numbers,
                    "synchronous": left,
                    "batch": right,
                }
            )
    return {
        "format": "phase1_sync_batch_extraction_comparison_v3.1",
        "document_id": sync.document_id,
        "comparison_scope": "extraction outputs only; source documents were not used for adjudication",
        "substantive_candidate_count": len(differences),
        "differences": differences,
    }


def compare_pilot_roots(sync_root: Path, batch_root: Path, output_path: Path) -> dict[str, Any]:
    document_ids = sorted(
        path.parent.name for path in Path(sync_root).glob("*/reported_extraction.json")
        if (Path(batch_root) / path.parent.name / "reported_extraction.json").is_file()
    )
    articles = [
        compare_transport_outputs(
            ArticlePaths(Path(sync_root), document_id),
            ArticlePaths(Path(batch_root), document_id),
        )
        for document_id in document_ids
    ]
    payload = {
        "format": "phase1_sync_batch_pilot_comparison_v3.1",
        "article_count": len(articles),
        "articles": articles,
    }
    write_json(output_path, payload)
    return payload
