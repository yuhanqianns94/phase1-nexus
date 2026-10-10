#!/usr/bin/env python3
"""Shared pre-trial similarity and dose-unit utilities for toxicity curves."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import requests

from .dose_unit_conversion import (
    canonical_dose_unit,
    canonical_dose_value,
    normalize_dose_unit_text,
    parse_dose_unit,
    parse_scaled_unit,
    units_match,
)


DEFAULT_EXTRACTION_DIR = Path("output/mtd_llm_extractions")
DEFAULT_EMBEDDING_CACHE = Path("output/mtd_similarity/paragraph_embedding_cache.json")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MANUAL_SIMILARITY_PARAMETERS = Path(__file__).with_name(
    "manual_similarity_parameters.json"
)

NON_EXTRACTION_SUFFIXES = (
    ".validation.json",
    ".run_metadata.json",
    ".raw_response.json",
    ".invalid.json",
    ".values.json",
    ".standardized.json",
)
MISSING_VALUES = {"", "not_reported", "unclear", "not_applicable", "unknown", None}
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._/#-]*")
DEFAULT_DLT_EVA_CYCLE_LENGTH_DAYS = 21.0

# Similarity field weights. dose_grid and design are excluded from scoring.
# Remaining weights are the old relative weights renormalized to sum to 1:
#   new_w = old_w / (1 - 0.16 - 0.07)
EXCLUDED_SIMILARITY_FIELDS = ("dose_grid", "design")
FIELD_WEIGHTS = {
    "drug": 18 / 82,
    "cancer_type": 11 / 82,
    "route": 7 / 82,
    "schedule": 6 / 82,
    "schema_embedding": 6 / 82,
    "dlt_definition": 8 / 82,
    "dlt_eva": 6 / 82,
    "target_moa": 6 / 82,
    "age": 4 / 82,
    "trial_type": 4 / 82,
    "eligibility_embedding": 4 / 82,
    "gender": 1 / 82,
    "dose_form": 1 / 82,
}


def normalize_field_weights(
    field_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Validate and normalize tunable similarity-field weights to sum to one."""

    from ..profile_similarity import LABELS, normalize_weights as normalize_profile_weights
    if field_weights and set(field_weights) <= set(LABELS):
        return normalize_profile_weights(field_weights)
    if field_weights is None:
        return dict(FIELD_WEIGHTS)
    unknown = set(field_weights) - set(FIELD_WEIGHTS)
    if unknown:
        raise ValueError(
            "Unknown similarity weight fields: " + ", ".join(sorted(unknown))
        )
    normalized: dict[str, float] = {}
    for field_name in FIELD_WEIGHTS:
        value = field_weights.get(field_name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Similarity weight for {field_name} must be a finite number."
            )
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Similarity weight for {field_name} must be non-negative and finite."
            )
        normalized[field_name] = value
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("At least one similarity field weight must be positive.")
    return {
        field_name: value / total for field_name, value in normalized.items()
    }


def validate_similarity_threshold(value: float | int | None) -> float:
    """Return an inclusive raw-similarity threshold in the closed interval [0, 1]."""

    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("similarity_threshold must be a number from 0 to 1.")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("similarity_threshold must be a number from 0 to 1.")
    return threshold


def load_similarity_parameters(
    path: Path,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Load a persisted field-weight/threshold configuration."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load similarity parameters from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Similarity parameter file {path} must contain a JSON object.")
    raw_weights = payload.get("field_weights")
    if not isinstance(raw_weights, dict):
        raise ValueError(f"Similarity parameter file {path} has no field_weights object.")
    return (
        normalize_field_weights(raw_weights),
        validate_similarity_threshold(payload.get("similarity_threshold")),
        payload,
    )


def scoring_library_fingerprint(records: list[dict[str, Any]]) -> str:
    """Fingerprint all extraction content that can influence similarity scoring."""

    payload = [
        {
            "pmcid": record.get("pmcid"),
            "data": record.get("data"),
        }
        for record in sorted(records, key=lambda record: str(record.get("pmcid", "")))
    ]
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def resolve_similarity_parameters(
    *,
    path: Path | None = None,
    use_manual_default: bool = True,
    threshold_override: float | int | None = None,
) -> tuple[dict[str, float], float, str]:
    """Resolve parameters with explicit, manual, then built-in precedence."""

    selected_path = path
    if (
        selected_path is None
        and use_manual_default
        and DEFAULT_MANUAL_SIMILARITY_PARAMETERS.exists()
    ):
        selected_path = DEFAULT_MANUAL_SIMILARITY_PARAMETERS
    if selected_path is None:
        field_weights = normalize_field_weights()
        threshold = 0.0
        source = "built_in_defaults"
    else:
        field_weights, threshold, payload = load_similarity_parameters(selected_path)
        parameter_format = payload.get("format")
        if isinstance(parameter_format, str) and parameter_format.startswith(
            "mtd_similarity_optimized_parameters_"
        ):
            raise ValueError(
                "Legacy optimized similarity parameters are retired. Use a "
                "manual parameter file instead."
            )
        source = str(selected_path)
    if threshold_override is not None:
        threshold = validate_similarity_threshold(threshold_override)
        source = f"{source}; threshold overridden at CLI"
    return field_weights, threshold, source


PARAGRAPH_FIELDS = {
    "schema_embedding": ["schema"],
    "eligibility_embedding": ["Trial_Inclusion", "Trial_Exclusion"],
}

OUTCOME_OR_POST_TRIAL_FIELDS = {
    "MTD_Value",
    "MTD_Unit",
    "MTD_regimen_description",
    "MTD_components",
    "MTD_missing_status",
    "MAD_Value",
    "MAD_Unit",
    "MAD_regimen_description",
    "MAD_components",
    "RP2D",
    "dlt_count_at_each_dose_level",
}


def is_extraction_json(path: Path) -> bool:
    return (
        path.suffix == ".json"
        and path.name != "mtd_extraction_state.json"
        and not any(path.name.endswith(suffix) for suffix in NON_EXTRACTION_SUFFIXES)
    )


def pmcid_from_path(path: Path) -> str:
    name = path.name
    for suffix in ("_gpt_5_5_compact.json", "_compact.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def load_extractions(extraction_dir: Path, pattern: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths_by_pmcid: dict[str, Path] = {}
    for path in sorted(extraction_dir.glob(pattern)):
        if not is_extraction_json(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("skipped_extraction"):
            continue
        pmcid = pmcid_from_path(path)
        previous = paths_by_pmcid.get(pmcid)
        if previous is not None:
            raise ValueError(
                f"Duplicate PMCID {pmcid!r} in extraction library: "
                f"{previous} and {path}. Remove or rename one duplicate before "
                "scoring, fitting, or validation."
            )
        paths_by_pmcid[pmcid] = path
        records.append({"pmcid": pmcid, "path": str(path), "data": payload})
    return records


def pretrial_query_only(query: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Remove outcome/post-trial fields so users cannot leak MTD evidence."""

    cleaned = dict(query)
    ignored = []
    for field_name in sorted(OUTCOME_OR_POST_TRIAL_FIELDS):
        if field_name in cleaned:
            ignored.append(field_name)
            cleaned.pop(field_name, None)
    return cleaned, ignored


def build_unit_only_query(source: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Build a user-style pre-trial query that only keeps the target dose unit."""

    target_unit = query_target_unit(source)
    query, ignored = pretrial_query_only(source)
    query.pop("dose_levels", None)
    if target_unit:
        query["Dose_Unit"] = target_unit
    return query, ignored


def is_wrapped_field(value: Any) -> bool:
    return isinstance(value, dict) and "value" in value and (
        "evidence" in value or "status" in value
    )


def unwrap(value: Any) -> Any:
    if is_wrapped_field(value):
        status = value.get("status")
        inner = value.get("value")
        if status not in (None, "", "extracted"):
            return status
        return inner
    return value


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if is_wrapped_field(value):
        return compact_text(unwrap(value))
    if isinstance(value, str):
        if value in MISSING_VALUES:
            return ""
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(compact_text(item) for item in value if compact_text(item))
    if isinstance(value, dict):
        parts: list[str] = []
        for item in value.values():
            text = compact_text(item)
            if text:
                parts.append(text)
        return "; ".join(parts)
    return str(value)


def normalize_text(value: Any) -> str:
    text = compact_text(value).lower()
    text = text.replace("μ", "u").replace("µ", "u")
    text = re.sub(r"[^a-z0-9+/#.^ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


def split_terms(value: Any) -> set[str]:
    if value is None:
        return set()
    if is_wrapped_field(value):
        return split_terms(unwrap(value))
    if isinstance(value, list):
        terms: set[str] = set()
        for item in value:
            terms.update(split_terms(item))
        return terms
    if isinstance(value, dict):
        terms: set[str] = set()
        if "core_term" in value:
            terms.add(normalize_text(value.get("core_term")))
        if "mention" in value:
            terms.add(normalize_text(value.get("mention")))
        for item in value.get("modifiers") or []:
            terms.add(normalize_text(item))
        if not terms:
            for child in value.values():
                terms.update(split_terms(child))
        return {term for term in terms if term}
    text = normalize_text(value)
    if not text:
        return set()
    return {
        term
        for term in re.split(r"\s*[;,|]\s*", text)
        if term and term not in MISSING_VALUES
    }


def jaccard(left: set[str], right: set[str]) -> float | None:
    if not left or not right:
        return None
    return len(left & right) / len(left | right)


def set_similarity(left: Any, right: Any) -> float | None:
    return jaccard(split_terms(left), split_terms(right))


def categorical_similarity(left: Any, right: Any, matrix: dict[tuple[str, str], float] | None = None) -> float | None:
    left_text = normalize_text(unwrap(left))
    right_text = normalize_text(unwrap(right))
    if not left_text or not right_text:
        return None
    if left_text == right_text:
        return 1.0
    if matrix:
        return matrix.get((left_text, right_text)) or matrix.get((right_text, left_text)) or 0.0
    return 0.0


def soft_overlap_similarity(left: set[str], right: set[str]) -> float | None:
    if not left or not right:
        return None
    exact = len(left & right)
    partial = 0.0
    for left_item in left - right:
        best = 0.0
        left_tokens = set(tokenize(left_item))
        for right_item in right - left:
            right_tokens = set(tokenize(right_item))
            token_score = jaccard(left_tokens, right_tokens) or 0.0
            if left_item in right_item or right_item in left_item:
                token_score = max(token_score, 0.6)
            best = max(best, token_score)
        partial += best
    return (exact + partial) / max(len(left), len(right), 1)


def drug_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    return soft_overlap_similarity(split_terms(query.get("Drug_Name")), split_terms(article.get("Drug_Name")))


def cancer_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    return soft_overlap_similarity(split_terms(query.get("cancer_type")), split_terms(article.get("cancer_type")))


def route_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    return set_similarity(query.get("Admin_Route"), article.get("Admin_Route"))


def target_moa_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    target_score = soft_overlap_similarity(
        split_terms(query.get("Drug_target_gene_target")),
        split_terms(article.get("Drug_target_gene_target")),
    )
    moa_score = soft_overlap_similarity(
        split_terms(query.get("Mechanism_of_action")),
        split_terms(article.get("Mechanism_of_action")),
    )
    return weighted_average({"target": target_score, "moa": moa_score}, {"target": 0.55, "moa": 0.45})


def age_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    q_age = unwrap(query.get("Age"))
    a_age = unwrap(article.get("Age"))
    if not isinstance(q_age, dict) or not isinstance(a_age, dict):
        return None

    q_min, q_max = normalize_age_bounds(q_age)
    a_min, a_max = normalize_age_bounds(a_age)
    range_score = range_overlap(q_min, q_max, a_min, a_max)
    group_score = age_group_similarity(q_age.get("age_group"), a_age.get("age_group"))
    return weighted_average({"range": range_score, "group": group_score}, {"range": 0.7, "group": 0.3})


def normalize_age_bounds(age: dict[str, Any]) -> tuple[float, float]:
    unit = normalize_text(age.get("age_unit") or "years")
    factor = 1.0
    if unit.startswith("month"):
        factor = 1 / 12
    elif unit.startswith("day"):
        factor = 1 / 365
    low = age.get("age_min")
    high = age.get("age_max")
    return (
        float(low) * factor if isinstance(low, (int, float)) else 0.0,
        float(high) * factor if isinstance(high, (int, float)) else 100.0,
    )


def range_overlap(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    overlap = max(0.0, min(left_max, right_max) - max(left_min, right_min))
    union = max(left_max, right_max) - min(left_min, right_min)
    return overlap / union if union > 0 else 0.0


def age_group_similarity(left: Any, right: Any) -> float | None:
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return None
    if left_text == right_text:
        return 1.0
    if "adult_and_children" in {left_text, right_text}:
        return 0.7
    if {left_text, right_text} <= {"adult", "adolescent"}:
        return 0.5
    if {left_text, right_text} <= {"children", "adolescent"}:
        return 0.5
    return 0.0


def gender_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    left = normalize_text(query.get("Gender_Sex"))
    right = normalize_text(article.get("Gender_Sex"))
    if not left or not right or left in {"unknown", "not_specified"} or right in {"unknown", "not_specified"}:
        return None
    if left == right:
        return 1.0
    if "all" in {left, right}:
        return 0.5
    return 0.0


def dose_form_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    return set_similarity(query.get("Dose_Form"), article.get("Dose_Form"))


def design_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    matrix = {
        ("3 + 3", "modified 3 + 3"): 0.9,
        ("crm", "modified crm"): 0.8,
        ("crm", "tite crm"): 0.8,
        ("boin", "mtpi"): 0.6,
        ("boin", "keyboard design"): 0.6,
        ("model assisted design", "mtpi"): 0.6,
        ("model assisted design", "boin"): 0.6,
    }
    return categorical_similarity(query.get("DesignType"), article.get("DesignType"), matrix)


def trial_type_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    matrix = {
        ("combination", "mixed"): 0.6,
        ("single", "combination"): 0.3,
        ("single", "mixed"): 0.5,
        ("non drug modality only", "single"): 0.0,
        ("non drug modality only", "combination"): 0.0,
    }
    return categorical_similarity(query.get("Trial_Type"), article.get("Trial_Type"), matrix)


def dlt_evaluation_window_days(value: Any) -> float | None:
    """Return a comparable DLT-window duration in days when evidence permits.

    Explicit durations remain authoritative. When a DLT window is stated in
    cycles but its cycle length is missing, the similarity model uses the
    configured 21-day default per cycle. This is a modeling convention for
    similarity only; it does not alter the extracted literature evidence.
    """

    eva = unwrap(value)
    if not isinstance(eva, dict):
        return None

    duration = eva.get("dlt_eval_duration_days")
    if (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration))
        and float(duration) > 0
    ):
        return float(duration)

    if eva.get("normalization_status") != "cycle_length_missing":
        return None

    cycle_count = eva.get("dlt_eval_cycle_number")
    if (
        isinstance(cycle_count, (int, float))
        and not isinstance(cycle_count, bool)
        and math.isfinite(float(cycle_count))
        and float(cycle_count) > 0
    ):
        return DEFAULT_DLT_EVA_CYCLE_LENGTH_DAYS * float(cycle_count)
    return DEFAULT_DLT_EVA_CYCLE_LENGTH_DAYS


def dlt_eva_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    q_days = dlt_evaluation_window_days(query.get("DLT_EVA"))
    a_days = dlt_evaluation_window_days(article.get("DLT_EVA"))
    if q_days is None or a_days is None:
        return None
    return math.exp(-abs(q_days - a_days) / 21.0)


def dlt_definition_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    q_rows = [row for row in query.get("DLT_definition") or [] if isinstance(row, dict)]
    a_rows = [row for row in article.get("DLT_definition") or [] if isinstance(row, dict)]
    if not q_rows or not a_rows:
        return None
    name_score = soft_overlap_similarity(
        {normalize_text(row.get("DLT_Name")) for row in q_rows if row.get("DLT_Name")},
        {normalize_text(row.get("DLT_Name")) for row in a_rows if row.get("DLT_Name")},
    )
    grade_score = numeric_list_similarity(
        [grade_numeric(row.get("DLT_definition_Grade")) for row in q_rows],
        [grade_numeric(row.get("DLT_definition_Grade")) for row in a_rows],
        scale=2.0,
    )
    duration_score = numeric_list_similarity(
        [duration_days(row.get("DLT_definition_Duration")) for row in q_rows],
        [duration_days(row.get("DLT_definition_Duration")) for row in a_rows],
        scale=14.0,
    )
    return weighted_average(
        {"name": name_score, "grade": grade_score, "duration": duration_score},
        {"name": 0.5, "grade": 0.3, "duration": 0.2},
    )


def grade_numeric(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    operator = value.get("operator")
    if operator == "range":
        low = value.get("grade_min")
        high = value.get("grade_max")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return (float(low) + float(high)) / 2
    grade = value.get("grade_value")
    if isinstance(grade, (int, float)):
        return float(grade)
    return None


def duration_days(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    duration = value.get("duration_value")
    unit = normalize_text(value.get("duration_unit"))
    if not isinstance(duration, (int, float)):
        return None
    factor = 1.0
    if unit.startswith("hour"):
        factor = 1 / 24
    elif unit.startswith("week"):
        factor = 7
    elif unit.startswith("cycle"):
        factor = 21
    return float(duration) * factor


def numeric_list_similarity(left: list[float | None], right: list[float | None], scale: float) -> float | None:
    left_values = [item for item in left if item is not None]
    right_values = [item for item in right if item is not None]
    if not left_values or not right_values:
        return None
    scores = []
    for item in left_values:
        best = max(math.exp(-abs(item - other) / scale) for other in right_values)
        scores.append(best)
    return sum(scores) / len(scores)


def dose_grid_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    q_rows = actual_dose_rows(query)
    a_rows = actual_dose_rows(article)
    if not a_rows:
        return None
    if not q_rows:
        query_unit = query_target_unit(query)
        if not query_unit:
            return None
        article_units = {
            canonical_dose_unit(row.get("Dose_Unit"))
            for row in a_rows
            if canonical_dose_unit(row.get("Dose_Unit"))
        }
        if not article_units:
            return None
        query_base = canonical_dose_unit(query_unit)
        if query_base and query_base in article_units:
            return 1.0
        return 0.0

    unit_score = soft_overlap_similarity(
        {
            canonical_dose_unit(row.get("Dose_Unit"))
            for row in q_rows
            if canonical_dose_unit(row.get("Dose_Unit"))
        },
        {
            canonical_dose_unit(row.get("Dose_Unit"))
            for row in a_rows
            if canonical_dose_unit(row.get("Dose_Unit"))
        },
    )
    dose_score = dose_value_range_similarity(q_rows, a_rows)
    level_count_score = math.exp(-abs(len(q_rows) - len(a_rows)) / max(len(q_rows), len(a_rows), 1))
    schedule_score = schedule_similarity_for_rows(q_rows, a_rows)
    drug_score = soft_overlap_similarity(
        {normalize_text(row.get("Drug_Name")) for row in q_rows if row.get("Drug_Name")},
        {normalize_text(row.get("Drug_Name")) for row in a_rows if row.get("Drug_Name")},
    )
    return weighted_average(
        {
            "drug": drug_score,
            "unit": unit_score,
            "dose_range": dose_score,
            "schedule": schedule_score,
            "level_count": level_count_score,
        },
        {"drug": 0.25, "unit": 0.2, "dose_range": 0.25, "schedule": 0.2, "level_count": 0.1},
    )


def schedule_similarity(query: dict[str, Any], article: dict[str, Any]) -> float | None:
    q_rows = actual_dose_rows(query)
    a_rows = actual_dose_rows(article)
    if not q_rows or not a_rows:
        return None
    return schedule_similarity_for_rows(q_rows, a_rows)


def actual_dose_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in data.get("dose_levels") or [] if isinstance(row, dict)]
    actual = [row for row in rows if row.get("is_actual_escalation_dose") is True]
    return actual or rows


def dose_value_range_similarity(q_rows: list[dict[str, Any]], a_rows: list[dict[str, Any]]) -> float | None:
    q_values = [dose_numeric(row.get("Dose_Value")) for row in q_rows]
    a_values = [dose_numeric(row.get("Dose_Value")) for row in a_rows]
    q_values = [value for value in q_values if value and value > 0]
    a_values = [value for value in a_values if value and value > 0]
    if not q_values or not a_values:
        return None
    q_low, q_high = math.log(min(q_values)), math.log(max(q_values))
    a_low, a_high = math.log(min(a_values)), math.log(max(a_values))
    if q_low == q_high and a_low == a_high:
        return math.exp(-abs(q_low - a_low))
    overlap = max(0.0, min(q_high, a_high) - max(q_low, a_low))
    union = max(q_high, a_high) - min(q_low, a_low)
    return overlap / union if union > 0 else 0.0


def dose_numeric(value: Any) -> float | None:
    if not isinstance(value, dict):
        if isinstance(value, (int, float)):
            return float(value)
        return None
    value_type = value.get("value_type")
    if value_type == "single" and isinstance(value.get("value"), (int, float)):
        return float(value["value"])
    if value_type == "range":
        low = value.get("min_value")
        high = value.get("max_value")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return (float(low) + float(high)) / 2
    if value_type == "multiple":
        values = [float(item) for item in value.get("values") or [] if isinstance(item, (int, float))]
        if values:
            return sum(values) / len(values)
    return None


def schedule_similarity_for_rows(q_rows: list[dict[str, Any]], a_rows: list[dict[str, Any]]) -> float | None:
    scores: list[float] = []
    for q_row in q_rows:
        best = None
        for a_row in a_rows:
            score = single_schedule_similarity(q_row, a_row)
            if score is not None:
                best = score if best is None else max(best, score)
        if best is not None:
            scores.append(best)
    return sum(scores) / len(scores) if scores else None


def schedule_repeat_interval_days(schedule: dict[str, Any]) -> float | None:
    """Return a schedule's normalized repeat interval in days.

    The extraction schema calls its source field ``cycle_length_days``, but
    the value is already a duration in days (for example, Q3W is 21). The
    similarity model compares this duration directly and never compares a
    cycle count or cycle label.
    """

    value = schedule.get("cycle_length_days")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    days = float(value)
    return days if math.isfinite(days) and days > 0 else None


def single_schedule_similarity(q_row: dict[str, Any], a_row: dict[str, Any]) -> float | None:
    freq_score = categorical_similarity(
        (q_row.get("Frequency") or {}).get("frequency_standard") if isinstance(q_row.get("Frequency"), dict) else None,
        (a_row.get("Frequency") or {}).get("frequency_standard") if isinstance(a_row.get("Frequency"), dict) else None,
    )
    q_schedule = q_row.get("Schedule") if isinstance(q_row.get("Schedule"), dict) else {}
    a_schedule = a_row.get("Schedule") if isinstance(a_row.get("Schedule"), dict) else {}
    q_interval_days = schedule_repeat_interval_days(q_schedule)
    a_interval_days = schedule_repeat_interval_days(a_schedule)
    interval_days_score = None
    if q_interval_days is not None and a_interval_days is not None:
        interval_days_score = math.exp(
            -abs(q_interval_days - a_interval_days) / 21.0
        )
    administration_days_score = None
    q_administration_days = set(q_schedule.get("administration_days") or [])
    a_administration_days = set(a_schedule.get("administration_days") or [])
    if q_administration_days and a_administration_days:
        administration_days_score = len(
            q_administration_days & a_administration_days
        ) / len(q_administration_days | a_administration_days)
    return weighted_average(
        {
            "frequency": freq_score,
            "interval_days": interval_days_score,
            "administration_days": administration_days_score,
        },
        {"frequency": 0.4, "interval_days": 0.3, "administration_days": 0.3},
    )


def weighted_average(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for key, score in scores.items():
        if score is None:
            continue
        weight = weights.get(key, 0.0)
        numerator += weight * max(0.0, min(1.0, score))
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def paragraph_text(data: dict[str, Any], field_names: list[str]) -> str:
    parts: list[str] = []
    for field_name in field_names:
        if field_name == "DLT_EVA_text":
            dlt_eva = unwrap(data.get("DLT_EVA"))
            if isinstance(dlt_eva, dict):
                parts.append(compact_text(dlt_eva.get("dlt_eval_text")))
            continue
        parts.append(compact_text(data.get(field_name)))
    return "; ".join(part for part in parts if part)


def paragraph_similarity_scores(
    query: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    embedding_backend: str,
    embedding_model: str,
    cache_path: Path,
) -> tuple[dict[str, dict[str, float]], str]:
    by_field: dict[str, dict[str, float]] = {}
    backend_used = embedding_backend
    for score_name, fields in PARAGRAPH_FIELDS.items():
        query_text = paragraph_text(query, fields)
        article_texts = {
            record["pmcid"]: paragraph_text(record["data"], fields)
            for record in records
            if paragraph_text(record["data"], fields)
        }
        if not query_text or not article_texts:
            continue
        scores, backend_used = score_paragraph_embedding(
            query_text=query_text,
            article_texts=article_texts,
            field_name=score_name,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            cache_path=cache_path,
        )
        by_field[score_name] = scores
    return by_field, backend_used


def score_paragraph_embedding(
    *,
    query_text: str,
    article_texts: dict[str, str],
    field_name: str,
    embedding_backend: str,
    embedding_model: str,
    cache_path: Path,
) -> tuple[dict[str, float], str]:
    if embedding_backend == "local":
        return score_paragraph_local(query_text, article_texts), "local"
    if embedding_backend == "openai":
        return score_paragraph_openai(query_text, article_texts, field_name, embedding_model, cache_path), "openai"
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return score_paragraph_openai(query_text, article_texts, field_name, embedding_model, cache_path), "openai"
        except Exception:
            return score_paragraph_local(query_text, article_texts), "local"
    return score_paragraph_local(query_text, article_texts), "local"


def score_paragraph_local(query_text: str, article_texts: dict[str, str]) -> dict[str, float]:
    article_ids = list(article_texts)
    vectors = vectorize_texts([article_texts[article_id] for article_id in article_ids] + [query_text])
    query_vector = vectors[-1]
    return {
        article_id: max(0.0, cosine_sparse(vectors[index], query_vector))
        for index, article_id in enumerate(article_ids)
    }


def token_features(text: str) -> list[str]:
    tokens = tokenize(text)
    features = list(tokens)
    features.extend(f"bg:{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def vectorize_texts(texts: list[str]) -> list[dict[str, float]]:
    counters = [Counter(token_features(text)) for text in texts]
    doc_freq: Counter[str] = Counter()
    for counter in counters:
        doc_freq.update(counter.keys())
    total = len(counters)
    idf = {token: math.log((1 + total) / (1 + freq)) + 1.0 for token, freq in doc_freq.items()}
    vectors: list[dict[str, float]] = []
    for counter in counters:
        weights = {token: count * idf[token] for token, count in counter.items()}
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if norm > 0:
            weights = {token: value / norm for token, value in weights.items()}
        vectors.append(weights)
    return vectors


def cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def score_paragraph_openai(
    query_text: str,
    article_texts: dict[str, str],
    field_name: str,
    embedding_model: str,
    cache_path: Path,
) -> dict[str, float]:
    with embedding_cache_lock(cache_path):
        cache = load_cache(cache_path)
        items = cache.setdefault("items", {})
        article_vectors: dict[str, list[float]] = {}
        missing: list[tuple[str, str, str]] = []
        for article_id, text in article_texts.items():
            key = f"{field_name}:{article_id}"
            text_hash = hash_text(text)
            cached = items.get(key)
            if (
                cached
                and cached.get("model") == embedding_model
                and cached.get("text_hash") == text_hash
            ):
                article_vectors[article_id] = cached["vector"]
            else:
                missing.append((key, article_id, text))
        if missing:
            vectors = embed_texts_openai(
                [text for _key, _article_id, text in missing], embedding_model
            )
            for (key, article_id, text), vector in zip(missing, vectors):
                normalized = normalize_dense(vector)
                article_vectors[article_id] = normalized
                items[key] = {
                    "model": embedding_model,
                    "text_hash": hash_text(text),
                    "vector": normalized,
                }
            save_cache(cache_path, cache)
    query_vector = normalize_dense(embed_texts_openai([query_text], embedding_model)[0])
    return {
        article_id: max(0.0, cosine_dense(vector, query_vector))
        for article_id, vector in article_vectors.items()
    }


def embed_texts_openai(texts: list[str], model: str) -> list[list[float]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json().get("data") or []
    if len(data) != len(texts):
        raise RuntimeError("Embedding API returned an unexpected number of vectors.")
    return [item["embedding"] for item in data]


def normalize_dense(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def cosine_dense(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"items": {}}
    if not isinstance(payload, dict):
        return {"items": {}}
    payload.setdefault("items", {})
    return payload


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def embedding_cache_lock(path: Path):
    """Serialize cache read-modify-write cycles across concurrent CLI processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - non-POSIX fallback
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                pass


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_article(
    query: dict[str, Any],
    article: dict[str, Any],
    paragraph_scores: dict[str, dict[str, float]],
    pmcid: str,
    *,
    field_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    weights = normalize_field_weights(field_weights)
    scores: dict[str, float | None] = {
        "drug": drug_similarity(query, article),
        "cancer_type": cancer_similarity(query, article),
        "route": route_similarity(query, article),
        "schedule": schedule_similarity(query, article),
        "dlt_definition": dlt_definition_similarity(query, article),
        "dlt_eva": dlt_eva_similarity(query, article),
        "target_moa": target_moa_similarity(query, article),
        "age": age_similarity(query, article),
        "trial_type": trial_type_similarity(query, article),
        "gender": gender_similarity(query, article),
        "dose_form": dose_form_similarity(query, article),
    }
    for field_name, by_article in paragraph_scores.items():
        scores[field_name] = by_article.get(pmcid)
    total = weighted_average(scores, weights)
    return {
        "similarity": total or 0.0,
        "available_weight": sum(
            weights[key]
            for key, value in scores.items()
            if key in weights and value is not None
        ),
        "field_scores": {key: value for key, value in scores.items() if value is not None},
    }


def rescore_library_from_field_scores(
    scored: list[dict[str, Any]],
    *,
    field_weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Recombine cached field scores under a candidate weight vector."""

    if any(article.get('version') == 'cohort_profile_similarity_v1' for article in scored):
        from ..profile_similarity import normalize_weights as profile_weights
        weights=profile_weights(field_weights)
        def recombine(result):
            available=sum(weights[k] for k,v in result.get('groups',{}).items() if v.get('score') is not None)
            value=sum(weights[k]*v['score'] for k,v in result.get('groups',{}).items() if v.get('score') is not None)/available if available else None
            return {**result,'weights':weights,'similarity':value,'available_weight':available}
        result=[]
        for article in scored:
            row=recombine(article)
            scopes={k:recombine(v) for k,v in article.get('scope_scores',{}).items()}
            valid=[k for k,v in scopes.items() if v['similarity'] is not None]
            if valid:
                best=max(sorted(valid),key=lambda k:scopes[k]['similarity'])
                row.update(scopes[best]);row['selected_scope']=best
            row['scope_scores']=scopes;row['comparison_similarity']=row['similarity'];row['similarity']=row['similarity'] or 0.
            result.append(row)
        return sorted(result,key=lambda r:(-r['similarity'],r['pmcid']))
    weights = normalize_field_weights(field_weights)
    rescored = []
    for article in scored:
        field_scores = article.get("field_scores", {})
        if not isinstance(field_scores, dict):
            raise ValueError("Scored article is missing field_scores.")
        similarity = weighted_average(field_scores, weights) or 0.0
        available_weight = sum(
            weights[field_name]
            for field_name in field_scores
            if field_name in weights and field_scores[field_name] is not None
        )
        rescored.append(
            {
                **article,
                "similarity": similarity,
                "available_weight": available_weight,
            }
        )
    return sorted(rescored, key=lambda item: item["similarity"], reverse=True)


def filter_scored_by_similarity_threshold(
    scored: list[dict[str, Any]], threshold: float | int | None
) -> list[dict[str, Any]]:
    """Keep articles whose raw field-weighted similarity is at least threshold."""

    resolved_threshold = validate_similarity_threshold(threshold)
    return [
        article
        for article in scored
        if float(article.get("similarity", 0.0)) >= resolved_threshold
    ]


def mtd_missing_status(data: dict[str, Any]) -> str:
    value = unwrap(data.get("MTD_missing_status"))
    return (normalize_text(value).replace(" ", "_") or "unknown")


def mtd_value_and_unit(data: dict[str, Any]) -> tuple[float | None, str | None]:
    value = dose_numeric(unwrap(data.get("MTD_Value")))
    raw_unit = unwrap(data.get("MTD_Unit"))
    unit = raw_unit.strip() if isinstance(raw_unit, str) and raw_unit.strip() else None
    return value, unit


def mad_value_and_unit(data: dict[str, Any]) -> tuple[float | None, str | None]:
    value = dose_numeric(unwrap(data.get("MAD_Value")))
    raw_unit = unwrap(data.get("MAD_Unit"))
    unit = raw_unit.strip() if isinstance(raw_unit, str) and raw_unit.strip() else None
    return value, unit


def unit_terms_from_field(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        terms: set[str] = set()
        for item in value:
            terms.update(unit_terms_from_field(item))
        return terms
    raw_unit = unwrap(value)
    text = normalize_dose_unit_text(raw_unit if isinstance(raw_unit, str) else None)
    return {text} if text else set()


def query_target_unit(query: dict[str, Any]) -> str | None:
    """Return the canonical dose unit declared for a target toxicity query."""

    for field_name in ("Dose_Unit", "MTD_Unit"):
        units = unit_terms_from_field(query.get(field_name))
        if len(units) == 1:
            return canonical_dose_unit(next(iter(units)))
        if len(units) > 1:
            canonical_units = {canonical_dose_unit(unit) for unit in units if canonical_dose_unit(unit)}
            if len(canonical_units) == 1:
                return next(iter(canonical_units))
            return sorted(unit for unit in canonical_units if unit)[0]

    rows = actual_dose_rows(query)
    row_units = {
        canonical_dose_unit(row.get("Dose_Unit"))
        for row in rows
        if canonical_dose_unit(row.get("Dose_Unit"))
    }
    if len(row_units) == 1:
        return next(iter(row_units))
    if len(row_units) > 1:
        return sorted(row_units)[0]
    return None


def effective_sample_size(weights: list[float]) -> float:
    denominator = sum(weight * weight for weight in weights)
    return 1 / denominator if denominator > 0 else 0.0


def score_library(
    query: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    embedding_backend: str = "auto",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: Path = DEFAULT_EMBEDDING_CACHE,
    field_weights: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Score the full historical library using pre-trial similarity fields."""

    if any("_comparison_profile" in r["data"] for r in records):
        from ..profile_scoring import score_native_library
        return score_native_library(query, records, weights=field_weights,
            embedding_backend=embedding_backend, embedding_model=embedding_model, embedding_cache=embedding_cache)
    paragraph_scores, backend_used = paragraph_similarity_scores(
        query,
        records,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        cache_path=embedding_cache,
    )
    scored: list[dict[str, Any]] = []
    for record in records:
        score = score_article(
            query,
            record["data"],
            paragraph_scores,
            record["pmcid"],
            field_weights=field_weights,
        )
        scored.append(
            {
                "pmcid": record["pmcid"],
                "path": record["path"],
                "data": record["data"],
                **score,
            }
        )
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored, backend_used
