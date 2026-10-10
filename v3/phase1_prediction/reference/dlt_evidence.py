"""Parse, validate, and group dose-level DLT evidence from compact extractions."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from .dose_unit_conversion import conversion_metadata, describe_dose_unit
from .toxicity_similarity import canonical_dose_value


MISSING_VALUES = {"", "not_reported", "unclear", "not_applicable", "unknown", None}
UNSUPPORTED_UNITS = {"other", "unknown", "unitless"}
INFERRED_ZERO_EVIDENCE = (
    "Model-derived zero: this article reports dose-level DLT results, but this "
    "actual escalation dose has no separate extracted DLT row."
)
_DESIGN_COHORT_SIZES = {
    "3 + 3": 3,
    "modified 3 + 3": 3,
    "6 + 6": 6,
    "rolling-six design": 6,
}


def is_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().casefold() in MISSING_VALUES
    )


def field_value(value: Any) -> Any:
    """Unwrap a compact evidence field without unwrapping nested Dose_Value."""

    if isinstance(value, dict) and "value" in value and "evidence" in value:
        value = value["value"]
    return None if is_missing(value) else value


def normalized_key(value: Any) -> str:
    value = field_value(value)
    return "" if is_missing(value) else str(value).strip().casefold()


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def integer_count(value: Any) -> int | None:
    """Return a finite integer count, accepting JSON values such as ``3.0``."""

    resolved = numeric(value)
    if resolved is None or not resolved.is_integer():
        return None
    return int(resolved)


def scalar_dose(dose_level: dict[str, Any]) -> float | None:
    dose_value = field_value(dose_level.get("Dose_Value"))
    if not isinstance(dose_value, dict) or dose_value.get("value_type") != "single":
        return None
    value = numeric(dose_value.get("value"))
    return value if value is not None and value > 0 else None


def dlt_join_key(dlt_row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalized_key(dlt_row.get("dose_level")),
        normalized_key(dlt_row.get("arm_or_cohort")),
        normalized_key(dlt_row.get("drug_name")),
    )


def match_dose_level(
    dlt_row: dict[str, Any], dose_levels: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    """Join a DLT row to exactly one dose-level row without guessing its dose.

    A globally unique dose-level label is the primary identity. Extractors
    sometimes put ``DL5`` in the dose table's ``arm_or_cohort`` field and
    ``250 mg group`` in the DLT row for that same level. Rejecting that row
    loses valid dose--DLT evidence, so arm text is used only to disambiguate
    repeated dose-level labels.
    """

    level, arm, drug = dlt_join_key(dlt_row)
    if not level:
        return None, "missing_dose_level_label"

    candidates = [
        row
        for row in dose_levels
        if normalized_key(row.get("Dose_Level")) == level
    ]
    if drug:
        narrowed = [
            row
            for row in candidates
            if normalized_key(row.get("Drug_Name")) == drug
        ]
        if not narrowed:
            return None, "dose_level_Drug_Name_mismatch"
        candidates = narrowed

    if len(candidates) > 1 and arm:
        narrowed = [
            row
            for row in candidates
            if normalized_key(row.get("arm_or_cohort")) == arm
        ]
        if not narrowed:
            return None, "dose_level_arm_or_cohort_mismatch"
        candidates = narrowed

    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, "dose_level_not_found"
    return None, "ambiguous_dose_level_join"


def denominator_and_source(dlt_row: dict[str, Any]) -> tuple[float | None, str | None]:
    source_override = dlt_row.get("denominator_source_override")
    if not isinstance(source_override, str) or not source_override:
        source_override = None
    evaluable = numeric(dlt_row.get("n_patients_evaluable_for_dlt"))
    if evaluable is not None:
        return evaluable, source_override or "evaluable_for_dlt"
    treated = numeric(dlt_row.get("n_patients_treated"))
    if treated is not None:
        return treated, source_override or "treated"
    return None, None


def _inferred_base_cohort_size(data: dict[str, Any]) -> tuple[int, str]:
    """Return a transparent effective denominator for an inferred zero row."""

    design = field_value(data.get("DesignType"))
    if isinstance(design, str):
        cohort_size = _DESIGN_COHORT_SIZES.get(design.strip())
        if cohort_size is not None:
            return cohort_size, "inferred_from_design_type"

    schema = field_value(data.get("schema"))
    if isinstance(schema, str):
        patterns = (
            r"cohorts?\s+(?:consisted\s+of\s+)?(?:at\s+least\s+)?(\d+)\s+"
            r"(?:evaluable\s+)?(?:patients|subjects|individuals)",
            r"(\d+)\s+(?:patients|subjects|individuals)\s+per\s+cohort",
        )
        for pattern in patterns:
            match = re.search(pattern, schema, flags=re.IGNORECASE)
            if match:
                cohort_size = int(match.group(1))
                if cohort_size > 0:
                    return cohort_size, "inferred_from_schema_cohort_size"

    # This is deliberately one effective observation, not a claim that one
    # patient was treated. It lets the requested implicit zero enter the curve
    # without inventing a large, highly influential cohort.
    return 1, "inferred_minimal_effective_denominator"


def _eligible_escalation_level(dose_level: dict[str, Any]) -> bool:
    return (
        field_value(dose_level.get("is_actual_escalation_dose")) is True
        and field_value(dose_level.get("drug_role")) == "escalated_study_drug"
    )


def _dose_level_identity(dose_level: dict[str, Any]) -> tuple[str, str]:
    return (
        normalized_key(dose_level.get("Dose_Level")),
        normalized_key(dose_level.get("Drug_Name")),
    )


def complete_unreported_dlt_rows(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Add auditable zero-DLT rows after an article reports any DLT result.

    The extraction remains untouched. Explicit rows keep priority. Missing
    actual escalation levels receive zero events and a conservative effective
    denominator. For standard 3+3 data, a reported single DLT with a missing
    denominator is treated as the protocol-required expanded cohort of six.
    """

    dose_levels = [
        row
        for row in data.get("dose_levels") or []
        if isinstance(row, dict) and _eligible_escalation_level(row)
    ]
    original_rows = [
        row
        for row in data.get("dlt_count_at_each_dose_level") or []
        if isinstance(row, dict)
    ]
    rows = [dict(row) for row in original_rows]
    reasons: set[str] = set()
    if not any(row.get("is_actual_escalation_dose") is True for row in original_rows):
        return rows, reasons

    identity_counts = Counter(_dose_level_identity(row) for row in dose_levels)
    matched_identities: set[tuple[str, str]] = set()
    event_anchor_found = False
    base_cohort_size, inferred_denominator_source = _inferred_base_cohort_size(data)
    design = field_value(data.get("DesignType"))
    standard_three_plus_three = design in {"3 + 3", "modified 3 + 3"}

    for row in rows:
        if row.get("is_actual_escalation_dose") is not True:
            continue
        matched, _reason = match_dose_level(row, dose_levels)
        if matched is None:
            continue
        matched_identities.add(_dose_level_identity(matched))
        events = integer_count(row.get("n_patients_with_dlt"))
        if events is not None:
            event_anchor_found = True
        denominator, _source = denominator_and_source(row)
        if events is None or denominator is not None or not standard_three_plus_three:
            continue
        inferred_denominator = 6 if events == 1 else max(3, events)
        row["n_patients_evaluable_for_dlt"] = inferred_denominator
        row["denominator_source_override"] = "inferred_standard_3_plus_3_cohort"
        row["dlt_rate"] = events / inferred_denominator
        row["dlt_value_origin"] = "extracted_events_inferred_3_plus_3_denominator"
        row["inference_note"] = (
            "The event count is extracted; the denominator follows the standard "
            "3+3 cohort rule because the article did not report it for this row."
        )

    # Do not manufacture an all-zero curve when the only extracted DLT rows
    # cannot be joined to this article's dose table (for example, a wrong drug
    # identifier). At least one real row must anchor the completion rule.
    if not matched_identities or not event_anchor_found:
        return rows, reasons

    for dose_level in dose_levels:
        identity = _dose_level_identity(dose_level)
        if not identity[0]:
            reasons.add("implicit_zero_missing_dose_level_label")
            continue
        if identity_counts[identity] != 1:
            reasons.add("implicit_zero_ambiguous_dose_level_identity")
            continue
        if identity in matched_identities:
            continue
        level = field_value(dose_level.get("Dose_Level"))
        drug = field_value(dose_level.get("Drug_Name"))
        arm = field_value(dose_level.get("arm_or_cohort"))
        rows.append(
            {
                "dose_level": level,
                "dose_description": None,
                "drug_name": drug,
                "arm_or_cohort": arm,
                "is_actual_escalation_dose": True,
                "n_patients_treated": None,
                "n_patients_evaluable_for_dlt": base_cohort_size,
                "denominator_source_override": inferred_denominator_source,
                "n_patients_with_dlt": 0,
                "dlt_rate": 0.0,
                "evidence": INFERRED_ZERO_EVIDENCE,
                "dlt_value_origin": "inferred_zero_for_unreported_dose_level",
                "inference_note": (
                    "Zero events are inferred by the requested article-level rule; "
                    f"the denominator source is {inferred_denominator_source}."
                ),
            }
        )
    if len(rows) > len(original_rows):
        reasons.add("implicit_zero_dlt_rows_added")
    return rows, reasons


def curve_arm_or_cohort(
    dose_level: dict[str, Any], arm_counts: Counter[str] | None = None
) -> str | None:
    """Discard dose-cohort labels masquerading as treatment-arm identities.

    A genuine treatment arm normally spans more than one dose level. If every
    level has its own one-off ``cohort 1``/``cohort 2`` label, grouping by that
    text would incorrectly create one curve per dose.
    """

    arm = field_value(dose_level.get("arm_or_cohort"))
    if not isinstance(arm, str):
        return None
    if dose_level.get("_v3_configuration_scope") is True:
        return arm
    if normalized_key(arm) == normalized_key(dose_level.get("Dose_Level")):
        return None
    if arm_counts is not None and arm_counts[normalized_key(arm)] <= 1:
        return None
    return arm


def _normalized_regimen_value(value: Any) -> Any:
    """Canonicalize structured schedule metadata without inferring omissions."""

    if isinstance(value, dict):
        normalized = {
            str(key): _normalized_regimen_value(child)
            for key, child in value.items()
        }
        administration_days = normalized.get("administration_days")
        if isinstance(administration_days, list):
            normalized["administration_days"] = sorted(
                administration_days,
                key=lambda day: json.dumps(
                    day, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            )
        return normalized
    if isinstance(value, list):
        return [_normalized_regimen_value(item) for item in value]
    if (
        not isinstance(value, bool)
        and isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
    ):
        return int(value)
    return value


def regimen_signature(
    frequency: Any,
    schedule: Any,
) -> str:
    """Return a stable identity for a dose regimen without inferring missing data."""

    payload = {
        "frequency": _normalized_regimen_value(field_value(frequency)),
        "schedule": _normalized_regimen_value(field_value(schedule)),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dose_level_regimen_signature(dose_level: dict[str, Any]) -> str:
    return regimen_signature(dose_level.get("Frequency"), dose_level.get("Schedule"))


def build_candidate_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    """Return valid, uniquely joined actual-escalation dose-level DLT rows."""

    dose_levels = [
        row for row in data.get("dose_levels") or [] if isinstance(row, dict)
    ]
    dlt_rows, reasons = complete_unreported_dlt_rows(data)
    arm_counts: Counter[str] = Counter(
        normalized_key(row.get("arm_or_cohort"))
        for row in dose_levels
        if _eligible_escalation_level(row)
        and normalized_key(row.get("arm_or_cohort"))
    )
    candidates: list[dict[str, Any]] = []
    dlt_key_counts: Counter[tuple[str, str, str]] = Counter(
        dlt_join_key(row)
        for row in dlt_rows
        if row.get("is_actual_escalation_dose") is True
    )
    duplicate_dlt_keys = {
        key for key, count in dlt_key_counts.items() if count > 1
    }

    for dlt_row in dlt_rows:
        if dlt_row.get("is_actual_escalation_dose") is not True:
            reasons.add("dlt_row_not_actual_escalation_dose")
            continue
        if dlt_join_key(dlt_row) in duplicate_dlt_keys:
            reasons.add("duplicate_dlt_join_key")
            continue
        dose_level, join_reason = match_dose_level(dlt_row, dose_levels)
        if join_reason:
            reasons.add(join_reason)
            continue
        assert dose_level is not None
        if field_value(dose_level.get("is_actual_escalation_dose")) is not True:
            reasons.add("dose_level_not_actual_escalation_dose")
            continue
        if field_value(dose_level.get("drug_role")) != "escalated_study_drug":
            reasons.add("dose_level_not_escalated_study_drug")
            continue

        raw_dose = scalar_dose(dose_level)
        raw_unit = field_value(dose_level.get("Dose_Unit"))
        if raw_dose is None:
            reasons.add("dose_not_single_positive_number")
            continue
        unit_info = describe_dose_unit(raw_unit if isinstance(raw_unit, str) else None)
        if (
            not isinstance(raw_unit, str)
            or raw_unit in UNSUPPORTED_UNITS
            or unit_info.canonical_unit is None
        ):
            reasons.add("unsupported_or_missing_dose_unit")
            continue

        raw_events = numeric(dlt_row.get("n_patients_with_dlt"))
        raw_denominator, denominator_source = denominator_and_source(dlt_row)
        if raw_events is None:
            reasons.add("missing_dlt_numerator")
            continue
        if raw_denominator is None:
            reasons.add("missing_dlt_denominator")
            continue
        invalid_count_integrality = False
        if not raw_events.is_integer():
            reasons.add("noninteger_dlt_numerator")
            invalid_count_integrality = True
        if not raw_denominator.is_integer():
            reasons.add("noninteger_dlt_denominator")
            invalid_count_integrality = True
        if invalid_count_integrality:
            continue
        events = int(raw_events)
        denominator = int(raw_denominator)
        if denominator <= 0:
            reasons.add("nonpositive_dlt_denominator")
            continue
        if events < 0 or events > denominator:
            reasons.add("invalid_dlt_counts")
            continue

        canonical_dose, canonical_unit = canonical_dose_value(raw_dose, raw_unit)
        if canonical_dose is None or canonical_unit is None:
            reasons.add("dose_canonicalization_failed")
            continue

        arm = curve_arm_or_cohort(dose_level, arm_counts)
        drug = field_value(dose_level.get("Drug_Name"))
        frequency = field_value(dose_level.get("Frequency"))
        schedule = field_value(dose_level.get("Schedule"))
        candidates.append(
            {
                "arm_or_cohort": arm if isinstance(arm, str) else None,
                "drug_name": drug if isinstance(drug, str) else None,
                "frequency": frequency,
                "schedule": schedule,
                "regimen_signature": regimen_signature(frequency, schedule),
                "dose_level": field_value(dose_level.get("Dose_Level")),
                "dose": raw_dose,
                "dose_unit": raw_unit,
                "canonical_dose": canonical_dose,
                "canonical_unit": canonical_unit,
                "unit_conversion": conversion_metadata(raw_unit),
                "n_patients_with_dlt": events,
                "n_patients": denominator,
                "denominator_source": denominator_source,
                "observed_dlt_probability": events / denominator,
                "evidence": dlt_row.get("evidence"),
                "dlt_value_origin": dlt_row.get("dlt_value_origin", "extracted"),
                "inference_note": dlt_row.get("inference_note"),
            }
        )
    return candidates, reasons


def group_candidate_rows(
    candidates: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], set[str]]:
    """Group evidence by arm, drug, and regimen, rejecting mixed-unit groups."""

    by_arm_drug_regimen: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            normalized_key(row["arm_or_cohort"]) or "unspecified_arm",
            normalized_key(row["drug_name"]) or "unspecified_drug",
            str(row["regimen_signature"]),
        )
        by_arm_drug_regimen[key].append(row)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    reasons: set[str] = set()
    for key, rows in by_arm_drug_regimen.items():
        units = {row["canonical_unit"] for row in rows}
        if len(units) != 1:
            reasons.add("mixed_dose_units_within_arm")
            continue
        groups[key] = rows
    return groups, reasons


def aggregate_duplicate_doses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine duplicate dose cohorts only within a homogeneous arm/drug group."""

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_dose"]].append(row)

    combined: list[dict[str, Any]] = []
    for canonical_dose, dose_rows in grouped.items():
        total_events = sum(row["n_patients_with_dlt"] for row in dose_rows)
        total_patients = sum(row["n_patients"] for row in dose_rows)
        combined.append(
            {
                "arm_or_cohort": dose_rows[0]["arm_or_cohort"],
                "drug_name": dose_rows[0]["drug_name"],
                "frequency": dose_rows[0]["frequency"],
                "schedule": dose_rows[0]["schedule"],
                "regimen_signature": dose_rows[0]["regimen_signature"],
                "dose_level": ", ".join(
                    str(row["dose_level"])
                    for row in dose_rows
                    if row["dose_level"] is not None
                )
                or None,
                "dose": dose_rows[0]["dose"],
                "dose_unit": dose_rows[0]["dose_unit"],
                "canonical_dose": canonical_dose,
                "canonical_unit": dose_rows[0]["canonical_unit"],
                "source_dose_units": sorted(
                    {str(row["dose_unit"]) for row in dose_rows}
                ),
                "unit_conversions": [
                    conversion
                    for _key, conversion in sorted(
                        {
                            json.dumps(
                                row["unit_conversion"],
                                ensure_ascii=False,
                                sort_keys=True,
                            ): row["unit_conversion"]
                            for row in dose_rows
                        }.items()
                    )
                ],
                "n_patients_with_dlt": total_events,
                "n_patients": total_patients,
                "denominator_source": "+".join(
                    sorted({row["denominator_source"] for row in dose_rows})
                ),
                "observed_dlt_probability": total_events / total_patients,
                "evidence": "; ".join(
                    str(row["evidence"])
                    for row in dose_rows
                    if row.get("evidence")
                )
                or None,
                "dlt_value_origins": sorted(
                    {str(row["dlt_value_origin"]) for row in dose_rows}
                ),
                "inference_notes": sorted(
                    {
                        str(row["inference_note"])
                        for row in dose_rows
                        if row.get("inference_note")
                    }
                ),
            }
        )
    return sorted(combined, key=lambda row: row["canonical_dose"])


def candidate_segments(
    data: dict[str, Any], *, target_unit: str, curve_drug: str | None = None
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Get comparable, per-arm DLT segments for a canonical dose unit.

    ``curve_drug`` is optional. When omitted, every actual escalated-drug arm
    in the matching unit remains eligible and receives its article's
    pre-trial similarity weight downstream.
    """

    candidates, reasons = build_candidate_rows(data)
    candidates = [
        row
        for row in candidates
        if row["canonical_unit"] == target_unit
        and (
            curve_drug is None
            or normalized_key(row["drug_name"]) == normalized_key(curve_drug)
        )
    ]
    if not candidates:
        return [], sorted(reasons)

    groups, group_reasons = group_candidate_rows(candidates)
    reasons.update(group_reasons)
    return [aggregate_duplicate_doses(rows) for rows in groups.values()], sorted(reasons)
