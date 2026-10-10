"""Shared target-query and historical-evidence helpers for toxicity curves."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .dlt_evidence import (
    candidate_segments,
    field_value,
    normalized_key,
    regimen_signature,
    scalar_dose,
)
from .toxicity_similarity import (
    canonical_dose_value,
    mad_value_and_unit,
    mtd_missing_status,
    mtd_value_and_unit,
    query_target_unit,
)


def normalized_trial_type(value: Any) -> str:
    """Normalize user/extraction trial type to the two supported target values."""

    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise ValueError("Trial_Type is required and must be 'single' or 'combination'.")
    normalized = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
    aliases = {
        "single": "single",
        "single drug": "single",
        "monotherapy": "single",
        "combination": "combination",
        "combo": "combination",
        "combined": "combination",
        "combination therapy": "combination",
    }
    if normalized not in aliases:
        raise ValueError(
            "Trial_Type must be 'single' or 'combination' for target toxicity prediction."
        )
    return aliases[normalized]


@dataclass(frozen=True)
class ToxicityQuery:
    curve_drug: str
    target_unit: str
    trial_type: str
    planned_doses: tuple[float, ...]


def parse_toxicity_query(query: dict[str, Any]) -> ToxicityQuery:
    curve_drug = query.get("toxicity_curve_drug")
    if not isinstance(curve_drug, str) or not curve_drug.strip():
        raise ValueError("toxicity_curve_drug must be a non-empty string.")

    target_unit = query_target_unit(query)
    if not target_unit:
        raise ValueError("Dose_Unit is required and must resolve to one canonical unit.")
    trial_type = normalized_trial_type(query.get("Trial_Type"))

    raw_doses = query.get("planned_doses")
    if raw_doses is not None and (
        not isinstance(raw_doses, list) or not raw_doses
    ):
        raise ValueError(
            "planned_doses must be omitted or be a non-empty list of positive numbers."
        )
    raw_unit = query.get("Dose_Unit")
    if isinstance(raw_unit, dict):
        raw_unit = raw_unit.get("value")
    if not isinstance(raw_unit, str) or not raw_unit.strip():
        raise ValueError("Dose_Unit must be one non-empty unit string.")
    planned_doses: list[float] = []
    for index, value in enumerate(raw_doses or []):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"planned_doses[{index}] must be a number.")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"planned_doses[{index}] must be positive and finite.")
        canonical_value, canonical_unit = canonical_dose_value(float(value), raw_unit)
        if canonical_value is None or canonical_unit != target_unit:
            raise ValueError(f"planned_doses[{index}] cannot be converted to Dose_Unit.")
        planned_doses.append(canonical_value)

    return ToxicityQuery(
        curve_drug=curve_drug.strip(),
        target_unit=target_unit,
        trial_type=trial_type,
        planned_doses=tuple(sorted(set(planned_doses))),
    )


def derive_auto_curve_doses(
    scored_articles: list[dict[str, Any]],
    spec: ToxicityQuery,
    *,
    additional_doses: list[float] | tuple[float, ...] = (),
    grid_points: int = 101,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Build a display grid from comparable historical evidence.

    ``additional_doses`` is reserved for evaluation-only points, such as a
    held-out article's actual dose levels in leave-one-out validation. Those
    points never enter fitting because the held-out article is absent from
    ``scored_articles``.
    """

    if grid_points < 2:
        raise ValueError("grid_points must be at least 2.")

    weighted_doses: list[tuple[float, float]] = []
    for article in scored_articles:
        segments, _reasons = article_segments(article, spec)
        if not segments:
            continue
        article_similarity = max(0.0, float(article.get("similarity", 0.0)))
        if article_similarity <= 0:
            continue
        from ..profile_scoring import segment_weight as configuration_weight
        for segment in segments:
            segment_weight = configuration_weight(article, segment, segments)
            for row in segment:
                weighted_doses.append(
                    (
                        row["canonical_dose"],
                        segment_weight * row["n_patients"],
                    )
                )

    valid_additional = [
        float(dose)
        for dose in additional_doses
        if isinstance(dose, (int, float))
        and not isinstance(dose, bool)
        and math.isfinite(float(dose))
        and dose > 0
    ]
    if not weighted_doses:
        return (), {
            "mode": "automatic_from_similarity_weighted_historical_evidence",
            "historical_observed_dose_count": 0,
            "additional_evaluation_dose_count": len(valid_additional),
        }

    observed_doses = [dose for dose, _weight in weighted_doses]
    positive_weights = [max(0.0, weight) for _dose, weight in weighted_doses]
    total_weight = sum(positive_weights)
    if total_weight > 0:
        weighted_pairs = sorted(
            zip(observed_doses, positive_weights, strict=True),
            key=lambda pair: pair[0],
        )

        def weighted_quantile(quantile: float) -> float:
            cutoff = total_weight * quantile
            cumulative = 0.0
            for dose, weight in weighted_pairs:
                cumulative += weight
                if cumulative >= cutoff:
                    return dose
            return weighted_pairs[-1][0]

        minimum = weighted_quantile(0.05)
        maximum = weighted_quantile(0.95)
    else:
        minimum = min(observed_doses)
        maximum = max(observed_doses)
    minimum = min(minimum, *valid_additional) if valid_additional else minimum
    maximum = max(maximum, *valid_additional) if valid_additional else maximum
    if minimum == maximum and len(set(observed_doses)) > 1:
        minimum = min(observed_doses)
        maximum = max(observed_doses)
    if minimum == maximum:
        generated = [minimum]
    else:
        generated = [
            minimum + (maximum - minimum) * index / (grid_points - 1)
            for index in range(grid_points)
        ]

    observed_within_display_range = [
        dose for dose in observed_doses if minimum <= dose <= maximum
    ]
    all_doses = tuple(
        sorted(set([*generated, *observed_within_display_range, *valid_additional]))
    )
    return all_doses, {
        "mode": "automatic_from_similarity_weighted_historical_evidence",
        "historical_observed_dose_count": len(observed_doses),
        "historical_observed_dose_min": min(observed_doses),
        "historical_observed_dose_max": max(observed_doses),
        "similarity_weighted_display_dose_min": minimum,
        "similarity_weighted_display_dose_max": maximum,
        "generated_grid_point_count": len(generated),
        "additional_evaluation_dose_count": len(valid_additional),
    }


def article_segments(
    article: dict[str, Any], spec: ToxicityQuery
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    segments,reasons=candidate_segments(article["data"],target_unit=spec.target_unit)
    if 'allowed_scopes' in article:
        allowed=set(article['allowed_scopes'])
        removed=[s for s in segments if s[0].get('arm_or_cohort') not in allowed]
        segments=[s for s in segments if s[0].get('arm_or_cohort') in allowed]
        if removed:reasons=[*reasons,'configuration_not_selected']
    return segments,reasons


def _component_dose_in_target_unit(
    data: dict[str, Any],
    *,
    segment: list[dict[str, Any]],
    target_unit: str,
    components_field: str,
    fallback_value_and_unit,
    require_component_regimen_match: bool = False,
) -> tuple[float | None, str | None]:
    """Resolve one unambiguous segment-specific component dose.

    MTD/MAD fields are often article-level.  They may be used for a segment
    only when one component can be matched by drug and, when needed, regimen;
    otherwise an article-level value could censor or annotate the wrong arm.
    """

    if not segment:
        return None, "empty_segment"
    drug_name = segment[0].get("drug_name")
    if not isinstance(drug_name, str) or not drug_name:
        return None, "segment_missing_drug_name"
    same_drug_segments, _reasons = candidate_segments(
        data,
        target_unit=target_unit,
        curve_drug=drug_name,
    )
    segment_regimen = str(segment[0].get("regimen_signature", ""))
    segment_arm = normalized_key(segment[0].get("arm_or_cohort"))

    components = [
        component
        for component in data.get(components_field) or []
        if isinstance(component, dict)
    ]
    matching_components = [
        component
        for component in components
        if normalized_key(component.get("Drug_Name")) == normalized_key(drug_name)
    ]
    explicitly_armed_components = [
        component
        for component in matching_components
        if normalized_key(component.get("arm_or_cohort"))
    ]
    if explicitly_armed_components:
        matching_components = [
            component
            for component in explicitly_armed_components
            if normalized_key(component.get("arm_or_cohort")) == segment_arm
        ]
    if matching_components:
        if len(matching_components) == 1:
            component = matching_components[0]
            component_regimen = regimen_signature(
                component.get("Frequency"), component.get("Schedule")
            )
            component_arm = normalized_key(component.get("arm_or_cohort"))
            matching_regimen_segments = [
                candidate
                for candidate in same_drug_segments
                if str(candidate[0].get("regimen_signature", ""))
                == component_regimen
                and (
                    not component_arm
                    or normalized_key(candidate[0].get("arm_or_cohort"))
                    == component_arm
                )
            ]
            if (
                (
                    require_component_regimen_match
                    and component_regimen != segment_regimen
                )
                or (
                    len(same_drug_segments) > 1
                    and (
                        component_regimen != segment_regimen
                        or len(matching_regimen_segments) != 1
                    )
                )
            ):
                return None, "ambiguous_component_regimen_for_segment"
        else:
            regimen_matches = [
                component
                for component in matching_components
                if regimen_signature(
                    component.get("Frequency"), component.get("Schedule")
                )
                == segment_regimen
            ]
            if len(regimen_matches) != 1:
                return None, "ambiguous_component_regimen_for_segment"
            component = regimen_matches[0]
        value = scalar_dose(component)
        unit = field_value(component.get("Dose_Unit"))
    elif components:
        return None, "no_matching_component_for_segment_drug"
    else:
        if len(same_drug_segments) != 1:
            return None, "article_level_value_ambiguous_across_segments"
        value, unit = fallback_value_and_unit(data)
    canonical_value, canonical_unit = canonical_dose_value(value, unit)
    if canonical_unit != target_unit:
        return None, "component_dose_not_convertible_to_target_unit"
    if canonical_value is None:
        return None, "component_dose_missing_or_invalid"
    return canonical_value, None


_EXPLICIT_MTD_EVIDENCE_RE = re.compile(
    r"(?:\bMTD\b|maximum[- ]tolerat(?:ed|able)\s+dose|"
    r"maximally[- ]tolerated\s+dose)",
    flags=re.IGNORECASE,
)


def _field_evidence(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    evidence = value.get("evidence")
    return evidence.strip() if isinstance(evidence, str) else ""


def _mtd_supporting_evidence(
    data: dict[str, Any], segment: list[dict[str, Any]]
) -> str:
    """Collect source snippets that specifically support the resolved MTD dose."""

    evidence = [
        _field_evidence(data.get("MTD_Value")),
        _field_evidence(data.get("MTD_Unit")),
    ]
    drug_name = normalized_key(segment[0].get("drug_name"))
    segment_arm = normalized_key(segment[0].get("arm_or_cohort"))
    segment_regimen = str(segment[0].get("regimen_signature", ""))
    for component in data.get("MTD_components") or []:
        if not isinstance(component, dict):
            continue
        if normalized_key(component.get("Drug_Name")) != drug_name:
            continue
        component_arm = normalized_key(component.get("arm_or_cohort"))
        if component_arm and component_arm != segment_arm:
            continue
        if (
            regimen_signature(component.get("Frequency"), component.get("Schedule"))
            != segment_regimen
        ):
            continue
        component_evidence = component.get("evidence")
        if isinstance(component_evidence, str):
            evidence.append(component_evidence.strip())
    return " | ".join(text for text in evidence if text)


def segment_mad_resolution(
    article: dict[str, Any],
    segment: list[dict[str, Any]],
    spec: ToxicityQuery,
) -> tuple[float | None, str | None]:
    """Return a valid MAD cap and explicit reason when it cannot be used."""

    value, reason = _component_dose_in_target_unit(
        article["data"],
        segment=segment,
        target_unit=spec.target_unit,
        components_field="MAD_components",
        fallback_value_and_unit=mad_value_and_unit,
    )
    if value is None:
        return None, reason
    observed_max = max(float(row["canonical_dose"]) for row in segment)
    if value < observed_max:
        return None, "mad_below_segment_observed_maximum"
    return value, None


def segment_mtd_resolution(
    article: dict[str, Any],
    segment: list[dict[str, Any]],
    target_unit: str,
) -> tuple[float | None, str | None]:
    """Return a conservatively verified MTD annotation for one segment.

    Recommended phase-2 doses, maximum administered/tested doses, and MTCDs
    are clinically distinct endpoints.  They are not accepted merely because
    an extractor placed their numeric value in ``MTD_Value``.
    """

    data = article["data"]
    status = mtd_missing_status(data)
    if status == "not_reached":
        return None, "mtd_not_reached"
    if status != "not_missing":
        return None, f"mtd_status_{status}"

    value, reason = _component_dose_in_target_unit(
        data,
        segment=segment,
        target_unit=target_unit,
        components_field="MTD_components",
        fallback_value_and_unit=mtd_value_and_unit,
        require_component_regimen_match=True,
    )
    if value is None:
        return None, reason

    evidence = _mtd_supporting_evidence(data, segment)
    if not _EXPLICIT_MTD_EVIDENCE_RE.search(evidence):
        return None, "mtd_not_explicitly_supported_by_evidence"
    return value, None


def segment_mad_in_target_unit(
    article: dict[str, Any],
    segment: list[dict[str, Any]],
    spec: ToxicityQuery,
) -> float | None:
    """Return MAD for the actual historical escalated drug in one segment."""

    return segment_mad_resolution(article, segment, spec)[0]


def segment_mtd_in_target_unit(
    article: dict[str, Any],
    segment: list[dict[str, Any]],
    target_unit: str,
) -> float | None:
    """Return the article MTD for this segment drug if it matches target_unit."""

    return segment_mtd_resolution(article, segment, target_unit)[0]


def is_not_reached(article: dict[str, Any]) -> bool:
    return mtd_missing_status(article["data"]) == "not_reached"


def segment_is_all_zero(segment: list[dict[str, Any]]) -> bool:
    return bool(segment) and all(row["n_patients_with_dlt"] == 0 for row in segment)


def segment_is_not_reached_all_zero(
    article: dict[str, Any],
    segment: list[dict[str, Any]],
    spec: ToxicityQuery,
) -> bool:
    """True for a not-reached historical arm with real zero-event evidence."""

    mad, _reason = segment_mad_resolution(article, segment, spec)
    return (
        is_not_reached(article)
        and mad is not None
        and segment_is_all_zero(segment)
    )


def split_article_weight(
    article_weight: float, segments: list[list[dict[str, Any]]]
) -> float:
    return article_weight / len(segments) if segments else 0.0
