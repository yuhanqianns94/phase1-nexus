"""Convert resolved v3 entities into prediction-ready, scope-safe evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from phase1_extraction.artifacts import ArticlePaths, read_json, sha256_file
from phase1_extraction.friendly_review import FriendlyArticleReview
from phase1_extraction.models import CurveRow, ReportedExtraction

from .models import EvidencePoint, EvidencePolicy, EvidenceSeries, LibrarySnapshot, ReviewPolicy
from .compact_adapter import matching_data, publication_matching_data


def _norm(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip().casefold()


def _join(values: Iterable[str | None]) -> str:
    return " · ".join(dict.fromkeys(cleaned for value in values if (cleaned := " ".join(str(value or "").split()))))


def canonical_dose_unit(value: str) -> tuple[str, str | None]:
    """Clean layout/schedule contamination without numerically rescaling a dose."""
    source = " ".join(value.split()).strip()
    compact = source.casefold().replace("μ", "u").replace("µ", "u").replace("²", "2").replace("^2", "2").replace(" ", "")
    aliases = {"mcg": "ug"}
    match = re.match(r"^(mcg|ug|mg|g)(/(?:m2|kg))?(/(?:day|dose|week|cycle))?", compact)
    if not match:
        return compact, None
    mass = aliases.get(match.group(1), match.group(1))
    canonical = mass + (match.group(2) or "") + (match.group(3) or "")
    remainder = compact[match.end():]
    return canonical, remainder or None


def _fingerprint(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def discover_article_paths(roots: Iterable[Path]) -> tuple[dict[str, ArticlePaths], dict[str, tuple[str, ...]]]:
    """Select the first root containing each document and report other versions."""
    selected: dict[str, ArticlePaths] = {}
    locations: dict[str, list[str]] = defaultdict(list)
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        candidates = [root] if (root / "reported_extraction.json").is_file() else [path.parent for path in root.glob("*/reported_extraction.json")]
        for article_root in sorted(candidates):
            if not (article_root / "curve_rows.json").is_file():
                continue
            document_id = article_root.name
            locations[document_id].append(str(article_root))
            selected.setdefault(document_id, ArticlePaths(article_root.parent, document_id))
    duplicates = {key: tuple(value) for key, value in locations.items() if len(value) > 1}
    return selected, duplicates


def _component_intervention(extraction: ReportedExtraction, regimen_id: str, component_id: str) -> tuple[str, Any | None]:
    interventions = {item.intervention_id: item for item in extraction.interventions}
    regimen = next((item for item in extraction.regimens if item.regimen_id == regimen_id), None)
    if regimen is None:
        return "Unknown intervention", None
    component = next((item for item in regimen.components if item.component_id == component_id), None)
    intervention = interventions.get(component.intervention_id) if component else None
    return (intervention.name_raw if intervention else component_id), intervention


def _review_overrides(paths: ArticlePaths) -> tuple[FriendlyArticleReview | None, dict[str, str]]:
    if not paths.friendly_human_review.is_file():
        return None, {}
    try:
        review = FriendlyArticleReview.model_validate(read_json(paths.friendly_human_review))
    except (OSError, ValueError):
        return None, {}
    overrides = {
        unit.module_id: unit.gold_answer
        for unit in review.units
        if unit.unit_kind == "text_field"
        and unit.decision in {"incorrect", "missing"}
        and unit.gold_answer
    }
    return review, overrides


def _features(extraction: ReportedExtraction, row: CurveRow, overrides: dict[str, str]) -> tuple[str, str, dict[str, str]]:
    series = next((item for item in extraction.dose_series if item.series_id == row.series_id), None)
    regimen = next((item for item in extraction.regimens if item.regimen_id == row.regimen_id), None)
    intervention_name, intervention = _component_intervention(extraction, row.regimen_id, row.varying_component_id)
    populations = [item for item in extraction.populations if item.study_id == row.study_id and item.part_id in {None, row.part_id}]
    windows = [item for item in extraction.dlt_windows if item.window_id == row.window_id]
    definitions = [item for item in extraction.dlt_definitions if item.study_id == row.study_id and item.part_id in {None, row.part_id}]
    designs = [item for item in extraction.designs if item.study_id == row.study_id and item.part_id in {None, row.part_id}]
    arm = next((item for item in extraction.arms if item.arm_id == row.arm_id), None)
    component = next((component for candidate in extraction.regimens if candidate.regimen_id == row.regimen_id for component in candidate.components if component.component_id == row.varying_component_id), None)
    dosing_phases = component.dosing_phases if component else []
    disease = _join([item.disease_core or item.disease_raw for item in populations] + ([arm.population_description] if arm else []))
    age = _join(f"{item.age_min_comparator or '>='} {item.age_min:g} to {item.age_max_comparator or '<='} {item.age_max:g} {item.age_unit or ''}" if item.age_min is not None and item.age_max is not None else None for item in populations)
    eligibility = _join(value for item in populations for value in [*item.inclusion_criteria, *item.exclusion_criteria, *item.prior_treatment_requirements, *item.clinical_requirements])
    dlt_definition = _join(value for item in definitions for value in [*item.criteria_raw, *item.exceptions_raw, item.logic_raw, item.grading_system_raw])
    dlt_window = _join(item.raw_definition for item in windows)
    route = _join(item.route_raw for item in dosing_phases)
    schedule = _join(_join([item.frequency_raw, item.administration_days_raw, item.cycle_length_raw, item.duration_raw]) for item in dosing_phases)
    dose_form = _join(item.dose_form_raw for item in dosing_phases)
    mechanism = _join([*(intervention.mechanism_raw if intervention else []), *(intervention.targets_raw if intervention else []), *(intervention.pathways_raw if intervention else [])])
    trial_type = regimen.composition if regimen else "unclear"
    features = {
        "drug": intervention_name,
        "cancer_type": disease,
        "route": route,
        "schedule": schedule,
        "schema_embedding": _join([regimen.label if regimen else None, regimen.shared_schedule_raw if regimen else None]),
        "dlt_definition": dlt_definition,
        "dlt_eva": dlt_window,
        "target_moa": mechanism,
        "age": age,
        "trial_type": trial_type,
        "eligibility_embedding": eligibility,
        "gender": _join(item.sex_eligibility_raw for item in populations),
        "dose_form": dose_form,
    }
    # Human text corrections improve matching without being parsed into invented structured facts.
    module_fields = {
        "population": ("cancer_type", "age", "gender"),
        "eligibility": ("eligibility_embedding",),
        "intervention": ("drug", "target_moa"),
        "dosing": ("route", "schedule", "dose_form"),
        "parts_arms": ("schema_embedding",),
        "dlt_definition": ("dlt_definition",),
        "dlt_window": ("dlt_eva",),
    }
    for module_id, answer in overrides.items():
        for field in module_fields.get(module_id, ()):
            features[field] = answer
    return (series.label if series else row.series_id), intervention_name, features


def _reviewed_rows(
    extraction: ReportedExtraction,
    raw_rows: list[CurveRow],
    review: FriendlyArticleReview | None,
) -> tuple[list[CurveRow], list[dict[str, Any]], Counter[str]]:
    if review is None:
        return raw_rows, [], Counter()
    units_by_scope = {
        (unit.predicted_point.series_id, unit.predicted_point.cohort_id): unit
        for unit in review.units
        if unit.unit_kind == "curve_point" and unit.origin == "prediction" and unit.predicted_point and unit.predicted_point.cohort_id
    }
    series_by_id = {item.series_id: item for item in extraction.dose_series}
    resolved: list[CurveRow] = []
    counter: Counter[str] = Counter()
    for row in raw_rows:
        unit = units_by_scope.get((row.series_id, row.cohort_id))
        if unit is None or unit.decision in {"pending", "correct", "excluded"}:
            resolved.append(row)
            continue
        if unit.decision == "spurious":
            counter["review_removed_spurious"] += 1
            continue
        if unit.decision == "incorrect" and unit.gold_point:
            point = unit.gold_point
            target_series_id = point.series_id or row.series_id
            target_series = series_by_id.get(target_series_id)
            if target_series is None:
                resolved.append(row)
                counter["invalid_review_series_correction"] += 1
                continue
            update = {
                "study_id": target_series.study_id,
                "part_id": target_series.part_id,
                "arm_id": target_series.arm_id,
                "series_id": target_series.series_id,
                "regimen_id": target_series.regimen_id,
                "varying_component_id": target_series.varying_component_id,
                "window_id": target_series.window_id,
                "dose_value": point.dose_value,
                "dose_unit": point.dose_unit,
                "dose_raw": point.dose_raw,
                "exposures": [item.model_copy(deep=True) for item in point.exposures],
                "exposure_coverage": point.exposure_coverage,
                "exposure_issues": list(point.exposure_issues),
                "n": point.n,
                "y": point.y,
                "rate": point.y / point.n if point.y is not None and point.n else None,
                "denominator_role": point.denominator_type if point.denominator_type in {"paired_dlt", "dlt_evaluable", "treated", "equivalent_treated", "design_assumed_cohort", "unknown"} else "unknown",
            }
            resolved.append(row.model_copy(update=update))
            counter["review_corrected_curve_row"] += 1
            continue
        # A malformed or unfinished correction must not silently remove a usable
        # extracted row from the prediction library.
        resolved.append(row)
        counter["invalid_review_curve_correction"] += 1
    manual: list[dict[str, Any]] = []
    for unit in review.units:
        if unit.unit_kind == "curve_point" and unit.origin == "manual_missing" and unit.decision == "missing" and unit.gold_point:
            manual.append({"unit": unit, "point": unit.gold_point})
    return resolved, manual, counter


def _article_series(
    paths: ArticlePaths, evidence_policy: EvidencePolicy, review_policy: ReviewPolicy,
) -> tuple[list[EvidenceSeries], Counter[str], Counter[str]]:
    extraction = ReportedExtraction.model_validate(read_json(paths.reported_extraction))
    payload = read_json(paths.curve_rows)
    rows = [CurveRow.model_validate(item) for item in payload.get(evidence_policy, [])]
    review, overrides = _review_overrides(paths) if review_policy != "extracted" else (None, {})
    skipped: Counter[str] = Counter()
    notices: Counter[str] = Counter()
    if review_policy == "reviewed_only" and (
        review is None
        or not review.module_complete.get("curve_results")
        or any(unit.module_id == "curve_results" and unit.decision == "pending" for unit in review.units)
    ):
        skipped["article_curve_review_incomplete"] += 1
        return [], skipped, notices
    manual: list[dict[str, Any]] = []
    if review_policy != "extracted":
        rows, manual, review_counts = _reviewed_rows(extraction, rows, review)
        notices.update(review_counts)
    grouped: dict[tuple[str, ...], list[CurveRow]] = defaultdict(list)
    for row in rows:
        if row.dose_value is None or row.dose_value <= 0 or not row.dose_unit or row.dose_unit == "unknown" or row.n is None or row.y is None:
            skipped["invalid_curve_row"] += 1
            continue
        if row.n <= 0 or row.y < 0 or row.y > row.n:
            skipped["invalid_binomial_counts"] += 1
            continue
        canonical_unit, _qualifier = canonical_dose_unit(row.dose_unit)
        if not canonical_unit:
            skipped["invalid_curve_row"] += 1
            continue
        from .profiles import background_signature
        background = background_signature(row.exposures, row.varying_component_id)
        key = (row.study_id, row.part_id or "", row.arm_id or "", row.series_id, row.regimen_id, row.varying_component_id, row.window_id or "", background, canonical_unit)
        grouped[key].append(row)
    result: list[EvidenceSeries] = []
    article_fingerprint = _fingerprint({"extraction": sha256_file(paths.reported_extraction), "curves": sha256_file(paths.curve_rows), "review": sha256_file(paths.friendly_human_review) if review_policy != "extracted" and paths.friendly_human_review.is_file() else None})
    for key, members in sorted(grouped.items()):
        first = members[0]
        label, intervention_name, features = _features(extraction, first, overrides if review_policy != "extracted" else {})
        source_units = tuple(dict.fromkeys(row.dose_unit for row in members))
        qualifiers = [canonical_dose_unit(value)[1] for value in source_units]
        if any(qualifiers):
            features["schedule"] = _join([features.get("schedule"), *(value for value in qualifiers if value)])
            notices["dose_unit_schedule_or_label_cleaned"] += sum(value is not None for value in qualifiers)
        corrected_scopes = {
            (item.gold_point.series_id or item.predicted_point.series_id, item.predicted_point.cohort_id)
            for item in (review.units if review else [])
            if item.decision == "incorrect" and item.predicted_point and item.gold_point
        }
        points = tuple(EvidencePoint(
            curve_row_id=row.curve_row_id,
            cohort_id=row.cohort_id,
            dose=float(row.dose_value), unit=key[-1], source_unit=row.dose_unit, n=int(row.n), y=int(row.y),
            denominator_role=row.denominator_role, denominator_basis=row.denominator_basis,
            numerator_basis=row.numerator_basis, rule_ids=tuple(row.rule_ids),
            evidence_source_ids=tuple(row.evidence_source_ids),
            exposures=tuple(item.model_dump(mode="json") for item in row.exposures),
            exposure_coverage=row.exposure_coverage,
            source="human_review" if review_policy != "extracted" and (row.series_id, row.cohort_id) in corrected_scopes else "extraction",
        ) for row in sorted(members, key=lambda item: (float(item.dose_value), item.cohort_id)))
        series_key = "|".join([paths.document_id, *key])
        result.append(EvidenceSeries(
            series_key=series_key, document_id=paths.document_id, study_id=first.study_id,
            part_id=first.part_id, arm_id=first.arm_id, series_id=first.series_id,
            regimen_id=first.regimen_id, varying_component_id=first.varying_component_id,
            window_id=first.window_id, label=label, intervention_name=intervention_name,
            unit=key[-1], source_units=source_units,
            study_group_id="|".join(filter(None, [paths.document_id, first.study_id, first.part_id, first.arm_id])),
            features=features, points=points, source_fingerprint=article_fingerprint,
            review_status="reviewed" if review and review.module_complete.get("curve_results") else "not_reviewed",
            matching_data=matching_data(extraction, first, features, overrides if review_policy != "extracted" else {}),
        ))
    # Manual omissions can be used only when their series association is unambiguous.
    for item in manual:
        point = item["point"]
        candidates = [series for series in result if (point.series_id and series.series_id == point.series_id) or (not point.series_id and _norm(series.label) == _norm(point.series_label))]
        if len(candidates) != 1 or point.dose_value is None or point.n is None or point.y is None or point.n <= 0 or point.y < 0 or point.y > point.n:
            skipped["unmapped_manual_curve_point"] += 1
            continue
        current = candidates[0]
        manual_unit, _manual_qualifier = canonical_dose_unit(point.dose_unit)
        if manual_unit != current.unit:
            skipped["manual_curve_point_unit_mismatch"] += 1
            continue
        new_point = EvidencePoint(
            curve_row_id=item["unit"].review_unit_id, cohort_id=point.cohort_id or item["unit"].review_unit_id,
            dose=float(point.dose_value), unit=current.unit, source_unit=point.dose_unit, n=int(point.n), y=int(point.y),
            denominator_role=point.denominator_type, denominator_basis="reported", numerator_basis="reported",
            source="human_review",
            exposures=tuple(e.model_dump(mode="json") for e in point.exposures),
            exposure_coverage=point.exposure_coverage,
        )
        replacement = EvidenceSeries(**{**current.__dict__, "points": tuple(sorted([*current.points, new_point], key=lambda value: (value.dose, value.cohort_id)))})
        result[result.index(current)] = replacement
    return result, skipped, notices


def load_evidence_library(
    roots: Iterable[Path], *, evidence_policy: EvidencePolicy = "project_default",
    review_policy: ReviewPolicy = "extracted",
) -> LibrarySnapshot:
    if evidence_policy not in {"project_default", "reported_only"}:
        raise ValueError("Unsupported evidence policy.")
    if review_policy not in {"extracted", "reviewed_if_available", "reviewed_only"}:
        raise ValueError("Unsupported review policy.")
    # Human gold is reserved for extraction-accuracy evaluation. Old callers and
    # saved settings may still send a review policy; it must not change evidence.
    review_policy = "extracted"
    normalized_roots = tuple(str(Path(root).expanduser().resolve()) for root in roots)
    selected, duplicates = discover_article_paths(Path(value) for value in normalized_roots)
    series: list[EvidenceSeries] = []
    matching_records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    notices: Counter[str] = Counter()
    for paths in selected.values():
        try:
            article_series, article_skips, article_notices = _article_series(paths, evidence_policy, review_policy)
            series.extend(article_series)
            skipped.update(article_skips)
            notices.update(article_notices)
            if not article_skips.get("article_curve_review_incomplete"):
                extraction = ReportedExtraction.model_validate(read_json(paths.reported_extraction))
                _, overrides = _review_overrides(paths) if review_policy != "extracted" else (None, {})
                matching_records.append({"pmcid": paths.document_id, "data": publication_matching_data(extraction, overrides),
                                         "source_fingerprint": sha256_file(paths.reported_extraction)})
        except (OSError, ValueError, KeyError, TypeError):
            skipped["invalid_article_artifacts"] += 1
    identity = [{"series_key": item.series_key, "fingerprint": item.source_fingerprint, "points": [point.__dict__ for point in item.points]} for item in series]
    return LibrarySnapshot(
        evidence_policy=evidence_policy, review_policy=review_policy,
        source_roots=normalized_roots, series=tuple(series), document_count=len(selected),
        duplicate_documents=duplicates, skipped=dict(sorted(skipped.items())),
        notices=dict(sorted(notices.items())), fingerprint=_fingerprint({"series": identity, "matching_records": matching_records}),
        matching_records=tuple(matching_records),
    )
