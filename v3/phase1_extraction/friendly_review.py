"""User-facing, single-reviewer validation for phase-I extractions.

The extraction schema is intentionally detailed.  The review UI is not: it
collapses that schema into nine clinical text fields plus a dose--DLT matrix.
Text fields and curve points are scored separately because they have different
granularity.  Legacy ``human_review.json`` files are never read or overwritten.
"""

from __future__ import annotations

import csv
import hashlib
from .drug_ordering import clinical_payload
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import ArticlePaths, read_json, sha256_file, write_json
from .models import CohortExposure, EvidenceLink, ReportedExtraction
from .exposures import exposure_identity, exposure_text, exposure_schedule


FRIENDLY_REVIEW_FORMAT = "phase1_friendly_review_v1"
FRIENDLY_REVIEW_POLICY = "phase1_friendly_field_and_curve_prf_v2"
LEGACY_FRIENDLY_REVIEW_POLICY = "phase1_friendly_field_and_curve_prf_v1"

Decision = Literal["pending", "correct", "incorrect", "spurious", "missing", "excluded"]
Origin = Literal["prediction", "manual_missing"]
UnitKind = Literal["text_field", "curve_point"]


MODULES: tuple[tuple[str, str, UnitKind], ...] = (
    ("population", "Disease and Study Population", "text_field"),
    ("eligibility", "Eligibility", "text_field"),
    ("intervention", "Intervention and Mechanism", "text_field"),
    ("dosing", "Dosage and Administration", "text_field"),
    ("parts_arms", "Study Parts and Patient Groups", "text_field"),
    ("design", "Dose-Escalation Design", "text_field"),
    ("population_counts", "Population Counts", "text_field"),
    ("dlt_definition", "DLT Definition and Manifestations", "text_field"),
    ("dlt_window", "DLT Assessment Window", "text_field"),
    ("curve_results", "Dose-DLT Results", "curve_point"),
)

MODULE_LABELS = {module_id: label for module_id, label, _kind in MODULES}
TEXT_MODULE_IDS = tuple(module_id for module_id, _label, kind in MODULES if kind == "text_field")
CURVE_MODULE_ID = "curve_results"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CurvePointAnswer(_StrictModel):
    cohort_id: str | None = None
    series_id: str | None = None
    series_label: str
    regimen_label: str | None = None
    part_label: str | None = None
    arm_label: str | None = None
    dose_value: float | None = None
    dose_unit: str
    dose_raw: str
    n: int | None = None
    y: int | None = None
    denominator_type: str = "unknown"
    treated_n: int | None = None
    dlt_evaluable_n: int | None = None
    window_label: str | None = None
    value_basis: str = "reported"
    curve_status: str = "included"
    association_note: str | None = None
    exposures: list[CohortExposure] = Field(default_factory=list)
    exposure_coverage: Literal["complete", "partial", "unknown"] = "unknown"
    exposure_issues: list[str] = Field(default_factory=list)
    varying_component_id: str | None = None

    def display(self) -> str:
        dose = f"{self.dose_value:g} {self.dose_unit}" if self.dose_value is not None else self.dose_raw
        outcome = f"{self.y if self.y is not None else '?'} / {self.n if self.n is not None else '?'}"
        configuration = " + ".join(exposure_text(item) for item in self.exposures) or dose
        return f"{self.series_label} | {configuration} | DLT/N = {outcome}"


class FriendlyReviewUnit(_StrictModel):
    review_unit_id: str
    module_id: str
    module_label: str
    unit_kind: UnitKind
    predicted_answer: str | None = None
    predicted_point: CurvePointAnswer | None = None
    gold_answer: str | None = None
    gold_point: CurvePointAnswer | None = None
    decision: Decision = "pending"
    origin: Origin = "prediction"
    source_record_ids: list[str] = Field(default_factory=list)
    evidence_source_ids: list[str] = Field(default_factory=list)
    evidence_text: list[str] = Field(default_factory=list)
    content_fingerprint: str
    note: str | None = None
    updated_at_utc: str | None = None

    def display_answer(self, *, gold: bool = False) -> str:
        if self.unit_kind == "curve_point":
            point = self.gold_point if gold else self.predicted_point
            return point.display() if point is not None else ""
        return (self.gold_answer if gold else self.predicted_answer) or ""


class FriendlyArticleReview(_StrictModel):
    format: Literal["phase1_friendly_review_v1"] = FRIENDLY_REVIEW_FORMAT
    metric_policy: Literal[
        "phase1_friendly_field_and_curve_prf_v1",
        "phase1_friendly_field_and_curve_prf_v2",
    ] = FRIENDLY_REVIEW_POLICY
    document_id: str
    source_fingerprint: str
    reviewer: str = ""
    module_complete: dict[str, bool] = Field(default_factory=dict)
    units: list[FriendlyReviewUnit] = Field(default_factory=list)
    created_at_utc: str
    updated_at_utc: str


class FriendlyReviewUndo(_StrictModel):
    previous_review: FriendlyArticleReview
    saved_state_fingerprint: str
    module_id: str | None = None


class MetricCounts(_StrictModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    pending: int = 0
    excluded: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class DomainSummary(_StrictModel):
    counts: MetricCounts
    all_predictions_reviewed: bool
    completeness_checked: bool
    precision_is_final: bool
    recall_is_final: bool


class FriendlyReviewSummary(_StrictModel):
    format: Literal["phase1_friendly_review_summary_v1"] = "phase1_friendly_review_summary_v1"
    metric_policy: Literal["phase1_friendly_field_and_curve_prf_v2"] = FRIENDLY_REVIEW_POLICY
    document_id: str
    source_fingerprint: str
    reviewer: str
    completed_modules: int
    total_modules: int
    text_fields: DomainSummary
    curve_points: DomainSummary
    by_module: dict[str, MetricCounts]
    generated_at_utc: str


class FriendlyCorpusSummary(_StrictModel):
    format: Literal["phase1_friendly_review_corpus_summary_v1"] = "phase1_friendly_review_corpus_summary_v1"
    metric_policy: Literal["phase1_friendly_field_and_curve_prf_v2"] = FRIENDLY_REVIEW_POLICY
    article_count: int
    text_fields: MetricCounts
    curve_points: MetricCounts
    fully_completed_articles: int
    generated_at_utc: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _unique(values: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value or "").split()).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _evidence(*groups: Iterable[EvidenceLink]) -> tuple[list[str], list[str]]:
    source_ids: list[str] = []
    texts: list[str] = []
    for links in groups:
        for link in links:
            source_ids.extend(link.source_ids)
            if link.source_text:
                texts.append(link.source_text)
            if link.locator:
                texts.append(link.locator)
    return _unique(source_ids), _unique(texts)


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "Not reported"
    return f"{value:g}"


def _block(title: str, values: Iterable[str | None]) -> list[str]:
    cleaned = _unique(values)
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return [f"**{title}**", "", cleaned[0], ""]
    return [f"**{title}**", "", *[f"- {value}" for value in cleaned], ""]


def _block_unseen(title: str, values: Iterable[str | None], seen: set[str]) -> list[str]:
    cleaned: list[str] = []
    for value in _unique(values):
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(value)
    return _block(title, cleaned)


def _text_unit(
    extraction: ReportedExtraction,
    module_id: str,
    lines: Iterable[str],
    record_ids: Iterable[str],
    evidence_groups: Iterable[Iterable[EvidenceLink]],
) -> FriendlyReviewUnit:
    answer = "\n".join(lines).strip() or None
    evidence_ids, evidence_text = _evidence(*evidence_groups)
    payload = {
        "module_id": module_id,
        "answer": answer,
        "record_ids": _unique(record_ids),
        "evidence_ids": evidence_ids,
        "evidence_text": evidence_text,
        "policy": FRIENDLY_REVIEW_POLICY,
    }
    return FriendlyReviewUnit(
        review_unit_id=f"fr_text_{module_id}",
        module_id=module_id,
        module_label=MODULE_LABELS[module_id],
        unit_kind="text_field",
        predicted_answer=answer,
        source_record_ids=payload["record_ids"],
        evidence_source_ids=evidence_ids,
        evidence_text=evidence_text,
        content_fingerprint=_canonical_hash(payload),
    )


def _build_text_units(extraction: ReportedExtraction) -> list[FriendlyReviewUnit]:
    units: list[FriendlyReviewUnit] = []

    population_lines: list[str] = []
    for study in extraction.studies:
        population_lines.extend(_block("Study", [study.label]))
        population_lines.extend(_block("Study purpose", [study.purpose_summary]))
        population_lines.extend(_block("Registration ID", study.registration_ids))
    for population in extraction.populations:
        population_lines.extend(_block("Disease", [population.disease_raw or population.disease_core]))
        population_lines.extend(_block("Disease qualifiers", population.modifiers))
        population_lines.extend(_block("Enrolled population and baseline", population.baseline_summary))
    units.append(
        _text_unit(
            extraction,
            "population",
            population_lines,
            [item.study_id for item in extraction.studies] + [item.population_id for item in extraction.populations],
            [item.evidence for item in extraction.studies] + [item.evidence for item in extraction.populations],
        )
    )

    eligibility_lines: list[str] = []
    eligibility_seen: set[str] = set()
    for population in extraction.populations:
        if population.age_min is not None or population.age_max is not None:
            low = f"{population.age_min_comparator or '>='}{_format_number(population.age_min)}" if population.age_min is not None else "No lower limit"
            high = f"{population.age_max_comparator or '<='}{_format_number(population.age_max)}" if population.age_max is not None else "No upper limit"
            eligibility_lines.extend(_block_unseen("Age", [f"{low} to {high} {population.age_unit or ''}".strip()], eligibility_seen))
        eligibility_lines.extend(_block_unseen("Sex and reproductive requirements", [population.sex_eligibility_raw], eligibility_seen))
        eligibility_lines.extend(_block_unseen("Inclusion criteria", population.inclusion_criteria, eligibility_seen))
        eligibility_lines.extend(_block_unseen("Clinical and organ function requirements", population.clinical_requirements, eligibility_seen))
        eligibility_lines.extend(_block_unseen("Prior therapy and washout requirements", population.prior_treatment_requirements, eligibility_seen))
        eligibility_lines.extend(_block_unseen("Exclusion criteria", population.exclusion_criteria, eligibility_seen))
    units.append(
        _text_unit(extraction, "eligibility", eligibility_lines, [p.population_id for p in extraction.populations], [p.evidence for p in extraction.populations])
    )

    intervention_lines: list[str] = []
    for item in extraction.interventions:
        intervention_lines.extend(_block("Drug or intervention", [item.name_raw]))
        intervention_lines.extend(_block("Aliases", item.aliases))
        intervention_lines.extend(_block("Modality", [item.modality]))
        intervention_lines.extend(_block("Product or formulation", [item.product_or_formulation]))
        intervention_lines.extend(_block("Drug class", [item.drug_class_raw]))
        intervention_lines.extend(_block("Mechanism of action", item.mechanism_raw))
        intervention_lines.extend(_block("Targets", item.targets_raw))
        intervention_lines.extend(_block("Pathways", item.pathways_raw))
    units.append(
        _text_unit(extraction, "intervention", intervention_lines, [i.intervention_id for i in extraction.interventions], [i.evidence for i in extraction.interventions])
    )

    intervention_names = {item.intervention_id: item.name_raw for item in extraction.interventions}
    dosing_lines: list[str] = []
    dosing_evidence: list[Iterable[EvidenceLink]] = []
    dosing_ids: list[str] = []
    for regimen in extraction.regimens:
        dosing_ids.append(regimen.regimen_id)
        dosing_evidence.append(regimen.evidence)
        dosing_lines.extend(_block("Regimen", [regimen.label]))
        dosing_lines.extend(_block("Shared schedule", [regimen.shared_schedule_raw]))
        for component in regimen.components:
            dosing_ids.append(component.component_id)
            dosing_lines.extend(_block("Regimen component", [intervention_names.get(component.intervention_id, component.intervention_id)]))
            for phase in component.dosing_phases:
                dosing_ids.append(phase.dosing_phase_id)
                dosing_evidence.append(phase.evidence)
                values = [
                    f"Dose = {phase.dose.raw_expression}" if phase.dose else None,
                    f"Route = {phase.route_raw}" if phase.route_raw else None,
                    f"Dosage form = {phase.dose_form_raw}" if phase.dose_form_raw else None,
                    f"Frequency = {phase.frequency_raw}" if phase.frequency_raw else None,
                    f"Administration days = {phase.administration_days_raw}" if phase.administration_days_raw else None,
                    f"Cycle length = {phase.cycle_length_raw}" if phase.cycle_length_raw else None,
                    f"Duration = {phase.duration_raw}" if phase.duration_raw else None,
                    f"Infusion details = {phase.infusion_duration_raw}" if phase.infusion_duration_raw else None,
                    f"Infusion rate = {phase.infusion_rate_raw}" if phase.infusion_rate_raw else None,
                    f"Start condition = {phase.start_condition_raw}" if phase.start_condition_raw else None,
                    f"End condition = {phase.end_condition_raw}" if phase.end_condition_raw else None,
                ]
                dosing_lines.extend(_block(phase.phase_label or "Dosing phase", values))
        for rule in regimen.dose_adjustment_rules:
            dosing_ids.append(rule.rule_id)
            dosing_evidence.append(rule.evidence)
            dosing_lines.extend(
                _block(
                    "Dose adjustment rule",
                    [f"{rule.action}: {rule.condition_raw}; {rule.action_raw or 'action detail not reported'}"],
                )
            )
    configurations: dict[str, tuple[list[str], list[CohortExposure]]] = {}
    for cohort in extraction.cohorts:
        if cohort.actual_status != "actual" or not cohort.exposures:
            continue
        identity = exposure_identity(cohort.exposures)
        if identity not in configurations:
            configurations[identity] = ([], cohort.exposures)
        configurations[identity][0].append(cohort.label)
        dosing_ids.append(cohort.cohort_id)
    for labels, exposures in configurations.values():
        values = []
        for exposure in exposures:
            schedule = exposure_schedule(exposure)
            values.append(exposure_text(exposure) + (f" — {schedule}" if schedule else ""))
            dosing_evidence.append(exposure.evidence)
            dosing_evidence.extend(phase.evidence for phase in exposure.dosing_phases)
        dosing_lines.extend(_block("Cohort configuration: " + "; ".join(labels), values))
    units.append(_text_unit(extraction, "dosing", dosing_lines, dosing_ids, dosing_evidence))

    parts_lines: list[str] = []
    for part in sorted(extraction.study_parts, key=lambda item: item.sequence_index or 0):
        parts_lines.extend(_block("Study part", [f"{part.sequence_index or '?'}: {part.label} ({part.phase_raw or 'Phase not reported'})"]))
    for arm in extraction.arms:
        parts_lines.extend(_block("Patient group", [f"{arm.label}: {arm.population_description or 'Definition not reported'}"]))
    units.append(
        _text_unit(
            extraction,
            "parts_arms",
            parts_lines,
            [p.part_id for p in extraction.study_parts] + [a.arm_id for a in extraction.arms],
            [p.evidence for p in extraction.study_parts] + [a.evidence for a in extraction.arms],
        )
    )

    design_lines: list[str] = []
    for design in extraction.designs:
        summary = [
            design.raw_name,
            f"Normalized type = {design.normalized_type}",
            f"Initial cohort size = {design.initial_cohort_n}" if design.initial_cohort_n is not None else None,
            f"Additional patients = {design.added_cohort_n}" if design.added_cohort_n is not None else None,
            f"Expanded cohort size = {design.expanded_total_n}" if design.expanded_total_n is not None else None,
            f"Replacement count = {design.replacement_count}" if design.replacement_count is not None else None,
            f"Nonevaluable count = {design.nonevaluable_count}" if design.nonevaluable_count is not None else None,
            f"Actual cohort additions = {design.cohort_addition_count}" if design.cohort_addition_count is not None else None,
            f"Stopping threshold = DLT >= {design.stop_when_y_at_least}" if design.stop_when_y_at_least is not None else None,
        ]
        design_lines.extend(_block("Design", summary))
        design_lines.extend(_block("Dose-escalation and expansion rules", design.decision_rules_raw))
        design_lines.extend(_block("Protocol changes", design.amendments_raw))
    units.append(_text_unit(extraction, "design", design_lines, [d.design_id for d in extraction.designs], [d.evidence for d in extraction.designs]))

    cohort_ids = {cohort.cohort_id for cohort in extraction.cohorts}
    count_lines: list[str] = []
    count_evidence: list[Iterable[EvidenceLink]] = []
    count_ids: list[str] = []
    for population in extraction.analysis_populations:
        count_lines.extend(_block("Analysis population", [f"{population.author_name}: {population.definition_raw or population.interpreted_role}"]))
        count_ids.append(population.analysis_population_id)
        count_evidence.append(population.evidence)
    arm_labels = {arm.arm_id: arm.label for arm in extraction.arms}
    for count in extraction.population_counts:
        if count.cohort_id in cohort_ids:
            continue
        label = arm_labels.get(count.arm_id or "", "Whole study")
        role = count.author_role_raw or count.role
        count_lines.extend(_block("Population count", [f"{label} | {role}: {_format_number(count.reported_value)}"]))
        count_ids.append(count.count_id)
        count_evidence.append(count.evidence)
    units.append(_text_unit(extraction, "population_counts", count_lines, count_ids, count_evidence))

    cohort_labels = {cohort.cohort_id: cohort.label for cohort in extraction.cohorts}
    dlt_lines: list[str] = []
    dlt_evidence: list[Iterable[EvidenceLink]] = []
    dlt_ids: list[str] = []
    for definition in extraction.dlt_definitions:
        dlt_lines.extend(_block("DLT decision logic", [definition.logic_raw]))
        dlt_lines.extend(_block("DLT criteria", definition.criteria_raw))
        dlt_lines.extend(_block("DLT exceptions", definition.exceptions_raw))
        dlt_lines.extend(_block("Toxicity grading system", [definition.grading_system_raw]))
        dlt_ids.append(definition.definition_id)
        dlt_evidence.append(definition.evidence)
    for observation in extraction.dlt_observations:
        if observation.toxicities_raw:
            label = {
                "reported": "Observed DLTs",
                "not_reported": "Toxicity notes — DLT result not reported",
                "explicitly_unreported": "Toxicity notes — DLT result explicitly unreported",
                "unclear": "Toxicity notes — DLT status unclear",
                "explicitly_not_evaluable": "Toxicity notes — DLT not evaluable",
                "not_applicable": "Toxicity notes — DLT not applicable",
                "conflicting": "Toxicity notes — conflicting DLT status",
            }[observation.reporting_status]
            cohort_label = cohort_labels.get(observation.cohort_id or "", "Cohort not identified")
            dlt_lines.extend(_block(f"{label} ({cohort_label})", observation.toxicities_raw))
            if observation.reporting_status != "reported":
                dlt_lines.extend(_block("Reporting note", [observation.provenance.note]))
            dlt_ids.append(observation.observation_id)
            dlt_evidence.append(observation.evidence)
    for unresolved in extraction.unresolved_items:
        if "dlt" in unresolved.category.casefold() or any(record_id in set(dlt_ids) for record_id in unresolved.affected_ids):
            dlt_lines.extend(_block("Source conflict or review note", [unresolved.description]))
            dlt_ids.append(unresolved.item_id)
            dlt_evidence.append(unresolved.evidence)
    units.append(_text_unit(extraction, "dlt_definition", dlt_lines, dlt_ids, dlt_evidence))

    window_lines: list[str] = []
    for window in extraction.dlt_windows:
        values = [
            "Window type = " + {"formal_dlt": "Formal DLT assessment", "safety_monitoring": "Ongoing safety monitoring", "unclear": "Unclear — formal endpoint requires source review"}[window.window_kind],
            window.raw_definition,
            f"Start = {window.start_raw}" if window.start_raw else None,
            f"End = {window.end_raw}" if window.end_raw else None,
            f"Duration basis = {window.basis}",
            f"Assessment cycles = {window.cycle_numbers_raw}" if window.cycle_numbers_raw else None,
            f"Normalized duration = {_format_number(window.normalized_days)} days" if window.normalized_days is not None else None,
        ]
        window_lines.extend(_block("DLT assessment window", values))
    units.append(_text_unit(extraction, "dlt_window", window_lines, [w.window_id for w in extraction.dlt_windows], [w.evidence for w in extraction.dlt_windows]))
    return units


def _reported_curve_payloads(extraction: ReportedExtraction) -> list[dict[str, Any]]:
    observations = {item.cohort_id: item for item in extraction.dlt_observations if item.cohort_id}
    counts_by_id = {item.count_id: item for item in extraction.population_counts}
    counts_by_cohort: dict[str, list[Any]] = defaultdict(list)
    for count in extraction.population_counts:
        if count.cohort_id:
            counts_by_cohort[count.cohort_id].append(count)
    rows: list[dict[str, Any]] = []
    for cohort in extraction.cohorts:
        if cohort.actual_status != "actual":
            continue
        observation = observations.get(cohort.cohort_id)
        denominator = counts_by_id.get(observation.denominator_count_id) if observation and observation.denominator_count_id else None
        cohort_counts = counts_by_cohort.get(cohort.cohort_id, [])
        dlt_count = next((item for item in cohort_counts if item.role == "dlt_evaluable" and item.reported_value is not None), None)
        treated_count = next((item for item in cohort_counts if item.role == "treated" and item.reported_value is not None), None)
        denominator = denominator or dlt_count or treated_count
        y = None
        if observation is not None:
            y = observation.explicit_patient_count
            if y is None and observation.count_kind in {"patients", "dlt_count"}:
                y = observation.reported_count
        evidence_ids, _ = _evidence(
            cohort.evidence,
            observation.evidence if observation else [],
            denominator.evidence if denominator else [],
        )
        rows.append(
            {
                "cohort_id": cohort.cohort_id,
                "series_id": cohort.series_id,
                "regimen_id": cohort.regimen_id,
                "dose_value": cohort.dose.value,
                "dose_unit": cohort.dose.unit_normalized or cohort.dose.unit_raw or "Unit not reported",
                "dose_raw": cohort.dose.raw_expression,
                "n": denominator.reported_value if denominator else None,
                "y": y,
                "denominator_role": denominator.role if denominator else "unknown",
                "reported_treated_n": treated_count.reported_value if treated_count else None,
                "reported_dlt_evaluable_n": dlt_count.reported_value if dlt_count else None,
                "window_id": observation.window_id if observation else None,
                "numerator_basis": "reported" if y is not None else "missing",
                "denominator_basis": "reported" if denominator else "missing",
                "rule_ids": [],
                "evidence_source_ids": evidence_ids,
            }
        )
    return rows


def _curve_payloads(paths: ArticlePaths, extraction: ReportedExtraction) -> list[dict[str, Any]]:
    fallback = _reported_curve_payloads(extraction)
    if not paths.curve_rows.is_file():
        return fallback
    raw = read_json(paths.curve_rows)
    rows = raw.get("all_rows", raw.get("project_default", [])) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return fallback
    resolved = [row for row in rows if isinstance(row, dict)]
    covered = {str(row.get("cohort_id")) for row in resolved if row.get("cohort_id")}
    # Every actual cohort remains visible even when the resolver failed to
    # produce a curve row.  Such rows need human attention most.
    resolved.extend(row for row in fallback if str(row.get("cohort_id")) not in covered)
    return resolved


def _build_curve_units(paths: ArticlePaths, extraction: ReportedExtraction) -> list[FriendlyReviewUnit]:
    cohorts = {item.cohort_id: item for item in extraction.cohorts}
    series = {item.series_id: item for item in extraction.dose_series}
    regimens = {item.regimen_id: item.label for item in extraction.regimens}
    parts = {item.part_id: item.label for item in extraction.study_parts}
    arms = {item.arm_id: item.label for item in extraction.arms}
    windows = {item.window_id: item.raw_definition for item in extraction.dlt_windows}
    window_items = {item.window_id: item for item in extraction.dlt_windows}
    observations_by_cohort: dict[str, list[Any]] = defaultdict(list)
    counts_by_cohort: dict[str, list[Any]] = defaultdict(list)
    for observation in extraction.dlt_observations:
        if observation.cohort_id:
            observations_by_cohort[observation.cohort_id].append(observation)
    for count in extraction.population_counts:
        if count.cohort_id:
            counts_by_cohort[count.cohort_id].append(count)
    units: list[FriendlyReviewUnit] = []
    for row in _curve_payloads(paths, extraction):
        cohort_id = str(row.get("cohort_id") or "")
        cohort = cohorts.get(cohort_id)
        if cohort is None:
            continue
        # The resolver may correctly retain a row in a series/regimen that
        # differs from an incorrectly linked source cohort. Review what the
        # curve actually uses so a wrong projection cannot be hidden by the UI.
        series_id = str(row.get("series_id") or cohort.series_id)
        series_item = series.get(series_id)
        regimen_id = str(
            row.get("regimen_id")
            or cohort.regimen_id
            or (series_item.regimen_id if series_item else "")
        ) or None
        part_id = row.get("part_id") or (series_item.part_id if series_item else None)
        arm_id = row.get("arm_id") or (series_item.arm_id if series_item else None)
        association_changes: list[str] = []
        if row.get("series_id") and row.get("series_id") != cohort.series_id:
            association_changes.append(
                f"exported series {row.get('series_id')} differs from source cohort series {cohort.series_id}"
            )
        if row.get("regimen_id") and cohort.regimen_id and row.get("regimen_id") != cohort.regimen_id:
            association_changes.append(
                f"exported regimen {row.get('regimen_id')} differs from source cohort regimen {cohort.regimen_id}"
            )
        point = CurvePointAnswer(
            cohort_id=cohort_id,
            series_id=series_id,
            series_label=series_item.label if series_item else series_id,
            regimen_label=regimens.get(regimen_id or "", regimen_id),
            part_label=parts.get(str(part_id or ""), str(part_id) if part_id else None),
            arm_label=arms.get(str(arm_id or ""), str(arm_id) if arm_id else None),
            dose_value=row.get("dose_value"),
            dose_unit=str(row.get("dose_unit") or cohort.dose.unit_normalized or cohort.dose.unit_raw or "Unit not reported"),
            dose_raw=str(row.get("dose_raw") or cohort.dose.raw_expression),
            n=row.get("n"),
            y=row.get("y"),
            denominator_type=str(row.get("denominator_role") or "unknown"),
            treated_n=row.get("reported_treated_n"),
            dlt_evaluable_n=row.get("reported_dlt_evaluable_n"),
            window_label=windows.get(row.get("window_id") or "", row.get("window_id")),
            value_basis=" + ".join(row.get("rule_ids") or []) or f"{row.get('numerator_basis', 'reported')} / {row.get('denominator_basis', 'reported')}",
            curve_status=(
                "included"
                if row.get("included_in_default", True)
                else "excluded: " + "; ".join(row.get("exclusion_reasons") or ["unresolved"])
            ),
            association_note="; ".join(association_changes) or None,
            exposures=row.get("exposures") or cohort.exposures,
            exposure_coverage=row.get("exposure_coverage", cohort.exposure_coverage),
            exposure_issues=row.get("exposure_issues", []),
            varying_component_id=row.get("varying_component_id") or (series_item.varying_component_id if series_item else None),
        )
        direct_evidence: list[Iterable[EvidenceLink]] = [cohort.evidence]
        direct_evidence.extend(item.evidence for item in point.exposures)
        direct_evidence.extend(phase.evidence for item in point.exposures for phase in item.dosing_phases)
        direct_evidence.extend(item.evidence for item in observations_by_cohort.get(cohort_id, []))
        direct_evidence.extend(item.evidence for item in counts_by_cohort.get(cohort_id, []))
        if series_item is not None:
            direct_evidence.append(series_item.evidence)
        if row.get("window_id") in window_items:
            direct_evidence.append(window_items[row["window_id"]].evidence)
        evidence_ids, evidence_text = _evidence(*direct_evidence)
        if row.get("rule_ids") or not evidence_ids:
            evidence_ids = _unique([*evidence_ids, *row.get("evidence_source_ids", [])])
        payload = clinical_payload(point)
        payload["evidence_source_ids"] = evidence_ids
        payload["policy"] = FRIENDLY_REVIEW_POLICY
        row_identity = str(row.get("curve_row_id") or cohort_id)
        units.append(
            FriendlyReviewUnit(
                review_unit_id="fr_curve_" + hashlib.sha256(row_identity.encode("utf-8")).hexdigest()[:16],
                module_id=CURVE_MODULE_ID,
                module_label=MODULE_LABELS[CURVE_MODULE_ID],
                unit_kind="curve_point",
                predicted_point=point,
                source_record_ids=[cohort_id],
                evidence_source_ids=evidence_ids,
                evidence_text=evidence_text,
                content_fingerprint=_canonical_hash(payload),
            )
        )
    return units


def source_fingerprint(paths: ArticlePaths) -> str:
    values = {"reported_extraction": sha256_file(paths.reported_extraction)}
    if paths.curve_rows.is_file():
        values["curve_rows"] = sha256_file(paths.curve_rows)
    return _canonical_hash(values)


def build_friendly_units(paths: ArticlePaths, extraction: ReportedExtraction) -> list[FriendlyReviewUnit]:
    return [*_build_text_units(extraction), *_build_curve_units(paths, extraction)]


def new_friendly_review(paths: ArticlePaths, extraction: ReportedExtraction, *, reviewer: str = "") -> FriendlyArticleReview:
    now = _now()
    return FriendlyArticleReview(
        document_id=extraction.document_id,
        source_fingerprint=source_fingerprint(paths),
        reviewer=reviewer,
        module_complete={module_id: False for module_id, _label, _kind in MODULES},
        units=build_friendly_units(paths, extraction),
        created_at_utc=now,
        updated_at_utc=now,
    )


def sync_friendly_review(review: FriendlyArticleReview, paths: ArticlePaths, extraction: ReportedExtraction) -> FriendlyArticleReview:
    old_by_id = {unit.review_unit_id: unit for unit in review.units if unit.origin == "prediction"}
    old_modules = defaultdict(list)
    for unit in review.units:
        if unit.origin == "prediction":
            old_modules[unit.module_id].append(unit)
    synced: list[FriendlyReviewUnit] = []
    unchanged_by_module: dict[str, bool] = {module_id: True for module_id, _label, _kind in MODULES}
    new_units = build_friendly_units(paths, extraction)
    for unit in new_units:
        previous = old_by_id.get(unit.review_unit_id)
        same_curve = (previous is not None and previous.predicted_point is not None
                      and unit.predicted_point is not None
                      and clinical_payload(previous.predicted_point) == clinical_payload(unit.predicted_point)
                      and previous.evidence_source_ids == unit.evidence_source_ids)
        if previous is not None and (previous.content_fingerprint == unit.content_fingerprint or same_curve):
            unit.gold_answer = previous.gold_answer
            unit.gold_point = previous.gold_point
            unit.decision = previous.decision
            unit.note = previous.note
            unit.updated_at_utc = previous.updated_at_utc
        else:
            unchanged_by_module[unit.module_id] = False
        synced.append(unit)
    new_ids = {unit.review_unit_id for unit in new_units}
    for module_id, old_units in old_modules.items():
        if any(unit.review_unit_id not in new_ids for unit in old_units):
            unchanged_by_module[module_id] = False
    source_changed = review.source_fingerprint != source_fingerprint(paths)
    for unit in review.units:
        if unit.origin != "manual_missing":
            continue
        if unit.unit_kind == "curve_point" and unit.gold_point is not None and any(
            candidate.predicted_point is not None
            and _curve_semantic_key(candidate.predicted_point) == _curve_semantic_key(unit.gold_point)
            for candidate in new_units
        ):
            # A later extraction now contains the formerly missed fact.  The
            # old manual FN must not survive and double-count that same point.
            unchanged_by_module[unit.module_id] = False
            continue
        if source_changed:
            unit.decision = "pending"
            unit.updated_at_utc = _now()
            unchanged_by_module[unit.module_id] = False
        synced.append(unit)
    review.units = synced
    review.metric_policy = FRIENDLY_REVIEW_POLICY
    review.source_fingerprint = source_fingerprint(paths)
    review.document_id = extraction.document_id
    review.module_complete = {
        module_id: bool(review.module_complete.get(module_id)) and unchanged_by_module.get(module_id, False)
        for module_id, _label, _kind in MODULES
    }
    review.updated_at_utc = _now()
    return review


def load_or_create_friendly_review(paths: ArticlePaths) -> FriendlyArticleReview:
    extraction = ReportedExtraction.model_validate(read_json(paths.reported_extraction))
    if not paths.friendly_human_review.is_file():
        return new_friendly_review(paths, extraction)
    review = FriendlyArticleReview.model_validate(read_json(paths.friendly_human_review))
    if review.document_id != extraction.document_id:
        raise ValueError("friendly review document_id does not match reported_extraction.json")
    return sync_friendly_review(review, paths, extraction)


def mark_module_correct(review: FriendlyArticleReview, module_id: str) -> None:
    if module_id not in MODULE_LABELS:
        raise ValueError(f"Unknown module: {module_id}")
    for unit in review.units:
        if unit.module_id != module_id or unit.origin != "prediction":
            continue
        if module_id == CURVE_MODULE_ID and unit.decision in {"incorrect", "spurious", "excluded"}:
            # The table displays saved corrections; confirming it must preserve them.
            continue
        if unit.predicted_answer is None and unit.predicted_point is None:
            continue
        unit.decision = "correct"
        unit.gold_answer = unit.predicted_answer
        unit.gold_point = unit.predicted_point.model_copy(deep=True) if unit.predicted_point else None
        unit.updated_at_utc = _now()
    review.module_complete[module_id] = True
    review.updated_at_utc = _now()


def add_missing_curve_point(review: FriendlyArticleReview, point: CurvePointAnswer, *, note: str | None = None) -> FriendlyReviewUnit:
    if any(
        unit.predicted_point is not None
        and unit.decision != "spurious"
        and _curve_semantic_key(unit.predicted_point) == _curve_semantic_key(point)
        for unit in review.units
    ):
        raise ValueError("This dose-DLT point already has a model prediction; correct that point instead")
    raw = f"{review.document_id}\x1fmissing_curve\x1f{point.model_dump_json()}\x1f{_now()}"
    unit = FriendlyReviewUnit(
        review_unit_id="fr_missing_curve_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
        module_id=CURVE_MODULE_ID,
        module_label=MODULE_LABELS[CURVE_MODULE_ID],
        unit_kind="curve_point",
        gold_point=point,
        decision="missing",
        origin="manual_missing",
        content_fingerprint=_canonical_hash(point.model_dump(mode="json")),
        note=(note or "").strip() or None,
        updated_at_utc=_now(),
    )
    review.units.append(unit)
    review.module_complete[CURVE_MODULE_ID] = True
    review.updated_at_utc = _now()
    return unit


def _curve_semantic_key(point: CurvePointAnswer) -> tuple[Any, ...]:
    return (
        _norm_text(point.series_label),
        point.dose_value,
        _norm_text(point.dose_unit),
        point.n,
        point.y,
        _norm_text(point.window_label),
        _norm_text(point.part_label),
        _norm_text(point.arm_label),
        exposure_identity(point.exposures),
    )


def _norm_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def validate_unit(unit: FriendlyReviewUnit) -> list[str]:
    errors: list[str] = []
    if unit.origin == "manual_missing" and unit.decision not in {"pending", "missing", "excluded"}:
        errors.append("A manually added omission may only be pending, missing, or excluded")
    if unit.decision == "correct" and unit.predicted_answer is None and unit.predicted_point is None:
        errors.append("A field without a model answer cannot be marked correct")
    if unit.decision == "incorrect":
        if unit.unit_kind == "text_field" and not (unit.gold_answer or "").strip():
            errors.append("A corrected text field must include the correct answer")
        if unit.unit_kind == "curve_point" and unit.gold_point is None:
            errors.append("A corrected curve point must include the correct values")
    if unit.decision == "missing" and unit.gold_answer is None and unit.gold_point is None:
        errors.append("A missing item must include the correct answer")
    return errors


def _counts(units: Iterable[FriendlyReviewUnit]) -> MetricCounts:
    result = MetricCounts()
    for unit in units:
        if unit.decision == "correct":
            result.tp += 1
        elif unit.decision == "incorrect":
            result.fp += 1
            result.fn += 1
        elif unit.decision == "spurious":
            result.fp += 1
        elif unit.decision == "missing":
            result.fn += 1
        elif unit.decision == "pending":
            result.pending += 1
        elif unit.decision == "excluded":
            result.excluded += 1
    if result.tp + result.fp:
        result.precision = result.tp / (result.tp + result.fp)
    if result.tp + result.fn:
        result.recall = result.tp / (result.tp + result.fn)
    denominator = 2 * result.tp + result.fp + result.fn
    if denominator:
        result.f1 = 2 * result.tp / denominator
    return result


def _domain_summary(review: FriendlyArticleReview, units: list[FriendlyReviewUnit], module_ids: Iterable[str]) -> DomainSummary:
    counts = _counts(units)
    all_reviewed = counts.pending == 0
    completeness = all(review.module_complete.get(module_id, False) for module_id in module_ids)
    return DomainSummary(
        counts=counts,
        all_predictions_reviewed=all_reviewed,
        completeness_checked=completeness,
        precision_is_final=all_reviewed,
        recall_is_final=all_reviewed and completeness,
    )


def summarize_friendly_review(review: FriendlyArticleReview) -> FriendlyReviewSummary:
    by_module: dict[str, list[FriendlyReviewUnit]] = defaultdict(list)
    for unit in review.units:
        by_module[unit.module_id].append(unit)
    text_units = [unit for unit in review.units if unit.unit_kind == "text_field"]
    curve_units = [unit for unit in review.units if unit.unit_kind == "curve_point"]
    completed = sum(bool(review.module_complete.get(module_id)) for module_id, _label, _kind in MODULES)
    return FriendlyReviewSummary(
        document_id=review.document_id,
        source_fingerprint=review.source_fingerprint,
        reviewer=review.reviewer,
        completed_modules=completed,
        total_modules=len(MODULES),
        text_fields=_domain_summary(review, text_units, TEXT_MODULE_IDS),
        curve_points=_domain_summary(review, curve_units, [CURVE_MODULE_ID]),
        by_module={MODULE_LABELS.get(module_id, module_id): _counts(units) for module_id, units in by_module.items()},
        generated_at_utc=_now(),
    )


CSV_FIELDS = (
    "document_id", "review_unit_id", "module_id", "module_label", "unit_kind", "origin",
    "predicted_answer", "gold_answer", "decision", "tp", "fp", "fn", "source_record_ids",
    "evidence_source_ids", "evidence_text", "note",
    "predicted_exposures_json", "gold_exposures_json",
)


def friendly_review_csv(review: FriendlyArticleReview) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for unit in review.units:
        counts = _counts([unit])
        writer.writerow(
            {
                "document_id": review.document_id,
                "review_unit_id": unit.review_unit_id,
                "module_id": unit.module_id,
                "module_label": unit.module_label,
                "unit_kind": unit.unit_kind,
                "origin": unit.origin,
                "predicted_answer": unit.display_answer(),
                "gold_answer": unit.display_answer(gold=True),
                "decision": unit.decision,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "source_record_ids": " | ".join(unit.source_record_ids),
                "evidence_source_ids": " | ".join(unit.evidence_source_ids),
                "evidence_text": " | ".join(unit.evidence_text),
                "note": unit.note or "",
                "predicted_exposures_json": json.dumps([e.model_dump(mode="json") for e in unit.predicted_point.exposures], ensure_ascii=False) if unit.predicted_point else "",
                "gold_exposures_json": json.dumps([e.model_dump(mode="json") for e in unit.gold_point.exposures], ensure_ascii=False) if unit.gold_point else "",
            }
        )
    return stream.getvalue()


def _review_content(review: FriendlyArticleReview) -> dict[str, Any]:
    content = review.model_dump(mode="json")
    content.pop("created_at_utc")
    content.pop("updated_at_utc")
    for unit in content["units"]:
        unit.pop("updated_at_utc", None)
    return content


def load_friendly_review_undo(paths: ArticlePaths) -> FriendlyReviewUndo | None:
    """Only offer undo for the saved state and extraction that created it."""
    if not paths.friendly_review_undo.is_file() or not paths.friendly_human_review.is_file():
        return None
    try:
        undo = FriendlyReviewUndo.model_validate(read_json(paths.friendly_review_undo))
        current = FriendlyArticleReview.model_validate(read_json(paths.friendly_human_review))
        if (
            undo.previous_review.document_id != paths.document_id
            or current.document_id != paths.document_id
            or undo.previous_review.source_fingerprint != current.source_fingerprint
            or current.source_fingerprint != source_fingerprint(paths)
            or undo.saved_state_fingerprint != _canonical_hash(_review_content(current))
        ):
            return None
        return undo
    except (OSError, ValueError):
        return None


def undo_friendly_review(paths: ArticlePaths) -> tuple[FriendlyArticleReview, str | None]:
    undo = load_friendly_review_undo(paths)
    if undo is None:
        raise ValueError("No matching saved review is available to undo.")
    review = undo.previous_review
    save_friendly_review(paths, review, record_undo=False)
    paths.friendly_review_undo.unlink(missing_ok=True)
    return review, undo.module_id


def restart_friendly_review(paths: ArticlePaths, *, reviewer: str | None = None) -> FriendlyArticleReview:
    previous = load_or_create_friendly_review(paths)
    extraction = ReportedExtraction.model_validate(read_json(paths.reported_extraction))
    review = new_friendly_review(paths, extraction, reviewer=previous.reviewer if reviewer is None else reviewer)
    save_friendly_review(paths, review)
    return review


def save_friendly_review(
    paths: ArticlePaths, review: FriendlyArticleReview, *, record_undo: bool = True
) -> FriendlyReviewSummary:
    for unit in review.units:
        errors = validate_unit(unit)
        if errors:
            raise ValueError(f"{unit.review_unit_id}: {'; '.join(errors)}")
    review.updated_at_utc = _now()
    summary = summarize_friendly_review(review)
    if record_undo:
        previous = load_or_create_friendly_review(paths)
        before, after = _review_content(previous), _review_content(review)
        if before != after:
            changed_module = next(
                (
                    module_id for module_id, _label, _kind in MODULES
                    if before["module_complete"].get(module_id) != after["module_complete"].get(module_id)
                    or [unit for unit in before["units"] if unit["module_id"] == module_id]
                    != [unit for unit in after["units"] if unit["module_id"] == module_id]
                ),
                None,
            )
            write_json(paths.friendly_review_undo, FriendlyReviewUndo(
                previous_review=previous,
                saved_state_fingerprint=_canonical_hash(after),
                module_id=changed_module,
            ))
    write_json(paths.friendly_human_review, review)
    paths.friendly_review_units_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.friendly_review_units_csv.with_name(f".{paths.friendly_review_units_csv.name}.tmp")
    temporary.write_text(friendly_review_csv(review), encoding="utf-8")
    temporary.replace(paths.friendly_review_units_csv)
    write_json(paths.friendly_review_summary, summary)
    return summary


def _review_roots(output_roots: Path | Iterable[Path]) -> list[Path]:
    if isinstance(output_roots, (str, Path)):
        values = [Path(output_roots)]
    else:
        values = [Path(value) for value in output_roots]
    return list(dict.fromkeys(value.expanduser().resolve() for value in values))


def discover_review_articles(output_roots: Path | Iterable[Path]) -> list[ArticlePaths]:
    articles: list[ArticlePaths] = []
    for output_root in _review_roots(output_roots):
        if output_root.is_dir():
            articles.extend(
                ArticlePaths(output_root, path.parent.name)
                for path in sorted(output_root.glob("*/reported_extraction.json"))
            )
    return sorted(articles, key=lambda paths: (paths.document_id, str(paths.root)))


def load_current_saved_reviews(output_root: Path | Iterable[Path]) -> tuple[list[FriendlyArticleReview], int]:
    reviews: list[FriendlyArticleReview] = []
    stale = 0
    for paths in discover_review_articles(output_root):
        if not paths.friendly_human_review.is_file():
            continue
        try:
            review = FriendlyArticleReview.model_validate(read_json(paths.friendly_human_review))
            if review.source_fingerprint != source_fingerprint(paths):
                stale += 1
                continue
        except Exception:
            stale += 1
            continue
        reviews.append(review)
    return reviews, stale


def summarize_friendly_corpus(reviews: Iterable[FriendlyArticleReview]) -> FriendlyCorpusSummary:
    review_list = list(reviews)
    text_units = [unit for review in review_list for unit in review.units if unit.unit_kind == "text_field"]
    curve_units = [unit for review in review_list for unit in review.units if unit.unit_kind == "curve_point"]
    completed = sum(
        summarize_friendly_review(review).completed_modules == len(MODULES)
        and all(unit.decision != "pending" for unit in review.units)
        for review in review_list
    )
    return FriendlyCorpusSummary(
        article_count=len(review_list),
        text_fields=_counts(text_units),
        curve_points=_counts(curve_units),
        fully_completed_articles=completed,
        generated_at_utc=_now(),
    )
