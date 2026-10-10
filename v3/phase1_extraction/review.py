"""Single-reviewer, metric-level validation for v3 extractions.

The extraction artifact stays immutable.  This module flattens it into one
review unit per metric answer and writes the human answers to separate JSON,
CSV, and summary artifacts.

Scoring policy:

* correct: TP
* incorrect value: FP + FN
* spurious extraction: FP
* manually added omission: FN
* excluded or pending: outside the scored denominator

Recall is final only after the reviewer explicitly confirms that the source
has been checked for omissions and every predicted unit has a final decision.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import ArticlePaths, read_json, sha256_file, write_json
from .models import EvidenceLink, ReportedExtraction


REVIEW_FORMAT = "phase1_metric_review_v3"
REVIEW_POLICY = "phase1_metric_prf_v1"
Decision = Literal["pending", "correct", "incorrect", "spurious", "missing", "excluded"]
Origin = Literal["prediction", "manual_missing"]


METRIC_GROUPS = (
    "Study and Population",
    "Intervention and Dosing",
    "Design and DLT Definition",
    "Dose Cohorts and Counts",
    "DLT Results",
    "Result Scope and Count Relations",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewUnit(_StrictModel):
    review_unit_id: str
    metric_group: str
    metric_name: str
    metric_label: str
    scope: str
    source_record_id: str | None = None
    source_field: str
    predicted_answer: str | None = None
    gold_answer: str | None = None
    decision: Decision = "pending"
    origin: Origin = "prediction"
    evidence_source_ids: list[str] = Field(default_factory=list)
    evidence_text: list[str] = Field(default_factory=list)
    note: str | None = None
    updated_at_utc: str | None = None


class ArticleReview(_StrictModel):
    format: Literal["phase1_metric_review_v3"] = REVIEW_FORMAT
    metric_policy: Literal["phase1_metric_prf_v1"] = REVIEW_POLICY
    document_id: str
    extraction_sha256: str
    reviewer: str = ""
    source_review_complete: bool = False
    units: list[ReviewUnit] = Field(default_factory=list)
    created_at_utc: str
    updated_at_utc: str


class MetricCounts(_StrictModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    pending: int = 0
    excluded: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class ReviewSummary(_StrictModel):
    format: Literal["phase1_metric_review_summary_v3"] = "phase1_metric_review_summary_v3"
    metric_policy: Literal["phase1_metric_prf_v1"] = REVIEW_POLICY
    document_id: str
    extraction_sha256: str
    reviewer: str
    source_review_complete: bool
    all_predictions_reviewed: bool
    precision_is_final: bool
    recall_is_final: bool
    overall: MetricCounts
    by_metric_group: dict[str, MetricCounts]
    by_metric_name: dict[str, MetricCounts]
    generated_at_utc: str


class CorpusReviewSummary(_StrictModel):
    format: Literal["phase1_metric_review_corpus_summary_v3"] = "phase1_metric_review_corpus_summary_v3"
    metric_policy: Literal["phase1_metric_prf_v1"] = REVIEW_POLICY
    article_count: int
    all_predictions_reviewed_articles: int
    source_review_complete_articles: int
    precision_is_final_for_saved_reviews: bool
    recall_is_final_for_saved_reviews: bool
    overall: MetricCounts
    by_metric_group: dict[str, MetricCounts]
    by_metric_name: dict[str, MetricCounts]
    generated_at_utc: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _answer(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).strip()


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_")
        return bool(normalized) and normalized not in {"unclear", "unknown"}
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _evidence(evidence: Iterable[EvidenceLink]) -> tuple[list[str], list[str]]:
    source_ids: list[str] = []
    texts: list[str] = []
    for link in evidence:
        for source_id in link.source_ids:
            if source_id not in source_ids:
                source_ids.append(source_id)
        parts = [part.strip() for part in (link.source_text, link.locator) if part and part.strip()]
        if parts:
            text = " | ".join(parts)
            if text not in texts:
                texts.append(text)
    return source_ids, texts


def _stable_id(document_id: str, source_record_id: str, source_field: str, ordinal: int) -> str:
    raw = f"{document_id}\x1f{source_record_id}\x1f{source_field}\x1f{ordinal}"
    return "rv_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class _UnitBuilder:
    def __init__(self, extraction: ReportedExtraction) -> None:
        self.extraction = extraction
        self.units: list[ReviewUnit] = []

    def add(
        self,
        *,
        group: str,
        name: str,
        label: str,
        scope: str,
        record_id: str,
        field: str,
        value: Any,
        evidence: Iterable[EvidenceLink] = (),
        ordinal: int = 0,
    ) -> None:
        if not _is_present(value):
            return
        source_ids, texts = _evidence(evidence)
        self.units.append(
            ReviewUnit(
                review_unit_id=_stable_id(self.extraction.document_id, record_id, field, ordinal),
                metric_group=group,
                metric_name=name,
                metric_label=label,
                scope=scope,
                source_record_id=record_id,
                source_field=field,
                predicted_answer=_answer(value),
                evidence_source_ids=source_ids,
                evidence_text=texts,
            )
        )

    def add_list(self, *, values: Iterable[Any], **kwargs: Any) -> None:
        for index, value in enumerate(values):
            self.add(value=value, ordinal=index, **kwargs)


def build_review_units(extraction: ReportedExtraction) -> list[ReviewUnit]:
    """Flatten a reported extraction into one unit per non-empty answer."""

    b = _UnitBuilder(extraction)
    study_labels = {item.study_id: item.label for item in extraction.studies}
    part_labels = {item.part_id: item.label for item in extraction.study_parts}
    arm_labels = {item.arm_id: item.label for item in extraction.arms}
    intervention_names = {item.intervention_id: item.name_raw for item in extraction.interventions}
    series_labels = {item.series_id: item.label for item in extraction.dose_series}
    cohort_labels = {item.cohort_id: item.label for item in extraction.cohorts}
    population_names = {item.analysis_population_id: item.author_name for item in extraction.analysis_populations}

    def scope(*parts: str | None) -> str:
        return " / ".join(str(part) for part in parts if part)

    for item in extraction.studies:
        s = scope(item.label, item.study_id)
        b.add(group="Study and Population", name="study.label", label="Study name", scope=s, record_id=item.study_id, field="label", value=item.label, evidence=item.evidence)
        b.add_list(group="Study and Population", name="study.registration_id", label="Registration ID", scope=s, record_id=item.study_id, field="registration_ids", values=item.registration_ids, evidence=item.evidence)
        b.add(group="Study and Population", name="study.purpose", label="Study purpose", scope=s, record_id=item.study_id, field="purpose_summary", value=item.purpose_summary, evidence=item.evidence)

    for item in extraction.study_parts:
        s = scope(study_labels.get(item.study_id, item.study_id), item.label, item.part_id)
        b.add(group="Study and Population", name="study_part.phase", label="Study phase", scope=s, record_id=item.part_id, field="phase_raw", value=item.phase_raw, evidence=item.evidence)
        b.add(group="Study and Population", name="study_part.role", label="Study-part role", scope=s, record_id=item.part_id, field="role", value=item.role, evidence=item.evidence)

    for item in extraction.arms:
        s = scope(study_labels.get(item.study_id, item.study_id), part_labels.get(item.part_id or ""), item.label, item.arm_id)
        b.add(group="Study and Population", name="arm.label", label="Treatment group name", scope=s, record_id=item.arm_id, field="label", value=item.label, evidence=item.evidence)
        b.add(group="Study and Population", name="arm.population", label="Treatment group population", scope=s, record_id=item.arm_id, field="population_description", value=item.population_description, evidence=item.evidence)

    for item in extraction.populations:
        s = scope(study_labels.get(item.study_id, item.study_id), part_labels.get(item.part_id or ""), item.population_id)
        scalar_fields = (
            ("disease_raw", "population.disease", "Disease / tumor type"),
            ("disease_core", "population.disease_core", "Normalized disease"),
            ("sex_eligibility_raw", "population.sex", "Sex eligibility"),
        )
        for field, name, label in scalar_fields:
            b.add(group="Study and Population", name=name, label=label, scope=s, record_id=item.population_id, field=field, value=getattr(item, field), evidence=item.evidence)
        if item.age_min is not None or item.age_max is not None:
            low = f"{item.age_min_comparator or '>='} {item.age_min:g}" if item.age_min is not None else "No lower limit"
            high = f"{item.age_max_comparator or '<='} {item.age_max:g}" if item.age_max is not None else "No upper limit"
            b.add(group="Study and Population", name="population.age", label="Age eligibility", scope=s, record_id=item.population_id, field="age", value=f"{low} to {high} {item.age_unit or ''}".strip(), evidence=item.evidence)
        for field, name, label in (
            ("modifiers", "population.disease_modifier", "Disease qualifiers"),
            ("inclusion_criteria", "population.inclusion", "Inclusion criteria"),
            ("exclusion_criteria", "population.exclusion", "Exclusion criteria"),
            ("prior_treatment_requirements", "population.prior_treatment", "Prior-treatment requirements"),
            ("clinical_requirements", "population.clinical_requirement", "Clinical requirements"),
            ("baseline_summary", "population.baseline", "Baseline characteristics"),
        ):
            b.add_list(group="Study and Population", name=name, label=label, scope=s, record_id=item.population_id, field=field, values=getattr(item, field), evidence=item.evidence)

    for item in extraction.interventions:
        s = scope(item.name_raw, item.intervention_id)
        for field, name, label in (
            ("name_raw", "intervention.name", "Intervention name"),
            ("modality", "intervention.modality", "Intervention modality"),
            ("product_or_formulation", "intervention.formulation", "Product / formulation"),
            ("drug_class_raw", "intervention.drug_class", "Drug class"),
        ):
            b.add(group="Intervention and Dosing", name=name, label=label, scope=s, record_id=item.intervention_id, field=field, value=getattr(item, field), evidence=item.evidence)
        for field, name, label in (
            ("aliases", "intervention.alias", "Intervention alias"),
            ("mechanism_raw", "intervention.mechanism", "Mechanism of action"),
            ("targets_raw", "intervention.target", "Target"),
            ("pathways_raw", "intervention.pathway", "Pathway"),
        ):
            b.add_list(group="Intervention and Dosing", name=name, label=label, scope=s, record_id=item.intervention_id, field=field, values=getattr(item, field), evidence=item.evidence)

    for item in extraction.regimens:
        s = scope(item.label, item.regimen_id)
        b.add(group="Intervention and Dosing", name="regimen.composition", label="Regimen composition", scope=s, record_id=item.regimen_id, field="composition", value=item.composition, evidence=item.evidence)
        b.add(group="Intervention and Dosing", name="regimen.shared_schedule", label="Shared regimen schedule", scope=s, record_id=item.regimen_id, field="shared_schedule_raw", value=item.shared_schedule_raw, evidence=item.evidence)
        for component in item.components:
            cs = scope(s, intervention_names.get(component.intervention_id, component.intervention_id), component.component_id)
            b.add(group="Intervention and Dosing", name="regimen.component", label="Regimen intervention", scope=cs, record_id=component.component_id, field="intervention_id", value=intervention_names.get(component.intervention_id, component.intervention_id), evidence=item.evidence)
            b.add(group="Intervention and Dosing", name="regimen.component_role", label="Regimen-component role", scope=cs, record_id=component.component_id, field="role", value=component.role, evidence=item.evidence)
            b.add(group="Intervention and Dosing", name="regimen.dose_finding_component", label="Participates in dose finding", scope=cs, record_id=component.component_id, field="participates_in_dose_finding", value=component.participates_in_dose_finding, evidence=item.evidence)
            for phase in component.dosing_phases:
                ps = scope(cs, phase.phase_label, phase.dosing_phase_id)
                if phase.dose is not None:
                    b.add(group="Intervention and Dosing", name="dosing.dose", label="Dose administered", scope=ps, record_id=phase.dosing_phase_id, field="dose.raw_expression", value=phase.dose.raw_expression, evidence=phase.evidence)
                for field, name, label in (
                    ("route_raw", "dosing.route", "Route"),
                    ("dose_form_raw", "dosing.form", "Dosage form"),
                    ("frequency_raw", "dosing.frequency", "Dosing frequency"),
                    ("administration_days_raw", "dosing.days", "Administration days"),
                    ("cycle_length_raw", "dosing.cycle_length", "Cycle length"),
                    ("duration_raw", "dosing.duration", "Course / duration"),
                    ("infusion_duration_raw", "dosing.infusion_duration", "Infusion duration"),
                    ("start_condition_raw", "dosing.start_condition", "Start condition"),
                    ("end_condition_raw", "dosing.end_condition", "End condition"),
                ):
                    b.add(group="Intervention and Dosing", name=name, label=label, scope=ps, record_id=phase.dosing_phase_id, field=field, value=getattr(phase, field), evidence=phase.evidence)

    for item in extraction.designs:
        s = scope(study_labels.get(item.study_id, item.study_id), part_labels.get(item.part_id or ""), item.design_id)
        for field, name, label in (
            ("raw_name", "design.name", "Dose-escalation design"),
            ("normalized_type", "design.type", "Normalized design type"),
            ("initial_cohort_n", "design.initial_cohort_n", "Initial cohort size"),
            ("added_cohort_n", "design.added_cohort_n", "Additional cohort size"),
            ("expanded_total_n", "design.expanded_total_n", "Expanded cohort size"),
            ("stop_when_y_at_least", "design.stop_threshold", "Dose-escalation stopping threshold"),
        ):
            b.add(group="Design and DLT Definition", name=name, label=label, scope=s, record_id=item.design_id, field=field, value=getattr(item, field), evidence=item.evidence)
        b.add_list(group="Design and DLT Definition", name="design.expand_threshold", label="Cohort-expansion DLT condition", scope=s, record_id=item.design_id, field="expand_when_y_equals", values=item.expand_when_y_equals, evidence=item.evidence)
        b.add_list(group="Design and DLT Definition", name="design.decision_rule", label="Dose-escalation rule", scope=s, record_id=item.design_id, field="decision_rules_raw", values=item.decision_rules_raw, evidence=item.evidence)
        b.add_list(group="Design and DLT Definition", name="design.amendment", label="Design amendment", scope=s, record_id=item.design_id, field="amendments_raw", values=item.amendments_raw, evidence=item.evidence)

    for item in extraction.dlt_windows:
        s = scope(study_labels.get(item.study_id, item.study_id), part_labels.get(item.part_id or ""), item.window_id)
        for field, name, label in (
            ("raw_definition", "dlt_window.definition", "DLT assessment window"),
            ("start_raw", "dlt_window.start", "DLT-window start"),
            ("end_raw", "dlt_window.end", "DLT-window end"),
            ("cycle_numbers_raw", "dlt_window.cycles", "DLT assessment cycle"),
            ("normalized_days", "dlt_window.days", "DLT-window duration in days"),
        ):
            b.add(group="Design and DLT Definition", name=name, label=label, scope=s, record_id=item.window_id, field=field, value=getattr(item, field), evidence=item.evidence)

    for item in extraction.dlt_definitions:
        s = scope(study_labels.get(item.study_id, item.study_id), part_labels.get(item.part_id or ""), item.version_label, item.definition_id)
        for field, name, label in (
            ("logic_raw", "dlt_definition.logic", "DLT decision logic"),
            ("grading_system_raw", "dlt_definition.grading", "Toxicity grading system"),
        ):
            b.add(group="Design and DLT Definition", name=name, label=label, scope=s, record_id=item.definition_id, field=field, value=getattr(item, field), evidence=item.evidence)
        b.add_list(group="Design and DLT Definition", name="dlt_definition.criterion", label="DLT criterion", scope=s, record_id=item.definition_id, field="criteria_raw", values=item.criteria_raw, evidence=item.evidence)
        b.add_list(group="Design and DLT Definition", name="dlt_definition.exception", label="DLT exception", scope=s, record_id=item.definition_id, field="exceptions_raw", values=item.exceptions_raw, evidence=item.evidence)

    for item in extraction.dose_series:
        s = scope(study_labels.get(item.study_id, item.study_id), part_labels.get(item.part_id or ""), arm_labels.get(item.arm_id or ""), item.label, item.series_id)
        b.add(group="Dose Cohorts and Counts", name="dose_series.varying_component", label="Dose-escalated component", scope=s, record_id=item.series_id, field="varying_component_id", value=intervention_names.get(item.varying_component_id, item.varying_component_id), evidence=item.evidence)
        b.add_list(group="Dose Cohorts and Counts", name="dose_series.fixed_component", label="Fixed-dose component", scope=s, record_id=item.series_id, field="fixed_component_ids", values=[intervention_names.get(v, v) for v in item.fixed_component_ids], evidence=item.evidence)
        b.add(group="Dose Cohorts and Counts", name="dose_series.unit", label="Comparable dose unit", scope=s, record_id=item.series_id, field="comparable_unit", value=item.comparable_unit, evidence=item.evidence)

    for item in extraction.cohorts:
        s = scope(series_labels.get(item.series_id, item.series_id), item.label, item.cohort_id)
        for field, name, label, value in (
            ("actual_status", "cohort.status", "Dose-cohort status", item.actual_status),
            ("cohort_role", "cohort.role", "Dose-cohort role", item.cohort_role),
            ("dose.raw_expression", "cohort.dose", "Cohort dose", item.dose.raw_expression),
            ("expansion_status", "cohort.expansion", "Cohort-expansion status", item.expansion_status),
        ):
            b.add(group="Dose Cohorts and Counts", name=name, label=label, scope=s, record_id=item.cohort_id, field=field, value=value, evidence=item.evidence)

    for item in extraction.analysis_populations:
        s = scope(study_labels.get(item.study_id, item.study_id), part_labels.get(item.part_id or ""), item.author_name, item.analysis_population_id)
        for field, name, label in (
            ("author_name", "analysis_population.name", "Analysis-population name"),
            ("interpreted_role", "analysis_population.role", "Analysis-population role"),
            ("definition_raw", "analysis_population.definition", "Analysis-population definition"),
        ):
            b.add(group="Dose Cohorts and Counts", name=name, label=label, scope=s, record_id=item.analysis_population_id, field=field, value=getattr(item, field), evidence=item.evidence)

    for item in extraction.population_counts:
        s = scope(
            study_labels.get(item.study_id, item.study_id),
            part_labels.get(item.part_id or ""),
            arm_labels.get(item.arm_id or ""),
            cohort_labels.get(item.cohort_id or ""),
            population_names.get(item.analysis_population_id or ""),
            item.count_id,
        )
        if item.reported_value is not None:
            b.add(group="Dose Cohorts and Counts", name=f"population_count.{item.role}", label=f"Population count: {item.role}", scope=s, record_id=item.count_id, field="reported_value", value=item.reported_value, evidence=item.evidence)
        if item.reporting_status != "reported":
            b.add(group="Dose Cohorts and Counts", name="population_count.reporting_status", label="Population-count reporting status", scope=s, record_id=item.count_id, field="reporting_status", value=item.reporting_status, evidence=item.evidence)
        for field, name, label in (
            ("author_role_raw", "population_count.author_role", "Author population term"),
            ("timepoint_raw", "population_count.timepoint", "Population-count time point"),
            ("is_cumulative", "population_count.cumulative", "Is cumulative"),
            ("equivalent_to_role", "population_count.equivalent_role", "Equivalent analysis-population role"),
        ):
            b.add(group="Dose Cohorts and Counts", name=name, label=label, scope=s, record_id=item.count_id, field=field, value=getattr(item, field), evidence=item.evidence)

    for item in extraction.result_scopes:
        s = scope(series_labels.get(item.series_id, item.series_id), item.result_scope_id)
        for field, name, label in (
            ("data_cutoff_raw", "result_scope.data_cutoff", "Data cutoff"),
            ("dlt_results_coverage", "result_scope.coverage", "DLT-results coverage"),
            ("has_observed_dlt_results", "result_scope.has_results", "Has DLT results in scope"),
            ("explicit_no_dlt_in_scope", "result_scope.explicit_no_dlt", "Explicitly no DLTs in scope"),
            ("unresolved_positive_dlt", "result_scope.unresolved_positive", "Has unresolved positive DLT"),
        ):
            b.add(group="Result Scope and Count Relations", name=name, label=label, scope=s, record_id=item.result_scope_id, field=field, value=getattr(item, field), evidence=item.evidence)

    for item in extraction.dlt_observations:
        s = scope(cohort_labels.get(item.cohort_id or "", "Unassigned cohort"), item.result_scope_id, item.observation_id)
        if item.reported_count is not None:
            b.add(group="DLT Results", name=f"dlt.count.{item.count_kind}", label=f"DLT count: {item.count_kind}", scope=s, record_id=item.observation_id, field="reported_count", value=item.reported_count, evidence=item.evidence)
        for field, name, label in (
            ("explicit_patient_count", "dlt.explicit_patient_count", "Patients explicitly reported with DLT"),
            ("same_patient_repeated_events", "dlt.repeated_events", "Repeated events in the same patient"),
        ):
            b.add(group="DLT Results", name=name, label=label, scope=s, record_id=item.observation_id, field=field, value=getattr(item, field), evidence=item.evidence)
        if item.reporting_status != "reported":
            b.add(group="DLT Results", name="dlt.reporting_status", label="DLT reporting status", scope=s, record_id=item.observation_id, field="reporting_status", value=item.reporting_status, evidence=item.evidence)
        b.add_list(group="DLT Results", name="dlt.toxicity", label="DLT manifestation", scope=s, record_id=item.observation_id, field="toxicities_raw", values=item.toxicities_raw, evidence=item.evidence)

    for item in extraction.count_equations:
        s = scope(item.scope_description, item.equation_id)
        expression = f"{item.total_record_id} = " + " + ".join(item.component_record_ids)
        b.add(group="Result Scope and Count Relations", name="count_equation", label="Count equation", scope=s, record_id=item.equation_id, field="equation", value=expression, evidence=item.evidence)

    return b.units


def new_review(extraction: ReportedExtraction, extraction_sha256: str, *, reviewer: str = "") -> ArticleReview:
    now = _now()
    return ArticleReview(
        document_id=extraction.document_id,
        extraction_sha256=extraction_sha256,
        reviewer=reviewer,
        units=build_review_units(extraction),
        created_at_utc=now,
        updated_at_utc=now,
    )


def sync_review(review: ArticleReview, extraction: ReportedExtraction, extraction_sha256: str) -> ArticleReview:
    """Merge a new extraction into a review without trusting stale verdicts."""

    extraction_changed = review.extraction_sha256 != extraction_sha256
    old = {unit.review_unit_id: unit for unit in review.units if unit.origin == "prediction"}
    synced: list[ReviewUnit] = []
    for unit in build_review_units(extraction):
        previous = old.get(unit.review_unit_id)
        if previous is not None and previous.predicted_answer == unit.predicted_answer:
            unit.gold_answer = previous.gold_answer
            unit.decision = previous.decision
            unit.note = previous.note
            unit.updated_at_utc = previous.updated_at_utc
        synced.append(unit)
    synced.extend(unit for unit in review.units if unit.origin == "manual_missing")
    review.document_id = extraction.document_id
    review.extraction_sha256 = extraction_sha256
    if extraction_changed:
        review.source_review_complete = False
    review.units = synced
    review.updated_at_utc = _now()
    return review


def load_or_create_review(paths: ArticlePaths) -> ArticleReview:
    extraction = ReportedExtraction.model_validate(read_json(paths.reported_extraction))
    extraction_hash = sha256_file(paths.reported_extraction)
    if paths.human_review.is_file():
        review = ArticleReview.model_validate(read_json(paths.human_review))
        if review.document_id != extraction.document_id:
            raise ValueError("Review document_id does not match reported_extraction.json")
        review = sync_review(review, extraction, extraction_hash)
    else:
        review = new_review(extraction, extraction_hash)
    save_review(paths, review)
    return review


def add_missing_unit(
    review: ArticleReview,
    *,
    metric_group: str,
    metric_name: str,
    metric_label: str,
    scope: str,
    gold_answer: str,
    note: str | None = None,
) -> ReviewUnit:
    gold_answer = gold_answer.strip()
    if not metric_name.strip() or not metric_label.strip() or not gold_answer:
        raise ValueError("A missing item requires a metric name, display label, and correct answer")
    ordinal = sum(unit.origin == "manual_missing" for unit in review.units)
    raw = f"{review.document_id}\x1fmanual_missing\x1f{metric_name}\x1f{ordinal}\x1f{_now()}"
    unit = ReviewUnit(
        review_unit_id="rv_missing_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
        metric_group=metric_group,
        metric_name=metric_name.strip(),
        metric_label=metric_label.strip(),
        scope=scope.strip() or "Whole article",
        source_field="manual_missing",
        predicted_answer=None,
        gold_answer=gold_answer,
        decision="missing",
        origin="manual_missing",
        note=(note or "").strip() or None,
        updated_at_utc=_now(),
    )
    review.units.append(unit)
    review.updated_at_utc = _now()
    return unit


def validate_unit(unit: ReviewUnit) -> list[str]:
    errors: list[str] = []
    if unit.origin == "manual_missing" and unit.decision not in {"missing", "excluded"}:
        errors.append("A manually added omission may only be marked missing or excluded")
    if unit.origin == "prediction" and unit.decision == "missing":
        errors.append("A model-generated answer cannot be marked missing")
    if unit.decision in {"incorrect", "missing"} and not (unit.gold_answer or "").strip():
        errors.append("Incorrect and missing items require a correct human answer")
    return errors


def _counts(units: Iterable[ReviewUnit]) -> MetricCounts:
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
    if result.precision is not None and result.recall is not None and result.precision + result.recall:
        result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
    return result


def summarize_review(review: ArticleReview) -> ReviewSummary:
    group_units: dict[str, list[ReviewUnit]] = defaultdict(list)
    metric_units: dict[str, list[ReviewUnit]] = defaultdict(list)
    for unit in review.units:
        group_units[unit.metric_group].append(unit)
        metric_units[unit.metric_name].append(unit)
    overall = _counts(review.units)
    all_reviewed = overall.pending == 0
    return ReviewSummary(
        document_id=review.document_id,
        extraction_sha256=review.extraction_sha256,
        reviewer=review.reviewer,
        source_review_complete=review.source_review_complete,
        all_predictions_reviewed=all_reviewed,
        precision_is_final=all_reviewed,
        recall_is_final=all_reviewed and review.source_review_complete,
        overall=overall,
        by_metric_group={name: _counts(units) for name, units in sorted(group_units.items())},
        by_metric_name={name: _counts(units) for name, units in sorted(metric_units.items())},
        generated_at_utc=_now(),
    )


def summarize_corpus(reviews: Iterable[ArticleReview]) -> CorpusReviewSummary:
    review_list = list(reviews)
    units = [unit for review in review_list for unit in review.units]
    group_units: dict[str, list[ReviewUnit]] = defaultdict(list)
    metric_units: dict[str, list[ReviewUnit]] = defaultdict(list)
    for unit in units:
        group_units[unit.metric_group].append(unit)
        metric_units[unit.metric_name].append(unit)
    completed = sum(_counts(review.units).pending == 0 for review in review_list)
    source_complete = sum(review.source_review_complete for review in review_list)
    all_predictions_reviewed = completed == len(review_list)
    all_sources_reviewed = source_complete == len(review_list)
    return CorpusReviewSummary(
        article_count=len(review_list),
        all_predictions_reviewed_articles=completed,
        source_review_complete_articles=source_complete,
        precision_is_final_for_saved_reviews=bool(review_list) and all_predictions_reviewed,
        recall_is_final_for_saved_reviews=bool(review_list) and all_predictions_reviewed and all_sources_reviewed,
        overall=_counts(units),
        by_metric_group={name: _counts(items) for name, items in sorted(group_units.items())},
        by_metric_name={name: _counts(items) for name, items in sorted(metric_units.items())},
        generated_at_utc=_now(),
    )


CSV_FIELDS = (
    "document_id", "review_unit_id", "metric_group", "metric_name", "metric_label",
    "scope", "origin", "predicted_answer", "gold_answer", "decision", "tp", "fp", "fn",
    "source_record_id", "source_field", "evidence_source_ids", "evidence_text", "note",
)


def review_csv(review: ArticleReview) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for unit in review.units:
        counts = _counts([unit])
        writer.writerow(
            {
                "document_id": review.document_id,
                "review_unit_id": unit.review_unit_id,
                "metric_group": unit.metric_group,
                "metric_name": unit.metric_name,
                "metric_label": unit.metric_label,
                "scope": unit.scope,
                "origin": unit.origin,
                "predicted_answer": unit.predicted_answer or "",
                "gold_answer": unit.gold_answer or "",
                "decision": unit.decision,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "source_record_id": unit.source_record_id or "",
                "source_field": unit.source_field,
                "evidence_source_ids": " | ".join(unit.evidence_source_ids),
                "evidence_text": " | ".join(unit.evidence_text),
                "note": unit.note or "",
            }
        )
    return stream.getvalue()


def corpus_review_csv(reviews: Iterable[ArticleReview]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for review in reviews:
        article_rows = csv.DictReader(io.StringIO(review_csv(review)))
        writer.writerows(article_rows)
    return stream.getvalue()


def save_review(paths: ArticlePaths, review: ArticleReview) -> ReviewSummary:
    for unit in review.units:
        errors = validate_unit(unit)
        if errors:
            raise ValueError(f"{unit.review_unit_id}: {'; '.join(errors)}")
    review.updated_at_utc = _now()
    summary = summarize_review(review)
    write_json(paths.human_review, review)
    paths.review_units_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.review_units_csv.with_name(f".{paths.review_units_csv.name}.tmp")
    temporary.write_text(review_csv(review), encoding="utf-8")
    temporary.replace(paths.review_units_csv)
    write_json(paths.review_summary, summary)
    return summary


def discover_review_articles(output_root: Path) -> list[ArticlePaths]:
    output_root = Path(output_root).expanduser().resolve()
    if not output_root.is_dir():
        return []
    return [
        ArticlePaths(output_root, path.parent.name)
        for path in sorted(output_root.glob("*/reported_extraction.json"))
    ]


def load_current_saved_reviews(output_root: Path) -> tuple[list[ArticleReview], int]:
    """Load saved reviews whose extraction hash still matches; count stale ones."""

    reviews: list[ArticleReview] = []
    stale = 0
    for paths in discover_review_articles(output_root):
        if not paths.human_review.is_file():
            continue
        try:
            review = ArticleReview.model_validate(read_json(paths.human_review))
        except Exception:
            stale += 1
            continue
        if review.extraction_sha256 != sha256_file(paths.reported_extraction):
            stale += 1
            continue
        reviews.append(review)
    return reviews, stale
