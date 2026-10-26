"""Typed source, extraction, policy, and curve contracts for v3."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .controlled_vocabulary import (
    SexEligibility, StudyPhase, ClinicalMetric, Comparator, ConditionCategory, AgeLowerComparator, AgeUpperComparator,
    ConditionUnit, AgeUnit, Route, DosageForm, TreatmentStage, FrequencyPattern,
    DoseUnit, DoseDenominator, GradingSystem, GradingVersion, WindowAnchor,
    DiseaseModifier, DoseBasis, IntervalUnit, CycleUnit, InfusionUnit, AdjustmentAction,
)


SCHEMA_VERSION = "phase1_extraction_v3.3"
IR_VERSION = "phase1_document_ir_v3.1"
POLICY_VERSION = "phase1_policy_v3.2_no_r3"


def has_retired_r3_provenance(record: object) -> bool:
    """Recognize historical design assumptions even if one legacy tag is missing."""
    def value(key: str, default=None):
        return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)

    return bool(
        "R3" in (value("rule_ids", []) or [])
        or value("denominator_basis") == "design_inferred"
        or value("denominator_role") == "design_assumed_cohort"
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KeyValue(StrictModel):
    name: str
    value: str


class Rect(StrictModel):
    x0: float
    y0: float
    x1: float
    y1: float


class Material(StrictModel):
    material_id: str
    path: str
    media_type: Literal["application/pdf", "application/xml", "text/xml", "image", "other"]
    role: Literal["main", "supplement", "table_attachment", "unknown"] = "unknown"
    sha256: str
    byte_count: int
    status: Literal["registered", "parsed", "partial", "failed"] = "registered"
    page_count: int | None = None
    title: str | None = None
    identifiers: list[KeyValue] = Field(default_factory=list)
    error: str | None = None


class PageIR(StrictModel):
    page_id: str
    material_id: str
    page_number: int
    printed_page_number: str | None = None
    width: float
    height: float
    rotation: int = 0
    native_text_status: Literal["readable", "sparse", "none", "failed"]
    ocr_status: Literal["not_needed", "not_run", "completed", "failed"] = "not_run"
    image_path: str | None = None
    text_char_count: int = 0
    visual_inventory_status: Literal["available", "missing", "not_rendered"] = "not_rendered"


class BlockIR(StrictModel):
    block_id: str
    material_id: str
    page_id: str | None = None
    block_type: Literal[
        "title", "abstract", "section_title", "paragraph", "list_item",
        "caption", "footnote", "table_text", "page_text", "other"
    ]
    text_raw: str
    text_normalized: str
    section_path: list[str] = Field(default_factory=list)
    reading_order: int
    bbox: Rect | None = None
    bbox_normalized: Rect | None = None
    parser: Literal["jats", "pymupdf", "ocr", "derived"]
    previous_block_id: str | None = None
    next_block_id: str | None = None


class TableCellIR(StrictModel):
    cell_id: str
    row_index: int
    column_index: int
    row_span: int = 1
    column_span: int = 1
    is_header: bool = False
    text_raw: str = ""
    native_text: str | None = None
    ocr_text: str | None = None
    visual_text: str | None = None
    row_header_path: list[str] = Field(default_factory=list)
    column_header_path: list[str] = Field(default_factory=list)
    footnote_markers: list[str] = Field(default_factory=list)
    bbox: Rect | None = None
    bbox_normalized: Rect | None = None
    parse_status: Literal["complete", "partial", "raw_only"] = "complete"


class TableFragmentIR(StrictModel):
    fragment_id: str
    material_id: str
    page_id: str | None = None
    bbox: Rect | None = None
    bbox_normalized: Rect | None = None
    image_path: str | None = None
    source: Literal["jats", "pymupdf_lines", "pymupdf_text", "visual_candidate"]
    continued_from_fragment_id: str | None = None


class TableBundle(StrictModel):
    table_id: str
    material_id: str
    source_label: str | None = None
    caption: str | None = None
    section_path: list[str] = Field(default_factory=list)
    fragments: list[TableFragmentIR] = Field(default_factory=list)
    row_count: int
    column_count: int
    cells: list[TableCellIR] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    referenced_by_block_ids: list[str] = Field(default_factory=list)
    parse_status: Literal["complete", "partial", "visual_only", "failed"]
    relevant_hint: bool = False
    detection_kind: Literal["data_table", "layout_candidate", "visual_candidate", "unknown"] = "unknown"
    visual_required: bool = True
    unresolved_notes: list[str] = Field(default_factory=list)


class CoverageRecord(StrictModel):
    source_id: str
    source_kind: Literal["material", "material_reference", "page", "block", "table", "table_fragment"]
    parse_status: Literal["complete", "partial", "unread", "failed"]
    included_in_extraction: bool = False
    potentially_relevant: bool = False
    reason: str | None = None


class MaterialReference(StrictModel):
    reference_id: str
    raw_text: str
    locator_source_ids: list[str] = Field(default_factory=list)
    reference_kind: Literal["supplement", "appendix", "protocol", "other"]
    availability: Literal["supplied", "missing", "unknown"] = "unknown"
    potentially_relevant_categories: list[str] = Field(default_factory=list)


class DocumentIR(StrictModel):
    # v3 remains readable so frozen evidence from an in-flight or completed
    # Batch run can be reviewed without mutating its source fingerprint.
    ir_version: Literal["phase1_document_ir_v3", "phase1_document_ir_v3.1"] = IR_VERSION
    document_id: str
    title: str | None = None
    identifiers: list[KeyValue] = Field(default_factory=list)
    materials: list[Material]
    pages: list[PageIR] = Field(default_factory=list)
    blocks: list[BlockIR] = Field(default_factory=list)
    tables: list[TableBundle] = Field(default_factory=list)
    coverage: list[CoverageRecord] = Field(default_factory=list)
    material_references: list[MaterialReference] = Field(default_factory=list)
    relations: list[KeyValue] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class EvidenceLink(StrictModel):
    source_ids: list[str] = Field(default_factory=list)
    source_text: str | None = None
    locator: str | None = None
    support_type: Literal["text", "table", "visual", "combined", "unknown"] = "unknown"


class ValueProvenance(StrictModel):
    """How a value entered the reported or resolved data model."""

    # r3_design_inferred is accepted only to read legacy artifacts; it is never
    # eligible as a current denominator or an arithmetic input.
    basis: Literal[
        "source_reported", "arithmetic_derived", "r1_inferred_patient_count",
        "r2_imputed_zero", "r3_design_inferred", "unknown",
    ] = "source_reported"
    input_record_ids: list[str] = Field(default_factory=list)
    expression: str | None = None
    note: str | None = None


class RuleGroup(StrictModel):
    group_id: str
    operator: Literal["AND", "OR"]
    children: list[str] = Field(default_factory=list)


class StructuredSchedule(StrictModel):
    frequency_pattern: FrequencyPattern | None = Field(default=None, description="Reported scheduling pattern; null if unstated. Intermittent schedules also use numerical frequency and interval fields.")
    frequency_count: float | None = Field(default=None, gt=0)
    interval_value: float | None = Field(default=None, gt=0)
    interval_unit: IntervalUnit | None = None
    cycle_value: float | None = Field(default=None, gt=0)
    cycle_unit: CycleUnit | None = None
    administration_days: list[int] = Field(default_factory=list)
    infusion_value: float | None = Field(default=None, gt=0)
    infusion_unit: InfusionUnit | None = None
    dose_basis: DoseBasis = "unclear"


class ConditionClause(StrictModel):
    """A reviewable eligibility or toxicity condition without losing the source wording."""

    clause_id: str
    metric_code: ClinicalMetric | None = Field(default=None, description="Choose the clinical measure/event, never an entire rule sentence; other preserves unlisted topics in metric/raw_text.")
    category_code: ConditionCategory | None = None
    comparator_code: Comparator | None = Field(default=None, description="Only an explicitly stated comparison or grade relation; never infer a threshold from an arbitrary number.")
    unit_code: ConditionUnit | None = None
    metric: str | None = None
    raw_text: str
    category: str | None = None
    applies_to_part_ids: list[str] = Field(default_factory=list)
    applies_to_arm_ids: list[str] = Field(default_factory=list)
    comparator: str | None = None
    threshold: float | None = None
    threshold_upper: float | None = None
    unit_raw: str | None = None
    timing_raw: str | None = None
    connective_to_previous: Literal["AND", "OR", "NONE", "UNCLEAR"] = "NONE"
    negated: bool | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class DoseAdjustmentRule(StrictModel):
    rule_id: str
    action: AdjustmentAction
    condition_raw: str
    condition_structured: ConditionClause | None = None
    action_raw: str | None = None
    applies_to_part_ids: list[str] = Field(default_factory=list)
    applies_to_arm_ids: list[str] = Field(default_factory=list)
    applies_to_component_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)


class Study(StrictModel):
    study_id: str
    label: str
    registration_ids: list[str] = Field(default_factory=list)
    is_current_study: bool = True
    purpose_summary: str | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class StudyPart(StrictModel):
    part_id: str
    study_id: str
    label: str
    phase_raw: str | None = None
    phase: StudyPhase | None = None
    role: Literal["escalation", "deescalation", "expansion", "confirmation", "lead_in", "maintenance", "mixed", "other", "unclear"] = "unclear"
    sequence_index: int | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class Arm(StrictModel):
    arm_id: str
    study_id: str
    part_id: str | None = None
    label: str
    population_description: str | None = None
    regimen_id: str | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class PopulationDescription(StrictModel):
    population_id: str
    study_id: str
    part_id: str | None = None
    disease_raw: str | None = None
    disease_core: str | None = None
    modifiers: list[str] = Field(default_factory=list)
    age_min: float | None = None
    age_min_comparator: str | None = None
    age_max: float | None = None
    age_max_comparator: str | None = None
    age_unit: str | None = None
    sex_eligibility_raw: str | None = None
    sex_eligibility: SexEligibility | None = Field(default=None, description="Female, male, or explicitly no restriction. Unreported eligibility is null; do not infer it from enrolled-patient sex counts.")
    age_min_operator: AgeLowerComparator | None = None
    age_max_operator: AgeUpperComparator | None = None
    age_unit_code: AgeUnit | None = None
    disease_modifier_codes: list[DiseaseModifier] = Field(default_factory=list, description="Supported disease qualifiers only. Keep exact original qualifiers in modifiers. Empty means no qualifier coded, not absence of disease characteristics.")
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    prior_treatment_requirements: list[str] = Field(default_factory=list)
    clinical_requirements: list[str] = Field(default_factory=list)
    baseline_summary: list[str] = Field(default_factory=list)
    eligibility_conditions: list[ConditionClause] = Field(default_factory=list)
    eligibility_logic_groups: list[RuleGroup] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)


    @model_validator(mode="after")
    def valid_rule_groups(self):
        from .clinical_protocol import validate_logic
        validate_logic([g.model_dump() for g in self.eligibility_logic_groups], {c.clause_id for c in self.eligibility_conditions})
        return self


class Intervention(StrictModel):
    intervention_id: str
    name_raw: str
    aliases: list[str] = Field(default_factory=list)
    modality: Literal["drug", "radiation", "cell", "gene", "virus", "vaccine", "procedure", "other", "unclear"] = "unclear"
    product_or_formulation: str | None = None
    drug_class_raw: str | None = None
    mechanism_raw: list[str] = Field(default_factory=list)
    targets_raw: list[str] = Field(default_factory=list)
    pathways_raw: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)


class DoseExpression(StrictModel):
    comparator_code: Comparator | None = None
    unit_code: DoseUnit | None = Field(default=None, description="Closed unit code for the numeric value as written. Alias spelling only: never turn micrograms into mg without changing numeric value; do not change numeric source values.")
    basis_code: DoseDenominator | None = None
    raw_expression: str
    value_kind: Literal["scalar", "range", "discrete", "formula", "conditional", "unparsed"]
    comparator: str | None = None
    value: float | None = None
    range_min: float | None = None
    range_max: float | None = None
    discrete_values: list[float] = Field(default_factory=list)
    coefficient: float | None = None
    exponent: int | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    denominator_basis: str | None = None
    dose_cap_raw: str | None = None
    formula_raw: str | None = None


class DosingPhase(StrictModel):
    dosing_phase_id: str
    structured: StructuredSchedule | None = None
    phase_label: str
    phase_kind: TreatmentStage | None = None
    route: Route | None = None
    dose_form: DosageForm | None = Field(default=None, description="Physical dosage form category only; preserve detailed product/formulation separately in dose_form_raw.")
    dose: DoseExpression | None = None
    route_raw: str | None = None
    dose_form_raw: str | None = None
    frequency_raw: str | None = None
    administration_days_raw: str | None = None
    cycle_length_raw: str | None = None
    duration_raw: str | None = None
    infusion_duration_raw: str | None = None
    infusion_rate_raw: str | None = None
    start_condition_raw: str | None = None
    end_condition_raw: str | None = None
    schedule_completeness: Literal["complete", "partial", "unclear"] = "unclear"
    evidence: list[EvidenceLink] = Field(default_factory=list)


class ResearchOrdering(StrictModel):
    """Model metadata, deliberately excluded from clinical/review identity."""

    research_priority: int | None = Field(default=None, ge=1)
    research_role: Literal["primary", "co_primary", "partner", "background", "unclear"] = Field(
        default="unclear", description="Research objective role: primary is the main investigated drug; co_primary marks every jointly investigated main drug. Distinguish fixed partners and background/supportive agents. Use source objectives or dose-finding evidence, not list order or dose magnitude.")
    priority_basis: Literal["source_reported", "human_confirmed", "deterministic_fallback", "unknown"] = "unknown"
    priority_reason: str | None = Field(default=None, description="Source-supported reason for the research role and ordering; explain uncertainty when unclear.")
    priority_evidence: list[EvidenceLink] = Field(default_factory=list, description="Source passages supporting which drug(s) are primarily investigated.")


class RegimenComponent(ResearchOrdering):
    component_id: str
    intervention_id: str
    role: Literal["escalated", "fixed_backbone", "combination_partner", "conditioning", "supportive", "radiation", "maintenance", "other", "unclear"] = "unclear"
    participates_in_dose_finding: bool | None = None
    dosing_phases: list[DosingPhase] = Field(default_factory=list)


class Regimen(StrictModel):
    regimen_id: str
    label: str
    composition: Literal["single_component", "combination", "mixed", "unclear"] = "unclear"
    components: list[RegimenComponent] = Field(default_factory=list)
    shared_schedule_raw: str | None = None
    dose_adjustment_rules: list[DoseAdjustmentRule] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)


class TrialDesign(StrictModel):
    design_id: str
    study_id: str
    part_id: str | None = None
    raw_name: str
    normalized_type: Literal[
        "classic_3plus3", "modified_3plus3", "accelerated_titration", "rolling_six",
        "crm", "boin", "mtpi", "keyboard", "model_based", "other", "unclear"
    ] = "unclear"
    initial_cohort_n: int | None = None
    added_cohort_n: int | None = None
    expanded_total_n: int | None = None
    replacement_count: int | None = None
    nonevaluable_count: int | None = None
    cohort_addition_count: int | None = None
    expand_when_y_equals: list[int] = Field(default_factory=list)
    stop_when_y_at_least: int | None = None
    decision_rules_raw: list[str] = Field(default_factory=list)
    amendments_raw: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)


class DLTWindow(StrictModel):
    window_kind: Literal["formal_dlt", "safety_monitoring", "unclear"] = "unclear"
    window_id: str
    study_id: str
    part_id: str | None = None
    raw_definition: str
    start_raw: str | None = None
    end_raw: str | None = None
    start_anchor: WindowAnchor | None = None
    end_anchor: WindowAnchor | None = None
    cycle_numbers_raw: str | None = None
    normalized_days: float | None = None
    basis: Literal["reported", "derived", "unclear"] = "reported"
    evidence: list[EvidenceLink] = Field(default_factory=list)


class DLTDefinition(StrictModel):
    definition_id: str
    study_id: str
    part_id: str | None = None
    version_label: str | None = None
    criteria_raw: list[str] = Field(default_factory=list)
    exceptions_raw: list[str] = Field(default_factory=list)
    logic_raw: str | None = None
    grading_system_raw: str | None = None
    grading_system: GradingSystem | None = None
    grading_version: GradingVersion | None = Field(default=None, description="Only an explicitly reported grading-system version; do not guess the version from publication year.")
    criteria_structured: list[ConditionClause] = Field(default_factory=list)
    criteria_logic_groups: list[RuleGroup] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)


    @model_validator(mode="after")
    def valid_rule_groups(self):
        from .clinical_protocol import validate_logic
        validate_logic([g.model_dump() for g in self.criteria_logic_groups], {c.clause_id for c in self.criteria_structured})
        return self


class DoseSeries(StrictModel):
    series_id: str
    study_id: str
    part_id: str | None = None
    arm_id: str | None = None
    label: str
    regimen_id: str
    varying_component_id: str
    fixed_component_ids: list[str] = Field(default_factory=list)
    window_id: str | None = None
    design_id: str | None = None
    comparable_unit: str | None = None
    comparable_unit_code: DoseUnit | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class DoseSubgroup(StrictModel):
    """Reported allocation at one dose; do not invent subgroup DLT counts."""
    dose: DoseExpression
    n: int = Field(gt=0)
    population_basis: str
    evidence: list[EvidenceLink] = Field(default_factory=list)


class CohortExposure(ResearchOrdering):
    """One source-supported drug/class exposure in a cohort, not a DLT outcome."""

    exposure_id: str
    name_raw: str
    entity_kind: Literal["drug", "drug_class", "non_drug", "unclear"] = "drug"
    intervention_id: str | None = None
    component_id: str | None = None
    role: Literal["escalated", "fixed_backbone", "combination_partner", "conditioning", "supportive", "concomitant", "radiation", "maintenance", "other", "unclear"] = "unclear"
    presence: Literal["present", "absent", "unknown", "mixed"] = "unknown"
    reporting_scope: Literal["cohort", "pooled", "unclear"] = "cohort"
    exposed_n: int | None = None
    dose: DoseExpression | None = None
    dose_status: Literal["reported", "not_reported", "unparsed", "conflicting", "not_applicable", "unknown"] = "unknown"
    dose_conflict_details: list[str] = Field(default_factory=list)
    dose_semantics: Literal["scalar", "alternative_unknown", "sequential", "subgroups", "patient_conditional", "unknown"] = "unknown"
    dose_subgroups: list[DoseSubgroup] = Field(default_factory=list)
    formulation_raw: str | None = None
    schedule_raw: str | None = None
    dosing_phases: list[DosingPhase] = Field(default_factory=list)
    window_id: str | None = None
    note: str | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class Cohort(StrictModel):
    cohort_id: str
    series_id: str
    label: str
    actual_status: Literal["actual", "planned", "not_used", "unclear"]
    cohort_role: Literal["escalation", "deescalation", "expansion", "confirmation", "lead_in", "maintenance", "other", "unclear"] = "unclear"
    regimen_id: str
    dose: DoseExpression
    exposures: list[CohortExposure] = Field(default_factory=list)
    exposure_coverage: Literal["complete", "partial", "unknown"] = "unknown"
    expansion_status: Literal["completed_expansion", "completed_to_six", "added_three", "initial_three_only", "stopped_before_expansion", "not_applicable", "unknown"] = "unknown"
    result_scope_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)


class AnalysisPopulation(StrictModel):
    analysis_population_id: str
    study_id: str
    part_id: str | None = None
    author_name: str
    interpreted_role: Literal["dlt_evaluable", "treated", "safety_evaluable", "other", "unclear"] = "unclear"
    definition_raw: str | None = None
    window_id: str | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class PopulationCount(StrictModel):
    count_id: str
    study_id: str
    part_id: str | None = None
    arm_id: str | None = None
    cohort_id: str | None = None
    result_scope_id: str | None = None
    analysis_population_id: str | None = None
    role: Literal[
        "screened", "consented", "planned", "enrolled", "assigned", "randomized",
        "treated", "dlt_evaluable", "safety_evaluable", "pk_evaluable",
        "efficacy_evaluable", "paired_dlt_denominator", "other", "unclear"
    ]
    author_role_raw: str | None = None
    reported_value: int | None = None
    reporting_status: Literal["reported", "not_reported", "explicitly_unreported", "unclear", "explicitly_not_evaluable", "not_applicable", "conflicting"] = "reported"
    timepoint_raw: str | None = None
    window_id: str | None = None
    data_cutoff_raw: str | None = None
    is_cumulative: bool | None = None
    equivalent_to_role: Literal["treated", "dlt_evaluable"] | None = None
    derived_from_count_ids: list[str] = Field(default_factory=list)
    provenance: ValueProvenance = Field(default_factory=ValueProvenance)
    evidence: list[EvidenceLink] = Field(default_factory=list)


class ResultScope(StrictModel):
    result_scope_id: str
    study_id: str
    part_id: str | None = None
    arm_id: str | None = None
    series_id: str
    cohort_ids: list[str]
    window_id: str | None = None
    data_cutoff_raw: str | None = None
    dlt_results_coverage: Literal["complete", "partial", "unread", "unclear"] = "unclear"
    has_observed_dlt_results: bool | None = None
    explicit_no_dlt_in_scope: bool | None = None
    unresolved_positive_dlt: bool | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class DLTObservation(StrictModel):
    observation_id: str
    study_id: str
    result_scope_id: str
    cohort_id: str | None = None
    window_id: str | None = None
    count_kind: Literal["patients", "dlt_count", "events", "episodes", "named_toxicities_only", "unclear"] = "unclear"
    reported_count: int | None = None
    explicit_patient_count: int | None = None
    same_patient_repeated_events: bool | None = None
    denominator_count_id: str | None = None
    reporting_status: Literal["reported", "not_reported", "explicitly_unreported", "unclear", "explicitly_not_evaluable", "not_applicable", "conflicting"] = "reported"
    toxicities_raw: list[str] = Field(default_factory=list)
    provenance: ValueProvenance = Field(default_factory=ValueProvenance)
    evidence: list[EvidenceLink] = Field(default_factory=list)


class CountEquation(StrictModel):
    equation_id: str
    value_type: Literal["population_count", "dlt_patient_count"]
    total_record_id: str
    component_record_ids: list[str]
    mutually_exclusive: bool
    exhaustive: bool
    scope_description: str | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)


class UnresolvedItem(StrictModel):
    item_id: str
    category: str
    description: str
    affected_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)


class ExtractionCoverage(StrictModel):
    category: Literal[
        "study_structure", "population", "eligibility", "interventions", "dosing",
        "design", "population_counts", "dlt_definition", "dlt_window", "dlt_results",
    ]
    status: Literal["extracted", "not_reported", "material_not_covered", "conflict"]
    source_ids: list[str] = Field(default_factory=list)
    note: str | None = None


class ReportedExtraction(StrictModel):
    schema_version: Literal["phase1_extraction_v3", "phase1_extraction_v3.2", "phase1_extraction_v3.3"] = SCHEMA_VERSION
    exposure_contract: Literal["cohort_exposures_v1"] | None = None
    document_id: str
    title: str | None = None
    studies: list[Study] = Field(default_factory=list)
    study_parts: list[StudyPart] = Field(default_factory=list)
    arms: list[Arm] = Field(default_factory=list)
    populations: list[PopulationDescription] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    regimens: list[Regimen] = Field(default_factory=list)
    designs: list[TrialDesign] = Field(default_factory=list)
    dlt_windows: list[DLTWindow] = Field(default_factory=list)
    dlt_definitions: list[DLTDefinition] = Field(default_factory=list)
    dose_series: list[DoseSeries] = Field(default_factory=list)
    cohorts: list[Cohort] = Field(default_factory=list)
    analysis_populations: list[AnalysisPopulation] = Field(default_factory=list)
    population_counts: list[PopulationCount] = Field(default_factory=list)
    result_scopes: list[ResultScope] = Field(default_factory=list)
    dlt_observations: list[DLTObservation] = Field(default_factory=list)
    count_equations: list[CountEquation] = Field(default_factory=list)
    unresolved_items: list[UnresolvedItem] = Field(default_factory=list)
    extraction_coverage: list[ExtractionCoverage] = Field(default_factory=list)
    material_coverage_complete: bool | None = None
    extraction_complete: bool = False
    coverage_notes: list[str] = Field(default_factory=list)


class RuleApplication(StrictModel):
    rule_application_id: str
    # R3 remains readable for historical audit only; the resolver never emits it.
    rule_id: Literal["R1", "R2", "R3", "D0", "DENOMINATOR_SELECTION", "DETERMINISTIC_DERIVATION"]
    version: str
    target_id: str
    target_field: str
    input_record_ids: list[str]
    output_value: int | float | str
    reason: str


class Issue(StrictModel):
    issue_id: str
    severity: Literal["warning", "row_blocker"]
    code: str
    affected_record_ids: list[str]
    field: str | None = None
    message: str
    source_ids: list[str] = Field(default_factory=list)
    suggested_action: str | None = None


class CurveRow(StrictModel):
    curve_row_id: str
    document_id: str
    study_id: str
    part_id: str | None = None
    arm_id: str | None = None
    cohort_id: str
    series_id: str
    result_scope_id: str
    regimen_id: str
    varying_component_id: str
    dose_value: float | None
    dose_unit: str
    dose_raw: str
    exposures: list[CohortExposure] = Field(default_factory=list)
    exposure_coverage: Literal["complete", "partial", "unknown"] = "unknown"
    exposure_issues: list[str] = Field(default_factory=list)
    window_id: str | None = None
    n: int | None = None
    y: int | None = None
    rate: float | None = None
    reported_treated_n: int | None = None
    reported_dlt_evaluable_n: int | None = None
    reported_dlt_count: int | None = None
    numerator_basis: Literal["reported", "derived", "inferred_patient_count", "imputed_zero", "missing"] = "missing"
    # Historical design fields remain readable, but cannot make a row usable.
    denominator_basis: Literal["reported", "derived", "design_inferred", "missing"] = "missing"
    denominator_role: Literal["paired_dlt", "dlt_evaluable", "treated", "equivalent_treated", "design_assumed_cohort", "unknown"] = "unknown"
    rule_ids: list[str] = Field(default_factory=list)
    design_candidate_ns: list[int] = Field(default_factory=list)
    included_in_default: bool = False
    included_in_reported_only: bool = False
    outcome_usable: bool = False
    outcome_exclusion_reasons: list[str] = Field(default_factory=list)
    exclusion_reasons: list[str] = Field(default_factory=list)
    evidence_source_ids: list[str] = Field(default_factory=list)
    policy_version: str = POLICY_VERSION

    @model_validator(mode="after")
    def validate_binomial(self) -> "CurveRow":
        if self.n is not None and self.n <= 0:
            raise ValueError("n must be positive")
        if self.y is not None and self.y < 0:
            raise ValueError("y cannot be negative")
        if has_retired_r3_provenance(self):
            self.included_in_default = False
            self.included_in_reported_only = False
            self.outcome_usable = False
            reason = "retired_r3_denominator"
            for reasons in (self.exclusion_reasons, self.outcome_exclusion_reasons):
                if reason not in reasons:
                    reasons.append(reason)
        # y > n is retained as an excluded row plus an Issue so the source
        # conflict remains inspectable instead of making the whole artifact
        # impossible to serialize.
        return self


class ScopeCoverageAssessment(StrictModel):
    result_scope_id: str
    status: Literal["ready", "missing", "unreadable", "conflict", "unknown"]
    reasons: list[str] = Field(default_factory=list)
    affected_source_ids: list[str] = Field(default_factory=list)
    basis: Literal["programmatic", "model_reported", "mixed"] = "mixed"


class ResolvedExtraction(StrictModel):
    schema_version: Literal["phase1_extraction_v3", "phase1_extraction_v3.2", "phase1_extraction_v3.3"] = SCHEMA_VERSION
    policy_version: str = POLICY_VERSION
    document_id: str
    reported_extraction: ReportedExtraction
    curve_rows: list[CurveRow] = Field(default_factory=list)
    scope_coverage: list[ScopeCoverageAssessment] = Field(default_factory=list)
    rule_applications: list[RuleApplication] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
