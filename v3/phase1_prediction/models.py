from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EvidencePolicy = Literal["project_default", "reported_only"]
ReviewPolicy = Literal["extracted", "reviewed_if_available", "reviewed_only"]


@dataclass(frozen=True)
class EvidencePoint:
    curve_row_id: str
    cohort_id: str
    dose: float
    unit: str
    source_unit: str
    n: int
    y: int
    denominator_role: str
    denominator_basis: str
    numerator_basis: str
    rule_ids: tuple[str, ...] = ()
    evidence_source_ids: tuple[str, ...] = ()
    source: str = "extraction"
    exposures: tuple[dict[str, Any], ...] = ()
    exposure_coverage: str = "unknown"


@dataclass(frozen=True)
class EvidenceSeries:
    series_key: str
    document_id: str
    study_id: str
    part_id: str | None
    arm_id: str | None
    series_id: str
    regimen_id: str
    varying_component_id: str
    window_id: str | None
    label: str
    intervention_name: str
    unit: str
    source_units: tuple[str, ...]
    study_group_id: str
    features: dict[str, str]
    points: tuple[EvidencePoint, ...]
    source_fingerprint: str
    review_status: str
    matching_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LibrarySnapshot:
    evidence_policy: EvidencePolicy
    review_policy: ReviewPolicy
    source_roots: tuple[str, ...]
    series: tuple[EvidenceSeries, ...]
    document_count: int
    duplicate_documents: dict[str, tuple[str, ...]] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    notices: dict[str, int] = field(default_factory=dict)
    fingerprint: str = ""
    matching_records: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TargetTrial:
    name: str
    drug: str
    disease: str
    planned_doses: tuple[float, ...]
    dose_unit: str
    target_dlt: float = 0.25
    route: str = ""
    schedule: str = ""
    regimen: str = ""
    dlt_definition: str = ""
    dlt_window: str = ""
    mechanism: str = ""
    age: str = ""
    trial_type: str = ""
    eligibility: str = ""
    sex: str = ""
    dose_form: str = ""
    query_fields: dict[str, Any] = field(default_factory=dict)
    exposure_rows: tuple[dict[str, Any], ...] = ()
    selected_scenario: str = "Main"

    def validated(self) -> "TargetTrial":
        if not self.drug.strip() or not self.dose_unit.strip():
            raise ValueError("Drug and dose unit are required.")
        doses = tuple(float(value) for value in self.planned_doses)
        if any(not math.isfinite(value) or value <= 0 for value in doses):
            raise ValueError("Planned doses must be finite positive values.")
        if any(right <= left for left, right in zip(doses, doses[1:])):
            raise ValueError("Planned doses must be strictly increasing.")
        if not math.isfinite(float(self.target_dlt)) or not 0 < float(self.target_dlt) < 1:
            raise ValueError("Target DLT probability must be a finite number between 0 and 1.")
        return TargetTrial(**{**asdict(self), "planned_doses": doses})

    def features(self) -> dict[str, str]:
        return {
            "drug": self.drug,
            "cancer_type": self.disease,
            "route": self.route,
            "schedule": self.schedule,
            "schema_embedding": self.regimen,
            "dlt_definition": self.dlt_definition,
            "dlt_eva": self.dlt_window,
            "target_moa": self.mechanism,
            "age": self.age,
            "trial_type": self.trial_type,
            "eligibility_embedding": self.eligibility,
            "gender": self.sex,
            "dose_form": self.dose_form,
        }


@dataclass(frozen=True)
class ScoredSeries:
    series: EvidenceSeries
    similarity: float
    available_weight: float
    field_scores: dict[str, float]


@dataclass(frozen=True)
class PredictionResult:
    format: str
    model_version: str
    run_id: str
    target: dict[str, Any]
    evidence_contract: dict[str, Any]
    selected_series: list[dict[str, Any]]
    excluded_series: list[dict[str, Any]]
    methods: dict[str, dict[str, Any]]
    generated_at_utc: str
    reference_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
