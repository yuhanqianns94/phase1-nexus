from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from phase1_prediction.profile_similarity import DEFAULT_MIN_DRUG_SIMILARITY


def fingerprint(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DrugSlot(Contract):
    model_drug_id: str
    name: str
    research_priority: int = Field(ge=1)
    research_role: str
    priority_basis: str = "deterministic_fallback"
    priority_reason: str | None = None
    priority_evidence: list[dict] = Field(default_factory=list)
    ambiguous: bool = False
    mode: Literal["numeric", "background"] = "background"
    reference_dose: float | None = Field(default=None, gt=0)
    unit: str | None = None
    profile: dict = Field(default_factory=dict)


class Exposure(Contract):
    drug_id: str
    presence: Literal["present", "absent", "unknown", "mixed"] = "unknown"
    dose: float | None = Field(default=None, gt=0)
    dose_status: str = "unknown"
    candidates: list[float] = Field(default_factory=list)
    lower: float | None = Field(default=None, gt=0)
    upper: float | None = Field(default=None, gt=0)
    # Computational coordinate bookkeeping, not an extracted clinical value.
    # Source axes use zero; transformed axes preserve the same physical prior.
    log_relative_dose_prior_mean: float = 0.
    exposed_n: int | None = Field(default=None, ge=0)
    dose_subgroups: list[tuple[float, int]] = Field(default_factory=list)
    background_id: str = "reported_background"
    source: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def valid_dose_support(self):
        if any(v <= 0 for v in self.candidates) or (self.lower is not None and self.upper is not None and self.lower > self.upper):
            raise ValueError('Invalid positive dose support')
        return self


class Observation(Contract):
    observation_id: str
    cohort_id: str
    y: int = Field(ge=0)
    n: int = Field(gt=0)
    exposures: list[Exposure]
    source_row_ids: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_counts(self):
        if self.y > self.n:
            raise ValueError("DLT patients exceed evaluable patients")
        if len({e.drug_id for e in self.exposures}) != len(self.exposures):
            raise ValueError("Duplicate drug exposure in an observation")
        for e in self.exposures:
            if e.dose_subgroups and (sum(n for _, n in e.dose_subgroups) != self.n or any(d <= 0 or n <= 0 for d, n in e.dose_subgroups)):
                raise ValueError('Dose subgroup counts must partition the DLT-evaluable denominator')
            if e.presence == "mixed" and e.exposed_n is not None and e.exposed_n > self.n:
                raise ValueError("Exposed subgroup exceeds cohort size")
        return self


class ModelScope(Contract):
    version: str = "model_scope_v1"
    document_id: str
    scope_id: str
    label: str
    drugs: list[DrugSlot]
    observations: list[Observation]
    profile: dict = Field(default_factory=dict)
    lineage: dict = Field(default_factory=dict)
    exclusions: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def stable_slots(self):
        if not self.drugs:
            raise ValueError("A model scope needs at least one reported drug")
        ids = [d.model_drug_id for d in self.drugs]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate drug slot")
        if [d.research_priority for d in self.drugs] != list(range(1, len(ids) + 1)):
            raise ValueError("Slots must be stored in consecutive research order")
        for d in self.drugs:
            if d.mode == "numeric" and (d.reference_dose is None or not d.unit):
                raise ValueError("Numeric slot requires reference dose and unit")
        for o in self.observations:
            if {e.drug_id for e in o.exposures} != set(ids):
                raise ValueError("Every observation must explicitly include every slot")
            by_id={e.drug_id:e for e in o.exposures}
            o.exposures=[by_id[drug_id] for drug_id in ids]
        return self


class PriorSpec(Contract):
    prior_id: str = "multidrug_weak_v1"
    intercept_mean: float = -1.1
    intercept_sd: float = Field(default=2.5, gt=0)
    log_slope_mean: float = 0.0
    log_slope_sd: float = Field(default=1.0, gt=0)
    pair_sd: float = Field(default=1.0, gt=0)
    triple_sd: float = Field(default=0.5, gt=0)
    missing_log_dose_sd: float = Field(default=1.0, gt=0)
    unknown_presence_probability: float = Field(default=0.5, gt=0, lt=1)


class SamplerSpec(Contract):
    warmup: int = Field(default=750, ge=1)
    samples: int = Field(default=750, ge=4)
    chains: int = Field(default=2, ge=2)
    seed: int = 20260919
    target_accept: float = Field(default=0.9, gt=0, lt=1)
    max_tree_depth: int = Field(default=10, ge=1, le=15)
    retry_count: int = Field(default=1, ge=0, le=3)
    max_rhat: float = Field(default=1.05, gt=1)
    min_ess: float = Field(default=100, gt=0)


class TargetDrug(Contract):
    drug_id: str
    name: str
    research_priority: int = Field(ge=1)
    research_role: str = "partner"
    mode: Literal["numeric", "background"] = "numeric"
    reference_dose: float | None = Field(default=None, gt=0)
    unit: str | None = None
    profile: dict = Field(default_factory=dict)
    transport_mode: Literal["relative_dose_analogy", "same_entity_unit_conversion"] = "relative_dose_analogy"
    allow_background_anchor: bool = False

    @model_validator(mode='after')
    def consistent_transport(self):
        if self.transport_mode == 'same_entity_unit_conversion' and (self.mode == 'background' or self.allow_background_anchor):
            raise ValueError('Same-drug unit conversion requires numeric doses and cannot use a background anchor. Choose relative-dose analogy for anchoring.')
        return self


class RegimenPoint(Contract):
    level_id: str
    # Positive number: administered dose; None: present without dose; 0: explicitly absent.
    doses: dict[str, float | None]

    @field_validator('doses', mode='before')
    @classmethod
    def no_boolean_doses(cls, value):
        if isinstance(value, dict) and any(isinstance(v, bool) for v in value.values()):
            raise ValueError('Use 0 for absent or null for unknown dose; true is not a dose of 1')
        return value

    @model_validator(mode="after")
    def finite_nonnegative(self):
        if any(v is not None and v < 0 for v in self.doses.values()):
            raise ValueError("Target doses must be nonnegative; zero means absent")
        return self


class TargetRegimen(Contract):
    version: str = "target_regimen_v1"
    name: str
    drugs: list[TargetDrug]
    points: list[RegimenPoint]
    path_level_ids: list[str] = Field(default_factory=list)
    profile: dict = Field(default_factory=dict)
    target_dlt: float = Field(default=0.25, gt=0, lt=1)
    mapping_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    minimum_drug_similarity: float = Field(default=DEFAULT_MIN_DRUG_SIMILARITY, ge=0, le=1)

    @model_validator(mode="after")
    def complete(self):
        from phase1_extraction.clinical_protocol import planned_issues
        if self.profile.get('planning'):
            issues=planned_issues(self.model_dump())
            if issues:raise ValueError('Incomplete planned trial: '+' '.join(issues))
        ids = [d.drug_id for d in self.drugs]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Unique target drug IDs are required")
        if [d.research_priority for d in self.drugs] != list(range(1, len(ids) + 1)):
            raise ValueError("Target drugs must follow stable research order")
        if not self.points or len({p.level_id for p in self.points}) != len(self.points):
            raise ValueError("At least one uniquely identified target point is required")
        for d in self.drugs:
            if d.mode == "numeric" and (d.reference_dose is None or not d.unit):
                raise ValueError("Each numeric target needs a saved reference dose and unit")
        for p in self.points:
            if set(p.doses) != set(ids):
                raise ValueError("Each point must specify every target drug")
            for d in self.drugs:
                if d.mode == 'background' and p.doses[d.drug_id] not in (None,0):
                    raise ValueError(f'{d.name}: dose-unknown coordinates accept only absent or used-without-dose; use numeric mode for positive doses.')
        if len(set(self.path_level_ids)) != len(self.path_level_ids) or not set(self.path_level_ids) <= {p.level_id for p in self.points}:
            raise ValueError("Path must contain distinct existing level IDs")
        return self
