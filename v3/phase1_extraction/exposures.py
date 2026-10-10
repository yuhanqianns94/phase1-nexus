"""Exposure identity and rendering, without imputing drugs or outcomes."""
from __future__ import annotations
import json
from typing import Iterable
from .models import Cohort, CohortExposure, DoseExpression
from .drug_ordering import clinical_payload


def dose_text(dose: DoseExpression | None) -> str:
    if dose is None:
        return ""
    if dose.value_kind == "scalar" and dose.value is not None:
        unit = dose.unit_normalized or dose.unit_raw or "unit not reported"
        return f"{dose.comparator or ''}{dose.value:g} {unit}"
    return dose.raw_expression


def exposure_identity(exposures: Iterable[CohortExposure | dict]) -> str:
    """Order-independent clinical identity; IDs/citations are not identity."""
    values = []
    for item in exposures:
        data = clinical_payload(item.model_dump(mode="json") if isinstance(item, CohortExposure) else dict(item))
        for key in ("exposure_id", "intervention_id", "component_id", "evidence", "note"):
            data.pop(key, None)
        data["name_raw"] = " ".join(str(data.get("name_raw", "")).casefold().split())
        data["dosing_phases"] = [{k: v for k, v in phase.items() if k not in {"dosing_phase_id", "evidence"}}
                                for phase in data.get("dosing_phases", [])]
        values.append(json.dumps(data, sort_keys=True, ensure_ascii=False))
    return json.dumps(sorted(values), ensure_ascii=False)


def exposure_problems(cohort: Cohort, varying_component_id: str | None = None) -> list[str]:
    problems, seen = [], set()
    for e in cohort.exposures:
        prefix = e.name_raw or e.exposure_id
        if e.exposure_id in seen:
            problems.append(f"{prefix}: duplicate exposure ID")
        seen.add(e.exposure_id)
        if not e.name_raw.strip():
            problems.append("Exposure name is missing")
        if e.presence == "absent" and (e.dose is not None or e.exposed_n not in {None, 0}):
            problems.append(f"{prefix}: absent exposure has a dose or exposed patients")
        if e.presence == "absent" and e.dose_status != "not_applicable":
            problems.append(f"{prefix}: absent exposure must use dose status not_applicable")
        if e.dose_status == "reported" and e.dose is None:
            problems.append(f"{prefix}: dose marked reported but missing")
        if e.dose_status in {"not_reported", "not_applicable"} and e.dose is not None:
            problems.append(f"{prefix}: dose conflicts with its reporting status")
        if e.presence == "present" and e.dose_status == "not_applicable":
            problems.append(f"{prefix}: present drug has dose status not_applicable")
        if e.dose_status == "conflicting":
            problems.append(f"{prefix}: source doses conflict")
        if e.exposed_n is not None and e.exposed_n < 0:
            problems.append(f"{prefix}: exposed patient count is negative")
        if e.dose and e.dose.value is not None and e.dose.value <= 0:
            problems.append(f"{prefix}: administered dose must be positive; absence is not dose zero")
        if varying_component_id and e.component_id == varying_component_id:
            if cohort.actual_status == "actual" and e.presence == "absent":
                problems.append(f"{prefix}: actual cohort's primary drug is marked absent")
            primary_unit = cohort.dose.unit_normalized or cohort.dose.unit_raw
            exposure_unit = (e.dose.unit_normalized or e.dose.unit_raw) if e.dose else None
            if (e.dose and e.dose.value_kind == cohort.dose.value_kind == "scalar"
                    and primary_unit == exposure_unit and e.dose.value != cohort.dose.value):
                problems.append(f"{prefix}: exposure dose disagrees with cohort primary dose")
    return problems


PRESENCE_LABELS = {"present": "true (used)", "absent": "false (not used)", "unknown": "unknown", "mixed": "mixed patients"}
DOSE_STATUS_LABELS = {"not_reported": "dose not reported", "not_applicable": "not applicable", "unknown": "dose unknown",
                      "unparsed": "dose needs parsing", "conflicting": "conflicting doses", "reported": "dose reported"}


def exposure_text(e: CohortExposure) -> str:
    label = e.name_raw + (" [class]" if e.entity_kind == "drug_class" else "")
    dose = dose_text(e.dose)
    if e.presence == "present" and e.dose_status == "reported" and dose:
        value = dose
    else:
        value = f"{PRESENCE_LABELS[e.presence]}; {DOSE_STATUS_LABELS[e.dose_status]}" + (f" ({dose})" if dose else "")
    if e.reporting_scope == "pooled":
        value += "; pooled / not stratified"
    if e.dose_status == "conflicting":
        details = e.dose_conflict_details or ([e.note] if e.note else [])
        if details:
            value += "; " + " | ".join(details)
    if e.formulation_raw:
        value += f"; {e.formulation_raw}"
    return f"{label}: {value}"


def exposure_schedule(e: CohortExposure) -> str:
    if e.schedule_raw:
        return e.schedule_raw
    phases = []
    for p in e.dosing_phases:
        parts = [p.phase_label, dose_text(p.dose), p.route_raw, p.frequency_raw, p.administration_days_raw,
                 p.cycle_length_raw, p.duration_raw, p.infusion_duration_raw, p.infusion_rate_raw,
                 p.start_condition_raw, p.end_condition_raw]
        phases.append(" · ".join(dict.fromkeys(v for v in parts if v)))
    return "; ".join(phases)
