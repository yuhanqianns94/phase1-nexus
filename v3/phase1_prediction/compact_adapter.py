"""Format-only boundary from resolved v3 evidence to the v1 computational contract.

One record is one publication, not one dose series. All actual resolved N/y
values are explicit, so the reference compact reader cannot impute new points.
Scope keys prevent pooling different arms, windows or varying components.
Unparseable/conflicting structured facts remain missing; no LLM is called here.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections import defaultdict
from dataclasses import asdict
from typing import Any
from types import SimpleNamespace

from .models import LibrarySnapshot, TargetTrial


def _wrapped_fields(data):
    # v1 intentionally unwraps only evidence-bearing top-level fields. Do not
    # apply recursively: Dose_Value has a numeric `value` but is not a wrapper.
    for value in data.values():
        if isinstance(value, dict) and "value" in value:
            value.setdefault("evidence", "Adapted from resolved v3 data; see source provenance")
    return data


def _days(raw):
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[- ]?\s*(days?|weeks?)\s*", str(raw or ""), re.I)
    return float(match[1]) * (7 if match[2].lower().startswith("week") else 1) if match else None


def _schedule(phase):
    raw = str(phase.get("frequency_raw") or "").strip()
    aliases = {"once daily": "QD", "daily": "QD", "twice daily": "BID", "three times daily": "TID",
               "every 2 weeks": "Q2W", "every 3 weeks": "Q3W", "every 4 weeks": "Q4W"}
    frequency = aliases.get(raw.casefold(), raw)
    days_raw = re.sub(r"^days?\s*", "", str(phase.get("administration_days_raw") or ""), flags=re.I)
    days = None
    if re.fullmatch(r"\d+(?:\s*[,;]\s*\d+)*", days_raw):
        days = sorted({int(v) for v in re.split(r"\s*[,;]\s*", days_raw)})
    elif re.fullmatch(r"\d+\s*[-–]\s*\d+", days_raw):
        lo, hi = [int(v) for v in re.split(r"\s*[-–]\s*", days_raw)]
        if 0 < lo <= hi <= 366:
            days = list(range(lo, hi + 1))
    return {"Frequency": {"frequency_standard": frequency or None},
            "Schedule": {"cycle_length_days": _days(phase.get("cycle_length_raw")), "administration_days": days}}


def _criterion(raw):
    text = str(raw)
    grade = re.search(r"\bgrade\s*(>=|≥|>|=)?\s*([0-5])\b", text, re.I)
    duration = re.search(r"(>=|≥|>|<=|≤|<)?\s*(\d+(?:\.\d+)?)\s*(hours?|days?|weeks?)\b", text, re.I)
    name = text
    if grade:
        name = (text[:grade.start()] + text[grade.end():]).strip(" ,;:-")
    return {"DLT_Name": name,
            "DLT_definition_Grade": {"operator": (grade[1] or "=").replace("≥", ">="), "grade_value": int(grade[2])} if grade else None,
            "DLT_definition_Duration": {"operator": (duration[1] or "=").replace("≥", ">=").replace("≤", "<="), "duration_value": float(duration[2]), "duration_unit": duration[3].rstrip("s")} if duration else None,
            "DLT_definition_comments": text}


def matching_data(extraction, row, features, overrides):
    """Preserve structured extraction fields that the former flat adapter lost."""
    populations = [p for p in extraction.populations if p.study_id == row.study_id and p.part_id in {None, row.part_id}]
    regimen = next((r for r in extraction.regimens if r.regimen_id == row.regimen_id), None)
    component = next((c for c in regimen.components if c.component_id == row.varying_component_id), None) if regimen else None
    intervention = next((i for i in extraction.interventions if component and i.intervention_id == component.intervention_id), None)
    windows = [w for w in extraction.dlt_windows if w.window_id == row.window_id]
    definitions = [d for d in extraction.dlt_definitions if d.study_id == row.study_id and d.part_id in {None, row.part_id}]
    data = flat_fields(features)
    data["Trial_Inclusion"] = "; ".join(v for p in populations for v in [*p.inclusion_criteria, *p.prior_treatment_requirements, *p.clinical_requirements])
    data["Trial_Exclusion"] = "; ".join(v for p in populations for v in p.exclusion_criteria)
    ages = [{"age_min": p.age_min, "age_max": p.age_max, "age_unit": p.age_unit, "age_group": None} for p in populations if p.age_min is not None or p.age_max is not None]
    if ages and all(a == ages[0] for a in ages):
        data["Age"] = {"value": ages[0]}
    if intervention:
        data["Mechanism_of_action"] = intervention.mechanism_raw
        data["Drug_target_gene_target"] = intervention.targets_raw
    data["DLT_definition"] = [_criterion(c) for d in definitions for c in d.criteria_raw]
    if len(windows) == 1:
        window = windows[0]
        days = window.normalized_days or _days(window.raw_definition)
        # Only explicit cycle counts are adapted. Native comparison never
        # imputes a cycle length.
        cycles = re.fullmatch(r"\s*(?:first\s+)?(\d+)\s*cycles?\s*", window.raw_definition, re.I)
        data["DLT_EVA"] = {"value": {"dlt_eval_duration_days": days, "dlt_eval_cycle_number": float(cycles[1]) if cycles else None,
                                      "normalization_status": "cycle_length_missing" if cycles and days is None else "reported",
                                      "dlt_eval_text": window.raw_definition}}
    phases = component.dosing_phases if component else []
    schedules = [_schedule(p.model_dump()) for p in phases]
    data["_schedule"] = schedules[0] if schedules and all(s == schedules[0] for s in schedules) else {}
    # Broad human text corrections do not authorize retaining contradicted
    # structured facts, or interpreting prose as a precise number/category.
    for module, keys in {"population": ("Age",), "intervention": ("Drug_target_gene_target",),
                         "dosing": ("_schedule",), "dlt_window": ("DLT_EVA",)}.items():
        if module in overrides:
            for key in keys:
                data.pop(key, None)
    if "eligibility" in overrides:
        data["Trial_Inclusion"], data["Trial_Exclusion"] = overrides["eligibility"], ""
    if "intervention" in overrides:
        data["Mechanism_of_action"] = [features["target_moa"]]
    if "dlt_definition" in overrides:
        data["DLT_definition"] = [_criterion(overrides["dlt_definition"])]
    from .profiles import extraction_profile
    data["_comparison_profile"] = extraction_profile(extraction, row, overrides)
    return _wrapped_fields(data)


def flat_fields(features):
    """Compatibility for in-memory callers lacking v3 structured metadata."""
    get = lambda key: str(features.get(key) or "")
    trial_type = {"single_component": "single", "single": "single", "combination": "combination"}.get(get("trial_type"), get("trial_type"))
    return _wrapped_fields({"Drug_Name": [get("drug")], "cancer_type": {"value": [{"core_term": get("cancer_type"), "mention": get("cancer_type")}]},
            "Admin_Route": [v.strip() for v in re.split(r"[·;,]", get("route")) if v.strip()],
            "Dose_Form": [v.strip() for v in re.split(r"[·;,]", get("dose_form")) if v.strip()],
            "schema": get("schema_embedding"), "Trial_Inclusion": get("eligibility_embedding"),
            "DLT_definition": [_criterion(get("dlt_definition"))] if get("dlt_definition") else [],
            "DLT_EVA": {"value": {"dlt_eval_duration_days": _days(get("dlt_eva")), "dlt_eval_text": get("dlt_eva")}},
            "Mechanism_of_action": [get("target_moa")] if get("target_moa") else [],
            "Trial_Type": {"value": trial_type}, "Gender_Sex": get("gender")})


def target_query(target: TargetTrial):
    target = target.validated()
    data = flat_fields(target.features())
    age = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(years?|months?|days?)\s*", target.age, re.I)
    if age:
        data["Age"] = {"value": {"age_min": float(age[1]), "age_max": float(age[2]), "age_unit": age[3]}}
    elif target.age in {"adult", "children", "adolescent", "adult_and_children"}:
        data["Age"] = {"value": {"age_group": target.age}}
    if target.schedule:
        data["dose_levels"] = [{**_schedule({"frequency_raw": target.schedule}), "is_actual_escalation_dose": True,
                                "Drug_Name": target.drug, "Dose_Unit": target.dose_unit}]
    data.update(target.query_fields)
    data.update({"toxicity_curve_drug": target.drug, "Dose_Unit": target.dose_unit})
    if not data.get("Trial_Type", {}).get("value"):
        data["Trial_Type"] = {"value": "single"}
    if target.planned_doses:
        data["planned_doses"] = list(target.planned_doses)
    if target.exposure_rows:
        from .target_configurations import parse_rows
        data = parse_rows(list(target.exposure_rows), data, target.selected_scenario)
    return _wrapped_fields(data)


def library_records(library: LibrarySnapshot):
    grouped = defaultdict(list)
    for series in library.series:
        grouped[series.document_id].append(series)
    metadata = {r["pmcid"]: r["data"] for r in library.matching_records}
    records = []
    for document_id in sorted(set(grouped) | set(metadata)):
        series_list = grouped.get(document_id, [])
        sources = [copy.deepcopy(s.matching_data or flat_fields(s.features)) for s in series_list]
        data = {}
        for key in set().union(*(s.keys() for s in sources)) - {"_schedule"}:
            values = [s[key] for s in sources if key in s and s[key] not in (None, "", [])]
            if not values:
                continue
            if all(v == values[0] for v in values):
                data[key] = values[0]
            elif all(isinstance(v, list) for v in values):
                data[key] = []
                for values_list in values:
                    for v in values_list:
                        if v not in data[key]:
                            data[key].append(v)
            elif key == "cancer_type":
                data[key] = {"value": [v for value in values for v in value.get("value", [])]}
            elif key in {"schema", "Trial_Inclusion", "Trial_Exclusion"}:
                data[key] = "; ".join(dict.fromkeys(str(v) for v in values))
            # Conflicting publication-level age/window/trial type stay missing.
        if document_id in metadata:
            data = copy.deepcopy(metadata[document_id])
        data.update({"dose_levels": [], "dlt_count_at_each_dose_level": [], "_v3_scope_map": {}, "_comparison_scopes": {}})
        for series, source in zip(series_list, sources):
            scope = "v3scope_" + hashlib.sha256(series.series_key.encode()).hexdigest()[:20]
            data["_v3_scope_map"][scope] = series.series_key
            if source.get("_comparison_profile"):
                data["_comparison_scopes"][scope] = source["_comparison_profile"]
            for point in series.points:
                level = scope + "_" + point.curve_row_id
                data["dose_levels"].append({"Dose_Level": level, "Dose_Value": {"value_type": "single", "value": point.dose},
                    "Dose_Unit": point.unit, "Drug_Name": series.intervention_name, "arm_or_cohort": scope,
                    "_v3_configuration_scope": True, "is_actual_escalation_dose": True, "drug_role": "escalated_study_drug", **source.get("_schedule", {})})
                data["dlt_count_at_each_dose_level"].append({"dose_level": level, "arm_or_cohort": scope,
                    "drug_name": series.intervention_name, "is_actual_escalation_dose": True,
                    "n_patients_with_dlt": point.y, "n_patients_evaluable_for_dlt": point.n,
                    "denominator_source_override": point.denominator_role + ":" + point.denominator_basis,
                    "dlt_value_origin": "v3_resolved_" + point.numerator_basis, "evidence": asdict(point)})
        records.append({"pmcid": document_id, "path": "v3-resolved://" + document_id, "data": _wrapped_fields(data)})
    return records


def publication_matching_data(extraction, overrides):
    """Include the entire accepted publication in the IDF corpus, even if no
    resolved dose–N–DLT point is usable. Missing outcomes never remove prose.
    """
    from .evidence import _features
    scopes = list(extraction.dose_series)
    if not scopes:
        scopes = [SimpleNamespace(study_id=s.study_id, part_id=None, arm_id=None,
                   series_id="metadata_only", regimen_id=extraction.regimens[0].regimen_id if extraction.regimens else "",
                   varying_component_id=extraction.regimens[0].components[0].component_id if extraction.regimens and extraction.regimens[0].components else "",
                   window_id=None) for s in extraction.studies]
    metadata = []
    for scope in scopes:
        _, _, features = _features(extraction, scope, overrides)
        metadata.append(matching_data(extraction, scope, features, overrides))
    # Reuse the same publication merge semantics, without manufacturing points.
    proxies = [SimpleNamespace(document_id=extraction.document_id, matching_data=d, features={}, series_key=f"metadata_{i}", points=())
               for i, d in enumerate(metadata)]
    records = library_records(LibrarySnapshot("project_default", "extracted", (), tuple(proxies), 1)) if proxies else []
    from .profiles import extraction_profile
    data = records[0]["data"] if records else {}
    data["_comparison_profile"] = extraction_profile(extraction, overrides=overrides)
    return data
