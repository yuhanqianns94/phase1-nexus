"""Stable regimen-wide research ordering. No drug-name-specific rules."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

ORDER_FIELDS = frozenset({"research_priority", "research_role", "priority_basis",
                         "priority_reason", "priority_evidence", "model_drug_id"})


def clinical_payload(value):
    """Strip model annotations without changing the clinical facts or citations."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: clinical_payload(v) for k, v in value.items() if k not in ORDER_FIELDS
                and not (k == 'dose_semantics' and v == 'unknown')
                and not (k == 'dose_subgroups' and not v)}
    if isinstance(value, list):
        result = [clinical_payload(v) for v in value]
        # Entity-array position is not a clinical fact; phase/event order is.
        for key in ('exposure_id', 'component_id'):
            if result and all(isinstance(v,dict) and v.get(key) for v in result):
                return sorted(result,key=lambda v:str(v[key]))
        return result
    return value


def entity_key(exposure):
    d = exposure.model_dump() if hasattr(exposure, "model_dump") else exposure
    # An ID is scoped by regimen by the caller. Class entities need no invented intervention.
    return d.get("component_id") or d.get("intervention_id") or (
        "entity_" + hashlib.sha256(json.dumps([
            " ".join(str(d.get("name_raw", "")).casefold().split()),
            d.get("entity_kind", "drug")], ensure_ascii=False).encode()).hexdigest()[:16])


def ordered_catalog(extraction, regimen_id, exposures=()):
    regimen = next((r for r in extraction.regimens if r.regimen_id == regimen_id), None)
    names = {i.intervention_id: i.name_raw for i in extraction.interventions}
    groups = defaultdict(list)
    for c in extraction.cohorts:
        if c.regimen_id == regimen_id:
            for e in c.exposures:
                groups[entity_key(e)].append(e)
    for e in exposures:
        groups[entity_key(e)].append(e)
    components = {c.component_id: c for c in regimen.components} if regimen else {}
    # Include every component even if absent from one cohort's exposure array.
    for key in components:
        groups.setdefault(key, [])
    varying = {s.varying_component_id for s in extraction.dose_series if s.regimen_id == regimen_id}
    values = []
    for key, records in groups.items():
        component = components.get(key)
        annotated = [r for r in [component, *records] if r and r.research_priority is not None]
        annotated.sort(key=lambda r: (r.priority_basis != "human_confirmed", r.research_priority))
        chosen = annotated[0] if annotated else None
        roles = {r.role for r in [component, *records] if r}
        main = key in varying or "escalated" in roles
        role = (chosen.research_role if chosen and chosen.research_role != "unclear" else
                "primary" if main else "partner" if roles & {"fixed_backbone", "combination_partner"} else "background")
        name = (names.get(component.intervention_id) if component else None) or next((r.name_raw for r in records), key)
        values.append({"model_drug_id": key, "name": name, "research_role": role,
                       "research_priority": chosen.research_priority if chosen else None,
                       "priority_basis": chosen.priority_basis if chosen else "deterministic_fallback",
                       "priority_reason": chosen.priority_reason if chosen else "Research role and source dose-finding component; stable identity breaks ties.",
                       "priority_evidence": [e.model_dump(mode="json") for e in chosen.priority_evidence] if chosen else [],
                       "ambiguous": len({r.research_priority for r in annotated}) > 1})
    rank = {"primary": 0, "co_primary": 0, "partner": 1, "background": 2, "unclear": 3}
    values.sort(key=lambda d: (rank[d["research_role"]], d["research_priority"] or 10**9, d["model_drug_id"]))
    primary_count = sum(v["research_role"] in {"primary", "co_primary"} for v in values)
    for index, value in enumerate(values, 1):
        value["research_priority"] = index
        if primary_count > 1 and value["research_role"] == "primary":
            value["research_role"] = "co_primary"
    return values


def apply_drug_ordering(extraction):
    """Return a copy; raw source and human gold records remain untouched."""
    from .models import EvidenceLink
    result = extraction.model_copy(deep=True)
    regimen_ids = {r.regimen_id for r in result.regimens} | {c.regimen_id for c in result.cohorts}
    for regimen_id in sorted(regimen_ids):
        catalog = {v["model_drug_id"]: v for v in ordered_catalog(result, regimen_id)}
        records = [e for c in result.cohorts if c.regimen_id == regimen_id for e in c.exposures]
        records += [c for r in result.regimens if r.regimen_id == regimen_id for c in r.components]
        for record in records:
            key = entity_key(record) if hasattr(record, "name_raw") else record.component_id
            for field in ORDER_FIELDS - {"model_drug_id", "priority_evidence"}:
                setattr(record, field, catalog[key][field])
            record.priority_evidence=[EvidenceLink.model_validate(e) for e in catalog[key]['priority_evidence']]
        for cohort in result.cohorts:
            if cohort.regimen_id == regimen_id:
                cohort.exposures.sort(key=lambda e: (e.research_priority or 10**9, entity_key(e)))
        for regimen in result.regimens:
            if regimen.regimen_id == regimen_id:
                regimen.components.sort(key=lambda c: (c.research_priority or 10**9, c.component_id))
    return result
