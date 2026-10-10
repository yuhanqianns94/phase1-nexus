"""Build full-regimen observations from original resolved rows (no human gold).

Never project through the legacy single-axis EvidencePoint eligibility filter.
Migration writes derived sidecars only; original extraction/reviews are read-only.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from phase1_extraction.artifacts import read_json, sha256_file
from phase1_extraction.dose_units import dose_unit_for_value
from phase1_extraction.drug_ordering import entity_key, ordered_catalog, clinical_payload
from phase1_extraction.models import CohortExposure, CurveRow, ReportedExtraction
from phase1_prediction.evidence import discover_article_paths
from phase1_prediction.profiles import exposure_profile, extraction_profile
from .grouping import group_rows
from .contracts import DrugSlot, Exposure, ModelScope, Observation, fingerprint


def _numbers(e):
    d = e.dose
    if e.presence == "absent":
        return None, None, [], None, None
    if e.dose_semantics == 'subgroups' and e.dose_subgroups:
        # Every allocation carries its own units. The parent expression can
        # summarize doses using a different mass prefix, or omit units entirely.
        allocations = [_numbers(e.model_copy(update={
            'dose': subgroup.dose, 'dose_semantics': 'scalar', 'dose_status': 'reported'}))
            for subgroup in e.dose_subgroups]
        if any(value is None or not unit for value, unit, *_ in allocations):
            raise ValueError('Invalid subgroup dose/unit')
        units = {unit for _, unit, *_ in allocations}
        if len(units) != 1:
            raise ValueError('Incompatible subgroup dose units')
        return None, units.pop(), [value for value, *_ in allocations], None, None
    if not d:
        return None, None, [], None, None
    unit = "".join((dose_unit_for_value(d) or "").casefold().split()).replace("²", "2").replace("^2", "2").replace("µ", "u").replace("μ", "u").replace("mcg", "ug")
    # Only dimensional mass-prefix conversion; no schedule/weight imputation.
    factor = 1.0
    for prefix, scale in (("ug", .001), ("mg", 1.), ("g", 1000.)):
        if unit == prefix or unit.startswith(prefix + "/"):
            unit, factor = "mg" + unit[len(prefix):], scale
            break
    value = d.value if d.value_kind == "scalar" and e.dose_status == "reported" else None
    # A sequence or patient-specific allocation is not a cohort-level lottery.
    candidates = list(d.discrete_values) if e.dose_semantics == 'alternative_unknown' else []
    if d.value_kind == "scalar" and e.dose_status == "conflicting":
        # A lone retained scalar is not the complete set of conflicting doses.
        value = None
    clean = lambda x: float(x) * factor if x is not None and math.isfinite(x) and x > 0 else None
    lo, hi = (d.range_min, d.range_max) if e.dose_semantics == 'alternative_unknown' else (None, None)
    return clean(value), unit or None, [clean(x) for x in candidates if clean(x)], clean(lo), clean(hi)


def _row_exposures(row, extraction):
    if row.exposures:
        return row.exposures
    cohort = next((c for c in extraction.cohorts if c.cohort_id == row.cohort_id), None)
    if cohort and cohort.exposures:
        return cohort.exposures
    regimen = next((r for r in extraction.regimens if r.regimen_id == row.regimen_id), None)
    names = {i.intervention_id: i.name_raw for i in extraction.interventions}
    return [CohortExposure(exposure_id=f"legacy_{row.cohort_id}_{c.component_id}",
                          name_raw=names.get(c.intervention_id, c.component_id),
                          component_id=c.component_id, intervention_id=c.intervention_id, role=c.role,
                          presence="present" if c.component_id == row.varying_component_id else "unknown",
                          dose=cohort.dose if cohort and c.component_id == row.varying_component_id else None,
                          dose_status="reported" if cohort and c.component_id == row.varying_component_id else "unknown")
            for c in (regimen.components if regimen else [])]


def outcome_reasons(row, evidence_policy):
    # The dedicated field can carry newer outcome-only exclusions that are not
    # duplicated in the legacy scalar-curve exclusion list.
    reasons = list(row.outcome_exclusion_reasons)
    if row.n is None or row.y is None or row.n <= 0 or not 0 <= row.y <= row.n:
        reasons.append("missing_or_invalid_binomial_counts")
    # Outcome validity is separate from scalar dose eligibility.
    for reason in row.exclusion_reasons:
        if any(word in reason.casefold() for word in ("denominator", "numerator", "not actual", "not used", "overlap", "not evaluable")):
            if reason not in reasons:
                reasons.append(reason)
    if evidence_policy == "reported_only" and any(rule in {"R1", "R2", "R3"} for rule in row.rule_ids):
        reasons.append("not_reported_only")
    return reasons


def _scope_key(row, extraction):
    # Background dose/presence and dose-series labels must not partition a surface.
    exposures = _row_exposures(row, extraction)
    schedules = []
    for e in exposures:
        phases = [{k: v for k, v in p.model_dump(mode="json").items()
                   if k not in {"dose", "dosing_phase_id", "evidence"}} for p in e.dosing_phases]
        if e.schedule_raw or phases:
            schedules.append((entity_key(e), e.schedule_raw, phases))
        if e.dose and e.dose.value_kind != 'scalar' and e.dose_semantics not in {'alternative_unknown','subgroups'}:
            # Until a numerical regimen coordinate is explicitly defined, retain
            # this complete schedule as its own background risk, not a fabricated
            # single dose or a random choice of sequential doses.
            schedules.append((entity_key(e), 'unresolved_regimen_coordinate', e.dose.model_dump(mode='json')))
    return (row.study_id, row.part_id, row.arm_id, row.regimen_id, row.window_id,
            fingerprint(sorted(schedules, key=lambda x: x[0])))


def build_article_scopes(paths, *, evidence_policy="project_default", review_policy="extracted"):
    extraction = ReportedExtraction.model_validate(read_json(paths.reported_extraction))
    payload = read_json(paths.curve_rows)
    rows = [CurveRow.model_validate(r) for r in payload.get("all_rows", payload.get(evidence_policy, []))]
    # Human review is an accuracy benchmark only, never a fitting input.
    review_policy = "extracted"
    overrides = {}
    series = {s.series_id: s for s in extraction.dose_series}
    # Preserve scopes with no usable outcomes so their prior-only status is visible.
    represented = {r.cohort_id for r in rows}
    for c in extraction.cohorts:
        if c.cohort_id in represented or c.actual_status != "actual" or c.series_id not in series:
            continue
        s = series[c.series_id]
        rows.append(CurveRow(curve_row_id=f"unresolved_{c.cohort_id}", document_id=extraction.document_id,
                             study_id=s.study_id, part_id=s.part_id, arm_id=s.arm_id,
                             cohort_id=c.cohort_id, series_id=s.series_id, result_scope_id="unresolved",
                             regimen_id=c.regimen_id, varying_component_id=s.varying_component_id,
                             window_id=s.window_id, dose_value=None, dose_unit="unknown",
                             dose_raw=c.dose.raw_expression, exposures=c.exposures))
    if not rows:
        for s in extraction.dose_series:
            rows.append(CurveRow(curve_row_id=f"prior_only_{s.series_id}", document_id=extraction.document_id,
                                 study_id=s.study_id, part_id=s.part_id, arm_id=s.arm_id,
                                 cohort_id=f"prior_only_{s.series_id}", series_id=s.series_id, result_scope_id="unresolved",
                                 regimen_id=s.regimen_id, varying_component_id=s.varying_component_id,
                                 window_id=s.window_id, dose_value=None, dose_unit="unknown", dose_raw=""))
    groups, all_notices = group_rows(extraction, rows, source_sha256=sha256_file(paths.reported_extraction),
                                    row_exposures=_row_exposures, legacy_key=_scope_key)
    interventions = {i.intervention_id: i.model_dump(mode="json") for i in extraction.interventions}
    scopes = []
    # Explicit exhaustive totals must not be counted alongside their components.
    records = {r.count_id: r.cohort_id for r in extraction.population_counts}
    records.update({r.observation_id: r.cohort_id for r in extraction.dlt_observations})
    pooled_totals = set()
    available = {r.cohort_id for r in rows if not outcome_reasons(r, evidence_policy)}
    for eq in extraction.count_equations:
        children = {records.get(x) for x in eq.component_record_ids} - {None}
        total = records.get(eq.total_record_id)
        if eq.exhaustive and eq.mutually_exclusive and children and children <= available and total not in children:
            pooled_totals.add(total)
    for key, members, grouping_audit in groups:
        exposures = [e for row in members for e in _row_exposures(row, extraction)]
        catalog = ordered_catalog(extraction, key[3], exposures)
        if not catalog:
            all_notices.append({"document_id": extraction.document_id, "reason": "no_reported_drugs", "scope": list(key)})
            continue
        regimen = next((r for r in extraction.regimens if r.regimen_id == key[3]), None)
        components = {c.component_id: c.model_dump(mode="json") for c in regimen.components} if regimen else {}
        slots = []
        for info in catalog:
            candidates = sorted([e for e in exposures if entity_key(e) == info["model_drug_id"]],
                                key=lambda e: ({"present":0,"mixed":1,"unknown":2,"absent":3}[e.presence], e.exposure_id))
            numbers = [_numbers(e) for e in candidates]
            unresolved = [e for e in candidates if e.dose and e.dose.value_kind != 'scalar'
                          and e.dose_semantics not in {'alternative_unknown','subgroups'}]
            if unresolved:
                all_notices.append({'document_id':extraction.document_id, 'drug_id':info['model_drug_id'],
                    'reason':'complete_regimen_background_pending_numerical_coordinate',
                    'raw_doses':sorted({e.dose.raw_expression for e in unresolved}),
                    'detail':'Outcomes retained at their complete reported regimen; no sequential-dose lottery or invented allocation.'})
            units = {unit for value, unit, discrete, lo, hi in numbers if unit and (value or discrete or lo or hi)}
            if len(units) > 1:
                raise ValueError(f"{extraction.document_id}: incompatible units for {info['name']}: {sorted(units)}")
            unit = next(iter(units), None)
            values = {v for value, u, ds, lo, hi in numbers if u == unit and unit
                      for v in [value, *ds, lo, hi] if v}
            profile = exposure_profile(candidates[0], interventions, components) if candidates else {
                "name": info["name"], "role": info["research_role"]}
            profile.update(model_drug_id=info["model_drug_id"], research_priority=info["research_priority"], research_role=info["research_role"])
            slots.append(DrugSlot(**info, mode="numeric" if values else "background",
                                  reference_dose=math.exp(sum(map(math.log, values)) / len(values)) if values else None,
                                  unit=unit, profile=profile))
        observations, rejected, seen = [], [], {}
        for row in sorted(members, key=lambda r: (r.cohort_id, r.curve_row_id)):
            reasons = outcome_reasons(row, evidence_policy)
            if row.cohort_id in pooled_totals:
                reasons.append("pooled_total_already_represented_by_exhaustive_subgroups")
            if reasons:
                rejected.append({"row_id": row.curve_row_id, "cohort_id": row.cohort_id, "reasons": reasons})
                continue
            row_exposures = {entity_key(e): e for e in _row_exposures(row, extraction)}
            states = []
            for slot in slots:
                e = row_exposures.get(slot.model_drug_id)
                if e is None:
                    states.append(Exposure(drug_id=slot.model_drug_id))
                    continue
                value, unit, discrete, lo, hi = _numbers(e)
                allocations = []
                if e.dose_semantics == 'subgroups' and e.dose_subgroups:
                    if any(g.population_basis != 'dlt_evaluable' for g in e.dose_subgroups):
                        raise ValueError('Dose allocation denominator is not confirmed DLT-evaluable')
                    for subgroup in e.dose_subgroups:
                        copy = e.model_copy(update={'dose':subgroup.dose, 'dose_semantics':'scalar', 'dose_status':'reported'})
                        dvalue, dunit, *_ = _numbers(copy)
                        if dvalue is None or dunit != slot.unit:
                            raise ValueError('Invalid subgroup dose/unit')
                        allocations.append((dvalue, subgroup.n))
                if slot.mode == "background" or unit != slot.unit:
                    value, discrete, lo, hi = None, [], None, None
                states.append(Exposure(drug_id=slot.model_drug_id, presence=e.presence, dose=value,
                                       candidates=discrete, lower=lo, upper=hi, dose_status=e.dose_status,
                                       exposed_n=e.exposed_n, dose_subgroups=allocations, source=clinical_payload(e)))
            obs = Observation(observation_id=row.curve_row_id, cohort_id=row.cohort_id, y=row.y, n=row.n,
                              exposures=states, source_row_ids=[row.curve_row_id],
                              provenance={"rule_ids": row.rule_ids, "denominator_basis": row.denominator_basis,
                                          "numerator_basis": row.numerator_basis, "evidence_ids": row.evidence_source_ids,
                                          **({"arm_id":row.arm_id,"series_id":row.series_id,"regimen_id":row.regimen_id,"window_id":row.window_id} if grouping_audit else {})})
            if row.cohort_id in seen:
                previous = seen[row.cohort_id]
                identity = lambda o: (o.y, o.n, [e.model_dump(exclude={"source"}) for e in o.exposures])
                if identity(previous) != identity(obs):
                    raise ValueError(f"Conflicting duplicate cohort: {extraction.document_id}/{row.cohort_id}")
                previous.source_row_ids.append(row.curve_row_id)
                continue
            observations.append(obs)
            seen[row.cohort_id] = obs
        first = members[0]
        profile = extraction_profile(extraction, first, overrides if review_policy != "extracted" else {})
        # Source-slot profiles, not a dose-dependent cohort array, are canonical for target mapping.
        configs = profile.get('configurations', [])
        config = dict(configs[0]) if configs else {'status':'actual'}
        for slot in slots:
            matching=[(r,e) for r in members for e in _row_exposures(r,extraction) if entity_key(e)==slot.model_drug_id]
            if matching:
                selected_row,selected_exposure=min(matching,key=lambda pair:({"present":0,"mixed":1,"unknown":2,"absent":3}[pair[1].presence],pair[0].cohort_id))
                selected_config=(extraction_profile(extraction,selected_row).get('configurations') or [{}])[0]
                selected_profile=next((d for d in selected_config.get('drugs',[]) if d.get('name')==selected_exposure.name_raw),slot.profile)
                slot.profile={**selected_profile,'model_drug_id':slot.model_drug_id,
                              'research_priority':slot.research_priority,'research_role':slot.research_role}
                states=sorted({e.presence for _,e in matching})
                if len(states)>1:
                    slot.profile['presence']=None
                    slot.profile['presence_states']=states
        config['drugs'] = [s.profile for s in slots]
        profile['configurations'] = [config]
        scopes.append(ModelScope(document_id=extraction.document_id,
                                 scope_id="scope_" + fingerprint([extraction.document_id, key])[:20],
                                 label=regimen.label if regimen else first.regimen_id,
                                 drugs=slots, observations=observations, profile=profile, exclusions=rejected,
                                 lineage={"root": str(paths.root), "scope_key": list(key),
                                          "series_ids": sorted({r.series_id for r in members}),
                                          "evidence_policy": evidence_policy, "review_policy": review_policy,
                                          **({"grouping":grouping_audit,"arm_ids":grouping_audit["arm_ids"]} if grouping_audit else {})}))
    return scopes, all_notices


def load_model_library(roots, *, evidence_policy="project_default", review_policy="extracted"):
    selected, duplicates = discover_article_paths(Path(p) for p in roots)
    scopes, notices = [], []
    for paths in selected.values():
        try:
            article_scopes, article_notices = build_article_scopes(paths, evidence_policy=evidence_policy, review_policy=review_policy)
            scopes.extend(article_scopes)
            notices.extend(article_notices)
        except (ValueError, OSError, KeyError) as error:
            notices.append({"document_id": paths.document_id, "reason": "needs_data_resolution", "detail": str(error)})
    return scopes, {"document_count": len(selected), "scope_count": len(scopes),
                    "observation_count": sum(len(s.observations) for s in scopes),
                    "duplicates": duplicates, "notices": notices,
                    "prior_only_scopes": [s.scope_id for s in scopes if not s.observations]}
