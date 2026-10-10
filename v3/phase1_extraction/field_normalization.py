"""Conservative, auditable field attribution; never changes counts or design."""
import re
from .models import ReportedExtraction


def biological_description(value):
    # A biological identity alone is not a physical formulation. Mixed strings
    # containing a genuine dosage form are left for source review.
    s=str(value or '').strip()
    return bool(re.search(r'\b(?:monoclonal antibody|igg\d? antibody|human igg\d?)\b',s,re.I)) and not bool(re.search(r'\b(?:solution|tablet|capsule|powder|suspension|lyophili|concentrate|vial)\b',s,re.I))


def normalize_fields(extraction):
    data=extraction.model_dump(mode='json');changes=[]
    interventions={i['intervention_id']:i for i in data['interventions']}
    for i in interventions.values():
        value=i.get('product_or_formulation')
        if biological_description(value):
            if value not in i.get('mechanism_raw',[]) and value!=i.get('drug_class_raw'):
                if not i.get('drug_class_raw'):i['drug_class_raw']=value
                else:i['mechanism_raw'].append(value)
            i['product_or_formulation']=None
            changes.append({'record':i['intervention_id'],'field':'product_or_formulation','before':value,'after':None,'reason':'Biological description retained under drug class/mechanism.'})
    for c in data['cohorts']:
        for e in c.get('exposures',[]):
            value=e.get('formulation_raw')
            if biological_description(value):
                i=interventions.get(e.get('intervention_id'))
                if not i:continue # Do not discard an unbound description.
                if value not in i.get('mechanism_raw',[]) and value!=i.get('drug_class_raw'):
                    if not i.get('drug_class_raw'):i['drug_class_raw']=value
                    else:i['mechanism_raw'].append(value)
                e['formulation_raw']=None
                changes.append({'record':e['exposure_id'],'cohort':c['cohort_id'],'field':'formulation_raw','before':value,'after':None,'reason':'Biological description retained under drug class/mechanism.'})
            if e.get('dose_status')=='conflicting':
                candidates=[]
                for u in data.get('unresolved_items',[]):
                    if set(u.get('affected_ids',[])) & {c['cohort_id'],e['exposure_id']}:
                        candidates.append(u.get('description',''))
                if e.get('note'):candidates.append(e['note'])
                for ev in e.get('evidence',[]):
                    if ev.get('source_text'):
                        candidates.append((ev.get('locator') or ', '.join(ev.get('source_ids',[]))) + ': ' + ev['source_text'])
                values=list(dict.fromkeys(x.strip() for x in candidates if x.strip()))
                if values and e.get('dose_conflict_details')!=values:
                    e['dose_conflict_details']=values
                    changes.append({'record':e['exposure_id'],'field':'dose_conflict_details','after':values,'reason':'Preserved source conflict wording; no conflicting numeric dose chosen.'})
    _normalize_controlled_fields(data, changes)
    # Existing unbounded monitoring prose remains unclear, never auto-promoted to
    # a formal DLT endpoint. New extractions explicitly classify window_kind.
    return ReportedExtraction.model_validate(data),changes


_MISSING = {'unknown', 'unclear', 'not specified', 'not reported', 'unreported', 'none reported', 'undecided', 'n/a'}


def _normalize_controlled_fields(data, changes):
    """Add conservative aliases in a copy; retain every raw field and explicit code.

    This compatibility pass does not interpret free clinical prose, choose a
    threshold, convert dose units, or replace model-supplied structured values.
    """
    from .controlled_vocabulary import canonical_value, choices

    def fill(record, target, catalog, raw):
        if record.get(target) is not None:
            return
        if not isinstance(raw, str) or not raw.strip() or raw.strip().casefold() in _MISSING:
            return
        value = canonical_value(catalog, raw)
        if value is None and 'other' in choices(catalog):
            value = 'other'
        if value is not None:
            record[target] = value
            changes.append({'field': target, 'after': value, 'reason': 'Added a controlled category; original source wording retained.'})

    def clause(c):
        if not c:
            return
        for target, catalog, source in [('metric_code', 'clinical_metric', 'metric'),
                ('category_code', 'condition_category', 'category'),
                ('comparator_code', 'comparison', 'comparator'),
                ('unit_code', 'condition_unit', 'unit_raw')]:
            fill(c, target, catalog, c.get(source))

    def dose(d):
        if not d:
            return
        fill(d, 'comparator_code', 'comparison', d.get('comparator'))
        # Numeric values remain exactly as transcribed. Prefer the unit attached
        # to that source number; a prior normalization may use a scaled unit.
        fill(d, 'unit_code', 'dose_unit', d.get('unit_raw') or d.get('unit_normalized'))
        fill(d, 'basis_code', 'dose_denominator', d.get('denominator_basis'))

    def phase(p):
        fill(p, 'phase_kind', 'treatment_stage', p.get('phase_label'))
        fill(p, 'route', 'route', p.get('route_raw'))
        fill(p, 'dose_form', 'dosage_form', p.get('dose_form_raw'))
        dose(p.get('dose'))
        schedule = p.get('structured')
        if schedule:
            fill(schedule, 'frequency_pattern', 'frequency_pattern', p.get('frequency_raw'))
            if (schedule.get('frequency_pattern') is None and schedule.get('frequency_count') is not None
                    and schedule.get('interval_value') is not None):
                schedule['frequency_pattern'] = 'intermittent'
                changes.append({'field':'frequency_pattern', 'after':'intermittent', 'reason':'Explicit numerical administrations and interval define intermittent dosing.'})

    for p in data.get('populations', []):
        fill(p, 'sex_eligibility', 'sex_eligibility', p.get('sex_eligibility_raw'))
        fill(p, 'age_unit_code', 'age_unit', p.get('age_unit'))
        fill(p, 'age_min_operator', 'age_min_comparison', p.get('age_min_comparator'))
        fill(p, 'age_max_operator', 'age_max_comparison', p.get('age_max_comparator'))
        if not p.get('disease_modifier_codes'):
            codes = []
            for raw in p.get('modifiers', []):
                if raw and raw.strip().casefold() not in _MISSING:
                    code = canonical_value('disease_modifier', raw) or 'other'
                    if code not in codes:
                        codes.append(code)
            if codes:
                p['disease_modifier_codes'] = codes
                changes.append({'field':'disease_modifier_codes', 'after':codes, 'reason':'Added controlled disease qualifiers; original qualifiers retained.'})
        for c in p.get('eligibility_conditions', []):
            clause(c)
    for part in data.get('study_parts', []):
        fill(part, 'phase', 'study_phase', part.get('phase_raw'))
    for regimen in data.get('regimens', []):
        for component in regimen.get('components', []):
            for p in component.get('dosing_phases', []):
                phase(p)
        for rule in regimen.get('dose_adjustment_rules', []):
            clause(rule.get('condition_structured'))
    for cohort in data.get('cohorts', []):
        dose(cohort.get('dose'))
        for e in cohort.get('exposures', []):
            dose(e.get('dose'))
            for group in e.get('dose_subgroups', []):
                dose(group.get('dose'))
            for p in e.get('dosing_phases', []):
                phase(p)
    for window in data.get('dlt_windows', []):
        fill(window, 'start_anchor', 'window_anchor', window.get('start_raw'))
        fill(window, 'end_anchor', 'window_anchor', window.get('end_raw'))
    for definition in data.get('dlt_definitions', []):
        raw = definition.get('grading_system_raw')
        # A grading version must follow an explicit named grading system, never
        # an arbitrary number in a toxicity paragraph or a publication year.
        match = re.fullmatch(r'\s*(CTCAE|NCI[- ]?CTC|CTC|WHO|RTOG/EORTC)(?:\s+(?:version|v)\.?\s*|\s+)([1-6](?:\.[0-9]+)?)\s*', str(raw or ''), re.I)
        fill(definition, 'grading_system', 'grading_system', match[1] if match else raw)
        if match:
            fill(definition, 'grading_version', 'grading_version', match[2])
        for c in definition.get('criteria_structured', []):
            clause(c)
    for series in data.get('dose_series', []):
        fill(series, 'comparable_unit_code', 'dose_unit', series.get('comparable_unit'))
