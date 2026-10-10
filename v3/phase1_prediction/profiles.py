"""Source-bound comparison views. No design, outcomes or dose magnitudes score."""
from __future__ import annotations
import copy
import hashlib
import json
import re
from .profile_similarity import condition, days, unique, canonical_name
from phase1_extraction.clinical_protocol import normalized_phase, complete_conditions, controlled_text, controlled_list, statement_key

PHASE_FIELDS = {'phase_label':'phase','route_raw':'route','dose_form_raw':'form','frequency_raw':'frequency',
 'administration_days_raw':'administration_days','cycle_length_raw':'cycle','duration_raw':'duration',
 'infusion_duration_raw':'infusion_duration','infusion_rate_raw':'infusion_rate','start_condition_raw':'start','end_condition_raw':'end'}


def dump(v): return v.model_dump(mode='json') if hasattr(v,'model_dump') else dict(v)
def grading_version(v):
    m=re.search(r'(?:version|\bv)\s*([0-9]+(?:\.[0-9]+)*)',str(v or ''),re.I)
    if m is None:
        m=re.fullmatch(r'\s*(?:CTCAE|NCI[- ]?CTC|CTC|WHO|RTOG/EORTC)\s+([0-9]+(?:\.[0-9]+)*)\s*',str(v or ''),re.I)
    return m[1] if m else None

def arr(v): return v if isinstance(v,list) else [v] if v not in (None,'') else []
def unbox(v): return v.get('value') if isinstance(v,dict) and 'value' in v else v


def phase(v):
    d=dump(v); p={b:d.get(a) for a,b in PHASE_FIELDS.items()}
    for field, source in [('route_code','route'),('form_code','dose_form'),('phase_code','phase_kind')]:
        if d.get(source):p[field]=d[source]
    aliases={'qd':'once daily','daily':'once daily','bid':'twice daily','tid':'three times daily','q2w':'every 2 weeks','q3w':'every 3 weeks','q4w':'every 4 weeks'}
    p['frequency']=aliases.get(str(p.get('frequency') or '').lower(),p.get('frequency'))
    if d.get("structured"):p["structured"]=d["structured"]
    return normalized_phase(p)


def adjustment(v):
    """Keep the trigger category and full action statement in one source rule."""
    data=dump(v)
    trigger=copy.deepcopy(data.get('condition_structured') or {})
    trigger['raw_text']='; '.join(filter(None,[data.get('condition_raw'),data.get('action_raw')]))
    trigger['action']=data.get('action')
    return condition(trigger)


def background_signature(exposures, varying_component_id):
    """Keep fixed partner doses; strip only the dose of the actual varying drug."""
    records=[]
    for obj in exposures:
        d=dump(obj)
        r={k:d.get(k) for k in ('name_raw','entity_kind','role','presence','reporting_scope','dose_status','dose','formulation_raw','schedule_raw','window_id')}
        phases=[]
        varying=d.get('component_id')==varying_component_id
        for p in d.get('dosing_phases',[]):
            q={k:v for k,v in p.items() if k not in {'evidence','dosing_phase_id'}}
            if varying:q.pop('dose',None)
            phases.append(q)
        # Raw schedule text can contain a restatement of the varying dose.
        # Prefer structured phases when available to avoid one curve per dose.
        if phases:r.pop('schedule_raw',None)
        r['phases']=unique(phases)
        if varying:
            r.pop('dose',None);r.pop('dose_status',None)
        records.append(r)
    return hashlib.sha256(json.dumps(unique(records),sort_keys=True).encode()).hexdigest()[:20] if records else 'legacy'


def exposure_profile(obj,interventions,components):
    d=dump(obj)
    c=components.get(d.get('component_id'),{})
    i=interventions.get(d.get('intervention_id') or c.get('intervention_id'),{})
    # Accept a documented name followed by a documented alias, while still
    # refusing to inherit metadata after an unrelated name edit.
    names=[i.get('name_raw'),*i.get('aliases',[])]
    known_names={canonical_name(n) for n in names if n}
    raw_name=d.get('name_raw') or ''
    parts=[canonical_name(p) for p in re.split(r'[()（）]',raw_name) if p.strip()]
    recognized=canonical_name(raw_name) in known_names or (len(parts)>1 and all(p in known_names for p in parts))
    if i and not recognized:i={}
    phases=d.get('dosing_phases') or c.get('dosing_phases',[])
    aliases=unique([*i.get('aliases',[]),i.get('name_raw')]) if i else []
    bases={(x.get('structured') or {}).get('dose_basis') for x in phases}
    bases-={None,'unclear'}
    return {'name':d.get('name_raw'),'aliases':aliases,'kind':d.get('entity_kind','drug'),
      'role':d.get('role'),'presence':d.get('presence'),'reporting_scope':d.get('reporting_scope'),
      'classes':arr(i.get('drug_class_raw')),'targets':i.get('targets_raw',[]),'pathways':i.get('pathways_raw',[]),
      'mechanisms':i.get('mechanism_raw',[]),'phases':[phase(p) for p in phases],
      **({'dose_basis':next(iter(bases))} if len(bases)==1 else {}),
      'formulation':d.get('formulation_raw') or i.get('product_or_formulation'),'schedule':d.get('schedule_raw')}


def review_sections(raw):
    chunks=re.split(r'(?m)^\*\*(.+?)\*\*\s*$',raw)
    result={}
    for i in range(1,len(chunks)-1,2):
        title=chunks[i].strip();body=chunks[i+1].strip()
        if body:result.setdefault(title,[]).append(body)
    return result


def _scoped_eligibility_logic(groups, clauses):
    """Keep complete connected group trees; never project a partial AND/OR."""
    nodes={group['group_id']:group for group in groups}
    clause_ids={clause['clause_id'] for clause in clauses if clause.get('clause_id')}
    neighbors={identity:set() for identity in nodes}
    for identity,group in nodes.items():
        for child in group['children']:
            if child in nodes:
                neighbors[identity].add(child)
                neighbors[child].add(identity)
    complete=set()
    visited=set()
    for identity in nodes:
        if identity in visited:continue
        component=set();pending=[identity]
        while pending:
            current=pending.pop()
            if current in component:continue
            component.add(current)
            pending.extend(neighbors[current]-component)
        visited.update(component)
        leaves={child for member in component for child in nodes[member]['children'] if child not in nodes}
        if leaves<=clause_ids:complete.update(component)
    return ([copy.deepcopy(group) for group in groups if group['group_id'] in complete],
            [copy.deepcopy(group) for group in groups if group['group_id'] not in complete])


def extraction_profile(extraction,row=None,overrides=None):
    e=dump(extraction); overrides=overrides or {}
    series={s['series_id']:s for s in e.get('dose_series',[])}
    selected_series=series.get(getattr(row,'series_id',None),{}) if row else {}
    study_id=getattr(row,'study_id',None);part_id=getattr(row,'part_id',None)
    def scoped(x):return row is None or (x.get('study_id')==study_id and x.get('part_id') in (None,part_id))
    pops=[p for p in e.get('populations',[]) if scoped(p)]
    p={'disease':unique([v for p in pops for v in arr(p.get('disease_core') or p.get('disease_raw'))]),
       'modifiers':unique([v for p in pops for v in controlled_list('disease_modifier',p.get('disease_modifier_codes'),p.get('modifiers',[]))]),
       'sex':unique([controlled_text('sex_eligibility', p.get('sex_eligibility'), p.get('sex_eligibility_raw')) for p in pops]),
       'purpose':'; '.join(s.get('purpose_summary') or '' for s in e.get('studies',[]) if not row or s['study_id']==study_id),
       'baseline':unique([v for p in pops for v in p.get('baseline_summary',[])])}
    for bound in ('age_min','age_max'):
        values=set()
        for pop in pops:
            unit=controlled_text('age_unit',pop.get('age_unit_code'),pop.get('age_unit'))
            factor={'year':1.,'years':1.,'month':1/12,'months':1/12,'week':7/365.25,'weeks':7/365.25,'day':1/365.25,'days':1/365.25}.get(str(unit or '').lower())
            if pop.get(bound) is not None and factor is not None:values.add(pop[bound]*factor)
        p[bound]=next(iter(values)) if len(values)==1 else None
        operators={controlled_text('comparison',pop.get(bound+'_operator'),pop.get(bound+'_comparator')) for pop in pops if pop.get(bound) is not None}
        operators.discard(None)
        if len(operators)==1:p[bound+'_operator']=next(iter(operators))
    eligibility={'inclusion':[],'exclusion':[]}
    for pop in pops:
        # Keep AND/OR clause context within its source statement; do not pool
        # eligibility clauses from different populations into a new predicate.
        source_clauses=pop.get('eligibility_conditions',[])
        structured=[c for c in source_clauses if not row or ((not c.get('applies_to_part_ids') or part_id in c['applies_to_part_ids']) and (not c.get('applies_to_arm_ids') or getattr(row,'arm_id',None) in c['applies_to_arm_ids']))]
        raw_sections={key:[v for field in fields for v in pop.get(field,[])] for key,fields in
                      [('inclusion',('inclusion_criteria','prior_treatment_requirements','clinical_requirements')),('exclusion',('exclusion_criteria',))]}
        raw_keys={key:{statement_key(v) for v in values} for key,values in raw_sections.items()}
        destinations=[[key for key,keys in raw_keys.items() if statement_key(c.get('raw_text') or c.get('raw')) in keys] for c in structured]
        for key,raw_values in raw_sections.items():
            applicable=[c for c,destination in zip(structured,destinations) if destination==[key]]
            eligibility[key].extend(complete_conditions(raw_values,applicable,condition))
        # The source schema supplies no inclusion/exclusion discriminator.
        # Preserve unmatched atomic predicates without inferring their polarity.
        unclassified=[condition(c) for c,destination in zip(structured,destinations) if len(destination)!=1]
        if unclassified:eligibility.setdefault('unclassified',[]).extend(unclassified)
        active,unresolved=_scoped_eligibility_logic(pop.get('eligibility_logic_groups',[]),structured)
        eligibility.setdefault('logic_groups',[]).extend(active)
        if unresolved:
            # Retain the full source tree and scoped clause evidence for review,
            # outside active target logic and clinical similarity calculations.
            eligibility.setdefault('unresolved_source_logic',[]).append({
                'population_id':pop.get('population_id'),'logic_groups':unresolved,
                'conditions':copy.deepcopy(source_clauses),
                'reason':'The complete source rule group cannot be resolved using only conditions applicable to this study part and arm.'})
    interventions={i['intervention_id']:i for i in e.get('interventions',[])}
    regimens={r['regimen_id']:r for r in e.get('regimens',[])}
    parts={p['part_id']:p for p in e.get('study_parts',[])}
    configs=[]
    cohorts=e.get('cohorts',[])
    if row:
        cohort=next((c for c in cohorts if c['cohort_id']==getattr(row,'cohort_id',None)),{})
        cohorts=[{**cohort,'series_id':getattr(row,'series_id',None),'regimen_id':getattr(row,'regimen_id',None),
                  'exposures':[dump(x) for x in getattr(row,'exposures',[])] or cohort.get('exposures',[])}]
    for c in cohorts:
        s=series.get(c.get('series_id'),{});r=regimens.get(c.get('regimen_id'),{});part=parts.get(s.get('part_id'),{})
        components={x['component_id']:x for x in r.get('components',[])}
        exposures=c.get('exposures',[])
        if not exposures:
            exposures=[{'name_raw':interventions.get(x['intervention_id'],{}).get('name_raw'),'intervention_id':x['intervention_id'],
              'component_id':x['component_id'],'role':x['role'],'presence':'unknown'} for x in components.values()]
        cfg={'phase':controlled_text('study_phase',part.get('phase'),part.get('phase_raw')),'role':c.get('cohort_role') or part.get('role'),
             'status':c.get('actual_status'),'drugs':unique([exposure_profile(x,interventions,components) for x in exposures]),
             'adjustments':[adjustment(a) for a in r.get('dose_adjustment_rules',[])]}
        for exp,d in zip(exposures,[exposure_profile(x,interventions,components) for x in exposures]):
            d['adjustments']=[adjustment(a) for a in r.get('dose_adjustment_rules',[]) if exp.get('component_id') in a.get('applies_to_component_ids',[])]
            cfg['drugs']=[x for x in cfg['drugs'] if not (x['name']==d['name'] and x['role']==d['role'])]+[d]
        cfg['adjustments']=[adjustment(a) for a in r.get('dose_adjustment_rules',[]) if not a.get('applies_to_component_ids')]
        cfg['drugs']=unique(cfg['drugs'])
        # Cohort labels often contain doses/design names; never send to scorer.
        if cfg not in configs:configs.append(cfg)
    definitions=[d for d in e.get('dlt_definitions',[]) if scoped(d)]
    dlt={'criteria':[c for d in definitions for c in complete_conditions(d.get('criteria_raw',[]), d.get('criteria_structured',[]), condition)],
      'exceptions':unique([condition(c) for d in definitions for c in d.get('exceptions_raw',[])]),
      'grading':'; '.join(dict.fromkeys(filter(None,(controlled_text('grading_system',d.get('grading_system'),d.get('grading_system_raw')) for d in definitions)))),
      'version':'; '.join(dict.fromkeys(d['version_label'] for d in definitions if d.get('version_label'))),
      'observed':unique([t for o in e.get('dlt_observations',[]) if not row or o.get('cohort_id')==getattr(row,'cohort_id',None) for t in o.get('toxicities_raw',[])])}
    dlt['logic_groups']=[g for d in definitions for g in d.get('criteria_logic_groups',[])]
    eligibility.setdefault('logic_groups',[])
    versions={d.get('grading_version') if d.get('grading_version') not in (None,'other') else grading_version(d.get('grading_system_raw')) for d in definitions}
    versions.discard(None)
    dlt['grading_version']=next(iter(versions)) if len(versions)==1 else grading_version(dlt.get('grading'))
    windows=[{'kind':w.get('window_kind','unclear'),'start':w.get('start_raw'),'end':w.get('end_raw'),
      **({'start_anchor':w['start_anchor']} if w.get('start_anchor') else {}),
      **({'end_anchor':w['end_anchor']} if w.get('end_anchor') else {}),
      'days':w.get('normalized_days'),'cycles':w.get('cycle_numbers_raw'),'raw':w.get('raw_definition'),'basis':w.get('basis')}
      for w in e.get('dlt_windows',[]) if (not row or w['window_id']==getattr(row,'window_id',None))]
    profile={'population':p,'eligibility':eligibility,'configurations':unique(configs),'dlt':dlt,'windows':unique(windows)}
    # Human prose corrections replace affected values instead of silently scoring
    # contradicted structured data. Design reviews have no path into this view.
    if 'population' in overrides:
        sections=review_sections(overrides['population'])
        profile['population']={'disease':sections.get('Disease',[]),'modifiers':sections.get('Disease qualifiers',[]),
            'purpose':'; '.join(sections.get('Study purpose',[])), 'baseline':sections.get('Enrolled population and baseline',[])}
    if 'eligibility' in overrides:
        sections=review_sections(overrides['eligibility'])
        profile['eligibility']={'inclusion':[condition(v) for k in ('Inclusion criteria','Clinical and organ function requirements','Prior therapy and washout requirements') for v in sections.get(k,[])],
                                'exclusion':[condition(v) for v in sections.get('Exclusion criteria',[])]}
        # Age/sex were reviewed within the eligibility module too.
        for k in ('age_min','age_max','sex'):profile['population'].pop(k,None)
        if not sections:profile['eligibility']={'inclusion':[condition(overrides['eligibility'])],'exclusion':[]}
    if 'dlt_definition' in overrides:
        sections=review_sections(overrides['dlt_definition'])
        # Only explicitly labelled protocol fields enter pretrial similarity.
        # An unstructured combined review may include observed outcomes.
        profile['dlt']={'criteria':[condition(v) for v in sections.get('DLT criteria',[])],
            'exceptions':[condition(v) for v in sections.get('DLT exceptions',[])],
            'grading':'; '.join(sections.get('Toxicity grading system',[])),
            'observed':[v for k,values in sections.items() if k.startswith('Observed DLTs') for v in values]}
        profile['dlt']['grading_version']=grading_version(profile['dlt']['grading'])
    if 'dlt_window' in overrides:profile['windows']=[{'raw':overrides['dlt_window'],'kind':'unclear'}]
    if 'intervention' in overrides:
        for c in configs:
            for d in c['drugs']:
                for k in ('classes','targets','pathways','mechanisms'):d[k]=[]
        profile['configurations']=configs
    if 'dosing' in overrides:
        for c in profile['configurations']:
            for d in c['drugs']:d['phases']=[];d['schedule']=None;d['formulation']=None
    if 'parts_arms' in overrides:
        for c in profile['configurations']:c['phase']=None;c['role']=None
    if overrides:profile['review_notes']={k:v for k,v in overrides.items() if k!='design'}
    return profile


def query_profile(query):
    if '_comparison_profile' in query:return copy.deepcopy(query['_comparison_profile'])
    disease=unbox(query.get('cancer_type')) or []
    age=copy.deepcopy(unbox(query.get('Age')) or {})
    unit=str(age.get('age_unit') or 'years').lower()
    for bound in ('age_min','age_max'):
        if age.get(bound) is not None:
            age[bound]=age[bound]/12 if unit in {'month','months'} else age[bound] if unit in {'year','years'} else None
    window=unbox(query.get('DLT_EVA')) or {}
    drugs=[]
    levels=query.get('dose_levels') or [{}];schedule=levels[0].get('Schedule',{})
    frequency=levels[0].get('Frequency',{})
    for index, name in enumerate(arr(query.get('Drug_Name') or query.get('toxicity_curve_drug'))):
        drugs.append({'name':name,'kind':'drug','role':'escalated' if index == 0 else 'combination_partner','presence':'present',
          'research_priority': index + 1, 'research_role': 'primary' if index == 0 else 'partner',
          'targets':arr(query.get('Drug_target_gene_target')) if index == 0 else [],'mechanisms':arr(query.get('Mechanism_of_action')) if index == 0 else [],
          'phases':[{'phase':'treatment','route':'; '.join(arr(query.get('Admin_Route'))),
          'form':'; '.join(arr(query.get('Dose_Form'))),'frequency':frequency.get('frequency_standard'),
          'cycle':schedule.get('cycle_length_days'),'administration_days':schedule.get('administration_days')}]})
    criteria=[]
    for c in arr(query.get('DLT_definition')):
        if isinstance(c,dict):
            raw=c.get('DLT_definition_comments') or c.get('DLT_Name','')
            grade=c.get('DLT_definition_Grade') or {}
            if grade.get('grade_value') is not None and not re.search(r'grade\s*[>=≤≥]*\s*\d',raw,re.I):raw+=f" grade {grade.get('operator') or '='} {grade['grade_value']}"
            criteria.append(condition(raw))
        else:criteria.append(condition(c))
    return {'population':{'disease':[x.get('core_term') or x.get('mention') if isinstance(x,dict) else x for x in arr(disease)],
        'sex':arr(query.get('Gender_Sex')),'age_min':age.get('age_min'),'age_max':age.get('age_max')},
      'eligibility':{'inclusion':[condition(x) for x in arr(query.get('Trial_Inclusion'))], 'exclusion':[condition(x) for x in arr(query.get('Trial_Exclusion'))]},
      'configurations':[{'status':'planned','drugs':drugs}], 'dlt':{'criteria':criteria},
      'windows':[{'kind':query.get('dlt_window_kind','formal_dlt'),'days':window.get('dlt_eval_duration_days'),
          'raw':window.get('dlt_eval_text'),'cycles':window.get('dlt_eval_cycle_number'),
          'start':query.get('dlt_window_start'),'end':query.get('dlt_window_end')}]}
