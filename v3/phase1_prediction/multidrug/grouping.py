"""Conservative within-article grouping; exposure values are model inputs, not arms.

Free-text population differences require a source-bound compatibility declaration.
Declarations never bypass regimen, part, window, eligibility or schedule checks.
No drug-name-specific branch or human accuracy-review data is used.
"""
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from phase1_extraction.dose_units import dose_unit_for_value
from phase1_extraction.drug_ordering import entity_key, ordered_catalog
from phase1_prediction.profiles import extraction_profile
from phase1_prediction.profile_similarity import known
from .contracts import fingerprint

POLICY_PATH=Path(__file__).with_name('scope_compatibility.json')


def _norm(value):
    return ' '.join(str(value or '').casefold().split())


def _declarations(extraction, source_sha256):
    payload=json.loads(POLICY_PATH.read_text()) if POLICY_PATH.exists() else {'groups':[]}
    result={};notices=[]
    for group in payload['groups']:
        if group['document_id']!=extraction.document_id:continue
        if group['source_sha256']!=source_sha256:
            notices.append({'document_id':extraction.document_id,'reason':'scope_compatibility_source_changed',
                            'detail':'Stored population compatibility no longer matches this extraction; arms remain separate.'})
            continue
        for arm in group['arm_ids']:
            result[arm]=group
    return result,notices


def _clinical_context(extraction,row,profile):
    dlt=copy.deepcopy(profile.get('dlt',{}));dlt.pop('observed',None)
    population=copy.deepcopy(profile.get('population',{}));population.pop('baseline',None)
    config=(profile.get('configurations') or [{}])[0]
    return {'population':population,'eligibility':profile.get('eligibility',{}),'dlt':dlt,
            'windows':profile.get('windows',[]),'phase':config.get('phase'),'cohort_role':config.get('role'),
            'status':config.get('status')}


def _schedule(extraction,row,exposures):
    regimen=next((r for r in extraction.regimens if r.regimen_id==row.regimen_id),None)
    components={c.component_id:c for c in regimen.components} if regimen else {}
    result={}
    for e in exposures:
        if e.presence=='absent':continue # No administration schedule exists for an unadministered drug.
        component=components.get(e.component_id)
        phases=e.dosing_phases or (component.dosing_phases if component else [])
        phases=[{k:v for k,v in p.model_dump(mode='json').items() if k not in {'dose','dosing_phase_id','evidence'}} for p in phases]
        # Prefer the full structured schedule to a redundant prose restatement.
        item={'phases':phases,'schedule':None if phases else _norm(e.schedule_raw),
              'formulation':_norm(e.formulation_raw),'window_id':e.window_id or row.window_id}
        unit=_norm(dose_unit_for_value(e.dose)).replace(' ','').replace('²','2').replace('^2','2').replace('µ','u').replace('μ','u')
        item['unit']=re.sub(r'^(?:mcg|ug|mg|g)(?=/|$)','mg',unit) or None
        if e.dose and e.dose.value_kind!='scalar' and e.dose_semantics not in {'alternative_unknown','subgroups'}:
            item['unresolved_coordinate']=e.dose.model_dump(mode='json')
        result[entity_key(e)]=item
    represented={entity_key(e) for e in exposures}
    for drug in ordered_catalog(extraction,row.regimen_id):
        if drug['model_drug_id'] not in represented:
            result[drug['model_drug_id']]={'unreported_exposure':True,'unit':None}
    return result


def _compatible_schedule(a,b):
    for drug in a.keys() & b.keys():
        x,y=a[drug],b[drug]
        if {k:v for k,v in x.items() if k!='unit'}!={k:v for k,v in y.items() if k!='unit'}:
            return False
        if x.get('unit') and y.get('unit') and x['unit']!=y['unit']:
            return False
    return True


def group_rows(extraction,rows,*,source_sha256,row_exposures,legacy_key):
    declarations,notices=_declarations(extraction,source_sha256)
    arms={a.arm_id:a for a in extraction.arms}
    base=defaultdict(list)
    for row in rows:
        arm=arms.get(row.arm_id)
        declaration=declarations.get(row.arm_id)
        if declaration:
            population_key=('declared_compatible',declaration['group_id'])
        elif arm and known(arm.population_description):
            population_key=('same_reported_population',_norm(arm.population_description))
        else:
            population_key=('unresolved_population',row.arm_id)
        profile=extraction_profile(extraction,row)
        clinical=_clinical_context(extraction,row,profile)
        key=(row.study_id,row.part_id,row.regimen_id,row.window_id,population_key,fingerprint(clinical))
        base[key].append((row,_schedule(extraction,row,row_exposures(row,extraction))))
    clusters=[]
    for base_key,records in sorted(base.items(),key=lambda item:str(item[0])):
        # Most specified rows first. A missing/absent drug may not bridge two
        # conflicting active schedules; an ambiguous row gets its own scope.
        local=[]
        for row,schedule in sorted(records,key=lambda item:(
                -len(item[1]), -sum(bool(v.get('unit')) for v in item[1].values()),
                fingerprint(item[1]),item[0].cohort_id,item[0].curve_row_id)):
            compatible=[c for c in local if _compatible_schedule(c['schedule'],schedule)]
            if len(compatible)==1:
                c=compatible[0];c['rows'].append(row)
                for drug,value in schedule.items():
                    existing=c['schedule'].get(drug,{})
                    c['schedule'][drug]={**value,'unit':value.get('unit') or existing.get('unit')}
            else:
                local.append({'rows':[row],'schedule':dict(schedule),'base':base_key})
        clusters.extend(local)
    old_keys=[{legacy_key(row,extraction) for row in c['rows']} for c in clusters]
    use_counts=Counter(key for keys in old_keys for key in keys)
    result=[]
    for c,old in zip(clusters,old_keys):
        members=sorted(c['rows'],key=lambda r:(r.cohort_id,r.curve_row_id))
        arm_ids=sorted({r.arm_id for r in members},key=str)
        audit={}
        if len(old)==1 and use_counts[next(iter(old))]==1:
            key=next(iter(old))
        else:
            first=members[0]
            arm_key=arm_ids[0] if len(arm_ids)==1 else 'merged_'+fingerprint(arm_ids)[:12]
            key=(first.study_id,first.part_id,arm_key,first.regimen_id,first.window_id,
                 fingerprint({'grouping':'compatible_scopes_v1','context':c['base'],'schedule':c['schedule']}))
            sources=[{'arm_id':a,'label':arms[a].label,'population_description':arms[a].population_description}
                     for a in arm_ids if a in arms]
            audit={'grouping_version':'compatible_scopes_v1','arm_ids':arm_ids,'source_arms':sources,
                   'population_basis':list(c['base'][4]),
                   'compatibility_declarations':list({declarations[a]['group_id']:declarations[a] for a in arm_ids if a in declarations}.values()),
                   'prior_scope_keys':[list(k) for k in sorted(old,key=str)]}
        result.append((key,members,audit))
    return sorted(result,key=lambda x:str(x[0])),notices
