"""Validated native target exposure configurations, independent of web UI."""
import copy
import math
import re
from .profiles import query_profile
from .profile_similarity import condition,unique

ROLES=['escalated','fixed_backbone','combination_partner','conditioning','supportive','concomitant','radiation','maintenance','other','unclear']


def parse_rows(rows,query,selected):
    base=query_profile(query);primary=query.get('toxicity_curve_drug') or query.get('Drug_Name',[None])[0]
    configs=[];native=[];seen={}
    for r in rows:
        if not str(r.get('drug') or '').strip():continue
        scenario=str(r.get('scenario') or 'Main').strip();name=str(r['drug']).strip()
        kind=r.get('kind') or 'drug';role=r.get('role') or 'unclear';presence=r.get('presence') or 'unknown'
        if kind not in {'drug','drug_class','non_drug','unclear'} or role not in ROLES or presence not in {'present','absent','unknown','mixed'}:raise ValueError('Invalid drug type, role or presence.')
        raw=str(r.get('doses') or '').strip();values=[]
        if raw:
            try:values=[float(v) for v in re.split(r'[,;\s]+',raw) if v]
            except ValueError:raise ValueError(f'{name}: enter numeric doses separated by commas.')
            if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError(f'{name}: doses must be finite and positive.')
        if presence=='absent' and values:raise ValueError(f'{name}: an absent drug cannot have a dose. Clear the dose.')
        if values and presence!='present':raise ValueError(f'{name}: a specified dose requires present use.')
        if values and not r.get('unit'):raise ValueError(f'{name}: supply the dose unit.')
        if name==primary and presence!='present':raise ValueError('The primary study drug must be present.')
        cycle=r.get('cycle_days');cycle=None if cycle in ('',None) else float(cycle)
        if cycle is not None and (not math.isfinite(cycle) or cycle<=0):raise ValueError('Cycle days must be positive or blank.')
        phase={k:r.get(k) or None for k in ('phase','route','form','frequency','administration_days','duration','infusion_duration','infusion_rate','start','end')};phase['cycle']=cycle
        key=(scenario,name)
        if key in seen:
            d=seen[key]
            if (d['kind'],d['role'],d['presence'])!=(kind,role,presence):raise ValueError(f'{name}: phase rows disagree on identity, role or use.')
            d['phases'].append(phase)
        else:
            d={'name':name,'kind':kind,'role':role,'presence':presence,'reporting_scope':'cohort', 'phases':[phase]}
            for k in ('aliases','classes','targets','mechanisms'):d[k]=[v.strip() for v in str(r.get(k) or '').split(';') if v.strip()]
            d['adjustments']=[condition(v) for v in str(r.get('adjustment_conditions') or '').splitlines() if v.strip()]
            seen[key]=d
        native.append({'scenario':scenario,'name':name,'kind':kind,'role':role,'presence':presence,'phase':r.get('phase'), 'doses':sorted(set(values)),'unit':r.get('unit') or None})
    scenarios=sorted({k[0] for k in seen})
    for scenario in scenarios:
        drugs=[d for (s,n),d in seen.items() if s==scenario]
        if not any(d['name']==primary for d in drugs):raise ValueError(f'{scenario}: include a row for the curve-axis drug {primary}.')
        configs.append({'label':scenario,'status':'planned','phase':query.get('_target_phase') or None,'role':query.get('_target_role') or None,'drugs':unique(drugs)})
    if not configs:return query
    selected=selected if selected in scenarios else scenarios[0]
    chosen=next(c for c in configs if c['label']==selected)
    # Blank primary cells inherit the corresponding primary-only legacy inputs.
    fallback=(base.get('configurations') or [{'drugs':[]}])[0].get('drugs',[])
    if fallback:
        primary_fallback=fallback[0]
        for d in chosen['drugs']:
            if d['name']!=primary:continue
            for k in ('targets','mechanisms'):
                if not d[k]:d[k]=primary_fallback.get(k,[])
            if len(d['phases'])==1:
                for k,v in (primary_fallback.get('phases') or [{}])[0].items():
                    if not d['phases'][0].get(k):d['phases'][0][k]=v
    base['configurations']=[chosen]
    result=copy.deepcopy(query);result['_comparison_profile']=base
    result['_target_configurations']=configs;result['_target_exposures']=native;result['_selected_scenario']=selected
    present=[d for d in chosen['drugs'] if d['presence']=='present' and d['role'] not in {'supportive','concomitant','conditioning','other','unclear'}]
    result['Trial_Type']={'value':'combination' if len(present)>1 else 'single','evidence':'derived from planned antitumor drugs'}
    primary_doses={tuple(x['doses']) for x in native if x['scenario']==selected and x['name']==primary and x['doses']}
    if len(primary_doses)>1:raise ValueError('Multiple phases specify different primary dose grids; select a single curve-axis dose grid before prediction.')
    if primary_doses:result['planned_doses']=list(next(iter(primary_doses)))
    return result
