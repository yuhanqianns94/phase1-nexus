"""Target-only exploration, Phase I eligibility and relevance budgets."""
import copy
import math
import re
from itertools import product
from .contracts import TargetRegimen


def phase_status(profile):
    values = [str(c.get('phase') or '').strip().lower() for c in profile.get('configurations', [])]
    if not values or not any(values):
        return 'unknown'
    def classify(v):
        v = re.sub(r'\bphase\s*', '', v).strip()
        v = re.sub(r'\s+(?:dose[- ]?)?(?:expansion|escalation)(?:\s+.*)?$', '', v)
        if v in {'i', 'ia', 'ib', '1', '1a', '1b', 'phase i component of i/ii', 'i component of i/ii'}:
            return 'phase_i'
        if re.fullmatch(r'(?:i|1)\s*[/–-]\s*(?:ii|2)[ab]?', v):
            return 'mixed_requires_separation'
        if v in {'ii','2','iia','iib','iii','3','iv','4'}:
            return 'out_of_scope'
        return 'unknown'
    statuses = {classify(v) for v in values}
    return 'phase_i' if statuses == {'phase_i'} else ('out_of_scope' if 'out_of_scope' in statuses else 'mixed_requires_separation' if 'mixed_requires_separation' in statuses else 'unknown')


def relevance_multiplier(document_id, decisions=None):
    if decisions is None:
        return 1.
    value = decisions.get(document_id, 'Not reviewed')
    if value not in {'Include','Exclude','Uncertain','Not reviewed'}:
        raise ValueError('Unknown relevance decision: ' + str(value))
    if value == 'Not reviewed':
        raise ValueError('Complete relevance review for ' + document_id)
    return {'Include':1.,'Exclude':0.,'Uncertain':.5}[value]


def article_budgets(rows, decisions=None):
    grouped = {}
    for row in rows:
        doc=row['document_id']; b=row.get('similarity') or 0.
        if doc not in grouped or b > grouped[doc]['base_weight']:
            grouped[doc]={'document_id':doc,'similarity':row.get('similarity'),'base_weight':b}
    for doc,r in grouped.items():
        r['decision']=(decisions or {}).get(doc,'Automatic')
        r['review_multiplier']=relevance_multiplier(doc,decisions)
        r['adjusted_weight']=r['base_weight']*r['review_multiplier']
    total=sum(r['adjusted_weight'] for r in grouped.values())
    for r in grouped.values():r['normalized_weight']=r['adjusted_weight']/total if total else 0.
    return list(grouped.values())


def exploration_target(draft, coordinates, axes, *, relative=True):
    """Make a computational grid without mutating the saved clinical plan.

    Each coordinate is an explicit list; None means present/dose unknown, 0 absent.
    Relative numeric references are 1 and never represent a clinical dose.
    """
    data=copy.deepcopy(draft)
    if not data.get('drugs'):raise ValueError('Add at least one drug first.')
    ids=[d['drug_id'] for d in data['drugs']]
    if set(coordinates)!=set(ids) or not set(axes)<=set(ids) or not 1<=len(axes)<=2:
        raise ValueError('Choose one or two numeric axes and specify every companion.')
    for d in data['drugs']:
        values=coordinates[d['drug_id']]
        if not values:raise ValueError('Every drug needs a coordinate or explicit use state.')
        if any(v is not None and (isinstance(v,bool) or not math.isfinite(v) or v<0) for v in values):raise ValueError('Coordinates must be finite nonnegative numbers.')
        if d['drug_id'] not in axes and len(values)!=1:raise ValueError('Fix all non-axis drugs.')
        positive=[v for v in values if v is not None and v>0]
        if d['drug_id'] in axes and (len(values)<2 or len(positive)!=len(values)):
            raise ValueError('An axis needs at least two positive numeric coordinates.')
        d['mode']='numeric' if positive else 'background'
        if positive:
            if relative:
                d.update(unit='relative dose',reference_dose=1.,transport_mode='relative_dose_analogy')
            elif not d.get('unit') or not d.get('reference_dose'):
                raise ValueError(d['name']+': enter a unit and computational reference for actual-dose exploration.')
    grid=list(product(*(coordinates[k] for k in ids)))
    if len(grid)>1600:raise ValueError('Limit the exploration grid to 1600 points.')
    data['points']=[{'level_id':f'G{i+1}','doses':dict(zip(ids,values))} for i,values in enumerate(grid)]
    data['path_level_ids']=[]
    planning=data.setdefault('profile',{}).setdefault('planning',{})
    planning['exposure_states']={}
    planning['exploration']={'relative':relative,'axes':list(axes),'auto_grid':True}
    return TargetRegimen.model_validate(data)


def validate_boin_target(target):
    data=target.model_dump() if hasattr(target,'model_dump') else target
    if data.get('profile',{}).get('planning',{}).get('exploration',{}).get('relative'):
        raise ValueError('BOIN requires actual candidate regimens; relative exploration coordinates are not doses.')
    if any(d.get('mode')=='numeric' and (not d.get('unit') or d.get('unit')=='relative dose') for d in data['drugs']):
        raise ValueError('Enter actual dose units for BOIN.')
    if len(data.get('path_level_ids',[]))<2:raise ValueError('Choose at least two distinct regimens in an explicitly ordered BOIN path.')
    by_id={p['level_id']:p for p in data['points']}
    signatures=[tuple((d['drug_id'],by_id[level]['doses'][d['drug_id']]) for d in data['drugs'])
                for level in data['path_level_ids']]
    if len(set(signatures)) != len(signatures):
        raise ValueError('Choose distinct regimens in the BOIN path; different labels for identical doses are not separate levels.')


def coordinate_support(entries,target,method):
    """Conservative coordinate diagnostics, not an identifiability test."""
    from .mapping import mapped_relative_dose
    byfit={m['source_fit_id']:s for s,a,m in entries}
    mappings={c['source_fit_id']:c['mapping']['pairs'] for c in method.get('contributors',[]) if 'mapping' in c}
    if method.get('evidence_construction') == 'component_assembly':
        # Only the usable components enter support checks. The preview mapping
        # may also contain a component rejected by the dose-coordinate rules.
        mappings={c['source_fit_id']:{w['target_id']:w['source_id'] for w in c['component_weights']}
                  for c in method.get('contributors',[])}
    result=[]
    for point in target.points:
        outside=missing=0
        for fit,pairs in mappings.items():
            scope=byfit[fit]
            for td in target.drugs:
                sd=next((d for d in scope.drugs if d.model_drug_id==pairs.get(td.drug_id)),None)
                if sd is None:continue
                values=[]
                for o in scope.observations:
                    for e in o.exposures:
                        if e.drug_id==sd.model_drug_id and e.presence=='present' and e.dose is not None:
                            values.append(e.dose/(sd.reference_dose or 1.))
                r=mapped_relative_dose(td,sd,point.doses[td.drug_id])
                if not values:missing+=1
                elif r<min(values)-1e-9 or r>max(values)+1e-9:outside+=1
        status='Outside at least one source dose range' if outside else 'Source dose information incomplete' if missing else 'Within marginal dose ranges; joint support not established'
        if not mappings:status='Component assembly: model-derived cross-article combination'
        elif method.get('evidence_construction') == 'component_assembly':
            status += '; assembled combination is model-derived'
        result.append({'level_id':point.level_id,'support':status,'outside_source_axes':outside,'missing_source_axes':missing})
    return result
