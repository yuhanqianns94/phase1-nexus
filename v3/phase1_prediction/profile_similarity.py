"""Seven-category, symmetric comparisons. Design and dose magnitudes never score.

Profiles are separate from extraction and statistical evidence. Unknown is not
absence. Matching is one-to-one; all operations are deterministic and outcome-
free in pretrial mode. A comparison scores only the available, comparable clinical facts.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
from collections import Counter
from contextvars import ContextVar
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment

VERSION = 'cohort_profile_similarity_v11'
DEFAULT_MIN_DRUG_SIMILARITY = 0.20
LABELS = {'population':'Disease and population', 'eligibility':'Eligibility',
          'drugs':'Drugs and mechanism', 'administration':'Administration',
          'structure':'Study parts and groups', 'dlt':'DLT definition and manifestations',
          'window':'DLT assessment window'}
# Preserve the previous six effective indicators' category-level allocation.
DEFAULT_WEIGHTS = {'population':1/6, 'eligibility':1/6, 'drugs':1/6,
                   'administration':1/6, 'structure':0., 'dlt':1/6, 'window':1/6}
LEGACY_GROUPS = {'population':('cancer_type','age','gender'), 'eligibility':('eligibility_embedding',),
                 'drugs':('drug','target_moa'), 'administration':('route','schedule','dose_form'),
                 'structure':('trial_type','schema_embedding'), 'dlt':('dlt_definition',), 'window':('dlt_eva',)}
MISSING = {'', 'unknown', 'unclear', 'not reported', 'not_reported', 'not specified', 'not_specified', '(blank)', 'undecided', 'not entered', 'not specified'}
_TEXT_VECTORS = ContextVar('profile_text_vectors', default=None)


def norm(v):
    if v is None: return ''
    if isinstance(v, dict): return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return ' '.join(str(v).casefold().replace('−','-').split())


def known(v):
    if isinstance(v, (list,dict,tuple)): return bool(v)
    return v is not None and norm(v) not in MISSING


def stable(v): return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',',':'))


def unique(values):
    return [json.loads(x) for x in sorted({stable(v) for v in values if known(v)})]


def normalize_weights(weights=None):
    if weights is None: return dict(DEFAULT_WEIGHTS)
    weights = dict(weights)
    if not set(weights) <= set(LABELS):
        allowed = {k for keys in LEGACY_GROUPS.values() for k in keys}
        if not set(weights) <= allowed: raise ValueError('Unknown similarity weight field (design is not a similarity indicator).')
        weights = {group:sum(weights.get(k,0) for k in keys) for group,keys in LEGACY_GROUPS.items()}
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('Similarity weights must be finite nonnegative numbers.')
    # Study labels remain available for review, never for borrowing strength.
    # Apply this to saved/custom settings as well as the default UI.
    weights['structure'] = 0.
    total = sum(weights.values())
    if total <= 0: raise ValueError('At least one clinical similarity weight must be positive; study labels cannot receive borrowing weight.')
    return {k:weights.get(k,0)/total for k in LABELS}


def tokens(v): return re.findall(r'[\w]+|>=|<=|>|<|=', norm(v).replace('≥','>=').replace('≤','<='))


def embedding_vector(value):
    """Reject malformed vectors instead of turning NaN into a perfect score."""
    try:vector=np.asarray(value,dtype=float)
    except (TypeError,ValueError) as error:raise ValueError('Embeddings must contain finite numeric vectors; no local fallback was used.') from error
    if vector.ndim!=1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError('Embeddings must contain nonempty, finite numeric vectors; no local fallback was used.')
    length=np.linalg.norm(vector)
    if not math.isfinite(float(length)) or length<=0:
        raise ValueError('Embeddings must have a finite, positive norm; no local fallback was used.')
    return vector


def text_score(a,b):
    if not known(a) or not known(b): return None
    if norm(a)==norm(b): return 1.
    vectors = _TEXT_VECTORS.get()
    if hasattr(vectors, 'ensure'):vectors.ensure([a,b])
    if vectors is not None and norm(a) in vectors and norm(b) in vectors:
        x,y=embedding_vector(vectors[norm(a)]),embedding_vector(vectors[norm(b)])
        if x.shape!=y.shape:raise ValueError('Embedding dimensions do not match; no local fallback was used.')
        return max(0.,min(1.,float((x/np.linalg.norm(x))@(y/np.linalg.norm(y)))))
    def features(v):
        t=tokens(v); return Counter(t + [' '.join(pair) for pair in zip(t,t[1:])])
    x,y=features(a),features(b)
    den=math.sqrt(sum(v*v for v in x.values())*sum(v*v for v in y.values()))
    return sum(v*y.get(k,0) for k,v in x.items())/den if den else None


def categorical(a,b):
    return float(norm(a)==norm(b)) if known(a) and known(b) else None


def numeric(a,b,scale):
    if a is None or b is None: return None
    try:
        a,b=float(a),float(b)
        return math.exp(-abs(a-b)/scale) if math.isfinite(a) and math.isfinite(b) else None
    except (TypeError,ValueError): return None


def item(label,a,b,score,*,applicable=True,note=''):
    return {'indicator':label,'left':a,'right':b,'score':score,'applicable':applicable,'note':note}


def combine(details):
    applicable=[d for d in details if d.get('applicable',True)]
    comparable=[d for d in applicable if d.get('score') is not None]
    return {'applicable':bool(applicable),'score':sum(d['score'] for d in comparable)/len(comparable) if comparable else None,
            'details':details}


def match(left,right,affinity):
    """Stable one-to-one assignment; no best-right reuse or order dependence."""
    left,right=unique(left),unique(right)
    if affinity is condition_affinity:
        # Equivalent structured codes are annotations after normalization.
        # Use the clinical predicate, not those annotations, to order ties.
        left,right=(sorted(rows,key=lambda c:stable(condition_signature(c,include_text=True)))
                    for rows in (left,right))
    # Canonical orientation makes ties identical after exchanging the articles.
    flip=stable(left)>stable(right)
    a,b=(right,left) if flip else (left,right)
    pairs=[]
    if a and b:
        matrix=np.array([[float(affinity(x,y) or 0) for y in b] for x in a])
        if affinity is drug_affinity:
            # Lexicographic identity priority scales with assignment size.
            matrix += (min(len(a), len(b)) + 1) * np.array([[drug_identity(x,y) == 1 for y in b] for x in a])
        if affinity is phase_affinity:
            matrix += (min(len(a), len(b)) + 1) * np.array([[categorical(x.get('phase'),y.get('phase')) == 1 for y in b] for x in a])
        if affinity is condition_affinity:
            matrix += (min(len(a), len(b)) + 1) * np.array([[categorical(x.get('topic'),y.get('topic')) == 1 for y in b] for x in a])
        if affinity in (drug_affinity, condition_affinity):
            allowed=np.array([[drug_correspondence(x,y)['eligible'] if affinity is drug_affinity else condition_affinity(x,y)>0 for y in b] for x in a])
            matrix=np.concatenate([np.where(allowed,matrix+1e-9,-1e9),np.zeros((len(a),len(a)))],axis=1)
        ri,ci=linear_sum_assignment(matrix,maximize=True)
        matched=[(i,j) for i,j in zip(ri,ci) if j<len(b) and matrix[i,j]>=0]
        pairs=[(a[i],b[j]) for i,j in matched]
        pairs += [(x,None) for i,x in enumerate(a) if i not in {i for i,j in matched}]
        pairs += [(None,y) for j,y in enumerate(b) if j not in {j for i,j in matched}]
    else: pairs=[(x,None) for x in a]+[(None,y) for y in b]
    return [(y,x) for x,y in pairs] if flip else pairs


def set_score(a,b):
    if not known(a) or not known(b): return None
    x,y={norm(v) for v in a if known(v)},{norm(v) for v in b if known(v)}
    return len(x&y)/len(x|y) if x and y else None


def list_text(label,a,b):
    pairs=match(a or [],b or [],text_score)
    return combine([item(label,x,y,text_score(x,y),note='Unmatched is unreported, not explicit absence.' if x is None or y is None else '') for x,y in pairs] or [item(label,a,b,None)])


def canonical_name(v):
    # Punctuation variants only. Drug/class equivalence needs explicit aliases.
    return re.sub(r'[^\w]', '',norm(v))


def drug_identity(a,b):
    na={canonical_name(x) for x in [a.get('name'),*(a.get('aliases') or [])] if known(x)}
    nb={canonical_name(x) for x in [b.get('name'),*(b.get('aliases') or [])] if known(x)}
    if not na or not nb: return None
    return float(bool(na&nb) and a.get('kind','drug')==b.get('kind','drug'))


def drug_affinity(a,b):
    fields = [(drug_identity(a,b), .15)]
    for key, weight in [('mechanisms', .25), ('targets', .25), ('classes', .15), ('pathways', .05)]:
        fields.append((list_text(key, a.get(key), b.get(key))['score'], weight))
    available = [(score, weight) for score, weight in fields if score is not None]
    return sum(score * weight for score, weight in available) / sum(weight for _, weight in available) if available else 0.


def drug_correspondence(a,b,minimum=DEFAULT_MIN_DRUG_SIMILARITY):
    """Drug evidence is required independently of role, disease and schedule."""
    if drug_identity(a,b)==1:
        return {'eligible':True,'score':1.,'basis':'explicit_identity','minimum':minimum}
    available=[]
    for key,weight in [('mechanisms',.25),('targets',.25),('classes',.15),('pathways',.05)]:
        score=list_text(key,a.get(key),b.get(key))['score']
        if score is not None:available.append((score,weight))
    score=sum(s*w for s,w in available)/sum(w for _,w in available) if available else None
    return {'eligible':score is not None and score>0 and score>=minimum,
            'score':score,'basis':'drug_features' if available else 'missing_drug_features','minimum':minimum}


def drug_pair(a,b,*,prospective=False):
    if a is None or b is None: return combine([item('Drug record',a,b,None,note='No corresponding reported drug; absence is not inferred.')])
    # Kind is an identity constraint, and roles are source metadata. Neither
    # earns a separate reward for two otherwise unrelated drugs.
    d=[item('Identity',a.get('name'),b.get('name'),drug_identity(a,b))]
    # Actual cohort use/reporting provenance is not a prospective drug attribute.
    # Planned use is supplied per candidate regimen to the risk model instead.
    for k in (() if prospective else ('presence','reporting_scope')):
        d.append(item(k,a.get(k),b.get(k),categorical(a.get(k),b.get(k)) if drug_identity(a,b)==1 else None))
    for k in ('classes','targets','pathways','mechanisms'):
        v=list_text(k,a.get(k),b.get(k)); d.append({'indicator':k,**v})
    return combine(d)


def route(v):
    s=norm(v).replace('.','')
    return {'iv':'intravenous','intravenously':'intravenous','po':'oral','orally':'oral','sc':'subcutaneous'}.get(s,s)


def days(v):
    if not known(v): return None
    if isinstance(v,(int,float)): return float(v)
    s=norm(v)
    if any(t in s for t in ('throughout','until progression','all cycles')): return None
    m=re.fullmatch(r'(?:every\s+|q)?(\d+(?:\.\d+)?)\s*(days?|d|weeks?|w|hours?|h|minutes?|min)',s)
    if not m:return None
    factor=7 if m[2].startswith('w') else 1/24 if m[2].startswith('h') else 1/1440 if m[2].startswith('m') else 1
    return float(m[1])*factor


def administration_days(v):
    if isinstance(v,list): return sorted(set(v))
    s=norm(v)
    if not re.fullmatch(r'(?:on\s+)?(?:days?\s*)?[\d,\s–\-and]+',s): return None
    result=[]
    for token in re.findall(r'\d+\s*[-–]\s*\d+|\d+',s):
        if '-' in token or '–' in token:
            lo,hi=map(int,re.split('[-–]',token))
            if hi-lo>366:return None
            result.extend(range(lo,hi+1))
        else:result.append(int(token))
    return sorted(set(result)) or None


def phase_pair(a,b):
    if a is None or b is None:return combine([item('Dosing phase',a,b,None)])
    from phase1_extraction.clinical_protocol import normalized_phase
    a,b=normalized_phase(a),normalized_phase(b)
    details=[]
    # Stage names align records in match(), but do not add clinical evidence.
    for k in ('route','form','frequency'):
        x,y=a.get(k),b.get(k)
        if k=='frequency':
            aliases={'qd':'once daily','daily':'once daily','bid':'twice daily','tid':'three times daily','q2w':'every 2 weeks','q3w':'every 3 weeks','q4w':'every 4 weeks'}
            x,y=aliases.get(norm(x),x),aliases.get(norm(y),y)
        details.append(item(k,x,y,categorical(route(x),route(y)) if k=='route' else categorical(x,y)))
    ap,bp=((p.get('structured') or {}).get('frequency_pattern') for p in (a,b))
    if ap or bp:details.append(item('Frequency pattern',ap,bp,categorical(ap,bp) if 'other' not in (ap,bp) else None))
    for k in ('cycle','duration','infusion_duration'):
        x,y=a.get(k),b.get(k); dx,dy=days(x),days(y)
        details.append(item(k,x,y,numeric(dx,dy,28.) if dx is not None and dy is not None else text_score(x,y)))
    x,y=a.get('administration_days'),b.get('administration_days')
    details.append(item('Administration days',x,y,set_score(administration_days(x),administration_days(y))))
    for k in ('infusion_rate','start','end'):
        details.append(item(k,a.get(k),b.get(k),text_score(a.get(k),b.get(k))))
    return combine(details)


def admin_pair(a,b):
    if a is None or b is None:return combine([item('Administration',a,b,None)])
    # The caller already established correspondence. Compare reported schedules
    # even across research/treatment roles or when entity kind is missing.
    if a.get('presence')=='absent' or b.get('presence')=='absent':
        return combine([item('Administration',a.get('presence'),b.get('presence'),None,applicable=False)])
    phases=match(a.get('phases',[]),b.get('phases',[]),phase_affinity)
    result=[{'indicator':'Dosing phase',**phase_pair(x,y)} for x,y in phases]
    result.append({'indicator':'Drug-specific dose adjustment rules',**conditions(a.get('adjustments',[]),b.get('adjustments',[]))})
    basis_aliases={'daily total':'daily_total','per administration':'per_administration','infusion rate':'infusion_rate'}
    ab,bb=(basis_aliases.get(norm(x.get('dose_basis')),norm(x.get('dose_basis'))) for x in (a,b))
    if known(ab) or known(bb):result.append(item('Dose basis',ab,bb,categorical(ab,bb)))
    result.append(item('Formulation',a.get('formulation'),b.get('formulation'),text_score(a.get('formulation'),b.get('formulation'))))
    # A text schedule is a fallback, not a second copy of structured attributes.
    if not a.get('phases') or not b.get('phases'):
        result.append(item('Schedule text',a.get('schedule'),b.get('schedule'),text_score(a.get('schedule'),b.get('schedule'))))
    return combine(result)


def phase_affinity(a,b):
    """Align repeated/unnamed stages by administration, after stage identity."""
    return phase_pair(a,b)['score']


def condition(v):
    """Keep logical predicates intact; do not guess a threshold from arbitrary prose."""
    if isinstance(v,str): v={'raw_text':v}
    from phase1_extraction.clinical_protocol import controlled_text, condition_topic
    v=dict(v)
    # Closed categories and explicit numeric predicates take precedence over
    # prose parsing. Preserve the original statement for exceptions and logic.
    for field, code, vocabulary, fallback in [
        ('metric','metric_code','clinical_metric',v.get('metric') or v.get('topic')),
        ('comparator','comparator_code','comparison',v.get('operator') or v.get('comparator')),
        ('unit_raw','unit_code','condition_unit',v.get('unit') or v.get('unit_raw'))]:
        value=controlled_text(vocabulary,v.get(code),fallback)
        if value is not None:
            v[field]=value
            if field=='comparator':v['operator']=value
            if field=='unit_raw':v['unit']=value
    raw=v.get('raw_text') or v.get('raw') or ''
    s=norm(raw).replace('≥','>=').replace('≤','<=')
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',s)
    match_=re.search(r'(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)\s*([\w/%²]+)?',s)
    grade_range=re.search(r'\bgrades?\s*([0-5])\s*(?:or|to|through|[-–])\s*(?:grade\s*)?([0-5])\b',s)
    grade=re.search(r'\bgrades?\s*(>=|<=|>|<|=)?\s*([0-5])\b',s)
    if grade and (match_ is None or grade.start()<=match_.start()<grade.end()):
        v=dict(v)
        if not v.get('operator') and not v.get('comparator'):v['comparator']='between' if grade_range else grade[1] or '='
        if v.get('threshold') is None:v['threshold']=float(grade[2])
        if not v.get('unit') and not v.get('unit_raw'):v['unit_raw']='grade'
    if grade_range and v.get('upper',v.get('threshold_upper')) is None and v.get('comparator',v.get('operator'))=='between':
        v['upper']=float(grade_range[2])
    operator=v.get('operator') or v.get('comparator') or (match_[1] if match_ else None)
    upper=v.get('upper',v.get('threshold_upper'))
    # Older records encoded an inclusive grade interval with '=' or '>=' and
    # a separate upper bound. Do not relax strict inequalities or change bounds.
    if (norm(v.get('unit') or v.get('unit_raw'))=='grade' and v.get('threshold') is not None
            and upper is not None and 0 <= v['threshold'] < upper <= 5 and operator in {'=','>=','between'}):
        operator='between'
    topic=v.get('metric') or v.get('topic') or v.get('category') or re.sub(r'\bgrades?\s*[<>=]*\s*\d+|[<>=]+\s*\d+(?:\.\d+)?', '',s).strip()
    logic_text=s[:grade_range.start()]+s[grade_range.end():] if grade_range else s
    return {**{key:v[key] for key in ('metric_code','comparator_code','unit_code','category_code','action') if v.get(key)},
            'clause_id':v.get('clause_id'), 'raw':raw,'topic':condition_topic(topic,v.get('metric_code')),
            'operator':operator,
            'threshold':v.get('threshold') if v.get('threshold') is not None else (float(match_[2]) if match_ else None),
            'upper':upper,'unit':v.get('unit') or v.get('unit_raw') or (match_[3] if match_ else None),
            'timing':v.get('timing') or v.get('timing_raw'), 'negated':v.get('negated'),
            'logic':v.get('logic') or (v.get('connective_to_previous') if v.get('connective_to_previous') not in {None,'NONE','UNCLEAR'} else None),
            'logic_tokens':sorted(set(re.findall(r'\b(?:and|or|not|without|except|unless)\b',logic_text)))}


def canonical_threshold(value,unit):
    if value is None:return value,norm(unit)
    u=norm(unit).replace('μ','u').replace('µ','u').replace(' ','')
    # Exact dimensional conversions only; no analyte-specific mmol conversion.
    units={'mg/dl':('g/l',.01),'g/dl':('g/l',10.),'g/l':('g/l',1.),'mg/l':('g/l',.001),
           '/ul':('/l',1e6),'cells/ul':('/l',1e6),'10^9/l':('/l',1e9),'/l':('/l',1.),
           'hour':('day',1/24),'hours':('day',1/24),'day':('day',1.),'days':('day',1.),'week':('day',7.),'weeks':('day',7.)}
    canonical,factor=units.get(u,(u,1.))
    return value*factor,canonical


def _logical_parentheses(raw):
    """Ignore parenthesized annotations that contain no Boolean operator."""
    stack=[[]]
    for token in re.findall(r'\(|\)|\band\b|\bor\b',norm(raw)):
        if token=='(':stack.append([])
        elif token==')' and len(stack)>1:
            group=stack.pop()
            if any(x in {'and','or'} for x in group):stack[-1].extend(['(',*group,')'])
        else:stack[-1].append(token)
    while len(stack)>1:
        group=stack.pop();stack[-1].extend(['(',*group])
    return stack[0]


def condition_comparison_text(c):
    """Use a simple predicate's topic once; retain prose with extra meaning.

    The numeric predicate is scored below. Only a complete, unambiguous
    restatement can be reduced to its topic. Exceptions, timing, additional
    measures and unknown wording remain available for text comparison.
    Explicit structured values still take precedence over source wording.
    """
    from phase1_extraction.clinical_protocol import condition_text
    raw, topic = condition_text(c.get('raw')), condition_text(c.get('topic'))
    if not raw:
        return topic
    if not topic or c.get('threshold') is None or not c.get('operator'):
        return raw
    number = r'[-+]?\d+(?:\.\d+)?'
    comparison = r'(?:>=|<=|!=|>|<|=|between|at least|at most|greater than|less than)'
    value = comparison + r'\s*' + number + r'(?:\s*(?:to|and|[-–])\s*' + number + r')?'
    unit = re.escape(condition_text(c.get('unit')))
    subject = re.escape(topic)
    predicates = [subject + r'\s*:?\s*' + value + (r'\s*' + unit if unit else '')]
    if norm(c.get('unit')) == 'grade':
        grade = r'grades?\s*(?:' + comparison + r'\s*)?' + number + r'(?:\s*to\s*' + number + r')?'
        predicates.extend([grade + r'\s+' + subject, subject + r'\s*:?\s*' + grade])
    return topic if any(re.fullmatch(pattern, raw) for pattern in predicates) else raw


def condition_pair(a,b):
    if a is None or b is None:return combine([item('Condition',a,b,None)])
    at,bt=condition_comparison_text(a),condition_comparison_text(b)
    d=[item('Clinical condition',at,bt,text_score(at,bt))]
    for k in ('operator','negated','logic'):
        d.append(item(k,a.get(k),b.get(k),categorical(a.get(k),b.get(k))))
    if a.get('action') or b.get('action'):
        aa,ba=a.get('action'),b.get('action')
        d.append(item('Adjustment action',aa,ba,categorical(aa,ba) if 'other' not in (aa,ba) else None))
    # Operators/negation must not be erased by high textual overlap.
    conflict=any(known(a.get(k)) and known(b.get(k)) and a[k]!=b[k] for k in ('operator','negated','logic'))
    if a.get('action') not in (None,'other') and b.get('action') not in (None,'other') and a['action']!=b['action']:conflict=True
    if a['logic_tokens']!=b['logic_tokens']:conflict=True
    if '(' in a['raw'] or '(' in b['raw']:
        if _logical_parentheses(a['raw']) != _logical_parentheses(b['raw']):conflict=True
    for k in ('threshold','upper'):
        av,au=canonical_threshold(a.get(k),a.get('unit'));bv,bu=canonical_threshold(b.get(k),b.get('unit'))
        scale=max(abs(av or 0),abs(bv or 0),1.)
        d.append(item(k,a.get(k),b.get(k),numeric(av,bv,scale) if au==bu else None,
                      note='Incompatible units; threshold not compared.' if au!=bu else ''))
    for label,pattern in [('Grade',r'\bgrade\s*(>=|<=|≥|≤|>|<|=)?\s*([0-5])\b'),
                          ('Duration',r'(>=|<=|≥|≤|>|<|=)?\s*(\d+(?:\.\d+)?)\s*(hours?|days?|weeks?)\b')]:
        ma,mb=re.search(pattern,norm(a['raw'])),re.search(pattern,norm(b['raw']))
        if ma and mb:
            dimension='grade' if label=='Grade' else 'day'
            if all(c.get('threshold') is not None and canonical_threshold(c['threshold'],c.get('unit'))[1]==dimension for c in (a,b)):
                continue  # The primary structured threshold already scores it.
            oa,ob=(ma[1] or '=').replace('≥','>=').replace('≤','<='),(mb[1] or '=').replace('≥','>=').replace('≤','<=')
            if oa!=ob:conflict=True
            va,vb=float(ma[2]),float(mb[2])
            if label=='Duration':va,_=canonical_threshold(va,ma[3]);vb,_=canonical_threshold(vb,mb[3])
            d.append(item(label,ma[0],mb[0],numeric(va,vb,max(va,vb,1.))))
    d.append(item('Timing',a.get('timing'),b.get('timing'),text_score(a.get('timing'),b.get('timing'))))
    result=combine(d)
    if conflict:result['score']=0.;result['note']='Explicit predicate/logic difference; text overlap cannot override it.'
    return result


def condition_affinity(a,b):
    from phase1_extraction.controlled_vocabulary import canonical_value
    left,right=(canonical_value('clinical_metric',c['topic']) for c in (a,b))
    if left not in (None,'other') and right not in (None,'other') and left!=right:
        return 0.
    score=text_score(a['topic'],b['topic']) or 0.
    # A dummy unmatched slot is preferable to pairing unrelated clinical rules.
    return score if score>=.35 else 0.


def condition_signature(c, *, include_text=False):
    from phase1_extraction.clinical_protocol import condition_text
    result={k:norm(c.get(k)) for k in ('topic','operator','timing','negated','logic','action','logic_tokens')}
    for k in ('threshold','upper'):
        result[k]=canonical_threshold(c.get(k),c.get('unit'))
    if include_text:result['raw']=condition_comparison_text(c)
    return result


def normalized_conditions(values):
    # Local IDs/provenance must not multiply one clinical statement's weight.
    result={}
    for value in values or []:
        c=condition(value)
        # IDs identify source/group references, not clinical correspondence.
        # Leaving them here makes equal-topic assignment ties depend on UUIDs.
        c.pop('clause_id',None)
        result.setdefault(stable(condition_signature(c,include_text=True)),c)
    return list(result.values())


def conditions(a,b):
    a,b=(normalized_conditions(values) for values in (a,b))
    return combine([{'indicator':'Condition',**condition_pair(x,y)} for x,y in match(a,b,condition_affinity)] or [item('Conditions',a,b,None)])


def logic_comparison(a,b,fields):
    from phase1_extraction.clinical_protocol import logic_tree
    leaf=lambda c:condition_signature(condition(c))
    left=logic_tree(a.get('logic_groups',[]),[c for f in fields for c in a.get(f,[])],leaf_key=leaf)
    right=logic_tree(b.get('logic_groups',[]),[c for f in fields for c in b.get(f,[])],leaf_key=leaf)
    return item('Logical grouping',left,right,float(left==right) if left is not None and right is not None else None,
                note='Explicit nested AND/OR structure; unresolved logic is not assumed equivalent.')


def window_pair(a,b):
    if a is None or b is None:return combine([item('Window',a,b,None)])
    d=[item('Window type',a.get('kind'),b.get('kind'),categorical(a.get('kind'),b.get('kind')))]
    type_conflict=known(a.get('kind')) and known(b.get('kind')) and a['kind']!=b['kind']
    anchors={'start of therapy':'first treatment dose','start of treatment':'first treatment dose',
             'treatment initiation':'first treatment dose','first dose':'first treatment dose'}
    for k in ('start','end'):
        av,bv=a.get(k),b.get(k)
        from phase1_extraction.clinical_protocol import controlled_text
        av=controlled_text('window_anchor', a.get(k+'_anchor'), av)
        bv=controlled_text('window_anchor', b.get(k+'_anchor'), bv)
        if k=='start':av,bv=anchors.get(norm(av),av),anchors.get(norm(bv),bv)
        d.append(item(k,a.get(k),b.get(k),text_score(av,bv)))
        # Category equality alone does not say which drug, stage or cycle.
        qualified={'first_specified_drug_dose','start_of_stage','end_of_specified_cycle'}
        if a.get(k+'_anchor') in qualified or b.get(k+'_anchor') in qualified:
            d.append(item(k+' specified drug / stage / cycle',a.get(k),b.get(k),text_score(a.get(k),b.get(k))))
    d.append(item('Days',a.get('days'),b.get('days'),numeric(a.get('days'),b.get('days'),28.) if not type_conflict else None))
    d.append(item('Cycles',a.get('cycles'),b.get('cycles'),text_score(a.get('cycles'),b.get('cycles'))))
    if a.get('days') is None or b.get('days') is None or not known(a.get('start')) or not known(b.get('start')):
        d.append(item('Window definition',a.get('raw'),b.get('raw'),text_score(a.get('raw'),b.get('raw'))))
    r=combine(d)
    if type_conflict:r['score']=0.;r['note']='Formal DLT window and safety monitoring are different endpoints.'
    return r


def compare_profiles(left,right,*,mode='pretrial',weights=None,text_vectors=None):
    if mode not in {'pretrial','full'}:raise ValueError('Comparison mode must be pretrial or full.')
    token=_TEXT_VECTORS.set(text_vectors if text_vectors is not None else _TEXT_VECTORS.get())
    try:return _compare(left,right,mode,normalize_weights(weights))
    finally:_TEXT_VECTORS.reset(token)


def compare_mapped_profiles(left, right, pairs, *, weights=None, text_vectors=None):
    """Directional target scoring; uses the SAME fixed mapping as transport."""
    token = _TEXT_VECTORS.set(text_vectors if text_vectors is not None else _TEXT_VECTORS.get())
    try:
        return _compare(left, right, 'pretrial', normalize_weights(weights), pairs)
    finally:
        _TEXT_VECTORS.reset(token)


def _compare(a,b,mode,weights,forced_pairs=None):
    groups={}
    pa,pb=a.get('population',{}),b.get('population',{})
    d=[]
    for key in ('disease','modifiers','sex'):
        av,bv=pa.get(key),pb.get(key)
        if key=='sex':
            from phase1_extraction.clinical_protocol import controlled_text
            av,bv=([controlled_text('sex_eligibility',None,x) for x in (v if isinstance(v,list) else [v]) if known(x)] for v in (av,bv))
        elif key=='modifiers':
            from phase1_extraction.clinical_protocol import controlled_list
            av,bv=(controlled_list('disease_modifier',p.get('disease_modifier_codes'),p.get(key)) for p in (pa,pb))
        r=list_text(key,av,bv);d.append({'indicator':key,**r})
    # Compare bounds only when actually given. Never impute 0/100 years.
    for key in ('age_min','age_max'):
        d.append(item(key,pa.get(key),pb.get(key),numeric(pa.get(key),pb.get(key),20.)))
        if pa.get(key+'_operator') or pb.get(key+'_operator'):
            d.append(item(key+' comparator',pa.get(key+'_operator'),pb.get(key+'_operator'),categorical(pa.get(key+'_operator'),pb.get(key+'_operator'))))
    if mode=='full':d.append({'indicator':'Observed baseline',**list_text('Baseline',pa.get('baseline'),pb.get('baseline'))})
    groups['population']=combine(d)
    from .evidence_relevance import toxicity_eligibility
    ea,eb=(toxicity_eligibility(p.get('eligibility',{})) for p in (a,b))
    eligibility_fields=['inclusion','exclusion','unclassified']
    groups['eligibility']=combine([{'indicator':k,**conditions(ea.get(k,[]),eb.get(k,[]))} for k in eligibility_fields])
    if ea.get('logic_groups') or eb.get('logic_groups'):
        groups['eligibility']=combine([*groups['eligibility']['details'],logic_comparison(ea,eb,eligibility_fields)])
    config_pairs=match(a.get('configurations',[]),b.get('configurations',[]),lambda x,y:
        sum(drug_affinity(l,r) for l,r in match(x.get('drugs',[]),y.get('drugs',[]),drug_affinity) if l and r))
    drug_results=[];admin_results=[];structure_results=[];pairing=[]
    for x,y in config_pairs:
        if x is None or y is None:
            for results in (drug_results,admin_results,structure_results):results.append(item('Configuration',x,y,None))
            continue
        if forced_pairs is None:
            pairs=match(x.get('drugs',[]),y.get('drugs',[]),drug_affinity)
        else:
            targets = x.get('drugs', [])
            sources = {d.get('model_drug_id'): d for d in y.get('drugs', [])}
            pairs = [(d, sources.get(forced_pairs.get(d.get('model_drug_id')))) for d in targets]
            used = {v for v in forced_pairs.values()}
            pairs.extend((None, d) for key, d in sources.items() if key not in used)
        pairing.append({'left':x.get('label'),'right':y.get('label'),'drugs':[(l.get('name') if l else None,r.get('name') if r else None) for l,r in pairs]})
        drug_results.append({'indicator':'Configuration',**combine([{'indicator':'Drug',**drug_pair(l,r,prospective=mode=='pretrial')} for l,r in pairs])})
        ar=[{'indicator':'Drug administration',**admin_pair(l,r)} for l,r in pairs]
        ar.append({'indicator':'Dose adjustment rules',**conditions(x.get('adjustments',[]),y.get('adjustments',[]))})
        admin_results.append({'indicator':'Configuration',**combine(ar)})
        st=[item(k,x.get(k),y.get(k),categorical(x.get(k),y.get(k))) for k in (('phase','role','status') if mode=='full' else ('phase','role'))]
        structure_results.append({'indicator':'Configuration',**combine(st)})
    groups['drugs']=combine(drug_results or [item('Drugs',None,None,None)])
    groups['administration']=combine(admin_results or [item('Administration',None,None,None)])
    groups['structure']=combine(structure_results or [item('Structure',None,None,None)])
    da,db=a.get('dlt',{}),b.get('dlt',{})
    ds=[{'indicator':'Criteria',**conditions(da.get('criteria',[]),db.get('criteria',[]))},
        {'indicator':'Exceptions',**conditions(da.get('exceptions',[]),db.get('exceptions',[]))},
        item('Grading system',da.get('grading'),db.get('grading'),text_score(da.get('grading'),db.get('grading'))),
        item('Grading system version',da.get('grading_version'),db.get('grading_version'),categorical(da.get('grading_version'),db.get('grading_version')))]
    if da.get('logic_groups') or db.get('logic_groups'):ds.append(logic_comparison(da,db,['criteria','exceptions']))
    if mode=='full':ds.append({'indicator':'Observed toxicities (separate from definition)',**list_text('Observed toxicities',da.get('observed'),db.get('observed'))})
    groups['dlt']=combine(ds)
    groups['window']=combine([{'indicator':'Window',**window_pair(x,y)} for x,y in match(a.get('windows',[]),b.get('windows',[]),lambda x,y:(categorical(x.get('kind'),y.get('kind')) or 0)+(text_score(x.get('raw'),y.get('raw')) or 0))] or [item('Window',None,None,None)])
    available=sum(weights[k] for k,v in groups.items() if v['score'] is not None)
    total=sum(weights[k]*v['score'] for k,v in groups.items() if v['score'] is not None)/available if available else None
    return {'version':VERSION,'mode':mode,'similarity':total,'available_weight':available,
            'weights':weights,'field_scores':{k:v['score'] for k,v in groups.items() if v['score'] is not None},
            'groups':groups,'pairing':pairing,'design_compared':False,'dose_magnitudes_compared':False}


def all_texts(profiles):
    values=set()
    metadata={'observed','baseline','note','evidence','review_notes','purpose',
              'research_role','role','phase','phase_code','label','version',
              'clause_id','group_id','model_drug_id','component_id','source_id',
              'unresolved_source_logic'}
    def walk(v):
        if isinstance(v,str) and known(v):values.add(norm(v))
        elif isinstance(v,dict):
            for k,x in v.items():
                if k=='eligibility':
                    from .evidence_relevance import toxicity_eligibility
                    walk(toxicity_eligibility(x))
                elif k not in metadata:walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    for p in profiles:walk(p)
    return sorted(values)
