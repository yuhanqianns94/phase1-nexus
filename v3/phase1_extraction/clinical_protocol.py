"""Shared, lossless clinical normalization for reported and planned protocols.

No clinical defaults, outcomes, dose conversion or missing-value imputation.
"""
from __future__ import annotations
import copy
import re
from functools import lru_cache

PROTOCOL_VERSION = 'clinical_protocol_v3'


def statement_key(value):
    """Formatting-only identity; retain numbers, operators and Boolean grouping."""
    return ' '.join(str(value or '').casefold().replace('≥', '>=').replace('≤', '<=').split()).rstrip('.').strip()


@lru_cache(maxsize=1)
def _metric_aliases():
    from .controlled_vocabulary import CATALOG
    spec = CATALOG['clinical_metric']
    aliases = {**{key.replace('_', ' '): key for key in spec['options'] if key != 'other'},
               **{label.casefold(): key for key, label in spec['labels'].items() if key != 'other'},
               **spec['aliases']}
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in sorted(aliases, key=lambda x: (-len(x), x))) + r')(?!\w)')
    return pattern, aliases, spec['labels']


def condition_text(value):
    """Exact controlled synonyms only; all clinical qualifiers stay in the text."""
    pattern, aliases, labels = _metric_aliases()
    text = pattern.sub(lambda m: labels[aliases[m[0]]].casefold(), statement_key(value))
    return re.sub(r'\bgrades?\s*([0-5])\s*(?:or|to|through|[-–])\s*(?:grade\s*)?([0-5])\b',
                  r'grade \1 to \2', text)


def condition_topic(value, code=None):
    """Recognize a single named measure, never a numeric value as a topic."""
    from .controlled_vocabulary import canonical_value
    canonical = canonical_value('clinical_metric', code or value)
    if canonical and canonical != 'other':
        return controlled_text('clinical_metric', canonical)
    pattern, aliases, labels = _metric_aliases()
    text = statement_key(value)
    measures = {aliases[m[0]] for m in pattern.finditer(text)}
    if not measures and re.search(r'\byears? (?:old|of age)\b', text):
        measures.add('age')
    return labels[next(iter(measures))] if len(measures) == 1 else value


def controlled_text(vocabulary, code, raw=None):
    """Use an explicit category before exact alias normalization of source prose.

    Other is a classification, not a shared clinical meaning. Its detailed
    wording remains the comparison input, and unknown never becomes absence.
    """
    from .controlled_vocabulary import canonical_value, labels
    canonical = canonical_value(vocabulary, code) if code else canonical_value(vocabulary, raw)
    if canonical == 'other' or canonical is None:
        return raw
    return labels(vocabulary).get(canonical, canonical)


def controlled_list(vocabulary, codes, raw_values):
    from .controlled_vocabulary import canonical_value
    raw_values = raw_values or []
    if not codes:
        return [controlled_text(vocabulary, None, raw) for raw in raw_values]
    values = [controlled_text(vocabulary, code) for code in codes if code != 'other']
    if 'other' in codes:
        values.extend(raw for raw in raw_values if canonical_value(vocabulary, raw) in (None, 'other'))
    return list(dict.fromkeys(value for value in values if value))


def normalized_phase(value):
    p = copy.deepcopy(value)
    for field, vocabulary, code_field in [('route', 'route', 'route_code'),
                                         ('form', 'dosage_form', 'form_code'),
                                         ('phase', 'treatment_stage', 'phase_code')]:
        if p.get(field) or p.get(code_field):
            p[field] = controlled_text(vocabulary, p.get(code_field), p.get(field))
    schedule = p.get('structured') or {}
    f = str(p.get('frequency') or '').strip().lower()
    aliases = {'qd':'once daily','daily':'once daily','bid':'twice daily',
               'tid':'three times daily','weekly':'every 7 days','once weekly':'every 7 days'}
    f = aliases.get(f, f)
    m = re.fullmatch(r'(?:q|every\s+)(\d+(?:\.\d+)?)\s*(w|weeks?|d|days?|h|hours?)', f)
    if m:
        amount = float(m[1]) * (7 if m[2].startswith('w') else 1/24 if m[2].startswith('h') else 1)
        f = f'every {amount:g} days'
    if schedule.get('frequency_count') is not None and schedule.get('interval_value') is not None:
        n, t = schedule['frequency_count'], schedule['interval_value']
        unit = schedule.get('interval_unit')
        factor = {'days':1,'weeks':7,'hours':1/24}.get(unit)
        if factor:
            interval = t * factor
            f = ('once daily' if n == 1 else 'twice daily' if n == 2 else 'three times daily' if n == 3 else f'{n:g} times daily') if interval == 1 else f'every {interval:g} days' if n == 1 else f'{n:g} times every {interval:g} days'
    if f:p['frequency'] = f
    for raw, number, unit in [('cycle','cycle_value','cycle_unit'),('infusion_duration','infusion_value','infusion_unit')]:
        if schedule.get(number) is not None and schedule.get(unit):
            p[raw] = f"{schedule[number]:g} {schedule[unit]}"
    if schedule.get('administration_days'):
        p['administration_days'] = list(schedule['administration_days'])
    return p


def complete_conditions(raw_values, structured_values, convert):
    """Preserve every statement, attach structured clauses without sorting logic."""
    result, seen = [], set()
    by_raw = {}
    for c in structured_values:
        by_raw.setdefault(statement_key(c.get('raw_text') or c.get('raw')), []).append(c)
    for raw in raw_values:
        candidates = by_raw.get(statement_key(raw)) or [raw]
        for c in candidates:
            item = convert(c)
            identity = item.get('clause_id') or repr(item)
            if identity not in seen: result.append(item); seen.add(identity)
    for c in structured_values:
        item = convert(c)
        identity = item.get('clause_id') or repr(item)
        if identity not in seen:result.append(item);seen.add(identity)
    return result


def validate_logic(groups, clause_ids):
    nodes = {g['group_id']:g for g in groups}
    if len(nodes)!=len(groups):raise ValueError('Rule group IDs must be unique')
    if set(nodes)&set(clause_ids):raise ValueError('Rule and group IDs must be distinct')
    def visit(key, stack):
        if key in stack:raise ValueError('Rule groups cannot contain a cycle')
        if key in clause_ids:return
        if key not in nodes:raise ValueError(f'Unknown rule or group: {key}')
        g=nodes[key]
        if g['operator'] not in {'AND','OR'} or not g['children']:raise ValueError('Each rule group needs AND/OR and at least one child')
        for child in g['children']:visit(child,stack|{key})
    for key in nodes:visit(key,set())


def planned_issues(draft, *, require_doses=True):
    issues=[]
    pop=draft.get('profile',{}).get('population',{})
    if pop.get('age_min') is not None and pop.get('age_max') is not None and pop['age_min']>pop['age_max']:
        issues.append('Minimum age must not exceed maximum age.')
    if not draft.get('drugs'):issues.append('Add at least one drug.')
    if require_doses and not draft.get('points'):issues.append('Add at least one candidate regimen.')
    planning=draft.get('profile',{}).get('planning',{})
    from phase1_prediction.multidrug.workflow import phase_status
    if phase_status(draft.get('profile',{})) in {'out_of_scope','mixed_requires_separation'}:
        issues.append('Only Phase I trials or the explicitly separated Phase I component of I/II are supported.')
    for drug in draft.get('drugs',[]):
        if not drug.get('name','').strip():issues.append('Enter a name or development code for each drug.')
        if require_doses and drug.get('mode')=='numeric' and (not drug.get('unit') or not drug.get('reference_dose')):
            issues.append(f"{drug.get('name') or 'Drug'}: enter a dose unit and positive candidate dose.")
    for point in (draft.get('points',[]) if require_doses else []):
        for drug in draft.get('drugs',[]):
            state=planning.get('exposure_states',{}).get(point['level_id'],{}).get(drug['drug_id'])
            if state in {'Not entered','Undecided','Use: dose specified'} and (state!='Use: dose specified' or point['doses'].get(drug['drug_id']) in (None,0)):
                issues.append(f"{point['level_id']} / {drug['name']}: specify use and dose, or explicitly select dose unknown.")
    for name,fields in [('eligibility',['inclusion','exclusion','unclassified']),('dlt',['criteria','exceptions'])]:
        section=draft.get('profile',{}).get(name,{})
        clauses=[c for field in fields for c in section.get(field,[]) if isinstance(c,dict)]
        try:validate_logic(section.get('logic_groups',[]),{c['clause_id'] for c in clauses if c.get('clause_id')})
        except ValueError as error:issues.append(f'{name}: {error}')
    return issues


def logic_tree(groups, clauses, *, leaf_key=None):
    """An ID-independent predicate tree; invalid or unresolved logic is unknown."""
    if not groups:return None
    lookup={c.get('clause_id'):c for c in clauses if c.get('clause_id')}
    try:validate_logic(groups,set(lookup))
    except ValueError:return None
    nodes={g['group_id']:g for g in groups}
    children={c for g in groups for c in g['children']}
    def expand(key):
        if key in lookup:
            c=lookup[key]
            label = (leaf_key(c) if leaf_key else statement_key(controlled_text('clinical_metric',
                c.get('metric_code'), c.get('topic') or c.get('metric') or c.get('category') or c.get('raw') or c.get('raw_text'))))
            return ['condition', label]
        g=nodes[key]
        return [g['operator'], sorted([expand(x) for x in g['children']],key=repr)]
    return sorted([expand(key) for key in nodes if key not in children],key=repr)
