"""Conservative clinical views of source eligibility, without changing source facts.

Only complete, unambiguous consent/participation statements are administrative.
A mention of consent inside an age, capacity, treatment, or other clinical rule
is not enough to discard it. OR/negated/uncertain logic is kept intact.
"""
import copy
import re


_CONSENT = re.compile(
    r"(?:the )?(?:(?:patients?|participants?|subjects?) (?:must |should |were required to )?)?"
    r"(?:(?:be )?(?:willing and able|able|willing) to |ability to |"
    r"(?:must |should )?(?:have )?|(?:has |have |had |having ))"
    r"(?:provide|give|sign|provided|given|signed)? ?"
    r"(?:a |the )?(?:written )?(?:informed )?consent(?: form)?"
    r"(?: (?:to participate in|for participation in|for) (?:the |this )?(?:study|trial))?"
    r"(?: (?:is |was )?required)?", re.I,
)
_PARTICIPATION = re.compile(
    r"(?:(?:patients?|participants?|subjects?) (?:must |should )?)?"
    r"(?:willingness to participate|willing to participate|agree to participate|agreed to participate)"
    r" in (?:the |this )?(?:study|trial)", re.I,
)
_CONSENT_REQUIRED = re.compile(
    r"(?:written )?(?:informed )?consent(?: form)? "
    r"(?:must be |should be |was |is )?(?:obtained|signed|provided|required)"
    r"(?: (?:before|prior to) (?:study |trial )?(?:enrollment|enrolment|participation))?", re.I,
)


def _pure_administrative(rule):
    """Do not infer irrelevance from keywords or discard mixed clinical prose."""
    if isinstance(rule, dict):
        # Explicit clinical structure overrides the prose classification.
        if any(rule.get(key) not in (None, '', [], 'other', 'unclear') for key in
               ('threshold', 'upper', 'operator', 'comparator_code', 'unit', 'unit_code',
                'timing', 'timing_raw')) or rule.get('negated') is True:
            return False
        if rule.get('metric_code') not in (None, '', 'other', 'unclear'):
            return False
        texts = [rule.get(key) for key in ('raw', 'raw_text', 'topic') if rule.get(key)]
    else:
        texts = [rule]
    if not texts:
        return False
    for value in texts:
        text = re.sub(r'\s+', ' ', str(value)).strip().strip(' .;:')
        if not (_CONSENT.fullmatch(text) or _PARTICIPATION.fullmatch(text) or _CONSENT_REQUIRED.fullmatch(text)):
            return False
    return True


def irrelevant_eligibility_indices(section):
    """Return removable row positions; complex Boolean branches stay untouched."""
    result = {field: set() for field in ('inclusion', 'exclusion', 'unclassified')}
    rules = [rule for field in result for rule in section.get(field, [])]
    groups = section.get('logic_groups', [])
    # Legacy OR linkage has no reliable group boundary: retain the whole section.
    if any(isinstance(rule, dict) and (
        str(rule.get('logic', '')).upper() not in ('', 'NONE', 'AND') or
        any(str(token).lower() in ('or', 'unless', 'except') for token in rule.get('logic_tokens', []) or [])
    ) for rule in rules):
        return result
    ids = [rule['clause_id'] for rule in rules if isinstance(rule, dict) and rule.get('clause_id')]
    if len(ids) != len(set(ids)):
        return result
    nodes = {group.get('group_id'): group for group in groups}
    if len(nodes) != len(groups) or None in nodes or set(nodes) & set(ids):
        return result
    protected = set()
    def visit(identity, stack, protect=False):
        if identity in stack:
            raise ValueError('Cyclic source eligibility logic')
        if identity in ids:
            if protect:
                protected.add(identity)
            return
        if identity not in nodes:
            raise ValueError('Unresolved source eligibility reference')
        group = nodes[identity]
        protect = protect or group.get('operator') != 'AND' or bool(group.get('negated'))
        for child in group.get('children', []):
            visit(child, stack | {identity}, protect)
    try:
        for identity in nodes:
            visit(identity, set())
    except (ValueError, TypeError):
        return result
    for field in result:
        result[field] = {i for i, rule in enumerate(section.get(field, []))
                         if _pure_administrative(rule) and
                         (not isinstance(rule, dict) or rule.get('clause_id') not in protected)}
    return result


def toxicity_eligibility(section):
    """Deep-copy a clinical-only eligibility view, preserving meaningful logic.

    Removed administrative AND conjuncts and now-empty groups are pruned from
    this view only. OR descendants are retained because deleting a disjunct can
    change the meaning of the remaining clinical requirement.
    """
    result = copy.deepcopy(section)
    hidden = irrelevant_eligibility_indices(section)
    if not any(hidden.values()):
        return result
    removed = set()
    for field, indices in hidden.items():
        if field not in result:
            continue
        removed.update(rule['clause_id'] for i, rule in enumerate(result[field])
                       if i in indices and isinstance(rule, dict) and rule.get('clause_id'))
        result[field] = [rule for i, rule in enumerate(result[field]) if i not in indices]
    if 'logic_groups' in result:
        groups = result['logic_groups']
        while True:
            for group in groups:
                group['children'] = [child for child in group.get('children', []) if child not in removed]
            empty = {group['group_id'] for group in groups if not group['children']}
            if not empty:
                break
            removed.update(empty)
            groups = [group for group in groups if group['group_id'] not in empty]
        # An AND wrapper around one surviving clinical condition adds no
        # clinical information. Remove it from the scoring view, including
        # references in parents, rather than rewarding the retained wrapper.
        while True:
            singles = {group['group_id']: group['children'][0] for group in groups
                       if group.get('operator') == 'AND' and not group.get('negated')
                       and len(group['children']) == 1}
            if not singles:
                break
            def resolve(identity):
                while identity in singles:
                    identity = singles[identity]
                return identity
            for group in groups:
                group['children'] = [resolve(child) for child in group['children']]
            groups = [group for group in groups if group['group_id'] not in singles]
        result['logic_groups'] = groups
    return result
