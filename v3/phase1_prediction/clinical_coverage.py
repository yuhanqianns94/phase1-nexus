"""Outcome-free information gate, evaluated before any similarity/embedding.

Within each clinical category, coverage is the fraction of reported field types
available on both sides (intersection / union). An entirely missing category
has zero coverage. The six clinical category weights are kept in the denominator.
Metadata, dose magnitudes and historical outcomes never establish coverage.
"""
from __future__ import annotations

from .profile_similarity import known, normalize_weights, LABELS

VERSION = 'clinical_information_coverage_v1'
DEFAULT_MIN_CLINICAL_COVERAGE = .20


def _reported(value):
    if isinstance(value, (list, tuple)):
        return any(_reported(item) for item in value)
    if isinstance(value, dict):
        # IDs and provenance alone are not clinical information.
        return any(_reported(value.get(key)) for key in (
            'raw_text', 'raw', 'topic', 'metric', 'threshold', 'upper',
            'operator', 'comparator', 'unit', 'timing', 'timing_raw'))
    return known(value)


def clinical_fields(profile):
    from .evidence_relevance import toxicity_eligibility
    from phase1_extraction.clinical_protocol import controlled_list, controlled_text, normalized_phase
    fields = {key: set() for key in LABELS if key != 'structure'}
    def add(group, key, value):
        if _reported(value):
            fields[group].add(key)
    population = profile.get('population', {})
    for key in ('disease', 'modifiers', 'sex', 'age_min', 'age_max'):
        value = population.get(key)
        if key == 'modifiers':
            value = controlled_list('disease_modifier', population.get('disease_modifier_codes'), value)
        add('population', key, value)
    eligibility = toxicity_eligibility(profile.get('eligibility', {}))
    # Classifying a condition as inclusion/exclusion/unclassified does not
    # create extra information. The substantive clinical condition counts once.
    for rows in (eligibility.get(key, []) for key in ('inclusion', 'exclusion', 'unclassified')):
        add('eligibility', 'clinical_conditions', rows)
    for config in profile.get('configurations', []):
        add('administration', 'adjustments', config.get('adjustments'))
        for drug in config.get('drugs', []):
            for key in ('name', 'classes', 'targets', 'pathways', 'mechanisms'):
                add('drugs', key, drug.get(key))
            if drug.get('presence') == 'absent':
                continue
            for key in ('formulation', 'dose_basis', 'adjustments'):
                add('administration', key, drug.get(key))
            if not drug.get('phases'):
                add('administration', 'schedule', drug.get('schedule'))
            for original in drug.get('phases', []):
                phase = normalized_phase(original)
                for key in ('route', 'form', 'frequency', 'cycle', 'duration',
                            'infusion_duration', 'administration_days'):
                    add('administration', key, phase.get(key))
                pattern = (phase.get('structured') or {}).get('frequency_pattern')
                if pattern != 'other':
                    add('administration', 'frequency', pattern)
    dlt = profile.get('dlt', {})
    for key in ('criteria', 'exceptions', 'grading', 'grading_version'):
        add('dlt', key, dlt.get(key))
    for window in profile.get('windows', []):
        for key in ('kind', 'days', 'cycles'):
            add('window', key, window.get(key))
        for key in ('start', 'end'):
            add('window', key, controlled_text('window_anchor', window.get(key + '_anchor'), window.get(key)))
        add('window', 'definition', window.get('raw'))
    return fields


def clinical_coverage(left, right, *, weights=None):
    weights = normalize_weights(weights)
    left, right = clinical_fields(left), clinical_fields(right)
    groups = {}
    for key in left:
        union, common = left[key] | right[key], left[key] & right[key]
        groups[key] = {'label': LABELS[key], 'coverage': len(common) / len(union) if union else 0.,
                       'common_fields': sorted(common), 'reported_fields': sorted(union)}
    return {'version': VERSION, 'coverage': sum(weights[key] * group['coverage'] for key, group in groups.items()),
            'weights': weights, 'groups': groups}
