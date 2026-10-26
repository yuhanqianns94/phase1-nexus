"""One-to-one drug-evidence matching with explicit dose transport.

Research roles set candidate priority; clinical similarity remains a separate score.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

from phase1_prediction.profile_similarity import drug_affinity, drug_identity, drug_pair, drug_correspondence, compare_mapped_profiles
from .contracts import fingerprint

MATCHING_POLICY = 'automatic_primary_first_clinical_coverage_v8'


def target_profile(target):
    profile = copy.deepcopy(target.profile)
    context=dict((profile.get('configurations') or [{}])[0])
    profile['configurations'] = [{**context, 'status': 'planned', 'drugs': [
        {**d.profile, 'name': d.name, 'model_drug_id': d.drug_id,
         'research_priority': d.research_priority, 'research_role': d.research_role}
        for d in target.drugs]}]
    return profile


def coverage_target_profile(target):
    """Keep all planned companions in the information gate for a focal curve."""
    if hasattr(target, 'coverage_profile'):
        return target.coverage_profile
    projection = target.profile.get('single_drug_projection')
    if projection:
        from types import SimpleNamespace
        original = SimpleNamespace(profile=projection['original_profile'],
            drugs=[SimpleNamespace(**drug) for drug in projection['original_drugs']])
        return target_profile(original)
    return target_profile(target)


def match_target(target, scope, *, weights=None):
    tp = target_profile(target)
    from phase1_prediction.clinical_coverage import clinical_coverage
    minimum = getattr(target, 'minimum_clinical_coverage', 0.)
    source_profile = scope.lineage.get('single_drug_projection', {}).get('original_profile', scope.profile)
    coverage = clinical_coverage(coverage_target_profile(target), source_profile, weights=weights) if minimum > 0 else None
    if coverage is not None and coverage['coverage'] + 1e-12 < minimum:
        # Return before correspondence, similarity or lazy embedding evaluation.
        payload = {'scope_id': scope.scope_id, 'document_id': scope.document_id,
            'matching_policy': MATCHING_POLICY, 'pairs': {}, 'details': [], 'rejected_pairs': [],
            'unmatched_target_ids': [drug.drug_id for drug in target.drugs],
            'removed_source_ids': [drug.model_drug_id for drug in scope.drugs],
            'clinical_coverage': coverage, 'minimum_clinical_coverage': minimum,
            'comparison_skipped': 'insufficient_clinical_coverage',
            'score': {'similarity': None, 'field_scores': {}, 'groups': {}, 'weights': coverage['weights']}}
        payload['mapping_id'] = fingerprint(payload)
        return payload
    td = tp['configurations'][0]['drugs']
    sd = [{**d.profile, 'name': d.name, 'model_drug_id': d.model_drug_id,
           'research_role':d.research_role} for d in scope.drugs]
    matrix = np.zeros((len(td), len(sd) + len(td)))
    raw_scores = np.array([[drug_affinity(a, b) for b in sd] for a in td])
    correspondence = [[drug_correspondence(a,b,target.minimum_drug_similarity) for b in sd] for a in td]
    primary = [drug.research_role in {'primary', 'co_primary'} for drug in scope.drugs]
    # Lexicographic assignment: eligible main-study drugs first, then identity,
    # then drug features. A role never makes an unsupported drug eligible.
    identity_bonus = len(td) + 1
    primary_bonus = identity_bonus ** 2
    overrides = target.mapping_overrides.get(scope.scope_id, {})
    if len(set(overrides.values())) != len(overrides) or not set(overrides) <= {d.drug_id for d in target.drugs} or not set(overrides.values()) <= {d.model_drug_id for d in scope.drugs}:
        raise ValueError('Invalid target mapping override')
    for i, a in enumerate(td):
        for j, b in enumerate(sd):
            eligible = correspondence[i][j]['eligible']
            if target.drugs[i].drug_id in overrides:
                eligible = overrides[target.drugs[i].drug_id] == scope.drugs[j].model_drug_id
            if scope.drugs[j].model_drug_id in overrides.values() and overrides.get(target.drugs[i].drug_id) != scope.drugs[j].model_drug_id:
                eligible = False
            matrix[i, j] = (primary_bonus * primary[j] + identity_bonus * (drug_identity(a, b) == 1)
                            + raw_scores[i, j] + 1e-9) if eligible else -1e9
    ri, ci = linear_sum_assignment(matrix, maximize=True)
    pairs, details = {}, []
    for i, j in zip(ri, ci):
        if j >= len(sd) or matrix[i, j] < 0:
            continue
        pairs[target.drugs[i].drug_id] = scope.drugs[j].model_drug_id
        allowed_scores = [matrix[i, x] for x in range(len(sd)) if matrix[i, x] >= 0]
        details.append({'target_id': target.drugs[i].drug_id, 'source_id': scope.drugs[j].model_drug_id,
                        'affinity': float(raw_scores[i, j]), 'comparison': drug_pair(td[i], sd[j],prospective=True),
                        'drug_evidence': correspondence[i][j],
                        'ambiguous': bool(sum(abs(x - matrix[i, j]) < 1e-8 for x in allowed_scores) > 1),
                        'role_rule': ('manual_override' if target.drugs[i].drug_id in overrides else
                                      'primary_investigated_first' if primary[j] else 'other_drug_fallback'),
                        'target_role': target.drugs[i].research_role,
                        'source_role': scope.drugs[j].research_role,
                        'role_transfer': target.drugs[i].research_role != scope.drugs[j].research_role})
    score = compare_mapped_profiles(tp, scope.profile, pairs, weights=weights)
    payload = {'scope_id': scope.scope_id, 'document_id': scope.document_id, 'matching_policy': MATCHING_POLICY, 'pairs': pairs,
               'rejected_pairs': [{'target_id':target.drugs[i].drug_id, 'source_id':scope.drugs[j].model_drug_id,
                                   **correspondence[i][j], 'reason':'insufficient_drug_similarity'}
                                  for i in range(len(td)) for j in range(len(sd))
                                  if not correspondence[i][j]['eligible'] and overrides.get(target.drugs[i].drug_id)!=scope.drugs[j].model_drug_id],
               'details': details, 'unmatched_target_ids': [d.drug_id for d in target.drugs if d.drug_id not in pairs],
               'removed_source_ids': [d.model_drug_id for d in scope.drugs if d.model_drug_id not in pairs.values()],
               'score': score}
    if coverage is not None:
        payload.update(clinical_coverage=coverage, minimum_clinical_coverage=minimum)
    payload['mapping_id'] = fingerprint(payload)
    return payload


def _unit_scale(source, target):
    def split(unit):
        from phase1_extraction.dose_units import unit_spelling
        unit = ''.join((unit_spelling(unit) or '').lower().split()).replace('²', '2').replace('^2', '2').replace('µ', 'u').replace('mcg', 'ug')
        for prefix, factor in [('mg', 1.), ('ug', .001), ('g', 1000.)]:
            if unit == prefix or unit.startswith(prefix + '/'):
                return unit[len(prefix):], factor
        return unit, None
    a, fa = split(source); b, fb = split(target)
    if source == target:
        return 1.
    if a != b or fa is None or fb is None:
        raise ValueError('Incompatible dose units; no weight/BSA/schedule is imputed')
    return fb / fa


def mapped_relative_dose(target_drug, source_drug, value):
    if target_drug.mode == 'numeric' and source_drug.mode == 'background':
        raise ValueError(f'{target_drug.name}: numerical target doses require a numeric source axis; dose-unknown anchoring is prohibited')
    if target_drug.mode == 'background' and value not in (0, None):
        raise ValueError('Background target accepts only 0 (absent) or null (present without dose)')
    if value == 0:
        return 0.
    if target_drug.transport_mode == 'same_entity_unit_conversion':
        if target_drug.allow_background_anchor or target_drug.mode != 'numeric' or source_drug.mode != 'numeric' or value is None:
            raise ValueError('Same-entity unit conversion requires known numeric doses; background anchoring must use relative_dose_analogy')
        if drug_identity({**target_drug.profile, 'name': target_drug.name}, source_drug.profile) != 1:
            raise ValueError('Real unit conversion requires the same explicitly identified drug')
    if source_drug.mode == 'background':
        if target_drug.mode == 'background' or value is None:
            return 1.
        raise ValueError(f'{target_drug.name}: source has no numerical dose axis')
    if value is None:
        raise ValueError(f'{target_drug.name}: numerical source needs an explicit target dose/reference mapping')
    if target_drug.transport_mode == 'same_entity_unit_conversion':
        # A mass-prefix conversion does not establish equivalent treatment schedules.
        from phase1_prediction.profile_similarity import norm
        tp,sp=target_drug.profile,source_drug.profile
        fields=('route','frequency','administration_days','cycle','duration','infusion_duration','infusion_rate')
        t_phases,s_phases=tp.get('phases',[]),sp.get('phases',[])
        if len(t_phases)!=len(s_phases) and (t_phases or s_phases):
            raise ValueError('Same-entity dose conversion needs corresponding treatment phases')
        for tphase,sphase in zip(t_phases,s_phases):
            from phase1_extraction.clinical_protocol import normalized_phase
            tphase,sphase=normalized_phase(tphase),normalized_phase(sphase)
            patterns=[(phase.get('structured') or {}).get('frequency_pattern') for phase in (tphase,sphase)]
            if all(pattern and pattern!='other' for pattern in patterns) and patterns[0]!=patterns[1]:
                raise ValueError('Same-entity dose conversion requires matching frequency patterns; use explicit relative-dose analogy otherwise')
            for field in fields:
                if norm(tphase.get(field)) != norm(sphase.get(field)):
                    raise ValueError('Same-entity dose conversion requires matching schedules; use explicit relative-dose analogy otherwise')
        if norm(tp.get('schedule'))!=norm(sp.get('schedule')) and (tp.get('schedule') or sp.get('schedule')):
            raise ValueError('Same-entity dose conversion requires matching reported schedules')
        return value * _unit_scale(source_drug.unit, target_drug.unit) / source_drug.reference_dose
    if target_drug.reference_dose is None:
        raise ValueError('Relative dose analogy requires a frozen target reference dose')
    return value / target_drug.reference_dose
