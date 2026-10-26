"""Project one focal dose axis while retaining whole-regimen outcomes and context.

The synthetic second slot records whether any companion was administered. Source
cohorts retain their original DLT numerator and denominator exactly once; they
are never relabelled as observed monotherapy outcomes.
"""
from __future__ import annotations

import copy
from collections import defaultdict

from .contracts import DrugSlot, Exposure, ModelScope, RegimenPoint, TargetDrug, TargetRegimen, fingerprint

CONTEXT_ID = '__combination_context__'
PROJECTION_VERSION = 'focal_context_projection_v1'


def _context_profile():
    return {'name': 'Combination treatment context', 'model_drug_id': CONTEXT_ID,
            'role': 'combination_partner', 'single_drug_context': True}


def _context_slot():
    return DrugSlot(model_drug_id=CONTEXT_ID, name='Combination treatment context',
                    research_priority=2, research_role='background', mode='background',
                    profile=_context_profile())


def _project_profile(original, focal_profile):
    profile = copy.deepcopy(original)
    profile.pop('planning', None)
    configurations = profile.get('configurations') or [{}]
    for configuration in configurations:
        configuration['drugs'] = [copy.deepcopy(focal_profile), _context_profile()]
    profile['configurations'] = configurations
    return profile


def resolve_focal_target_id(target, focal_id=None):
    chosen = focal_id or target.profile.get('single_drug_analysis', {}).get('focal_drug_id')
    if chosen is None:
        chosen = next((drug.drug_id for drug in target.drugs if drug.mode == 'numeric'), None)
    drug = next((drug for drug in target.drugs if drug.drug_id == chosen), None)
    if drug is None or drug.mode != 'numeric':
        raise ValueError('Choose one focal drug with a numeric dose axis.')
    if chosen == CONTEXT_ID:
        raise ValueError('The combination context cannot be the focal drug.')
    return chosen


def project_target(target, focal_target_id=None):
    """Return numeric focal + binary context; preserve the original full plan."""
    target = target if isinstance(target, TargetRegimen) else TargetRegimen.model_validate(target)
    chosen = resolve_focal_target_id(target, focal_target_id)
    focal = next(drug.model_copy(deep=True) for drug in target.drugs if drug.drug_id == chosen)
    companions = [drug for drug in target.drugs if drug.drug_id != chosen]
    if any(drug.drug_id == CONTEXT_ID for drug in companions):
        if target.profile.get('single_drug_projection') and len(target.drugs) == 2:
            return target.model_copy(deep=True)
        raise ValueError('The reserved combination-context identifier is already in use.')
    companion_coordinates = [{drug.drug_id: point.doses[drug.drug_id] for drug in companions}
                             for point in target.points]
    if any(values != companion_coordinates[0] for values in companion_coordinates[1:]):
        raise ValueError('Keep every companion dose and use state fixed while varying the focal drug.')
    if any(point.doses[chosen] is None for point in target.points):
        raise ValueError('Every focal point needs a numeric dose; use zero for explicit absence.')
    context_present = any(value is None or value > 0 for value in companion_coordinates[0].values())
    focal.research_priority = 1
    context = TargetDrug(drug_id=CONTEXT_ID, name='Combination treatment context', research_priority=2,
                         research_role='background', mode='background', profile=_context_profile())
    profile = _project_profile(target.profile, {**focal.profile, 'name': focal.name, 'model_drug_id': chosen})
    profile['single_drug_analysis'] = {**profile.get('single_drug_analysis', {}), 'focal_drug_id': chosen}
    profile['single_drug_projection'] = {'version': PROJECTION_VERSION, 'focal_drug_id': chosen,
        'original_drugs': [drug.model_dump(mode='json') for drug in target.drugs],
        'companion_doses': companion_coordinates[0], 'combination_present': context_present,
        'original_profile': copy.deepcopy(target.profile)}
    points = [RegimenPoint(level_id=point.level_id,
                           doses={chosen: point.doses[chosen], CONTEXT_ID: None if context_present else 0.})
              for point in target.points]
    values = target.model_dump(mode='json')
    values.update(drugs=[focal.model_dump(), context.model_dump()], points=[point.model_dump() for point in points],
                  profile=profile, mapping_overrides={})
    return TargetRegimen.model_validate(values)


def _context_exposure(companions):
    original = [exposure.model_dump(mode='json') for exposure in companions]
    source = {'projection_version': PROJECTION_VERSION, 'original_companion_exposures': original}
    if any(exposure.presence == 'present' for exposure in companions):
        return Exposure(drug_id=CONTEXT_ID, presence='present', source=source)
    uncertain = [exposure for exposure in companions if exposure.presence != 'absent']
    if not uncertain:
        return Exposure(drug_id=CONTEXT_ID, presence='absent', source=source)
    if len(uncertain) > 1:
        raise ValueError('Joint use allocation is unavailable for multiple unknown or mixed companion exposures.')
    exposure = uncertain[0]
    return Exposure(drug_id=CONTEXT_ID, presence=exposure.presence,
                    exposed_n=exposure.exposed_n if exposure.presence == 'mixed' else None, source=source)


def project_scope(scope, focal_source_id, *, exclusions=None):
    """Split fixed companion signatures; report unsupported cohorts explicitly.

    Pass a list as ``exclusions`` to collect cohort exclusion records even when
    every cohort is unsupported. Each retained outcome occurs in one projection.
    """
    scope = scope if isinstance(scope, ModelScope) else ModelScope.model_validate(scope)
    focal = next((drug for drug in scope.drugs if drug.model_drug_id == focal_source_id), None)
    if focal is None or focal.mode != 'numeric':
        raise ValueError('A focal numeric target requires a reported numeric source dose axis.')
    if any(drug.model_drug_id == CONTEXT_ID for drug in scope.drugs):
        raise ValueError('Project the original source scope, not an already projected context.')
    excluded = [] if exclusions is None else exclusions
    groups = defaultdict(list)
    contexts = {}
    cohorts, source_rows = set(), set()
    for observation in scope.observations:
        if observation.cohort_id in cohorts or source_rows.intersection(observation.source_row_ids):
            raise ValueError('Source cohorts overlap; projection cannot count the same patients twice.')
        cohorts.add(observation.cohort_id)
        source_rows.update(observation.source_row_ids)
        focal_exposure = next(e for e in observation.exposures if e.drug_id == focal_source_id)
        companions = [e for e in observation.exposures if e.drug_id != focal_source_id]
        try:
            context = _context_exposure(companions)
            if context.presence == 'mixed' and (focal_exposure.dose_subgroups or focal_exposure.presence in {'mixed', 'unknown'}):
                raise ValueError('Focal dose/use and mixed companion exposure require a reported joint patient allocation.')
            if focal_exposure.presence == 'mixed' and context.presence == 'unknown':
                raise ValueError('Mixed focal exposure and unknown companion use require a reported joint patient allocation.')
        except ValueError as error:
            excluded.append({'document_id': scope.document_id, 'scope_id': scope.scope_id,
                             'observation_id': observation.observation_id, 'cohort_id': observation.cohort_id,
                             'reason': str(error)})
            continue
        signature = [{key: value for key, value in exposure.model_dump(mode='json').items()
                      if key not in {'source', 'background_id'}} for exposure in companions]
        key = fingerprint(signature)
        contexts[key] = signature
        projected = observation.model_copy(deep=True)
        projected.exposures = [focal_exposure.model_copy(deep=True), context]
        projected.provenance = {**copy.deepcopy(observation.provenance), 'single_drug_projection': {
            'original_scope_id': scope.scope_id, 'original_observation_id': observation.observation_id,
            'original_exposures': [e.model_dump(mode='json') for e in observation.exposures],
            'whole_regimen_outcomes_retained': True}}
        groups[key].append(projected)
    if not scope.observations:
        key = fingerprint([])
        contexts[key] = []
        groups[key] = []
    result = []
    for context_key, observations in groups.items():
        selected = focal.model_copy(deep=True)
        selected.research_priority = 1
        selected.profile = {**selected.profile, 'name': selected.name, 'model_drug_id': focal_source_id}
        projection = {'version': PROJECTION_VERSION, 'original_scope_id': scope.scope_id,
                      'focal_source_id': focal_source_id, 'context_signature': contexts[context_key],
                      'original_drugs': [drug.model_dump(mode='json') for drug in scope.drugs],
                      'original_profile': copy.deepcopy(scope.profile)}
        identity = fingerprint(projection)[:20]
        result.append(ModelScope(document_id=scope.document_id, scope_id=f'{scope.scope_id}__focal_{identity}',
            label=f'{scope.label} · {selected.name} with fixed background', drugs=[selected, _context_slot()],
            observations=observations, profile=_project_profile(scope.profile, selected.profile),
            lineage={**copy.deepcopy(scope.lineage), 'original_scope_id': scope.scope_id,
                     'focal_source_id': focal_source_id, 'single_drug_projection': projection},
            exclusions=[*copy.deepcopy(scope.exclusions), *copy.deepcopy(excluded)]))
    return result
