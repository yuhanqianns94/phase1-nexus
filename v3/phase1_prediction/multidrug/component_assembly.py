"""Assemble frozen component posteriors, then average or mix whole curves.

Similarity chooses sources, never interaction coefficients. Within an output
draw, every use of a source fit shares its joint draw and every missing target
interaction shares one prior draw, including across Method 1 recipes.
"""
from collections import Counter, defaultdict
from itertools import product
import math

import numpy as np
from scipy.special import expit, logit

from phase1_prediction.profile_similarity import compare_mapped_profiles
from .contracts import PriorSpec, fingerprint
from .mapping import match_target, mapped_relative_dose, target_profile
from .math import interaction_sets
from .transport import _base_result, _summary, flat_samples
from .workflow import relevance_multiplier

# Larger recipe spaces use a reproducible, explicitly reported quadrature bank.
# This bounds work independently of the Cartesian size; it is not new evidence.
AVERAGE_RECIPE_LIMIT = 256


def _identity(candidate):
    return candidate['manifest']['source_fit_id'], candidate['source_index']


def _component_pools(entries, target, weights, decisions, result):
    profile = target_profile(target)
    mapped = []
    for scope, arrays, manifest in entries:
        if relevance_multiplier(scope.document_id, decisions) == 0:
            continue
        if manifest['backend'] != result['backend'] or manifest['status'] != 'fitted':
            result['excluded'].append({'source_fit_id': manifest['source_fit_id'],
                'reason': 'backend_mismatch' if manifest['backend'] != result['backend'] else manifest['status']})
            continue
        mapped.append((scope, arrays, manifest, match_target(target, scope, weights=weights)))
    pools = []
    for td in target.drugs:
        candidates = []
        for scope, arrays, manifest, mapping in mapped:
            for j, sd in enumerate(scope.drugs):
                # Keep preview and manual assignments as actual constraints.
                if mapping['pairs'].get(td.drug_id) != sd.model_drug_id:
                    continue
                try:
                    r = np.array([mapped_relative_dose(td, sd, p.doses[td.drug_id]) for p in target.points])
                except ValueError as error:
                    result['excluded'].append({'target_drug': td.drug_id,
                        'source_fit_id': manifest['source_fit_id'], 'reason': str(error)})
                    continue
                score = compare_mapped_profiles(profile, scope.profile,
                    {td.drug_id: sd.model_drug_id}, weights=weights)
                strength = score['similarity'] or 0.
                if strength > 0:
                    candidates.append(dict(scope=scope, arrays=arrays, manifest=manifest,
                        source_index=j, r=r, strength=strength, score=score, mapping=mapping))
        if not candidates:
            result['excluded'].append({'target_drug': td.drug_id, 'reason': 'no_mapped_component_evidence'})
            return []
        groups = defaultdict(list)
        for c in candidates:
            groups[c['scope'].document_id].append(c)
        budgets = {doc: max(c['strength'] for c in cs) * relevance_multiplier(doc, decisions)
                   for doc, cs in groups.items()}
        for doc, cs in groups.items():
            local_total = sum(c['strength'] for c in cs)
            for c in cs:
                c['weight'] = budgets[doc] / sum(budgets.values()) * c['strength'] / local_total
        pools.append(candidates)
    contributors = {}
    result['component_pools'] = []
    for td, pool in zip(target.drugs, pools):
        rows = []
        for c in pool:
            fit, j = _identity(c)
            sd = c['scope'].drugs[j]
            rows.append({'source_fit_id': fit, 'document_id': c['scope'].document_id,
                'source_index': j, 'source_id': sd.model_drug_id, 'source_name': sd.name,
                'mapping_id': c['mapping']['mapping_id'], 'candidate_weight': float(c['weight'])})
            entry = contributors.setdefault(fit, {'source_fit_id': fit,
                'document_id': c['scope'].document_id, 'mapping': c['mapping'],
                'review_multiplier': relevance_multiplier(c['scope'].document_id, decisions),
                'component_weights': []})
            entry['component_weights'].append({'target_id': td.drug_id,
                'source_id': sd.model_drug_id, 'weight': float(c['weight'])})
        result['component_pools'].append({'target_id': td.drug_id, 'candidates': rows})
    result['contributors'] = list(contributors.values())
    return pools


def _valid_recipe(pools, indices):
    slots = [_identity(pool[i]) for pool, i in zip(pools, indices)]
    return len(set(slots)) == len(slots)


def _sample_recipes(pools, count, rng):
    probabilities = [np.array([c['weight'] for c in pool]) for pool in pools]
    accepted = []
    for _ in range(max(10000, count * 1000)):
        indices = tuple(int(rng.choice(len(pool), p=p)) for pool, p in zip(pools, probabilities))
        if _valid_recipe(pools, indices):
            accepted.append(indices)
            if len(accepted) == count:
                return accepted
    raise ValueError('No feasible injective assembly recipe within the sampling limit; source components cannot be reused as different target drugs')


def _average_recipes(pools, rng):
    space = math.prod(len(p) for p in pools)
    if space <= AVERAGE_RECIPE_LIMIT:
        recipes = [(indices, math.prod(pool[i]['weight'] for pool, i in zip(pools, indices)))
                   for indices in product(*(range(len(p)) for p in pools)) if _valid_recipe(pools, indices)]
        if not recipes:
            raise ValueError('No feasible injective assembly recipe')
        total = sum(w for _, w in recipes)
        bank = [(indices, w / total) for indices, w in recipes]
        evaluation = 'exact_recipe_average'
        samples = None
    else:
        counts = Counter(_sample_recipes(pools, AVERAGE_RECIPE_LIMIT, rng))
        bank = [(indices, n / AVERAGE_RECIPE_LIMIT) for indices, n in sorted(counts.items())]
        evaluation = 'monte_carlo_recipe_average'
        samples = AVERAGE_RECIPE_LIMIT
    return bank, {'evaluation': evaluation, 'cartesian_recipe_count': space,
                  'evaluated_recipe_count': len(bank), 'recipe_sample_count': samples,
                  'exact_recipe_limit': AVERAGE_RECIPE_LIMIT}


def _draw_state(pools, target, prior, count, rng):
    sources = {}
    for td, pool in zip(target.drugs, pools):
        for c in pool:
            fit, j = _identity(c)
            if fit not in sources:
                flat = flat_samples(c['arrays'])
                sources[fit] = {'flat': flat, 'ids': rng.integers(len(flat['a']), size=count)}
    interactions = {g: rng.normal(0, prior.pair_sd, count)
                    for g in interaction_sets(len(target.drugs))}
    return dict(sources=sources, interactions=interactions)


def _source_interaction(chosen, group):
    slots = [_identity(chosen[i]) for i in group]
    source_groups = interaction_sets(len(chosen[group[0]]['scope'].drugs))
    source_group = tuple(sorted(j for _, j in slots))
    if len({fit for fit, _ in slots}) == 1 and source_group in source_groups:
        return slots[0][0], source_groups.index(source_group)
    return None


def _basis(r, group):
    present = np.all(r[:, group] > 0, axis=1)
    return np.where(present, 2 * expit(np.log(np.maximum(r[:, group], 1e-300)).sum(axis=1)), 0.)


def _evaluate(chosen, target, state, outputs):
    """Vectorized output draws; memory does not scale with the recipe count."""
    r = np.array([c['r'] for c in chosen]).T
    log_survival = np.zeros((len(outputs), len(target.points)))
    for td, c in zip(target.drugs, chosen):
        fit, j = _identity(c)
        source = state['sources'][fit]
        ids = source['ids'][outputs]
        a = source['flat']['a'][ids, j]
        b = source['flat']['log_b'][ids, j]
        numeric = c['scope'].drugs[j].mode == 'numeric'
        q = expit(a[:, None] + (np.exp(b[:, None]) * np.log(np.maximum(c['r'], 1e-300)) if numeric else 0.))
        q = np.where(c['r'] > 0, q, 0.)
        log_survival += np.log1p(-np.clip(q, 0, 1-1e-15))
    p0 = -np.expm1(log_survival)
    shift = np.zeros_like(p0)
    source_r = r.copy()
    for i, c in enumerate(chosen):
        if c['scope'].drugs[c['source_index']].mode == 'background':
            source_r[:, i] = (r[:, i] > 0).astype(float)
    target_r = np.array([[0. if p.doses[d.drug_id] == 0 else
        (p.doses[d.drug_id] / d.reference_dose if d.mode == 'numeric' and p.doses[d.drug_id] is not None else 1.)
        for d in target.drugs] for p in target.points])
    for group in interaction_sets(len(chosen)):
        inherited = _source_interaction(chosen, group)
        if inherited:
            fit, gi = inherited
            source = state['sources'][fit]
            eta = source['flat']['eta'][source['ids'][outputs], gi]
        else:
            eta = state['interactions'][group][outputs]
        shift += eta[:, None] * _basis(source_r if inherited else target_r, group)
    values = expit(logit(np.clip(p0, 1e-15, 1-1e-15)) + shift)
    values[:, np.all(r == 0, axis=1)] = 0.
    return values


def _recipe_trace(chosen, target, state, m):
    components, interactions = [], []
    for td, c in zip(target.drugs, chosen):
        fit, j = _identity(c)
        sd = c['scope'].drugs[j]
        source = state['sources'][fit]
        draw = int(source['ids'][m])
        components.append({'target_id': td.drug_id, 'source_fit_id': fit, 'source_index': j,
            'source_id': sd.model_drug_id, 'mapping_id': c['mapping']['mapping_id'],
            'flat_draw': draw, 'candidate_weight': c['weight'],
            'log_slope': float(source['flat']['log_b'][draw, j]),
            'target_role': td.research_role, 'source_role': sd.research_role,
            'role_transfer': td.research_role != sd.research_role, 'dose_mapping': td.transport_mode,
            'source_reference_dose': sd.reference_dose, 'target_reference_dose': td.reference_dose})
    for group in interaction_sets(len(chosen)):
        inherited = _source_interaction(chosen, group)
        if inherited:
            fit, gi = inherited
            source = state['sources'][fit]
            value = source['flat']['eta'][source['ids'][m], gi]
        else:
            value = state['interactions'][group][m]
        interactions.append({'target_indices': list(group),
            'basis': 'source_joint_draw' if inherited else 'fixed_prior', 'value': float(value)})
    return dict(components=components, interactions=interactions)


def _average_trace(state, m):
    # Recipe definitions live once in recipe_bank; avoid repeating thousands of
    # identical component descriptions for every posterior draw.
    return {'source_draws': [{'source_fit_id': fit, 'flat_draw': int(s['ids'][m])}
                             for fit, s in state['sources'].items()],
            'prior_interactions': [{'target_indices': list(g), 'value': float(v[m])}
                                   for g, v in state['interactions'].items()],
            'anchor_log_slopes': []}


def assemble_posteriors(entries, target, *, aggregation='mixture', backend='logistic',
                        draw_count=1000, seed=20260919, prior=None, weights=None, relevance_decisions=None,
                        pair_prior_scenario=None):
    if aggregation not in {'mixture', 'average'}:
        raise ValueError('Unknown component curve aggregation')
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 1:
        raise ValueError('draw_count must be positive')
    prior, rng = prior or PriorSpec(), np.random.default_rng(seed)
    synthesis = 'cross_article_component_assembly' + ('_average' if aggregation == 'average' else '')
    result = _base_result(target, backend, synthesis, seed)
    if backend != 'logistic':
        raise ValueError('Component assembly requires Component Logistic. Product-interaction Logistic has one shared intercept and no defined cross-article intercept assembly rule; choose complete-regimen borrowing for this model.')
    if pair_prior_scenario is not None:
        from .interaction_pilot import PairPriorScenario
        pair_prior_scenario = PairPriorScenario.model_validate(pair_prior_scenario)
        if len(target.drugs) != 2 or any(len(scope.drugs) != 1 for scope, _, _ in entries):
            raise ValueError('A pair-prior scenario requires two targets and monotherapy source fits; it cannot replace an inherited clinical interaction posterior.')
        result['pair_prior_scenario'] = pair_prior_scenario.model_dump(mode='json')
    result.update(evidence_construction='component_assembly', aggregation=aggregation)
    pools = _component_pools(entries, target, weights, relevance_decisions, result)
    if not pools:
        return result
    if aggregation == 'average':
        bank, result['assembly_computation'] = _average_recipes(pools, rng)
        result['recipe_bank'] = [{'candidate_indices': list(indices), 'weight': float(w)} for indices, w in bank]
        destinations = {indices: np.arange(draw_count) for indices, _ in bank}
    else:
        selected = _sample_recipes(pools, draw_count, rng)
        grouped = defaultdict(list)
        for m, indices in enumerate(selected):
            grouped[indices].append(m)
        destinations = {indices: np.array(outputs) for indices, outputs in grouped.items()}
        bank = [(indices, len(outputs) / draw_count) for indices, outputs in destinations.items()]
    state = _draw_state(pools, target, prior, draw_count, rng)
    if pair_prior_scenario is not None:
        # Transform the existing standard-normal draws, keeping exactly the same
        # source draws/recipes across sensitivity settings and the entire grid.
        state['interactions'][(0, 1)] = (pair_prior_scenario.mean +
            state['interactions'][(0, 1)] / prior.pair_sd * pair_prior_scenario.sd)
    matrix = np.zeros((draw_count, len(target.points)))
    provenance = [None] * draw_count
    frequencies = defaultdict(float)
    support = {g: 0. for g in interaction_sets(len(target.drugs))}
    for indices, weight in bank:
        chosen = [p[i] for p, i in zip(pools, indices)]
        outputs = destinations[indices]
        # Bound temporary evaluation arrays even on dense two-dimensional grids.
        for offset in range(0, len(outputs), 128):
            chunk = outputs[offset:offset+128]
            values = _evaluate(chosen, target, state, chunk)
            if aggregation == 'average':
                matrix[chunk] += weight * values
            else:
                matrix[chunk] = values
        for td, c in zip(target.drugs, chosen):
            frequencies[(td.drug_id, *_identity(c))] += weight
        for group in support:
            if _source_interaction(chosen, group):
                support[group] += weight
        if aggregation == 'mixture':
            for m in outputs:
                provenance[m] = _recipe_trace(chosen, target, state, m)
    if aggregation == 'average':
        provenance = [_average_trace(state, m) for m in range(draw_count)]
    result.update(status='fitted', posterior_draws=matrix.tolist(), summary=_summary(matrix, target),
        provenance=provenance, recipe_distribution='product_candidate_weights_conditioned_on_joint_injectivity',
        selection_frequencies=[{'target_id': t, 'source_fit_id': fit, 'source_index': j, 'frequency': float(w)}
                               for (t, fit, j), w in sorted(frequencies.items())],
        interaction_support=[{'target_ids': [target.drugs[j].drug_id for j in g],
            'source_joint_draw_fraction': float(np.clip(w, 0., 1.)),
            'fixed_prior_fraction': float(np.clip(1-w, 0., 1.))} for g, w in support.items()],
        aggregation_note=('Draw-wise weighted average of complete assembled curves. Shared source and missing-interaction draws are reused across recipes; intervals describe this average.'
            if aggregation == 'average' else 'One complete assembled curve per output draw. Source choices and interaction draws stay fixed across the entire dose grid.'))
    result['prediction_id'] = fingerprint(result)
    return result
