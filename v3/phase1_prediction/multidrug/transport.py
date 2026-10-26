"""Transport frozen joint draws. Never re-fit an article with other data."""
from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from .contracts import PriorSpec, fingerprint
from . import TARGET_PREDICTION_VERSION, validate_backend
from .workflow import relevance_multiplier
from .mapping import match_target, mapped_relative_dose
from .math import component_risks, logistic_risk


def flat_samples(arrays):
    return {k: v.reshape((v.shape[0] * v.shape[1], *v.shape[2:])) for k, v in arrays.items()}


def _summary(matrix, target):
    return [{'level_id': p.level_id, 'doses': p.doses,
             'posterior_mean': float(np.mean(matrix[:, i])),
             'posterior_median': float(np.median(matrix[:, i])),
             'low_95': float(np.quantile(matrix[:, i], .025)),
             'high_95': float(np.quantile(matrix[:, i], .975)),
             'probability_above_target': float(np.mean(matrix[:, i] > target.target_dlt))}
            for i, p in enumerate(target.points)]


def _mapped_r(target, scope, mapping, backend='logistic'):
    indices = {d.model_drug_id: i for i, d in enumerate(scope.drugs)}
    matrix = np.zeros((len(target.points), len(scope.drugs)))
    for td in target.drugs:
        j = indices[mapping['pairs'][td.drug_id]]
        sd = scope.drugs[j]
        for pi, p in enumerate(target.points):
            matrix[pi, j] = mapped_relative_dose(td, sd, p.doses[td.drug_id])
    return matrix, []


def _evaluate(scope, arrays, draw_ids, target, mapping, backend, prior, rng):
    validate_backend(backend)
    flat = flat_samples(arrays)
    log_b = flat['log_b'][draw_ids].copy()
    r, _ = _mapped_r(target, scope, mapping, backend)
    if backend == 'single_drug_logistic':
        from .math import single_drug_logistic_risk
        values = jax.jit(jax.vmap(single_drug_logistic_risk, in_axes=(0, 0, None, 0, None)))(
            flat['alpha'][draw_ids], log_b[:, 0], jnp.asarray(r[:, 0]),
            flat['delta'][draw_ids], jnp.asarray(r[:, 1] > 0))
        matrix = np.asarray(values)
        if not np.isfinite(matrix).all():
            raise ValueError('Single-drug Logistic produced non-finite risks.')
        return matrix, {}
    if backend == 'logistic_product':
        from .math import product_logistic_risk
        values = jax.jit(jax.vmap(product_logistic_risk, in_axes=(0, 0, None, 0)))(
            flat['alpha'][draw_ids], jnp.asarray(log_b), jnp.asarray(r), flat['eta'][draw_ids])
        matrix = np.asarray(values)
        if not np.isfinite(matrix).all():
            raise ValueError('Product-interaction Logistic produced non-finite risks at these coordinates. Check dose scales and prior settings; no fallback curve was used.')
        return matrix, {}
    a = flat['a'][draw_ids]
    numeric = np.array([d.mode == 'numeric' for d in scope.drugs])
    def one(a, b, eta, doses, interaction_doses, numeric_mask):
        q = component_risks(a, b, doses, numeric_mask, doses > 0)
        return logistic_risk(q, interaction_doses, eta)
    values = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None, None, None)))(
        a, log_b, flat['eta'][draw_ids], jnp.asarray(r), jnp.asarray(r), jnp.asarray(numeric))
    return np.asarray(values), {}


def _base_result(target, backend, synthesis, seed):
    validate_backend(backend)
    return {'format': 'multidrug_target_prediction_v1', 'prediction_engine_version': TARGET_PREDICTION_VERSION,
            'backend': backend, 'synthesis': synthesis,
            'target': target.model_dump(mode='json'), 'seed': seed,
            'path_level_ids': target.path_level_ids, 'status': 'no_similarity_support',
            'posterior_draws': [], 'summary': [], 'provenance': [], 'contributors': [], 'excluded': []}


def transport_joint_posteriors(entries, target, *, backend='logistic', draw_count=1000, seed=20260919,
                               prior=None, weights=None, threshold=0., top_articles=None,
                               aggregation='mixture', relevance_decisions=None, mappings=None):
    if aggregation not in {'mixture', 'average'}:
        raise ValueError('Unknown whole-regimen posterior aggregation')
    prior = prior or PriorSpec()
    rng = np.random.default_rng(seed)
    result = _base_result(target, backend, 'same_article_joint_posterior_' + aggregation, seed)
    by_doc = defaultdict(list)
    for scope, arrays, manifest in entries:
        if relevance_multiplier(scope.document_id,relevance_decisions)==0:continue
        mapping = ((mappings or {}).get(scope.scope_id) if mappings is not None
                   else match_target(target, scope, weights=weights))
        if mapping is None:
            raise ValueError('Missing frozen focal-drug mapping for ' + scope.scope_id)
        reason = None
        if manifest['backend'] != backend:
            reason = 'backend_mismatch'
        elif manifest['status'] != 'fitted':
            reason = manifest['status']
        elif mapping['unmatched_target_ids']:
            reason = 'incomplete_target_drug_mapping'
        else:
            try:
                _mapped_r(target, scope, mapping, backend)
            except ValueError as error:
                reason = str(error)
        if reason:
            result['excluded'].append({'source_fit_id': manifest['source_fit_id'], 'reason': reason})
            continue
        score = mapping['score']
        similarity = score['similarity'] or 0.
        strength = similarity
        if strength > 0 and similarity >= threshold:
            by_doc[scope.document_id].append((strength, scope, arrays, manifest, mapping))
    # Each article has one budget; the best applicable scope sets its score.
    docs = sorted(by_doc, key=lambda d: (-max(x[0] for x in by_doc[d]), d))
    if top_articles is not None:
        docs = docs[:top_articles]
    if not docs:
        return result
    article_raw = np.array([max(x[0] for x in by_doc[d])*relevance_multiplier(d,relevance_decisions) for d in docs])
    article_weights = article_raw / article_raw.sum()
    candidates, probabilities = [], []
    for doc, budget in zip(docs, article_weights):
        local = by_doc[doc]
        local_weights = np.array([x[0] for x in local]); local_weights /= local_weights.sum()
        for entry, local_weight in zip(local, local_weights):
            candidates.append(entry); probabilities.append(float(budget * local_weight))
            result['contributors'].append({'document_id': doc, 'source_fit_id': entry[3]['source_fit_id'],
                                           'review_multiplier':relevance_multiplier(doc,relevance_decisions), 'article_weight': float(budget), 'scope_weight': float(local_weight),
                                           'weight': float(budget * local_weight), 'mapping': entry[4]})
    if aggregation == 'average':
        # Each article has one budget. Its scopes are alternative source curves,
        # not independent studies; choose one scope per article and whole draw.
        matrix = np.zeros((draw_count, len(target.points)))
        provenance = [{'article_curves': []} for _ in range(draw_count)]
        for doc, budget in zip(docs, article_weights):
            local = by_doc[doc]
            local_weights = np.array([x[0] for x in local]); local_weights /= local_weights.sum()
            selected = rng.choice(len(local), size=draw_count, p=local_weights)
            for index, (_, scope, arrays, manifest, mapping) in enumerate(local):
                dest = np.where(selected == index)[0]
                if not len(dest):
                    continue
                chains, draws = arrays['log_b'].shape[:2]
                ids = rng.integers(0, chains * draws, len(dest))
                values, anchors = _evaluate(scope, arrays, ids, target, mapping, backend,
                                            prior, rng)
                matrix[dest] += float(budget) * values
                for j, out in enumerate(dest):
                    provenance[out]['article_curves'].append({
                        'document_id': doc, 'source_fit_id': manifest['source_fit_id'],
                        'weight': float(budget), 'mapping_id': mapping['mapping_id'],
                        'chain': int(ids[j] // draws), 'draw': int(ids[j] % draws),
                        'anchor_log_slopes': {key: values[j] for key, values in anchors.items()}})
        result.update(status='fitted', posterior_draws=matrix.tolist(), summary=_summary(matrix, target),
                      provenance=provenance,
                      aggregation_note='Draw-wise average across articles; one whole scope curve per article and draw. Fixed article weights across the target grid.')
        result['prediction_id'] = fingerprint(result)
        return result
    selected = rng.choice(len(candidates), size=draw_count, p=probabilities)
    matrix = np.zeros((draw_count, len(target.points)))
    provenance = [None] * draw_count
    for index, (_, scope, arrays, manifest, mapping) in enumerate(candidates):
        dest = np.where(selected == index)[0]
        if not len(dest):
            continue
        chains, draws = arrays['log_b'].shape[:2]
        ids = rng.integers(0, chains * draws, len(dest))
        values, anchors = _evaluate(scope, arrays, ids, target, mapping, backend, prior, rng)
        matrix[dest] = values
        for local, out in enumerate(dest):
            provenance[out] = {'source_fit_id': manifest['source_fit_id'], 'mapping_id': mapping['mapping_id'],
                               'chain': int(ids[local] // draws), 'draw': int(ids[local] % draws),
                               'anchor_log_slopes': {j: v[local] for j, v in anchors.items()}}
    result.update(status='fitted', posterior_draws=matrix.tolist(), summary=_summary(matrix, target), provenance=provenance)
    result['prediction_id'] = fingerprint(result)
    return result


def average_joint_posteriors(entries, target, **kwargs):
    """Average frozen article posterior curves without re-fitting any source."""
    return transport_joint_posteriors(entries, target, aggregation='average', **kwargs)


def assemble_components(entries, target, **kwargs):
    """Method 2: mixture of assembled curves (legacy public mode name)."""
    from .component_assembly import assemble_posteriors
    return assemble_posteriors(entries, target, aggregation='mixture', **kwargs)


def average_assembled_components(entries, target, **kwargs):
    """Method 1: draw-wise weighted average of complete assembled curves."""
    from .component_assembly import assemble_posteriors
    return assemble_posteriors(entries, target, aggregation='average', **kwargs)
