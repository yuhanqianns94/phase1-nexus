"""Experimental allocation of monotherapy evidence in target research order.

This module does not change production matching or fit any model. Reservations
belong to an article's drug entity, not a posterior-cache slot. A reservation
therefore persists across that article's clinical scopes and documented aliases.
"""
from collections import defaultdict
import math

from phase1_prediction.profile_similarity import (
    compare_mapped_profiles, drug_correspondence, drug_identity,
)
from .contracts import fingerprint
from .mapping import mapped_relative_dose, target_profile
from .workflow import phase_status, relevance_multiplier


def _drug_profile(drug):
    return {**drug.profile, 'name': drug.name, 'research_role': drug.research_role}


def _article_entities(scopes):
    """Cluster definite source identity only; biological similarity is not identity."""
    rows = [(scope, scope.drugs[0]) for scope in scopes]
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, (left, a) in enumerate(rows):
        for j in range(i):
            right, b = rows[j]
            if left.document_id == right.document_id and (
                a.model_drug_id == b.model_drug_id
                or drug_identity(_drug_profile(a), _drug_profile(b)) == 1
            ):
                parents[root(i)] = root(j)
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[root(i)].append(row)
    by_scope, groups = {}, []
    for members in grouped.values():
        identities = sorted((s.scope_id, d.model_drug_id) for s, d in members)
        document = members[0][0].document_id
        entity = 'article_entity_' + fingerprint([document, identities])[:20]
        for scope, _ in members:
            by_scope[scope.scope_id] = entity
        groups.append({'article_entity_id': entity, 'document_id': document,
                       'members': [{'scope_id': s, 'source_id': d} for s, d in identities]})
    return by_scope, sorted(groups, key=lambda x: (x['document_id'], x['article_entity_id']))


def allocate_primary_first(scopes, target, weights=None, relevance_decisions=None,
                          top_articles_per_drug=None, require_usable_coordinates=False):
    """Return a deterministic, JSON-serializable allocation audit without mutation.

    All retained eligible scopes for the primary target are reserved before the
    next target is considered. An optional article limit is applied separately
    for each target. Explicit mappings reserve their article entity first and
    are never silently displaced by automatic primary-first allocation. Clinical
    previews need no dose points; curve generation can require usable coordinates
    before making automatic reservations.
    """
    if top_articles_per_drug is not None and (
        isinstance(top_articles_per_drug, bool)
        or not isinstance(top_articles_per_drug, int) or top_articles_per_drug < 1
    ):
        raise ValueError('top_articles_per_drug must be a positive integer or null')
    if require_usable_coordinates and not getattr(target, 'points', None):
        raise ValueError('Usable-coordinate allocation requires target dose points')
    scopes = sorted(scopes, key=lambda s: (s.document_id, s.scope_id))
    by_id = {s.scope_id: s for s in scopes}
    if len(by_id) != len(scopes):
        raise ValueError('Source scope IDs must be unique')
    target_ids = {d.drug_id for d in target.drugs}
    if len(target_ids) != len(target.drugs) or not target_ids:
        raise ValueError('Unique target drug IDs are required')
    overrides = target.mapping_overrides
    if set(overrides) - set(by_id):
        raise ValueError('Manual mapping refers to a missing source scope')
    manual = {}
    for scope_id, pairs in overrides.items():
        source_ids = {d.model_drug_id for d in by_id[scope_id].drugs}
        if (not set(pairs) <= target_ids or not set(pairs.values()) <= source_ids
                or len(set(pairs.values())) != len(pairs)):
            raise ValueError('Invalid or conflicting manual source mapping')
        manual.update({(scope_id, sid): tid for tid, sid in pairs.items()})

    rank = {'primary': 0, 'co_primary': 1}
    drugs = sorted(target.drugs, key=lambda d: (
        rank.get(d.research_role, 2), d.research_priority, d.drug_id))
    result = {'format': 'primary_first_allocation_v1', 'experimental': True,
              'scope_policy': 'monotherapy_only',
              'priority_order': [d.drug_id for d in drugs],
              'allocations': [], 'exclusions': [], 'ambiguities': [],
              'top_articles_per_drug': top_articles_per_drug,
              'require_usable_coordinates': require_usable_coordinates}
    usable, review = [], {}
    for scope in scopes:
        phase = phase_status(scope.profile)
        multiplier = relevance_multiplier(scope.document_id, relevance_decisions)
        reason = ('not_phase_i' if phase != 'phase_i' else
                  'review_excluded' if multiplier == 0 else
                  'combination_scope_not_monotherapy' if len(scope.drugs) != 1 else
                  'no_observed_outcomes' if not scope.observations else
                  'no_present_source_exposure' if not any(
                      e.presence != 'absent' for o in scope.observations for e in o.exposures) else None)
        if reason:
            if overrides.get(scope.scope_id):
                raise ValueError(f'Manual mapping uses an ineligible source scope: {scope.scope_id} ({reason})')
            result['exclusions'].append({'document_id': scope.document_id,
                                         'scope_id': scope.scope_id, 'reason': reason})
        else:
            usable.append(scope)
            review[scope.document_id] = multiplier
    entities, result['entity_groups'] = _article_entities(usable)
    reservations = {}
    for scope in usable:
        owner = manual.get((scope.scope_id, scope.drugs[0].model_drug_id))
        if owner is not None:
            entity = entities[scope.scope_id]
            if entity in reservations and reservations[entity] != owner:
                raise ValueError('Conflicting manual mappings: one article drug entity cannot support two target drugs')
            reservations[entity] = owner

    profile = target_profile(target)
    candidates = defaultdict(list)
    for drug in drugs:
        for scope in usable:
            source = scope.drugs[0]
            entity = entities[scope.scope_id]
            forced = reservations.get(entity) == drug.drug_id
            evidence = drug_correspondence(_drug_profile(drug), _drug_profile(source),
                                           target.minimum_drug_similarity)
            if not forced and not evidence['eligible']:
                result['exclusions'].append({'target_id': drug.drug_id, 'scope_id': scope.scope_id,
                    'document_id': scope.document_id, 'source_id': source.model_drug_id,
                    'reason': 'insufficient_drug_similarity', 'drug_evidence': evidence})
                continue
            if require_usable_coordinates:
                try:
                    for point in target.points:
                        mapped_relative_dose(drug, source, point.doses[drug.drug_id])
                except ValueError as error:
                    if forced:
                        raise ValueError(f'Manual mapping has unusable coordinates: {drug.drug_id} / {scope.scope_id}: {error}') from error
                    result['exclusions'].append({'target_id': drug.drug_id, 'scope_id': scope.scope_id,
                        'document_id': scope.document_id, 'source_id': source.model_drug_id,
                        'reason': 'unusable_target_coordinates', 'detail': str(error)})
                    continue
            score = compare_mapped_profiles(profile, scope.profile,
                {drug.drug_id: source.model_drug_id}, weights=weights)
            similarity = float(score['similarity'] or 0.)
            strength = similarity * review[scope.document_id]
            if not math.isfinite(strength) or strength <= 0:
                if forced:
                    raise ValueError('Manual mapping has no positive comparison support')
                result['exclusions'].append({'target_id': drug.drug_id, 'scope_id': scope.scope_id,
                    'document_id': scope.document_id, 'reason': 'no_positive_comparison_support'})
                continue
            candidates[drug.drug_id].append({'target_id': drug.drug_id,
                'document_id': scope.document_id, 'scope_id': scope.scope_id,
                'source_id': source.model_drug_id, 'source_index': 0,
                'article_entity_id': entity, 'similarity': similarity,
                'strength': strength, 'review_multiplier': review[scope.document_id],
                'drug_evidence': evidence, 'manual_override': forced})

    # Report target competition even though research order makes the outcome deterministic.
    competition = defaultdict(set)
    for tid, rows in candidates.items():
        for row in rows:
            competition[row['article_entity_id']].add(tid)
    result['ambiguities'] = [{'article_entity_id': entity, 'eligible_target_ids': sorted(tids),
                              'resolution': 'manual_reservation_then_primary_first'}
                             for entity, tids in sorted(competition.items()) if len(tids) > 1]
    for drug in drugs:
        available = []
        for row in candidates[drug.drug_id]:
            owner = reservations.get(row['article_entity_id'])
            if owner is not None and owner != drug.drug_id:
                result['exclusions'].append({**row, 'reason': 'article_entity_reserved',
                                             'reserved_for': owner})
            else:
                available.append(row)
        article_scores = {}
        for row in available:
            doc = row['document_id']
            article_scores[doc] = max(article_scores.get(doc, 0.), row['strength'])
        docs = sorted(article_scores, key=lambda doc: (-article_scores[doc], doc))
        forced_docs = {row['document_id'] for row in available if row['manual_override']}
        if top_articles_per_drug is not None:
            if len(forced_docs) > top_articles_per_drug:
                raise ValueError('The article limit is smaller than the explicit manual source selection')
            ranked = [doc for doc in docs if doc not in forced_docs]
            selected = forced_docs | set(ranked[:top_articles_per_drug-len(forced_docs)])
        else:
            selected = set(docs)
        for row in sorted(available, key=lambda x: (-x['strength'], x['document_id'], x['scope_id'])):
            if row['document_id'] not in selected:
                result['exclusions'].append({**row, 'reason': 'outside_per_drug_article_limit'})
                continue
            reservations[row['article_entity_id']] = drug.drug_id
            result['allocations'].append(row)
        if not any(row['target_id'] == drug.drug_id for row in result['allocations']):
            result['exclusions'].append({'target_id': drug.drug_id,
                'reason': 'no_support_after_primary_reservations'})
    supported = {row['target_id'] for row in result['allocations']}
    result['unsupported_target_ids'] = [d.drug_id for d in drugs if d.drug_id not in supported]
    result['complete'] = not result['unsupported_target_ids']
    result['allocation_id'] = fingerprint(result)
    return result
