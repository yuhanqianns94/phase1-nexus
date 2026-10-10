"""Experimental article-entity reservations and deterministic primary-first evidence."""
import json
from types import SimpleNamespace

import pytest

from phase1_prediction.multidrug.contracts import Exposure, Observation
from phase1_prediction.multidrug.mapping import match_target
from phase1_prediction.multidrug.primary_first import allocate_primary_first
from test_multidrug_transport import fixture


def mono(document='paper', scope_id='mono', name='B', source_id='source', aliases=()):
    scope, _, _, target = fixture(2)
    scope.document_id, scope.scope_id = document, scope_id
    scope.drugs = scope.drugs[:1]
    source = scope.drugs[0]
    source.name, source.model_drug_id = name, source_id
    source.profile = {**source.profile, 'name': name, 'model_drug_id': source_id,
                      'aliases': list(aliases)}
    scope.profile['configurations'][0]['drugs'] = [source.profile]
    scope.observations = [Observation(observation_id='obs', cohort_id='cohort', y=2, n=10,
        exposures=[Exposure(drug_id=source_id, presence='present', dose=100.)])]
    return scope, target


def test_primary_first_can_differ_from_global_assignment_and_exposes_starvation():
    scope, target = mono()
    assert match_target(target, scope)['pairs'] == {'B': 'source'}
    saved = scope.model_dump(), target.model_dump()
    result = allocate_primary_first([scope], target)
    assert [(r['target_id'], r['source_id']) for r in result['allocations']] == [('A', 'source')]
    assert result['unsupported_target_ids'] == ['B'] and not result['complete']
    assert result['ambiguities'][0]['eligible_target_ids'] == ['A', 'B']
    assert any(e.get('reserved_for') == 'A' for e in result['exclusions'])
    assert (scope.model_dump(), target.model_dump()) == saved
    json.dumps(result, allow_nan=False)


def test_entity_reservation_spans_scope_ids_and_explicit_aliases():
    first, target = mono(scope_id='first', name='drug-X', source_id='cmp_one', aliases=['code-123'])
    second, _ = mono(scope_id='second', name='code-123', source_id='cmp_two')
    # Manual B assignment must reserve both clinically distinct scopes of X.
    target.mapping_overrides = {'first': {'B': 'cmp_one'}}
    result = allocate_primary_first([first, second], target)
    rows = result['allocations']
    assert {r['target_id'] for r in rows} == {'B'}
    assert {r['scope_id'] for r in rows} == {'first', 'second'}
    assert len({r['article_entity_id'] for r in rows}) == 1
    assert len(result['entity_groups']) == 1
    assert result['unsupported_target_ids'] == ['A']


@pytest.mark.parametrize('same_id', [False, True])
def test_conflicting_manual_assignments_across_scopes_fail(same_id):
    first, target = mono(scope_id='one', name='same drug', source_id='cmp1')
    second, _ = mono(scope_id='two', name='different name' if same_id else 'same drug',
                     source_id='cmp1' if same_id else 'cmp2')
    target.mapping_overrides = {'one': {'A': 'cmp1'}, 'two': {'B': second.drugs[0].model_drug_id}}
    with pytest.raises(ValueError, match='one article drug entity'):
        allocate_primary_first([first, second], target)


def test_same_drug_in_different_articles_can_support_distinct_targets_with_explicit_assignments():
    first, target = mono(document='first', scope_id='one')
    second, _ = mono(document='second', scope_id='two')
    target.mapping_overrides = {'one': {'A': 'source'}, 'two': {'B': 'source'}}
    result = allocate_primary_first([first, second], target)
    assert result['complete']
    assert {(r['target_id'], r['document_id']) for r in result['allocations']} == {('A', 'first'), ('B', 'second')}
    assert len(result['entity_groups']) == 2


def test_top_n_leaves_unselected_articles_available_for_secondary_and_is_order_invariant():
    first, target = mono(document='a', scope_id='one', name='X')
    second, _ = mono(document='b', scope_id='two', name='X')
    first.drugs[0].research_role = second.drugs[0].research_role = 'background'
    # Equal article scores are resolved deterministically by document ID.
    result = allocate_primary_first([second, first], target, top_articles_per_drug=1)
    assert [(r['target_id'], r['document_id']) for r in result['allocations']] == [('A', 'a'), ('B', 'b')]
    assert result == allocate_primary_first([first, second], target, top_articles_per_drug=1)


def test_manual_mapping_has_priority_and_cannot_be_silently_truncated_by_article_limit():
    first, target = mono(document='a', scope_id='one')
    second, _ = mono(document='b', scope_id='two')
    target.mapping_overrides = {'one': {'B': 'source'}, 'two': {'B': 'source'}}
    with pytest.raises(ValueError, match='article limit'):
        allocate_primary_first([first, second], target, top_articles_per_drug=1)


def test_combination_and_phase_ii_do_not_become_monotherapy_evidence():
    combined, _, _, target = fixture(2)
    phase2, _ = mono(document='phase2', scope_id='phase2')
    phase2.profile['configurations'][0]['phase'] = 'Phase II'
    empty, _ = mono(document='empty', scope_id='empty')
    empty.observations = []
    excluded, _ = mono(document='review', scope_id='review')
    result = allocate_primary_first([combined, phase2, empty, excluded], target,
        relevance_decisions={'paper': 'Include', 'phase2': 'Include', 'empty': 'Include', 'review': 'Exclude'})
    assert not result['allocations']
    reasons = {r['scope_id']: r['reason'] for r in result['exclusions'] if 'scope_id' in r}
    assert reasons == {'scope': 'combination_scope_not_monotherapy', 'phase2': 'not_phase_i',
                       'empty': 'no_observed_outcomes', 'review': 'review_excluded'}


def test_explicit_primary_role_precedes_array_position_and_source_role_is_not_a_gate():
    scope, target = mono()
    target.drugs[0].research_role = 'partner'
    target.drugs[1].research_role = 'primary'
    scope.drugs[0].research_role = 'background'
    result = allocate_primary_first([scope], target)
    assert result['priority_order'] == ['B', 'A']
    assert result['allocations'][0]['target_id'] == 'B'


def test_missing_drug_features_cannot_be_replaced_by_similar_roles_or_context():
    scope, target = mono(name='unrelated name')
    scope.drugs[0].profile = {'name': 'unrelated name', 'model_drug_id': 'source'}
    scope.profile['configurations'][0]['drugs'] = [scope.drugs[0].profile]
    result = allocate_primary_first([scope], target)
    assert not result['allocations']
    assert sum(e['reason'] == 'insufficient_drug_similarity' for e in result['exclusions']) == 2


def test_review_multiplier_is_applied_once_to_candidate_strength():
    scope, target = mono()
    included = allocate_primary_first([scope], target, relevance_decisions={'paper': 'Include'})
    uncertain = allocate_primary_first([scope], target, relevance_decisions={'paper': 'Uncertain'})
    assert uncertain['allocations'][0]['strength'] == pytest.approx(included['allocations'][0]['strength']/2)
    assert uncertain['allocations'][0]['similarity'] == included['allocations'][0]['similarity']


def test_unusable_primary_coordinates_do_not_reserve_evidence_from_usable_partner():
    scope, target = mono()
    scope.drugs[0].mode = 'background'
    scope.drugs[0].reference_dose = None
    scope.drugs[0].unit = None
    target.drugs[1].mode = 'background'
    target.drugs[1].reference_dose = None
    for point in target.points:
        point.doses['B'] = None
    clinical = allocate_primary_first([scope], target)
    assert clinical['allocations'][0]['target_id'] == 'A'
    usable = allocate_primary_first([scope], target, require_usable_coordinates=True)
    assert usable['allocations'][0]['target_id'] == 'B'
    assert usable['unsupported_target_ids'] == ['A']
    assert any(row.get('target_id') == 'A' and row['reason'] == 'unusable_target_coordinates'
               for row in usable['exclusions'])
    target.mapping_overrides = {scope.scope_id: {'A': 'source'}}
    with pytest.raises(ValueError, match='Manual mapping has unusable coordinates'):
        allocate_primary_first([scope], target, require_usable_coordinates=True)


def test_clinical_preview_needs_no_target_points():
    scope, target = mono()
    preview = SimpleNamespace(drugs=target.drugs, profile=target.profile,
        mapping_overrides=target.mapping_overrides,
        minimum_drug_similarity=target.minimum_drug_similarity)
    assert allocate_primary_first([scope], preview)['allocations']
    with pytest.raises(ValueError, match='requires target dose points'):
        allocate_primary_first([scope], preview, require_usable_coordinates=True)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_article_limit_is_rejected(value):
    scope, target = mono()
    with pytest.raises(ValueError, match='positive integer'):
        allocate_primary_first([scope], target, top_articles_per_drug=value)
