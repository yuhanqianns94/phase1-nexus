"""Cross-role semantic matching and cross-paper assembly use the same sources."""
from types import SimpleNamespace

import numpy as np
import pytest

from phase1_prediction.multidrug.mapping import MATCHING_POLICY, match_target
from phase1_prediction.multidrug.service import preview
from phase1_prediction.multidrug.transport import assemble_components
from test_multidrug_transport import fixture


def profile(name, family, role):
    return {'name': name, 'kind': 'drug', 'model_drug_id': name,
            'role': 'escalated' if role in {'primary', 'co_primary'} else 'combination_partner',
            'mechanisms': [family], 'targets': [family],
            'classes': [family], 'pathways': [family]}


def three_papers(backend='logistic'):
    _, _, _, target = fixture(2, backend)
    for drug, family in zip(target.drugs, ('alpha', 'beta')):
        drug.profile = profile(drug.name, family, drug.research_role)
    entries = []
    # Source model slots stay primary-first, independently of the target mapping.
    for paper, names, families in [('paper1', ('D', 'C'), ('alpha', 'gamma')),
                                    ('paper2', ('F', 'E'), ('beta', 'alpha')),
                                    ('paper3', ('G', 'H'), ('beta', 'delta'))]:
        scope, arrays, manifest, _ = fixture(2, backend)
        scope.document_id = paper
        scope.scope_id = paper + '_scope'
        for drug, name, family in zip(scope.drugs, names, families):
            drug.name = drug.model_drug_id = name
            drug.profile = profile(name, family, drug.research_role)
        scope.profile = {'configurations': [{'phase':'Phase I','drugs': [d.profile for d in scope.drugs]}]}
        manifest['source_fit_id'] = paper + '_fit'
        entries.append((scope, arrays, manifest))
    return entries, target


@pytest.mark.parametrize('role', ['primary', 'co_primary', 'partner'])
def test_first_or_primary_target_can_match_a_source_partner(role):
    entries, target = three_papers()
    scope = entries[1][0]  # F primary, E partner.
    target.drugs = target.drugs[:1]
    target.drugs[0].research_role = role
    for point in target.points:
        point.doses = {'A': point.doses['A']}
    mapping = match_target(target, scope)
    assert mapping['pairs'] == {'A': 'E'}
    assert mapping['details'][0]['source_role'] == 'partner'
    assert mapping['details'][0]['role_rule'] == 'other_drug_fallback'


@pytest.mark.parametrize('backend', ['logistic'])
def test_focal_preview_and_legacy_full_assembly_use_automatic_matching(backend, monkeypatch):
    from phase1_prediction.multidrug import service
    entries, target = three_papers(backend)
    monkeypatch.setattr(service, 'load_model_library', lambda *a, **kw:
                        ([s for s, _, _ in entries], {'notices': []}))
    library = SimpleNamespace(source_roots=[], evidence_policy='resolved', review_policy='extracted')
    rows = preview(library, target)['selected_series']
    preview_mappings = {r['document_id']: r['mapping'] for r in rows}
    # Current web prediction varies A only; B is fixed treatment context.
    assert {doc: m['pairs'] for doc, m in preview_mappings.items()} == {
        'paper1': {'A': 'D'}, 'paper2': {'A': 'E'}, 'paper3': {}}
    mappings = {s.document_id: match_target(target, s) for s, _, _ in entries}
    assert {doc: m['pairs'] for doc, m in mappings.items()} == {
        'paper1': {'A': 'D'},
        'paper2': {'A': 'E', 'B': 'F'},
        'paper3': {'B': 'G'}}
    assert all(m['matching_policy'] == MATCHING_POLICY for m in mappings.values())

    result = assemble_components(entries, target, backend=backend, draw_count=48)
    assert result['status'] == 'fitted'
    assert np.isfinite(result['posterior_draws']).all()
    used = set()
    saw_same_paper = saw_cross_paper = False
    for recipe in result['provenance']:
        components = recipe['components']
        assert len({(c['source_fit_id'], c['source_id']) for c in components}) == 2
        for c in components:
            mapping = mappings[c['source_fit_id'].removesuffix('_fit')]
            assert c['source_id'] == mapping['pairs'][c['target_id']]
            assert c['mapping_id'] == mapping['mapping_id']
            used.add((c['target_id'], c['source_id']))
        same_paper = len({c['source_fit_id'] for c in components}) == 1
        saw_same_paper |= same_paper
        saw_cross_paper |= not same_paper
        if same_paper:
            assert len({c['flat_draw'] for c in components}) == 1
        if backend == 'logistic':
            expected = 'source_joint_draw' if same_paper else 'fixed_prior'
            assert recipe['interactions'][0]['basis'] == expected
        else:
            assert bool(recipe['interactions']['retained_pairs']) == same_paper
    assert used == {('A', 'D'), ('A', 'E'), ('B', 'F'), ('B', 'G')}
    assert saw_same_paper and saw_cross_paper


def test_role_does_not_score_but_manual_mapping_remains_a_constraint():
    entries, target = three_papers()
    scope = entries[1][0]
    # Identical drug evidence has a deterministic tie, independent of role.
    target.drugs[1].profile=profile('B','alpha','partner')
    for drug in scope.drugs:
        drug.profile = profile(drug.name, 'alpha', drug.research_role)
    original=match_target(target, scope)
    target.drugs[0].research_role='concomitant'
    target.drugs[1].research_role='primary'
    changed=match_target(target, scope)
    assert original['pairs']==changed['pairs']
    assert [d['affinity'] for d in original['details']]==[d['affinity'] for d in changed['details']]
    target.mapping_overrides = {scope.scope_id: {'A': 'E'}}
    assert match_target(target, scope)['pairs'] == {'A': 'E', 'B': 'F'}
    target.mapping_overrides[scope.scope_id]['B'] = 'E'
    with pytest.raises(ValueError, match='Invalid target mapping override'):
        match_target(target, scope)


def test_supported_primary_precedes_exact_identity_partner_without_inflating_score():
    entries, target = three_papers()
    scope = entries[0][0]
    target.drugs = target.drugs[:1]
    # D shares mechanism etc.; partner A is the same named drug.
    scope.drugs[1].name = 'A'
    scope.drugs[1].profile = profile('A', 'alpha', 'partner')
    result = match_target(target, scope)
    assert result['pairs'] == {'A': 'D'}
    detail = result['details'][0]
    assert detail['role_rule'] == 'primary_investigated_first'
    assert 0 < detail['affinity'] < 1
    scope.drugs = scope.drugs[:1]
    scope.drugs[0].research_role = 'background'
    without_role_priority = match_target(target, scope)
    assert without_role_priority['details'][0]['affinity'] == detail['affinity']
    assert without_role_priority['score'] == result['score']


def test_same_drug_and_alias_take_priority_among_coprimary_drugs():
    entries, target = three_papers()
    scope = entries[0][0]
    target.drugs = target.drugs[:1]
    # Exact identity has sparse features; the other co-primary has rich matches.
    scope.drugs[1].research_role = scope.drugs[0].research_role = 'co_primary'
    scope.drugs[1].profile = {'name': scope.drugs[1].name, 'aliases': ['A']}
    assert match_target(target, scope)['pairs'] == {'A': scope.drugs[1].model_drug_id}


def test_features_choose_among_coprimary_drugs_independent_of_order_or_rank():
    entries, target = three_papers()
    scope = entries[0][0]
    target.drugs = target.drugs[:1]
    scope.drugs[1].research_role = scope.drugs[0].research_role = 'co_primary'
    # Both eligible, but the second has the closer target/mechanism profile.
    scope.drugs[0].profile = profile('D', 'gamma', 'co_primary')
    scope.drugs[0].profile['classes'] = ['alpha']
    scope.drugs[1].profile = profile('C', 'alpha', 'co_primary')
    assert match_target(target, scope)['pairs'] == {'A': 'C'}
    scope.drugs.reverse()
    assert match_target(target, scope)['pairs'] == {'A': 'C'}


def test_unsupported_primary_does_not_create_a_match():
    entries, target = three_papers()
    scope = entries[2][0]
    target.drugs = target.drugs[:1]
    result = match_target(target, scope)
    assert not result['pairs']
    assert result['unmatched_target_ids'] == ['A']
