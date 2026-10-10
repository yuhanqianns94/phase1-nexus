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
    assert mapping['details'][0]['role_rule'] == 'roles_not_scored'


@pytest.mark.parametrize('backend', ['logistic'])
def test_three_article_cross_role_preview_and_assembly_agree(backend, monkeypatch):
    from phase1_prediction.multidrug import service
    entries, target = three_papers(backend)
    monkeypatch.setattr(service, 'load_model_library', lambda *a, **kw:
                        ([s for s, _, _ in entries], {'notices': []}))
    library = SimpleNamespace(source_roots=[], evidence_policy='resolved', review_policy='extracted')
    rows = preview(library, target)['selected_series']
    mappings = {r['document_id']: r['mapping'] for r in rows}
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
