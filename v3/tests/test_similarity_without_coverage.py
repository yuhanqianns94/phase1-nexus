"""Borrowing uses similarity alone; missing facts never create a second multiplier."""
import copy
from types import SimpleNamespace

import pytest

from phase1_prediction.profile_similarity import compare_profiles
from phase1_prediction.multidrug import component_assembly, service, transport
from phase1_prediction.multidrug.mapping import match_target
from phase1_prediction.multidrug.power_prior import PowerPriorSpec
from phase1_web.evidence_review import power_prior_review_preview
from test_multidrug_average import two_articles
from test_multidrug_power_prior import article


def assert_no_coverage(value):
    if isinstance(value, dict):
        assert 'coverage' not in value
        for child in value.values():
            assert_no_coverage(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_coverage(child)


def test_sparse_and_missing_comparisons_do_not_compute_coverage():
    sparse = {'population': {'disease': ['glioma']}}
    before = copy.deepcopy(sparse)
    result = compare_profiles(sparse, sparse)
    assert result['similarity'] == 1.
    assert result['groups']['dlt']['score'] is None
    assert compare_profiles({}, {})['similarity'] is None
    assert_no_coverage(result)
    assert_no_coverage(compare_profiles({}, {}))
    assert sparse == before


@pytest.mark.parametrize('method', [
    transport.average_joint_posteriors,
    transport.transport_joint_posteriors,
    transport.assemble_components,
    transport.average_assembled_components,
])
def test_all_posterior_aggregation_weights_use_similarity_only(monkeypatch, method):
    entries, target = two_articles('logistic')
    for i, (scope, _, _) in enumerate(entries):
        scope.profile['test_source_index'] = i
    def mapping(target, scope, **kwargs):
        result = match_target(target, scope, **kwargs)
        result['score']['similarity'] = .8 if scope.document_id == 'paper_0' else .4
        return result
    monkeypatch.setattr(transport, 'match_target', mapping)
    monkeypatch.setattr(component_assembly, 'match_target', mapping)
    monkeypatch.setattr(component_assembly, 'compare_mapped_profiles',
        lambda left, right, *args, **kwargs: {'similarity': .8 if right['test_source_index'] == 0 else .4})
    result = method(entries, target, draw_count=12)
    assert result['status'] == 'fitted'
    expected = {'paper_0': 2/3, 'paper_1': 1/3}
    for row in result['contributors']:
        actual = row['component_weights'][0]['weight'] if 'component_weights' in row else row['weight']
        assert actual == pytest.approx(expected[row['document_id']])
    assert_no_coverage(result)


def test_selection_preview_and_power_discount_use_same_similarity(monkeypatch):
    first, target = article(doc='first', strength=.8)
    second, _ = article(doc='second', strength=.4)
    rows = [first, second]
    monkeypatch.setattr(service, 'load_model_library', lambda *a, **kw: ([r[0] for r in rows], {'notices': []}))
    by_id = {r[0].scope_id: r[1] for r in rows}
    monkeypatch.setattr(service, 'match_target', lambda target, scope, **kw: by_id[scope.scope_id])
    library = SimpleNamespace(source_roots=[], evidence_policy='project_default', review_policy='extracted')
    selected, _ = service._select(library, target, None, 0., 1)
    assert [r[0].document_id for r in selected] == ['first']
    assert selected[0][2] == .8
    preview = service.preview(library, target)
    assert_no_coverage(preview)
    result = power_prior_review_preview(preview['selected_series'], [r[0] for r in rows], target,
        PowerPriorSpec(strength=.5, patient_budget=None), {'first': 'Include', 'second': 'Uncertain'})
    assert [r['Likelihood power'] for r in result['rows']] == pytest.approx([.4, .1])
    assert [r['Similarity'] for r in result['rows']] == [.8, .4]
    assert_no_coverage(result)
