import copy

import numpy as np
import pytest

from test_multidrug_transport import fixture
from phase1_prediction.multidrug.transport import average_joint_posteriors, transport_joint_posteriors
from phase1_simulation.multidrug import simulate_regimen_path
from phase1_simulation.service import BoinSettings


def two_articles(backend):
    scope, arrays, manifest, target = fixture(1, backend)
    entries = []
    for i, risk in enumerate((.1, .5)):
        s, a, m = scope.model_copy(deep=True), copy.deepcopy(arrays), dict(manifest)
        s.document_id = f'paper_{i}'
        s.scope_id = f'scope_{i}'
        m['source_fit_id'] = f'fit_{i}'
        a['a'][:] = np.log(risk / (1-risk))
        entries.append((s, a, m))
    return entries, target


@pytest.mark.parametrize('backend', ['logistic'])
def test_average_is_not_mixture_and_uses_joint_curves(backend):
    entries, target = two_articles(backend)
    result = average_joint_posteriors(entries, target, backend=backend, draw_count=20)
    mixture = transport_joint_posteriors(entries, target, backend=backend, draw_count=200)
    assert result['synthesis'] == 'same_article_joint_posterior_average'
    assert np.allclose(np.array(result['posterior_draws'])[:, 1], .3)
    assert np.std(np.array(mixture['posterior_draws'])[:, 1]) > .15
    for row, trace in zip(result['posterior_draws'], result['provenance']):
        expected = np.zeros(3)
        assert len(trace['article_curves']) == 2
        for c in trace['article_curves']:
            i = int(c['document_id'].split('_')[-1])
            a = entries[i][1]['a'][c['chain'], c['draw'], 0]
            expected += c['weight'] / (1 + np.exp(-a-np.log([.5, 1., 2.])))
        np.testing.assert_allclose(row, expected)
    assert result == average_joint_posteriors(entries, target, backend=backend, draw_count=20)
    simulation = simulate_regimen_path(result, settings=BoinSettings(), posterior_curve_count=2, trials_per_curve=2)
    assert simulation['total_trials'] == 4
    assert simulation['synthesis'] == result['synthesis']


def test_additional_scope_does_not_increase_article_budget():
    entries, target = two_articles('logistic')
    scope, arrays, manifest = entries[0]
    extra = scope.model_copy(deep=True)
    extra.scope_id = 'additional_scope'
    result = average_joint_posteriors(entries + [(extra, arrays, {**manifest, 'source_fit_id': 'fit_extra'})],
                                      target, draw_count=30)
    assert np.allclose(np.array(result['posterior_draws'])[:, 1], .3)
    for trace in result['provenance']:
        assert len(trace['article_curves']) == 2
        assert {r['document_id'] for r in trace['article_curves']} == {'paper_0', 'paper_1'}


def test_failed_source_and_incomplete_mapping_are_not_averaged():
    entries, target = two_articles('logistic')
    entries[1][2]['status'] = 'numerical_failure'
    result = average_joint_posteriors(entries, target, draw_count=4)
    assert np.allclose(np.array(result['posterior_draws'])[:, 1], .1)
    assert result['excluded'][0]['reason'] == 'numerical_failure'


def test_prediction_service_routes_average_and_preserves_method_identity(monkeypatch):
    from phase1_prediction.multidrug import service
    entries, target = two_articles('logistic')
    monkeypatch.setattr(service, '_select', lambda *args, **kwargs:
        ([(s, {'score':{'similarity':1.}}, 1.) for s, _, _ in entries], {'notices': []}))
    monkeypatch.setattr(service, 'fit_library', lambda *args, **kwargs: (entries, []))
    result = service.predict(None, target, synthesis='same_article_joint_posterior_average', draw_count=4).to_dict()
    method = result['methods']['same_article_joint_posterior_average']
    assert method['summary'][1]['posterior_mean'] == pytest.approx(.3)
    assert result['evidence_contract']['synthesis'] == method['synthesis']


def test_batch_quality_audit_can_be_explicitly_skipped(tmp_path, monkeypatch):
    from test_openai_batch import request
    import phase1_extraction.openai_batch as batch
    from phase1_extraction.models import ReportedExtraction
    from phase1_extraction.artifacts import read_json
    def forbidden(*args, **kwargs):
        raise AssertionError('Quality audit must not run')
    monkeypatch.setattr(batch, 'audit_reported_extraction', forbidden)
    result = batch.persist_transport_result(output_root=tmp_path, request=request('TEST'),
        parsed=ReportedExtraction(document_id='TEST'), raw_response={}, transport='openai_batch', run_quality_audit=False)
    assert read_json(result.extraction_audit)['status'] == 'not_run'
    assert read_json(result.request_summary)['quality_audit_performed'] is False
    assert result.reported_extraction.is_file() and result.curve_rows.is_file()
