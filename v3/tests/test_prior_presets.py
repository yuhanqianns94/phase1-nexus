"""Preset equivalence, partial caches and target-only cache preparation."""
import copy
import json

import numpy as np
import pytest

from phase1_prediction.multidrug import prior_presets as presets
from phase1_prediction.multidrug.contracts import PriorSpec, SamplerSpec
from phase1_prediction.multidrug.power_prior import PowerPriorStore, prepare_power_prior
from phase1_prediction.multidrug.store import PosteriorStore, source_fit_id
from test_multidrug_power_prior import article


@pytest.mark.parametrize('backend', ['logistic'])
def test_default_prior_preserves_existing_numbers_and_cache_label(backend):
    default = PriorSpec()
    assert presets.risk_prior_preset('balanced', backend) == default
    custom = default.model_copy(update={'prior_id': 'my preferred label'})
    assert presets.canonical_prior(custom, backend) == default
    for name in presets.RISK_PRESETS:
        prior = presets.risk_prior_preset(name, backend)
        assert prior.intercept_mean == -1.1
        assert prior.log_slope_mean == 0.
        assert prior.unknown_presence_probability == .5
    assert presets.power_prior_preset('default').model_dump() == {
        'strength': 1., 'patient_budget': 30., 'heterogeneity_scale': .5, 'slope_heterogeneity_scale': .5}


def test_custom_equal_preset_shares_source_identity_but_changed_parameters_do_not():
    backend, active = 'logistic', 'pair_sd'
    (scope, _, _), _ = article()
    prior = presets.risk_prior_preset('regularized', backend)
    custom = prior.model_copy(update={'prior_id': 'custom'})
    snapshot = custom.model_dump()
    sampler = SamplerSpec()
    assert presets.canonical_prior(custom, backend) == prior
    assert source_fit_id(scope, backend, prior, sampler) == source_fit_id(scope, backend, custom, sampler)
    assert source_fit_id(scope, backend, prior, sampler) != source_fit_id(
        scope, backend, custom.model_copy(update={active: 6.}), sampler)
    assert custom.model_dump() == snapshot


def test_invalid_backend_and_preset_names_are_rejected():
    with pytest.raises(ValueError, match='Unsupported probability model'):
        presets.canonical_prior(None, 'anything')
    with pytest.raises(ValueError, match='preset'):
        presets.risk_prior_preset('anything', 'logistic')
    with pytest.raises(ValueError, match='preset'):
        presets.power_prior_preset('anything')


def test_prepare_source_cache_only_fits_missing_eligible_outcome_scopes(tmp_path, monkeypatch):
    (first, _, _), _ = article(doc='ready')
    (second, _, _), _ = article(doc='missing')
    (failed, _, _), _ = article(doc='failed')
    phase_two = copy.deepcopy(first)
    phase_two.document_id, phase_two.scope_id = 'phase_two', 'phase_two_scope'
    phase_two.profile['configurations'][0]['phase'] = 'Phase II'
    no_outcomes = copy.deepcopy(first)
    no_outcomes.document_id, no_outcomes.scope_id = 'empty', 'empty_scope'
    no_outcomes.observations = []
    prior = presets.risk_prior_preset('regularized', 'logistic')
    sampler = SamplerSpec()
    store = PosteriorStore(tmp_path)
    store.save(source_fit_id(first, 'logistic', prior, sampler), first, {'a': np.zeros((2, 4, 2))},
               {'status': 'fitted'})
    store.save(source_fit_id(failed, 'logistic', prior, sampler), failed, {},
               {'status': 'numerical_failure', 'reason': 'ESS below configured minimum'})
    calls = []

    def fit_source(scope, backend, received, configured):
        calls.append(scope.document_id)
        assert received == prior
        assert configured == sampler
        return {'a': np.zeros((2, 4, 2))}, {'status': 'fitted'}

    monkeypatch.setattr('phase1_prediction.multidrug.fitting.fit_source', fit_source)
    scopes = [first, second, failed, phase_two, no_outcomes]
    before = presets.source_cache_status(scopes, backend='logistic', prior=prior, sampler=sampler,
                                         store_root=tmp_path)
    assert before['counts'] == {'ready': 1, 'missing': 1, 'cached_failure': 1,
                                'invalid_cache': 0, 'phase_excluded': 1, 'no_outcomes': 1}
    assert not calls
    after = presets.prepare_source_cache(scopes, backend='logistic', prior=prior, sampler=sampler,
                                          store_root=tmp_path)
    assert calls == ['missing']
    assert after['counts']['ready'] == 2
    assert after['counts']['cached_failure'] == 1
    assert after['errors'] == []
    assert after['failed'][0]['reason'] == 'ESS below configured minimum'
    assert len(after['skipped']) == 2


def test_status_is_cheap_and_does_not_load_posterior_arrays(tmp_path, monkeypatch):
    (scope, _, _), _ = article()
    fit_id = source_fit_id(scope, 'logistic', PriorSpec(), SamplerSpec())
    PosteriorStore(tmp_path).save(fit_id, scope, {}, {'status': 'fitted'})
    monkeypatch.setattr(PosteriorStore, 'load', lambda *args: pytest.fail('Status must not load arrays'))
    monkeypatch.setattr(PosteriorStore, 'fit', lambda *args: pytest.fail('Status must not fit'))
    status = presets.source_cache_status([scope], store_root=tmp_path)
    assert status['counts']['ready'] == 1
    (tmp_path / fit_id / 'draws.npz').unlink()
    status = presets.source_cache_status([scope], store_root=tmp_path)
    assert status['counts']['invalid_cache'] == 1


def test_status_reports_non_object_manifest_as_invalid(tmp_path):
    (scope, _, _), _ = article()
    fit_id = source_fit_id(scope, 'logistic', PriorSpec(), SamplerSpec())
    directory = tmp_path / fit_id
    directory.mkdir()
    (directory / 'manifest.json').write_text('[]')
    status = presets.source_cache_status([scope], store_root=tmp_path)
    assert status['counts']['invalid_cache'] == 1
    assert 'object' in status['failed'][0]['reason']


@pytest.mark.parametrize('backend', ['logistic'])
def test_prepared_target_cache_matches_prediction_cache_and_custom_reuses_it(tmp_path, monkeypatch, backend):
    row, target = article()
    original = row[0].model_dump()
    prepared = prepare_power_prior([row], target, presets.power_prior_preset('cautious'))
    prior = presets.risk_prior_preset('broad', backend)
    sampler = SamplerSpec(samples=4, warmup=4)
    custom = prior.model_copy(update={'prior_id': 'human-readable custom name'})
    fit_id = PowerPriorStore.identity(prepared, backend, prior, sampler)
    assert fit_id == PowerPriorStore.identity(prepared, backend, custom, sampler)
    calls = []

    def fit_target(received, model, received_prior, configured):
        assert received == prepared and model == backend and received_prior == prior and configured == sampler
        calls.append(received)
        return {'a': np.zeros((2, 4, 2))}, {'status': 'fitted'}

    monkeypatch.setattr('phase1_prediction.multidrug.power_prior.fit_power_prior', fit_target)
    before = presets.prepared_target_cache_status(prepared, backend=backend, prior=custom,
                                                  sampler=sampler, store_root=tmp_path)
    assert before['cache_status'] == 'missing' and before['fit_id'] == fit_id
    after = presets.prepare_prepared_target_cache(prepared, backend=backend, prior=custom,
                                                  sampler=sampler, store_root=tmp_path)
    assert after['cache_status'] == 'ready' and len(calls) == 1
    presets.prepare_prepared_target_cache(prepared, backend=backend, prior=prior,
                                          sampler=sampler, store_root=tmp_path)
    assert len(calls) == 1
    assert row[0].model_dump() == original
    assert not list(tmp_path.glob('fit_*'))
    restored, _, manifest = PowerPriorStore(tmp_path).load(fit_id)
    assert restored == json.loads(json.dumps(prepared)) and manifest['prior'] == prior.model_dump()
    changed = prepare_power_prior([row], target, presets.power_prior_preset('default'))
    assert PowerPriorStore.identity(changed, backend, prior, sampler) != fit_id
    changed = prepare_power_prior([row], target, presets.power_prior_preset('cautious'), {'paper': 'Uncertain'})
    assert PowerPriorStore.identity(changed, backend, prior, sampler) != fit_id


def test_target_without_eligible_evidence_does_not_fit(tmp_path, monkeypatch):
    row, target = article()
    prepared = prepare_power_prior([row], target, presets.power_prior_preset('default'), {'paper': 'Exclude'})
    monkeypatch.setattr(PowerPriorStore, 'fit', lambda *args: pytest.fail('No eligible outcomes must not fit'))
    status = presets.prepare_prepared_target_cache(prepared, store_root=tmp_path)
    assert status['cache_status'] == 'no_eligible_evidence'
    assert status['fit_id'] is None


def test_custom_default_predict_contract_is_canonical(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase1_prediction.multidrug import service
    row, target = article(k=1)
    monkeypatch.setattr(service, '_select', lambda *args: ([row], {'notices': []}))
    prior_received = []

    def fit_library(scopes, **kwargs):
        prior_received.append(kwargs['prior'])
        return [], []

    monkeypatch.setattr(service, 'fit_library', fit_library)
    # No contributing fits returns an empty prediction, sufficient to inspect
    # the contract without inventing posterior samples for this boundary test.
    result = service.predict(SimpleNamespace(), target, prior=PriorSpec(prior_id='custom'),
                              store_root=tmp_path, draw_count=3)
    assert prior_received == [PriorSpec()]
    assert result.evidence_contract['prior'] == PriorSpec().model_dump()
