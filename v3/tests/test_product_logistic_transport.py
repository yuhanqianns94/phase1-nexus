"""Ordinary product Logistic remains the same model throughout transport."""
import numpy as np
import pytest
from scipy.special import expit

from phase1_prediction.multidrug import service
from phase1_prediction.multidrug.contracts import PriorSpec, RegimenPoint, SamplerSpec
from phase1_prediction.multidrug.math import interaction_sets
from phase1_prediction.multidrug.store import PosteriorStore, source_fit_id
from phase1_prediction.multidrug.transport import (assemble_components, average_assembled_components,
    transport_joint_posteriors, average_joint_posteriors)
from phase1_simulation.multidrug import simulate_regimen_path
from phase1_simulation.service import BoinSettings
from test_multidrug_transport import fixture


def product_fixture(k=2):
    scope, _, manifest, target = fixture(k, 'logistic_product')
    # Different intercepts, slopes, and interactions covary across joint draws.
    index = np.arange(40).reshape(2, 20)
    arrays = {'alpha': -3.+index/100,
              'log_b': np.stack([index/100 + j/5 for j in range(k)], axis=-1),
              'eta': np.stack([index/100-.5+j/10 for j in range(len(interaction_sets(k)))], axis=-1)
                     if k > 1 else np.zeros((2, 20, 0))}
    return scope, arrays, manifest, target


@pytest.mark.parametrize('k', [1, 2, 3, 4])
@pytest.mark.parametrize('aggregation', [transport_joint_posteriors, average_joint_posteriors])
def test_transport_matches_product_formula_with_zero_and_partial_absence(k, aggregation):
    scope, arrays, manifest, target = product_fixture(k)
    target.points += [RegimenPoint(level_id='none', doses={d.drug_id: 0. for d in target.drugs}),
                      RegimenPoint(level_id='only_A', doses={d.drug_id: 50. if i == 0 else 0.
                                                           for i, d in enumerate(target.drugs)})]
    result = aggregation([(scope, arrays, manifest)], target, backend='logistic_product', draw_count=8)
    assert result['status'] == 'fitted'
    r = np.array([[p.doses[d.drug_id]/d.reference_dose for d in target.drugs] for p in target.points])
    for row, trace in zip(result['posterior_draws'], result['provenance']):
        t = trace['article_curves'][0] if 'article_curves' in trace else trace
        c, m = t['chain'], t['draw']
        score = arrays['alpha'][c, m] + r @ np.exp(arrays['log_b'][c, m])
        for j, group in enumerate(interaction_sets(k)):
            score += arrays['eta'][c, m, j] * np.prod(r[:, group], axis=1)
        np.testing.assert_allclose(row, expit(score), rtol=1e-12)
        assert row[-2] == pytest.approx(expit(arrays['alpha'][c, m]))
    simulation = simulate_regimen_path(result, settings=BoinSettings(), posterior_curve_count=2, trials_per_curve=2)
    assert simulation['total_trials'] == 4


def test_transport_preserves_swapped_manual_mapping_and_source_joint_interaction():
    scope, arrays, manifest, target = product_fixture()
    target.mapping_overrides = {scope.scope_id: {'A': 'B', 'B': 'A'}}
    target.points = [RegimenPoint(level_id='asymmetric', doses={'A': 25., 'B': 100.})]
    target.path_level_ids = []
    result = transport_joint_posteriors([(scope, arrays, manifest)], target, backend='logistic_product', draw_count=5)
    for row, trace in zip(result['posterior_draws'], result['provenance']):
        c, m = trace['chain'], trace['draw']
        score = arrays['alpha'][c, m] + np.exp(arrays['log_b'][c, m]) @ [2., .5] + arrays['eta'][c, m, 0]
        assert row[0] == pytest.approx(expit(score))
    assert result['contributors'][0]['mapping']['pairs'] == {'A': 'B', 'B': 'A'}


def test_product_source_interactions_preserved_under_permutation_of_any_target_slots():
    scope, arrays, manifest, target = product_fixture(4)
    target.mapping_overrides = {scope.scope_id: {'A': 'D', 'B': 'A', 'C': 'B', 'D': 'C'}}
    result = transport_joint_posteriors([(scope, arrays, manifest)], target, backend='logistic_product', draw_count=3)
    assert result['status'] == 'fitted'
    assert not result['excluded']
    from phase1_prediction.multidrug.math import product_logistic_risk
    # Source A,B,C,D correspond to target B,C,D,A, including the fourth slot.
    r = np.array([[p.doses[name]/50 for name in ('B', 'C', 'D', 'A')]
                  for p in target.points])
    for row, trace in zip(result['posterior_draws'], result['provenance']):
        c, m = trace['chain'], trace['draw']
        np.testing.assert_allclose(row, product_logistic_risk(
            arrays['alpha'][c, m], arrays['log_b'][c, m], r, arrays['eta'][c, m]))


def test_background_exposure_uses_binary_coordinate_and_cannot_invent_numeric_axis():
    scope, arrays, manifest, target = product_fixture()
    scope.drugs[1].mode = 'background'
    scope.drugs[1].reference_dose = None
    scope.drugs[1].unit = None
    target.drugs[1].mode = 'background'
    target.drugs[1].reference_dose = None
    for i, point in enumerate(target.points):
        point.doses['B'] = None if i < 2 else 0.
    result = transport_joint_posteriors([(scope, arrays, manifest)], target, backend='logistic_product', draw_count=5)
    for row, trace in zip(result['posterior_draws'], result['provenance']):
        c, m = trace['chain'], trace['draw']
        r = np.array([[.5, 1.], [1., 1.], [2., 0.]])
        score = arrays['alpha'][c, m] + r @ np.exp(arrays['log_b'][c, m]) + r.prod(axis=1)*arrays['eta'][c, m, 0]
        np.testing.assert_allclose(row, expit(score), rtol=1e-12)
    target.drugs[1].mode = 'numeric'; target.drugs[1].reference_dose = 30.; target.drugs[1].unit = 'mg'
    target.drugs[1].allow_background_anchor = True
    for point in target.points:
        point.doses['B'] = 30.
    rejected = transport_joint_posteriors([(scope, arrays, manifest)], target, backend='logistic_product', draw_count=5)
    assert rejected['status'] == 'no_similarity_support'
    assert 'numeric source' in rejected['excluded'][0]['reason']


def test_extreme_product_coordinates_fail_explicitly_instead_of_saving_nan():
    scope, arrays, manifest, target = product_fixture()
    target.points = [RegimenPoint(level_id='extreme', doses={'A': 1e308, 'B': 1e308})]
    target.path_level_ids = []
    for drug in target.drugs:
        drug.reference_dose = 1.
    with pytest.raises(ValueError, match='non-finite risks'):
        transport_joint_posteriors([(scope, arrays, manifest)], target, backend='logistic_product', draw_count=3)


@pytest.mark.parametrize('mode', ['same_article_joint_posterior_average', 'same_article_joint_posterior_mixture'])
def test_product_service_and_frozen_redraw_preserve_backend_and_cache(tmp_path, monkeypatch, mode):
    scope, arrays, manifest, target = product_fixture()
    store = PosteriorStore(tmp_path)
    entry = store.save(manifest['source_fit_id'], scope, arrays, manifest)
    monkeypatch.setattr(service, '_select', lambda *args, **kwargs:
        ([(scope, {'score': {'similarity': 1.}}, 1.)], {'notices': []}))
    monkeypatch.setattr(service, 'fit_library', lambda *args, **kwargs: ([entry], []))
    saved = service.predict(None, target, backend='logistic_product', synthesis=mode,
                            store_root=tmp_path, draw_count=5).to_dict()
    target.points = list(reversed(target.points))
    monkeypatch.setattr(PosteriorStore, 'fit', lambda *args, **kwargs: pytest.fail('No refit for redraw'))
    repeated = service.evaluate_frozen_prediction(saved, target)
    old, new = saved['methods'][mode], repeated['methods'][mode]
    assert old['backend'] == new['backend'] == 'logistic_product'
    assert old['provenance'] == new['provenance']
    np.testing.assert_allclose(new['posterior_draws'], np.array(old['posterior_draws'])[:, ::-1])
    assert source_fit_id(scope, 'logistic', PriorSpec(), SamplerSpec()) != source_fit_id(scope, 'logistic_product', PriorSpec(), SamplerSpec())


def test_product_power_prior_service_saves_and_redraws_scalar_intercept_cache(tmp_path, monkeypatch):
    from test_multidrug_power_prior import article
    from phase1_prediction.multidrug.power_prior import PowerPriorStore, PowerPriorSpec
    row, target = article()
    monkeypatch.setattr(service, '_select', lambda *args, **kwargs: ([row], {'notices': []}))
    monkeypatch.setattr(service, 'fit_library', lambda *args, **kwargs: pytest.fail('Method 3 never fits source articles'))
    saved = service.predict(None, target, backend='logistic_product', synthesis='target_power_prior',
        store_root=tmp_path, draw_count=8, power_prior=PowerPriorSpec(strength=0),
        sampler=SamplerSpec(samples=4, warmup=1)).to_dict()
    assert saved['methods']['target_power_prior']['status'] == 'fitted'
    _, arrays, manifest = PowerPriorStore(tmp_path).load(saved['evidence_contract']['target_fit_id'])
    assert manifest['backend'] == 'logistic_product'
    assert arrays['alpha'].shape == (2, 4) and 'a' not in arrays
    monkeypatch.setattr(PowerPriorStore, 'fit', lambda *args, **kwargs: pytest.fail('No refit for frozen redraw'))
    target.points.reverse()
    repeated = service.evaluate_frozen_prediction(saved, target)
    old = saved['methods']['target_power_prior']; new = repeated['methods']['target_power_prior']
    np.testing.assert_allclose(new['posterior_draws'], np.array(old['posterior_draws'])[:, ::-1])
    assert old['provenance'] == new['provenance']


@pytest.mark.parametrize('mode, fn', [('cross_article_component_assembly', assemble_components),
                                    ('cross_article_component_assembly_average', average_assembled_components)])
def test_product_cannot_silently_assemble_incompatible_source_intercepts(monkeypatch, mode, fn):
    scope, arrays, manifest, target = product_fixture()
    with pytest.raises(ValueError, match='shared intercept'):
        fn([(scope, arrays, manifest)], target, backend='logistic_product')
    monkeypatch.setattr(service, 'fit_library', lambda *a, **kw: pytest.fail('Reject before source fitting'))
    with pytest.raises(ValueError, match='shared-intercept'):
        service.predict(None, target, backend='logistic_product', synthesis=mode)
