"""Component borrowing, aggregation uncertainty, frozen redraw and UI dispatch."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit, logit
from streamlit.testing.v1 import AppTest

from phase1_prediction.multidrug import component_assembly, service
from phase1_prediction.multidrug.contracts import PriorSpec, RegimenPoint
from phase1_prediction.multidrug.store import PosteriorStore
from phase1_prediction.multidrug.transport import assemble_components, average_assembled_components, transport_joint_posteriors
from phase1_simulation.multidrug import simulate_regimen_path
from phase1_simulation.service import BoinSettings
from test_multidrug_transport import fixture


def mono_library(extra_a=False):
    original, arrays, manifest, target = fixture(2)
    entries = []
    for name, j, risk in [('mono_a', 0, .1), ('mono_b', 1, .2)] + ([('extra_a', 0, .5)] if extra_a else []):
        scope = original.model_copy(deep=True)
        scope.document_id = name; scope.scope_id = name
        scope.drugs = [scope.drugs[j]]
        scope.drugs[0].research_priority = 1
        scope.profile['configurations'][0]['drugs'] = [scope.drugs[0].profile]
        a = {'a': np.full((2, 20, 1), logit(risk)), 'log_b': np.zeros((2, 20, 1)),
             'eta': np.zeros((2, 20, 0))}
        entries.append((scope, a, {**manifest, 'source_fit_id': name}))
    # A dose-response slice at a fixed B dose, not simultaneous escalation.
    for point in target.points:
        point.doses['B'] = 30.
    target.drugs[1].reference_dose = 30.
    return entries, target


@pytest.mark.parametrize('fn', [assemble_components, average_assembled_components])
def test_two_single_drug_articles_support_fixed_b_curve_and_prior_interaction(fn):
    entries, target = mono_library()
    untouched = [s.model_dump() for s, _, _ in entries]
    result = fn(entries, target, draw_count=20)
    assert result['status'] == 'fitted'
    assert transport_joint_posteriors(entries, target, draw_count=5)['status'] == 'no_similarity_support'
    assert [s.model_dump() for s, _, _ in entries] == untouched
    assert {c['document_id'] for c in result['contributors']} == {'mono_a', 'mono_b'}
    assert result['interaction_support'] == [{'target_ids': ['A', 'B'],
        'source_joint_draw_fraction': 0., 'fixed_prior_fraction': 1.}]
    r_a = np.array([.5, 1., 2.])
    p0 = 1 - (1-expit(logit(.1)+np.log(r_a))) * .8
    h = 2*r_a/(1+r_a)
    for row, trace in zip(result['posterior_draws'], result['provenance']):
        eta = (trace['interactions'] if fn == assemble_components else trace['prior_interactions'])[0]['value']
        np.testing.assert_allclose(row, expit(logit(p0)+eta*h), rtol=1e-12)
    assert len(np.unique(np.array(result['posterior_draws'])[:, 1])) > 1


def test_average_and_mixture_have_different_distributions_not_different_component_evidence():
    entries, target = mono_library(extra_a=True)
    prior = PriorSpec(pair_sd=1e-10)
    avg = average_assembled_components(entries, target, draw_count=30, prior=prior)
    mix = assemble_components(entries, target, draw_count=800, prior=prior)
    assert avg['component_pools'] == mix['component_pools']
    assert avg['synthesis'] != mix['synthesis']
    # A's alternative .1 and .5 curves are assembled with B=.2 FIRST.
    np.testing.assert_allclose(np.array(avg['posterior_draws'])[:, 1], .44, atol=1e-9)
    assert np.std(np.array(mix['posterior_draws'])[:, 1]) > .15
    assert avg['assembly_computation']['evaluation'] == 'exact_recipe_average'
    assert sum(c['weight'] for c in avg['recipe_bank']) == pytest.approx(1.)
    sim = simulate_regimen_path(avg, settings=BoinSettings(), posterior_curve_count=2, trials_per_curve=2)
    assert sim['total_trials'] == 4 and sim['synthesis'] == avg['synthesis']


def test_average_does_not_shrink_a_shared_unlearned_interaction_by_recipe_count():
    entries, target = mono_library(extra_a=True)
    entries[2][1]['a'][:] = logit(.1)
    result = average_assembled_components(entries, target, draw_count=400)
    # Both recipes have the same known component risks. Averaging must therefore
    # leave the full target eta prior, rather than average two independent etas.
    implied = logit(np.array(result['posterior_draws'])[:, 1]) - logit(.28)
    saved = [t['prior_interactions'][0]['value'] for t in result['provenance']]
    np.testing.assert_allclose(implied, saved, rtol=1e-12, atol=1e-12)
    assert np.std(implied) > .9


def test_average_preserves_joint_source_component_draw_and_nonzero_interaction():
    scope, arrays, manifest, target = fixture(2)
    arrays['a'][:, :, 0] = np.arange(40).reshape(2, 20)/40-2
    arrays['a'][:, :, 1] = -arrays['a'][:, :, 0]-2
    arrays['eta'][:, :, 0] = np.arange(40).reshape(2, 20)/20-.5
    target.mapping_overrides = {scope.scope_id: {'A': 'B', 'B': 'A'}}
    result = average_assembled_components([(scope, arrays, manifest)], target, draw_count=30)
    for row, trace in zip(result['posterior_draws'], result['provenance']):
        m = trace['source_draws'][0]['flat_draw']
        q = expit(arrays['a'].reshape(40, 2)[m] + np.log(np.array([.5, 1., 2.])[:, None]))
        p0 = 1-np.prod(1-q, axis=1)
        product = np.array([.25, 1., 4.])
        expected = expit(logit(p0)+arrays['eta'].reshape(40)[m]*2*product/(1+product))
        np.testing.assert_allclose(row, expected, rtol=1e-12)
    assert result['interaction_support'][0]['source_joint_draw_fraction'] == 1.


@pytest.mark.parametrize('fn', [assemble_components, average_assembled_components])
def test_mix_of_single_and_combination_evidence_records_interaction_origins(fn):
    entries, target = mono_library()
    s, a, m, _ = fixture(2)
    a['eta'][:] = .73
    entries.append((s, a, m))
    result = fn(entries, target, draw_count=100)
    fraction = result['interaction_support'][0]['source_joint_draw_fraction']
    assert 0 < fraction < 1
    if fn == assemble_components:
        for trace in result['provenance']:
            if trace['interactions'][0]['basis'] == 'source_joint_draw':
                assert trace['interactions'][0]['value'] == .73
                assert len({c['flat_draw'] for c in trace['components']}) == 1
                assert {c['source_fit_id'] for c in trace['components']} == {'fit_x'}


@pytest.mark.parametrize('mode', ['cross_article_component_assembly', 'cross_article_component_assembly_average'])
def test_service_frozen_surface_and_fixed_b_redraw_reuse_the_same_recipes(tmp_path, monkeypatch, mode):
    entries, target = mono_library(extra_a=True)
    store = PosteriorStore(tmp_path)
    entries = [store.save(m['source_fit_id'], s, a, m) for s, a, m in entries]
    monkeypatch.setattr(service, 'load_model_library', lambda *a, **kw: ([s for s, _, _ in entries], {'notices': []}))
    monkeypatch.setattr(service, 'fit_library', lambda scopes, **kw: (entries, []))
    lib = SimpleNamespace(source_roots=[], evidence_policy='project_default', review_policy='extracted')
    saved = service.predict(lib, target, synthesis=mode, store_root=tmp_path, draw_count=10).to_dict()
    assert len(saved['selected_series']) == 3
    monkeypatch.setattr(service, 'fit_library', lambda *a, **kw: pytest.fail('No refit on redraw'))
    extended = target.model_copy(deep=True)
    extended.points += [RegimenPoint(level_id='other_b', doses={'A': 50., 'B': 60.})]
    redrawn = service.evaluate_frozen_prediction(saved, extended)
    old = saved['methods'][mode]; new = redrawn['methods'][mode]
    assert old['provenance'] == new['provenance']
    assert old['component_pools'] == new['component_pools']
    np.testing.assert_allclose(np.array(new['posterior_draws'])[:, :3], old['posterior_draws'])
    assert new['coordinate_support'][0]['missing_source_axes'] == 3
    assert 'assembled combination is model-derived' in new['coordinate_support'][0]['support']


def test_review_weights_and_failed_sources_apply_per_component():
    entries, target = mono_library(extra_a=True)
    decisions = {'mono_a': 'Include', 'mono_b': 'Include', 'extra_a': 'Uncertain'}
    result = average_assembled_components(entries, target, draw_count=3, relevance_decisions=decisions)
    pool = result['component_pools'][0]['candidates']
    assert [c['candidate_weight'] for c in pool] == pytest.approx([2/3, 1/3])
    decisions['mono_b'] = 'Exclude'
    assert assemble_components(entries, target, draw_count=3, relevance_decisions=decisions)['status'] == 'no_similarity_support'
    entries[1][2]['status'] = 'numerical_failure'
    failed = average_assembled_components(entries, target, draw_count=3)
    assert failed['status'] == 'no_similarity_support'
    assert any(c['reason'] == 'numerical_failure' for c in failed['excluded'])


def test_large_recipe_approximation_is_explicit_bounded_and_frozen(monkeypatch):
    entries, target = mono_library(extra_a=True)
    monkeypatch.setattr(component_assembly, 'AVERAGE_RECIPE_LIMIT', 1)
    result = average_assembled_components(entries, target, draw_count=3)
    assert result['assembly_computation']['evaluation'] == 'monte_carlo_recipe_average'
    assert result['assembly_computation']['recipe_sample_count'] == 1
    target.points.reverse()
    repeated = average_assembled_components(entries, target, draw_count=3)
    assert result['recipe_bank'] == repeated['recipe_bank']
    assert result['provenance'] == repeated['provenance']
    np.testing.assert_allclose(repeated['posterior_draws'], np.array(result['posterior_draws'])[:, ::-1])


def test_ui_offers_both_assembly_methods_and_dispatches_distinct_settings(tmp_path, monkeypatch):
    from test_prediction_pipeline import _article
    from phase1_web import multidrug_app
    _article(tmp_path, 'PMC123', rows=[(.1, 3, 0), (.3, 6, 2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS', str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT', str(tmp_path/'state'))
    app = AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py', default_timeout=30).run()
    target = multidrug_app.example_target()
    target['drugs'][0]['name'] = 'IPH2101'
    app.session_state['multidrug_target_text'] = json.dumps(target)
    app.sidebar.radio[0].set_value('Prediction Results').run()
    app.selectbox(key='workflow_construction').set_value('Component assembly (advanced)').run()
    picker = app.selectbox(key='workflow_component_aggregation')
    assert len(picker.options) == 2
    seen = []
    def fake(*args, **kwargs):
        seen.append(kwargs['synthesis'])
        return SimpleNamespace(to_dict=lambda: {'run_id': 'fake_'+kwargs['synthesis'], 'methods': {}, 'target': target})
    monkeypatch.setattr(multidrug_app, 'predict', fake)
    for label in picker.options:
        app.selectbox(key='workflow_component_aggregation').set_value(label).run()
        next(b for b in app.button if b.label == 'Generate prediction').click().run()
        assert not app.exception and not app.error
    assert seen == ['cross_article_component_assembly_average', 'cross_article_component_assembly']
    # Return to complete borrowing without invalid selectbox state.
    app.selectbox(key='workflow_construction').set_value('Complete-regimen borrowing').run()
    assert len(app.selectbox(key='workflow_aggregation').options) == 3
    assert not app.exception
