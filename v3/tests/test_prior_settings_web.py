"""A prior selection must reach prediction, persist, and invalidate old results."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from phase1_prediction.multidrug.contracts import PriorSpec
from phase1_prediction.multidrug.prior_presets import canonical_prior, risk_prior_preset
from phase1_web.multidrug_app import prediction_result_key
from test_prediction_pipeline import _article


def target_app(tmp_path, monkeypatch):
    _article(tmp_path, 'PMC123', rows=[(.1, 3, 0), (.3, 6, 2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS', str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT', str(tmp_path / 'state'))
    app = AppTest.from_file(Path(__file__).parents[1] / 'phase1_web/app.py', default_timeout=30).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    from test_multidrug_ui_state_audit import seed_test_plan
    seed_test_plan(app,tmp_path)
    app.sidebar.radio[0].set_value('Prediction Results').run()
    return app


def choose(app, label, value):
    next(widget for widget in app.selectbox if widget.label == label).set_value(value).run()
    assert not app.exception and not app.error


def test_matching_custom_values_and_preset_share_result_identity():
    preset = risk_prior_preset('regularized', 'logistic')
    custom = preset.model_copy(update={'prior_id': 'my arbitrary label'})
    assert prediction_result_key('logistic', 'same_article_joint_posterior_average', prior=custom) == prediction_result_key(
        'logistic', 'same_article_joint_posterior_average', prior=preset)
    altered = preset.model_copy(update={'pair_sd': .50000001})
    assert prediction_result_key('logistic', 'same_article_joint_posterior_average', prior=altered) != prediction_result_key(
        'logistic', 'same_article_joint_posterior_average', prior=preset)


def test_custom_fields_persist_across_pages_and_sessions(tmp_path, monkeypatch):
    app = target_app(tmp_path, monkeypatch)
    choose(app, 'Risk-parameter prior', 'regularized')
    choose(app, 'Risk-parameter prior', 'custom')
    assert app.number_input(key='prior_settings_logistic_risk_pair_sd').value == .5
    app.number_input(key='prior_settings_logistic_risk_pair_sd').set_value(.65).run()
    assert app.number_input(key='prior_settings_logistic_risk_pair_sd').value == .65
    app.sidebar.radio[0].set_value('Library').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.number_input(key='prior_settings_logistic_risk_pair_sd').value == .65
    preferences = json.loads((tmp_path / 'state/bayesian_prior_preferences.json').read_text())
    assert preferences['backends']['logistic']['custom_risk']['pair_sd'] == .65
    # A new browser session can resume the saved custom values.
    fresh = AppTest.from_file(Path(__file__).parents[1] / 'phase1_web/app.py', default_timeout=30).run()
    fresh.sidebar.radio[0].set_value('Prediction Results').run()
    assert not fresh.exception
    assert fresh.number_input(key='prior_settings_logistic_risk_pair_sd').value == .65


def test_selected_priors_reach_prediction_and_do_not_reactivate_different_results(tmp_path, monkeypatch):
    import phase1_web.multidrug_app as workspace
    calls = []
    def predict(library, target, **kwargs):
        calls.append(kwargs)
        result = {'run_id': 'test_run_' + str(len(calls)), 'target': target.model_dump(),
                  'evidence_contract': {'prior': kwargs['prior'].model_dump()}, 'methods': {}}
        return SimpleNamespace(to_dict=lambda: result)
    monkeypatch.setattr(workspace, 'predict', predict)
    app = target_app(tmp_path, monkeypatch)
    choose(app, 'Risk-parameter prior', 'regularized')
    next(button for button in app.button if button.label == 'Generate prediction').click().run()
    assert calls[-1]['prior'] == risk_prior_preset('regularized', 'logistic')
    first = app.session_state['multidrug_prediction']['run_id']
    choose(app, 'Risk-parameter prior', 'broad')
    assert app.session_state['multidrug_prediction'] is None
    choose(app, 'Risk-parameter prior', 'regularized')
    assert app.session_state['multidrug_prediction']['run_id'] == first
    choose(app, 'Curve aggregation', 'Power prior from discounted historical data (Method 3)')
    choose(app, 'Historical borrowing settings', 'cautious')
    next(button for button in app.button if button.label == 'Generate prediction').click().run()
    assert calls[-1]['power_prior'] == {'strength': .5, 'patient_budget': 15., 'heterogeneity_scale': .75}
    assert calls[-1]['prior'] == risk_prior_preset('regularized', 'logistic')
    choose(app, 'Historical borrowing settings', 'custom')
    app.number_input(key='prior_settings_logistic_power_strength').set_value(.3).run()
    assert app.session_state['multidrug_prediction'] is None
    app.checkbox(key='prior_settings_logistic_power_use_cap').set_value(False).run()
    next(button for button in app.button if button.label == 'Generate prediction').click().run()
    assert calls[-1]['power_prior']['strength'] == .3
    assert calls[-1]['power_prior']['patient_budget'] is None
    assert not app.exception and not app.error


def test_previous_method3_inputs_are_retained_as_custom_settings(tmp_path, monkeypatch):
    app = target_app(tmp_path, monkeypatch)
    app.session_state['power_prior_strength'] = .4
    app.session_state['power_prior_budget'] = 20.
    app.session_state['power_prior_use_cap'] = False
    app.session_state['power_prior_heterogeneity'] = .8
    choose(app, 'Curve aggregation', 'Power prior from discounted historical data (Method 3)')
    assert app.selectbox(key='prior_settings_logistic_power_preset').value == 'custom'
    assert app.number_input(key='prior_settings_logistic_power_strength').value == .4
    assert app.number_input(key='prior_settings_logistic_power_heterogeneity').value == .8
    assert not app.checkbox(key='prior_settings_logistic_power_use_cap').value
