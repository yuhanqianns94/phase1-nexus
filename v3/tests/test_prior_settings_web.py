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
    next(widget for widget in [*app.selectbox, *app.radio] if widget.label == label).set_value(value).run()
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
    choose(app, 'Prior scales', 'regularized')
    choose(app, 'Prior scales', 'custom')
    assert app.number_input(key='prior_settings_single_drug_logistic_risk_log_slope_sd').value == .5
    app.number_input(key='prior_settings_single_drug_logistic_risk_log_slope_sd').set_value(.65).run()
    assert app.number_input(key='prior_settings_single_drug_logistic_risk_log_slope_sd').value == .65
    app.sidebar.radio[0].set_value('Library').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.number_input(key='prior_settings_single_drug_logistic_risk_log_slope_sd').value == .65
    preferences = json.loads((tmp_path / 'state/bayesian_prior_preferences.json').read_text())
    assert preferences['backends']['single_drug_logistic']['custom_risk']['log_slope_sd'] == .65
    # A new browser session can resume the saved custom values.
    fresh = AppTest.from_file(Path(__file__).parents[1] / 'phase1_web/app.py', default_timeout=30).run()
    fresh.sidebar.radio[0].set_value('Prediction Results').run()
    assert not fresh.exception
    assert fresh.number_input(key='prior_settings_single_drug_logistic_risk_log_slope_sd').value == .65


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
    choose(app, 'Prior scales', 'regularized')
    next(button for button in app.button if button.label == 'Generate prediction').click().run()
    assert calls[-1]['prior'] == risk_prior_preset('regularized', 'single_drug_logistic')
    first = app.session_state['multidrug_prediction']['run_id']
    choose(app, 'Prior scales', 'broad')
    assert app.radio(key='prior_settings_single_drug_logistic_risk_preset').value == 'broad'
    assert any(label.startswith('Broader variation') and r'\sigma' in label for label in app.radio(key='prior_settings_single_drug_logistic_risk_preset').options)
    assert app.session_state['multidrug_prediction'] is None
    choose(app, 'Prior scales', 'regularized')
    assert app.session_state['multidrug_prediction']['run_id'] == first
    choose(app, 'Curve distribution', 'average')
    next(button for button in app.button if button.label == 'Generate prediction').click().run()
    assert calls[-1]['synthesis'] == 'same_article_joint_posterior_average'
    assert calls[-1]['backend'] == 'single_drug_logistic'
    assert calls[-1]['power_prior'] is None
    choose(app, 'Curve distribution', 'historical')
    choose(app, 'Historical borrowing settings', 'cautious')
    next(button for button in app.button if button.label == 'Generate prediction').click().run()
    assert calls[-1]['synthesis'] == 'target_power_prior'
    assert calls[-1]['power_prior']['strength'] == .5
    assert calls[-1]['power_prior']['slope_heterogeneity_scale'] == .75
    assert calls[-1]['prior'] == risk_prior_preset('regularized', 'single_drug_logistic')
    assert not app.exception and not app.error


def test_random_slope_custom_scale_persists_and_reaches_prediction(tmp_path, monkeypatch):
    import phase1_web.multidrug_app as workspace
    calls = []
    def predict(library, target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {'run_id': 'slope-test', 'target': target.model_dump(),
            'evidence_contract': {'prior': kwargs['prior'].model_dump()}, 'methods': {}})
    monkeypatch.setattr(workspace, 'predict', predict)
    app = target_app(tmp_path, monkeypatch)
    choose(app, 'Curve distribution', 'historical')
    choose(app, 'Historical borrowing settings', 'custom')
    key = 'prior_settings_single_drug_logistic_power_slope_heterogeneity'
    app.number_input(key=key).set_value(.35).run()
    next(button for button in app.button if button.label == 'Generate prediction').click().run()
    assert calls[-1]['power_prior']['slope_heterogeneity_scale'] == .35
    app.number_input(key=key).set_value(.8).run()
    assert app.session_state['multidrug_prediction'] is None
    app.sidebar.radio[0].set_value('Library').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.number_input(key=key).value == .8
    preferences = json.loads((tmp_path / 'state/bayesian_prior_preferences.json').read_text())
    assert preferences['backends']['single_drug_logistic']['custom_power']['slope_heterogeneity_scale'] == .8
    assert any('v_i' in item.value and 'tau_\\ell' in item.value for item in app.latex)
    assert not app.exception and not app.error


def test_saved_legacy_selectors_cannot_reenable_incompatible_construction(tmp_path, monkeypatch):
    app = target_app(tmp_path, monkeypatch)
    app.session_state['workflow_probability_model'] = 'logistic_product'
    app.session_state['workflow_aggregation'] = 'Power prior from discounted historical data (Method 3)'
    app.run()
    labels = [widget.label for widget in app.selectbox]
    assert 'Probability model' not in labels
    assert 'Evidence construction' not in labels
    assert not any('historical contributions' in item.label for item in app.expander)
    assert app.radio(key='workflow_formula_aggregation').value == 'mixture'
    assert not app.exception
