"""Risk-family selection preserves the meaning of priors and evidence methods."""
from types import SimpleNamespace

from phase1_prediction.multidrug.contracts import PriorSpec
from phase1_web.multidrug_app import prediction_result_key, prediction_result_label
from phase1_web.prior_cache import MODEL_LABELS
from phase1_web.prior_settings import prior_parameter_rows
from test_prior_settings_web import target_app, choose


def test_product_labels_and_result_identity_are_model_specific():
    assert prediction_result_label('logistic_product · same_article_joint_posterior_average') == (
        'Product-interaction Logistic · Method 1: weighted average')
    assert prediction_result_key('logistic_product', 'same_article_joint_posterior_average', prior=PriorSpec()) != (
        prediction_result_key('logistic', 'same_article_joint_posterior_average', prior=PriorSpec()))
    rows = prior_parameter_rows(PriorSpec(), 'logistic_product')
    assert 'all-zero doses' in rows[0]['Parameter']
    assert 'linear relative-dose' in rows[1]['Parameter']
    assert MODEL_LABELS['logistic_product'] == 'Product-interaction Logistic'


def test_product_selection_routes_complete_methods_and_disallows_assembly(tmp_path, monkeypatch):
    from phase1_web import multidrug_app
    calls = []
    def fake_predict(library, target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {
            'run_id': f'test_{len(calls)}', 'target': target.model_dump(), 'methods': {}})
    monkeypatch.setattr(multidrug_app, 'predict', fake_predict)
    app = target_app(tmp_path, monkeypatch)
    choose(app, 'Evidence construction', 'Component assembly (advanced)')
    app.selectbox(key='workflow_probability_model').set_value('logistic_product').run()
    assert not app.exception and not app.error
    assert app.selectbox(key='workflow_product_construction').options == ['Complete-regimen borrowing']
    assert not any(box.key == 'workflow_component_aggregation' for box in app.selectbox)
    assert any('shared baseline intercept' in cap.value for cap in app.caption)
    for method in list(app.selectbox(key='workflow_aggregation').options):
        choose(app, 'Curve aggregation', method)
        next(button for button in app.button if button.label == 'Generate prediction').click().run()
        assert not app.exception and not app.error
        assert calls[-1]['backend'] == 'logistic_product'
    assert [call['synthesis'] for call in calls] == [
        'same_article_joint_posterior_average', 'same_article_joint_posterior_mixture', 'target_power_prior']
    choose(app, 'Risk-parameter prior', 'custom')
    assert app.number_input(key='prior_settings_logistic_product_risk_intercept_mean').label == (
        'Baseline log-odds at all-zero doses: mean')
    app.number_input(key='prior_settings_logistic_product_risk_intercept_mean').set_value(-2.).run()
    app.selectbox(key='workflow_probability_model').set_value('logistic').run()
    choose(app, 'Risk-parameter prior', 'custom')
    assert app.number_input(key='prior_settings_logistic_risk_intercept_mean').value == PriorSpec().intercept_mean
    choose(app, 'Evidence construction', 'Component assembly (advanced)')
    assert len(app.selectbox(key='workflow_component_aggregation').options) == 2
    app.selectbox(key='workflow_probability_model').set_value('logistic_product').run()
    assert app.number_input(key='prior_settings_logistic_product_risk_intercept_mean').value == -2.
    assert not app.exception and not app.error
