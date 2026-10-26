"""Risk-family selection preserves the meaning of priors and evidence methods."""
from types import SimpleNamespace

from phase1_prediction.multidrug.contracts import PriorSpec
from phase1_web.multidrug_app import prediction_result_key, prediction_result_label
from phase1_web.prior_cache import MODEL_LABELS
from phase1_web.prior_settings import prior_parameter_rows
from test_prior_settings_web import target_app, choose


def test_product_labels_and_result_identity_are_model_specific():
    assert prediction_result_label('logistic_product · same_article_joint_posterior_average') == (
        'Product-interaction Logistic · Posterior average')
    assert prediction_result_key('logistic_product', 'same_article_joint_posterior_average', prior=PriorSpec()) != (
        prediction_result_key('logistic', 'same_article_joint_posterior_average', prior=PriorSpec()))
    rows = prior_parameter_rows(PriorSpec(), 'logistic_product')
    assert 'all-zero doses' in rows[0]['Parameter']
    assert 'linear relative-dose' in rows[1]['Parameter']
    assert MODEL_LABELS['logistic_product'] == 'Product-interaction Logistic'



def test_formula_rendering_keeps_one_focal_curve_for_combination_plans():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string("""
import streamlit as st
from phase1_web.model_formulas import render_model, render_aggregation, target_combination
from phase1_web.prior_settings import render_prior_formulas
from phase1_prediction.multidrug.contracts import PriorSpec
from phase1_web.multidrug_app import example_target
count = st.number_input('Drugs', 1, 4, 1)
used = st.checkbox('Use companions', value=True)
target = example_target()
for i in range(1, count):
    target['drugs'].append({**target['drugs'][0], 'drug_id': str(i), 'name': str(i), 'research_priority': i + 1})
    for point in target['points']: point['doses'][str(i)] = 10. if used else 0.
combination = target_combination(target)
render_model(st, target['drugs'], 'D', combination=combination)
render_prior_formulas(st, PriorSpec(), backend='single_drug_logistic', drug_count=count, combination=combination)
render_aggregation(st, combination=combination)
""").run()
    assert not app.exception
    source_curve = lambda: next(item.value for item in app.latex if r'p_{\mathrm{source}}' in item.value)
    single = source_curve()
    assert r'\alpha+e^\ell r' in single and r'\delta' not in single
    assert any(item.value == 'Historical combination adjustment' for item in app.caption)
    assert not app.expander
    assert not any('No companion is used' in item.value or 'This is one source curve' in item.value
                   or 'Each source uses its own dose reference' in item.value for item in app.caption)
    assert any(r'\Pr(I_s=i)' in item.value for item in app.latex)
    assert not any(r'\bar p^{(s)}' in item.value or r'Y_{ik}' in item.value for item in app.latex)
    app.radio(key='workflow_formula_aggregation').set_value('average').run()
    assert any(r'\bar p^{(s)}' in item.value for item in app.latex)
    assert not any(r'\Pr(I_s=i)' in item.value or r'Y_{ik}' in item.value for item in app.latex)
    app.number_input[0].set_value(3).run()
    assert not app.exception
    multiple = source_curve()
    assert r'+\delta' in multiple and r'\delta C' not in multiple
    assert not any(item.value == 'Historical combination adjustment' for item in app.caption)
    app.checkbox[0].set_value(False).run()
    assert source_curve() == single
    app.radio(key='workflow_formula_aggregation').set_value('historical').run()
    assert not app.expander
    assert any('$u_i$' in item.value and '$v_i$' in item.value for item in app.caption)
    assert any('$C_{ik}$' in item.value for item in app.caption)
    assert any('$a_i=0.5$' in item.value for item in app.caption)
    assert not any(r'\Pr(I_s=i)' in item.value or r'\bar p^{(s)}' in item.value for item in app.latex)
    new_curve = next(item.value for item in app.latex if r'p_{\mathrm{new}}' in item.value)
    assert r'v_{\mathrm{new}}' in new_curve and r'\delta' not in new_curve
    app.checkbox[0].set_value(True).run()
    assert all(r'+\delta' in item.value for item in app.latex if r'p_{\mathrm{new}}' in item.value)
    assert len(app.radio(key='workflow_formula_aggregation').options) == 3
    assert not app.exception
