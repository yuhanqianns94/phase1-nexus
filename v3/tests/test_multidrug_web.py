from pathlib import Path
from streamlit.testing.v1 import AppTest
from test_prediction_pipeline import _article


def test_native_workspace_and_target_editor(tmp_path,monkeypatch):
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value=='PHASE I NEXUS'
    app.sidebar.radio[0].set_value('Target Trial').run()
    assert not app.exception and not app.error
    assert app.button(key='plan_add_drug')
    assert not any(x.label=='Probability model' for x in app.selectbox)
    from test_multidrug_ui_state_audit import seed_test_plan
    seed_test_plan(app,tmp_path)
    assert not app.exception and not app.error
    assert not any(x.label == 'Probability model' for x in app.selectbox)
    assert not any(x.label == 'Curve aggregation' for x in app.selectbox)
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert not app.exception and not app.error
    assert not any(x.label in ('Probability model', 'Evidence construction') for x in app.selectbox)
    borrowing = app.radio(key='workflow_formula_aggregation')
    assert borrowing.value == 'mixture'
    borrowing.set_value('average').run()
    assert not app.exception and not app.error
    assert not any('uplift' in str(x.label).lower() for x in app.number_input)
    next(x for x in app.radio if x.label == 'Prior scales').set_value('custom').run()
    app.number_input(key='prior_settings_single_drug_logistic_risk_intercept_mean').set_value(-1.5).run()
    app.radio(key='workflow_formula_aggregation').set_value('mixture').run()
    assert app.number_input(key='prior_settings_single_drug_logistic_risk_intercept_mean').value == -1.5
    assert not app.exception


def test_grid_is_not_automatically_a_simulation_path():
    from phase1_prediction.multidrug.contracts import TargetRegimen
    from phase1_web.multidrug_app import example_target
    example=example_target();example['path_level_ids']=[]
    target=TargetRegimen.model_validate(example)
    assert not target.path_level_ids


def test_legacy_draft_migration_preserves_drug_and_doses():
    from phase1_web.multidrug_draft import migrate_legacy_draft
    target=migrate_legacy_draft({'values':{'target_drug':'A','target_dose_unit':'mg','planned_doses_text':'10,20',
        'opt_cancer_type':'lung cancer','opt_mechanism':'kinase inhibitor'}})
    assert target.drugs[0].name=='A'
    assert [p.doses['drug_1'] for p in target.points]==[10.,20.]
    assert target.drugs[0].profile['mechanisms']==['kinase inhibitor']
