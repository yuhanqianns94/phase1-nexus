"""User edits must not leave evidence weights or BOIN results out of date."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phase1_prediction.multidrug.transport import transport_joint_posteriors
from phase1_prediction.profile_similarity import DEFAULT_WEIGHTS
from test_multidrug_transport import fixture
from test_prediction_pipeline import _article


def app_with_library(tmp_path, monkeypatch, *, articles=1):
    for doc in ['PMC123','PMC456'][:articles]:
        _article(tmp_path,doc,rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    return AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()


def target_app(tmp_path, monkeypatch, *, articles=1):
    app_with_library(tmp_path,monkeypatch,articles=articles)
    from phase1_prediction.multidrug.observations import load_model_library
    from phase1_web.multidrug_examples import target_from_scope
    scopes,_=load_model_library([tmp_path])
    target=target_from_scope(scopes[0])
    (tmp_path/'state'/'multidrug_target_draft.json').write_text(target.model_dump_json())
    return AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()


def seed_test_plan(app, root):
    """Supply fixture data without a user-facing library-import control."""
    from phase1_prediction.multidrug.observations import load_model_library
    from phase1_web.multidrug_examples import target_from_scope
    scopes,_=load_model_library([root])
    app.session_state['multidrug_target_text']=target_from_scope(scopes[0]).model_dump_json()
    app.session_state['multidrug_reset_editor']=True
    app.run()


def seed_outputs(app):
    app.session_state['multidrug_prediction']={'run_id':'previous_prediction'}
    app.session_state['multidrug_results']={'logistic':{'run_id':'previous_prediction'}}
    app.session_state['multidrug_preview']={'selected_series':[]}
    for key in ('multidrug_simulation','multidrug_search'):
        app.session_state[key]={'prediction_run_id':'previous_prediction'}


def assert_invalidated(app):
    assert not app.exception
    assert not app.session_state['multidrug_prediction']
    assert not app.session_state['multidrug_results']
    assert 'multidrug_preview' not in app.session_state
    assert 'multidrug_simulation' not in app.session_state
    assert 'multidrug_search' not in app.session_state


@pytest.mark.parametrize('invalid',[
    '{invalid JSON','[]','null','{"drugs":{}}','{"points":{}}',
    '{"profile":[]}','{"drugs":[{"profile":null}]}',
    '{"points":[{"doses":[]}]}','{"profile":{"population":null}}',
])
def test_invalid_target_clears_outputs_without_overwriting_saved_draft(tmp_path,monkeypatch,invalid):
    app=target_app(tmp_path,monkeypatch)
    next(c for c in app.checkbox if c.label=='Advanced JSON editor').check().run()
    path=tmp_path/'state'/'multidrug_target_draft.json'
    original=path.read_text()
    seed_outputs(app)
    app.text_area(key='multidrug_target_json_editor').set_value(invalid).run()
    assert app.error
    assert_invalidated(app)
    assert path.read_text()==original
    assert app.session_state['multidrug_target_text']==original


def test_zero_analysis_weights_clear_outputs(tmp_path,monkeypatch):
    app=target_app(tmp_path,monkeypatch)
    app.sidebar.radio[0].set_value('Review evidence').run()
    seed_outputs(app)
    assert not any(widget.key=='multidrug_weight_structure' for widget in app.number_input)
    for key in DEFAULT_WEIGHTS:
        if key!='structure':app.number_input(key='multidrug_weight_'+key).set_value(0.)
    app.run()
    assert any('At least one indicator weight' in e.value for e in app.error)
    assert_invalidated(app)


def test_similarity_thresholds_survive_conditional_controls_and_navigation(tmp_path,monkeypatch):
    app=target_app(tmp_path,monkeypatch)
    app.sidebar.radio[0].set_value('Review evidence').run()
    app.slider(key='threshold_local').set_value(.35).run()
    app.selectbox(key='workflow_similarity_engine').set_value('openai').run()
    app.slider(key='threshold_openai').set_value(.65).run()
    app.selectbox(key='workflow_similarity_engine').set_value('local').run()
    assert app.slider(key='threshold_local').value==.35
    next(s for s in app.selectbox if s.label=='Candidate pool').set_value('Top N articles').run()
    next(s for s in app.selectbox if s.label=='Candidate pool').set_value('Similarity threshold').run()
    assert app.slider(key='threshold_local').value==.35
    app.sidebar.radio[0].set_value('Library').run()
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert app.slider(key='threshold_local').value==.35
    app.selectbox(key='workflow_similarity_engine').set_value('openai').run()
    assert app.slider(key='threshold_openai').value==.65
    assert not app.error and not app.exception


def test_explicit_draft_settings_rebase_controls_on_other_pages(tmp_path,monkeypatch):
    app=target_app(tmp_path,monkeypatch)
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.number_input(key='workflow_target_dlt').value==.25
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert app.number_input(key='workflow_min_clinical_coverage').value==0.
    app.sidebar.radio[0].set_value('Target Trial').run()
    next(c for c in app.checkbox if c.label=='Advanced JSON editor').check().run()
    draft=json.loads(app.text_area(key='multidrug_target_json_editor').value)
    draft.update(target_dlt=.35,minimum_clinical_coverage=.65)
    app.text_area(key='multidrug_target_json_editor').set_value(json.dumps(draft)).run()
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert app.number_input(key='workflow_min_clinical_coverage').value==.65
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.number_input(key='workflow_target_dlt').value==.35
    saved=json.loads((tmp_path/'state'/'multidrug_target_draft.json').read_text())
    assert saved['minimum_clinical_coverage']==.65 and saved['target_dlt']==.35
    assert not app.exception and not app.error


def test_review_only_selected_articles_and_preserve_deselected_article_decisions(tmp_path,monkeypatch):
    app=target_app(tmp_path,monkeypatch,articles=2)
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(r for r in app.radio if r.label=='Evidence screening').set_value('Review article abstracts').run()
    next(b for b in app.button if b.label=='Find matching articles').click().run()
    selector=next(s for s in app.multiselect if s.label=='Articles to use')
    first,second=selector.value
    radios=[r for r in app.radio if r.label=='Relevance decision']
    radios[0].set_value('Include').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert next(b for b in app.button if b.label=='Generate prediction').disabled
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(s for s in app.multiselect if s.label=='Articles to use').set_value([first]).run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert not next(b for b in app.button if b.label=='Generate prediction').disabled
    app.sidebar.radio[0].set_value('Review evidence').run()
    from phase1_prediction.multidrug.workflow import article_budgets
    review=app.session_state['multidrug_review_state']
    rows=[row for row in app.session_state['multidrug_preview']['selected_series'] if row['series_key'] in review['included']]
    budget=article_budgets(rows,review['decisions'])
    assert [row['document_id'] for row in budget]==['PMC123']
    assert [row['normalized_weight'] for row in budget]==[1.]
    assert not app.dataframe and not app.json
    next(s for s in app.multiselect if s.label=='Articles to use').set_value([second]).run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert next(b for b in app.button if b.label=='Generate prediction').disabled
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(r for r in app.radio if r.label=='Relevance decision').set_value('Uncertain').run()
    record=json.loads(next((tmp_path/'state'/'relevance_reviews').glob('*.json')).read_text())
    assert record['decisions']=={'PMC123':'Include','PMC456':'Uncertain'}
    next(s for s in app.multiselect if s.label=='Articles to use').set_value([first,second]).run()
    assert [r.value for r in app.radio if r.label=='Relevance decision']==['Include','Uncertain']
    review=app.session_state['multidrug_review_state']
    budget=article_budgets(app.session_state['multidrug_preview']['selected_series'],review['decisions'])
    assert [row['normalized_weight'] for row in budget]==pytest.approx([2/3,1/3])
    assert not app.dataframe and not app.json
    assert not any(widget.label=='Adjust drug correspondence (advanced)' for widget in app.expander)
    next(s for s in app.multiselect if s.label=='Articles to use').set_value([]).run()
    assert not [r for r in app.radio if r.label=='Relevance decision']
    assert not app.session_state['multidrug_review_state']['ready']
    assert next(b for b in app.button if b.label=='Continue to risk curves').disabled
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert next(b for b in app.button if b.label=='Generate prediction').disabled
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert next(s for s in app.multiselect if s.label=='Articles to use').value==[]
    next(s for s in app.multiselect if s.label=='Articles to use').set_value([first,second]).run()
    assert [r.value for r in app.radio if r.label=='Relevance decision']==['Include','Uncertain']
    assert not app.error and not app.exception


def test_boin_settings_invalidate_only_affected_outputs_and_survive_navigation(tmp_path,monkeypatch):
    app=app_with_library(tmp_path,monkeypatch)
    scope,arrays,manifest,target=fixture(1)
    method=transport_joint_posteriors([(scope,arrays,manifest)],target,draw_count=4)
    # This state test reuses a fixed probability matrix under the active UI family.
    method['backend'] = 'single_drug_logistic'
    prediction={'run_id':'prediction_ui_audit','model_version':'v3_multidrug_bayes_1',
        'target':target.model_dump(mode='json'),'methods':{method['synthesis']:method},'evidence_contract':{}}
    app.session_state['multidrug_prediction']=prediction
    app.session_state['multidrug_results']={'logistic':prediction}
    app.sidebar.radio[0].set_value('OC Simulation').run()
    app.number_input(key='boin_design_scenarios').set_value(1).run()
    app.number_input(key='boin_design_repetitions').set_value(2).run()
    next(b for b in app.button if b.label=='Run BOIN').click().run()
    app.radio(key='boin_design_task').set_value('Compare multiple designs').run()
    app.multiselect(key='boin_design_search_sizes').set_value([12,18]).run()
    next(b for b in app.button if b.label=='Compare designs').click().run()
    simulation_id=app.session_state['multidrug_simulation']['run_id']
    search_id=app.session_state['multidrug_search']['run_id']
    app.sidebar.radio[0].set_value('Download results').run()
    app.sidebar.radio[0].set_value('OC Simulation').run()
    assert app.number_input(key='boin_design_scenarios').value==1
    assert app.multiselect(key='boin_design_search_sizes').value==[12,18]
    assert app.session_state['multidrug_simulation']['run_id']==simulation_id
    assert app.session_state['multidrug_search']['run_id']==search_id
    app.radio(key='boin_design_task').set_value('Simulate one design').run()
    assert not any(b.label=='Compare designs' for b in app.button)
    app.number_input(key='boin_design_nmax').set_value(60).run()
    assert 'multidrug_simulation' not in app.session_state
    assert app.session_state['multidrug_search']['run_id']==search_id
    next(b for b in app.button if b.label=='Run BOIN').click().run()
    assert app.session_state['multidrug_simulation']['settings']['max_sample_size']==60
    app.radio(key='boin_design_task').set_value('Compare multiple designs').run()
    assert not any(b.label=='Run BOIN' for b in app.button)
    app.multiselect(key='boin_design_search_sizes').set_value([18,24]).run()
    assert 'multidrug_search' not in app.session_state
    assert 'multidrug_simulation' in app.session_state
    next(b for b in app.button if b.label=='Compare designs').click().run()
    app.number_input(key='boin_design_high').set_value(.4).run()
    assert 'multidrug_search' not in app.session_state
    assert 'multidrug_simulation' not in app.session_state
    assert not app.error and not app.exception
