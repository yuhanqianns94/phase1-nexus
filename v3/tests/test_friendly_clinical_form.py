"""Clinical form shortcuts must preserve deliberate and imported clinical facts."""
import copy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phase1_web.multidrug_form import blank_target


def _draft():
    draft=blank_target()
    draft['drugs']=[{'drug_id':'A','name':'Drug A','research_role':'primary',
                     'mode':'background','unit':None,'profile':{}}]
    return draft


def _form_app(draft=None):
    app=AppTest.from_string('''
import json
import streamlit as st
from phase1_web.multidrug_form import render_draft
if st.session_state.pop('multidrug_rebase_editor',False):
    for key in list(st.session_state):
        if key.startswith('plan_'):del st.session_state[key]
    st.session_state['base']=json.loads(st.session_state['multidrug_target_text'])
result=render_draft(st.session_state['base'])
st.session_state['result']=result
st.session_state['multidrug_target_text']=json.dumps(result)
''',default_timeout=30)
    app.session_state['base']=copy.deepcopy(draft if draft is not None else _draft())
    app.run()
    assert not app.exception
    return app


def _phase(app):
    return app.session_state['result']['drugs'][0]['profile']['phases'][0]


def test_first_drug_can_be_named_directly_without_adding_an_empty_record():
    app=_form_app(blank_target())
    assert app.session_state['result']['drugs']==[]
    app.text_input(key='plan_first_drug_name').set_value('Investigational A').run()
    assert not app.exception
    result=app.session_state['result']
    assert len(result['drugs'])==1
    assert result['drugs'][0]['name']=='Investigational A'
    assert result['drugs'][0]['research_role']=='unclear'
    assert result['points']==[]
    assert 'presence' not in result['drugs'][0]['profile']
    assert result['profile']['planning']['exposure_states']=={}
    assert any('Plan saved. Next, review matching articles.' in message.value for message in app.success)


def test_clinical_pages_preserve_imported_facts():
    draft=_draft()
    original={'raw':'Grade 3 toxicity unless reversible within 48 hours',
              'threshold':3.,'unit':'grade','timing':'cycle 1','source_id':'source-7',
              'logic_tokens':['unless'],'custom_fact':{'exception':'retain'}}
    draft['profile']['dlt']={'criteria':[original]}
    draft['drugs'][0]['profile']['phases']=[{'cycle':'3 weeks',
        'administration_days':'Days 1–5 unless interrupted for toxicity',
        'source':{'page':3},'structured':{'source_schedule':'retain'}}]
    app=_form_app(draft)
    assert [tab.label for tab in app.tabs]==['1. Drugs','2. Administration','3. Population','4. DLT','5. Doses (optional)','6. Summary']
    assert not app.expander
    app.text_input(key='plan_A_name').set_value('Edited drug name').run()
    result=app.session_state['result']
    assert result['profile']['dlt']['criteria'][0]==original
    assert _phase(app)==draft['drugs'][0]['profile']['phases'][0]
    assert not app.exception


def test_structured_rule_generates_and_updates_wording_without_double_entry():
    app=_form_app()
    app.button(key='plan_elig_inclusion_add').click().run()
    app.selectbox(key='plan_elig_inclusion_0_topic').set_value('Hemoglobin')
    app.selectbox(key='plan_elig_inclusion_0_op').set_value('>=')
    app.number_input(key='plan_elig_inclusion_0_threshold').set_value(9.)
    app.selectbox(key='plan_elig_inclusion_0_unit').set_value('g/dL').run()
    rule=app.session_state['result']['profile']['eligibility']['inclusion'][0]
    assert rule['raw']=='Hemoglobin >= 9 g/dL'
    assert (rule['topic'],rule['operator'],rule['threshold'],rule['unit'])==('Hemoglobin','>=',9.,'g/dL')
    identity=rule['clause_id']
    app.number_input(key='plan_elig_inclusion_0_threshold').set_value(10.).run()
    rule=app.session_state['result']['profile']['eligibility']['inclusion'][0]
    assert rule['raw']=='Hemoglobin >= 10 g/dL'
    assert rule['topic']=='Hemoglobin' and rule['clause_id']==identity
    app.text_area(key='plan_elig_inclusion_0_raw').set_value('Hemoglobin >= 10 g/dL without transfusion in the past week').run()
    app.number_input(key='plan_elig_inclusion_0_threshold').set_value(11.).run()
    rule=app.session_state['result']['profile']['eligibility']['inclusion'][0]
    assert rule['raw']=='Hemoglobin >= 10 g/dL without transfusion in the past week'
    assert rule['threshold']==11. and not rule.get('_generated_statement')
    assert not app.exception


def test_existing_complex_rule_never_loses_exceptions_when_threshold_changes():
    draft=_draft()
    original={'raw':'Grade >= 3 toxicity except alopecia during cycle 1',
              'topic':'Nonhematologic toxicity','operator':'>=','threshold':3.,
              'unit':'grade','timing':'cycle 1','logic_tokens':['except'],
              'source_id':'source-9','custom_fact':{'reviewed':True}}
    draft['profile']['dlt']={'criteria':[original]}
    app=_form_app(draft)
    app.number_input(key='plan_dlt_criteria_0_threshold').set_value(4.).run()
    rule=app.session_state['result']['profile']['dlt']['criteria'][0]
    assert rule['raw']==original['raw']
    assert rule['threshold']==4.
    for field in ('logic_tokens','timing','source_id','custom_fact'):assert rule[field]==original[field]
    assert not app.exception


def test_dosing_days_follow_actual_cycle_and_keep_other_stage_metadata():
    draft=_draft()
    draft['drugs'][0]['profile']['phases']=[{'cycle':'8 weeks','source_id':'stage-1',
        'structured':{'administration_days':[1,8],'custom':'keep'}}]
    app=_form_app(draft)
    app.selectbox(key='plan_A_phase_0_days_mode').set_value('Every day').run()
    assert _phase(app)['administration_days']==list(range(1,57))
    assert _phase(app)['source_id']=='stage-1'
    assert _phase(app)['structured']['custom']=='keep'
    assert 'administration_days' not in _phase(app)['structured']
    app.selectbox(key='plan_A_phase_0_days_mode').set_value('Specific days').run()
    assert '56' in app.multiselect(key='plan_A_phase_0_days').options
    app.multiselect(key='plan_A_phase_0_days').set_value([1,8,56]).run()
    assert _phase(app)['administration_days']==[1,8,56]
    app.selectbox(key='plan_A_phase_0_days_mode').set_value('Every day')
    app.number_input(key='plan_A_phase_0_cycle').set_value(3.).run()
    assert _phase(app)['administration_days']==list(range(1,22))
    assert not app.exception


def test_custom_schedule_and_hidden_edits_survive_switching_shortcuts():
    draft=_draft()
    draft['drugs'][0]['profile']['phases']=[{'cycle':'21 days',
        'administration_days':'Days 1–5, except during dose interruption'}]
    app=_form_app(draft)
    app.text_area(key='plan_A_phase_0_days_description').set_value('Days 1–7, except during dose interruption').run()
    app.selectbox(key='plan_A_phase_0_days_mode').set_value('Every day').run()
    assert _phase(app)['administration_days']==list(range(1,22))
    app.selectbox(key='plan_A_phase_0_days_mode').set_value('Custom description').run()
    assert _phase(app)['administration_days']=='Days 1–7, except during dose interruption'
    assert not app.exception


def test_unknown_cycle_is_not_imputed_by_every_day_shortcut():
    draft=_draft()
    draft['drugs'][0]['profile']['phases']=[{'administration_days':'Treatment days depend on recovery'}]
    app=_form_app(draft)
    app.selectbox(key='plan_A_phase_0_days_mode').set_value('Every day').run()
    assert _phase(app)['administration_days']=='Treatment days depend on recovery'
    assert not _phase(app).get('cycle')
    assert any('Enter a whole-day cycle length' in message.value for message in app.info)
    app.number_input(key='plan_A_phase_0_cycle').set_value(14.).run()
    assert _phase(app)['administration_days']==list(range(1,15))
    assert not app.exception


def test_imported_normalized_schedule_is_preserved_until_explicitly_edited():
    draft=_draft()
    stage={'cycle':'21 days','structured':{'cycle_value':4.,'cycle_unit':'weeks',
             'administration_days':[1,8,15],'custom_fact':'retain'}}
    draft['drugs'][0]['profile']['phases']=[stage]
    app=_form_app(draft)
    assert _phase(app)==stage
    assert '28' in app.multiselect(key='plan_A_phase_0_days').options
    app.selectbox(key='plan_A_phase_0_days_mode').set_value('Every day').run()
    assert _phase(app)['administration_days']==list(range(1,29))
    app.number_input(key='plan_A_phase_0_cycle').set_value(14.).run()
    assert _phase(app)['administration_days']==list(range(1,15))
    assert _phase(app)['structured']=={'custom_fact':'retain'}
    assert not app.exception


def test_infusion_fields_follow_form_changes_in_additional_details():
    draft=_draft()
    draft['drugs'][0]['profile']['phases']=[{}]
    app=_form_app(draft)
    assert not any(widget.key=='plan_A_phase_0_infusion_duration' for widget in app.number_input)
    app.selectbox(key='plan_A_phase_0_form').set_value('infusion').run()
    app.number_input(key='plan_A_phase_0_infusion_duration').set_value(60.).run()
    assert _phase(app)['infusion_duration']=='60 minutes'
    assert not app.exception


def test_candidate_page_preserves_distinct_use_states():
    draft=_draft()
    draft['points']=[{'level_id':f'L{index}','doses':{'A':value}} for index,value in enumerate([None,None,None,0.],1)]
    states=['Not entered','Undecided','Use: dose unknown','Do not use']
    draft['profile']['planning']['exposure_states']={f'L{index}':{'A':state} for index,state in enumerate(states,1)}
    app=_form_app(draft)
    result=app.session_state['result']
    assert result['profile']['planning']['exposure_states']==draft['profile']['planning']['exposure_states']
    assert result['points']==draft['points']
    assert app.session_state['trial_plan_step']=='1. Drugs'
    assert app.tabs[4].label=='5. Doses (optional)'
    assert not app.exception


def test_duplicate_schedule_keeps_current_edits_and_independent_copy():
    draft=_draft()
    draft['drugs'][0]['profile']['phases']=[{'cycle':'21 days','route':'oral','source_id':'stage-2'}]
    app=_form_app(draft)
    app.number_input(key='plan_A_phase_0_cycle').set_value(28.)
    app.button(key='plan_A_duplicate_phase_0').click().run()
    phases=app.session_state['result']['drugs'][0]['profile']['phases']
    assert phases==[{'cycle':'28 days','route':'oral','source_id':'stage-2'}]*2
    app.number_input(key='plan_A_phase_1_cycle').set_value(35.).run()
    phases=app.session_state['result']['drugs'][0]['profile']['phases']
    assert phases[0]['cycle']=='28 days' and phases[1]['cycle']=='35 days'
    assert not app.exception


def _workflow_app(tmp_path,monkeypatch,*,with_article=True):
    from phase1_web.multidrug_app import example_target
    from test_prediction_pipeline import _article
    if with_article:_article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    app.session_state['multidrug_target_text']=json.dumps(example_target())
    app.session_state['multidrug_reset_editor']=True
    app.run()
    assert not app.exception and not app.error
    return app


@pytest.mark.parametrize('page,setting_key,field,value',[
    ('Review evidence','workflow_min_clinical_coverage','minimum_clinical_coverage',.45),
    ('Prediction Results','workflow_target_dlt','target_dlt',.3),
])
def test_visiting_trial_plan_preserves_result_for_already_saved_settings(tmp_path,monkeypatch,page,setting_key,field,value):
    app=_workflow_app(tmp_path,monkeypatch)
    app.sidebar.radio[0].set_value(page).run()
    app.number_input(key=setting_key).set_value(value).run()
    saved=json.loads(app.session_state['multidrug_target_text'])
    assert saved[field]==value
    # Simulate a calculation completed after the setting was already saved.
    # No posterior fitting is necessary to verify navigation invalidation.
    result={'run_id':'after-settings-change','target':saved,'methods':{}}
    app.session_state['multidrug_prediction']=copy.deepcopy(result)
    app.session_state['multidrug_results']={'example':copy.deepcopy(result)}
    app.sidebar.radio[0].set_value('Target Trial').run()
    assert app.session_state['multidrug_prediction']['run_id']=='after-settings-change'
    assert app.session_state['multidrug_results']['example']['run_id']=='after-settings-change'
    assert json.loads(app.session_state['multidrug_target_text'])[field]==value
    assert not app.exception and not app.error


def test_changed_plan_units_and_reference_rebase_without_losing_explored_doses(tmp_path,monkeypatch):
    app=_workflow_app(tmp_path,monkeypatch)
    app.sidebar.radio[0].set_value('Prediction Results').run()
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    assert app.text_input(key='explore_D_unit').value=='mg'
    app.number_input(key='explore_D_actual_lo').set_value(70.)
    app.number_input(key='explore_D_actual_hi').set_value(300.).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    app.selectbox(key='plan_D_unit').set_value('mg/kg').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.text_input(key='explore_D_unit').value=='mg/kg'
    app.checkbox(key='explore_D_custom_ref').check().run()
    app.number_input(key='explore_D_ref').set_value(200.).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    assert 'plan_D_ref' not in [item.key for item in app.number_input]
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.number_input(key='explore_D_ref').value==200.
    assert app.number_input(key='explore_D_actual_lo').value==70.
    assert app.number_input(key='explore_D_actual_hi').value==300.
    assert not app.exception and not app.error


def test_previous_draft_can_be_restored_without_any_library_examples(tmp_path,monkeypatch):
    app=_workflow_app(tmp_path,monkeypatch,with_article=False)
    before=json.loads(app.session_state['multidrug_target_text'])
    next(button for button in app.button if button.label=='Start new trial').click().run()
    assert not json.loads(app.session_state['multidrug_target_text'])['drugs']
    next(button for button in app.button if button.label=='Restore previous draft').click().run()
    restored=json.loads(app.session_state['multidrug_target_text'])
    assert restored['drugs']==before['drugs']
    assert restored['points']==before['points']
    assert not app.exception and not app.error
