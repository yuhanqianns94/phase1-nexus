import copy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from phase1_extraction.clinical_protocol import normalized_phase, complete_conditions, planned_issues, validate_logic, logic_tree
from phase1_extraction.models import ReportedExtraction
from phase1_prediction.profile_similarity import phase_pair, condition, logic_comparison
from phase1_prediction.multidrug.contracts import TargetRegimen
from phase1_prediction.multidrug.service import preview_planned
from phase1_prediction.evidence import load_evidence_library
from phase1_web.multidrug_form import blank_target
from phase1_web.multidrug_state import simulation_view
from phase1_web.multidrug_app import example_target
from test_prediction_pipeline import _article


def app_for(tmp_path,monkeypatch):
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    return app


@pytest.mark.parametrize('left,right',[('q3w','every 21 days'),('every 3 weeks','every 21 days'),('daily','once daily')])
def test_reported_and_planned_frequency_equivalence(left,right):
    a={'frequency':left};b={'frequency':right}
    assert normalized_phase(a)['frequency']==normalized_phase(b)['frequency']
    assert next(d for d in phase_pair(a,b)['details'] if d['indicator']=='frequency')['score']==1


def test_schedule_units_and_unknowns():
    a={'infusion_duration':'120 minutes'};b={'infusion_duration':'2 hours'}
    assert next(d for d in phase_pair(a,b)['details'] if d['indicator']=='infusion_duration')['score']==1
    assert normalized_phase({'frequency':None})['frequency'] is None
    s=normalized_phase({'structured':{'frequency_count':1,'interval_value':3,'interval_unit':'weeks'}})
    assert s['frequency']=='every 21 days'


def test_partial_structuring_never_drops_other_dlt_criteria():
    result=complete_conditions(['ANC <= 500','Febrile neutropenia'],[{'clause_id':'r1','raw_text':'ANC <= 500','threshold':500,'comparator':'<='}],condition)
    assert len(result)==2 and result[1]['raw']=='Febrile neutropenia'
    assert result[0]['clause_id']=='r1'
    assert condition('Platelet count <=25,000 platelets/mm3')['threshold']==25000


def test_nested_logic_is_id_independent_and_detects_and_or_changes():
    clauses=[{'clause_id':i,'topic':i} for i in 'abc']
    groups=[{'group_id':'inner','operator':'AND','children':['b','c']},{'group_id':'outer','operator':'OR','children':['a','inner']}]
    validate_logic(groups,set('abc'))
    left={'criteria':clauses,'logic_groups':groups}
    right=copy.deepcopy(left);right['logic_groups'][0]['operator']='OR'
    assert logic_comparison(left,right,['criteria'])['score']==0
    renamed=copy.deepcopy(groups);renamed[0]['group_id']='nested';renamed[1]['children']=['nested','a']
    assert logic_tree(groups,clauses)==logic_tree(renamed,clauses)
    groups[0]['children'].append('outer')
    with pytest.raises(ValueError,match='cycle'):validate_logic(groups,set('abc'))


def test_unknown_use_is_not_present_unknown_dose():
    draft=example_target()
    draft['profile']['planning']={'exposure_states':{'L1':{'D':'Not entered'}}}
    draft['points'][0]['doses']['D']=None
    assert planned_issues(draft)
    with pytest.raises(ValueError,match='Incomplete planned'):TargetRegimen.model_validate(draft)
    draft['profile']['planning']['exposure_states']['L1']['D']='Use: dose unknown'
    assert not planned_issues(draft)


def test_clinical_preview_does_not_require_or_create_doses(tmp_path):
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    draft=blank_target();draft['drugs']=[{'drug_id':'new','name':'A','profile':{},'research_role':'primary'}]
    draft['minimum_clinical_coverage']=0.  # This test isolates dose-optional preview, not the coverage gate.
    original=copy.deepcopy(draft)
    result=preview_planned(load_evidence_library([tmp_path]),draft)
    assert result['selected_series'] and result['prediction_ready'] is False
    assert draft==original and draft['points']==[]


def test_blank_plan_survives_navigation(tmp_path,monkeypatch):
    app=app_for(tmp_path,monkeypatch)
    app.text_area(key='plan_pop_disease').set_value('Advanced solid tumors').run()
    app.text_input(key='plan_first_drug_name').set_value('Investigational X').run()
    draft=json.loads(app.session_state['multidrug_target_text'])
    did=draft['drugs'][0]['drug_id']
    app.text_input(key='plan_'+did+'_name').set_value('Investigational X').run()
    app.button(key='plan_add_point').click().run()
    app.sidebar.radio[0].set_value('Library').run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    assert not app.exception and not app.error
    saved=json.loads(app.session_state['multidrug_target_text'])
    assert saved['profile']['population']['disease']==['Advanced solid tumors']
    assert saved['profile']['planning']['exposure_states']['L1'][did]=='Not entered'
    assert 'Generate prediction' not in [b.label for b in app.button]


def test_new_trial_structured_entry_and_reference_freezing(tmp_path,monkeypatch):
    app=app_for(tmp_path,monkeypatch)
    app.text_input(key='plan_first_drug_name').set_value('Investigational X').run()
    did=json.loads(app.session_state['multidrug_target_text'])['drugs'][0]['drug_id']
    app.text_input(key='plan_'+did+'_name').set_value('Investigational X').run()
    app.selectbox(key='plan_'+did+'_unit').set_value('mg').run()
    app.button(key='plan_add_point').click().run()
    app.selectbox(key='plan_L1_'+did+'_state').set_value('Use: dose specified').run()
    app.number_input(key='plan_L1_'+did+'_dose').set_value(50.).run()
    assert not app.exception and not app.error
    before=json.loads(app.session_state['multidrug_target_text'])
    assert before['drugs'][0]['reference_dose']==pytest.approx(50)
    app.number_input(key='plan_L1_'+did+'_dose').set_value(100.).run()
    after=json.loads(app.session_state['multidrug_target_text'])
    assert after['drugs'][0]['reference_dose']==before['drugs'][0]['reference_dose']
    assert after['points'][0]['doses'][did]==100
    assert not any(b.label=='Generate prediction' for b in app.button)
    app.sidebar.radio[0].set_value('Prediction Results').run()
    app.radio(key='explore_goal').set_value('Compare candidate doses').run()
    assert any(b.label=='Generate prediction' for b in app.button)


def test_saved_source_plan_and_rule_group_edit(tmp_path,monkeypatch):
    app=app_for(tmp_path,monkeypatch)
    from test_multidrug_ui_state_audit import seed_test_plan
    seed_test_plan(app,tmp_path)
    assert not app.exception and not app.error
    draft=json.loads(app.session_state['multidrug_target_text'])
    assert len(draft['points'])==2
    assert draft['profile']['example_source']['document_id']=='PMC123'
    assert 'baseline' not in draft['profile']['population']
    assert 'observed' not in draft['profile']['dlt']
    app.button(key='plan_dlt_group_add').click().run()
    assert not app.exception and not app.error
    draft=json.loads(app.session_state['multidrug_target_text'])
    assert draft['profile']['dlt']['logic_groups']
    app.selectbox(key='plan_dlt_group_0_op').set_value('OR').run()
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['dlt']['logic_groups'][0]['operator']=='OR'


def test_simulation_path_preserves_prediction_and_joint_draws():
    target=example_target();target['path_level_ids']=[]
    original={'run_id':'original','target':target,'methods':{'m':{'target':copy.deepcopy(target),'posterior_draws':[[.1,.2,.3]]}}}
    view=simulation_view(original,['L1','L3'])
    assert original['target']['path_level_ids']==[]
    assert view['run_id']=='original'
    assert view['methods']['m']['posterior_draws']==original['methods']['m']['posterior_draws']
    assert view['methods']['m']['target']['path_level_ids']==['L1','L3']
    with pytest.raises(ValueError):simulation_view(original,['L4','L1'])


def test_extraction_schema_accepts_old_and_new_versions():
    assert ReportedExtraction.model_fields['schema_version'].default=='phase1_extraction_v3.3'
    from phase1_extraction.models import DosingPhase, StructuredSchedule
    p=DosingPhase(dosing_phase_id='p',phase_label='treatment',structured=StructuredSchedule(frequency_count=1,interval_value=3,interval_unit='weeks'))
    from phase1_prediction.profiles import phase
    assert phase(p)['frequency']=='every 21 days'


def test_clearing_window_duration_removes_old_numeric_value(tmp_path,monkeypatch):
    app=app_for(tmp_path,monkeypatch)
    from test_multidrug_ui_state_audit import seed_test_plan
    seed_test_plan(app,tmp_path)
    assert app.number_input(key='plan_window_days').value==28
    app.selectbox(key='plan_window_mode').set_value('Undecided').run()
    draft=json.loads(app.session_state['multidrug_target_text'])
    assert draft['profile']['windows'][0]['days'] is None
    assert draft['profile']['windows'][0]['source_raw']
    assert not app.exception


def test_extraction_rejects_invalid_logic_and_nonpositive_schedule():
    from phase1_extraction.models import DLTDefinition, StructuredSchedule
    with pytest.raises(ValueError,match='Unknown rule'):
        DLTDefinition(definition_id='d',study_id='s',criteria_logic_groups=[{'group_id':'g','operator':'AND','children':['missing']}])
    with pytest.raises(ValueError):StructuredSchedule(frequency_count=0)


@pytest.mark.parametrize('review_mode', ['Automatic inclusion', 'Review article abstracts'])
def test_web_correspondence_is_automatic_without_explanation_panel(tmp_path,monkeypatch,review_mode):
    app=app_for(tmp_path,monkeypatch)
    from test_multidrug_ui_state_audit import seed_test_plan
    seed_test_plan(app,tmp_path)
    app.sidebar.radio[0].set_value('Review evidence').run()
    app.radio(key='workflow_screening').set_value(review_mode).run()
    next(b for b in app.button if b.label=='Find matching articles').click().run()
    row=app.session_state['multidrug_preview']['selected_series'][0]
    assert row['mapping']['pairs']
    assert not any(x.label.startswith('Source for ') or x.label=='Source regimen to map' for x in app.selectbox)
    assert not any(x.label=='Apply correspondence' for x in app.button)
    labels={x.label for x in app.expander}
    assert not labels & {'Adjust drug correspondence (advanced)', 'Detailed comparison evidence'}
    assert 'How automatic drug matching works' not in labels
    assert not any('Drug matching is automatic:' in x.value for x in app.caption)
    assert not app.exception and not app.error


def test_legacy_override_is_backed_up_and_cleared_before_matching(tmp_path,monkeypatch):
    app=app_for(tmp_path,monkeypatch)
    from test_multidrug_ui_state_audit import seed_test_plan
    seed_test_plan(app,tmp_path)
    old=json.loads(app.session_state['multidrug_target_text'])
    old['mapping_overrides']={'old-scope': {'old-target': 'old-source'}}
    app.session_state['multidrug_target_text']=json.dumps(old)
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert not app.exception and not app.error
    saved=json.loads(app.session_state['multidrug_target_text'])
    assert saved['mapping_overrides']=={}
    backup,=list((tmp_path/'state').glob('multidrug_target_draft.before_automatic_*.json'))
    assert json.loads(backup.read_text())==old
    assert json.loads((tmp_path/'state'/'multidrug_target_draft.json').read_text())==saved
    next(b for b in app.button if b.label=='Find matching articles').click().run()
    assert app.session_state['multidrug_preview']['selected_series'][0]['mapping']['pairs']
    # Imports through the JSON editor must not reintroduce hidden overrides.
    app.sidebar.radio[0].set_value('Target Trial').run()
    next(c for c in app.checkbox if c.label=='Advanced JSON editor').check().run()
    app.text_area(key='multidrug_target_json_editor').set_value(json.dumps(old)).run()
    assert not json.loads(app.session_state['multidrug_target_text'])['mapping_overrides']
    assert not app.exception and not app.error


def test_explicit_window_start_synonyms_compare_equally():
    from phase1_prediction.profile_similarity import window_pair
    comparison=window_pair({'start':'start of therapy','days':28},{'start':'First treatment dose','days':28})
    assert next(d for d in comparison['details'] if d['indicator']=='start')['score']==1
