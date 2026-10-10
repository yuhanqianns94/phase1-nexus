"""Administrative matching views and lighter forms retain original provenance."""
import copy

import pytest

from phase1_prediction.evidence_relevance import toxicity_eligibility
from test_friendly_clinical_form import _draft, _form_app


def _consent():
    return {'clause_id':'consent', 'raw':'Ability to provide written informed consent',
            'topic':'ability to provide written informed consent', 'source_id':'page-2'}


def _renal():
    return {'clause_id':'renal', 'raw':'Creatinine clearance >= 60 mL/min',
            'topic':'Creatinine clearance', 'operator':'>=', 'threshold':60.,
            'unit':'mL/min', 'source_id':'page-2'}


@pytest.mark.parametrize('text', [
    'Ability to provide written informed consent',
    'Willing and able to provide written informed consent.',
    'Written informed consent must be obtained before enrollment',
    'Patients must sign written informed consent',
    'Willingness to participate in the trial',
])
def test_only_complete_administrative_statements_are_removed(text):
    source={'inclusion':[{'raw':text},_renal()], 'source_id':'eligibility-table'}
    before=copy.deepcopy(source)
    assert toxicity_eligibility(source)=={'inclusion':[_renal()], 'source_id':'eligibility-table'}
    assert source==before


@pytest.mark.parametrize('rule', [
    {'raw':'Age >= 18 years on the day informed consent is signed'},
    {'raw':'Written informed consent and creatinine clearance >= 60 mL/min'},
    {'raw':'Unable to consent because of cognitive impairment'},
    {'raw':'Able to swallow tablets and provide written consent'},
    {'raw':'Consent to pregnancy testing and effective contraception'},
    {'raw':'Ability to provide written informed consent', 'metric_code':'age', 'threshold':18.},
    {'raw':'Ability to provide written informed consent', 'negated':True},
    {'raw':'No active uncontrolled infection'},
])
def test_clinical_mixed_and_unknown_requirements_remain(rule):
    source={'inclusion':[rule]}
    assert toxicity_eligibility(source)==source


def test_and_pruning_is_recursive_but_or_dependencies_remain_intact():
    source={'inclusion':[_consent(),_renal()], 'logic_groups':[
        {'group_id':'admin','operator':'AND','children':['consent'],'source_id':'paragraph-1'},
        {'group_id':'all','operator':'AND','children':['admin','renal'],'source_id':'paragraph-1'},
    ]}
    original=copy.deepcopy(source)
    assert toxicity_eligibility(source)=={'inclusion':[_renal()], 'logic_groups':[]}
    assert source==original
    source['logic_groups'][1]['operator']='OR'
    assert toxicity_eligibility(source)==source
    source['logic_groups'][1]['operator']='AND'
    source['logic_groups'][1]['negated']=True
    assert toxicity_eligibility(source)==source


def test_legacy_or_or_unresolved_reference_is_kept_without_guessing():
    source={'inclusion':[_consent(),dict(_renal(),logic='OR')]}
    assert toxicity_eligibility(source)==source
    source={'inclusion':[_consent()], 'logic_groups':[
        {'group_id':'g','operator':'AND','children':['consent','unknown-source-clause']} ]}
    assert toxicity_eligibility(source)==source


def test_nested_single_and_wrappers_resolve_to_leaf_without_dangling_ids():
    source={'inclusion':[_consent(),_renal(),{'clause_id':'age','raw':'Age >= 18 years'}],
            'logic_groups':[
                {'group_id':'inner','operator':'AND','children':['renal','consent']},
                {'group_id':'middle','operator':'AND','children':['inner']},
                {'group_id':'root','operator':'AND','children':['middle','age']},
            ]}
    result=toxicity_eligibility(source)
    assert result['logic_groups']==[{'group_id':'root','operator':'AND','children':['renal','age']}]


def test_hidden_descriptive_metadata_survives_no_edit_and_clinical_edit():
    draft=_draft()
    draft['drugs'][0]['research_role']='co_primary'
    draft['drugs'][0]['profile'].update(role='maintenance', role_code='maintenance',
        phases=[{'phase':'Lead-in X','phase_code':'lead_in','route':'oral','source_id':'stage-x'}])
    draft['profile'].update(population={'purpose':'Determine a recommended dose'},
        configurations=[{'phase':'I','role':'escalation','role_code':'dose_escalation'}],
        dlt={'version':'Protocol definition 3','grading':'CTCAE version 5.0'})
    original=copy.deepcopy(draft)
    app=_form_app(draft)
    controls={w.label for name in ('text_input','text_area','selectbox') for w in getattr(app,name)}
    assert not controls & {'Study objective','Research role','Treatment role','Study part',
                           'DLT definition label / version (if specified)','Treatment stage'}
    assert {'Type of intervention','Dose represents','Planned study phase','Grading version'} <= controls
    assert any('Lead-in X' in item.value for item in app.markdown)
    app.text_area(key='plan_pop_disease').set_value('Advanced solid tumors').run()
    result=app.session_state['result']
    assert result['drugs'][0]['research_role']=='co_primary'
    assert result['drugs'][0]['profile']==original['drugs'][0]['profile']|{'name':'Drug A'}
    assert result['profile']['population']['purpose']=='Determine a recommended dose'
    assert result['profile']['configurations']==original['profile']['configurations']
    assert result['profile']['dlt']==original['profile']['dlt']
    assert draft==original
    assert not app.exception


def test_hidden_consent_and_ids_groups_survive_editing_visible_rule():
    draft=_draft()
    source={'inclusion':[_consent(),_renal()], 'logic_groups':[
        {'group_id':'g','operator':'AND','children':['consent','renal'],'source_id':'p2'}]}
    draft['profile']['eligibility']=copy.deepcopy(source)
    app=_form_app(draft)
    assert app.session_state['result']['profile']['eligibility']==source
    assert not any(w.key=='plan_elig_inclusion_0_raw' for w in app.text_area)
    assert app.text_area(key='plan_elig_inclusion_1_raw').value==_renal()['raw']
    members=app.multiselect(key='plan_elig_group_0_children')
    assert all('consent' not in name.lower() for name in members.options)
    assert not any(w.key=='plan_elig_group_0_op' for w in app.selectbox)
    app.number_input(key='plan_elig_inclusion_1_threshold').set_value(75.).run()
    result=app.session_state['result']['profile']['eligibility']
    assert result['inclusion'][0]==source['inclusion'][0]
    assert result['logic_groups']==source['logic_groups']
    assert result['inclusion'][1]['threshold']==75.
    assert result['inclusion'][1]['clause_id']=='renal'
    assert result['inclusion'][1]['source_id']=='page-2'
    assert draft['profile']['eligibility']==source
    assert not app.exception


def test_administrative_rules_without_ids_are_not_added_to_new_clinical_groups():
    draft=_draft()
    consent=_consent();consent.pop('clause_id')
    draft['profile']['eligibility']={'inclusion':[consent,_renal()]}
    app=_form_app(draft)
    app.button(key='plan_elig_group_add').click().run()
    result=app.session_state['result']['profile']['eligibility']
    assert result['inclusion'][0]==consent
    assert result['logic_groups'][0]['children']==['renal']
    assert not app.exception


def test_only_hidden_administrative_rules_cannot_create_empty_clinical_group():
    draft=_draft();draft['profile']['eligibility']={'inclusion':[_consent()]}
    app=_form_app(draft)
    assert app.button(key='plan_elig_group_add').disabled
    assert app.session_state['result']['profile']['eligibility']==draft['profile']['eligibility']
