"""Closed extraction choices are closed in Plan; prose remains lossless prose."""
import copy

from phase1_extraction.controlled_vocabulary import choices, labels
from phase1_prediction.multidrug.contracts import TargetRegimen
from test_friendly_clinical_form import _draft, _form_app, _phase


def test_imported_clinical_sentence_is_never_a_dropdown_candidate():
    draft=_draft()
    sentence='If a patient developed DLT and toxicity resolved, resume at the previous dose level'
    original={'raw':sentence,'topic':sentence,'source_id':'source-1'}
    draft['drugs'][0]['profile']['adjustments']=[copy.deepcopy(original)]
    app=_form_app(draft)
    topic=app.selectbox(key='plan_A_rules_0_topic')
    assert topic.value=='Other' and sentence not in topic.options
    assert app.text_input(key='plan_A_rules_0_topic_custom').value==sentence
    assert app.session_state['result']['drugs'][0]['profile']['adjustments']==[original]
    assert draft['drugs'][0]['profile']['adjustments']==[original]
    assert all(sentence not in widget.options for widget in app.selectbox)


def test_aliases_choose_closed_controls_without_overwriting_source_values():
    draft=_draft()
    original={'route':'IV','form':'capsules','phase':'lead-in','frequency':'q3w',
              'source_id':'stage-original'}
    draft['drugs'][0]['profile']['phases']=[copy.deepcopy(original)]
    draft['profile']['eligibility']={'inclusion':[{'raw':'ANC >= 1500 cells/uL','topic':'ANC',
        'operator':'≥','threshold':1500.,'unit':'cells/uL'}]}
    before=copy.deepcopy(draft)
    app=_form_app(draft)
    assert app.selectbox(key='plan_A_phase_0_route').value=='intravenous'
    assert app.selectbox(key='plan_A_phase_0_form').value=='capsule'
    assert app.selectbox(key='plan_A_phase_0_freq').value=='every 21 days'
    assert app.selectbox(key='plan_elig_inclusion_0_topic').value=='Absolute neutrophil count'
    assert app.selectbox(key='plan_elig_inclusion_0_op').value=='>='
    assert _phase(app)==original
    assert app.session_state['result']['profile']['eligibility']==before['profile']['eligibility']
    assert draft==before


def test_sex_has_only_three_business_choices_and_unknown_never_means_unrestricted():
    draft=_draft();draft['profile']['population']={'sex':['Not reported']}
    app=_form_app(draft)
    selector=app.selectbox(key='plan_pop_sex')
    assert set(selector.options)=={'Undecided','Female','Male','No restriction'}
    assert selector.value=='Undecided'
    assert app.session_state['result']['profile']['population']['sex']==['Not reported']
    selector.set_value('Male').run()
    app.selectbox(key='plan_pop_sex').set_value('Undecided').run()
    assert app.session_state['result']['profile']['population']['sex']==[]
    assert draft['profile']['population']['sex']==['Not reported']


def test_scientific_names_are_text_and_modifiers_are_closed():
    draft=_draft()
    draft['drugs'][0]['profile'].update(mechanisms=['Novel kinase inhibition'],targets=['Target X'])
    draft['profile']['population']={'disease':['Uncommon disease X'],'modifiers':['advanced','Locally unusual disease']}
    app=_form_app(draft)
    text_keys={widget.key for widget in app.text_area}
    assert {'plan_A_mechanisms','plan_A_targets','plan_A_classes','plan_pop_disease'}<=text_keys
    modifier=app.multiselect(key='plan_pop_modifiers')
    assert set(modifier.options)==set(labels('disease_modifier').values())
    assert 'Locally unusual disease' not in modifier.options
    assert app.text_area(key='plan_pop_modifiers_other').value=='Locally unusual disease'
    assert app.session_state['result']['profile']['population']==draft['profile']['population']


def test_structured_edit_refreshes_authoritative_codes_and_preserves_full_wording():
    draft=_draft()
    raw='Hemoglobin >= 9 g/dL without transfusion in the previous week'
    draft['profile']['eligibility']={'inclusion':[{'raw':raw,'topic':'Hemoglobin',
        'metric_code':'hemoglobin','operator':'>=','comparator_code':'>=',
        'threshold':9.,'unit':'g/dL','unit_code':'g/dL','source_id':'original'}]}
    app=_form_app(draft)
    app.selectbox(key='plan_elig_inclusion_0_topic').set_value('Creatinine clearance')
    app.selectbox(key='plan_elig_inclusion_0_op').set_value('<=')
    app.selectbox(key='plan_elig_inclusion_0_unit').set_value('mL/min').run()
    rule=app.session_state['result']['profile']['eligibility']['inclusion'][0]
    assert rule['metric_code']=='creatinine_clearance'
    assert rule['comparator_code']=='<=' and rule['unit_code']=='mL/min'
    assert rule['raw']==raw and rule['source_id']=='original'
    app.selectbox(key='plan_elig_inclusion_0_topic').set_value('Other').run()
    app.text_input(key='plan_elig_inclusion_0_topic_custom').set_value('Novel laboratory measure').run()
    rule=app.session_state['result']['profile']['eligibility']['inclusion'][0]
    assert rule['metric_code']=='other' and rule['topic']=='Novel laboratory measure'


def test_grading_system_and_version_are_separate_closed_fields():
    draft=_draft();draft['profile']['dlt']={'grading':'CTCAE version 5.0','grading_version':'5.0'}
    app=_form_app(draft)
    assert app.selectbox(key='plan_grading').value=='CTCAE'
    assert app.selectbox(key='plan_grading_version').value=='5.0'
    assert app.session_state['result']['profile']['dlt']==draft['profile']['dlt']
    app.selectbox(key='plan_grading_version').set_value('4.03').run()
    dlt=app.session_state['result']['profile']['dlt']
    assert dlt['grading_system']=='ctcae' and dlt['grading_version']=='4.03'
    assert dlt['grading']=='CTCAE version 4.03'
    assert all('CTCAE version 5.0' not in item.options for item in app.selectbox)


def test_age_bound_operators_are_optional_and_explicit():
    app=_form_app()
    assert not any(widget.key=='plan_age_min_operator' for widget in app.selectbox)
    app.number_input(key='plan_age_min').set_value(18.).run()
    assert app.selectbox(key='plan_age_min_operator').value=='Undecided'
    assert not app.session_state['result']['profile']['population'].get('age_min_operator')
    app.selectbox(key='plan_age_min_operator').set_value('>').run()
    assert app.session_state['result']['profile']['population']['age_min_operator']=='>'


def test_new_closed_dose_units_still_validate_complete_target_contract():
    draft=_draft();draft['points']=[{'level_id':'L1','doses':{'A':10.}}]
    draft['drugs'][0].update(mode='numeric',reference_dose=10.)
    app=_form_app(draft)
    app.selectbox(key='plan_A_unit').set_value('micrograms/kg').run()
    saved=app.session_state['result']
    assert TargetRegimen.model_validate(saved).drugs[0].unit=='micrograms/kg'
    assert saved['drugs'][0]['profile']['dose_unit_code']=='micrograms/kg'
    assert 'unit_code' not in saved['drugs'][0]


def test_frequency_pattern_and_numeric_interval_are_independent_of_source_text():
    draft=_draft();draft['drugs'][0]['profile']['phases']=[{}]
    app=_form_app(draft)
    app.selectbox(key='plan_A_phase_0_frequency_pattern').set_value('Intermittent').run()
    app.selectbox(key='plan_A_phase_0_freq').set_value('Custom interval').run()
    app.number_input(key='plan_A_phase_0_freq_n').set_value(2.)
    app.number_input(key='plan_A_phase_0_interval').set_value(3.)
    app.selectbox(key='plan_A_phase_0_interval_unit').set_value('weeks').run()
    stage=_phase(app)
    assert stage['structured']['frequency_pattern']=='intermittent'
    assert stage['structured']['frequency_count']==2.
    assert stage['frequency']=='2 times every 21 days'
    assert not app.exception


def test_adjustment_action_uses_closed_pool_and_keeps_complex_condition():
    draft=_draft()
    raw='If grade 3 toxicity resolves within 48 hours, resume at previous dose'
    draft['drugs'][0]['profile']['adjustments']=[{'raw':raw,'topic':'DLT','action':'resume'}]
    app=_form_app(draft)
    action=app.selectbox(key='plan_A_rules_0_action')
    assert set(action.options)=={'Not specified',*labels('adjustment_action').values()}
    action.set_value('Reduce').run()
    rule=app.session_state['result']['drugs'][0]['profile']['adjustments'][0]
    assert rule['action']=='reduce' and rule['raw']==raw


def test_imported_window_anchor_keeps_specified_drug_wording_until_selection():
    draft=_draft()
    original={'start':'First dose of Drug A','start_anchor':'first_specified_drug_dose','days':28.}
    draft['profile']['windows']=[copy.deepcopy(original)]
    app=_form_app(draft)
    assert app.selectbox(key='plan_window_start').value=='First dose of a specified drug'
    assert app.session_state['result']['profile']['windows'][0]==original
    app.selectbox(key='plan_window_start').set_value('Last treatment dose').run()
    window=app.session_state['result']['profile']['windows'][0]
    assert window['start']=='Last treatment dose' and window['start_anchor']=='last_treatment_dose'
    assert draft['profile']['windows']==[original]


def test_direct_tab_click_and_previous_next_share_one_navigation_state():
    from phase1_web import plan_navigation
    app=_form_app()
    # Streamlit AppTest does not expose a Tab.click method. Deliver the same
    # widget state/callback as a browser tab click, then rerun the real form.
    app.session_state[plan_navigation.TAB_KEY]='3. Population'
    app.session_state[plan_navigation.STEP_KEY]='3. Population'
    app.run()
    assert any('Section 3 of 6' in caption.value for caption in app.caption)
    assert not app.button(key='trial_plan_previous_top').disabled
    assert app.button(key='trial_plan_next_top').label=='Next: DLT'
    app.button(key='trial_plan_next_top').click().run()
    assert app.session_state[plan_navigation.STEP_KEY]=='4. DLT'
    assert app.session_state[plan_navigation.TAB_KEY]=='4. DLT'
    assert any('Section 4 of 6' in caption.value for caption in app.caption)


def test_tab_callback_commits_selected_page_before_progress_is_rendered(monkeypatch):
    from types import SimpleNamespace
    from phase1_web import plan_navigation
    state={plan_navigation.STEP_KEY:'1. Drugs',plan_navigation.TAB_KEY:'3. Population'}
    monkeypatch.setattr(plan_navigation,'st',SimpleNamespace(session_state=state))
    plan_navigation._tab_selected()
    assert state[plan_navigation.STEP_KEY]=='3. Population'
