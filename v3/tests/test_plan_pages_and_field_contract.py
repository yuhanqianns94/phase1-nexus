"""Prospective controls must round-trip into the same fields as extraction."""
import copy
import json

from phase1_extraction.models import (CohortExposure, DosingPhase, StructuredSchedule,
                                      Intervention, ConditionClause)
from phase1_prediction.profiles import exposure_profile
from phase1_prediction.profile_similarity import (admin_pair, condition, condition_pair,
                                                 drug_pair, compare_profiles)
from test_friendly_clinical_form import _draft, _form_app


def test_next_previous_and_structural_edits_preserve_all_clinical_pages():
    app=_form_app()
    app.text_input(key='plan_A_name').set_value('Investigational A')
    app.button(key='trial_plan_next').click().run()
    assert app.session_state['trial_plan_step']=='2. Administration'
    assert app.session_state['result']['drugs'][0]['name']=='Investigational A'
    app.button(key='plan_A_phase_add').click().run()
    assert app.session_state['trial_plan_step']=='2. Administration'
    app.selectbox(key='plan_A_phase_0_route').set_value('oral')
    app.button(key='trial_plan_next').click().run()
    assert app.session_state['trial_plan_step']=='3. Population'
    app.number_input(key='plan_age_min').set_value(18.)
    app.button(key='trial_plan_previous').click().run()
    result=app.session_state['result']
    assert result['profile']['population']['age_min']==18.
    assert result['drugs'][0]['profile']['phases'][0]['route']=='oral'
    assert result['points']==[]
    assert not app.exception


def test_extracted_drug_and_stage_fields_have_equivalent_form_representation():
    source=CohortExposure(exposure_id='e',name_raw='kinase inhibitor',entity_kind='drug_class',
        intervention_id='i',role='fixed_backbone',dosing_phases=[DosingPhase(
        dosing_phase_id='stage',phase_label='induction',route_raw='oral',dose_form_raw='tablet',
        structured=StructuredSchedule(frequency_count=1,interval_value=1,interval_unit='days',
            cycle_value=4,cycle_unit='weeks',administration_days=[1,2,3,4,5],dose_basis='daily_total'))])
    intervention=Intervention(intervention_id='i',name_raw='kinase inhibitor',
        mechanism_raw=['kinase inhibition'],targets_raw=['KIT'],drug_class_raw='small molecule')
    untouched=source.model_dump()
    reported=exposure_profile(source,{'i':intervention.model_dump()}, {})
    draft=_draft();draft['drugs'][0].update(name='kinase inhibitor',profile=copy.deepcopy(reported))
    app=_form_app(draft)
    assert app.selectbox(key='plan_A_kind').value=='Drug class'
    assert app.selectbox(key='plan_A_basis').value=='daily_total'
    assert app.session_state['result']['drugs'][0]['profile']==reported
    app.selectbox(key='plan_A_basis').set_value('per_administration').run()
    app.selectbox(key='plan_A_kind').set_value('Named drug').run()
    edited=app.session_state['result']['drugs'][0]['profile']
    assert edited['kind']=='drug' and edited['dose_basis']=='per_administration'
    assert next(row for row in admin_pair(edited,reported)['details'] if row['indicator']=='Dose basis')['score']==0
    assert source.model_dump()==untouched
    assert not app.exception


def test_negation_reaches_comparison_and_definition_version_is_preserved_metadata():
    source=ConditionClause(clause_id='c',raw_text='Prior therapy',metric='Prior therapy',negated=False)
    reported=condition(source.model_dump())
    draft=_draft();draft['profile']['dlt']={'criteria':[reported],'version':'Protocol A'}
    app=_form_app(draft)
    app.selectbox(key='plan_dlt_criteria_0_negated').set_value('NOT (the condition is negated)').run()
    assert not any(widget.key=='plan_dlt_version' for widget in app.text_input)
    dlt=app.session_state['result']['profile']['dlt']
    assert dlt['criteria'][0]['negated'] is True
    assert next(row for row in condition_pair(dlt['criteria'][0],reported)['details'] if row['indicator']=='negated')['score']==0
    compared=compare_profiles({'dlt':dlt},{'dlt':draft['profile']['dlt']})
    assert dlt['version']=='Protocol A'
    assert not any(row['indicator']=='Definition version' for row in compared['groups']['dlt']['details'])
    assert source.negated is False
    assert not app.exception


def test_unused_fields_are_not_requested_and_legacy_content_is_preserved():
    draft=_draft()
    draft['profile']['planning'].update(evaluability='legacy evaluability',safety_monitoring='legacy monitoring')
    draft['drugs'][0]['profile']['phases']=[{'formulation':'legacy stage formulation'}]
    app=_form_app(draft)
    keys={item.key for group in (app.text_input,app.text_area,app.number_input,app.checkbox,app.selectbox) for item in group}
    assert not {'plan_evaluability','plan_safety','plan_A_phase_0_formulation','plan_A_ref','plan_A_transport','plan_A_anchor','plan_name'} & keys
    assert not app.expander
    result=app.session_state['result']
    assert result['profile']['planning']['evaluability']=='legacy evaluability'
    assert result['drugs'][0]['profile']['phases'][0]['formulation']=='legacy stage formulation'


def test_source_reporting_scope_and_use_do_not_change_prospective_drug_similarity():
    a={'name':'A','kind':'drug','presence':'present','reporting_scope':'cohort'}
    b={**a,'presence':'mixed','reporting_scope':'pooled'}
    assert drug_pair(a,b,prospective=True)==drug_pair(a,a,prospective=True)
    assert drug_pair(a,b)['score'] < drug_pair(a,a)['score']


def test_each_assessment_window_is_editable_and_keeps_its_cycle_link():
    draft=_draft()
    draft['drugs'][0]['profile']['phases']=[{'cycle':'28 days'}]
    draft['profile']['windows']=[{'kind':'formal_dlt','days':28.},
                                  {'kind':'safety_monitoring','days':84.}]
    app=_form_app(draft)
    app.number_input(key='plan_window_1_window_days').set_value(112.).run()
    assert app.session_state['result']['profile']['windows'][0]==draft['profile']['windows'][0]
    assert app.session_state['result']['profile']['windows'][1]['days']==112.
    app.selectbox(key='plan_window_1_window_mode').set_value('Number of cycles').run()
    app.selectbox(key='plan_window_1_window_cycle_source').set_value(json.dumps(['A',0]))
    app.number_input(key='plan_window_1_window_cycles').set_value(2.).run()
    assert app.session_state['result']['profile']['windows'][1]['days']==56.
    app.number_input(key='plan_A_phase_0_cycle').set_value(21.).run()
    assert app.session_state['result']['profile']['windows'][1]['days']==42.
    app.button(key='plan_window_1_remove').click().run()
    assert app.session_state['result']['profile']['windows']==draft['profile']['windows'][:1]
    assert not app.exception


def test_rules_are_paged_without_losing_edits_or_other_rule_metadata():
    draft=_draft()
    rules=[{'raw':'Hemoglobin >= 9 g/dL','topic':'Hemoglobin','threshold':9.,'operator':'>=','unit':'g/dL'},
           {'raw':'Prior therapy','topic':'Prior therapy','source_id':'retain'}]
    draft['profile']['eligibility']={'inclusion':copy.deepcopy(rules)}
    app=_form_app(draft)
    assert len([x for x in app.text_area if str(x.key).startswith('plan_elig_inclusion')])==1
    app.number_input(key='plan_elig_inclusion_0_threshold').set_value(10.).run()
    app.button(key='plan_elig_inclusion_next_rule').click().run()
    app.text_area(key='plan_elig_inclusion_1_raw').set_value('No prior therapy').run()
    app.button(key='plan_elig_inclusion_previous_rule').click().run()
    result=app.session_state['result']['profile']['eligibility']['inclusion']
    assert result[0]['threshold']==10.
    assert result[1]['raw']=='No prior therapy' and result[1]['source_id']=='retain'
    assert app.number_input(key='plan_elig_inclusion_0_threshold').value==10.
    assert not app.exception
