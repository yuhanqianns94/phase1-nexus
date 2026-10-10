import copy
import json
import hashlib

import pytest
from phase1_extraction.models import Arm, DosingPhase, PopulationDescription, ConditionClause
from phase1_prediction.multidrug.observations import build_article_scopes
from phase1_prediction.multidrug import grouping
from test_cohort_exposures import extraction, write_article


def two_arms():
    e=extraction();e.cohorts=e.cohorts[:2]
    first=e.dose_series[0];first.arm_id='on'
    second=first.model_copy(update={'series_id':'off_series','arm_id':'off'})
    e.dose_series.append(second);e.cohorts[1].series_id='off_series'
    e.arms=[Arm(arm_id='on',study_id='study',label='With B',population_description='Adults with the same disease'),
            Arm(arm_id='off',study_id='study',label='Without B',population_description='Adults with the same disease')]
    e.cohorts[0].exposures[1].schedule_raw='concurrent use'
    for c in e.cohorts:
        c.exposures[0].dosing_phases=[DosingPhase(dosing_phase_id='p',phase_label='treatment',route_raw='intravenous',cycle_length_raw='28 days')]
    return e


def test_on_off_arms_share_model_keep_independent_observations(tmp_path):
    e=two_arms();paths=write_article(tmp_path,e)
    before=paths.reported_extraction.read_bytes()
    scopes,_=build_article_scopes(paths)
    assert len(scopes)==1 and len(scopes[0].observations)==2
    assert sum(o.n for o in scopes[0].observations)==6
    assert {o.exposures[1].presence for o in scopes[0].observations}=={'present','absent'}
    assert {o.provenance['arm_id'] for o in scopes[0].observations}=={'on','off'}
    assert scopes[0].drugs[1].profile['presence'] is None
    assert scopes[0].drugs[1].profile['schedule']=='concurrent use'
    assert paths.reported_extraction.read_bytes()==before
    e.cohorts.reverse();e.dose_series.reverse()
    again,_=build_article_scopes(write_article(tmp_path/'reordered',e))
    assert again[0].scope_id==scopes[0].scope_id
    assert again[0].observations==scopes[0].observations


@pytest.mark.parametrize('difference',['population','eligibility','formulation','schedule','window','regimen','unit'])
def test_incompatible_contexts_stay_separate(tmp_path,difference):
    e=two_arms()
    if difference=='population':e.arms[1].population_description='Children with another disease'
    elif difference=='eligibility':
        e.populations=[PopulationDescription(population_id='p',study_id='study',inclusion_criteria=['Organ function threshold'],
            eligibility_conditions=[ConditionClause(clause_id='c',raw_text='Organ function threshold',threshold=5,applies_to_arm_ids=['on'])])]
    elif difference=='formulation':e.cohorts[1].exposures[0].formulation_raw='different formulation'
    elif difference=='schedule':e.cohorts[1].exposures[0].dosing_phases[0].cycle_length_raw='14 days'
    elif difference=='window':e.dose_series[1].window_id='different_window'
    elif difference=='regimen':e.dose_series[1].regimen_id='other';e.cohorts[1].regimen_id='other'
    elif difference=='unit':
        e.cohorts[1].dose.unit_raw=e.cohorts[1].dose.unit_normalized='mg/kg'
        e.cohorts[1].exposures[0].dose.unit_raw=e.cohorts[1].exposures[0].dose.unit_normalized='mg/kg'
    scopes,_=build_article_scopes(write_article(tmp_path,e))
    assert len(scopes)==2


def test_explicit_population_declaration_is_source_bound_not_a_drug_exception(tmp_path,monkeypatch):
    e=two_arms();e.arms[0].population_description='Patients using companion medicine'
    e.arms[1].population_description='Patients not using companion medicine'
    for c in e.cohorts:c.exposures[1].name_raw='Any other drug group'
    paths=write_article(tmp_path,e)
    policy=tmp_path/'compatibility.json'
    declaration={'document_id':e.document_id,'source_sha256':hashlib.sha256(paths.reported_extraction.read_bytes()).hexdigest(),
                 'arm_ids':['on','off'],'group_id':'same_population','reason':'Only exposure differs'}
    policy.write_text(json.dumps({'groups':[declaration]}));monkeypatch.setattr(grouping,'POLICY_PATH',policy)
    scopes,_=build_article_scopes(paths)
    assert len(scopes)==1
    e.arms[1].population_description+='; now a different population'
    scopes,notices=build_article_scopes(write_article(tmp_path,e))
    assert len(scopes)==2
    assert notices[0]['reason']=='scope_compatibility_source_changed'


def test_absent_drug_does_not_bridge_incompatible_active_schedules(tmp_path):
    e=two_arms()
    third=e.cohorts[0].model_copy(deep=True);third.cohort_id='c3';third.exposures[1].schedule_raw='different regimen'
    e.cohorts.append(third)
    scopes,_=build_article_scopes(write_article(tmp_path,e))
    assert len(scopes)==3 # Absent-B row matches both, so must not choose one arbitrarily.


def test_unknown_population_text_does_not_authorize_cross_arm_merge(tmp_path):
    e=two_arms()
    for arm in e.arms:arm.population_description='not reported'
    scopes,_=build_article_scopes(write_article(tmp_path,e))
    assert len(scopes)==2


def test_omitted_exposure_is_not_an_absence_schedule_wildcard(tmp_path):
    e=two_arms();e.cohorts[1].exposures=e.cohorts[1].exposures[:1]
    scopes,_=build_article_scopes(write_article(tmp_path,e))
    assert len(scopes)==2


def test_regrouping_does_not_silently_ignore_a_saved_manual_mapping(monkeypatch):
    from types import SimpleNamespace
    from test_multidrug_transport import fixture
    from phase1_prediction.multidrug import service
    scope,_,_,target=fixture()
    target.mapping_overrides={'old_scope':{'A':'A'}}
    monkeypatch.setattr(service,'load_model_library',lambda *a,**kw:([scope],{}))
    library=SimpleNamespace(source_roots=(),evidence_policy='project_default',review_policy='extracted')
    with pytest.raises(ValueError,match='regrouped source scope'):
        service._select(library,target,None,0,None)
