"""Regression cases from the September audit, including independent numerics."""
import copy
import json
from itertools import combinations
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.special import expit, logit
from streamlit.testing.v1 import AppTest

from phase1_prediction.multidrug import TARGET_PREDICTION_VERSION, VERSION
from phase1_prediction.multidrug.contracts import TargetRegimen
from phase1_prediction.multidrug.mapping import match_target, mapped_relative_dose, target_profile
from phase1_prediction.multidrug.math import interaction_basis
from phase1_prediction.multidrug.transport import assemble_components, transport_joint_posteriors
from phase1_prediction.profile_similarity import drug_affinity, compare_mapped_profiles
from phase1_prediction.profiles import exposure_profile
from phase1_project.analyses import load_analysis
from phase1_simulation.multidrug import path_matrix
from phase1_web.multidrug_form import prepare_drug_rows
from test_multidrug_transport import fixture
from test_multidrug_role_matching import three_papers
from test_multidrug_revisions import detailed_target


def independent_basis(r):
    terms=[]
    for n in (2,3):
        for group in combinations(range(min(3,r.shape[1])),n):
            product=np.prod(r[:,group],axis=1)
            terms.append(2*product/(1+product))
    return np.array(terms).T


@pytest.mark.parametrize('k',[2,3,4])
@pytest.mark.parametrize('levels',[1,2,3])
def test_interaction_basis_matches_formula_in_constant_and_dynamic_jit(k,levels):
    r=np.array([[.5,.3,.9,.7],[1.,.6,1.8,1.4],[2.,1.2,3.6,2.8]])[:levels,:k]
    expected=independent_basis(r)
    for actual in (interaction_basis(r),jax.jit(lambda:interaction_basis(r))(),jax.jit(interaction_basis)(r)):
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('k',[2,3,4])
def test_nonzero_interactions_joint_assembly_and_formula_agree_after_mapping(k):
    scope,arrays,manifest,target=fixture(k)
    a=np.array([-2.,-1.,-3.,-2.5])[:k]
    b=np.array([.3,.7,-.2,.1])[:k]
    eta=np.array([.8,-.4,.2,-.3])[:arrays['eta'].shape[-1]]
    arrays['a'][:]=a; arrays['log_b'][:]=b; arrays['eta'][:]=eta
    mapping={target.drugs[i].drug_id:scope.drugs[(i+1)%min(k,3) if i<3 else i].model_drug_id for i in range(k)}
    target.mapping_overrides={scope.scope_id:mapping}
    r=np.zeros((len(target.points),k))
    for i,p in enumerate(target.points):
        for j,d in enumerate(target.drugs):
            p.doses[d.drug_id]*=(j+1)/2
            source_index=next(x for x,sd in enumerate(scope.drugs) if sd.model_drug_id==mapping[d.drug_id])
            r[i,source_index]=p.doses[d.drug_id]/d.reference_dose
    q=expit(a+np.exp(b)*np.log(r))
    expected=expit(logit(1-np.prod(1-q,axis=1))+independent_basis(r)@eta)
    for result in (transport_joint_posteriors([(scope,arrays,manifest)],target,draw_count=4),
                   assemble_components([(scope,arrays,manifest)],target,draw_count=4)):
        assert result['status']=='fitted'
        np.testing.assert_allclose(result['posterior_draws'],np.tile(expected,(4,1)),rtol=1e-11)


def test_declared_parenthetical_alias_retains_metadata_but_unrelated_rename_does_not():
    intervention={'name_raw':'sepantronium bromide','aliases':['YM155','S'],
                  'mechanism_raw':['survivin suppression'],'targets_raw':['survivin'],
                  'pathways_raw':['apoptosis'],'drug_class_raw':'survivin suppressant'}
    raw={'name_raw':'sepantronium bromide (YM155)','intervention_id':'i1'}
    p=exposure_profile(raw,{'i1':intervention},{})
    assert p['mechanisms']==['survivin suppression'] and p['targets']==['survivin']
    assert 'sepantronium bromide' in p['aliases'] and 'YM155' in p['aliases']
    assert raw['name_raw']=='sepantronium bromide (YM155)'
    wrong=exposure_profile({**raw,'name_raw':'unrelated drug (YM155)'},{'i1':intervention},{})
    assert not wrong['mechanisms'] and not wrong['aliases']


def test_research_role_and_original_treatment_role_remain_separate():
    scope,_,_,target=fixture(2)
    for profiles in ([d.profile for d in scope.drugs],[d.profile for d in target.drugs]):
        profiles[1]['role']='fixed_backbone'
        profiles[1]['research_role']='partner'
    p=target_profile(target)['configurations'][0]['drugs'][1]
    assert p['role']=='fixed_backbone' and p['research_role']=='partner'
    assert drug_affinity(p,scope.drugs[1].profile)==1


@pytest.mark.parametrize('kind',[None,'drug_class'])
def test_confirmed_mapping_compares_administration_across_role_and_kind(kind):
    scope,_,_,target=fixture(1)
    target.drugs[0].name='D'
    target.drugs[0].profile.pop('kind',None)
    if kind:target.drugs[0].profile['kind']=kind
    phase={'route':'IV','frequency':'q3w','cycle':'21 days'}
    target.drugs[0].profile.update(role='escalated',phases=[phase])
    scope.drugs[0].profile.update(role='fixed_backbone',phases=[dict(phase)],kind='drug')
    scope.profile['configurations'][0]['drugs']=[d.profile for d in scope.drugs]
    mapping=match_target(target,scope)
    assert mapping['pairs']=={'A':'A'}
    admin=mapping['score']['groups']['administration']
    assert admin['score']==1 and 'coverage' not in admin
    scope.drugs[0].profile['phases'][0]['frequency']='weekly'
    assert match_target(target,scope)['score']['groups']['administration']['score']<1


def test_population_similarity_cannot_admit_unrelated_drug_but_manual_mapping_can():
    entries,target=three_papers()
    scope,arrays,manifest=entries[2]
    target.drugs=target.drugs[:1]
    target.points=[p.model_copy(update={'doses':{'A':p.doses['A']}}) for p in target.points]
    target.profile['population']=scope.profile['population']={'disease':['same disease'],'purpose':'same purpose'}
    mapping=match_target(target,scope)
    assert not mapping['pairs'] and mapping['rejected_pairs']
    assert assemble_components([(scope,arrays,manifest)],target,draw_count=2)['status']=='no_similarity_support'
    target.mapping_overrides={scope.scope_id:{'A':'H'}}
    assert match_target(target,scope)['pairs']=={'A':'H'}
    assert assemble_components([(scope,arrays,manifest)],target,draw_count=2)['status']=='fitted'


@pytest.mark.parametrize('value',[25,50,100])
def test_background_dose_is_not_silently_ignored(value):
    scope,_,_,target=fixture(1)
    payload=target.model_dump()
    payload['drugs'][0].update(mode='background',reference_dose=None,unit=None)
    payload['points'][0]['doses']['A']=value
    with pytest.raises(ValueError,match='dose-unknown'):
        TargetRegimen.model_validate(payload)
    target.drugs[0].mode='background';scope.drugs[0].mode='background'
    with pytest.raises(ValueError,match='Background target'):
        mapped_relative_dose(target.drugs[0],scope.drugs[0],value)


def test_strict_unit_conversion_never_degrades_to_background_anchoring():
    scope,_,_,target=fixture(1)
    drug=target.drugs[0];drug.transport_mode='same_entity_unit_conversion'
    scope.drugs[0].mode='background'
    with pytest.raises(ValueError,match='known numeric doses'):
        mapped_relative_dose(drug,scope.drugs[0],50)
    drug.allow_background_anchor=True
    with pytest.raises(ValueError,match='background'):
        TargetRegimen.model_validate(target.model_dump())
    drug.transport_mode='relative_dose_analogy'
    assert mapped_relative_dose(drug,scope.drugs[0],100)==2


def test_existing_drug_ids_are_locked_and_new_rows_get_unused_ids():
    initial=detailed_target()
    with pytest.raises(ValueError,match='IDs cannot be changed'):
        prepare_drug_rows([{'__original_id':'D','ID':'renamed_D','Drug name':'D'}],initial['drugs'])
    rows=prepare_drug_rows([{'__original_id':'D','ID':'D','Drug name':'Renamed display'},
                            {'Drug name':'New drug'}],initial['drugs'])
    assert rows[0]['ID']=='D' and rows[1]['ID']=='drug_1'


def test_drug_display_name_edit_preserves_doses_and_phases():
    app=AppTest.from_string('import streamlit as st\nfrom phase1_web.multidrug_form import render\nst.session_state["result"]=render(st.session_state["initial"]).model_dump(mode="json")')
    initial=detailed_target();app.session_state['initial']=copy.deepcopy(initial);app.run()
    app.text_input(key='plan_D_name').set_value('Updated name').run()
    assert not app.exception
    result=app.session_state['result']
    assert result['drugs'][0]['name']=='Updated name'
    assert result['drugs'][0]['profile']['phases']==initial['drugs'][0]['profile']['phases']
    assert result['points']==initial['points']


def test_search_survives_web_navigation_and_saves_with_prediction(tmp_path,monkeypatch):
    from test_prediction_pipeline import _article
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    scope,arrays,manifest,target=fixture(2)
    method=transport_joint_posteriors([(scope,arrays,manifest)],target,draw_count=4)
    key=method['synthesis']
    prediction={'format':'phase1_v3_multidrug_prediction_v1','run_id':'prediction_audit_regression','model_version':VERSION,
                'target':target.model_dump(mode='json'),'methods':{key:method},'evidence_contract':{}}
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30)
    app.session_state['multidrug_target_text']=json.dumps(target.model_dump(mode='json'))
    app.session_state['explore_goal']='Compare planned regimens'
    app.run()
    app.session_state['multidrug_prediction']=prediction
    app.session_state['multidrug_results']={'logistic':prediction}
    app.sidebar.radio[0].set_value('OC Simulation').run()
    next(x for x in app.number_input if x.label=='Posterior scenarios').set_value(1).run()
    next(x for x in app.number_input if x.label=='Trials per scenario').set_value(2).run()
    next(x for x in app.multiselect if x.label=='Candidate maximum sample sizes').set_value([12]).run()
    next(x for x in app.button if x.label=='Compare designs').click().run()
    assert not app.exception and not app.error
    search=app.session_state['multidrug_search']
    assert search['prediction_run_id']==prediction['run_id'] and search['run_id']
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.session_state['multidrug_search']['run_id']==search['run_id']
    next(x for x in app.button if x.label=='Save analysis').click().run()
    assert not app.exception and not app.error
    record=load_analysis(next((tmp_path/'state'/'analyses').glob('*/analysis.json')))
    assert record['design_comparison']['run_id']==search['run_id']


def test_old_target_predictions_cannot_seed_new_simulations():
    scope,arrays,manifest,target=fixture(2)
    method=transport_joint_posteriors([(scope,arrays,manifest)],target,draw_count=2)
    assert method['prediction_engine_version']==TARGET_PREDICTION_VERSION
    method.pop('prediction_engine_version')
    with pytest.raises(ValueError,match='older calculation version'):
        path_matrix(method)


def test_web_new_drug_ids_stay_attached_after_deleting_a_different_row(tmp_path,monkeypatch):
    from test_prediction_pipeline import _article
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30)
    app.session_state['multidrug_target_text']=json.dumps(detailed_target())
    app.run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    original=json.loads(app.session_state['multidrug_target_text'])
    app.button(key='plan_add_drug').click().run()
    first=json.loads(app.session_state['multidrug_target_text'])['drugs'][-1]['drug_id']
    app.text_input(key='plan_'+first+'_name').set_value('Partner one').run()
    app.button(key='plan_add_drug').click().run()
    second=json.loads(app.session_state['multidrug_target_text'])['drugs'][-1]['drug_id']
    app.text_input(key='plan_'+second+'_name').set_value('Partner two').run()
    app.text_area(key='plan_'+second+'_schedule').set_value('every 21 days').run()
    app.button(key='plan_'+first+'_remove').click().run()
    assert not app.error and not app.exception
    after=json.loads(app.session_state['multidrug_target_text'])
    assert [d['drug_id'] for d in after['drugs']]==['D',second]
    assert after['drugs'][1]['profile']['schedule']=='every 21 days'
    assert [p['doses']['D'] for p in after['points']]==[p['doses']['D'] for p in original['points']]
