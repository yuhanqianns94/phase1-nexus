import copy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phase1_prediction.multidrug.mapping import match_target
from phase1_prediction.multidrug.transport import assemble_components
from phase1_prediction.multidrug.observations import build_article_scopes
from phase1_prediction.evidence import load_evidence_library
from phase1_project.analyses import save_analysis
from phase1_web.multidrug_state import activate_prediction, discard_stale_dependents
from phase1_web.multidrug_form import merge_phases, phase_rows, merge_conditions
from phase1_web.multidrug_examples import target_from_scope
from phase1_web.multidrug_app import example_target
from test_multidrug_transport import fixture
from test_cohort_exposures import extraction, write_article
from phase1_extraction.friendly_review import load_or_create_friendly_review, save_friendly_review, summarize_friendly_review


@pytest.mark.parametrize('backend',['logistic'])
def test_assembly_uses_preview_mapping_and_manual_role_override(backend):
    scope,arrays,manifest,target=fixture(2,backend)
    target.mapping_overrides={scope.scope_id:{'A':'B','B':'A'}}
    mapping=match_target(target,scope)
    result=assemble_components([(scope,arrays,manifest)],target,backend=backend,draw_count=8)
    assert result['status']=='fitted'
    for trace in result['provenance']:
        for component in trace['components']:
            assert component['source_id']==mapping['pairs'][component['target_id']]
            assert component['mapping_id']==mapping['mapping_id']


def test_assembly_obeys_automatic_assignment_for_all_primary_targets():
    scope,arrays,manifest,target=fixture(3)
    scope.drugs[1].research_role='co_primary'
    target.drugs[1].research_role='co_primary'
    target.drugs[1].name='C' # A co-primary target may correspond to a source partner.
    target.drugs[2].name='Z' # Do not create two targets claiming the same identity.
    mapping=match_target(target,scope)
    assert mapping['pairs']['B']=='C'
    result=assemble_components([(scope,arrays,manifest)],target,draw_count=8)
    assert result['status']=='fitted'
    for trace in result['provenance']:
        assert {c['target_id']:c['source_id'] for c in trace['components']}==mapping['pairs']


def test_human_gold_changes_accuracy_only_not_models_or_similarity(tmp_path):
    paths=write_article(tmp_path,extraction())
    before_scopes,_=build_article_scopes(paths)
    before_library=load_evidence_library([tmp_path])
    original={p:p.read_bytes() for p in (paths.reported_extraction,paths.curve_rows)}
    review=load_or_create_friendly_review(paths)
    unit=next(u for u in review.units if u.unit_kind=='curve_point')
    unit.decision='incorrect';unit.gold_point=unit.predicted_point.model_copy(deep=True)
    unit.gold_point.y=1;unit.gold_point.n=5
    unit.gold_point.exposures[0].name_raw='Human corrected name'
    unit.gold_point.exposures[0].component_id=None
    text=next(u for u in review.units if u.unit_kind=='text_field')
    text.decision='incorrect';text.gold_answer='Human corrected disease or mechanism'
    save_friendly_review(paths,review)
    for policy in ('extracted','reviewed_only','reviewed_if_available'):
        after,_=build_article_scopes(paths,review_policy=policy)
        assert after==before_scopes
        lib=load_evidence_library([tmp_path],review_policy=policy)
        assert lib.fingerprint==before_library.fingerprint
        assert lib.matching_records==before_library.matching_records
        assert lib.review_policy=='extracted'
    assert all(p.read_bytes()==data for p,data in original.items())
    assert summarize_friendly_review(review).model_dump()!=summarize_friendly_review(load_or_create_friendly_review(write_article(tmp_path/'unreviewed',extraction()))).model_dump()


def test_prediction_replacement_and_saving_cannot_mix_runs(tmp_path):
    state={'multidrug_prediction':{'run_id':'prediction_old'},
           'multidrug_simulation':{'prediction_run_id':'prediction_old'},
           'multidrug_search':{'prediction_run_id':'prediction_old'}}
    activate_prediction(state,{'run_id':'prediction_new'})
    assert 'multidrug_simulation' not in state and 'multidrug_search' not in state
    state['multidrug_simulation']={'prediction_run_id':'prediction_old'}
    discard_stale_dependents(state)
    assert 'multidrug_simulation' not in state
    state['multidrug_simulation']={'prediction_run_id':'prediction_new'}
    activate_prediction(state,{'run_id':'prediction_new'},recomputed=True)
    assert 'multidrug_simulation' not in state
    for key in ('simulation','design_comparison'):
        with pytest.raises(ValueError,match='different prediction'):
            save_analysis(tmp_path,{'prediction':{'run_id':'prediction_new'},key:{'prediction_run_id':'prediction_old'}})


def detailed_target():
    target=example_target()
    target['drugs'][0]['profile']['phases']=[
        {'phase':'lead-in','route':'IV','frequency':'weekly','cycle':21,'infusion_duration':'2 hours','custom_fact':'keep me'},
        {'phase':'maintenance','route':'IV','frequency':'monthly','cycle':'28 days','duration':'until progression'}]
    target['profile']={'dlt':{'criteria':[{'raw':'grade 3 toxicity','timing':'cycle 1','logic':'AND','unit':'grade'}],
                            'grading':'CTCAE version 5.0','grading_version':'5.0','custom_definition':'retain'},
                       'windows':[{'kind':'formal_dlt','days':28,'cycles':'1','basis':'reported'},
                                  {'kind':'safety_monitoring','raw':'throughout trial'}],
                       'configurations':[{'phase':'I','role':'escalation'},{'phase':'I','role':'expansion'}]}
    return target


def test_phase_edits_keep_other_phases_and_structured_conditions():
    initial=detailed_target();phases=initial['drugs'][0]['profile']['phases']
    rows=phase_rows(phases);rows[0]['Frequency']='twice weekly'
    out=merge_phases(phases,rows)
    assert out[0]['frequency']=='twice weekly' and out[0]['custom_fact']=='keep me'
    assert out[1]==phases[1]
    assert merge_phases(phases,rows[1:])==[phases[1]]
    criteria=initial['profile']['dlt']['criteria']
    assert merge_conditions(criteria,'grade 3 toxicity')==criteria


def test_simple_form_roundtrip_keeps_detailed_json():
    initial=detailed_target()
    app=AppTest.from_string('import streamlit as st\nfrom phase1_web.multidrug_form import render\nst.session_state["result"]=render(st.session_state["initial"]).model_dump(mode="json")',default_timeout=30)
    app.session_state['initial']=copy.deepcopy(initial);app.run()
    assert not app.exception and not app.error
    output=app.session_state['result']
    assert output['drugs'][0]['profile']['phases']==initial['drugs'][0]['profile']['phases']
    for key in ('dlt','windows','configurations'):
        assert output['profile'][key]==initial['profile'][key]
    assert app.session_state['initial']==initial


def test_example_import_copies_clinical_facts_without_outcomes(tmp_path):
    e=extraction();e.cohorts=e.cohorts[:1]
    scopes,_=build_article_scopes(write_article(tmp_path,e))
    target=target_from_scope(scopes[0])
    assert target.points[0].doses[target.drugs[0].drug_id]==750
    assert target.points[0].doses[target.drugs[1].drug_id] is None
    assert not target.path_level_ids
    assert 'observed' not in target.profile.get('dlt',{})
    assert 'observations' not in target.model_dump()
    assert target.drugs[0].profile['phases']==scopes[0].drugs[0].profile.get('phases',[])


def test_web_json_form_switch_and_navigation_preserve_edits(tmp_path,monkeypatch):
    from test_prediction_pipeline import _article
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    next(x for x in app.checkbox if x.label=='Advanced JSON editor').check().run()
    target=detailed_target()
    app.text_area(key='multidrug_target_json_editor').set_value(json.dumps(target)).run()
    next(x for x in app.checkbox if x.label=='Advanced JSON editor').uncheck().run()
    assert not app.exception and not app.error
    app.text_area(key='plan_pop_disease').set_value('Edited disease').run()
    app.sidebar.radio[0].set_value('Library').run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    assert app.text_area(key='plan_pop_disease').value=='Edited disease'
    next(x for x in app.checkbox if x.label=='Advanced JSON editor').check().run()
    actual=json.loads(app.text_area(key='multidrug_target_json_editor').value)
    assert actual['profile']['population']['disease']==['Edited disease']
    assert actual['drugs'][0]['profile']['phases']==target['drugs'][0]['profile']['phases']
    assert actual['profile']['dlt']==target['profile']['dlt']
    assert actual['profile']['windows']==target['profile']['windows']


def test_web_scope_selection_shared_across_methods_and_invalidates_outputs(tmp_path,monkeypatch):
    from test_prediction_pipeline import _article
    from types import SimpleNamespace
    from phase1_web import multidrug_app
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    _article(tmp_path,'PMC456',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    draft=example_target();draft['drugs'][0]['name']='IPH2101'
    app.session_state['multidrug_target_text']=json.dumps(draft)
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(x for x in app.button if x.label=='Find matching articles').click().run()
    selector=next(x for x in app.multiselect if x.label=='Included source model scopes')
    chosen=selector.value[:1]
    selector.set_value(chosen).run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    next(x for x in app.selectbox if x.label=='Curve aggregation').set_value('Average of article posterior curves (Method 1)').run()
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert next(x for x in app.multiselect if x.label=='Included source model scopes').value==chosen
    app.sidebar.radio[0].set_value('Prediction Results').run()
    fake={'run_id':'prediction_new','methods':{},'target':example_target()}
    monkeypatch.setattr(multidrug_app,'predict',lambda *a,**kw:SimpleNamespace(to_dict=lambda:fake))
    monkeypatch.setattr(multidrug_app,'_result',lambda *a,**kw:None)
    app.session_state['multidrug_prediction']={'run_id':'prediction_old'}
    app.session_state['multidrug_simulation']={'prediction_run_id':'prediction_old'}
    app.session_state['multidrug_search']={'prediction_run_id':'prediction_old'}
    next(x for x in app.button if x.label=='Generate prediction').click().run()
    assert app.session_state['multidrug_prediction']['run_id']=='prediction_new'
    assert 'multidrug_simulation' not in app.session_state and 'multidrug_search' not in app.session_state
    app.session_state['multidrug_simulation']={'prediction_run_id':'prediction_new'}
    app.sidebar.radio[0].set_value('Review evidence').run()
    selector=next(x for x in app.multiselect if x.label=='Included source model scopes')
    selector.set_value([r['series_key'] for r in app.session_state['multidrug_preview']['selected_series']]).run()
    assert not app.session_state['multidrug_prediction']
    assert not app.session_state['multidrug_results']
    assert 'multidrug_simulation' not in app.session_state
