"""Identical source examples, duplicate rules and clinical predicate boundaries."""
import copy
import json

import pytest

from phase1_extraction.clinical_protocol import complete_conditions
from phase1_prediction.profile_similarity import condition, conditions, logic_comparison, match, condition_affinity
from phase1_prediction.multidrug.contracts import TargetRegimen
from phase1_prediction.multidrug.mapping import match_target
from phase1_prediction.multidrug.observations import load_model_library
from phase1_web.multidrug_examples import target_from_scope
from test_prediction_pipeline import _article
from test_friendly_clinical_form import _form_app
from test_multidrug_ui_state_audit import target_app


def test_raw_and_structured_duplicates_preserve_explicit_predicate_and_source():
    raw=['GRADE 3 OR 4 NONHEMATOLOGIC TOXICITY.']
    structured=[{'clause_id':'dlt','raw_text':'Grade 3 or 4 nonhematologic toxicity',
                 'comparator':'>=','threshold':3.,'threshold_upper':4.,'unit_raw':'grade'}]
    before=copy.deepcopy(structured)
    result=complete_conditions(raw,structured,condition)
    assert len(result)==1
    assert (result[0]['operator'],result[0]['threshold'],result[0]['upper'])==('between',3.,4.)
    assert structured==before


def test_duplicate_rule_ids_do_not_change_score():
    first={'clause_id':'one','raw':'Platelet count <= 25000 /uL','topic':'Platelets'}
    duplicate={**first,'clause_id':'two','raw':'PLATELETS <= 25000 /uL.'}
    right=[{**first,'threshold':30000.,'operator':'<=','unit':'/uL'}]
    expected=conditions([first],right)
    actual=conditions([first,duplicate],right)
    assert actual['score']==expected['score']
    assert len(actual['details'])==1


@pytest.mark.parametrize('left,right', [
    ('Grade 3 or 4 nonhematologic toxicity','Grade 3–4 nonhematologic toxicity'),
    ('PLATELET COUNT <= 25000 /uL','Platelets <= 25000 /uL'),
])
def test_known_equivalent_expressions_match_perfectly(left,right):
    assert conditions([left],[right])['score']==1.


def test_explicit_conflicts_survive_normalization():
    left={'raw':'Grade 3 or 4 toxicity','operator':'>','threshold':4.,'unit':'grade'}
    right={'raw':'Grade 3 or 4 toxicity','operator':'>=','threshold':3.,'upper':4.,'unit':'grade'}
    assert condition(left)['operator']=='>' and condition(left)['threshold']==4.
    assert conditions([left],[right])['score']==0.
    assert conditions(['ANC >= 500 /uL'],['ANC < 500 /uL'])['score']==0.
    assert conditions(['Grade 3 toxicity'],['Grade 4 toxicity'])['score']<1.
    assert condition('Cycle 1 toxicity assessment')['threshold'] is None


def test_unrelated_conditions_are_unmatched_not_forced_pairs():
    left=[condition('At least 3 months since completion of radiation therapy'),condition('At least 18 years old')]
    right=[condition('Age >= 18 years'),condition('Acceptable birth control')]
    pairs=match(left,right,condition_affinity)
    paired=[(a,b) for a,b in pairs if a is not None and b is not None]
    assert len(paired)==1 and paired[0][0]['topic']==paired[0][1]['topic']=='Age'
    score=conditions(left,right)
    reverse=conditions(right,left)
    assert score['score']==reverse['score']
    assert any(row['score'] is None for row in score['details'])


def test_logical_grouping_uses_normalized_predicates_but_keeps_boolean_structure():
    left={'criteria':[{'clause_id':'a','topic':'Platelets','raw':'Platelets < 25000 /uL'},
                      {'clause_id':'b','topic':'Febrile neutropenia','raw':'Febrile neutropenia'}],
          'logic_groups':[{'group_id':'g','operator':'OR','children':['a','b']}]}
    right=copy.deepcopy(left)
    right['criteria'][0].update(topic='platelet count',raw='platelet count < 25000 /uL')
    right['criteria'][1]['topic']='febrile neutropenia'
    assert logic_comparison(left,right,['criteria'])['score']==1.
    right['logic_groups'][0]['operator']='AND'
    assert logic_comparison(left,right,['criteria'])['score']==0.
    right['logic_groups'][0]['operator']='OR'
    right['criteria'][0]['raw']='platelet count > 25000 /uL'
    assert logic_comparison(left,right,['criteria'])['score']==0.


def test_source_plan_stays_exact_through_form_without_changing_extraction(tmp_path):
    paths=_article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    original=paths.reported_extraction.read_bytes()
    scopes,_=load_model_library([tmp_path])
    scope=scopes[0]
    target=target_from_scope(scope)
    assert all(v==1. for v in match_target(target,scope)['score']['field_scores'].values())
    form=_form_app(target.model_dump(mode='json'))
    result=TargetRegimen.model_validate(form.session_state['result'])
    assert all(v==1. for v in match_target(result,scope)['score']['field_scores'].values())
    assert result.profile['windows']==target.profile['windows']
    assert 'coverage' not in match_target(result,scope)['score']
    edited=result.model_copy(deep=True)
    edited.profile['population']['age_min']=21.
    assert match_target(edited,scope)['score']['field_scores']['population']<1.
    assert paths.reported_extraction.read_bytes()==original


def test_plan_autosaves_without_library_fill_controls_and_reopens(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    app=target_app(tmp_path,monkeypatch)
    # Older source metadata must no longer produce a library-example warning.
    draft=json.loads(app.session_state['multidrug_target_text'])
    draft['profile']['example_source'].pop('source_fingerprint')
    app.session_state['multidrug_target_text']=json.dumps(draft)
    app.session_state['multidrug_reset_editor']=True
    app.run()
    assert not app.warning
    assert not any(b.label in {'Load selected example','Fill blank drug details'} for b in app.button)
    assert not any(s.label in {'Example article and regimen','Library drug'} for s in app.selectbox)
    app.text_area(key='plan_pop_disease').set_value('My planned disease').run()
    saved=json.loads((tmp_path/'state'/'multidrug_target_draft.json').read_text())
    assert saved['profile']['population']['disease']==['My planned disease']
    assert saved['drugs']==json.loads(app.session_state['multidrug_target_text'])['drugs']
    reopened=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    assert not reopened.error and not reopened.exception and not reopened.warning
    assert reopened.text_area(key='plan_pop_disease').value=='My planned disease'
    assert json.loads(reopened.session_state['multidrug_target_text'])==saved
    reopened.sidebar.radio[0].set_value('Review evidence').run()
    assert not reopened.warning
