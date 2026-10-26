"""User-visible target editing and curve-coordinate state transitions."""
import copy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phase1_web.multidrug_app import example_target
from test_multidrug_transport import fixture
from test_prediction_pipeline import _article


def _target_app(tmp_path, monkeypatch, draft=None):
    _article(tmp_path, 'PMC123', rows=[(.1, 3, 0), (.3, 6, 2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS', str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT', str(tmp_path / 'state'))
    app = AppTest.from_file(Path(__file__).parents[1] / 'phase1_web/app.py', default_timeout=30).run()
    if draft is None:
        draft = example_target()
        draft['profile'] = {'windows': [{'days': 28., 'kind': 'formal_dlt'}],
                            'dlt': {'criteria': [{'raw': 'Grade 3 toxicity'}]}}
    app.session_state['multidrug_target_text'] = json.dumps(draft)
    app.session_state['multidrug_reset_editor'] = True
    app.sidebar.radio[0].set_value('Target Trial').run()
    assert not app.exception and not app.error
    return app


def test_conditional_assessment_and_dose_inputs_retain_edits(tmp_path, monkeypatch):
    app = _target_app(tmp_path, monkeypatch)
    app.number_input(key='plan_window_days').set_value(45.).run()
    app.selectbox(key='plan_window_mode').set_value('Until an event').run()
    app.selectbox(key='plan_window_end').set_value('Other').run()
    app.text_input(key='plan_window_end_custom').set_value('After recovery').run()
    app.selectbox(key='plan_window_mode').set_value('Number of days').run()
    assert app.number_input(key='plan_window_days').value == 45.
    app.selectbox(key='plan_window_mode').set_value('Until an event').run()
    assert app.text_input(key='plan_window_end_custom').value == 'After recovery'
    app.number_input(key='plan_L1_D_dose').set_value(75.).run()
    app.selectbox(key='plan_L1_D_state').set_value('Use: dose unknown').run()
    app.selectbox(key='plan_L1_D_state').set_value('Use: dose specified').run()
    assert app.number_input(key='plan_L1_D_dose').value == 75.
    assert not app.exception and not app.error


def test_rule_rebase_preserves_explorer_but_new_trial_resets_it(tmp_path, monkeypatch):
    app = _target_app(tmp_path, monkeypatch)
    app.sidebar.radio[0].set_value('Prediction Results').run()
    app.radio(key='explore_goal').set_value('Vary one drug').run()
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    app.number_input(key='explore_D_lo').set_value(.7).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    app.text_area(key='plan_dlt_criteria_0_raw').set_value('Grade 3 toxicity during cycle 1').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.radio(key='explore_mode').value == 'Relative-dose exploration'
    assert app.number_input(key='explore_D_lo').value == .7
    app.sidebar.radio[0].set_value('Target Trial').run()
    next(b for b in app.button if b.label == 'Start new trial').click().run()
    assert 'explore_D_lo' not in app.session_state
    assert json.loads(app.session_state['multidrug_target_text'])['drugs'] == []
    assert not app.exception and not app.error


@pytest.mark.parametrize('axes', [['A', 'B'], ['B']])
def test_removing_selected_drug_reconciles_explorer_axes(tmp_path, monkeypatch, axes):
    *_, target = fixture()
    app = _target_app(tmp_path, monkeypatch, target.model_dump())
    app.sidebar.radio[0].set_value('Prediction Results').run()
    app.session_state['explore_axes'] = axes  # Migrate stale multi-axis selections as well.
    app.run()
    assert len(app.multiselect(key='explore_axes').value) == 1
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    app.multiselect(key='explore_axes').set_value(axes[:1]).run()
    if 'A' in axes:
        app.number_input(key='explore_A_lo').set_value(.7).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    app.button(key='plan_B_remove').click().run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert app.radio(key='explore_mode').value == 'Relative-dose exploration'
    assert app.multiselect(key='explore_axes').value == ['A']
    if 'A' in axes:
        assert app.number_input(key='explore_A_lo').value == .7
    assert not app.exception and not app.error


def _coordinate_app(draft):
    app = AppTest.from_string('''
import streamlit as st
from phase1_web.curve_explorer import coordinate_editor
result = coordinate_editor(st, st.session_state['draft'])
st.session_state['result'] = result.model_dump() if result else None
''')
    app.session_state['draft'] = copy.deepcopy(draft)
    return app.run()


@pytest.mark.parametrize('value, state', [(0., 'Not used'), (None, 'Used, dose unknown')])
def test_explorer_preserves_explicit_background_use_state(value, state):
    *_, target = fixture()
    draft = target.model_dump()
    draft['drugs'][1]['mode'] = 'background'
    for point in draft['points']:
        point['doses']['B'] = value
    app = _coordinate_app(draft)
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    assert app.selectbox(key='explore_B_state').value == state
    assert all(p['doses']['B'] == value for p in app.session_state['result']['points'])
    assert app.session_state['draft'] == draft
    assert not app.exception


def test_explorer_requires_choice_for_mixed_background_states():
    *_, target = fixture()
    draft = target.model_dump()
    draft['drugs'][1]['mode'] = 'background'
    for i, point in enumerate(draft['points']):
        point['doses']['B'] = 0. if i == 0 else None
    app = _coordinate_app(draft)
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    assert app.session_state['result'] is None
    app.selectbox(key='explore_B_state').set_value('Not used').run()
    assert all(p['doses']['B'] == 0. for p in app.session_state['result']['points'])
    assert not app.exception


@pytest.mark.parametrize('relative, expected', [(True, 1.5), (False, 75.)])
def test_explorer_preserves_fixed_numeric_companion(relative, expected):
    *_, target = fixture()
    draft = target.model_dump()
    for point in draft['points']:
        point['doses']['B'] = 75.
    app = _coordinate_app(draft)
    app.radio(key='explore_mode').set_value('Relative-dose exploration' if relative else 'Actual-dose exploration').run()
    fixed_key = 'explore_B_fixed' if relative else 'explore_B_actual_fixed'
    assert app.number_input(key=fixed_key).value == expected
    assert all(p['doses']['B'] == expected for p in app.session_state['result']['points'])
    assert not app.exception


def test_no_candidate_plan_requires_explicit_companion_state():
    *_, target = fixture()
    draft = target.model_dump()
    draft['points'] = []
    draft['path_level_ids'] = []
    for drug in draft['drugs']:
        drug['mode'] = 'background'
        drug['reference_dose'] = None
        drug['unit'] = None
    app = _coordinate_app(draft)
    assert app.session_state['result'] is None
    app.selectbox(key='explore_B_state').set_value('Used, dose unknown').run()
    result = app.session_state['result']
    assert result['profile']['planning']['exploration']['axes'] == ['A']
    assert all(p['doses']['A'] > 0 and p['doses']['B'] is None for p in result['points'])
    assert app.session_state['draft'] == draft
    assert not app.exception


def test_rule_text_parses_once_and_explicit_controls_override_it(tmp_path, monkeypatch):
    draft=example_target()
    draft['profile']['eligibility']={'inclusion':[{'raw':'','source_id':'source-7',
        'timing':'before treatment','negated':False,'custom_fact':{'retain':True}}]}
    app=_target_app(tmp_path,monkeypatch,draft)
    app.text_area(key='plan_elig_inclusion_0_raw').set_value('Age >= 18 years').run()
    rule=json.loads(app.session_state['multidrug_target_text'])['profile']['eligibility']['inclusion'][0]
    assert (rule['operator'],rule['threshold'],rule['unit'])==('>=',18.,'years')
    assert app.number_input(key='plan_elig_inclusion_0_threshold').value==18.
    app.text_area(key='plan_elig_inclusion_0_raw').set_value('Age >= 20 years')
    app.number_input(key='plan_elig_inclusion_0_threshold').set_value(21.).run()
    rule=json.loads(app.session_state['multidrug_target_text'])['profile']['eligibility']['inclusion'][0]
    assert rule['threshold']==21. and rule['raw']=='Age >= 20 years'
    assert rule['source_id']=='source-7' and rule['custom_fact']=={'retain':True}
    assert rule['timing']=='before treatment' and rule['negated'] is False
    assert any('21' in c.value and ('Condition:' in c.value or 'Structured' in c.value or '>=' in c.value) for c in app.caption)
    assert not app.exception and not app.error


def _cycle_draft():
    draft=example_target()
    draft['drugs'][0]['profile']['phases']=[{'phase':'treatment','cycle':'3 weeks'}]
    draft['profile']={'windows':[{'kind':'formal_dlt','days':28.,'cycles':'1'}],
        'planning':{'window_mode':'Number of cycles','window_cycles':1.,'window_cycle_days':28.},
        'dlt':{'criteria':[{'raw':'Grade 3 toxicity'}]}}
    return draft


def test_assessment_reuses_stage_cycle_and_preserves_custom_value(tmp_path, monkeypatch):
    app=_target_app(tmp_path,monkeypatch,_cycle_draft())
    assert app.number_input(key='plan_window_cycle_days').value==28.
    app.number_input(key='plan_window_cycle_days').set_value(35.).run()
    app.selectbox(key='plan_window_cycle_source').select_index(1).run()
    assert not any(n.key=='plan_window_cycle_days' for n in app.number_input)
    app.number_input(key='plan_window_cycles').set_value(2.).run()
    saved=json.loads(app.session_state['multidrug_target_text'])
    assert saved['profile']['windows'][0]['days']==42.
    app.text_area(key='plan_dlt_criteria_0_raw').set_value('Grade 3 toxicity in cycle 1').run()
    app.selectbox(key='plan_window_cycle_source').set_value('custom').run()
    assert app.number_input(key='plan_window_cycle_days').value==35.
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['windows'][0]['days']==70.
    assert not app.exception and not app.error


def test_selected_assessment_cycle_follows_stage_edits(tmp_path, monkeypatch):
    app=_target_app(tmp_path,monkeypatch,_cycle_draft())
    app.selectbox(key='plan_window_cycle_source').select_index(1).run()
    app.number_input(key='plan_D_phase_0_cycle').set_value(4.).run()
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['windows'][0]['days']==28.
    assert app.selectbox(key='plan_window_cycle_source').value!='custom'
    app.button(key='plan_D_remove_phase_0').click().run()
    assert app.selectbox(key='plan_window_cycle_source').value is None
    assert any('no longer has an available cycle' in info.value for info in app.info)
    app.selectbox(key='plan_window_cycle_source').set_value('custom').run()
    assert app.number_input(key='plan_window_cycle_days').value==28.
    assert not app.exception and not app.error


def test_removing_prior_stage_keeps_assessment_link_on_same_stage(tmp_path, monkeypatch):
    draft=_cycle_draft()
    draft['drugs'][0]['profile']['phases'].append({'phase':'maintenance','cycle':'28 days'})
    app=_target_app(tmp_path,monkeypatch,draft)
    app.selectbox(key='plan_window_cycle_source').select_index(2).run()
    app.button(key='plan_D_remove_phase_0').click().run()
    app.number_input(key='plan_D_phase_0_cycle').set_value(35.).run()
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['windows'][0]['days']==35.
    assert app.selectbox(key='plan_window_cycle_source').value!='custom'
    assert not app.exception and not app.error


def test_imported_cycle_length_does_not_change_when_cycle_count_changes(tmp_path, monkeypatch):
    draft=_cycle_draft()
    draft['profile']['planning']={'window_mode':'Number of cycles'}
    app=_target_app(tmp_path,monkeypatch,draft)
    assert app.number_input(key='plan_window_cycle_days').value==28.
    app.number_input(key='plan_window_cycles').set_value(2.).run()
    assert app.number_input(key='plan_window_cycle_days').value==28.
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['windows'][0]['days']==56.
    assert not app.exception and not app.error


def test_new_rules_use_groups_and_imported_linkage_is_preserved(tmp_path, monkeypatch):
    draft=example_target()
    draft['profile']['dlt']={'criteria':[{'raw':'Grade 3 toxicity','logic':'AND'}, {'raw':'Persistent toxicity'}]}
    app=_target_app(tmp_path,monkeypatch,draft)
    assert not any(s.label=='Relationship to preceding rule' for s in app.selectbox)
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['dlt']['criteria'][0]['logic']=='AND'
    app.checkbox(key='plan_dlt_criteria_0_edit_logic').check().run()
    app.selectbox(key='plan_dlt_criteria_0_logic').set_value('OR').run()
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['dlt']['criteria'][0]['logic']=='OR'
    assert not app.exception and not app.error


@pytest.mark.parametrize('original', [['Female','Male','No restriction'], ['Postmenopausal women only'], ['female']])
def test_sex_eligibility_is_exclusive_and_preserves_imported_values(tmp_path, monkeypatch, original):
    draft=example_target()
    draft['profile']['population']={'sex':original}
    app=_target_app(tmp_path,monkeypatch,draft)
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['population']['sex']==original
    assert not any(m.label=='Sex eligibility' for m in app.multiselect)
    app.selectbox(key='plan_pop_sex').set_value('Male').run()
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['population']['sex']==['Male']
    app.selectbox(key='plan_pop_sex').set_value('No restriction').run()
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['population']['sex']==['No restriction']
    app.selectbox(key='plan_pop_sex').set_value('Undecided').run()
    assert json.loads(app.session_state['multidrug_target_text'])['profile']['population']['sex']==[]
    assert not app.exception and not app.error
