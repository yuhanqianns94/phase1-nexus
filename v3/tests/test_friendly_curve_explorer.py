"""Question-led exploration preserves dosing meaning and the clinical draft."""
import copy

from streamlit.testing.v1 import AppTest

from test_multidrug_transport import fixture


def app_for(draft):
    app = AppTest.from_string('''
import streamlit as st
from phase1_web.curve_explorer import coordinate_editor
result = coordinate_editor(st, st.session_state['draft'])
st.session_state['result'] = result.model_dump() if result else None
''')
    app.session_state['draft'] = copy.deepcopy(draft)
    return app.run()


def draft_for(k=2):
    *_, target = fixture(k)
    draft = target.model_dump()
    for point in draft['points']:
        if k > 1:
            point['doses']['B'] = 30.
    return draft


def test_question_first_allows_a_range_with_b_fixed_30_and_keeps_plan():
    draft = draft_for()
    app = app_for(draft)
    assert app.radio[0].key == 'explore_goal'
    assert app.radio(key='explore_goal').value == 'Vary one drug with fixed companions'
    assert app.number_input(key='explore_B_actual_fixed').value == 30.
    app.number_input(key='explore_A_actual_lo').set_value(10.).run()
    app.number_input(key='explore_A_actual_hi').set_value(40.).run()
    result = app.session_state['result']
    assert result['points'][0]['doses'] == {'A': 10., 'B': 30.}
    assert result['points'][-1]['doses'] == {'A': 40., 'B': 30.}
    assert result['profile']['planning']['exploration']['axes'] == ['A']
    assert app.session_state['draft'] == draft
    assert not app.exception


def test_relative_actual_and_planned_views_preserve_separate_edits():
    app = app_for(draft_for())
    app.number_input(key='explore_A_actual_lo').set_value(10.).run()
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    app.number_input(key='explore_A_lo').set_value(.4).run()
    app.number_input(key='explore_B_fixed').set_value(1.5).run()
    app.radio(key='explore_goal').set_value('Compare planned regimens').run()
    assert app.session_state['result']['points'][0]['doses'] == {'A': 25., 'B': 30.}
    app.radio(key='explore_goal').set_value('Vary one drug with fixed companions').run()
    assert app.number_input(key='explore_A_lo').value == .4
    assert app.number_input(key='explore_B_fixed').value == 1.5
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    assert app.number_input(key='explore_A_actual_lo').value == 10.
    assert app.number_input(key='explore_B_actual_fixed').value == 30.
    assert not app.exception


def test_two_drug_goal_creates_grid_and_retains_fixed_companion_in_three_drug_plan():
    draft = draft_for(3)
    for point in draft['points']:
        point['doses']['C'] = 15.
    app = app_for(draft)
    app.radio(key='explore_goal').set_value('Explore two drugs').run()
    assert app.multiselect(key='explore_axes').value == ['A', 'B']
    app.number_input(key='explore_B_actual_hi').set_value(60.).run()
    result = app.session_state['result']
    assert result['profile']['planning']['exploration']['axes'] == ['A', 'B']
    assert len({p['doses']['A'] for p in result['points']}) > 1
    assert len({p['doses']['B'] for p in result['points']}) > 1
    assert {p['doses']['C'] for p in result['points']} == {15.}
    assert app.session_state['draft'] == draft
    assert not app.exception


def test_undecided_companion_is_neither_absent_nor_used_unknown():
    draft = draft_for()
    draft['points'] = []
    draft['path_level_ids'] = []
    for drug in draft['drugs']:
        drug.update(mode='background', unit=None, reference_dose=None)
    app = app_for(draft)
    assert app.selectbox(key='explore_B_state').value == 'Not decided'
    assert app.session_state['result'] is None
    app.selectbox(key='explore_B_state').set_value('Numeric dose').run()
    result = app.session_state['result']
    assert all(p['doses']['B'] == 1. for p in result['points'])
    assert result['drugs'][1]['unit'] == 'relative dose'
    app.selectbox(key='explore_B_state').set_value('Used, dose unknown').run()
    assert all(p['doses']['B'] is None for p in app.session_state['result']['points'])
    app.selectbox(key='explore_B_state').set_value('Not used').run()
    assert all(p['doses']['B'] == 0. for p in app.session_state['result']['points'])
    assert app.session_state['draft'] == draft
    assert not app.exception


def test_unknown_actual_doses_and_reference_stay_blank_until_entered():
    draft = draft_for(1)
    draft['points'] = []
    draft['path_level_ids'] = []
    draft['drugs'][0].update(mode='background', unit=None, reference_dose=None)
    app = app_for(draft)
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    assert app.number_input(key='explore_A_ref').value is None
    assert app.number_input(key='explore_A_actual_lo').value is None
    assert app.number_input(key='explore_A_actual_hi').value is None
    assert app.session_state['result'] is None
    app.text_input(key='explore_A_unit').set_value('mg').run()
    app.number_input(key='explore_A_ref').set_value(20.).run()
    app.number_input(key='explore_A_actual_lo').set_value(10.).run()
    app.number_input(key='explore_A_actual_hi').set_value(40.).run()
    assert app.session_state['result']['drugs'][0]['reference_dose'] == 20.
    assert app.session_state['draft'] == draft
    assert not app.exception


def test_unentered_candidate_state_does_not_masquerade_as_unknown_use():
    draft = draft_for()
    draft['drugs'][1]['mode'] = 'background'
    draft['profile']['planning'] = {'exposure_states': {}}
    for point in draft['points']:
        point['doses']['B'] = None
        draft['profile']['planning']['exposure_states'][point['level_id']] = {'B': 'Not entered'}
    app = app_for(draft)
    assert app.selectbox(key='explore_B_state').value == 'Not decided'
    assert app.session_state['result'] is None
    assert app.session_state['draft'] == draft
    assert not app.exception


def test_two_drug_exploration_opens_with_joint_heatmap():
    from test_curve_coordinate_display_state import _chart_app, _method
    app = _chart_app(_method({'A': [10., 20.], 'B': [30., 60.]}, axes=['A', 'B']))
    assert app.radio(key='curve_view').value == 'Two-drug heatmap'
    assert app.session_state['rendered']['encoding']['y']['field'] == 'dose_B'
    assert not app.exception
