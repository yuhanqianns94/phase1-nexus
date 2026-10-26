"""Automatic scaling must not change predictions merely by changing plot density."""
import copy
import math

import pytest

from phase1_web.dose_reference import geometric_reference
from test_friendly_curve_explorer import app_for, draft_for


def unscaled_draft():
    draft = draft_for(1)
    draft['points'] = []
    draft['path_level_ids'] = []
    draft['drugs'][0].update(mode='background', unit='mg', reference_dose=None)
    return draft


def reference(app):
    assert not app.exception
    return app.session_state['result']['drugs'][0]['reference_dose']


def test_geometric_scale_uses_distinct_positive_doses():
    assert geometric_reference([None, 0]) is None
    assert geometric_reference([1100, 1700, 2200, 1100, 0, None]) == pytest.approx(1602.340325129886)


def test_range_scale_is_automatic_and_stays_fixed_across_grid_range_and_view_changes():
    draft = unscaled_draft()
    app = app_for(draft)
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    app.number_input(key='explore_A_actual_lo').set_value(1100.).run()
    assert app.session_state['result'] is None
    app.number_input(key='explore_A_actual_hi').set_value(2200.).run()
    expected = math.sqrt(1100 * 2200)
    assert reference(app) == pytest.approx(expected)
    assert not any(w.key == 'explore_A_ref' for w in app.number_input)
    app.number_input(key='explore_A_actual_n').set_value(40).run()
    app.number_input(key='explore_A_actual_hi').set_value(3300.).run()
    assert reference(app) == pytest.approx(expected)
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    assert reference(app) == pytest.approx(expected)
    assert app.session_state['draft'] == draft
    app.checkbox(key='explore_A_custom_ref').check().run()
    app.number_input(key='explore_A_ref').set_value(2000.).run()
    assert reference(app) == 2000.
    app.checkbox(key='explore_A_custom_ref').uncheck().run()
    assert reference(app) == pytest.approx(expected)
    app.checkbox(key='explore_A_custom_ref').check().run()
    assert reference(app) == 2000.
    assert app.session_state['result']['points'][0]['doses']['A'] == 1100.


def test_candidate_doses_supply_missing_reference_and_saved_scale_wins():
    draft = unscaled_draft()
    draft['drugs'][0]['mode'] = 'numeric'
    draft['points'] = [{'level_id': f'L{i}', 'doses': {'A': d}}
                       for i, d in enumerate([1100., 1700., 2200.])]
    original = copy.deepcopy(draft)
    app = app_for(draft)
    app.radio(key='explore_goal').set_value('Compare candidate doses').run()
    assert reference(app) == pytest.approx(1602.340325129886)
    assert app.session_state['draft'] == original
    draft['drugs'][0]['reference_dose'] = 1500.
    app = app_for(draft)
    assert reference(app) == 1500.
    app.number_input(key='explore_A_actual_hi').set_value(3300.).run()
    assert reference(app) == 1500.


def test_new_unit_does_not_inherit_the_previous_unit_custom_scale():
    app = app_for(draft_for(1))
    app.checkbox(key='explore_A_custom_ref').check().run()
    app.number_input(key='explore_A_ref').set_value(999.).run()
    app.text_input(key='explore_A_unit').set_value('mg/kg').run()
    assert reference(app) == pytest.approx(50.)
    assert not app.checkbox(key='explore_A_custom_ref').value
    assert not any(w.key == 'explore_A_ref' for w in app.number_input)
