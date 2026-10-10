"""Coordinate-unit transitions and charts must reflect the evaluated grid."""
import copy
from itertools import product

import pytest
from streamlit.testing.v1 import AppTest

from test_multidrug_transport import fixture


def _coordinate_app():
    *_, target=fixture()
    draft=target.model_dump()
    for point in draft['points']:point['doses']['B']=75.
    app=AppTest.from_string('''
import streamlit as st
from phase1_web.curve_explorer import coordinate_editor
result=coordinate_editor(st,st.session_state['draft'])
st.session_state['result']=result.model_dump() if result else None
''')
    app.session_state['draft']=draft
    return app.run()


def test_relative_and_actual_coordinates_keep_separate_values():
    app=_coordinate_app()
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    app.number_input(key='explore_A_lo').set_value(.5).run()
    app.number_input(key='explore_A_hi').set_value(3.).run()
    app.number_input(key='explore_B_fixed').set_value(2.).run()
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    assert app.number_input(key='explore_A_actual_lo').value==25.  # smallest entered actual dose, not an invented fraction
    assert app.number_input(key='explore_A_actual_hi').value==100.
    assert app.number_input(key='explore_B_actual_fixed').value==75.
    app.number_input(key='explore_A_actual_lo').set_value(30.).run()
    app.number_input(key='explore_A_actual_hi').set_value(150.).run()
    app.number_input(key='explore_B_actual_fixed').set_value(90.).run()
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    assert app.number_input(key='explore_A_lo').value==.5
    assert app.number_input(key='explore_A_hi').value==3.
    assert app.number_input(key='explore_B_fixed').value==2.
    assert app.session_state['result']['points'][0]['doses']=={'A':.5,'B':2.}
    app.radio(key='explore_goal').set_value('Compare planned regimens').run()
    app.radio(key='explore_goal').set_value('Vary one drug with fixed companions').run()
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    assert app.number_input(key='explore_A_actual_lo').value==30.
    assert app.number_input(key='explore_A_actual_hi').value==150.
    assert app.number_input(key='explore_B_actual_fixed').value==90.
    assert app.session_state['result']['points'][0]['doses']=={'A':30.,'B':90.}
    assert not app.exception


def test_actual_reference_edits_do_not_rescale_entered_actual_doses():
    app=_coordinate_app()
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    app.number_input(key='explore_A_actual_lo').set_value(30.).run()
    app.number_input(key='explore_A_actual_hi').set_value(150.).run()
    app.number_input(key='explore_B_actual_fixed').set_value(90.).run()
    app.number_input(key='explore_A_ref').set_value(200.).run()
    app.number_input(key='explore_B_ref').set_value(100.).run()
    assert app.number_input(key='explore_A_actual_lo').value==30.
    assert app.number_input(key='explore_A_actual_hi').value==150.
    assert app.number_input(key='explore_B_actual_fixed').value==90.
    assert [d['reference_dose'] for d in app.session_state['result']['drugs']]==[200.,100.]
    assert any('does not rescale actual doses' in c.value for c in app.caption)
    app.radio(key='explore_mode').set_value('Relative-dose exploration').run()
    app.radio(key='explore_mode').set_value('Actual-dose exploration').run()
    assert app.number_input(key='explore_A_ref').value==200.
    assert app.number_input(key='explore_B_ref').value==100.
    assert app.number_input(key='explore_A_actual_lo').value==30.
    assert not app.exception


def _method(coordinates, *, axes=()):
    ids=list(coordinates)
    drugs=[{'drug_id':did,'name':did,'mode':'numeric','unit':'mg','reference_dose':1.} for did in ids]
    rows=[]
    for i,values in enumerate(product(*coordinates.values())):
        rows.append({'level_id':f'G{i+1}','doses':dict(zip(ids,values)),
                     'posterior_median':.2+i*.01,'low_95':.1+i*.01,'high_95':.3+i*.01})
    return {'target':{'drugs':drugs,'profile':{'planning':{'exploration':{'axes':list(axes)}}}},'summary':rows}


def _chart_app(method):
    app=AppTest.from_string('''
from unittest.mock import patch
import streamlit as st
from phase1_web.curve_explorer import render_chart
st.session_state['rendered']=None
st.session_state['surface']=None
def capture_chart(chart, **kwargs):
    st.session_state['rendered']=chart.to_dict()
def capture_surface(figure, **kwargs):
    axis=figure.axes[0]
    st.session_state['surface']={'x':axis.get_xlabel(),'y':axis.get_ylabel(),'collections':len(axis.collections)}
with patch.object(st,'altair_chart',capture_chart), patch.object(st,'pyplot',capture_surface):
    render_chart(st,st.session_state['method'])
''',default_timeout=30)
    app.session_state['method']=copy.deepcopy(method)
    return app.run()


def test_one_varying_axis_ignores_fixed_numeric_partner_as_chart_axis():
    app=_chart_app(_method({'A':[10.],'B':[1.,2.,3.]},axes=['B']))
    assert app.selectbox(key='curve_x').value=='B'
    assert app.selectbox(key='curve_x').options==['B']
    assert app.radio(key='curve_view').options==['Conditional curve']
    assert app.selectbox(key='curve_fix_A').value==10.
    assert not any(s.label=='Second dose axis' for s in app.selectbox)
    chart=app.session_state['rendered']
    assert chart['layer'][1]['encoding']['x']['field']=='dose_B'
    assert app.session_state['surface'] is None
    assert not app.exception


def test_two_varying_axes_prefer_declared_order_and_render_both_views():
    app=_chart_app(_method({'A':[10.],'B':[1.,2.],'C':[3.,4.]},axes=['C','B']))
    assert app.selectbox(key='curve_x').value=='C'
    assert app.selectbox(key='curve_x').options==['C','B']
    app.radio(key='curve_view').set_value('Two-drug heatmap').run()
    assert app.selectbox(key='curve_y').value=='B'
    assert app.selectbox(key='curve_fix_A').value==10.
    chart=app.session_state['rendered']
    assert chart['encoding']['x']['field']=='dose_C'
    assert chart['encoding']['y']['field']=='dose_B'
    app.radio(key='curve_view').set_value('Three-dimensional surface').run()
    assert app.session_state['surface']=={'x':'C (mg)','y':'B','collections':1}
    assert not app.exception


def test_grid_change_clears_unavailable_display_axes_and_surface_mode():
    app=_chart_app(_method({'A':[1.,2.],'B':[3.,4.]},axes=['B','A']))
    app.radio(key='curve_view').set_value('Three-dimensional surface').run()
    app.session_state['method']=_method({'A':[1.,2.],'B':[3.]},axes=['B','A'])
    app.run()
    assert app.selectbox(key='curve_x').value=='A'
    assert app.radio(key='curve_view').value=='Conditional curve'
    assert app.radio(key='curve_view').options==['Conditional curve']
    assert app.selectbox(key='curve_fix_B').value==3.
    assert app.session_state['surface'] is None
    assert app.session_state['rendered'] is not None
    assert not app.exception


@pytest.mark.parametrize('coordinates',[{'A':[10.],'B':[20.]},{'A':[10.],'B':[None,0.]}])
def test_no_varying_numeric_coordinate_uses_table_guidance(coordinates):
    method=_method(coordinates)
    if None in coordinates['B']:method['target']['drugs'][1]['mode']='background'
    app=_chart_app(method)
    assert not app.selectbox and not app.radio
    assert app.session_state['rendered'] is None and app.session_state['surface'] is None
    assert any('No numeric coordinate varies' in c.value for c in app.caption)
    assert not app.exception
