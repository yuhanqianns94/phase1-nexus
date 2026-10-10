"""Reuse evaluated doses without changing their posterior or computational scale."""
import copy
import json

import numpy as np
import pandas as pd
import streamlit as st

from phase1_prediction.multidrug import service
from phase1_prediction.multidrug.contracts import fingerprint
from phase1_prediction.multidrug.store import PosteriorStore
from phase1_prediction.multidrug.workflow import exploration_target
from test_multidrug_transport import fixture
from test_multidrug_ui_state_audit import app_with_library


def prediction_from_cached_fixture(tmp_path,monkeypatch,*,exploration=None):
    scope,arrays,manifest,target=fixture(1)
    if exploration is not None:
        coordinates={'A':[.5,1.,2.]} if exploration else {'A':[25.,50.,100.]}
        target=exploration_target(target.model_dump(mode='json'),coordinates,['A'],relative=exploration)
    store=PosteriorStore(tmp_path/'cache')
    entry=store.save('fit_x',scope,arrays,manifest)
    monkeypatch.setattr(service,'_select',lambda *a,**kw: ([(scope,{'score':{'similarity':1.}},1.)],{'notices':[]}))
    monkeypatch.setattr(service,'fit_library',lambda *a,**kw: ([entry],[]))
    prediction=service.predict(None,target,store_root=tmp_path/'cache',draw_count=4).to_dict()
    monkeypatch.setattr(PosteriorStore,'fit',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('BOIN must not fit source models')))
    return prediction


def boin_app(tmp_path,monkeypatch,prediction):
    app=app_with_library(tmp_path,monkeypatch)
    app.session_state['multidrug_prediction']=copy.deepcopy(prediction)
    app.session_state['multidrug_results']={'logistic':copy.deepcopy(prediction)}
    return app.sidebar.radio[0].set_value('OC Simulation').run()


def test_actual_exploration_reuses_regimens_and_original_prediction(tmp_path,monkeypatch):
    prediction=prediction_from_cached_fixture(tmp_path,monkeypatch,exploration=False)
    identity=fingerprint(prediction)
    app=boin_app(tmp_path,monkeypatch,prediction)
    assert next(r for r in app.radio if r.label=='Candidate regimen source').value=='Use evaluated actual-dose regimens'
    assert not any(b.label=='Evaluate BOIN candidate regimens' for b in app.button)
    assert not any(t.label=='A actual dose unit' for t in app.text_input)
    path_key='simulation_path_'+prediction['run_id']
    app.multiselect(key=path_key).set_value(['G3','G1']).run()
    app.number_input(key='boin_design_scenarios').set_value(1).run()
    app.number_input(key='boin_design_repetitions').set_value(2).run()
    next(b for b in app.button if b.label=='Run BOIN').click().run()
    simulation=app.session_state['multidrug_simulation']
    assert simulation['prediction_run_id']==prediction['run_id']
    assert [p['level_id'] for p in simulation['path']]==['G3','G1']
    assert [p['doses']['A'] for p in simulation['path']]==[100.,25.]
    assert fingerprint(app.session_state['multidrug_prediction'])==identity
    assert not app.error and not app.exception


def test_inherited_scale_is_unchanged_until_explicit_edit_and_evaluation(tmp_path,monkeypatch):
    prediction=prediction_from_cached_fixture(tmp_path,monkeypatch)
    app=boin_app(tmp_path,monkeypatch,prediction)
    prefix='boin_candidates_'+prediction['run_id']
    assert not any(b.label=='Evaluate BOIN candidate regimens' for b in app.button)
    app.checkbox(key=prefix+'_edit_units').check().run()
    assert app.text_input(key=prefix+'A_unit').value=='mg'
    assert app.number_input(key=prefix+'A_ref').value==50.
    assert not any(b.label=='Evaluate BOIN candidate regimens' for b in app.button)
    app.number_input(key=prefix+'A_ref').set_value(25.).run()
    assert app.session_state['multidrug_prediction']==prediction
    assert not any(s.label=='BOIN path: select regimens in exploration order' for s in app.multiselect)
    app.session_state['multidrug_simulation']={'prediction_run_id':prediction['run_id']}
    next(b for b in app.button if b.label=='Evaluate BOIN candidate regimens').click().run()
    updated=app.session_state['multidrug_prediction']
    assert updated['run_id']!=prediction['run_id']
    assert updated['evidence_contract']['regridded_from']==prediction['run_id']
    assert updated['evidence_contract']['source_fit_ids']==prediction['evidence_contract']['source_fit_ids']
    assert updated['target']['drugs'][0]['reference_dose']==25.
    assert 'multidrug_simulation' not in app.session_state
    assert not any(b.label=='Evaluate BOIN candidate regimens' for b in app.button)
    assert not app.error and not app.exception


def test_relative_coordinates_require_explicit_actual_doses_and_reference(tmp_path,monkeypatch):
    prediction=prediction_from_cached_fixture(tmp_path,monkeypatch,exploration=True)
    app=boin_app(tmp_path,monkeypatch,prediction)
    prefix='boin_candidates_'+prediction['run_id']
    assert not any(r.label=='Candidate regimen source' for r in app.radio)
    assert app.text_input(key=prefix+'A_unit').value==''
    assert app.number_input(key=prefix+'A_ref').value is None
    assert not any(b.label=='Run BOIN' for b in app.button)
    app.text_input(key=prefix+'A_unit').set_value('mg').run()
    app.number_input(key=prefix+'A_ref').set_value(50.).run()
    # AppTest cannot edit data_editor cells; supply its edited table output.
    original_editor=st.data_editor
    def entered_candidates(*args,**kwargs):
        rendered=original_editor(*args,**kwargs)
        if kwargs.get('key')==prefix+'_new':
            return pd.DataFrame([{'Regimen':'R1','A':25.},{'Regimen':'R2','A':50.},{'Regimen':'R3','A':100.}])
        return rendered
    monkeypatch.setattr(st,'data_editor',entered_candidates)
    app.run()
    assert not app.exception
    next(b for b in app.button if b.label=='Evaluate BOIN candidate regimens').click().run()
    updated=app.session_state['multidrug_prediction']
    assert updated['run_id']!=prediction['run_id']
    assert updated['target']['drugs'][0]['unit']=='mg'
    assert updated['target']['drugs'][0]['reference_dose']==50.
    assert not updated['target']['profile']['planning'].get('exploration')
    np.testing.assert_allclose(next(iter(updated['methods'].values()))['posterior_draws'],next(iter(prediction['methods'].values()))['posterior_draws'])
    assert not app.error and not app.exception


def test_pending_candidate_rows_survive_navigation_without_reapplying_deltas(tmp_path,monkeypatch):
    prediction=prediction_from_cached_fixture(tmp_path,monkeypatch)
    app=boin_app(tmp_path,monkeypatch,prediction)
    editor_key='boin_candidates_'+prediction['run_id']
    editor=next(d for d in app.dataframe if d.proto.id.endswith(editor_key))
    # Send the same serialized table event that the browser sends for cell edits.
    events=app._tree.get_widget_states()
    events.widgets.add(id=editor.proto.id,string_value=json.dumps({
        'edited_rows':{'0':{'A':30.}},'added_rows':[{'Regimen':'new','A':125.}],'deleted_rows':[]}))
    app._run(events)
    expected=[{'Regimen':p['level_id'],'A':p['doses']['A']} for p in prediction['target']['points']]
    expected[0]['A']=30.
    expected.append({'Regimen':'new','A':125.})
    assert app.session_state['multidrug_boin_candidate_drafts'][editor_key]['edited']==expected
    app.run()
    assert app.session_state['multidrug_boin_candidate_drafts'][editor_key]['edited']==expected
    for _ in range(2):
        app.sidebar.radio[0].set_value('Prediction Results').run()
        app.sidebar.radio[0].set_value('OC Simulation').run()
        editor=next(d for d in app.dataframe if d.proto.id.endswith(editor_key))
        assert editor.value.to_dict('records')==expected
        assert app.session_state['multidrug_boin_candidate_drafts'][editor_key]['edited']==expected
        assert app.session_state['multidrug_prediction']==prediction
        assert any(b.label=='Evaluate BOIN candidate regimens' for b in app.button)
        assert not app.error and not app.exception
    next(b for b in app.button if b.label=='Evaluate BOIN candidate regimens').click().run()
    updated=app.session_state['multidrug_prediction']
    assert updated['run_id']!=prediction['run_id']
    assert [{'Regimen':p['level_id'],'A':p['doses']['A']} for p in updated['target']['points']]==expected
    assert not app.error and not app.exception
