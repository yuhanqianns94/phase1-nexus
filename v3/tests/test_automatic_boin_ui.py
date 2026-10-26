"""Only dose coordinates order BOIN; posterior probabilities are never sorted."""
import copy
import json

import numpy as np
import pytest

from phase1_web.multidrug_state import automatic_boin_path, simulation_view
from phase1_simulation.multidrug import path_matrix
from test_result_exports import prediction
from test_multidrug_transport import fixture
from test_boin_candidate_usability import prediction_from_cached_fixture, boin_app


def test_path_sorts_doses_and_selects_matching_posterior_columns_without_mutation():
    result=prediction()
    target=result['target']
    target['points']=[target['points'][2],target['points'][0],target['points'][1]]
    method=next(iter(result['methods'].values()))
    method['target']=copy.deepcopy(target)
    method['posterior_draws']=[[.3,.1,.2],[.5,.2,.4]]
    before=copy.deepcopy(result)
    path=automatic_boin_path(target)
    assert path==['L0','L1','L2']
    matrix,_,_=path_matrix(simulation_view(result,path),next(iter(result['methods'])))
    np.testing.assert_array_equal(matrix,[[.1,.2,.3],[.2,.4,.5]])
    assert result==before


def test_auto_path_requires_unique_doses_and_fixed_companions():
    *_,target=fixture(2)
    target.profile['single_drug_analysis']={'focal_drug_id':'A'}
    with pytest.raises(ValueError,match='fixed'):
        automatic_boin_path(target.model_dump())
    for point in target.points:point.doses['B']=50.
    assert automatic_boin_path(target.model_dump())==['L0','L1','L2']
    target.points[0].doses['A']=target.points[1].doses['A']
    with pytest.raises(ValueError,match='different'):
        automatic_boin_path(target.model_dump())


def test_editor_hides_manual_ids_and_modes_show_only_relevant_inputs(tmp_path,monkeypatch):
    result=prediction_from_cached_fixture(tmp_path,monkeypatch)
    app=boin_app(tmp_path,monkeypatch,result)
    editor=next(d for d in app.dataframe if d.proto.id.endswith('boin_candidates_'+result['run_id']))
    assert json.loads(editor.proto.columns)['Regimen']['hidden'] is True
    assert not any(w.label.startswith('BOIN path:') for w in app.multiselect)
    assert not any(b.label=='Compare designs' for b in app.button)
    assert not app.expander
    app.radio(key='boin_design_task').set_value('Compare multiple designs').run()
    assert not any(b.label=='Run BOIN' for b in app.button)
    app.multiselect(key='boin_design_search_sizes').set_value([30]).run()
    assert next(b for b in app.button if b.label=='Compare designs').disabled
    app.multiselect(key='boin_design_search_sizes').set_value([30,36]).run()
    assert not next(b for b in app.button if b.label=='Compare designs').disabled
    assert not app.exception and not app.error
