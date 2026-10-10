"""Current multidrug wrapper predictions follow the same path as inner methods."""
import copy

import numpy as np
import pytest

from phase1_prediction.multidrug import VERSION, TARGET_PREDICTION_VERSION
from phase1_simulation.multidrug import REGIMEN_PATH_SIMULATION_VERSION, simulate_regimen_path
from phase1_simulation.service import (
    BoinDesignCandidate, BoinSettings, compare_boin_designs, run_boin_simulation,
    _posterior_matrix,
)
from test_multidrug_transport import fixture


def wrapper():
    *_, target = fixture(1)
    method = {'format': 'multidrug_target_prediction_v1', 'status': 'fitted',
              'prediction_engine_version': TARGET_PREDICTION_VERSION, 'prediction_id': 'current_method',
              'backend': 'logistic', 'synthesis': 'same_article_joint_posterior_mixture',
              'target': target.model_dump(mode='json'), 'posterior_draws': [[.1, .2, .3], [.2, .4, .6]]}
    return {'format': 'phase1_v3_multidrug_prediction_v1', 'model_version': VERSION,
            'run_id': 'current_wrapper', 'target': method['target'], 'methods': {'m': method}}


def test_current_wrapper_matches_direct_path_and_design_comparison():
    prediction = wrapper()
    settings = BoinSettings(max_sample_size=6)
    options = dict(method_key='m', settings=settings, posterior_curve_count=2, trials_per_curve=3, seed=7)
    direct = simulate_regimen_path(prediction, **options)
    routed = run_boin_simulation(prediction, **options)
    assert direct['trials'] == routed['trials']
    np.testing.assert_array_equal(_posterior_matrix(prediction, 'm')[0], prediction['methods']['m']['posterior_draws'])
    comparison = compare_boin_designs(prediction, method_key='m', target_dlt=.25,
        candidates=[BoinDesignCandidate('Current wrapper', settings)],
        posterior_curve_count=2, trials_per_curve=3, seed=7)
    assert not comparison['invalid']
    assert len(comparison['ranked']) == 1
    assert comparison['ranked'][0]['result']['trials'] == direct['trials']
    assert comparison['simulation_model_version'] == REGIMEN_PATH_SIMULATION_VERSION
    assert comparison['prediction_run_id'] == prediction['run_id']


def test_contract_dispatch_still_rejects_obsolete_calculation_versions():
    prediction = wrapper()
    prediction['methods']['m']['prediction_engine_version'] = 'obsolete'
    with pytest.raises(ValueError, match='older calculation version'):
        run_boin_simulation(prediction, method_key='m', settings=BoinSettings(),
                            posterior_curve_count=1, trials_per_curve=1)


def test_wrapper_respects_chosen_path_without_sorting_or_mutating_prediction():
    prediction = wrapper()
    prediction['methods']['m']['target']['path_level_ids'] = ['L2', 'L0']
    original = copy.deepcopy(prediction)
    matrix, positions = _posterior_matrix(prediction, 'm')
    np.testing.assert_allclose(matrix, [[.3, .1], [.6, .2]])
    assert positions == (1., 2.)
    assert prediction == original
