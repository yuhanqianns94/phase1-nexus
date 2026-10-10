"""Saved adjusted results cannot be silently reused after adjustment removal."""
from copy import deepcopy

import pytest

from phase1_prediction.reference.prediction_contract import ensure_current_prediction
from phase1_prediction.reference.prediction_service import predict_probability_at_dose
from phase1_prediction.reference.oc_service import joint_target_posterior_draws
from phase1_simulation.service import _posterior_matrix


@pytest.mark.parametrize("adjustment", [
    {"applied": True},
    {"enabled": True, "trial_type": "combination"},
    {"enabled": True},
])
@pytest.mark.parametrize("wrapper", ["reference_result", "methods", "bayesian_curve_parameters"])
def test_obsolete_adjusted_results_rejected_at_public_replay_boundaries(adjustment, wrapper):
    payload = {wrapper: {"target_trial_type_prior_adjustment": adjustment}}
    original = deepcopy(payload)
    for replay in (
        ensure_current_prediction,
        lambda p: predict_probability_at_dose(p, 1.),
        joint_target_posterior_draws,
        lambda p: _posterior_matrix(p, "article_line_weighted"),
    ):
        with pytest.raises(ValueError, match="Regenerate the prediction"):
            replay(payload)
    assert payload == original


def test_unadjusted_results_and_external_joint_draws_remain_usable():
    ensure_current_prediction({"target_trial_type_prior_adjustment": {
        "enabled": True, "applied": False, "trial_type": "single",
    }})
    payload = {"methods": {"article_line_weighted": {
        "status": "fitted", "doses": [1., 2.], "posterior_draws": [[.1, .2], [.2, .3]],
        "target_trial_type_prior_adjustment": {"enabled": False, "applied": False},
    }}}
    matrix, doses = _posterior_matrix(payload, "article_line_weighted")
    assert matrix.tolist() == [[.1, .2], [.2, .3]] and doses == (1., 2.)
