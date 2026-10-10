"""Regression coverage for removing trial-label-based toxicity shifts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect

import numpy as np
import pytest

from phase1_prediction.reference.bayesian_toxicity import (
    BayesianPrior, fit_bayesian_binomial_logistic, posterior_probability_draws,
)
from phase1_prediction.reference.posterior_curve_source_mixture import (
    estimate_article_posterior_draw_pool, sample_historical_posterior_curve_mixture,
    source_mixture_draws_at_doses,
)
from phase1_prediction.reference.power_likelihood_curve import (
    estimate_power_likelihood_curve, power_likelihood_draws_at_doses,
)
from phase1_prediction.reference.power_likelihood_random_effects import (
    estimate_power_likelihood_random_effects_curve, random_effects_draws_at_doses,
)
from phase1_prediction.reference.power_likelihood_random_intercept_slope import (
    estimate_power_likelihood_random_intercept_slope_curve,
    random_intercept_slope_draws_at_doses,
)
from phase1_prediction.reference.similarity_weighted_curve import (
    estimate_article_line_weighted, posterior_mixture_draws_at_doses,
    posterior_mixture_summary_at_dose, predict_article_line_weighted_at_doses,
)
from phase1_prediction.reference.toxicity_curve_common import ToxicityQuery


@pytest.fixture(scope="module")
def historical_lines():
    prior = BayesianPrior("test", "ordinary curve prior", -1.5, 2.5, -0.3, 1.5)
    lines = []
    for index, counts in enumerate(((0, 1, 3), (1, 2, 4))):
        rows = [
            {"canonical_dose": dose, "n_patients": 12,
             "n_patients_with_dlt": y, "denominator_role": "dlt_evaluable"}
            for dose, y in zip((1., 2., 4.), counts, strict=True)
        ]
        fit = fit_bayesian_binomial_logistic(rows, prior, draw_count=64, random_seed=index)
        assert fit["status"] == "fitted"
        lines.append({
            "pmcid": f"PMC{index}", "document_id": f"PMC{index}",
            "segment_index": 0, "study_unit_id": "study", "escalation_series_id": "series",
            "arm_or_cohort": "A with background therapy", "drug_name": "A", "dose_unit": "mg",
            "line_type": "multi_observation_bayesian_binomial_logistic", "similarity": .8,
            "article_similarity_weight": .8, "segment_similarity_weight": .8,
            "reliability_weight": 1., "reliability_policy": "test",
            "source_patient_count": 36, "observed_dose_min": 1., "observed_dose_max": 4.,
            "source_points": rows, "bayesian_fit": fit,
        })
    return lines


ESTIMATORS = [
    estimate_article_line_weighted,
    estimate_article_posterior_draw_pool,
    estimate_power_likelihood_curve,
    estimate_power_likelihood_random_effects_curve,
    estimate_power_likelihood_random_intercept_slope_curve,
]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_trial_label_alone_cannot_change_any_curve(estimator, historical_lines):
    """Evidence/configuration matching may differ; identical evidence cannot be uplifted."""
    spec = ToxicityQuery("A", "mg", "single", (1., 2., 4.))
    kwargs = {"bayesian_prior": "test"}
    inputs = historical_lines
    if estimator is estimate_article_line_weighted:
        kwargs.update(bayesian_cache={}, historical_lines=historical_lines)
        inputs = []
    single = estimator(inputs, spec, **kwargs)
    combination = estimator(inputs, replace(spec, trial_type="combination"), **kwargs)
    assert single["status"] == combination["status"] == "fitted"
    assert single["curve_points"] == combination["curve_points"]
    assert "target_trial_type_prior_adjustment" not in combination
    assert all(not any(key.startswith("unadjusted_") for key in row)
               for row in combination["curve_points"])


def test_line_helpers_preserve_historical_posterior_without_additional_shift(historical_lines):
    doses = (1., 2., 4.)
    expected = np.mean([
        np.column_stack([posterior_probability_draws(line["bayesian_fit"], d) for d in doses])
        for line in historical_lines
    ], axis=0)
    actual = posterior_mixture_draws_at_doses(historical_lines, doses)["posterior_draws"]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    predictions = predict_article_line_weighted_at_doses(historical_lines, doses)
    for column, dose in enumerate(doses):
        assert predictions[dose] == pytest.approx(expected[:, column].mean())
        assert posterior_mixture_summary_at_dose(historical_lines, dose)["dlt_probability"] == pytest.approx(predictions[dose])


@pytest.mark.parametrize("helper", [*ESTIMATORS, sample_historical_posterior_curve_mixture,
    predict_article_line_weighted_at_doses, posterior_mixture_draws_at_doses,
    posterior_mixture_summary_at_dose])
def test_deleted_adjustment_keywords_are_not_accepted(helper):
    parameters = inspect.signature(helper).parameters
    assert "combination_alpha_uplift_prior" not in parameters
    assert "apply_combination_prior_adjustment" not in parameters
    assert "adjustment_seed_material" not in parameters


@pytest.mark.parametrize("estimator,replay", [
    (estimate_article_posterior_draw_pool, source_mixture_draws_at_doses),
    (estimate_power_likelihood_curve, power_likelihood_draws_at_doses),
    (estimate_power_likelihood_random_effects_curve, random_effects_draws_at_doses),
    (estimate_power_likelihood_random_intercept_slope_curve, random_intercept_slope_draws_at_doses),
])
def test_replay_matches_display_and_rejects_previously_adjusted_methods(estimator, replay, historical_lines):
    spec = ToxicityQuery("A", "mg", "combination", (1., 2., 4.))
    method = estimator(historical_lines, spec, bayesian_prior="test")
    assert method["status"] == "fitted"
    sampled = replay(method, spec.planned_doses)
    draws = sampled["posterior_draws"] if isinstance(sampled, dict) else sampled
    np.testing.assert_allclose(
        np.mean(draws, axis=0), [row["dlt_probability"] for row in method["curve_points"]],
        rtol=1e-12, atol=1e-12,
    )
    old_method = deepcopy(method)
    old_method["target_trial_type_prior_adjustment"] = {
        "trial_type": "combination", "enabled": True, "applied": True,
        "prior": {"mean": .3, "sd": .2},
    }
    with pytest.raises(ValueError, match="Regenerate"):
        replay(old_method, spec.planned_doses)
    old_method["target_trial_type_prior_adjustment"].update(enabled=False, applied=False)
    legacy_unadjusted = replay(old_method, spec.planned_doses)
    legacy_draws = (legacy_unadjusted["posterior_draws"]
                    if isinstance(legacy_unadjusted, dict) else legacy_unadjusted)
    np.testing.assert_array_equal(legacy_draws, draws)
