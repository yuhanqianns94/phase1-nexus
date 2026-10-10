"""Names and mathematical helpers for the three approved v1-baseline methods.

The former independent v3 optimizer has been removed. Prediction requires the
full target query and publication corpus; call phase1_prediction.service.
"""
import numpy as np
from .reference.power_likelihood_random_intercept_slope import (
    DEFAULT_INTERCEPT_SCALE_LATENT_MEAN as DEFAULT_RANDOM_INTERCEPT_SCALE_LATENT_MEAN,
    DEFAULT_INTERCEPT_SCALE_LATENT_SD as DEFAULT_RANDOM_INTERCEPT_SCALE_LATENT_SD,
    DEFAULT_SLOPE_SCALE_LATENT_MEAN as DEFAULT_RANDOM_SLOPE_SCALE_LATENT_MEAN,
    DEFAULT_SLOPE_SCALE_LATENT_SD as DEFAULT_RANDOM_SLOPE_SCALE_LATENT_SD,
    hierarchical_linear_predictor,
)
from .service import PREDICTION_MODEL_VERSION

METHOD_LABELS = {
    "article_line_weighted": "Average of article posterior curves",
    "article_posterior_draw_pool": "Mixture of complete article posterior curves",
    "power_likelihood_random_intercept_slope": "Power-likelihood model with random intercepts and slopes",
}


def _hierarchical_linear_predictor(alpha, eta, offsets, group_array, *, intercept_effects=None, slope_effects=None):
    return hierarchical_linear_predictor(alpha, eta, offsets,
        random_intercept=0.0 if intercept_effects is None else np.asarray(intercept_effects)[group_array],
        random_slope_latent=0.0 if slope_effects is None else np.asarray(slope_effects)[group_array])
