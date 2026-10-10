"""V3 independent-article Bayesian models and frozen posterior transport."""

VERSION = "v3_multidrug_bayes_2"
TARGET_PREDICTION_VERSION = "multidrug_target_20261006_v14"

BACKENDS = ('logistic', 'logistic_product')


def validate_backend(backend):
    if backend not in BACKENDS:
        raise ValueError('Unsupported probability model; expected logistic or logistic_product')
