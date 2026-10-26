"""V3 independent-article Bayesian models and frozen posterior transport."""

VERSION = "v3_multidrug_bayes_4_single_drug"
TARGET_PREDICTION_VERSION = "multidrug_target_20261025_v17_single_drug"

BACKENDS = ('logistic', 'logistic_product', 'single_drug_logistic')


def validate_backend(backend):
    if backend not in BACKENDS:
        raise ValueError('Unsupported probability model; expected logistic, logistic_product, or single_drug_logistic')
