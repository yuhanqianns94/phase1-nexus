"""A coordinate change must preserve the historical latent-dose prior."""
import numpy as np
import pytest
from numpyro import handlers

from phase1_prediction.multidrug import power_prior as pp
from phase1_prediction.multidrug.contracts import PriorSpec
from test_multidrug_power_prior import article


@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('backend', ['logistic'])
def test_same_entity_conversion_preserves_unknown_physical_dose_prior(bounded, backend):
    row, target = article(k=1)
    exposure = row[0].observations[0].exposures[0]
    exposure.dose = None
    if bounded:
        exposure.lower, exposure.upper = 20., 500.
    original = row[0].model_dump()
    target.drugs[0].transport_mode = 'same_entity_unit_conversion'
    target.drugs[0].unit = 'g'
    distributions = []
    for target_reference in (.1, 1.):
        target.drugs[0].reference_dose = target_reference
        prepared = pp.prepare_power_prior([row], target, pp.PowerPriorSpec())
        scope = pp.ModelScope.model_validate(prepared['sources'][0]['scope'])
        trace = handlers.trace(handlers.seed(pp.power_prior_model, rng_seed=3)).get_trace(
            [scope], [.5], backend, PriorSpec(), pp.PowerPriorSpec())
        distributions.append(trace['article_0/missing_dose_0_0']['fn'])
    # The same physical doses expressed relative to different target references
    # must have the same density with respect to log physical dose.
    physical_mg = np.array([25., 50., 100., 200., 400.])
    np.testing.assert_allclose(distributions[0].log_prob(np.log(physical_mg / 100.)),
                               distributions[1].log_prob(np.log(physical_mg / 1000.)),
                               rtol=1e-10, atol=1e-10)
    assert row[0].model_dump() == original


def test_relative_analogy_keeps_source_center_at_one():
    row, target = article(k=1)
    row[0].observations[0].exposures[0].dose = None
    target.drugs[0].reference_dose = 1000.
    prepared = pp.prepare_power_prior([row], target, pp.PowerPriorSpec())
    scope = pp.ModelScope.model_validate(prepared['sources'][0]['scope'])
    trace = handlers.trace(handlers.seed(pp.power_prior_model, rng_seed=3)).get_trace(
        [scope], [.5], 'logistic', PriorSpec(), pp.PowerPriorSpec())
    assert float(trace['article_0/missing_dose_0_0']['fn'].loc) == 0.
