import numpy as np
import pytest
from scipy.stats import binom
from numpyro import handlers
from phase1_prediction.multidrug.contracts import Observation, Exposure, PriorSpec, RegimenPoint
from phase1_prediction.multidrug.fitting import source_model, _allocation_logprob
from test_multidrug_transport import fixture


@pytest.mark.parametrize('backend',['logistic'])
def test_completely_unknown_numeric_dose_has_finite_prior(backend):
    scope,*_=fixture(1,backend)
    scope.observations=[Observation(observation_id='one',cohort_id='one',y=0,n=3,
        exposures=[Exposure(drug_id='A',presence='present')])]
    trace=handlers.trace(handlers.seed(source_model,rng_seed=6)).get_trace(scope,backend,PriorSpec())
    site=trace['missing_dose_0_0']
    assert np.isfinite(site['fn'].log_prob(site['value']))
    assert np.isfinite(trace['cohort_0']['fn'].log_factor)


def test_known_dose_subgroups_use_exact_aggregate_likelihood():
    value=float(np.exp(_allocation_logprob(1,[(3,.1),(1,.4)])))
    expected=binom.pmf(0,3,.1)*binom.pmf(1,1,.4)+binom.pmf(1,3,.1)*binom.pmf(0,1,.4)
    assert value==pytest.approx(expected)


@pytest.mark.parametrize('lower,upper',[(50.,None),(None,200.),(50.,200.)])
def test_one_sided_missing_dose_bounds_initialize(lower,upper):
    import jax
    from numpyro.infer.util import initialize_model
    scope,*_=fixture(1)
    scope.observations=[Observation(observation_id='one',cohort_id='one',y=0,n=3,
        exposures=[Exposure(drug_id='A',presence='present',lower=lower,upper=upper)])]
    initialized=initialize_model(jax.random.PRNGKey(3),source_model,model_args=(scope,'logistic',PriorSpec()))
    assert np.isfinite(initialized.param_info.potential_energy)


def test_slot_order_is_canonicalized_and_bool_is_not_one_mg():
    scope,*_=fixture()
    payload=scope.model_dump()
    payload['observations']=[Observation(observation_id='one',cohort_id='one',y=0,n=3,
        exposures=[Exposure(drug_id='B',dose=200),Exposure(drug_id='A',dose=50)]).model_dump()]
    fixed=type(scope).model_validate(payload)
    assert [(e.drug_id,e.dose) for e in fixed.observations[0].exposures]==[('A',50.),('B',200.)]
    with pytest.raises(ValueError,match='true is not a dose'):
        RegimenPoint(level_id='L1',doses={'A':True})
