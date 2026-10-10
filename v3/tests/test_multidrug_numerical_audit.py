"""Independent checks for allocation likelihoods, fit diagnostics and cache IDs."""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from scipy.stats import binom

from phase1_prediction.multidrug import fitting
from phase1_prediction.multidrug.contracts import Exposure, Observation, PriorSpec, SamplerSpec
from phase1_prediction.multidrug.store import PosteriorStore, engine_hash, source_fit_id
from phase1_prediction.multidrug.transport import assemble_components
from test_multidrug_average import two_articles
from test_multidrug_transport import fixture


@pytest.mark.parametrize('y',[0,1,3,5,8])
def test_allocation_dynamic_program_matches_explicit_count_enumeration(y):
    groups=[(2,.1),(1,.4),(3,.7),(2,.25)]
    expected=sum(np.prod([binom.pmf(c,n,p) for c,(n,p) in zip(counts,groups)])
        for counts in product(*(range(n+1) for n,p in groups)) if sum(counts)==y)
    assert float(fitting._allocation_logprob(y,groups))==pytest.approx(np.log(expected),abs=1e-12)


def test_many_subgroups_use_exact_convolution_with_finite_gradients():
    # Cartesian enumeration would inspect 4**20 allocations for this cohort.
    groups=[(3,p) for p in np.linspace(.05,.8,20)]
    distribution=np.array([1.])
    for n,p in groups:
        distribution=np.convolve(distribution,binom.pmf(np.arange(n+1),n,p))
    value=float(fitting._allocation_logprob(15,groups))
    assert value==pytest.approx(np.log(distribution[15]),abs=1e-11)
    fn=lambda p:fitting._allocation_logprob(3,[(3,p),(2,.3),(2,.6)])
    derivative=float(jax.grad(fn)(.2))
    def independent(p):
        pmf=np.convolve(np.convolve(binom.pmf(np.arange(4),3,p),binom.pmf(np.arange(3),2,.3)),binom.pmf(np.arange(3),2,.6))
        return np.log(pmf[3])
    expected=(independent(.200001)-independent(.199999))/.000002
    assert derivative==pytest.approx(expected,rel=1e-7)


@pytest.mark.parametrize('field,bad_value',[('r_hat',np.nan),('n_eff',np.nan),('n_eff',np.inf)])
def test_one_nonfinite_diagnostic_component_rejects_the_source_fit(monkeypatch,field,bad_value):
    scope,arrays,_,_=fixture(2)
    scope.observations=[Observation(observation_id='row',cohort_id='cohort',n=3,y=1,
        exposures=[Exposure(drug_id='A',presence='present',dose=50),Exposure(drug_id='B',presence='present',dose=50)])]
    class FakeMCMC:
        def __init__(self,*args,**kwargs):pass
        def run(self,*args,**kwargs):pass
        def get_samples(self,**kwargs):return arrays
        def get_extra_fields(self):return {'diverging':np.zeros((2,20),dtype=bool)}
    stats={'r_hat':np.array([1.01,1.01]),'n_eff':np.array([500.,500.])}
    stats[field][1]=bad_value
    monkeypatch.setattr(fitting,'MCMC',FakeMCMC)
    monkeypatch.setattr(fitting,'summary',lambda *a,**kw:{'a':stats})
    _,diagnostics=fitting.fit_source(scope,'logistic',sampler=SamplerSpec(retry_count=0))
    assert diagnostics['status']=='numerical_failure'
    assert diagnostics['attempts'][0]['passed'] is False


def test_numerical_runtime_upgrade_changes_new_fit_id_without_breaking_old_cache(tmp_path,monkeypatch):
    scope,arrays,manifest,_=fixture(1)
    prior,sampler=PriorSpec(),SamplerSpec()
    old_engine=engine_hash()
    old_id=source_fit_id(scope,'logistic',prior,sampler)
    store=PosteriorStore(tmp_path)
    store.save(old_id,scope,arrays,{**manifest,'engine_hash':old_engine,'prior':prior.model_dump(),'sampler':sampler.model_dump()})
    monkeypatch.setattr(numpyro,'__version__',numpyro.__version__+'-test-upgrade')
    assert engine_hash()!=old_engine
    assert source_fit_id(scope,'logistic',prior,sampler)!=old_id
    restored_scope,restored_arrays,restored_manifest=store.load(old_id)
    assert restored_scope==scope and restored_manifest['source_fit_id']==old_id
    np.testing.assert_array_equal(restored_arrays['a'],arrays['a'])


def test_component_pool_metadata_covers_unsampled_candidates_and_is_coordinate_independent():
    entries,target=two_articles('logistic')
    result=assemble_components(entries,target,draw_count=1)
    pool=result['component_pools'][0]
    assert pool['target_id']=='A'
    assert {c['source_fit_id'] for c in pool['candidates']}=={'fit_0','fit_1'}
    assert sum(c['candidate_weight'] for c in pool['candidates'])==pytest.approx(1.)
    assert len(result['provenance'][0]['components'])==1
    target.points=list(reversed(target.points))
    redrawn=assemble_components(entries,target,draw_count=1)
    assert redrawn['component_pools']==result['component_pools']
    np.testing.assert_allclose(redrawn['posterior_draws'],np.array(result['posterior_draws'])[:,::-1])
