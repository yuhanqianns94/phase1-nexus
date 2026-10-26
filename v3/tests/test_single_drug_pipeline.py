"""Focal conditional-regimen transport and all three inference routes."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from phase1_prediction.multidrug import service, power_prior as pp
from phase1_prediction.multidrug.contracts import Exposure, Observation, RegimenPoint, PriorSpec
from phase1_prediction.multidrug.store import PosteriorStore
from phase1_prediction.multidrug.transport import transport_joint_posteriors
from test_multidrug_transport import fixture

BACKEND = 'single_drug_logistic'
MODES = ('same_article_joint_posterior_average', 'same_article_joint_posterior_mixture', 'target_power_prior')


def example(source_combo=True, target_combo=True):
    scope, _, _, target = fixture(2)
    target.profile['single_drug_analysis'] = {'focal_drug_id': 'A'}
    for point in target.points:
        point.doses['B'] = 25. if target_combo else 0.
    scope.observations = [Observation(observation_id=f'o{i}', cohort_id=f'c{i}', source_row_ids=[f'r{i}'],
        y=i+1, n=10, exposures=[Exposure(drug_id='A', presence='present', dose=dose),
            Exposure(drug_id='B', presence='present' if source_combo else 'absent', dose=25. if source_combo else None)])
        for i, dose in enumerate((50.,100.,200.))]
    mapping = service.match_target(service._focal_matching_target(target), scope)
    return [(scope, mapping, mapping['score']['similarity'])], target


def arrays():
    return {'alpha': np.full((2,20), -2.), 'log_b': np.zeros((2,20,2)),
            'eta': np.zeros((2,20,0)), 'delta': np.full((2,20), .4), 'tau': np.zeros((2,20)),
            'tau_slope': np.zeros((2,20))}


@pytest.mark.parametrize('source_combo', [False, True])
@pytest.mark.parametrize('target_combo', [False, True])
def test_context_enters_once_in_all_transferred_curves(source_combo, target_combo):
    scored, target = example(source_combo, target_combo)
    projected, computational, notices = service.prepare_single_drug_evidence(scored, target)
    assert not notices
    scope, mapping, _ = projected[0]
    assert len(scope.observations) == 3
    assert sum(o.n for o in scope.observations) == 30
    assert all(o.exposures[1].presence == ('present' if source_combo else 'absent') for o in scope.observations)
    entries=[(scope, arrays(), {'source_fit_id':'fixed','status':'fitted','backend':BACKEND})]
    for aggregation in ('average', 'mixture'):
        result=transport_joint_posteriors(entries, computational, backend=BACKEND,
            aggregation=aggregation, draw_count=8, mappings={scope.scope_id:mapping})
        np.testing.assert_allclose(result['posterior_draws'], np.broadcast_to(
            expit(-2+np.array([.5,1,2])+.4*target_combo), (8,3)))


def test_power_prior_preserves_split_contexts_and_does_not_duplicate_patients():
    scored, target = example()
    scored[0][0].observations[0].exposures[1].presence='absent'
    scored[0][0].observations[0].exposures[1].dose=None
    projected, computational, _=service.prepare_single_drug_evidence(scored,target)
    assert len(projected)==2
    prepared=pp.prepare_power_prior(projected,computational,pp.PowerPriorSpec())
    assert len(prepared['sources'])==1
    assert prepared['sources'][0]['patient_count']==30
    observations=prepared['sources'][0]['scope']['observations']
    assert len(observations)==3
    assert {o['exposures'][1]['presence'] for o in observations}=={'absent','present'}
    duplicated=pp.prepare_power_prior(projected+projected,computational,pp.PowerPriorSpec())
    assert not duplicated['sources']
    assert any('twice' in e['reason'] for e in duplicated['excluded'])


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('explicit_context', [False, True])
def test_service_routes_restore_full_plan_and_frozen_redraw(tmp_path,monkeypatch,mode,explicit_context):
    scored,target=example()
    if explicit_context:
        from phase1_web.single_drug_plan import single_drug_plan
        from phase1_prediction.multidrug.contracts import TargetRegimen
        target = TargetRegimen.model_validate(single_drug_plan(target.model_dump(mode='json')))
    monkeypatch.setattr(service,'_select',lambda *a,**kw:(scored,{'notices':[]}))
    def fit(scopes,**kw):
        store=PosteriorStore(kw['store_root']);entries=[]
        for scope in scopes:
            entries.append(store.save('fit_'+scope.scope_id,scope,arrays(),
                {'backend':BACKEND,'status':'fitted'}))
        return entries,[]
    monkeypatch.setattr(service,'fit_library',fit)
    monkeypatch.setattr(pp,'fit_power_prior',lambda *a:(arrays(),{'status':'fitted','inference':'test_fixed_joint_draws'}))
    result=service.predict(SimpleNamespace(),target,backend=BACKEND,synthesis=mode,
        store_root=tmp_path,draw_count=12,relevance_decisions={'paper':'Include'})
    method=result.methods[mode]
    assert method['status']=='fitted'
    assert method['target']==target.model_dump(mode='json')
    assert set(method['summary'][0]['doses'])==({'A'} if explicit_context else {'A','B'})
    assert method['evidence_construction']=='single_drug_conditional_regimen'
    np.testing.assert_allclose(method['posterior_draws'],np.broadcast_to(expit(-2+np.array([.5,1,2])+.4),(12,3)))
    revised=target.model_copy(deep=True)
    revised.points[0].doses['A']=12.5
    redrawn=service.evaluate_frozen_prediction(asdict(result),revised)
    redrawn_method=redrawn['methods'][mode]
    assert redrawn_method['target']==revised.model_dump(mode='json')
    assert redrawn_method['provenance']==method['provenance']
    np.testing.assert_allclose(np.array(redrawn_method['posterior_draws'])[:,0],expit(-2+.25+.4))
    if explicit_context:
        revised.profile['single_drug_analysis']['combination_present'] = False
    else:
        revised.points[0].doses['B']=0.
    with pytest.raises(ValueError,match='context changed'):
        service.evaluate_frozen_prediction(asdict(result),revised)


def test_backend_rejects_varying_companions_even_outside_ui():
    scored,target=example()
    target.points[0].doses['B']=10.
    with pytest.raises(ValueError,match='companion'):
        service.prepare_single_drug_evidence(scored,target)


def test_single_power_likelihood_context_and_discount_once():
    import jax
    from numpyro import handlers
    from scipy.stats import binom
    scored,target=example(source_combo=True)
    projected,computational,_=service.prepare_single_drug_evidence(scored,target)
    prepared=pp.prepare_power_prior(projected,computational,pp.PowerPriorSpec())
    scope=pp.ModelScope.model_validate(prepared['sources'][0]['scope'])
    values={'alpha':-2.,'ell':0.,'delta':.4,'tau':0.,'study_z':np.array([0.]),
            'tau_slope':0.,'study_slope_z':np.array([0.])}
    # Exact conditional likelihood values are compared across powers; priors are unchanged.
    traces=[handlers.trace(handlers.substitute(handlers.seed(pp.power_prior_model,0),data=values)).get_trace(
        [scope],[power],BACKEND,PriorSpec(),pp.PowerPriorSpec()) for power in (1.,.3)]
    likelihood_names=[name for name in traces[0] if 'cohort_' in name]
    assert len(likelihood_names)==3
    for i, name in enumerate(likelihood_names):
        first=traces[0][name]['fn'].log_prob(traces[0][name]['value'])
        second=traces[1][name]['fn'].log_prob(traces[1][name]['value'])
        np.testing.assert_allclose(second,.3*first)
        np.testing.assert_allclose(first,binom.logpmf(i+1,10,expit(-2+(.5,1.,2.)[i]+.4)))


def test_explicit_single_backend_precache_projects_each_numeric_focal(tmp_path,monkeypatch):
    scored,_=example()
    seen=[]
    def cache_fit(self,scope,backend,**kwargs):
        assert backend==BACKEND and scope.lineage.get('single_drug_projection')
        seen.append(scope)
        return scope,arrays(),{'backend':backend,'status':'fitted','source_fit_id':scope.scope_id}
    monkeypatch.setattr(PosteriorStore,'fit',cache_fit)
    entries,errors=service.fit_library([scored[0][0]],backends=(BACKEND,),store_root=tmp_path)
    assert entries and not errors
    assert {s.lineage['focal_source_id'] for s in seen}=={'A','B'}
    for focal in ('A','B'):
        assert sum(o.n for s in seen if s.lineage['focal_source_id']==focal for o in s.observations)==30
