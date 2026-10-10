import numpy as np
import pytest
from phase1_prediction.multidrug.contracts import DrugSlot, ModelScope, TargetRegimen, TargetDrug, RegimenPoint, PriorSpec, SamplerSpec
from phase1_prediction.multidrug.store import PosteriorStore, source_fit_id
from phase1_prediction.multidrug.mapping import match_target
from phase1_prediction.multidrug.transport import transport_joint_posteriors, assemble_components
from phase1_simulation.multidrug import path_matrix, simulate_regimen_path
from phase1_simulation.service import BoinSettings


def fixture(k=2, backend='logistic'):
    profiles=[{'name':chr(65+i),'mechanisms':['shared kinase mechanism'],'role':'escalated' if i==0 else 'combination_partner',
               'model_drug_id':chr(65+i)} for i in range(k)]
    scope=ModelScope(document_id='paper',scope_id='scope',label='A+B',drugs=[
        DrugSlot(model_drug_id=chr(65+i),name=chr(65+i),research_priority=i+1,research_role='primary' if i==0 else 'partner',
                 mode='numeric',reference_dose=100,unit='mg',profile=profiles[i]) for i in range(k)],
        observations=[],profile={'configurations':[{'phase':'Phase I','drugs':profiles}]})
    arrays={'a':np.full((2,20,k),-2.),'log_b':np.zeros((2,20,k)),
            'eta':np.zeros((2,20,0 if k==1 else 1 if k==2 else 4))}
    manifest={'source_fit_id':'fit_x','backend':backend,'status':'fitted','sampler':{}}
    target=TargetRegimen(name='target',drugs=[TargetDrug(drug_id=chr(65+i),name=chr(65+i),research_priority=i+1,
        research_role='primary' if i==0 else 'partner',reference_dose=50,unit='mg',profile=profiles[i]) for i in range(k)],
        points=[RegimenPoint(level_id=f'L{j}',doses={chr(65+i):float(d) for i in range(k)}) for j,d in enumerate([25,50,100])],
        path_level_ids=['L0','L1','L2'])
    return scope,arrays,manifest,target


@pytest.mark.parametrize('backend',['logistic'])
def test_joint_transport_keeps_draws_and_is_reproducible(backend):
    s,a,m,t=fixture(3,backend)
    result=transport_joint_posteriors([(s,a,m)],t,backend=backend,draw_count=10)
    again=transport_joint_posteriors([(s,a,m)],t,backend=backend,draw_count=10)
    assert result==again and result['status']=='fitted'
    assert len(result['provenance'])==10
    assert np.all(np.diff(result['posterior_draws'],axis=1)>0)


def test_single_target_removes_other_source_drugs():
    s,a,m,t=fixture()
    t=TargetRegimen(name='single',drugs=t.drugs[:1],points=[RegimenPoint(level_id='one',doses={'A':50.})])
    result=transport_joint_posteriors([(s,a,m)],t,draw_count=4)
    assert np.allclose(result['posterior_draws'],1/(1+np.exp(2)))
    assert result['contributors'][0]['mapping']['removed_source_ids']==['B']


def test_whole_source_parameter_draw_stays_joint():
    s,a,m,t=fixture()
    a['a'][:,:,0]=np.arange(40).reshape(2,20)/40-2
    a['a'][:,:,1]=-a['a'][:,:,0]
    result=transport_joint_posteriors([(s,a,m)],t,draw_count=20)
    for row,trace in zip(result['posterior_draws'],result['provenance']):
        logits=a['a'][trace['chain'],trace['draw']]
        q=1/(1+np.exp(-logits))
        assert row[1]==pytest.approx(1-np.prod(1-q))


@pytest.mark.parametrize('backend',['logistic'])
def test_assembly_preserves_same_fit_interactions(backend):
    s,a,m,t=fixture(2,backend)
    result=assemble_components([(s,a,m)],t,backend=backend,draw_count=5)
    assert result['status']=='fitted'
    for trace in result['provenance']:
        assert len({c['flat_draw'] for c in trace['components']})==1
        if backend=='logistic':assert trace['interactions'][0]['basis']=='source_joint_draw'
        else:assert len(trace['interactions']['retained_pairs'])==1


def test_cache_is_source_only_and_checksummed(tmp_path):
    s,a,m,t=fixture()
    prior,sampler=PriorSpec(),SamplerSpec()
    fid=source_fit_id(s,'logistic',prior,sampler)
    t.drugs[0].reference_dose=200
    assert fid==source_fit_id(s,'logistic',prior,sampler)
    store=PosteriorStore(tmp_path);store.save(fid,s,a,m)
    with (tmp_path/fid/'draws.npz').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='checksum'):store.load(fid)


def test_path_keeps_nonmonotone_truth_and_full_regimen():
    s,a,m,t=fixture()
    result=transport_joint_posteriors([(s,a,m)],t,draw_count=5)
    result['posterior_draws']=[[.1,.4,.2]]*5
    matrix,points,_=path_matrix(result)
    assert matrix[0].tolist()==[.1,.4,.2] and points[0]['doses']=={'A':25.,'B':25.}
    sim=simulate_regimen_path(result,settings=BoinSettings(),posterior_curve_count=2,trials_per_curve=3)
    assert sim['nonmonotone_truth_fraction']==1 and not sim['truth_was_projected_or_sorted']
    assert sim['total_trials']==6
    result['target']['path_level_ids']=[]
    with pytest.raises(ValueError,match='explicitly ordered'):path_matrix(result)


def test_one_point_and_prior_only_status():
    s,a,m,t=fixture(1)
    t.points=t.points[:1];t.path_level_ids=[]
    assert transport_joint_posteriors([(s,a,m)],t,draw_count=2)['status']=='fitted'
    m['status']='prior_only'
    result=transport_joint_posteriors([(s,a,m)],t,draw_count=2)
    assert result['status']=='no_similarity_support'
    assert result['excluded'][0]['reason']=='prior_only'
