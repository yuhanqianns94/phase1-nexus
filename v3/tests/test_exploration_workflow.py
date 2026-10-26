import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from test_multidrug_transport import fixture
from test_multidrug_average import two_articles
from phase1_prediction.multidrug.workflow import phase_status, article_budgets, exploration_target, validate_boin_target
from phase1_prediction.multidrug.transport import average_joint_posteriors, transport_joint_posteriors, assemble_components
from phase1_prediction.multidrug import service
from phase1_prediction.multidrug.contracts import PriorSpec, SamplerSpec, fingerprint
from phase1_prediction.multidrug.store import PosteriorStore


@pytest.mark.parametrize('value,expected', [('Phase I','phase_i'),('Phase 1','phase_i'),('Ia','phase_i'),('Phase I expansion','phase_i'),('I component of I/II','phase_i'),('Phase II','out_of_scope'),('Phase 1/2a','mixed_requires_separation'),('Phase I/II','mixed_requires_separation'),(None,'unknown')])
def test_phase_filter(value,expected):
    assert phase_status({'configurations':[{'phase':value}]})==expected


def test_article_review_is_once_per_article_not_per_scope():
    rows=[{'document_id':'a','similarity':.8,'coverage':.5}, {'document_id':'a','similarity':.4,'coverage':.5}, {'document_id':'b','similarity':.4}]
    values=article_budgets(rows,{'a':'Uncertain','b':'Include'})
    assert [r['normalized_weight'] for r in values]==pytest.approx([.5,.5])
    assert all(r['normalized_weight']==0 for r in article_budgets(rows,{'a':'Exclude','b':'Exclude'}))
    with pytest.raises(ValueError,match='Complete relevance'):article_budgets(rows,{'a':'Include'})


@pytest.mark.parametrize('fn',[average_joint_posteriors,transport_joint_posteriors,assemble_components])
def test_review_multipliers_enter_all_borrowing_methods(fn):
    entries,target=two_articles('logistic')
    r=fn(entries,target,draw_count=100,relevance_decisions={'paper_0':'Exclude','paper_1':'Include'})
    assert np.allclose(np.array(r['posterior_draws'])[:,1],.5)
    r=fn(entries,target,draw_count=10,relevance_decisions={'paper_0':'Exclude','paper_1':'Exclude'})
    assert r['status']!='fitted' and not r['posterior_draws']


def test_optional_plan_stays_unchanged_relative_grid_is_not_boin():
    _,_,_,t=fixture()
    draft=t.model_dump();draft['points']=[];draft['path_level_ids']=[]
    original=copy.deepcopy(draft)
    grid=exploration_target(draft,{'A':[.25,.5,1.,2.],'B':[None]},['A'])
    assert draft==original
    assert grid.drugs[0].reference_dose==1 and grid.drugs[0].unit=='relative dose'
    assert grid.drugs[1].mode=='background' and grid.points[0].doses['B'] is None
    grid.path_level_ids=['G1','G2']
    with pytest.raises(ValueError,match='actual candidate'):validate_boin_target(grid)
    with pytest.raises(ValueError,match='positive numeric'):exploration_target(draft,{'A':[1.,2.],'B':[None,None]},['A','B'])


@pytest.mark.parametrize('synthesis',service.SYNTHESIS_MODES)
def test_regrid_reuses_exact_draws_without_fitting(tmp_path,monkeypatch,synthesis):
    scope,arrays,manifest,target=fixture()
    mapping={'score':{'similarity':1.}}
    if synthesis == 'target_power_prior':
        from test_multidrug_power_prior import article, fake_fit
        from phase1_prediction.multidrug import power_prior
        (scope,mapping,_),target=article()
        monkeypatch.setattr(power_prior,'fit_power_prior',fake_fit)
    store=PosteriorStore(tmp_path)
    entry=store.save('fit_x',scope,arrays,manifest)
    monkeypatch.setattr(service,'_select',lambda *a,**kw: ([(scope,mapping,1.)],{'notices':[]}))
    monkeypatch.setattr(service,'fit_library',lambda *a,**kw: ([entry],[]))
    p=service.predict(None,target,synthesis=synthesis,store_root=tmp_path,draw_count=5).to_dict()
    monkeypatch.setattr(PosteriorStore,'fit',lambda *a,**kw:pytest.fail('Redraw must not fit'))
    if synthesis == 'target_power_prior':
        monkeypatch.setattr(power_prior.PowerPriorStore,'fit',lambda *a,**kw:pytest.fail('Redraw must not fit the target model'))
    target=target.model_copy(deep=True)
    target.points=list(reversed(target.points))
    new=service.evaluate_frozen_prediction(p,target)
    old=next(iter(p['methods'].values()));result=next(iter(new['methods'].values()))
    np.testing.assert_allclose(result['posterior_draws'],np.array(old['posterior_draws'])[:,::-1])
    assert result['provenance']==old['provenance']
    assert new['run_id']!=p['run_id']


def test_excluded_and_phase_ii_sources_never_fit(monkeypatch):
    a,arrays,manifest,t=fixture(1)
    b=a.model_copy(deep=True);b.document_id='excluded';b.scope_id='excluded'
    c=a.model_copy(deep=True);c.document_id='phase2';c.scope_id='phase2';c.profile['configurations'][0]['phase']='Phase II'
    monkeypatch.setattr(service,'load_model_library',lambda *a,**k:([a_scope,b,c],{'notices':[]}))
    a_scope=a
    seen=[]
    def fit(scopes,**kwargs):
        seen.extend(s.document_id for s in scopes)
        return [(a,arrays,manifest)],[]
    monkeypatch.setattr(service,'fit_library',fit)
    lib=SimpleNamespace(source_roots=[],evidence_policy='project_default',review_policy='extracted')
    service.predict(lib,t,draw_count=2,relevance_decisions={'paper':'Include','excluded':'Exclude'})
    assert seen==['paper']
    with pytest.raises(ValueError,match='No included'):service.predict(lib,t,relevance_decisions={'paper':'Exclude','excluded':'Exclude'})


def test_embedding_context_covers_correspondence_and_nested_scoring(tmp_path,monkeypatch):
    from phase1_prediction.similarity_engine import similarity_context
    from phase1_prediction.profile_similarity import text_score,compare_mapped_profiles
    from phase1_prediction.multidrug.mapping import match_target
    from phase1_prediction.reference import toxicity_similarity
    calls=[]
    def embed(texts,model):calls.extend(texts);return [[1.,0.] for _ in texts]
    monkeypatch.setattr(toxicity_similarity,'embed_texts_openai',embed)
    s,a,m,t=fixture(1)
    s.drugs[0].name='X';s.drugs[0].profile.update(name='X',mechanisms=['alpha'])
    s.profile['configurations'][0]['drugs']=[s.drugs[0].profile]
    t.drugs[0].profile.update(name='Y',mechanisms=['beta']);t.drugs[0].name='Y'
    with similarity_context('local'):local=match_target(t,s)
    with similarity_context('openai',cache=tmp_path/'cache.json'):
        result=match_target(t,s)
        assert text_score('alpha','beta')==1.
    assert result['pairs'] and result['score']['similarity']!=local['score']['similarity']
    count=len(calls)
    with similarity_context('openai',cache=tmp_path/'cache.json'):match_target(t,s)
    assert len(calls)==count
    with similarity_context('local'):assert text_score('alpha','beta')==0.
    def broken(*a,**kw):raise RuntimeError('offline')
    monkeypatch.setattr(toxicity_similarity,'embed_texts_openai',broken)
    with pytest.raises(RuntimeError,match='no local fallback'):
        with similarity_context('openai',cache=tmp_path/'other.json'):text_score('new','text')


def test_main_xml_abstract_separate_from_keypoints(tmp_path):
    from phase1_web.evidence_review import abstract_for
    xml=tmp_path/'paper.xml';xml.write_text('<article><front><article-meta><abstract abstract-type="keypoints"><p>Extra</p></abstract><abstract><sec><title>Methods</title><p>Original <italic>words</italic>.</p></sec></abstract></article-meta></front></article>')
    (tmp_path/'document_ir.json').write_text(json.dumps({'materials':[{'path':str(xml)}],'blocks':[]}))
    result=abstract_for(tmp_path)
    assert result['main']==[{'heading':'Methods'},{'text':'Original words.'}]
    assert result['additional'][0]['sections']==[{'text':'Extra'}]


def test_optional_target_and_abstract_review_web(tmp_path,monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from test_prediction_pipeline import _article
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    from phase1_prediction.multidrug.observations import load_model_library
    from phase1_web.multidrug_examples import target_from_scope
    scopes,_=load_model_library([tmp_path])
    app.session_state['multidrug_target_text']=target_from_scope(scopes[0]).model_dump_json()
    draft=json.loads(app.session_state['multidrug_target_text']);draft['points']=[];draft['path_level_ids']=[]
    app.session_state['multidrug_target_text']=json.dumps(draft);app.session_state['multidrug_reset_editor']=True
    app.run()
    assert not app.exception and not app.error
    assert 'II' not in next(s for s in app.selectbox if s.label=='Planned study phase').options
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert next(r for r in app.radio if r.label=='Have you chosen actual doses?').value=='Relative-dose exploration'
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(r for r in app.radio if r.label=='Evidence screening').set_value('Review article abstracts').run()
    next(b for b in app.button if b.label=='Find matching articles').click().run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert next(b for b in app.button if b.label=='Generate prediction').disabled
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(r for r in app.radio if r.label=='Relevance decision').set_value('Uncertain').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert not next(b for b in app.button if b.label=='Generate prediction').disabled
    assert list((tmp_path/'state'/'relevance_reviews').glob('*.json'))
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(r for r in app.radio if r.label=='Relevance decision').set_value('Exclude').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert next(b for b in app.button if b.label=='Generate prediction').disabled


def test_relative_prediction_results_and_boin_editor_web(tmp_path,monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from test_prediction_pipeline import _article
    _article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    from test_single_drug_pipeline import example, arrays, BACKEND
    scored,t=example()
    target=exploration_target(t.model_dump(),{'A':[.5,1.,2.],'B':[1.]},['A'])
    projected,computational,notices=service.prepare_single_drug_evidence(scored,target)
    assert not notices
    entries=[(scope,arrays(),{'source_fit_id':scope.scope_id,'backend':BACKEND,'status':'fitted'})
             for scope,_,_ in projected]
    method=transport_joint_posteriors(entries,computational,backend=BACKEND,draw_count=4,
        mappings={scope.scope_id:mapping for scope,mapping,_ in projected})
    service._restore_single_target(method,target)
    assert method['computational_target'] != method['target']
    result={'run_id':'test_relative','target':target.model_dump(mode='json'),
            'methods':{'same_article_joint_posterior_mixture':method},'evidence_contract':{'backend':BACKEND}}
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    app.session_state['multidrug_prediction']=result
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert not app.exception and not app.error
    app.sidebar.radio[0].set_value('OC Simulation').run()
    assert not app.exception
    assert not any(b.label=='Run BOIN' for b in app.button)
    assert any(x.label=='A actual dose unit' for x in app.text_input)
