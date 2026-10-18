"""Offline end-to-end safeguards for exact cohorts and restart-safe paid submission."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from phase1_acquisition import discovery, campaign, preparation
from phase1_acquisition.screening import MODEL, EFFORT, CRITERIA
from phase1_extraction.artifacts import read_json, write_json


class Metadata:
    def __init__(self):
        self.searches = 0
        self.short = False
        self.missing = False
    def ncbi_params(self):
        return {}
    def get(self, url, params):
        assert url == discovery.EUTILS + 'esearch.fcgi'
        assert params['db']=='pubmed' and params['sort']=='pub_date' and params['retstart']==0
        self.searches += 1
        return {'esearchresult':{'count':'48000','idlist':[str(n) for n in range(discovery.SEARCH_LIMIT if not self.short else discovery.SEARCH_LIMIT - 1,0,-1)]}}
    def pubmed_records(self, ids, origin):
        # EFetch is allowed to return a different order; preserve search ranking.
        return [{'pmid':p,'title':'Phase I trial '+p,'abstract':'Actual dose and safety.', 'origins':[origin]}
                for p in reversed(ids[:-1] if self.missing else ids)]


def test_frozen_ranking_preserves_order_and_ignores_prior_library(tmp_path):
    write_json(tmp_path/'screening_ledger.json',{'previously_screened':['4000']})
    client=Metadata()
    found=discovery.discover(tmp_path,through='2026/10/12',client=client)
    assert found['candidate_count']==discovery.SEARCH_LIMIT
    assert [c['pmid'] for c in found['candidates']]==[str(n) for n in range(discovery.SEARCH_LIMIT,0,-1)]
    assert found['candidates'][0]['rank']==1 and found['candidates'][-1]['rank']==discovery.SEARCH_LIMIT
    assert discovery.discover(tmp_path,through='2026/10/12',client=client)==found
    assert client.searches==1
    with pytest.raises(ValueError,match='configuration changed'):
        discovery.discover(tmp_path,through='2026/10/13',client=client)


def test_short_selection_or_missing_metadata_cannot_shrink_cohort(tmp_path):
    client=Metadata();client.short=True
    with pytest.raises(ValueError,match='distinct PMIDs'):
        discovery.discover(tmp_path,through='2026/10/12',client=client)
    client.short=False;client.missing=True
    with pytest.raises(ValueError,match='all requested PMIDs'):
        discovery.discover(tmp_path,through='2026/10/12',client=client)
    frozen=read_json(tmp_path/'pubmed_search.json')
    client.missing=False
    assert discovery.discover(tmp_path,through='2026/10/12',client=client)['candidate_count']==discovery.SEARCH_LIMIT
    assert read_json(tmp_path/'pubmed_search.json')==frozen
    assert client.searches==2


def test_pubmed_book_record_is_not_silently_dropped(tmp_path,monkeypatch):
    c=discovery.PublicMetadata(tmp_path)
    xml='<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>42</PMID><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList><Book><BookTitle>A study</BookTitle></Book><Abstract><AbstractText>Actual results.</AbstractText></Abstract></BookDocument></PubmedBookArticle></PubmedArticleSet>'
    monkeypatch.setattr(c,'get',lambda *a,**k:xml)
    r=c.pubmed_records(['42'],{'source':'pubmed'})[0]
    assert r['pmid']=='42' and r['title']=='A study' and r['abstract']=='Actual results.'
    assert r['publication_types']==['Book']


class BatchClient:
    def __init__(self, fail_at=None):
        self.creates=0;self.uploads=[];self.inputs={};self.fail_at=fail_at
        self.files=SimpleNamespace(create=self.upload,content=self.content)
        self.batches=SimpleNamespace(create=self.create,retrieve=self.retrieve)
    def with_options(self,**kw):
        assert kw=={'max_retries':0}
        return self
    def upload(self, file, **kw):
        lines=[json.loads(l) for l in file.read().decode().splitlines()]
        self.uploads.append(lines)
        return SimpleNamespace(id=str(len(self.uploads)))
    def create(self,input_file_id,**kw):
        self.creates+=1
        if self.creates==self.fail_at:raise TimeoutError('unknown create outcome')
        self.inputs['batch'+input_file_id]=self.uploads[int(input_file_id)-1]
        return SimpleNamespace(id='batch'+input_file_id,status='validating')
    def retrieve(self,identity):
        return SimpleNamespace(id=identity,status='completed',output_file_id=identity,error_file_id=None,
            request_counts=SimpleNamespace(model_dump=lambda:{'completed':len(self.inputs[identity]),'failed':0}))
    def content(self,identity):
        lines=[]
        for r in self.inputs[identity]:
            doc=json.loads(r['body']['input'][0]['content'][0]['text'])
            result={'document_id':doc['document_id'],'decision':'exclude','reason':'The required trial evidence is not established',
                    'checks':[{'criterion':c,'satisfied':False,'reason':'Not established','evidence':[]} for c in CRITERIA]}
            body={'status':'completed','model':MODEL,'reasoning':{'effort':EFFORT},
                  'usage':{'input_tokens':100,'output_tokens':100},
                  'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(result)}]}]}
            lines.append({'custom_id':r['custom_id'],'response':{'status_code':200,'body':body}})
        return SimpleNamespace(content='\n'.join(json.dumps(x) for x in lines).encode())


@pytest.fixture
def small_cohort(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign,'ARTICLE_COUNT',5)
    monkeypatch.setattr(campaign,'MAX_BATCH_REQUESTS',2)
    candidates=[{'document_id':f'PMID{n}','pmid':str(n),'title':f'Study {n}'} for n in range(5)]
    def found(workspace,**kw):
        obj={'candidate_count':len(candidates),'candidates':candidates}
        write_json(workspace/'ranked_candidates.json',obj)
        return obj
    monkeypatch.setattr(campaign,'discover',found)
    for c in candidates:
        path = tmp_path / (c['document_id'] + '.xml')
        path.write_text('<article><front><article-meta><article-id pub-id-type="pmid">'+c['pmid']+'</article-id></article-meta></front><body><sec><title>Results</title><p>Patients received treatment. Observed safety results.</p></sec></body></article>')
        c['fixture_path'] = str(path)
    monkeypatch.setattr(preparation,'fetch_fulltext',lambda c,*a,**kw:Path(c['fixture_path']))
    return candidates


def test_every_article_submitted_once_then_collected_without_inference(tmp_path,small_cohort):
    client=BatchClient()
    r=campaign.run(tmp_path,client=client)
    assert r['submitted']==5 and len(r['batch_ids'])==3 and not r['monitoring_enabled']
    campaign.run(tmp_path,client=client)
    assert client.creates==3
    result=campaign.collect_results(tmp_path,client=client)
    assert result['complete'] and result['excluded']==5 and result['technical_failures']==0
    assert campaign.run(tmp_path,client=client)['status']=='completed'
    assert client.creates==3
    plan=campaign.validate_campaign(tmp_path)
    assert plan['document_ids']==[c['document_id'] for c in small_cohort]


def test_whole_budget_checked_before_any_upload(tmp_path,small_cohort):
    client=BatchClient()
    with pytest.raises(ValueError,match='Total reserve'):
        campaign.run(tmp_path,max_usd=.000001,client=client)
    assert not client.uploads and client.creates==0


def test_ambiguous_create_never_retries_or_submits_later_shards(tmp_path,small_cohort):
    client=BatchClient(fail_at=2)
    with pytest.raises(TimeoutError):campaign.run(tmp_path,client=client)
    assert campaign.status(tmp_path)['submitted']==2
    with pytest.raises(RuntimeError,match='unknown'):campaign.run(tmp_path,client=client)
    assert client.creates==2 and len(client.uploads)==2
    assert not (tmp_path/'batches/003/screening_state.json').exists()


def test_preparation_failure_stops_all_paid_submission_and_resumes_free(tmp_path,small_cohort,monkeypatch):
    def fail(c,*a,**kw):
        if c['pmid']=='3':raise TimeoutError('download interrupted')
        return Path(c['fixture_path'])
    monkeypatch.setattr(preparation,'fetch_fulltext',fail)
    client=BatchClient()
    with pytest.raises(ValueError,match='no partial cohort'):campaign.run(tmp_path,client=client)
    assert client.creates==0 and len(list((tmp_path/'documents').glob('*.json')))==4
    calls=[]
    def recover(c,*a,**kw):
        calls.append(c['pmid'])
        return Path(c['fixture_path'])
    monkeypatch.setattr(preparation,'fetch_fulltext',recover)
    assert campaign.run(tmp_path,client=client)['submitted']==5
    assert calls==['3']


def test_changed_checkpoint_or_plan_stops_before_submission(tmp_path,small_cohort):
    preparation.prepare_documents(small_cohort,tmp_path)
    p=tmp_path/'documents/PMID0.json';r=read_json(p);r['document']['blocks'][0]['text']='altered';write_json(p,r)
    client=BatchClient()
    with pytest.raises(ValueError,match='no partial cohort'):campaign.run(tmp_path,client=client)
    assert client.creates==0


def test_size_sharding_preserves_every_document(tmp_path,small_cohort,monkeypatch):
    docs=preparation.prepare_documents(small_cohort,tmp_path)
    size=len(json.dumps(campaign.make_request(docs[0]),ensure_ascii=False).encode())+512
    monkeypatch.setattr(campaign,'MAX_BATCH_BYTES',size+1)
    groups=list(campaign._document_groups(docs))
    assert len(groups)==5 and [d for group in groups for d in group]==docs


def test_missing_fulltext_does_not_consume_places_and_scanning_continues(tmp_path, small_cohort, monkeypatch):
    for n in range(5, 8):
        path = tmp_path / f'PMID{n}.xml'
        path.write_text('<article><body><p>Actual patients, dose and safety.</p></body></article>')
        small_cohort.append({'document_id':f'PMID{n}', 'pmid':str(n), 'title':'Study', 'fixture_path':str(path)})
    def source(c, *a, **kw):
        return None if c['pmid'] in {'1', '3'} else Path(c['fixture_path'])
    monkeypatch.setattr(preparation, 'fetch_fulltext', source)
    client = BatchClient()
    state = campaign.run(tmp_path, client=client)
    assert state['selected'] == state['submitted'] == 5
    assert state['skipped_without_fulltext'] == 2 and state['examined_candidates'] == 7
    docs = [json.loads(r['body']['input'][0]['content'][0]['text']) for rows in client.uploads for r in rows]
    assert [d['document_id'] for d in docs] == ['PMID0', 'PMID2', 'PMID4', 'PMID5', 'PMID6']
    assert all(d['fulltext_available'] and d['materials'] for d in docs)
    assert not (tmp_path/'documents/PMID7.json').exists()
    result = campaign.collect_results(tmp_path, client=client)
    assert result['excluded'] == result['model_screening_count'] == 5
    assert result['excluded_without_fulltext'] == result['pending'] == 0
    assert result['technical_failures'] == 0
    assert not read_json(tmp_path/'retained_candidates.json')['articles']


def test_insufficient_fulltext_stops_without_model_calls(tmp_path, small_cohort, monkeypatch):
    monkeypatch.setattr(preparation, 'fetch_fulltext', lambda *a, **kw: None)
    client = BatchClient()
    with pytest.raises(ValueError, match='pool exhausted'):
        campaign.run(tmp_path, client=client)
    assert campaign.status(tmp_path)['status'] == 'stopped'
    assert not client.uploads and not (tmp_path/'campaign_plan.json').exists()


def test_title_abstract_only_xml_is_not_fulltext(tmp_path, small_cohort):
    for c in small_cohort:
        Path(c['fixture_path']).write_text('<article><front><article-meta><abstract><p>Detailed dose and safety.</p></abstract></article-meta></front></article>')
    client = BatchClient()
    with pytest.raises(ValueError, match='pool exhausted'):
        campaign.run(tmp_path, client=client)
    assert client.creates == 0


def test_changed_selection_receipt_blocks_resume(tmp_path, small_cohort):
    client = BatchClient()
    campaign.run(tmp_path, client=client)
    receipt = read_json(tmp_path/'selection_receipt.json')
    receipt['examined_count'] = 4
    write_json(tmp_path/'selection_receipt.json', receipt)
    with pytest.raises(ValueError, match='selection receipt changed'):
        campaign.run(tmp_path, client=client)
    assert client.creates == 3


def test_changed_availability_checkpoint_blocks_resume(tmp_path, small_cohort):
    client = BatchClient()
    campaign.run(tmp_path, client=client)
    checkpoint = tmp_path/'documents/PMID0.json'
    saved = read_json(checkpoint)
    saved['document']['fulltext_available'] = False
    write_json(checkpoint, saved)
    with pytest.raises(ValueError, match='checkpoint changed'):
        campaign.run(tmp_path, client=client)
    assert client.creates == 3


def test_rescreen_reuses_exact_fulltext_cohort_but_creates_fresh_paid_requests(tmp_path, small_cohort, monkeypatch):
    source_client = BatchClient()
    campaign.run(tmp_path, client=source_client)
    original_plan = (tmp_path/'campaign_plan.json').read_bytes()
    original_inputs = [(p, p.read_bytes()) for p in (tmp_path/'batches').glob('*/screening_requests.jsonl')]
    def no_network(*args, **kwargs):
        raise AssertionError('Rescreening must not discover or retrieve any new articles')
    monkeypatch.setattr(campaign, 'discover', no_network)
    monkeypatch.setattr(preparation, 'fetch_fulltext', no_network)
    target = tmp_path/'rescreen'
    client = BatchClient()
    result = campaign.run(target, source_workspace=tmp_path, client=client)
    assert result['submitted'] == 5 and client.creates == 3
    assert read_json(target/'candidates.json') == read_json(tmp_path/'candidates.json')
    assert read_json(target/'selection_receipt.json')['previous_decisions_reused'] is False
    assert (tmp_path/'campaign_plan.json').read_bytes() == original_plan
    assert all(p.read_bytes() == data for p, data in original_inputs)
    campaign.run(target, source_workspace=tmp_path, client=client)
    assert client.creates == 3
    summary = campaign.collect_results(target, client=client)
    assert summary['excluded'] == 5 and summary['technical_failures'] == 0


def test_rescreen_cannot_replace_source_workspace(tmp_path, small_cohort):
    campaign.run(tmp_path, client=BatchClient())
    client = BatchClient()
    with pytest.raises(ValueError, match='separate workspace'):
        campaign.run(tmp_path, source_workspace=tmp_path, client=client)
    assert client.creates == 0


def test_rescreen_detects_changed_source_cohort_before_payment(tmp_path, small_cohort):
    campaign.run(tmp_path, client=BatchClient())
    path = tmp_path/'candidates.json'
    selected = read_json(path)
    selected['candidates'].reverse()
    write_json(path, selected)
    client = BatchClient()
    with pytest.raises(ValueError, match='selection changed'):
        campaign.run(tmp_path/'rerun', source_workspace=tmp_path, client=client)
    assert client.creates == 0
