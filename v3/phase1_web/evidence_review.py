"""Target-specific relevance screening, separate from extraction accuracy review."""
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from phase1_prediction.multidrug.contracts import fingerprint
from phase1_prediction.multidrug.workflow import article_budgets
from phase1_extraction.artifacts import write_json


REVIEW_OPTIONS = ('Not reviewed', 'Include', 'Exclude', 'Uncertain')
REVIEW_LABELS = {'Not reviewed': 'Not reviewed', 'Include': 'Include',
                 'Exclude': 'Exclude', 'Uncertain': 'Unsure'}


def abstract_for(root):
    path=Path(root)/'document_ir.json'
    if not path.exists():return {'title':'','year':None,'main':[],'additional':[],'source':'Unavailable','content_id':'missing'}
    ir=json.loads(path.read_text());blocks=ir.get('blocks',[])
    title=next((b.get('text_raw','') for b in blocks if b.get('block_type')=='title'),'')
    result={'title':title,'year':None,'main':[],'additional':[],'source':'Document IR abstract (XML fallback)'}
    for m in ir.get('materials',[]):
        source=m.get('local_path') or m.get('path') or m.get('source_path')
        if not source or not str(source).lower().endswith('.xml') or not Path(source).exists():continue
        try:
            tree=ET.parse(source)
            for el in tree.iter():el.tag=el.tag.split('}')[-1]
            meta=tree.getroot().find('./front/article-meta')
            if meta is None:continue
            article_title=meta.find('./title-group/article-title')
            if article_title is not None:
                result['title']=' '.join(''.join(article_title.itertext()).split()) or title
            dates=meta.findall('pub-date')
            dates.sort(key=lambda date: {'epub':0,'ppub':1}.get(date.get('pub-type'),2))
            result['year']=next((date.findtext('year') for date in dates
                                 if re.fullmatch(r'\d{4}', date.findtext('year') or '')),None)
            for node in meta.findall('abstract'):
                kind=node.get('abstract-type','').lower()
                chunks=[]
                def visit(el):
                    if el.tag=='title':chunks.append({'heading':''.join(el.itertext()).strip()})
                    elif el.tag=='p':chunks.append({'text':' '.join(''.join(el.itertext()).split())})
                    else:
                        for child in el:visit(child)
                visit(node)
                if not chunks:chunks=[{'text':' '.join(''.join(node.itertext()).split())}]
                if kind in {'','main','summary'}:result['main'].extend(chunks)
                else:result['additional'].append({'type':kind,'sections':chunks})
            result['source']='Original JATS XML abstract'
            break
        except (ET.ParseError,OSError):continue
    if not result['main']:
        result['main']=[{'text':b.get('text_raw') or b.get('text_normalized','')} for b in blocks if b.get('block_type')=='abstract']
    # Publication metadata alone does not change a saved abstract decision.
    result['content_id']=fingerprint({k:v for k,v in result.items() if k!='year'})
    return result


def render_article_selection(st, rows, scopes, identity):
    """Select papers once and carry their matching scopes into every curve method."""
    docs=list(dict.fromkeys(row['document_id'] for row in rows))
    st.caption(f"{len(docs)} matching {'article' if len(docs)==1 else 'articles'}")
    if not docs:
        return []
    bydoc={scope.document_id:scope for scope in scopes}
    labels={}
    for doc in docs:
        scope=bydoc.get(doc)
        title=abstract_for(scope.lineage['root'])['title'] if scope else ''
        labels[doc]=f'{doc} · {title}' if title else doc
    key='multidrug_included_articles_'+identity[:12]
    if key not in st.session_state:
        previous=st.session_state.get('multidrug_review_state') or {}
        # Existing scope-level choices become whole-article choices on upgrade.
        old_ids=(previous.get('included') if previous.get('identity')==identity
                 else st.session_state.get('multidrug_included_'+identity[:12]))
        selected_docs=(set(docs) if old_ids is None else
                       {row['document_id'] for row in rows if row['series_key'] in old_ids})
        st.session_state[key]=[doc for doc in docs if doc in selected_docs]
    selected=st.multiselect('Articles to use',docs,format_func=labels.__getitem__,key=key)
    st.caption('Each article appears once. Its matching study data are linked automatically.')
    return [row['series_key'] for row in rows if row['document_id'] in selected]


def _render_abstract(st, abstract, doc, *, article_root=None, library_source='local', workspace=None):
    from phase1_web.article_source import render_original_article
    render_original_article(st,doc,article_root,library_source=library_source,workspace=workspace)
    with st.expander('Read the original abstract'):
        if not abstract['main']:
            st.info('An abstract is not available. Open the original article above to review it.')
        for section in abstract['main']:
            if 'heading' in section:
                st.markdown('**'+section['heading']+'**')
            else:
                st.write(section['text'])
        for other in abstract['additional']:
            st.caption('Additional abstract: '+other['type'])
            for section in other['sections']:
                st.write(section.get('heading') or section.get('text',''))


def power_prior_review_preview(rows, scopes, target, settings, decisions=None):
    """Use the prediction's exact scope selection and absolute-power rules.

    Preview mappings and scores are already calculated. This performs no new
    similarity calls or fitting, and preserves the pre-review borrowing cap.
    """
    from phase1_prediction.multidrug.power_prior import PowerPriorSpec, prepare_power_prior
    by_id = {scope.scope_id: scope for scope in scopes}
    if any(row['series_key'] not in by_id for row in rows):
        raise ValueError('The library changed. Preview the article candidates again.')
    scored = [(by_id[row['series_key']], row['mapping'],
               row.get('similarity') or 0.) for row in rows]
    spec = settings if isinstance(settings, PowerPriorSpec) else PowerPriorSpec.model_validate(settings)
    prepared = prepare_power_prior(scored, target, spec, decisions)
    table = [{'Article': item['document_id'], 'Selected scope': item['scope_id'],
              'Similarity': item['strength'],
              'Review multiplier': item['review_multiplier'],
              'Likelihood power': item['power'], 'Patients': item['patient_count'],
              'Discounted patients': item['power'] * item['patient_count']}
             for item in prepared['sources']]
    return {'rows': table, 'excluded': prepared['excluded'],
            'cap_factor': prepared['cap_factor'],
            'discounted_patient_count': prepared['discounted_patient_count'],
            'ready': bool(table) and any(item['review_multiplier'] > 0 for item in prepared['sources'])}


def render_review(st, rows, scopes, state_root, identity, review_mode, *, included_scope_ids=None,
                  power_prior_target=None, power_prior_settings=None, target_drug_names=None,
                  library_source='local', workspace=None):
    all_docs=list(dict.fromkeys(r['document_id'] for r in rows))
    bydoc={s.document_id:s for s in scopes}
    byscope={s.scope_id:s for s in scopes}
    if any(doc not in bydoc for doc in all_docs) or any(r['series_key'] not in byscope for r in rows):
        st.error('The library changed. Find matching articles again to refresh the review list.')
        return None,False
    abstracts={d:abstract_for(bydoc[d].lineage['root']) for d in all_docs}
    review_id=fingerprint([identity,{d:a['content_id'] for d,a in abstracts.items()}])
    path=Path(state_root)/'relevance_reviews'/(review_id+'.json')
    saved=json.loads(path.read_text()) if path.exists() else {}
    if included_scope_ids is not None:
        included=set(included_scope_ids)
        rows=[r for r in rows if r['series_key'] in included]
    docs=list(dict.fromkeys(r['document_id'] for r in rows))
    if not docs:
        if all_docs:
            st.info('Select at least one article above to continue.')
        return {} if review_mode else None,False
    decisions={d:saved.get('decisions',{}).get(d,'Not reviewed') for d in docs}
    decisions={d:value if value in REVIEW_OPTIONS else 'Not reviewed' for d,value in decisions.items()}
    if review_mode:
        st.caption('Read the original article, then decide whether to include it in your planned study.')
        batch=st.selectbox('Batch review action',['No batch action','Include all pending','Mark all pending uncertain'],
                           format_func=lambda value:'Mark all pending unsure' if value=='Mark all pending uncertain' else value,
                           key='review_batch_'+review_id)
        if st.button('Apply batch action',key='review_apply_'+review_id):
            for d in docs:
                if decisions[d]=='Not reviewed':
                    decisions[d]='Include' if batch=='Include all pending' else 'Uncertain' if batch=='Mark all pending uncertain' else 'Not reviewed'
                    st.session_state['relevance_'+review_id+d]=decisions[d]
        for number,doc in enumerate(docs,1):
            a=abstracts[doc]
            with st.container(border=True):
                st.subheader(f"{number}. {a['title'] or 'Article title unavailable'}")
                st.caption(f"{doc} · {a.get('year') or 'Year unavailable'}")
                _render_abstract(st,a,doc,article_root=bydoc[doc].lineage['root'],
                                 library_source=library_source,workspace=workspace)
                decisions[doc]=st.radio('Relevance decision',REVIEW_OPTIONS,index=REVIEW_OPTIONS.index(decisions[doc]),
                                       format_func=REVIEW_LABELS.__getitem__,key='relevance_'+review_id+doc,horizontal=True,
                                       help='Include keeps the article; Exclude removes it; Unsure applies a 50% review weight before normalization.')
        # Article selection does not discard earlier decisions for deselected papers.
        stored_decisions={**saved.get('decisions',{}),**decisions}
        write_json(path,{'kind':'target_relevance_only','review_id':review_id,'decisions':stored_decisions,'abstract_versions':{d:a['content_id'] for d,a in abstracts.items()}})
        if any(v=='Not reviewed' for v in decisions.values()):
            st.info('Review every selected article or use an explicit batch action before prediction.')
            return decisions,False
    else:decisions=None
    if review_mode and power_prior_target is None:
        budgets=article_budgets(rows,decisions)
        ready=bool(budgets) and any(item['normalized_weight']>0 for item in budgets)
        if ready:
            st.success('Article review saved. Continue to risk curves when you are ready.')
        elif all(value=='Exclude' for value in decisions.values()):
            st.info('All selected articles are excluded. Include an article or mark one Unsure to continue.')
        else:
            st.info('Your decisions are saved. No selected article has usable matching evidence for this plan.')
        return decisions,ready
    if power_prior_settings is not None and power_prior_target is None:
        # Article screening needs no dose coordinates. Exact likelihood powers
        # are checked later, once the target has computational dose references.
        st.info('Article review is saved. Method 3 contribution checks will appear after you choose curve coordinates and dose references.')
        budgets=article_budgets(rows,decisions)
        return decisions,bool(budgets) and any(item['adjusted_weight']>0 for item in budgets)
    if power_prior_settings is not None:
        try:
            result = power_prior_review_preview(rows, scopes, power_prior_target, power_prior_settings, decisions)
        except ValueError as error:
            st.error(str(error))
            return decisions, False
        if review_mode:
            return decisions,result['ready']
        st.dataframe(result['rows'], hide_index=True)
        st.caption('Method 3 uses the absolute likelihood powers shown above, not normalized article weights. '
                   'One best eligible complete scope is selected per article. The patient-budget cap is fixed '
                   'before review: Uncertain halves that article\'s contribution, and Exclude removes it without '
                   'redistributing its share. Marking all articles uncertain halves the discounted patient count; '
                   'it does not prescribe how much posterior uncertainty changes.')
        st.caption(f"Patient-budget cap factor: {result['cap_factor']:.4g}. "
                   f"Discounted patient count: {result['discounted_patient_count']:.4g} "
                   '(a borrowing budget, not an exact Bayesian effective sample size).')
        if result['excluded']:
            with st.expander('Scopes not used by Method 3'):
                st.dataframe(result['excluded'], hide_index=True)
        if not result['rows']:
            st.info('No candidate scope provides an eligible complete regimen for Method 3.')
        elif result['ready'] and result['discounted_patient_count'] == 0:
            st.info('Historical borrowing strength is zero. Prediction will use the specified priors only.')
        return decisions, result['ready']
    budgets=article_budgets(rows,decisions)
    st.dataframe([{'Article':item['document_id'],
                   'Decision':REVIEW_LABELS.get(item['decision'],item['decision']),
                   'Review multiplier':item['review_multiplier'],
                   'Candidate share':item['normalized_weight']}
                  for item in budgets],hide_index=True,use_container_width=True)
    st.caption('Base weight = similarity of the best scope. The review multiplier is applied once per article, then weights are normalized. Marking every article uncertain does not halve patient counts or uncertainty.')
    st.caption('These are screening shares. Final contributions also depend on whether a study group is usable for the chosen curve method.')
    with st.expander('Detailed evidence weights'):
        st.dataframe(budgets,hide_index=True,use_container_width=True)
    return decisions, bool(budgets) and any(r['normalized_weight']>0 for r in budgets)
