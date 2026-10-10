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


def _percentage(value):
    return 'Not available' if value is None else f'{value:.0%}'


def _mapping_rows(row, scope, target_drug_names=None):
    """Describe existing mappings; never choose a new correspondence in the UI."""
    target_drug_names=target_drug_names or {}
    mapping=row.get('mapping') or {}
    source_names={drug.model_drug_id:drug.name for drug in scope.drugs}
    details={item['target_id']:item for item in mapping.get('details',[])}
    result=[]
    for target_id,source_id in mapping.get('pairs',{}).items():
        detail=details.get(target_id,{})
        identity=next((item for item in detail.get('comparison',{}).get('details',[])
                       if item.get('indicator')=='Identity'),{})
        result.append({'Your drug':target_drug_names.get(target_id) or identity.get('left') or target_id,
                       'Article drug':source_names.get(source_id) or identity.get('right') or source_id,
                       'Match': 'Chosen by you' if detail.get('role_rule')=='manual_override' else 'Automatic'})
    for target_id in mapping.get('unmatched_target_ids',[]):
        result.append({'Your drug':target_drug_names.get(target_id,target_id),'Article drug':'No suitable match','Match':'Unmatched'})
    return result


def _render_abstract(st, abstract, doc):
    with st.expander('Read the original abstract'):
        if re.fullmatch(r'PMC\d+', doc):
            st.markdown(f'[Open full article on PMC](https://pmc.ncbi.nlm.nih.gov/articles/{doc}/)')
        st.caption(abstract['source'])
        if not abstract['main']:
            st.info('No abstract is available in the saved source files. Check the original article before deciding.')
        for section in abstract['main']:
            if 'heading' in section:
                st.markdown('**'+section['heading']+'**')
            else:
                st.write(section['text'])
        for other in abstract['additional']:
            st.caption('Additional abstract: '+other['type'])
            for section in other['sections']:
                st.write(section.get('heading') or section.get('text',''))


def _render_comparison(st, rows, scopes, target_drug_names=None):
    with st.expander('See matching and similarity details'):
        for row in rows:
            scope=scopes[row['series_key']]
            st.markdown('**'+str(row.get('label') or scope.label or 'Study group')+'**')
            matches=_mapping_rows(row,scope,target_drug_names)
            if matches:st.dataframe(matches,hide_index=True,use_container_width=True)
            score=(row.get('mapping') or {}).get('score') or {}
            groups=score.get('groups') or {}
            from phase1_prediction.profile_similarity import LABELS
            table=[{'Comparison':LABELS.get(key,key),'Similarity':_percentage(value.get('score'))}
                   for key,value in groups.items()]
            if table:
                st.dataframe(table,hide_index=True,use_container_width=True)
                st.caption('Similarity compares the available clinical facts. Missing or unmatched fields are not assigned a similarity score.')
            if row.get('prior_only'):
                st.caption('No usable DLT observations in this group; matching alone does not make it usable for borrowing.')
            elif 'point_count' in row:
                st.caption(f"{row['point_count']} usable dose–DLT observations in this group.")
            st.caption('Detailed comparison records')
            st.json(score,expanded=False)


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
                  power_prior_target=None, power_prior_settings=None, target_drug_names=None):
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
    decisions={d:saved.get('decisions',{}).get(d,'Not reviewed') for d in docs}
    decisions={d:value if value in REVIEW_OPTIONS else 'Not reviewed' for d,value in decisions.items()}
    if review_mode:
        st.caption('Choose which articles to use for your planned study. Include keeps the original contribution; '
                   'Exclude removes it; Unsure halves its contribution before any method-specific normalization. '
                   'Not reviewed is a separate pending state.')
        st.caption('These decisions affect borrowing for this target only. Extraction accuracy assessments never enter prediction. Exclusions do not backfill the candidate pool.')
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
            article_rows=[row for row in rows if row['document_id']==doc]
            best=max(article_rows,key=lambda row:row.get('similarity') or 0.)
            drugs=list(dict.fromkeys(drug.name for row in article_rows for drug in byscope[row['series_key']].drugs))
            with st.container(border=True):
                st.subheader(f"{number}. {a['title'] or 'Article title unavailable'}")
                st.caption(f"{doc} · {a.get('year') or 'Year unavailable'} · {len(article_rows)} study group(s)")
                st.write('**Drugs studied:** '+', '.join(drugs))
                st.caption(f"Similarity: {_percentage(best.get('similarity'))} "
                           '(best-ranked study group; these are matching scores, not toxicity probabilities).')
                matches=_mapping_rows(best,byscope[best['series_key']],target_drug_names)
                if matches:
                    st.write('**Drug correspondence:** '+'; '.join(
                        f"{item['Your drug']} → {item['Article drug']}" for item in matches))
                if len(article_rows)>1:
                    st.caption('The summary shows the best-ranked group. Other groups may have different matches; see the details below.')
                decisions[doc]=st.radio('Relevance decision',REVIEW_OPTIONS,index=REVIEW_OPTIONS.index(decisions[doc]),
                                       format_func=REVIEW_LABELS.__getitem__,key='relevance_'+review_id+doc,horizontal=True)
                _render_abstract(st,a,doc)
                _render_comparison(st,article_rows,byscope,target_drug_names)
        # Scope toggles do not discard earlier decisions for deselected articles.
        stored_decisions={**saved.get('decisions',{}),**decisions}
        write_json(path,{'kind':'target_relevance_only','review_id':review_id,'decisions':stored_decisions,'abstract_versions':{d:a['content_id'] for d,a in abstracts.items()}})
        if any(v=='Not reviewed' for v in decisions.values()):
            st.info('Review every candidate article or use an explicit batch action before prediction.')
            return decisions,False
    else:decisions=None
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
