"""Readable publication-pair comparisons with symmetric similarity matrices."""
import json
import pandas as pd
from phase1_prediction.profile_similarity import LABELS, compare_profiles, normalize_weights


def detail_rows(node,path=''):
    result=[]
    if 'details' in node:
        for d in node['details']:result.extend(detail_rows(d,' / '.join(filter(None,[path,d.get('indicator')]))))
    else:
        # Source values can mix prose and numbers in the same display column.
        # Keep scores numeric, but render source facts consistently as text.
        def show(v):
            if v is None:return None
            return json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else str(v)
        result.append({'Indicator':path,'Article A':show(node.get('left')),'Article B':show(node.get('right')),
                       'Similarity':node.get('score'),'Note':node.get('note') or ('Not comparable / not reported' if node.get('score') is None else '')})
    return result


def _render(st,library):
    from phase1_web.hosted import is_public
    public = is_public()
    st.title('Article-to-article similarity')
    st.caption('Seven indicator groups. Design and dose magnitudes do not score. Each drug, condition and treatment phase is aligned once. Missing information is not assigned a similarity score.')
    mode=st.radio('Comparison scope',['pretrial','full'],format_func=lambda x:'Pre-trial information only' if x=='pretrial' else 'Full article (includes observed baseline and toxicity descriptions)',horizontal=True)
    records={r['pmcid']:r['data']['_comparison_profile'] for r in library.matching_records if '_comparison_profile' in r['data']}
    ids=sorted(records)
    if not ids:st.info('No structured profiles available.');return
    from phase1_web.reference.helpers import load_default_similarity_parameters
    weights,_=load_default_similarity_parameters();weights=normalize_weights(weights)
    with st.expander('Comparison weights'):
        cols=st.columns(3)
        for i,k in enumerate(LABELS):
            with cols[i%3]:weights[k]=st.number_input(LABELS[k],min_value=0.,max_value=1.,value=weights[k],step=.05,key='pairweight_'+k)
    if sum(weights.values())<=0:st.error('Set at least one positive weight.');return
    selected=st.multiselect('Articles in matrix',ids,default=ids[:min(20,len(ids))],
                           **({'max_selections':20} if public else {}))
    if any(article_id not in records for article_id in selected):
        st.error('The selected articles are unavailable in this library. Select the articles again.')
        return
    if public and len(selected)>20:
        st.error('Select at most 20 articles for a comparison matrix on the public website.')
        return
    matrix=pd.DataFrame(index=selected,columns=selected,dtype=float)
    for i,a in enumerate(selected):
        for b in selected[i:]:
            r=compare_profiles(records[a],records[b],mode=mode,weights=weights)
            matrix.loc[a,b]=matrix.loc[b,a]=r['similarity']
    st.write('Similarity (0–1; blank = insufficient comparable information)')
    st.dataframe(matrix, column_config={k:st.column_config.NumberColumn(k,format='%.3f') for k in selected},use_container_width=True)
    if not selected:return
    cols=st.columns(2)
    with cols[0]:a=st.selectbox('Article A',selected)
    with cols[1]:b=st.selectbox('Article B',selected,index=min(1,len(selected)-1))
    r=compare_profiles(records[a],records[b],mode=mode,weights=weights)
    st.dataframe([{'Category':LABELS[k],'Similarity':v['score'],'Weight':r['weights'][k]} for k,v in r['groups'].items()],hide_index=True,use_container_width=True,column_config={'Similarity':st.column_config.NumberColumn(format='%.3f'),'Weight':st.column_config.NumberColumn(format='%.3f')})
    for k,v in r['groups'].items():
        with st.expander(LABELS[k]):
            st.dataframe(detail_rows(v),hide_index=True,use_container_width=True)
    with st.expander('Drug and configuration correspondence'):st.json(r['pairing'])
    st.download_button('Download comparison details',json.dumps({'article_a':a,'article_b':b,**r},ensure_ascii=False,indent=2),file_name=f'{a}_{b}_comparison.json',mime='application/json')


def render(st,library):
    from phase1_prediction.similarity_engine import similarity_context
    from phase1_web.hosted import is_public
    engines = ['local'] if is_public() else ['local','openai']
    preferred = st.session_state.get('similarity_engine_preference','local')
    engine=st.selectbox('Similarity engine',engines,format_func=lambda x:'Local structured + text comparison' if x=='local' else 'OpenAI embeddings + structured comparison',index=engines.index(preferred) if preferred in engines else 0,key='pair_similarity_engine')
    st.session_state['similarity_engine_preference']=engine
    model=st.selectbox('Embedding model',['text-embedding-3-small','text-embedding-3-large'],key='pair_embedding_model') if engine=='openai' else 'text-embedding-3-small'
    if engine=='openai':
        st.caption('Uncached normalized clinical text is sent to the OpenAI embedding API. Numeric rules remain structured. API failure does not switch engines.')
        if not st.checkbox('Enable OpenAI comparison for this page',key='pair_enable_openai'):return
    try:
        with similarity_context(engine,model):_render(st,library)
    except Exception as error:st.error('Similarity comparison stopped: '+str(error))
