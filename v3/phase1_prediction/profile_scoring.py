"""Shared native scorer for article pairs, target ranking and configuration weights."""
from __future__ import annotations
from .profiles import query_profile
from .profile_similarity import compare_profiles, all_texts, norm, normalize_weights


def score_native_library(query,records,*,weights=None,embedding_backend='local',embedding_model='',embedding_cache=None):
    # Pretrial is enforced on the prediction path even if an imported profile
    # contains observed baseline or toxicity descriptions.
    target=query_profile(query); vectors=None; backend='local_token_cosine'
    if embedding_backend=='openai':
        from .reference.toxicity_similarity import embed_texts_openai, embedding_cache_lock, load_cache, save_cache, hash_text
        texts=all_texts([target,*[r['data'].get('_comparison_profile',{}) for r in records],
                         *[p for r in records for p in r['data'].get('_comparison_scopes',{}).values()]])
        vectors={}
        with embedding_cache_lock(embedding_cache):
            cache=load_cache(embedding_cache);items=cache.setdefault('items',{})
            keys={t:'profile:'+embedding_model+':'+hash_text(t) for t in texts}
            missing=[t for t in texts if keys[t] not in items]
            for start in range(0,len(missing),128):
                chunk=missing[start:start+128]
                for t,v in zip(chunk,embed_texts_openai(chunk,embedding_model)):
                    items[keys[t]]={'model':embedding_model,'text_hash':hash_text(t),'vector':v}
            if missing:save_cache(embedding_cache,cache)
            vectors={t:items[k]['vector'] for t,k in keys.items()}
        backend='openai'
    normalized=normalize_weights(weights);scored=[]
    for record in records:
        profile=record['data'].get('_comparison_profile') or query_profile(record['data'])
        result=compare_profiles(target,profile,weights=normalized,text_vectors=vectors)
        scope_scores={scope:compare_profiles(target,p,weights=normalized,text_vectors=vectors)
                      for scope,p in record['data'].get('_comparison_scopes',{}).items()}
        # One publication has one budget. Prefer its best comparable configuration;
        # additional configurations only divide that budget and cannot multiply it.
        eligible_scopes=[k for k,r in scope_scores.items() if r['similarity'] is not None and ('_allowed_scopes' not in query or k in query['_allowed_scopes'])]
        values=[scope_scores[k]['similarity'] for k in eligible_scopes]
        if values:
            best=max(sorted(eligible_scopes),key=lambda k:scope_scores[k]['similarity'] if scope_scores[k]['similarity'] is not None else -1.)
            result={**scope_scores[best],'article_profile_similarity':result['similarity'],'selected_scope':best}
        result['comparison_similarity']=result['similarity']
        # Legacy regression expects a scalar; missing is still explicit in
        # diagnostics, and a zero weight cannot enter fitting.
        result['similarity']=result['similarity'] if result['similarity'] is not None else 0.
        scored.append({**record,**result,'scope_scores':scope_scores,
                       **({'allowed_scopes':list(query['_allowed_scopes'])} if '_allowed_scopes' in query else {})})
    return sorted(scored,key=lambda r:(-r['similarity'],r['pmcid'])),backend


def segment_weight(article,segment,segments):
    scopes=article.get('scope_scores') or {}
    budget=max(0.,float(article.get('similarity') or 0))
    if not segments:return 0.
    if not scopes:return budget/len(segments)
    def score(s):
        r=scopes.get(s[0].get('arm_or_cohort'),{})
        return max(0.,float(r.get('similarity') or 0))
    total=sum(score(s) for s in segments)
    return budget*score(segment)/total if total else 0.
