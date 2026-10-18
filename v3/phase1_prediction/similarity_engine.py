"""One engine context for correspondence, ranking, rescoring and article pairs."""
from contextlib import contextmanager
from pathlib import Path
from .profile_similarity import _TEXT_VECTORS, all_texts, norm, embedding_vector

NORMALIZATION_VERSION='profile_text_v1'
DEFAULT_MODEL='text-embedding-3-small'
DEFAULT_CACHE=Path(__file__).resolve().parents[1]/'output'/'profile_embeddings.json'


class EmbeddingVectors(dict):
    """Resolve only missing normalized text, with a persistent versioned cache."""
    def __init__(self, model, path):
        super().__init__();self.model=model;self.path=Path(path);self.dimension=None
    def ensure(self, texts):
        from .reference.toxicity_similarity import embed_texts_openai, embedding_cache_lock, load_cache, save_cache, hash_text
        texts=sorted({norm(t) for t in texts if norm(t)}-set(self))
        if not texts:return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with embedding_cache_lock(self.path):
            cache=load_cache(self.path);items=cache.setdefault('items',{})
            keys={t:f'{NORMALIZATION_VERSION}:{self.model}:{hash_text(t)}' for t in texts}
            missing=[t for t in texts if keys[t] not in items]
            for start in range(0,len(missing),128):
                chunk=missing[start:start+128]
                try:vectors=embed_texts_openai(chunk,self.model)
                except Exception as error:raise RuntimeError('OpenAI embedding comparison failed; no local fallback was used. '+str(error)) from error
                if len(vectors)!=len(chunk):raise ValueError('Embedding service returned an incomplete batch.')
                for t,v in zip(chunk,vectors):items[keys[t]]={'model':self.model,'text_hash':hash_text(t),'vector':v}
            resolved={}
            dimension=self.dimension
            for t in texts:
                try:vector=embedding_vector(items[keys[t]]['vector'])
                except (KeyError,TypeError) as error:raise ValueError('Embedding cache entry is invalid; no local fallback was used.') from error
                if dimension is not None and vector.size!=dimension:
                    raise ValueError('Embedding dimensions do not match; no local fallback was used.')
                dimension=vector.size
                resolved[t]=vector.tolist()
            if missing:save_cache(self.path,cache)
            self.dimension=dimension
            self.update(resolved)


@contextmanager
def similarity_context(engine='local', model=DEFAULT_MODEL, cache=DEFAULT_CACHE, profiles=()):
    if engine not in {'local','openai'}:raise ValueError('Unknown similarity engine.')
    import os
    if engine == 'openai' and os.environ.get('PHASE1_DEPLOYMENT', 'local').lower() == 'public':
        raise ValueError('The public website uses local similarity comparison. No site-owner API is available.')
    vectors=None
    if engine=='openai':
        vectors=EmbeddingVectors(model,cache)
        vectors.ensure(all_texts(profiles))
    token=_TEXT_VECTORS.set(vectors)
    try:yield
    finally:_TEXT_VECTORS.reset(token)


def with_similarity(fn):
    from functools import wraps
    @wraps(fn)
    def wrapped(*args, similarity_engine='local', embedding_model=DEFAULT_MODEL, embedding_cache=DEFAULT_CACHE, **kwargs):
        from phase1_prediction.multidrug.contracts import fingerprint
        with similarity_context(similarity_engine,embedding_model,embedding_cache):
            result=fn(*args,**kwargs)
        spec={'engine':similarity_engine,'model':embedding_model if similarity_engine=='openai' else None,'normalization_version':NORMALIZATION_VERSION}
        if isinstance(result,dict):result['similarity_engine']=spec
        else:
            from dataclasses import replace
            result=replace(result,evidence_contract={**result.evidence_contract,'similarity_engine':spec},run_id='prediction_'+fingerprint([result.run_id,spec])[:24])
        return result
    return wrapped
