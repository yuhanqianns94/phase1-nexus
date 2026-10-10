"""Compatibility facade over v1 publication-level similarity, not a second scorer."""
from collections import Counter
from .compact_adapter import library_records, target_query
from .models import LibrarySnapshot, ScoredSeries
from .reference.toxicity_similarity import FIELD_WEIGHTS, normalize_field_weights, resolve_similarity_parameters, score_library


def normalize_weights(weights=None):
    if weights is None:
        return resolve_similarity_parameters(use_manual_default=True)[0]
    return normalize_field_weights(weights)


def score_series(target, series, *, weights=None):
    """A single-publication convenience call; use score_publications for full-library IDF."""
    return score_publications(target, (series,), weights=weights)[0]


def score_publications(target, series, *, weights=None):
    library = LibrarySnapshot("project_default", "extracted", (), tuple(series), len({s.document_id for s in series}))
    scored, _ = score_library(target_query(target), library_records(library), embedding_backend="local", field_weights=normalize_weights(weights))
    by_id = {row["pmcid"]: row for row in scored}
    result=[]
    for s in series:
        article=by_id[s.document_id]
        scope=next((k for k,v in article['data'].get('_v3_scope_map',{}).items() if v==s.series_key),None)
        score=article.get('scope_scores',{}).get(scope,article)
        result.append(ScoredSeries(s,score['similarity'] or 0.,score['available_weight'],score['field_scores']))
    return result


def select_series(target, series, *, threshold=None, top_articles=None, weights=None, include_series_keys=(), exclude_series_keys=()):
    from .service import preview_target_evidence
    library = LibrarySnapshot("project_default", "extracted", (), tuple(series), len({s.document_id for s in series}))
    preview = preview_target_evidence(library, target, threshold=threshold, top_articles=top_articles, weights=weights,
                                      include_series_keys=include_series_keys, exclude_series_keys=exclude_series_keys)
    ids = {r["series_key"] for r in preview["selected_series"]}
    return [s for s in score_publications(target, series, weights=weights) if s.series.series_key in ids], preview["excluded_series"]


def allocated_series_weights(scored):
    """Equal segment split per publication, before reference reliability/support adjustments."""
    scores={}
    for row in scored:scores.setdefault(row.series.document_id,[]).append(max(0.,row.similarity))
    raw={row.series.series_key:(max(scores[row.series.document_id])*max(0.,row.similarity)/sum(scores[row.series.document_id]) if sum(scores[row.series.document_id]) else 0.) for row in scored}
    total=sum(raw.values())
    return {key:value/total for key,value in raw.items()} if total else {}
