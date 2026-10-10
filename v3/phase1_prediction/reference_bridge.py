"""v3 evidence provenance around the unchanged v1 downstream engine."""
from pathlib import Path
import json
from .compact_adapter import library_records
from .reference.bayesian_curve_cache import build_bayesian_curve_cache
from .reference.bayesian_toxicity import DEFAULT_POSTERIOR_DRAW_COUNT, DEFAULT_PRIOR_CATALOG
from .reference.prediction_service import PredictionWorkspace, predict_target_curve


def build_workspace(library, *, prior_name=None, prior_catalog_path=DEFAULT_PRIOR_CATALOG,
                    draw_count=DEFAULT_POSTERIOR_DRAW_COUNT, seed=20260725):
    prior_name = prior_name or json.loads(Path(prior_catalog_path).read_text())["default_prior"]
    records = library_records(library)
    if not records:
        raise ValueError("No usable resolved v3 evidence is available.")
    cache = build_bayesian_curve_cache(records, prior_catalog_path=Path(prior_catalog_path),
                                       prior_names=[prior_name], posterior_draw_count=draw_count, random_seed=seed)
    return PredictionWorkspace(records, cache, prior_name, Path("<v3-in-memory>"),
                               Path("<v3-resolved>"), "v3-resolved")


def evidence_contract(library):
    return {"schema": "phase1_extraction_v3", "policy_version": "v3_configuration_adapter_v2", "similarity_version": "cohort_profile_similarity_v1",
            "evidence_policy": library.evidence_policy, "review_policy": library.review_policy,
            "library_fingerprint": library.fingerprint, "library_document_count": library.document_count,
            "library_series_count": len(library.series), "source_roots": list(library.source_roots),
            "duplicate_documents": library.duplicate_documents,
            "per_article_curve_cache_read": False, "per_article_curve_cache_written": False,
            "downstream_baseline": "v1-working-tree-20260908"}


def predict(library, query, workspace, **options):
    if options.get("power_likelihood_enabled") or options.get("power_likelihood_random_effects_enabled"):
        raise ValueError("Only the three approved methods are available in v3.")
    result = predict_target_curve(query, workspace, **options)
    result["v3_evidence_contract"] = evidence_contract(library)
    return result
