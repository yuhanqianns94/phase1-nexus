"""Public v3 API. The calculation and selection contract is the v1 baseline."""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np
from .compact_adapter import target_query
from .models import PredictionResult
from .reference_bridge import build_workspace, evidence_contract, predict
from .reference.bayesian_toxicity import DEFAULT_PRIOR_CATALOG, DEFAULT_POSTERIOR_DRAW_COUNT
from .reference.prediction_service import preview_historical_articles
from .reference.oc_service import joint_target_posterior_draws
from .reference.toxicity_similarity import resolve_similarity_parameters
from .reference.toxicity_similarity import score_library
from .reference.toxicity_curve_common import parse_toxicity_query
from .reference.dlt_toxicity_regression import build_similarity_historical_lines
from .reference.power_likelihood_random_intercept_slope import (
    DEFAULT_INTERCEPT_SCALE_LATENT_MEAN, DEFAULT_INTERCEPT_SCALE_LATENT_SD,
    DEFAULT_SLOPE_SCALE_LATENT_MEAN, DEFAULT_SLOPE_SCALE_LATENT_SD,
)

from .reference.prediction_contract import PREDICTION_MODEL_VERSION


def _selection(library, query, workspace, threshold, top_articles, weights, include_series_keys, exclude_series_keys):
    if top_articles is not None and (isinstance(top_articles, bool) or not isinstance(top_articles, int) or top_articles <= 0):
        raise ValueError("top_articles must be a positive integer or None.")
    defaults, default_threshold, _ = resolve_similarity_parameters(use_manual_default=True)
    options = {"field_weights": dict(weights) if weights is not None else defaults,
               "similarity_threshold": default_threshold if threshold is None else threshold}
    included, excluded = set(include_series_keys), set(exclude_series_keys)
    if included & excluded:
        raise ValueError("Historical evidence cannot be both included and excluded.")
    known = {s.series_key for s in library.series}
    if (included | excluded) - known:
        raise ValueError("Unknown historical series identifier.")
    by_doc = {}
    for s in library.series:
        by_doc.setdefault(s.document_id, set()).add(s.series_key)
    if included or excluded:
        allowed=(included if included else known)-excluded
        query['_allowed_scopes']=[scope for r in workspace.records for scope,key in r['data'].get('_v3_scope_map',{}).items() if key in allowed]
    preview = preview_historical_articles(query, workspace, **options,
        candidate_basis="top_n" if top_articles is not None else "threshold", top_n=top_articles or 40)
    if included or excluded or top_articles is not None:
        candidate_ids = {r["document_id"] for r in preview["candidates"]}
        include_ids = {doc for doc, group in by_doc.items() if group & included}
        exclude_ids = {doc for doc, group in by_doc.items() if group <= excluded}
        if include_ids - candidate_ids:
            raise ValueError("Manual publications must belong to the v1 candidate pool; change threshold/top-N first.")
        options.update(historical_article_selection_mode="manual",
                       historical_article_candidate_basis="top_n" if top_articles is not None else "threshold",
                       historical_article_top_n=top_articles or 40,
                       historical_article_ids=sorted((include_ids or candidate_ids) - exclude_ids),
                       historical_article_candidate_snapshot_sha256=preview["candidate_snapshot_sha256"])
    return options, preview


def _scope_keys(workspace, lines):
    maps = {r["pmcid"]: r["data"].get("_v3_scope_map", {}) for r in workspace.records}
    return {maps.get(line["pmcid"], {}).get(line["arm_or_cohort"]) for line in lines} - {None}


def _rows(library, preview, selected_ids=None, eligible_keys=None):
    candidates = {r["document_id"]: r for r in preview["candidates"]}
    ids = set(candidates) if selected_ids is None else set(selected_ids)
    selected, excluded = [], []
    for s in library.series:
        if s.document_id not in ids or (eligible_keys is not None and s.series_key not in eligible_keys):
            excluded.append({"series_key": s.series_key, "document_id": s.document_id,
                             "reason": "reference_article_selection" if s.document_id not in ids else "no_eligible_reference_curve_line"})
            continue
        c = candidates.get(s.document_id, {})
        scope=next((k for k,v in c.get('configuration_keys',{}).items() if v==s.series_key),None)
        score=c.get('configuration_scores',{}).get(scope,c)
        selected.append({"series_key": s.series_key, "document_id": s.document_id, "study_id": s.study_id,
                         "part_id": s.part_id, "arm_id": s.arm_id, "series_id": s.series_id,
                         "window_id": s.window_id, "regimen_id": s.regimen_id,
                         "label": s.label, "intervention_name": s.intervention_name, "unit": s.unit,
                         "similarity": score.get("similarity", 0), "available_weight": score.get("available_weight", 0),
                         "field_scores": score.get("field_scores", {}), "point_count": len(s.points),
                         "patient_count": sum(p.n for p in s.points), "review_status": s.review_status,
                         "selection_basis": "v1_publication_selection", "source_points": [asdict(p) for p in s.points]})
    return selected, excluded


def preview_target_evidence(library, target, *, threshold=None, top_articles=None,
                            weights=None, include_series_keys=(), exclude_series_keys=()):
    query = target_query(target)
    workspace = build_workspace(library)
    options, preview = _selection(library, query, workspace, threshold, top_articles, weights, include_series_keys, exclude_series_keys)
    ids = options.get("historical_article_ids", [row["document_id"] for row in preview["candidates"]])
    scored, _ = score_library(query, workspace.records, embedding_backend="local", field_weights=options["field_weights"])
    lines, _ = build_similarity_historical_lines([r for r in scored if r["pmcid"] in ids], parse_toxicity_query(query),
        bayesian_cache=workspace.bayesian_cache, bayesian_prior=workspace.bayesian_prior)
    selected, excluded = _rows(library, preview, ids, _scope_keys(workspace, lines))
    return {"format": "phase1_v3_evidence_preview_v2", "target": asdict(target),
            **evidence_contract(library), "selected_series": selected, "excluded_series": excluded,
            "selected_article_count": len({r["document_id"] for r in selected}), "selected_series_count": len(selected),
            "selected_point_count": sum(r["point_count"] for r in selected), "reference_preview": preview}


def predict_target_curve(library, target, *, prior_name=None,
        prior_catalog_path=DEFAULT_PRIOR_CATALOG, threshold=None, top_articles=None, weights=None,
        draw_count=DEFAULT_POSTERIOR_DRAW_COUNT, seed=20260725, borrowing_strength=1.0, historical_ess_cap=30.0,
        random_intercept_scale_prior_mean=DEFAULT_INTERCEPT_SCALE_LATENT_MEAN,
        random_intercept_scale_prior_sd=DEFAULT_INTERCEPT_SCALE_LATENT_SD,
        random_slope_scale_prior_mean=DEFAULT_SLOPE_SCALE_LATENT_MEAN,
        random_slope_scale_prior_sd=DEFAULT_SLOPE_SCALE_LATENT_SD,
        include_series_keys=(), exclude_series_keys=()):
    include_series_keys, exclude_series_keys = tuple(include_series_keys), tuple(exclude_series_keys)
    query = target_query(target)
    workspace = build_workspace(library, prior_name=prior_name, prior_catalog_path=prior_catalog_path, draw_count=draw_count, seed=seed)
    options, preview = _selection(library, query, workspace, threshold, top_articles, weights, include_series_keys, exclude_series_keys)
    raw = predict(library, query, workspace, **options,
                  power_likelihood_borrowing_strength=borrowing_strength, power_likelihood_ess_cap=historical_ess_cap,
                  random_intercept_scale_prior_mean=random_intercept_scale_prior_mean,
                  random_intercept_scale_prior_sd=random_intercept_scale_prior_sd,
                  random_slope_scale_prior_mean=random_slope_scale_prior_mean,
                  random_slope_scale_prior_sd=random_slope_scale_prior_sd)
    selected, excluded = _rows(library, preview, options.get("historical_article_ids"),
        _scope_keys(workspace, raw["toxicity_curve"]["article_line_weighted"].get("historical_lines", [])))
    methods = {}
    for row in excluded:
        if row["series_key"] in set(exclude_series_keys):
            row["reason"] = "manually_excluded"
    doses = raw["query"]["planned_doses"]
    for key, method in raw["toxicity_curve"].items():
        summary = [{"dose": point["dose"], "posterior_mean": point.get("dlt_probability"),
                    "low_95": point.get("posterior_credible_interval_95", {}).get("low"),
                    "high_95": point.get("posterior_credible_interval_95", {}).get("high")}
                   for point in method.get("curve_points", [])]
        normalized = {**method, "doses": doses, "summary": summary, "posterior_draws": [], "intermediate_series_fits_persisted": False}
        if method.get("status") == "fitted" and raw["query"]["dose_grid"]["mode"] == "explicit_query_planned_doses" and len(doses) >= 2:
            sampled = joint_target_posterior_draws(raw, curve_method_key=key, requested_draw_count=draw_count, seed=seed)
            matrix = np.asarray(sampled["posterior_draws"])
            normalized.update(posterior_draws=matrix.tolist(), contributors=method.get("historical_lines", []))
        methods[key] = normalized
    contract = {**evidence_contract(library), "prior_name": workspace.bayesian_prior,
                "prior_catalog_fingerprint": workspace.bayesian_cache.get("prior_catalog_fingerprint_sha256"),
                "random_seed": seed, "posterior_draw_count": draw_count,
                "borrowing_strength": borrowing_strength, "historical_ess_cap": historical_ess_cap,
                "prediction_model_version": PREDICTION_MODEL_VERSION}
    identity = {"query": query, "options": options, "contract": contract,
                "reference_identity": raw["prediction_run_identity_sha256"]}
    run_id = "prediction_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return PredictionResult("phase1_v3_target_prediction_v3", PREDICTION_MODEL_VERSION, run_id,
        {**asdict(target), "dose_unit": raw["query"]["target_unit"], "planned_doses": doses,
         "similarity_threshold": options["similarity_threshold"], "top_articles": top_articles,
         "similarity_weights": raw["similarity_parameters"]["field_weights"]},
        contract, selected, excluded, methods, datetime.now(timezone.utc).isoformat(), raw)
