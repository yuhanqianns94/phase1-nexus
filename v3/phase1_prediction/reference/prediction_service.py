"""Programmatic prediction workspace and arbitrary-dose probability service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bayesian_curve_cache import (
    DEFAULT_BAYESIAN_CURVE_CACHE,
    build_bayesian_curve_cache,
    cache_metadata_for_result,
    load_bayesian_curve_cache,
    write_bayesian_curve_cache,
)
from .bayesian_toxicity import DEFAULT_POSTERIOR_DRAW_COUNT, DEFAULT_PRIOR_CATALOG
from .estimate_toxicity_curve import estimate_toxicity_curve
from .historical_article_selection import (
    AUTOMATIC_SELECTION_MODE,
    DEFAULT_HISTORICAL_ARTICLE_TOP_N,
    MANUAL_SELECTION_MODE,
    THRESHOLD_CANDIDATE_BASIS,
    apply_historical_article_selection,
    preview_historical_article_selection,
    retain_curve_eligible_preview_candidates,
)
from .dlt_toxicity_regression import build_similarity_historical_lines
from .similarity_weighted_curve import posterior_mixture_summary_at_dose
from .power_likelihood_curve import (
    DEFAULT_BORROWING_STRENGTH,
    DEFAULT_HISTORICAL_ESS_CAP,
    power_likelihood_probability_summary,
)
from .power_likelihood_random_effects import random_effects_probability_summary
from .power_likelihood_random_intercept_slope import (
    DEFAULT_INTERCEPT_SCALE_LATENT_MEAN,
    DEFAULT_INTERCEPT_SCALE_LATENT_SD,
    DEFAULT_SLOPE_SCALE_LATENT_MEAN,
    DEFAULT_SLOPE_SCALE_LATENT_SD,
    random_intercept_slope_probability_summary,
)
from .posterior_curve_source_mixture import source_mixture_probability_summary
from .toxicity_similarity import (
    DEFAULT_EMBEDDING_CACHE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXTRACTION_DIR,
    canonical_dose_value,
    load_extractions,
    normalize_field_weights,
    pretrial_query_only,
    resolve_similarity_parameters,
    score_library,
)
from .toxicity_curve_common import parse_toxicity_query
from .prediction_contract import ensure_current_prediction


@dataclass(frozen=True)
class PredictionWorkspace:
    """A validated historical extraction library and compatible posterior cache."""

    records: list[dict[str, Any]]
    bayesian_cache: dict[str, Any]
    bayesian_prior: str
    bayesian_cache_path: Path
    extraction_dir: Path
    pattern: str


def load_prediction_workspace(
    *,
    extraction_dir: Path = DEFAULT_EXTRACTION_DIR,
    pattern: str = "*_compact.json",
    bayesian_cache_path: Path = DEFAULT_BAYESIAN_CURVE_CACHE,
    bayesian_prior: str,
    bayesian_prior_catalog_path: Path = DEFAULT_PRIOR_CATALOG,
) -> PredictionWorkspace:
    """Load records and only a fingerprint-compatible cache for one prior."""

    records = load_extractions(extraction_dir, pattern)
    if not records:
        raise ValueError(f"No extraction records found in {extraction_dir}")
    cache = load_bayesian_curve_cache(
        bayesian_cache_path,
        records=records,
        prior_name=bayesian_prior,
        prior_catalog_path=bayesian_prior_catalog_path,
    )
    return PredictionWorkspace(
        records=records,
        bayesian_cache=cache,
        bayesian_prior=bayesian_prior,
        bayesian_cache_path=bayesian_cache_path,
        extraction_dir=extraction_dir,
        pattern=pattern,
    )


def rebuild_bayesian_cache_for_library(
    *,
    extraction_dir: Path,
    bayesian_cache_path: Path,
    bayesian_prior_catalog_path: Path = DEFAULT_PRIOR_CATALOG,
    pattern: str = "*_compact.json",
    posterior_draw_count: int = DEFAULT_POSTERIOR_DRAW_COUNT,
    random_seed: int = 20260725,
) -> dict[str, Any]:
    """Explicitly rebuild a cache after a reviewed library promotion."""

    records = load_extractions(extraction_dir, pattern)
    if not records:
        raise ValueError(f"No extraction records found in {extraction_dir}")
    cache = build_bayesian_curve_cache(
        records,
        prior_catalog_path=bayesian_prior_catalog_path,
        posterior_draw_count=posterior_draw_count,
        random_seed=random_seed,
    )
    write_bayesian_curve_cache(bayesian_cache_path, cache)
    return cache_metadata_for_result(
        cache,
        prior_name="all",
        path=bayesian_cache_path,
    )


def predict_target_curve(
    raw_query: dict[str, Any],
    workspace: PredictionWorkspace,
    *,
    embedding_backend: str = "local",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: Path = DEFAULT_EMBEDDING_CACHE,
    top_k: int = 10,
    auto_grid_points: int = 101,
    similarity_parameters_path: Path | None = None,
    use_manual_similarity_parameters: bool = True,
    similarity_threshold_override: float | None = None,
    field_weights: dict[str, float] | None = None,
    similarity_threshold: float | None = None,
    power_likelihood_enabled: bool = False,
    power_likelihood_random_effects_enabled: bool = False,
    power_likelihood_random_intercept_slope_enabled: bool = True,
    power_likelihood_borrowing_strength: float = DEFAULT_BORROWING_STRENGTH,
    power_likelihood_ess_cap: float = DEFAULT_HISTORICAL_ESS_CAP,
    random_intercept_scale_prior_mean: float = DEFAULT_INTERCEPT_SCALE_LATENT_MEAN,
    random_intercept_scale_prior_sd: float = DEFAULT_INTERCEPT_SCALE_LATENT_SD,
    random_slope_scale_prior_mean: float = DEFAULT_SLOPE_SCALE_LATENT_MEAN,
    random_slope_scale_prior_sd: float = DEFAULT_SLOPE_SCALE_LATENT_SD,
    historical_article_selection_mode: str = AUTOMATIC_SELECTION_MODE,
    historical_article_candidate_basis: str = THRESHOLD_CANDIDATE_BASIS,
    historical_article_top_n: int = DEFAULT_HISTORICAL_ARTICLE_TOP_N,
    historical_article_ids: list[str] | tuple[str, ...] | None = None,
    historical_article_candidate_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Estimate a curve from a user query without CLI file-output side effects."""

    query, ignored_fields = pretrial_query_only(raw_query)
    if field_weights is not None:
        resolved_weights = field_weights
        resolved_threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else (
                float(similarity_threshold_override)
                if similarity_threshold_override is not None
                else 0.0
            )
        )
        source = "ui_session_weights"
    else:
        resolved_weights, resolved_threshold, source = resolve_similarity_parameters(
            path=similarity_parameters_path,
            use_manual_default=use_manual_similarity_parameters,
            threshold_override=(
                similarity_threshold
                if similarity_threshold is not None
                else similarity_threshold_override
            ),
        )
    return estimate_toxicity_curve(
        query,
        workspace.records,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_cache=embedding_cache,
        top_k=top_k,
        ignored_query_fields=ignored_fields,
        auto_grid_points=auto_grid_points,
        field_weights=resolved_weights,
        similarity_threshold=resolved_threshold,
        similarity_parameter_source=source,
        bayesian_cache=workspace.bayesian_cache,
        bayesian_prior=workspace.bayesian_prior,
        bayesian_cache_metadata=cache_metadata_for_result(
            workspace.bayesian_cache,
            workspace.bayesian_prior,
            workspace.bayesian_cache_path,
        ),
        power_likelihood_enabled=power_likelihood_enabled,
        power_likelihood_random_effects_enabled=(
            power_likelihood_random_effects_enabled
        ),
        power_likelihood_random_intercept_slope_enabled=(
            power_likelihood_random_intercept_slope_enabled
        ),
        power_likelihood_borrowing_strength=power_likelihood_borrowing_strength,
        power_likelihood_ess_cap=power_likelihood_ess_cap,
        random_intercept_scale_prior_mean=random_intercept_scale_prior_mean,
        random_intercept_scale_prior_sd=random_intercept_scale_prior_sd,
        random_slope_scale_prior_mean=random_slope_scale_prior_mean,
        random_slope_scale_prior_sd=random_slope_scale_prior_sd,
        historical_article_selection_mode=historical_article_selection_mode,
        historical_article_candidate_basis=historical_article_candidate_basis,
        historical_article_top_n=historical_article_top_n,
        historical_article_ids=historical_article_ids,
        historical_article_candidate_snapshot_sha256=(
            historical_article_candidate_snapshot_sha256
        ),
    )


def preview_historical_articles(
    raw_query: dict[str, Any],
    workspace: PredictionWorkspace,
    *,
    embedding_backend: str = "local",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: Path = DEFAULT_EMBEDDING_CACHE,
    similarity_parameters_path: Path | None = None,
    use_manual_similarity_parameters: bool = True,
    similarity_threshold_override: float | None = None,
    field_weights: dict[str, float] | None = None,
    similarity_threshold: float | None = None,
    candidate_basis: str = THRESHOLD_CANDIDATE_BASIS,
    top_n: int = DEFAULT_HISTORICAL_ARTICLE_TOP_N,
) -> dict[str, Any]:
    """Score and rank review candidates without fitting any toxicity curve."""

    query, ignored_fields = pretrial_query_only(raw_query)
    if field_weights is not None:
        resolved_weights = normalize_field_weights(field_weights)
        resolved_threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else (
                float(similarity_threshold_override)
                if similarity_threshold_override is not None
                else 0.0
            )
        )
        source = "ui_session_weights"
    else:
        resolved_weights, resolved_threshold, source = resolve_similarity_parameters(
            path=similarity_parameters_path,
            use_manual_default=use_manual_similarity_parameters,
            threshold_override=(
                similarity_threshold
                if similarity_threshold is not None
                else similarity_threshold_override
            ),
        )
    if any("_comparison_profile" in r["data"] for r in workspace.records):
        from ..profile_similarity import normalize_weights as profile_weights
        resolved_weights = profile_weights(resolved_weights)
    scored, backend_used = score_library(
        query,
        workspace.records,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_cache=embedding_cache,
        field_weights=resolved_weights,
    )
    preview = preview_historical_article_selection(
        scored,
        similarity_threshold=resolved_threshold,
        candidate_basis=candidate_basis,
        top_n=top_n,
    )
    candidate_ids = [
        str(candidate["document_id"])
        for candidate in preview.get("candidates", [])
    ]
    if candidate_ids:
        preview_rows, _selection_audit = apply_historical_article_selection(
            scored,
            similarity_threshold=resolved_threshold,
            selection_mode=MANUAL_SELECTION_MODE,
            candidate_basis=candidate_basis,
            top_n=top_n,
            selected_document_ids=candidate_ids,
            candidate_snapshot_sha256=preview["candidate_snapshot_sha256"],
        )
    else:
        preview_rows = []
    spec = parse_toxicity_query(query)
    historical_lines, _skipped = build_similarity_historical_lines(
        preview_rows,
        spec,
        bayesian_cache=workspace.bayesian_cache,
        bayesian_prior=workspace.bayesian_prior,
    )
    preview = retain_curve_eligible_preview_candidates(preview, historical_lines)
    return {
        "evidence_contract": "legacy_compact_extraction",
        "query": query,
        "ignored_query_fields": ignored_fields,
        "similarity_parameters": {
            "field_weights": resolved_weights,
            "similarity_threshold": resolved_threshold,
            "source": source,
        },
        "embedding_backend_requested": embedding_backend,
        "embedding_backend_used": backend_used,
        "embedding_model": embedding_model if backend_used == "openai" else None,
        **preview,
    }


def predict_probability_at_dose(
    curve_result: dict[str, Any],
    dose: float,
    *,
    method_key: str = "article_line_weighted",
) -> dict[str, Any]:
    """Evaluate the already-built historical mixture at one positive target dose."""

    ensure_current_prediction(curve_result)
    query = curve_result.get("query", {})
    target_unit = query.get("target_unit")
    if not isinstance(target_unit, str) or not target_unit:
        raise ValueError("Curve result does not contain a target unit.")
    if isinstance(dose, bool) or not isinstance(dose, (int, float)) or dose <= 0:
        raise ValueError("Dose must be a positive finite number.")
    canonical_dose, canonical_unit = canonical_dose_value(float(dose), target_unit)
    if canonical_dose is None or canonical_unit != target_unit:
        raise ValueError("Dose cannot be converted to the target unit.")
    method = curve_result.get("toxicity_curve", {}).get(method_key, {})
    if method_key == "power_likelihood":
        if not isinstance(method, dict) or method.get("status") != "fitted":
            raise ValueError("A fitted power-likelihood curve is not available.")
        return {
            **power_likelihood_probability_summary(method, float(canonical_dose)),
            "unit": target_unit,
        }
    if method_key == "power_likelihood_random_effects":
        if not isinstance(method, dict) or method.get("status") != "fitted":
            raise ValueError(
                "A fitted random-effects power-likelihood curve is not available."
            )
        return {
            **random_effects_probability_summary(method, float(canonical_dose)),
            "unit": target_unit,
        }
    if method_key == "power_likelihood_random_intercept_slope":
        if not isinstance(method, dict) or method.get("status") != "fitted":
            raise ValueError(
                "A fitted random-intercept/slope power-likelihood curve is not available."
            )
        return {
            **random_intercept_slope_probability_summary(
                method, float(canonical_dose)
            ),
            "unit": target_unit,
        }
    if method_key == "article_posterior_draw_pool":
        if not isinstance(method, dict) or method.get("status") != "fitted":
            raise ValueError(
                "A fitted article posterior draw-pool curve is not available."
            )
        return {
            **source_mixture_probability_summary(method, float(canonical_dose)),
            "unit": target_unit,
        }
    if method_key != "article_line_weighted":
        raise ValueError(f"Unsupported toxicity curve method: {method_key!r}.")
    summary = posterior_mixture_summary_at_dose(
        method.get("historical_lines", []),
        float(canonical_dose),
    )
    return {**summary, "unit": target_unit}
