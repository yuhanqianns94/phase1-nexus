"""Persistent, validated cache of Bayesian historical arm curves."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .bayesian_toxicity import (
    DEFAULT_POSTERIOR_DRAW_COUNT,
    DEFAULT_PRIOR_CATALOG,
    MODEL_NAME,
    POSTERIOR_APPROXIMATION,
    fit_bayesian_binomial_logistic,
    load_prior_catalog,
    resolve_prior,
)
from .dlt_evidence import (
    aggregate_duplicate_doses,
    build_candidate_rows,
    group_candidate_rows,
    normalized_key,
)


CACHE_FORMAT = "mtd_similarity_bayesian_historical_curve_cache_v4"
DEFAULT_BAYESIAN_CURVE_CACHE = Path(
    "output/mtd_similarity/bayesian_historical_curve_cache.json"
)
# Fits loaded through ``load_bayesian_curve_cache`` have already passed a
# whole-payload check, so they need not be JSON-hashed again for every target
# during full-library optimization. Fresh/in-memory cache objects remain
# checked on every use, which catches mutations made by callers or tests.
_TRUSTED_LOADED_CACHE_IDS: set[int] = set()
_RUNTIME_CACHE_OBJECTS: dict[int, dict[str, Any]] = {}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _cache_payload_fingerprint(cache: dict[str, Any]) -> str:
    """Hash every persistent field except the digest itself."""

    return _hash_payload(
        {
            key: value
            for key, value in cache.items()
            if key != "payload_fingerprint_sha256"
        }
    )


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _validate_fitted_payload(
    fit: dict[str, Any],
    *,
    prior_name: str,
    expected_draw_count: int,
) -> str | None:
    """Return a cache-payload violation instead of trusting stored draws."""

    if fit.get("model") != MODEL_NAME:
        return "fit_model_mismatch"
    if fit.get("posterior_approximation") != POSTERIOR_APPROXIMATION:
        return "fit_posterior_approximation_mismatch"
    if fit.get("prior_id") != prior_name:
        return "fit_prior_id_mismatch"
    standardization = fit.get("dose_standardization")
    posterior = fit.get("posterior")
    diagnostics = fit.get("fit_diagnostics")
    if not isinstance(standardization, dict) or not isinstance(posterior, dict):
        return "fit_missing_standardization_or_posterior"
    if not isinstance(diagnostics, dict):
        return "fit_missing_diagnostics"
    if (
        standardization.get("type")
        != "natural_log_ratio_to_geometric_mean_observed_dose"
    ):
        return "fit_dose_standardization_mismatch"
    reference_dose = _finite_float(standardization.get("reference_dose"))
    reference_log_dose = _finite_float(standardization.get("reference_log_dose"))
    if (
        reference_dose is None
        or reference_dose <= 0
        or reference_log_dose is None
    ):
        return "fit_invalid_dose_standardization"
    if posterior.get("draw_count") != expected_draw_count:
        return "fit_draw_count_mismatch"
    try:
        draws = np.asarray(posterior.get("draws_alpha_eta"), dtype=float)
        covariance = np.asarray(posterior.get("covariance_alpha_eta"), dtype=float)
    except (TypeError, ValueError):
        return "fit_non_numeric_posterior"
    if (
        draws.shape != (expected_draw_count, 2)
        or not np.isfinite(draws).all()
    ):
        return "fit_invalid_posterior_draws"
    if (
        covariance.shape != (2, 2)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)
    ):
        return "fit_invalid_covariance"
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return "fit_covariance_not_positive_definite"
    mode = posterior.get("mode")
    if not isinstance(mode, dict) or any(
        _finite_float(mode.get(key)) is None
        for key in ("alpha", "eta", "monotone_slope")
    ):
        return "fit_invalid_posterior_mode"
    for key in (
        "negative_log_posterior_at_mode",
        "gradient_norm",
        "hessian_condition_number",
    ):
        if _finite_float(diagnostics.get(key)) is None:
            return f"fit_nonfinite_diagnostic:{key}"
    eigenvalues = diagnostics.get("hessian_eigenvalues")
    if (
        not isinstance(eigenvalues, list)
        or not eigenvalues
        or any(_finite_float(value) is None for value in eigenvalues)
    ):
        return "fit_invalid_hessian_eigenvalues"
    return None


def _validate_cache_payload(
    cache: dict[str, Any],
    *,
    records: list[dict[str, Any]] | None = None,
    prior_name: str | None = None,
) -> str | None:
    """Validate cache entries and optional source-segment identity."""

    if cache.get("payload_fingerprint_sha256") != _cache_payload_fingerprint(cache):
        return "cache_payload_fingerprint_mismatch"
    embedded_catalog = cache.get("prior_catalog")
    if (
        not isinstance(embedded_catalog, dict)
        or _hash_payload(embedded_catalog)
        != cache.get("prior_catalog_fingerprint_sha256")
    ):
        return "cache_embedded_prior_catalog_mismatch"
    expected_draw_count = cache.get("posterior_draw_count")
    if (
        isinstance(expected_draw_count, bool)
        or not isinstance(expected_draw_count, int)
        or expected_draw_count < 32
    ):
        return "cache_invalid_posterior_draw_count"
    selected_priors = cache.get("selected_priors")
    segments = cache.get("segments")
    if (
        not isinstance(selected_priors, list)
        or not selected_priors
        or len(set(selected_priors)) != len(selected_priors)
        or not all(isinstance(name, str) and name for name in selected_priors)
        or not isinstance(segments, list)
    ):
        return "cache_invalid_segment_or_prior_metadata"
    expected_sources: dict[str, tuple[str, str, int]] | None = None
    if records is not None:
        expected_sources = {}
        for record in records:
            pmcid = record.get("pmcid")
            if not isinstance(pmcid, str) or not pmcid:
                continue
            for segment in _article_segments(record)[0]:
                key = segment_cache_key(pmcid, segment)
                if key in expected_sources:
                    return "cache_expected_duplicate_segment_key"
                expected_sources[key] = (
                    segment_source_fingerprint(segment),
                    pmcid,
                    len(segment),
                )
    seen_keys: set[str] = set()
    for entry in segments:
        if not isinstance(entry, dict):
            return "cache_non_object_segment_entry"
        key = entry.get("segment_key")
        if not isinstance(key, str) or not key or key in seen_keys:
            return "cache_duplicate_or_missing_segment_key"
        seen_keys.add(key)
        if expected_sources is not None:
            expected_source = expected_sources.get(key)
            if (
                expected_source is None
                or entry.get("source_fingerprint_sha256") != expected_source[0]
                or entry.get("pmcid") != expected_source[1]
                or entry.get("source_point_count") != expected_source[2]
            ):
                return "cache_segment_source_mismatch"
        fits = entry.get("fits")
        fit_fingerprints = entry.get("fit_payload_fingerprints_sha256")
        if not isinstance(fits, dict) or not isinstance(fit_fingerprints, dict):
            return "cache_missing_segment_fits"
        names_to_validate = [prior_name] if prior_name else selected_priors
        for name in names_to_validate:
            fit = fits.get(name)
            if not isinstance(fit, dict):
                return "cache_missing_selected_prior_fit"
            if fit_fingerprints.get(name) != _hash_payload(fit):
                return "cache_fit_payload_fingerprint_mismatch"
            status = fit.get("status")
            if status == "fitted":
                violation = _validate_fitted_payload(
                    fit,
                    prior_name=name,
                    expected_draw_count=expected_draw_count,
                )
                if violation:
                    return violation
            elif status == "not_fit_eligible":
                skip_reasons = fit.get("skip_reasons")
                if (
                    not isinstance(skip_reasons, list)
                    or not skip_reasons
                    or not all(
                        isinstance(reason, str) and reason
                        for reason in skip_reasons
                    )
                ):
                    return "cache_invalid_nonfitted_skip_reasons"
            else:
                return "cache_unknown_fit_status"
    if expected_sources is not None and seen_keys != set(expected_sources):
        return "cache_segment_set_mismatch"
    return None


def _relevant_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    if not isinstance(data, dict):
        data = {}
    return {
        "pmcid": record.get("pmcid"),
        # These fields can determine the conservative effective denominator
        # used for an inferred zero-DLT dose and therefore belong in cache
        # identity even though the extraction itself remains unchanged.
        "DesignType": data.get("DesignType"),
        "schema": data.get("schema"),
        "dose_levels": data.get("dose_levels") or [],
        "dlt_count_at_each_dose_level": data.get("dlt_count_at_each_dose_level") or [],
    }


def _validate_unique_pmcids(records: list[dict[str, Any]]) -> None:
    pmcids = [
        record.get("pmcid")
        for record in records
        if isinstance(record.get("pmcid"), str) and record.get("pmcid")
    ]
    duplicates = sorted(
        pmcid for pmcid, count in Counter(pmcids).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Duplicate PMCIDs are not allowed in Bayesian curve cache input: "
            + ", ".join(duplicates)
        )


def source_library_fingerprint(records: list[dict[str, Any]]) -> str:
    """Fingerprint actual curve-source evidence, not merely filenames."""

    _validate_unique_pmcids(records)
    return _hash_payload(
        [
            _relevant_record_payload(record)
            for record in sorted(records, key=lambda row: str(row.get("pmcid", "")))
        ]
    )


def segment_source_payload(segment: list[dict[str, Any]]) -> dict[str, Any]:
    """Only model-relevant fields are allowed to define cache identity."""

    if not segment:
        raise ValueError("Cannot fingerprint an empty DLT segment.")
    return {
        "arm_or_cohort": segment[0].get("arm_or_cohort"),
        "drug_name": segment[0].get("drug_name"),
        "regimen_signature": segment[0].get("regimen_signature"),
        "frequency": segment[0].get("frequency"),
        "schedule": segment[0].get("schedule"),
        "canonical_unit": segment[0].get("canonical_unit"),
        "points": [
            {
                "canonical_dose": row.get("canonical_dose"),
                "n_patients_with_dlt": row.get("n_patients_with_dlt"),
                "n_patients": row.get("n_patients"),
            }
            for row in segment
        ],
    }


def segment_source_fingerprint(segment: list[dict[str, Any]]) -> str:
    return _hash_payload(segment_source_payload(segment))


def segment_cache_key(pmcid: str, segment: list[dict[str, Any]]) -> str:
    return _hash_payload(
        {
            "pmcid": pmcid,
            "source": segment_source_payload(segment),
        }
    )


def _draw_seed(base_seed: int, *, segment_key: str, prior_name: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{segment_key}|{prior_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _article_segments(record: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], list[str]]:
    data = record.get("data")
    if not isinstance(data, dict):
        return [], ["record_data_not_object"]
    candidates, reasons = build_candidate_rows(data)
    groups, group_reasons = group_candidate_rows(candidates)
    reasons.update(group_reasons)
    return (
        [aggregate_duplicate_doses(rows) for rows in groups.values() if rows],
        sorted(reasons),
    )


def build_bayesian_curve_cache(
    records: list[dict[str, Any]],
    *,
    prior_catalog_path: Path = DEFAULT_PRIOR_CATALOG,
    prior_names: list[str] | None = None,
    posterior_draw_count: int = DEFAULT_POSTERIOR_DRAW_COUNT,
    random_seed: int = 20260725,
) -> dict[str, Any]:
    """Fit every eligible historical arm under every requested candidate prior."""

    if posterior_draw_count < 32:
        raise ValueError("posterior_draw_count must be at least 32.")
    _validate_unique_pmcids(records)
    catalog, catalog_payload, catalog_fingerprint = load_prior_catalog(prior_catalog_path)
    selected_names = sorted(prior_names or catalog.keys())
    if not selected_names:
        raise ValueError("At least one Bayesian prior must be selected.")
    selected_priors = {
        name: resolve_prior(name, catalog) for name in selected_names
    }

    entries: list[dict[str, Any]] = []
    source_reason_counts: Counter[str] = Counter()
    fit_status_counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda row: str(row.get("pmcid", ""))):
        pmcid = record.get("pmcid")
        if not isinstance(pmcid, str) or not pmcid:
            source_reason_counts["missing_pmcid"] += 1
            continue
        segments, reasons = _article_segments(record)
        source_reason_counts.update(reasons)
        for segment in segments:
            key = segment_cache_key(pmcid, segment)
            fits: dict[str, Any] = {}
            for prior_name, prior in selected_priors.items():
                fit = fit_bayesian_binomial_logistic(
                    segment,
                    prior,
                    draw_count=posterior_draw_count,
                    random_seed=_draw_seed(
                        random_seed, segment_key=key, prior_name=prior_name
                    ),
                )
                fits[prior_name] = fit
                fit_status_counts[f"{prior_name}:{fit['status']}"] += 1
                for reason in fit.get("skip_reasons", []):
                    fit_status_counts[f"{prior_name}:skip:{reason}"] += 1
            entries.append(
                {
                    "segment_key": key,
                    "pmcid": pmcid,
                    "arm_or_cohort": segment[0].get("arm_or_cohort"),
                    "normalized_arm_or_cohort": normalized_key(
                        segment[0].get("arm_or_cohort")
                    ),
                    "drug_name": segment[0].get("drug_name"),
                    "normalized_drug_name": normalized_key(segment[0].get("drug_name")),
                    "regimen_signature": segment[0].get("regimen_signature"),
                    "canonical_unit": segment[0].get("canonical_unit"),
                    "source_fingerprint_sha256": segment_source_fingerprint(segment),
                    "source_point_count": len(segment),
                    "source_patient_count": sum(
                        float(row["n_patients"]) for row in segment
                    ),
                    "observed_dose_min": min(
                        float(row["canonical_dose"]) for row in segment
                    ),
                    "observed_dose_max": max(
                        float(row["canonical_dose"]) for row in segment
                    ),
                    "fits": fits,
                    "fit_payload_fingerprints_sha256": {
                        prior_name: _hash_payload(fit)
                        for prior_name, fit in fits.items()
                    },
                }
            )

    cache = {
        "format": CACHE_FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "source_library_fingerprint_sha256": source_library_fingerprint(records),
        "prior_catalog_path": str(prior_catalog_path),
        "prior_catalog_fingerprint_sha256": catalog_fingerprint,
        "prior_catalog": catalog_payload,
        "selected_priors": selected_names,
        "posterior_draw_count": posterior_draw_count,
        "random_seed": random_seed,
        "segments": entries,
        "summary": {
            "source_article_count": len(records),
            "cached_segment_count": len(entries),
            "source_skip_reasons": dict(sorted(source_reason_counts.items())),
            "fit_status_counts": dict(sorted(fit_status_counts.items())),
        },
    }
    cache["payload_fingerprint_sha256"] = _cache_payload_fingerprint(cache)
    return cache


def write_bayesian_curve_cache(path: Path, cache: dict[str, Any]) -> None:
    """Write atomically so target estimation never reads a partial cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_bayesian_curve_cache(
    path: Path,
    *,
    records: list[dict[str, Any]],
    prior_name: str,
    prior_catalog_path: Path = DEFAULT_PRIOR_CATALOG,
) -> dict[str, Any]:
    """Load only a cache proven compatible with current evidence and priors."""

    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"Bayesian curve cache does not exist: {path}. "
            "Run python -m mtd_similarity.fit_bayesian_toxicity_library first."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Bayesian curve cache is invalid JSON: {path}") from error
    if not isinstance(cache, dict) or cache.get("format") != CACHE_FORMAT:
        raise ValueError(f"Bayesian curve cache has an unsupported format: {path}")
    if cache.get("model") != MODEL_NAME:
        raise ValueError(
            "Bayesian curve cache was generated by a different model version. "
            "Rebuild it before prediction."
        )
    expected_library_fingerprint = source_library_fingerprint(records)
    if cache.get("source_library_fingerprint_sha256") != expected_library_fingerprint:
        raise ValueError(
            "Bayesian curve cache does not match the current dose--DLT extraction "
            "library. Rebuild it before prediction."
        )
    catalog, _payload, expected_catalog_fingerprint = load_prior_catalog(
        prior_catalog_path
    )
    resolve_prior(prior_name, catalog)
    if cache.get("prior_catalog_fingerprint_sha256") != expected_catalog_fingerprint:
        raise ValueError(
            "Bayesian curve cache does not match the current prior catalog. "
            "Rebuild it before prediction."
        )
    if prior_name not in cache.get("selected_priors", []):
        raise ValueError(
            f"Bayesian curve cache does not contain prior {prior_name!r}. "
            "Rebuild it with that prior."
        )
    payload_violation = _validate_cache_payload(
        cache,
        records=records,
        prior_name=prior_name,
    )
    if payload_violation:
        raise ValueError(
            "Bayesian curve cache payload is invalid or stale "
            f"({payload_violation}). Rebuild it before prediction."
        )
    _RUNTIME_CACHE_OBJECTS[id(cache)] = cache
    _TRUSTED_LOADED_CACHE_IDS.add(id(cache))
    return cache


def cached_segment_fit(
    cache: dict[str, Any],
    *,
    pmcid: str,
    segment: list[dict[str, Any]],
    prior_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a prior-specific fitted posterior for exactly one evidence segment."""

    source_fingerprint = segment_source_fingerprint(segment)
    key = segment_cache_key(pmcid, segment)
    matched = next(
        (
            entry
            for entry in cache.get("segments", [])
            if entry.get("segment_key") == key
            and entry.get("source_fingerprint_sha256") == source_fingerprint
        ),
        None,
    )
    if matched is None:
        return None, "bayesian_cache_segment_missing_or_stale"
    fit = matched.get("fits", {}).get(prior_name)
    if not isinstance(fit, dict):
        return None, "bayesian_cache_prior_missing"
    fit_fingerprints = matched.get("fit_payload_fingerprints_sha256")
    _RUNTIME_CACHE_OBJECTS[id(cache)] = cache
    if id(cache) not in _TRUSTED_LOADED_CACHE_IDS:
        if (
            not isinstance(fit_fingerprints, dict)
            or fit_fingerprints.get(prior_name) != _hash_payload(fit)
        ):
            return None, "bayesian_cache_fit_payload_fingerprint_mismatch"
        if fit.get("status") == "fitted":
            expected_draw_count = cache.get("posterior_draw_count")
            if not isinstance(expected_draw_count, int):
                return None, "bayesian_cache_invalid_posterior_draw_count"
            fit_violation = _validate_fitted_payload(
                fit,
                prior_name=prior_name,
                expected_draw_count=expected_draw_count,
            )
            if fit_violation:
                return None, f"bayesian_cache_fit_invalid:{fit_violation}"
    if fit.get("status") != "fitted":
        reasons = fit.get("skip_reasons")
        if isinstance(reasons, list) and reasons:
            return None, f"bayesian_cached_fit_not_available:{reasons[0]}"
        return None, "bayesian_cached_fit_not_available"
    return {
        **fit,
        "cache_segment_metadata": {
            key: matched[key]
            for key in (
                "segment_key",
                "source_fingerprint_sha256",
                "source_point_count",
                "source_patient_count",
                "observed_dose_min",
                "observed_dose_max",
            )
            if key in matched
        },
    }, None


def cache_metadata_for_result(cache: dict[str, Any], prior_name: str, path: Path) -> dict[str, Any]:
    """Compact provenance persisted in every target/LOO/optimization result."""

    return {
        "path": str(path),
        "format": cache.get("format"),
        "model": cache.get("model"),
        "prior_name": prior_name,
        "posterior_draw_count": cache.get("posterior_draw_count"),
        "source_library_fingerprint_sha256": cache.get(
            "source_library_fingerprint_sha256"
        ),
        "prior_catalog_fingerprint_sha256": cache.get(
            "prior_catalog_fingerprint_sha256"
        ),
        "payload_fingerprint_sha256": cache.get("payload_fingerprint_sha256"),
        "generated_at_utc": cache.get("generated_at_utc"),
    }
