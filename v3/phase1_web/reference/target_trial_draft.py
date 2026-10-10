"""Short-lived, local persistence for Target Trial widget values.

Only an explicit allowlist of non-secret Target Trial fields is written.  The
draft is a convenience for reopening the local Streamlit app; it is not a
prediction input artifact and is deliberately expired after a short period.
"""

from __future__ import annotations
import os

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .helpers import (
    ADMIN_ROUTE_OPTIONS,
    AGE_GROUP_OPTIONS,
    DOSE_FORM_OPTIONS,
    FREQUENCY_OPTIONS,
    GENDER_SEX_OPTIONS,
    PROJECT_ROOT,
    SIMILARITY_FIELD_ORDER,
    format_similarity_weight_input,
)


TARGET_TRIAL_DRAFT_FORMAT = "toxicity_ui_target_trial_draft_v1"
TARGET_TRIAL_DRAFT_TTL = timedelta(days=1)
DEFAULT_TARGET_TRIAL_DRAFT = (
    Path(os.environ.get("PHASE1_V3_STATE_ROOT", str(Path(__file__).resolve().parents[2] / "output"))) / "toxicity_ui" / "target_trial_draft.json"
)
TARGET_TRIAL_DRAFT_FINGERPRINT_KEY = "_target_trial_draft_saved_fingerprint"
TARGET_TRIAL_DRAFT_SKIP_SAVE_KEY = "_target_trial_draft_skip_save_once"

def _draft_path(path):
    if path is not None:return Path(path)
    root=os.environ.get('PHASE1_V3_STATE_ROOT')
    return Path(root)/'toxicity_ui'/'target_trial_draft.json' if root else DEFAULT_TARGET_TRIAL_DRAFT


_TEXT_KEYS = {
    "native_target_rows_json", "native_target_scenario", "native_purpose", "native_modifiers",
    "native_study_phase", "native_study_role", "native_dlt_exceptions", "native_grading",
    "native_dlt_version", "native_window_kind", "native_window_start", "native_window_end", "native_adjustments",
    "target_drug",
    "target_dose_unit",
    "planned_doses_text",
    "opt_cancer_type",
    "opt_schedule_admin_days",
    "opt_schema",
    "opt_inclusion",
    "opt_exclusion",
    "opt_dlt_definition",
    "opt_dlt_eva_text",
    "opt_gene_targets",
    "opt_mechanism",
    *(f"sim_weight_text_{field}" for field in SIMILARITY_FIELD_ORDER),
}
_BOOLEAN_KEYS = {
    "power_likelihood_enabled",
    "power_likelihood_random_effects_enabled",
    "power_likelihood_random_intercept_slope_enabled",
}
_NONNEGATIVE_NUMBER_KEYS = {
    "native_window_cycle_days",
    "opt_schedule_cycle_days",
    "opt_dlt_eva_days",
    "opt_dlt_eva_cycles",
}
_OPTION_KEYS: dict[str, set[str]] = {
    "trial_type_control": {"single", "combination"},
    "target_curve_method": {
        "article_line_weighted",
        "article_posterior_draw_pool",
        "power_likelihood_random_intercept_slope",
    },
    "embedding_backend_select": {"local", "openai"},
    "embedding_model_select": {
        "text-embedding-3-small",
        "text-embedding-3-large",
    },
    "historical_article_selection_mode": {"automatic", "manual"},
    "historical_article_candidate_basis": {"threshold", "top_n"},
    "opt_schedule_frequency": {"(blank)", *FREQUENCY_OPTIONS},
    "opt_age_group": {"(blank)", *AGE_GROUP_OPTIONS},
    "opt_age_unit": {"years", "months", "days"},
    "opt_gender": {"(blank)", *GENDER_SEX_OPTIONS},
}
_MULTISELECT_KEYS: dict[str, set[str]] = {
    "opt_routes": set(ADMIN_ROUTE_OPTIONS),
    "opt_dose_forms": set(DOSE_FORM_OPTIONS),
}
TARGET_TRIAL_WIDGET_KEYS = frozenset(
    _TEXT_KEYS
    | _BOOLEAN_KEYS
    | _NONNEGATIVE_NUMBER_KEYS
    | set(_OPTION_KEYS)
    | set(_MULTISELECT_KEYS)
    | {
        "bayesian_prior_select",
        "similarity_threshold_input",
        "opt_age_min",
        "opt_age_max",
        "power_likelihood_borrowing_strength",
        "power_likelihood_ess_cap",
        "random_intercept_scale_prior_mean",
        "random_intercept_scale_prior_sd",
        "random_slope_scale_prior_mean",
        "random_slope_scale_prior_sd",
        "historical_article_top_n",
    }
)


@dataclass(frozen=True)
class TargetTrialDraft:
    """Validated draft values plus their persistence timestamps."""

    values: dict[str, Any]
    saved_at_utc: datetime
    expires_at_utc: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc_datetime(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Target Trial draft {label} must be an ISO timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"Target Trial draft {label} must be an ISO timestamp."
        ) from error
    if parsed.tzinfo is None:
        raise ValueError(f"Target Trial draft {label} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return value


def sanitize_target_trial_values(
    values: Mapping[str, Any],
    *,
    prior_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    """Return widget-safe values from an untrusted or older draft payload."""

    sanitized: dict[str, Any] = {}
    for key in _TEXT_KEYS:
        value = values.get(key)
        if isinstance(value, str):
            sanitized[key] = value
    for key in _BOOLEAN_KEYS:
        value = values.get(key)
        if isinstance(value, bool):
            sanitized[key] = value
    for key, allowed in _OPTION_KEYS.items():
        value = values.get(key)
        if isinstance(value, str) and value in allowed:
            sanitized[key] = value
    for key, allowed in _MULTISELECT_KEYS.items():
        value = values.get(key)
        if isinstance(value, list) and all(
            isinstance(item, str) and item in allowed for item in value
        ):
            sanitized[key] = list(dict.fromkeys(value))
    for key in _NONNEGATIVE_NUMBER_KEYS:
        value = _finite_number(values.get(key))
        if value is not None and float(value) >= 0:
            sanitized[key] = value
    for key in ("opt_age_min", "opt_age_max"):
        if key not in values:
            continue
        raw = values.get(key)
        if raw is None:
            sanitized[key] = None
            continue
        value = _finite_number(raw)
        if value is not None and float(value) >= 0:
            sanitized[key] = value
    threshold = _finite_number(values.get("similarity_threshold_input"))
    if threshold is not None and 0 <= float(threshold) <= 1:
        sanitized["similarity_threshold_input"] = threshold
    borrowing_strength = _finite_number(
        values.get("power_likelihood_borrowing_strength")
    )
    if borrowing_strength is not None and 0.01 <= float(borrowing_strength) <= 1:
        sanitized["power_likelihood_borrowing_strength"] = borrowing_strength
    ess_cap = _finite_number(values.get("power_likelihood_ess_cap"))
    if ess_cap is not None and float(ess_cap) >= 0.1:
        sanitized["power_likelihood_ess_cap"] = ess_cap
    for key in (
        "random_intercept_scale_prior_mean",
        "random_slope_scale_prior_mean",
    ):
        value = _finite_number(values.get(key))
        if value is not None:
            sanitized[key] = value
    for key in (
        "random_intercept_scale_prior_sd",
        "random_slope_scale_prior_sd",
    ):
        value = _finite_number(values.get(key))
        if value is not None and float(value) >= 0.05:
            sanitized[key] = value
    top_n = _finite_number(values.get("historical_article_top_n"))
    if (
        top_n is not None
        and float(top_n).is_integer()
        and 1 <= int(top_n) <= 200
    ):
        sanitized["historical_article_top_n"] = int(top_n)
    prior = values.get("bayesian_prior_select")
    allowed_priors = set(prior_names or [])
    if isinstance(prior, str) and (not allowed_priors or prior in allowed_priors):
        sanitized["bayesian_prior_select"] = prior
    return sanitized


def target_trial_default_values(
    *,
    default_prior: str,
    default_weights: Mapping[str, float],
    default_threshold: float,
) -> dict[str, Any]:
    """Return every persisted widget default so restoration is deterministic."""

    values: dict[str, Any] = {
        "target_drug": "",
        "target_dose_unit": "",
        "trial_type_control": "single",
        "bayesian_prior_select": default_prior,
        "planned_doses_text": "",
        "target_curve_method": "article_line_weighted",
        "similarity_threshold_input": float(default_threshold),
        "embedding_backend_select": "local",
        "embedding_model_select": "text-embedding-3-small",
        "power_likelihood_enabled": False,
        "power_likelihood_random_effects_enabled": False,
        "power_likelihood_random_intercept_slope_enabled": True,
        "power_likelihood_borrowing_strength": 1.0,
        "power_likelihood_ess_cap": 30.0,
        "random_intercept_scale_prior_mean": -1.1,
        "random_intercept_scale_prior_sd": 1.0,
        "random_slope_scale_prior_mean": -1.5,
        "random_slope_scale_prior_sd": 0.8,
        "historical_article_selection_mode": "automatic",
        "historical_article_candidate_basis": "threshold",
        "historical_article_top_n": 40,
        "opt_cancer_type": "",
        "opt_routes": [],
        "opt_schedule_frequency": "(blank)",
        "opt_schedule_cycle_days": 0.0,
        "opt_schedule_admin_days": "",
        "opt_schema": "",
        "opt_inclusion": "",
        "opt_exclusion": "",
        "opt_dlt_definition": "",
        "opt_dlt_eva_days": 0.0,
        "opt_dlt_eva_cycles": 0.0,
        "opt_dlt_eva_text": "",
        "opt_age_group": "(blank)",
        "opt_age_min": None,
        "opt_age_max": None,
        "opt_age_unit": "years",
        "opt_gender": "(blank)",
        "opt_dose_forms": [],
        "opt_gene_targets": "",
        "opt_mechanism": "",
    }
    for key in _TEXT_KEYS:
        values.setdefault(key, "")
    for key in _NONNEGATIVE_NUMBER_KEYS:
        values.setdefault(key, 0.)
    values.update(native_target_rows_json="[]", native_target_scenario="Main",
                  native_study_role="unclear", native_window_kind="formal_dlt")
    values.update(
        {
            f"sim_weight_text_{field}": format_similarity_weight_input(
                float(default_weights.get(field, 0.0))
            )
            for field in SIMILARITY_FIELD_ORDER
        }
    )
    return values


def target_trial_values_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only JSON-safe, allowlisted Target Trial fields from Session State."""

    return sanitize_target_trial_values(
        {key: state[key] for key in TARGET_TRIAL_WIDGET_KEYS if key in state}
    )


def target_trial_values_fingerprint(state: Mapping[str, Any]) -> str | None:
    """Return a stable fingerprint, or ``None`` when no Target field exists."""

    if not any(key in state for key in TARGET_TRIAL_WIDGET_KEYS):
        return None
    values = target_trial_values_from_state(state)
    rendered = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def save_target_trial_draft(
    state: Mapping[str, Any],
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> TargetTrialDraft:
    """Atomically save one 24-hour local draft with restrictive permissions."""
    path = _draft_path(path)

    saved_at = (now or _utc_now()).astimezone(timezone.utc)
    expires_at = saved_at + TARGET_TRIAL_DRAFT_TTL
    values = target_trial_values_from_state(state)
    payload = {
        "format": TARGET_TRIAL_DRAFT_FORMAT,
        "saved_at_utc": saved_at.isoformat(),
        "expires_at_utc": expires_at.isoformat(),
        "values": values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return TargetTrialDraft(values, saved_at, expires_at)


def save_target_trial_draft_if_changed(
    state: MutableMapping[str, Any],
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> TargetTrialDraft | None:
    """Persist changed Target values without extending an unchanged draft TTL."""
    path = _draft_path(path)

    fingerprint = target_trial_values_fingerprint(state)
    if fingerprint is None:
        return None
    if state.get(TARGET_TRIAL_DRAFT_FINGERPRINT_KEY) == fingerprint:
        return None
    draft = save_target_trial_draft(state, path=path, now=now)
    state[TARGET_TRIAL_DRAFT_FINGERPRINT_KEY] = fingerprint
    return draft


def load_target_trial_draft(
    *,
    path: Path | None = None,
    prior_names: list[str] | tuple[str, ...] | set[str] | None = None,
    now: datetime | None = None,
) -> TargetTrialDraft | None:
    """Load a current draft; missing and expired drafts are treated as absent."""
    path = _draft_path(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read the saved Target Trial draft: {error}"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("format") != TARGET_TRIAL_DRAFT_FORMAT
    ):
        raise ValueError("Saved Target Trial draft has an unsupported format.")
    saved_at = _as_utc_datetime(payload.get("saved_at_utc"), label="saved_at_utc")
    stored_expires_at = _as_utc_datetime(
        payload.get("expires_at_utc"), label="expires_at_utc"
    )
    # The file is local and editable, so its declared expiry is provenance only.
    # Never allow a modified/corrupt payload to extend the 24-hour contract.
    expires_at = min(
        stored_expires_at,
        saved_at + TARGET_TRIAL_DRAFT_TTL,
    )
    current = (now or _utc_now()).astimezone(timezone.utc)
    if expires_at <= current:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Expiry is enforced even if this process cannot remove the stale
            # file; a later load will still refuse to restore it.
            pass
        return None
    raw_values = payload.get("values")
    if not isinstance(raw_values, dict):
        raise ValueError("Saved Target Trial draft values must be an object.")
    values = sanitize_target_trial_values(raw_values, prior_names=prior_names)
    return TargetTrialDraft(values, saved_at, expires_at)


def clear_target_trial_draft(*, path: Path | None = None) -> bool:
    """Remove the persisted draft and report whether one existed."""
    path = _draft_path(path)

    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def clear_target_trial_state(state: dict[str, Any] | Any) -> None:
    """Remove Target Trial widgets and dependent prediction results in-place."""

    for key in TARGET_TRIAL_WIDGET_KEYS:
        state.pop(key, None)
    for key in ("curve_result", "oc_result", "oc_result_signature"):
        state.pop(key, None)
    for key in list(state):
        if str(key).startswith("historical_article_review_"):
            state.pop(key, None)
    state.pop(TARGET_TRIAL_DRAFT_FINGERPRINT_KEY, None)
