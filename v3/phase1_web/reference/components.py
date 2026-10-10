"""Streamlit page components for the local toxicity interface."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np

from .article_review import (
    candidate_table_rows,
    extracted_content_summary,
    load_extracted_content,
    normalize_document_id,
    resolve_verified_pdf,
    selected_document_ids,
)
from .helpers import (
    ADMIN_ROUTE_OPTIONS,
    AGE_GROUP_OPTIONS,
    DOSE_FORM_OPTIONS,
    FREQUENCY_OPTIONS,
    GENDER_SEX_OPTIONS,
    SIMILARITY_FIELD_LABELS,
    SIMILARITY_FIELD_ORDER,
    build_optional_query_fields,
    build_target_query,
    count_builtin_extractions,
    dose_support_summary,
    friendly_library_label,
    format_similarity_weight_input,
    load_default_prior_name,
    load_default_similarity_parameters,
    load_json_file,
    pick_local_directory,
    parse_similarity_weight_inputs,
    parse_probability_list,
    prediction_json,
    validate_similarity_inputs,
)
from .target_trial_draft import (
    TARGET_TRIAL_DRAFT_FINGERPRINT_KEY,
    TARGET_TRIAL_DRAFT_SKIP_SAVE_KEY,
    TARGET_TRIAL_WIDGET_KEYS,
    clear_target_trial_draft,
    clear_target_trial_state,
    load_target_trial_draft,
    sanitize_target_trial_values,
    save_target_trial_draft_if_changed,
    target_trial_default_values,
    target_trial_values_fingerprint,
)


@dataclass(frozen=True)
class PredictionRequest:
    """Validated target input plus selected similarity and prior settings."""

    query: dict[str, Any]
    bayesian_prior: str
    field_weights: dict[str, float]
    similarity_threshold: float
    curve_method: str
    embedding_backend: str
    embedding_model: str
    power_likelihood_enabled: bool
    power_likelihood_random_effects_enabled: bool
    power_likelihood_random_intercept_slope_enabled: bool
    power_likelihood_borrowing_strength: float
    power_likelihood_ess_cap: float
    random_intercept_scale_prior_mean: float
    random_intercept_scale_prior_sd: float
    random_slope_scale_prior_mean: float
    random_slope_scale_prior_sd: float
    historical_article_selection_mode: str = "automatic"
    historical_article_candidate_basis: str = "threshold"
    historical_article_top_n: int = 40
    historical_article_ids: tuple[str, ...] = ()
    historical_article_candidate_snapshot_sha256: str | None = None
    submission_action: str = "predict"


_LIBRARY_SOURCE_KEY = "library_source_dir"
_PENDING_LIBRARY_SOURCE_KEY = "_pending_library_source_dir"
_SCANNED_LIBRARY_SOURCE_KEY = "_scanned_library_source_dir"


_DIAGNOSTIC_LABELS = {
    "q05": "5th percentile",
    "q95": "95th percentile",
    "median": "Median",
    "dlt": "DLT",
    "ess": "ESS",
    "oc": "OC",
    "mtd": "MTD",
}

_CURVE_METHOD_LABELS = {
    "article_line_weighted": "Current curve — average of article posterior curves",
    "article_posterior_draw_pool": (
        "Posterior mixture — sample whole article-component curves by article similarity"
    ),
    "power_likelihood": "Legacy method 3 — shared power-likelihood curve",
    "power_likelihood_random_effects": (
        "Legacy method 4 — power likelihood with study random intercepts"
    ),
    "power_likelihood_random_intercept_slope": (
        "Method 3 — power likelihood with random intercepts and slopes"
    ),
}

_EMBEDDING_BACKEND_LABELS = {
    "local": "Local text cosine (no API cost)",
    "openai": "OpenAI semantic embedding",
}
_OPENAI_EMBEDDING_MODEL_LABELS = {
    "text-embedding-3-small": "text-embedding-3-small (lower cost; default)",
    "text-embedding-3-large": "text-embedding-3-large (more capable)",
}


# Editable starting values for the BOIN parameter search.  These are real widget
# values rather than placeholders so a fresh page opens with one valid, useful
# comparison grid (3 x 2 x 3 = 18 candidates, below the 96-candidate
# safety limit).  Users remain free to replace or clear every value.
_DEFAULT_BOIN_CALIBRATION_CANDIDATES = "automatic, 0.10:0.40, 0.20:0.30"
_DEFAULT_BOIN_ELIMINATION_CUTOFF_CANDIDATES = "0.90, 0.95"
_DEFAULT_BOIN_ELIMINATION_MIN_N = 1
_DEFAULT_BOIN_STOP_RULE_CANDIDATES = "off, 9, 12"
_DEFAULT_BOIN_POSTERIOR_CURVE_COUNT = 50


def _readable_field_label(value: Any) -> str:
    """Turn one machine-readable result key into a compact display label."""

    text = str(value).strip()
    if not text:
        return "Value"
    lowered = text.lower()
    if lowered in _DIAGNOSTIC_LABELS:
        return _DIAGNOSTIC_LABELS[lowered]
    words = text.replace("_", " ").replace("-", " ").split()
    rendered = [_DIAGNOSTIC_LABELS.get(word.lower(), word) for word in words]
    label = " ".join(rendered)
    return f"{label[:1].upper()}{label[1:]}"


def _readable_diagnostic_rows(
    payload: Any,
    *,
    root_label: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten nested diagnostic data into a complete human-readable table.

    Lists and mappings are traversed recursively, so replacing ``st.json`` with
    this representation does not hide nested audit fields. Empty collections are
    represented explicitly instead of disappearing from the display.
    """

    rows: list[dict[str, Any]] = []

    def walk(value: Any, path: list[str]) -> None:
        if isinstance(value, Mapping):
            if not value:
                rows.append(
                    {
                        "Field": " › ".join(path) or "Value",
                        "Value": "— (empty section)",
                    }
                )
                return
            for key, child in value.items():
                walk(child, [*path, _readable_field_label(key)])
            return
        if isinstance(value, (list, tuple)):
            if not value:
                rows.append(
                    {
                        "Field": " › ".join(path) or "Value",
                        "Value": "— (empty list)",
                    }
                )
                return
            for index, child in enumerate(value, start=1):
                walk(child, [*path, f"Item {index}"])
            return
        if value is None:
            rendered: Any = "—"
        elif isinstance(value, (bool, np.bool_)):
            rendered = "Yes" if bool(value) else "No"
        else:
            # A single string-valued column avoids mixed-type Arrow coercion in
            # Streamlit while ``str`` keeps the complete scalar representation.
            rendered = str(value)
        rows.append(
            {
                "Field": " › ".join(path) or "Value",
                "Value": rendered,
            }
        )

    walk(payload, [_readable_field_label(root_label)] if root_label else [])
    return rows


def _quantile_rows(payload: Any) -> list[dict[str, Any]]:
    """Return the standard OC across-truth quantiles as one readable table."""

    if not isinstance(payload, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for metric, values in payload.items():
        if not isinstance(values, Mapping):
            continue
        rows.append(
            {
                "Metric": _readable_field_label(metric),
                "5th percentile": values.get("q05"),
                "Median": values.get("median"),
                "95th percentile": values.get("q95"),
            }
        )
    return rows


def _export_overview_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Summarize a downloadable payload without rendering raw JSON in the page."""

    rows: list[dict[str, Any]] = []
    for section, value in payload.items():
        if isinstance(value, Mapping):
            contents = f"{len(value)} field(s)"
        elif isinstance(value, (list, tuple)):
            contents = f"{len(value)} item(s)"
        elif value is None:
            contents = "Not set"
        elif isinstance(value, (bool, np.bool_)):
            contents = "Yes" if bool(value) else "No"
        else:
            contents = str(value)
        rows.append(
            {
                "Export section": _readable_field_label(section),
                "Contents": contents,
            }
        )
    return rows


def _normalized_library_source_dir(value: Any) -> str | None:
    """Return one stable folder identity for UI session comparisons."""

    if not isinstance(value, str) or not value.strip():
        return None
    return str(Path(value.strip()).expanduser().resolve())


def _queue_library_source_dir(
    session_state: MutableMapping[str, Any], selected: str
) -> None:
    """Queue a native-dialog selection without mutating an instantiated widget."""

    session_state[_PENDING_LIBRARY_SOURCE_KEY] = selected


def _apply_pending_library_source_dir(
    session_state: MutableMapping[str, Any],
) -> None:
    """Apply a queued folder before the source text widget is instantiated."""

    selected = session_state.pop(_PENDING_LIBRARY_SOURCE_KEY, None)
    if not isinstance(selected, str) or not selected.strip():
        return
    previous = _normalized_library_source_dir(session_state.get(_LIBRARY_SOURCE_KEY))
    current = _normalized_library_source_dir(selected)
    session_state[_LIBRARY_SOURCE_KEY] = selected
    if previous != current:
        session_state.pop(_SCANNED_LIBRARY_SOURCE_KEY, None)


def _has_current_staging_scan(
    session_state: Mapping[str, Any], source_dir: Any
) -> bool:
    """Return whether this browser session scanned the currently selected folder."""

    current = _normalized_library_source_dir(source_dir)
    return bool(current and session_state.get(_SCANNED_LIBRARY_SOURCE_KEY) == current)


def _document_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in manifest.get("documents", []):
        source = str(document.get("source_path") or "")
        display_name = Path(source).name if source else "Untitled document"
        display_name = display_name.replace("_", " ")
        extension = str(document.get("extension") or Path(source).suffix or "")
        rows.append(
            {
                "Document": display_name,
                "Type": extension.lstrip(".").upper() or "—",
                "Status": str(document.get("status") or "—").replace("_", " "),
                "Text length": document.get("normalized_text_chars"),
                "Note": str(document.get("rejection_code") or "—").replace("_", " "),
            }
        )
    return rows




def render_target_trial_page(
    st: Any,
    prior_options: list[tuple[str, str, str]],
    *,
    use_manual_similarity_defaults: bool = True,
    historical_article_preview: Mapping[str, Any] | None = None,
    article_review_context_fingerprint: str | None = None,
) -> PredictionRequest | None:
    """Render target pre-trial inputs and return a request on submission."""

    prior_names = [name for name, _label, _description in prior_options]
    prior_labels = {name: label for name, label, _description in prior_options}
    prior_descriptions = {
        name: description for name, _label, description in prior_options
    }
    configured_default_prior = load_default_prior_name()
    default_prior = (
        configured_default_prior
        if configured_default_prior in prior_names
        else prior_names[0]
    )
    default_weights, default_threshold = load_default_similarity_parameters(
        use_manual=use_manual_similarity_defaults
    )
    restored_draft = None
    draft_load_error: str | None = None
    try:
        restored_draft = load_target_trial_draft(prior_names=prior_names)
    except ValueError as error:
        draft_load_error = str(error)
    restored_count = 0
    if restored_draft is not None:
        for key, value in restored_draft.values.items():
            if key not in st.session_state:
                st.session_state[key] = value
                restored_count += 1
    target_defaults = target_trial_default_values(
        default_prior=default_prior,
        default_weights=default_weights,
        default_threshold=default_threshold,
    )
    for key, value in target_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Revalidate an already-open session too. This protects widgets when an app
    # update removes a prior or option while the old value remains in state.
    valid_session_values = sanitize_target_trial_values(
        {
            key: st.session_state[key]
            for key in TARGET_TRIAL_WIDGET_KEYS
            if key in st.session_state
        },
        prior_names=prior_names,
    )
    for key in TARGET_TRIAL_WIDGET_KEYS:
        if key not in valid_session_values:
            st.session_state[key] = target_defaults[key]
    if (
        restored_draft is not None
        and restored_count == len(restored_draft.values)
        and TARGET_TRIAL_DRAFT_FINGERPRINT_KEY not in st.session_state
    ):
        restored_fingerprint = target_trial_values_fingerprint(st.session_state)
        if restored_fingerprint is not None:
            st.session_state[TARGET_TRIAL_DRAFT_FINGERPRINT_KEY] = restored_fingerprint

    st.markdown(
        '<p class="tox-kicker">Step 2 of 4 · Configure</p>', unsafe_allow_html=True
    )
    st.markdown("## Target trial")
    st.markdown(
        '<p class="tox-lead">Configure the pre-trial profile, required similarity '
        "weights, and Bayesian prior. This is an exploratory historical estimate, "
        "not a clinical dose recommendation.</p>",
        unsafe_allow_html=True,
    )
    draft_cols = st.columns([4, 1])
    with draft_cols[0]:
        st.caption(
            "Target Trial inputs are saved only on this computer and expire "
            "1 day (24 hours) after the last change. API keys are never included."
        )
    with draft_cols[1]:
        if st.button(
            "Clear saved inputs",
            key="clear_target_trial_draft",
            use_container_width=True,
            help=(
                "Clear this form, its local 1-day draft, and prediction/OC "
                "results derived from the old target. The literature library and "
                "API key are not changed."
            ),
        ):
            try:
                clear_target_trial_draft()
            except OSError as error:
                st.error(f"Could not remove the saved Target Trial draft: {error}")
                return None
            clear_target_trial_state(st.session_state)
            st.session_state[TARGET_TRIAL_DRAFT_SKIP_SAVE_KEY] = True
            st.session_state["_target_trial_draft_notice"] = (
                "Target Trial inputs and the saved local draft were cleared."
            )
            st.rerun()
    notice = st.session_state.pop("_target_trial_draft_notice", None)
    if isinstance(notice, str):
        st.success(notice)
    elif restored_count:
        saved_time = restored_draft.saved_at_utc.astimezone().strftime("%H:%M:%S")
        st.success(f"Restored the local Target Trial draft saved at {saved_time}.")
    if draft_load_error:
        st.warning(
            f"The previous Target Trial draft could not be restored: {draft_load_error}"
        )
    early_save_error = st.session_state.pop("_target_trial_draft_save_error", None)
    if isinstance(early_save_error, str):
        st.warning(f"Could not save the local Target Trial draft: {early_save_error}")

    with st.container(border=True):
        st.markdown("#### 1. Target identity")
        st.caption("Start with the minimum information needed to define the curve.")
        c1, c2 = st.columns(2)
        with c1:
            drug = st.text_input(
                "Primary curve-axis drug *",
                placeholder="e.g., PF-03446962; add partners in Drug configurations below",
                key="target_drug",
            )
            dose_unit = st.text_input(
                "Dose unit *",
                placeholder="e.g., mg/kg, mg, mg/m2, mg/day",
                key="target_dose_unit",
            )
        with c2:
            try:
                native_rows_for_type=json.loads(st.session_state.get('native_target_rows_json','[]'))
            except (ValueError,TypeError):native_rows_for_type=[]
            selected_scenario=st.session_state.get('native_target_scenario','Main')
            antitumor_names={r.get('drug') for r in native_rows_for_type if r.get('drug')
                and (r.get('scenario') or 'Main')==selected_scenario
                and r.get('presence')=='present'
                and r.get('role') in {'escalated','fixed_backbone','combination_partner','maintenance','radiation'}}
            trial_type='combination' if len(antitumor_names)>1 else 'single'
            st.caption('Treatment type: '+trial_type+' · derived from the drug configuration below')
            bayesian_prior = st.selectbox(
                "Bayesian historical prior *",
                options=prior_names,
                format_func=lambda name: prior_labels[name],
                key="bayesian_prior_select",
                help=(
                    "Choose a broad weakly informative prior before seeing "
                    "target-trial outcomes. The catalog prevents narrow "
                    "historical-curve priors."
                ),
            )
            st.caption(prior_descriptions.get(bayesian_prior, ""))
        planned_doses = st.text_area(
            "Planned doses (optional)",
            placeholder=(
                "Leave blank for an automatic historical dose grid, or enter "
                "positive values such as: 0.1, 0.3, 1, 3"
            ),
            height=80,
            key="planned_doses_text",
        )

    with st.container(border=True):
        title_col, reset_col = st.columns([4, 1])
        with title_col:
            st.markdown("#### 2. Similarity weights")
            st.caption(
                "Use non-negative decimals (0.2) or fractions (1/5). "
                "The exact total must equal 1."
            )
        with reset_col:
            st.write("")
            if st.button(
                "Restore defaults",
                key="restore_similarity_defaults",
                use_container_width=True,
                help=(
                    "Reload legacy manual weights."
                    if use_manual_similarity_defaults
                    else "Reload built-in v2 weights; human-evaluation data is not used."
                ),
            ):
                for field in SIMILARITY_FIELD_ORDER:
                    st.session_state[f"sim_weight_text_{field}"] = (
                        format_similarity_weight_input(
                            float(default_weights.get(field, 0.0))
                        )
                    )
                st.session_state["similarity_threshold_input"] = float(
                    default_threshold
                )
                st.session_state["embedding_backend_select"] = "local"
                st.session_state["embedding_model_select"] = (
                    "text-embedding-3-small"
                )
                st.rerun()

        weight_text_values: dict[str, str] = {}
        rows = [
            SIMILARITY_FIELD_ORDER[index : index + 3]
            for index in range(0, len(SIMILARITY_FIELD_ORDER), 3)
        ]
        for row_fields in rows:
            columns = st.columns(3)
            for column, field in zip(columns, row_fields):
                with column:
                    weight_text_values[field] = st.text_input(
                        SIMILARITY_FIELD_LABELS[field],
                        placeholder="0.2 or 1/5",
                        key=f"sim_weight_text_{field}",
                    )
        weight_values: dict[str, float] = {}
        weight_total_exact = None
        weight_input_error: str | None = None
        try:
            weight_values, weight_total_exact = parse_similarity_weight_inputs(
                weight_text_values
            )
        except ValueError as error:
            weight_input_error = str(error)
            st.warning(weight_input_error)

        status_col, threshold_col = st.columns([2, 1])
        with status_col:
            if weight_total_exact is not None:
                if weight_total_exact == 1:
                    st.success("Weights ready · exact total = 1")
                else:
                    st.warning(
                        f"Current total: {float(weight_total_exact):.8g}. "
                        "Adjust the weights to exactly 1."
                    )
        with threshold_col:
            similarity_threshold = st.number_input(
                "Similarity threshold *",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.3f",
                help="Articles below this inclusive raw-similarity cutoff are excluded.",
                key="similarity_threshold_input",
            )

        st.divider()
        st.markdown("##### Semantic text comparison")
        st.caption(
            "This choice affects only the free-text schema/regimen and eligibility "
            "similarity fields. The other 11 indicators keep their structured "
            "comparison rules."
        )
        embedding_backend = st.radio(
            "Text similarity method *",
            options=tuple(_EMBEDDING_BACKEND_LABELS),
            format_func=lambda value: _EMBEDDING_BACKEND_LABELS[value],
            horizontal=True,
            key="embedding_backend_select",
            help=(
                "Local text cosine compares words and adjacent word pairs. OpenAI "
                "embeddings can match differently worded text with similar meaning."
            ),
        )
        embedding_model = str(
            st.session_state.get(
                "embedding_model_select", "text-embedding-3-small"
            )
        )
        if embedding_backend == "openai":
            embedding_model = st.selectbox(
                "OpenAI embedding model *",
                options=tuple(_OPENAI_EMBEDDING_MODEL_LABELS),
                format_func=lambda value: _OPENAI_EMBEDDING_MODEL_LABELS[value],
                key="embedding_model_select",
                help=(
                    "Historical article vectors are cached by model and source-text "
                    "hash. Later runs normally embed only the new target text."
                ),
            )
            if os.environ.get("OPENAI_API_KEY"):
                st.success(
                    "OPENAI_API_KEY is available. Uncached historical text and the "
                    "current target text will be sent to the Embeddings API."
                )
            else:
                st.warning(
                    "OPENAI_API_KEY is not available to this web process. Configure "
                    "it before running an OpenAI-embedding prediction."
                )
        else:
            st.info(
                "Local text cosine runs entirely on this computer and sends no similarity "
                "text to OpenAI."
            )
        st.caption(
            "TF-IDF and semantic-embedding scores are on different empirical scales. "
            "Validate the similarity threshold separately for each method."
        )

        st.divider()
        st.markdown("##### Historical article inclusion")
        st.caption(
            "Choose whether the system includes every eligible article automatically "
            "or first presents a ranked shortlist for human review. Manual review "
            "changes inclusion only: it does not let a reviewer overwrite similarity, "
            "reliability, likelihood power, or ESS calculations."
        )
        historical_article_selection_mode = st.radio(
            "How should articles be included? *",
            options=("automatic", "manual"),
            format_func=lambda value: (
                "Use the system's eligible articles and calculated weights"
                if value == "automatic"
                else "Review ranked articles and choose which ones to include"
            ),
            horizontal=True,
            key="historical_article_selection_mode",
        )
        historical_article_candidate_basis = st.session_state.get(
            "historical_article_candidate_basis", "threshold"
        )
        historical_article_top_n = int(
            st.session_state.get("historical_article_top_n", 40)
        )
        if historical_article_selection_mode == "manual":
            historical_article_candidate_basis = st.radio(
                "Which articles should be offered for review? *",
                options=("threshold", "top_n"),
                format_func=lambda value: (
                    f"All articles with system similarity S ≥ {float(similarity_threshold):.3f}"
                    if value == "threshold"
                    else "The top N most similar articles"
                ),
                horizontal=True,
                key="historical_article_candidate_basis",
                help=(
                    "Threshold and top-N are alternative shortlist rules. Top-N is "
                    "ranked across unique articles and can include articles below the "
                    "displayed threshold."
                ),
            )
            if historical_article_candidate_basis == "top_n":
                historical_article_top_n = int(
                    st.number_input(
                        "Maximum articles offered for review",
                        min_value=1,
                        max_value=200,
                        step=1,
                        key="historical_article_top_n",
                        help=(
                            "Default 40. The actual list can be shorter when the "
                            "library has fewer curve-eligible articles."
                        ),
                    )
                )
            st.info(
                "Candidate ranking is calculated locally and does not call the LLM. "
                "The displayed similarity is a ranking score, not necessarily the "
                "final method-specific curve weight."
            )

        st.divider()
        st.markdown("##### Curve synthesis method")
        selected_curve_method = st.radio(
            "Method used for prediction and OC simulation *",
            options=(
                "article_line_weighted",
                "article_posterior_draw_pool",
                "power_likelihood_random_intercept_slope",
            ),
            format_func=lambda key: _CURVE_METHOD_LABELS.get(key, key),
            horizontal=True,
            key="target_curve_method",
            help=(
                "Choose the curve-construction method before fitting. Method 3 "
                "has additional borrowing and random-effect controls."
            ),
        )

        power_likelihood_enabled = False
        power_likelihood_random_effects_enabled = False
        power_likelihood_random_intercept_slope_enabled = True
        power_likelihood_borrowing_strength = float(
            st.session_state["power_likelihood_borrowing_strength"]
        )
        power_likelihood_ess_cap = float(
            st.session_state["power_likelihood_ess_cap"]
        )
        random_intercept_scale_prior_mean = float(
            st.session_state["random_intercept_scale_prior_mean"]
        )
        random_intercept_scale_prior_sd = float(
            st.session_state["random_intercept_scale_prior_sd"]
        )
        random_slope_scale_prior_mean = float(
            st.session_state["random_slope_scale_prior_mean"]
        )
        random_slope_scale_prior_sd = float(
            st.session_state["random_slope_scale_prior_sd"]
        )
        if selected_curve_method == "power_likelihood_random_intercept_slope":
            with st.expander("Method 3 power-likelihood controls", expanded=True):
                st.caption(
                    "Method 3 allows both the toxicity level and the dose-response "
                    "slope to vary independently between study groups while "
                    "preserving monotone curves."
                )
                power_cols = st.columns(2)
                with power_cols[0]:
                    power_likelihood_borrowing_strength = st.number_input(
                        "Overall borrowing strength",
                        min_value=0.01,
                        max_value=1.0,
                        step=0.05,
                        format="%.2f",
                        key="power_likelihood_borrowing_strength",
                        help=(
                            "Multiplies every outcome-free similarity/reliability "
                            "power. It is an information discount, not a probability "
                            "that a paper is correct."
                        ),
                    )
                with power_cols[1]:
                    power_likelihood_ess_cap = st.number_input(
                        "Historical effective-N cap (ESS max)",
                        min_value=0.1,
                        step=1.0,
                        format="%.1f",
                        key="power_likelihood_ess_cap",
                        help=(
                            "Upper limit on borrowed historical information in "
                            "patient-equivalent units. Powers are scaled down "
                            "uniformly only when the preliminary ESS exceeds it."
                        ),
                    )
                st.info(
                    f"This prediction will use ESS max = "
                    f"{float(power_likelihood_ess_cap):.1f}. Powers never exceed "
                    "1 and are only scaled down to satisfy this cap."
                )
                prior_cols = st.columns(4)
                with prior_cols[0]:
                    random_intercept_scale_prior_mean = st.number_input(
                        "Intercept-scale latent mean",
                        step=0.1,
                        key="random_intercept_scale_prior_mean",
                    )
                with prior_cols[1]:
                    random_intercept_scale_prior_sd = st.number_input(
                        "Intercept-scale latent SD",
                        min_value=0.05,
                        step=0.1,
                        key="random_intercept_scale_prior_sd",
                    )
                with prior_cols[2]:
                    random_slope_scale_prior_mean = st.number_input(
                        "Slope-scale latent mean",
                        step=0.1,
                        key="random_slope_scale_prior_mean",
                    )
                with prior_cols[3]:
                    random_slope_scale_prior_sd = st.number_input(
                        "Slope-scale latent SD",
                        min_value=0.05,
                        step=0.1,
                        key="random_slope_scale_prior_sd",
                    )

    with st.container(border=True):
        st.markdown("#### 3. Optional pre-trial profile")
        st.caption(
            "Add only information known before the trial. Blank fields are omitted, "
            "and seven structured similarity categories are available. Design is retained for denominator inference only."
        )

        st.markdown("##### Study and disease")
        study_cols = st.columns(2)
        with study_cols[0]:
            cancer_type = st.text_input(
                "Cancer type",
                placeholder="e.g., advanced solid tumors; NSCLC",
                help="Separate multiple cancer types with commas, semicolons, or new lines.",
                key="opt_cancer_type",
            )
        with study_cols[1]:
            routes = st.multiselect(
                "Administration route",
                options=ADMIN_ROUTE_OPTIONS,
                placeholder="Select known routes",
                key="opt_routes",
            )
        st.caption(
            "Dosing schedule · frequency, repeat interval, and administration days"
        )
        schedule_cols = st.columns(3)
        with schedule_cols[0]:
            schedule_frequency = st.selectbox(
                "Dosing frequency",
                options=["(blank)", *FREQUENCY_OPTIONS],
                key="opt_schedule_frequency",
            )
        with schedule_cols[1]:
            schedule_cycle_days = st.number_input(
                "Repeat interval (days)",
                min_value=0.0,
                step=1.0,
                format="%.1f",
                help="For example, enter 21 for Q3W. Leave 0 if unknown.",
                key="opt_schedule_cycle_days",
            )
        with schedule_cols[2]:
            schedule_administration_days = st.text_input(
                "Administration days",
                placeholder="e.g., 1 or 1, 8, 15",
                key="opt_schedule_admin_days",
            )
        schema = st.text_area(
            "Reference note (not scored)",
            placeholder="Optional reference note; this free text is not scored.",
            key="opt_schema",
        )

        st.divider()
        st.markdown("##### Eligibility and safety")
        eligibility_cols = st.columns(2)
        with eligibility_cols[0]:
            inclusion = st.text_area(
                "Inclusion criteria",
                placeholder="e.g., Advanced solid tumors; adequate organ function.",
                key="opt_inclusion",
            )
        with eligibility_cols[1]:
            exclusion = st.text_area(
                "Exclusion criteria",
                placeholder="e.g., Uncontrolled cardiovascular disease.",
                key="opt_exclusion",
            )
        dlt_definition = st.text_area(
            "Planned DLT definition",
            placeholder=(
                "One line per definition: Name; grade; duration; comments\n"
                "e.g., hematologic toxicity; >=3; >7 days; treatment-related"
            ),
            help=(
                "Duration is optional. The legacy three-column form "
                "Name; grade; comments is still accepted."
            ),
            key="opt_dlt_definition",
        )
        st.caption("DLT evaluation window")
        eva_cols = st.columns(3)
        with eva_cols[0]:
            dlt_eva_days = st.number_input(
                "Window length (days)",
                min_value=0.0,
                step=1.0,
                format="%.1f",
                help="Prefer an explicit duration when known. Leave 0 if unknown.",
                key="opt_dlt_eva_days",
            )
        with eva_cols[1]:
            dlt_eva_cycles = st.number_input(
                "Window length (cycles)",
                min_value=0.0,
                step=1.0,
                format="%.1f",
                help="Converted only when the actual cycle length is supplied below; otherwise days remain unknown.",
                key="opt_dlt_eva_cycles",
            )
        with eva_cols[2]:
            dlt_eva_text = st.text_input(
                "Window note",
                placeholder="e.g., during the first 6 weeks",
                key="opt_dlt_eva_text",
            )

        st.caption("Age eligibility")
        age_cols = st.columns(4)
        with age_cols[0]:
            age_group = st.selectbox(
                "Age population",
                options=["(blank)", *AGE_GROUP_OPTIONS],
                key="opt_age_group",
            )
        with age_cols[1]:
            age_min = st.number_input(
                "Minimum age",
                min_value=0.0,
                step=1.0,
                placeholder="Unknown",
                key="opt_age_min",
            )
        with age_cols[2]:
            age_max = st.number_input(
                "Maximum age",
                min_value=0.0,
                step=1.0,
                placeholder="Unknown",
                key="opt_age_max",
            )
        with age_cols[3]:
            age_unit = st.selectbox(
                "Age unit",
                options=["years", "months", "days"],
                key="opt_age_unit",
            )
        eligibility_detail_cols = st.columns(2)
        with eligibility_detail_cols[0]:
            gender = st.selectbox(
                "Sex / gender eligibility",
                options=["(blank)", *GENDER_SEX_OPTIONS],
                key="opt_gender",
            )
        with eligibility_detail_cols[1]:
            dose_forms = st.multiselect(
                "Dose form",
                options=DOSE_FORM_OPTIONS,
                placeholder="Select known forms",
                key="opt_dose_forms",
            )

        st.divider()
        st.markdown("##### Biology")
        biology_cols = st.columns(2)
        with biology_cols[0]:
            gene_targets = st.text_input(
                "Gene / molecular target",
                placeholder="e.g., ALK-1, EGFR",
                key="opt_gene_targets",
            )
        with biology_cols[1]:
            mechanism = st.text_area(
                "Mechanism of action",
                placeholder="e.g., anti-ALK-1 monoclonal antibody",
                key="opt_mechanism",
            )

    skip_draft_save = bool(
        st.session_state.pop(TARGET_TRIAL_DRAFT_SKIP_SAVE_KEY, False)
    )
    if skip_draft_save:
        # Clear must keep the file absent until the user makes a real change.
        cleared_fingerprint = target_trial_values_fingerprint(st.session_state)
        if cleared_fingerprint is not None:
            st.session_state[TARGET_TRIAL_DRAFT_FINGERPRINT_KEY] = cleared_fingerprint
    else:
        try:
            save_target_trial_draft_if_changed(st.session_state)
        except (OSError, TypeError, ValueError) as error:
            st.warning(f"Could not save the local Target Trial draft: {error}")

    from phase1_web.target_profile import render as render_native_target
    native_rows, native_scenario = render_native_target(st, drug, dose_unit)

    st.caption("* Required fields · Optional blanks are safely omitted from the query.")
    current_form_fingerprint = target_trial_values_fingerprint(st.session_state)
    preview_current = bool(
        historical_article_selection_mode == "manual"
        and isinstance(historical_article_preview, Mapping)
        and historical_article_preview.get("_ui_form_fingerprint")
        == current_form_fingerprint
        and historical_article_preview.get("_ui_context_fingerprint")
        == article_review_context_fingerprint
    )
    selected_article_ids: tuple[str, ...] = ()
    candidate_snapshot_sha256: str | None = None
    submission_action: str | None = None

    if historical_article_selection_mode == "automatic":
        if st.button(
            "Run toxicity prediction  →", type="primary", use_container_width=True
        ):
            submission_action = "predict"
    elif not preview_current:
        if historical_article_preview:
            st.warning(
                "The target profile, weights, shortlist rule, evidence contract, or "
                "library location changed. Refresh the ranked shortlist before using it."
            )
        if st.button(
            "Prepare ranked article review  →",
            type="primary",
            use_container_width=True,
        ):
            submission_action = "preview"
    else:
        candidates = historical_article_preview.get("candidates") or []
        candidate_snapshot_sha256 = str(
            historical_article_preview.get("candidate_snapshot_sha256") or ""
        ) or None
        st.markdown("### Review historical articles before curve construction")
        before_curve_eligibility = int(
            historical_article_preview.get(
                "candidate_document_count_before_curve_eligibility",
                len(candidates),
            )
            or 0
        )
        curve_ineligible_count = int(
            historical_article_preview.get("curve_ineligible_candidate_count", 0)
            or 0
        )
        basis_label = (
            f"S ≥ {float(historical_article_preview.get('similarity_threshold', 0.0)):.3f}"
            if historical_article_preview.get("candidate_basis") == "threshold"
            else f"top {int(historical_article_preview.get('top_n') or len(candidates))}"
        )
        st.caption(
            f"The {basis_label} rule found {before_curve_eligibility} unique "
            f"article(s); {len(candidates)} can supply a fitted DLT curve line for "
            "the current drug and dose unit and are shown below. Uncheck an "
            "article to remove all of its study units and cohorts before every "
            "curve method is refitted."
        )
        if curve_ineligible_count:
            st.info(
                f"{curve_ineligible_count} shortlisted article(s) were not offered "
                "for inclusion because they contain no usable DLT posterior line "
                "matching the current drug and canonical dose unit."
            )
        if not candidates:
            st.error("No articles are available under this shortlist rule.")
        else:
            editor_rows = candidate_table_rows(candidates)
            editor_key = (
                "historical_article_review_editor_"
                f"{(candidate_snapshot_sha256 or 'preview')[:16]}"
            )
            edited_rows = st.data_editor(
                editor_rows,
                hide_index=True,
                use_container_width=True,
                key=editor_key,
                disabled=[
                    "Rank",
                    "Article",
                    "Title",
                    "System similarity",
                    "Available field weight",
                    "Provisional candidate share",
                    "Study units",
                    "Usable curve lines",
                    "Dose–DLT points",
                    "Observed dose min",
                    "Observed dose max",
                    "Original article",
                ],
                column_config={
                    "Include": st.column_config.CheckboxColumn(
                        "Include", help="Include the whole article in curve construction."
                    ),
                    "System similarity": st.column_config.NumberColumn(
                        "System similarity S", format="%.4f"
                    ),
                    "Available field weight": st.column_config.NumberColumn(
                        "Available evidence weight", format="%.4f"
                    ),
                    "Provisional candidate share": st.column_config.NumberColumn(
                        "Shortlist similarity share", format="%.4f"
                    ),
                    "Usable curve lines": st.column_config.NumberColumn(
                        "Usable curve lines", format="%d"
                    ),
                    "Dose–DLT points": st.column_config.NumberColumn(
                        "Dose–DLT points", format="%d"
                    ),
                    "Observed dose min": st.column_config.NumberColumn(
                        "Observed dose min", format="%.4g"
                    ),
                    "Observed dose max": st.column_config.NumberColumn(
                        "Observed dose max", format="%.4g"
                    ),
                    "Original article": st.column_config.LinkColumn(
                        "PMC article", display_text="Open"
                    ),
                },
            )
            selected_article_ids = selected_document_ids(edited_rows)
            st.caption(
                f"Selected {len(selected_article_ids)} of {len(candidates)} article(s). "
                "Final weights and ESS are recalculated only after this selection."
            )

            candidate_by_id = {
                normalize_document_id(
                    item.get("document_id") or item.get("pmcid")
                ): item
                for item in candidates
            }
            detail_options = list(candidate_by_id)
            detail_key = (
                "historical_article_review_detail_"
                f"{(candidate_snapshot_sha256 or 'preview')[:16]}"
            )
            detail_id = st.selectbox(
                "Inspect one article's source and extracted content",
                options=detail_options,
                format_func=lambda document_id: next(
                    (
                        row["Title"]
                        for row in editor_rows
                        if row["Article"] == document_id
                    ),
                    document_id,
                ),
                key=detail_key,
            )
            detail_candidate = candidate_by_id[detail_id]
            with st.expander("Seven-category similarity details and configuration scores"):
                from phase1_web.article_comparison import detail_rows
                groups=detail_candidate.get("comparison_groups", {})
                st.dataframe([{"Category":SIMILARITY_FIELD_LABELS.get(k,k),"Score":v.get("score")} for k,v in groups.items()],hide_index=True)
                for k,v in groups.items():
                    st.write(SIMILARITY_FIELD_LABELS.get(k,k))
                    st.dataframe(detail_rows(v),hide_index=True)
                st.dataframe([{"Configuration":detail_candidate.get("configuration_keys",{}).get(k,k),"Similarity":v.get("similarity")} for k,v in detail_candidate.get("configuration_scores",{}).items()],hide_index=True)
            detail_source_url = next(
                (
                    row.get("Original article")
                    for row in editor_rows
                    if row.get("Article") == detail_id
                ),
                None,
            )
            detail_cols = st.columns([1, 1, 2])
            with detail_cols[0]:
                if detail_source_url:
                    st.link_button(
                        "Open original article",
                        str(detail_source_url),
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "Original link unavailable",
                        disabled=True,
                        use_container_width=True,
                        key=(
                            "historical_article_review_link_missing_"
                            f"{detail_id}_{(candidate_snapshot_sha256 or '')[:12]}"
                        ),
                    )
            verified_pdf = resolve_verified_pdf(detail_id)
            with detail_cols[1]:
                if verified_pdf.get("status") == "ready":
                    pdf_path = Path(str(verified_pdf["path"]))
                    st.download_button(
                        "Download verified PDF",
                        data=pdf_path.read_bytes(),
                        file_name=str(verified_pdf["filename"]),
                        mime="application/pdf",
                        use_container_width=True,
                        key=(
                            "historical_article_review_pdf_"
                            f"{detail_id}_{(candidate_snapshot_sha256 or '')[:12]}"
                        ),
                    )
                else:
                    st.button(
                        "Local PDF unavailable",
                        disabled=True,
                        use_container_width=True,
                        key=(
                            "historical_article_review_pdf_missing_"
                            f"{detail_id}_{(candidate_snapshot_sha256 or '')[:12]}"
                        ),
                    )
            with detail_cols[2]:
                if verified_pdf.get("status") == "ready":
                    st.caption(
                        f"Verified local PDF · SHA-256 "
                        f"{str(verified_pdf.get('sha256'))[:16]}… · "
                        f"{int(verified_pdf.get('bytes') or 0) / 1_048_576:.1f} MB"
                    )
                else:
                    st.caption(str(verified_pdf.get("reason") or "No local PDF."))

            units = detail_candidate.get("study_units") or []
            if units:
                st.markdown("**System similarity evidence for this article**")
                st.dataframe(
                    [
                        {
                            "Study unit": unit.get("study_unit_id") or "article",
                            "Similarity": unit.get("similarity"),
                            "Available weight": unit.get("available_weight"),
                        }
                        for unit in units
                        if isinstance(unit, Mapping)
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            extraction = load_extracted_content(detail_candidate)
            if isinstance(extraction.get("payload"), Mapping):
                payload = extraction["payload"]
                st.markdown("**Extracted content summary**")
                st.json(extracted_content_summary(payload), expanded=True)
                with st.expander("View complete extracted JSON", expanded=False):
                    st.caption(
                        f"Source: {extraction.get('path')} · SHA-256 "
                        f"{str(extraction.get('sha256'))[:16]}…"
                    )
                    st.json(payload, expanded=False)
                    st.download_button(
                        "Download extracted JSON",
                        data=json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        ).encode("utf-8"),
                        file_name=f"{detail_id}_extracted.json",
                        mime="application/json",
                        key=(
                            "historical_article_review_json_"
                            f"{detail_id}_{(candidate_snapshot_sha256 or '')[:12]}"
                        ),
                    )
            else:
                st.warning(
                    "The extracted JSON could not be opened for this article: "
                    f"{extraction.get('error')}"
                )

        action_cols = st.columns([1, 3])
        with action_cols[0]:
            if st.button(
                "Refresh ranking",
                use_container_width=True,
                key="historical_article_review_refresh",
            ):
                submission_action = "preview"
        with action_cols[1]:
            if st.button(
                f"Generate curve using {len(selected_article_ids)} selected article(s)  →",
                type="primary",
                use_container_width=True,
                disabled=not selected_article_ids,
                key="historical_article_review_confirm",
            ):
                submission_action = "predict"

    submitted = submission_action is not None
    if not submitted:
        return None
    try:
        if weight_input_error:
            raise ValueError(weight_input_error)
        if weight_total_exact != 1:
            raise ValueError(
                "Similarity weights must sum to exactly 1 "
                f"(current sum = {float(weight_total_exact or 0):.8g})."
            )
        field_weights, threshold = validate_similarity_inputs(
            weight_values, similarity_threshold
        )
        if embedding_backend == "openai" and not os.environ.get(
            "OPENAI_API_KEY"
        ):
            raise ValueError(
                "OpenAI semantic embedding requires OPENAI_API_KEY in the web "
                "process environment. Choose Local text cosine or configure the key and "
                "restart the web app."
            )
        query = build_target_query(
            drug=drug,
            trial_type=trial_type,
            dose_unit=dose_unit,
            planned_doses_text=planned_doses,
            optional_fields=build_optional_query_fields(
                cancer_type=cancer_type,
                routes=routes,
                schema=schema,
                inclusion=inclusion,
                exclusion=exclusion,
                dlt_definition=dlt_definition,
                mechanism=mechanism,
                gene_targets=gene_targets,
                age_group=age_group,
                age_min=age_min,
                age_max=age_max,
                age_unit=age_unit,
                gender=gender,
                dose_forms=dose_forms,
                schedule_frequency=schedule_frequency,
                schedule_cycle_days=(
                    schedule_cycle_days if schedule_cycle_days > 0 else None
                ),
                schedule_administration_days=schedule_administration_days,
                dlt_eva_days=dlt_eva_days if dlt_eva_days > 0 else None,
                dlt_eva_cycles=dlt_eva_cycles if dlt_eva_cycles > 0 else None,
                dlt_eva_text=dlt_eva_text,
            ),
        )
        from phase1_web.target_profile import apply_extras
        query = apply_extras(query, native_rows, native_scenario, st.session_state)
    except ValueError as error:
        st.error(str(error))
        return None
    if (
        historical_article_selection_mode == "manual"
        and submission_action == "predict"
        and not selected_article_ids
    ):
        st.error("Select at least one historical article before generating a curve.")
        return None
    return PredictionRequest(
        query=query,
        bayesian_prior=bayesian_prior,
        field_weights=field_weights,
        similarity_threshold=threshold,
        curve_method=str(selected_curve_method),
        embedding_backend=str(embedding_backend),
        embedding_model=str(embedding_model),
        power_likelihood_enabled=bool(power_likelihood_enabled),
        power_likelihood_random_effects_enabled=bool(
            power_likelihood_random_effects_enabled
        ),
        power_likelihood_random_intercept_slope_enabled=bool(
            power_likelihood_random_intercept_slope_enabled
        ),
        power_likelihood_borrowing_strength=float(power_likelihood_borrowing_strength),
        power_likelihood_ess_cap=float(power_likelihood_ess_cap),
        random_intercept_scale_prior_mean=float(random_intercept_scale_prior_mean),
        random_intercept_scale_prior_sd=float(random_intercept_scale_prior_sd),
        random_slope_scale_prior_mean=float(random_slope_scale_prior_mean),
        random_slope_scale_prior_sd=float(random_slope_scale_prior_sd),
        historical_article_selection_mode=str(
            historical_article_selection_mode
        ),
        historical_article_candidate_basis=str(
            historical_article_candidate_basis
        ),
        historical_article_top_n=int(historical_article_top_n),
        historical_article_ids=selected_article_ids,
        historical_article_candidate_snapshot_sha256=(
            candidate_snapshot_sha256
        ),
        submission_action=str(submission_action),
    )


def _curve_figure(
    curve_result: Mapping[str, Any],
    method_key: str = "article_line_weighted",
) -> Any:
    """Create a compact plot of the fitted mean and its reported 95% interval."""

    import matplotlib.pyplot as plt
    from phase1_prediction.reference.bayesian_toxicity import annotate_figure_with_prior_panel

    method = curve_result.get("toxicity_curve", {}).get(method_key, {})
    points = [
        point
        for point in method.get("curve_points", [])
        if point.get("dlt_probability") is not None
    ]
    figure, axis = plt.subplots(figsize=(14, 5))
    if not points:
        axis.text(
            0.5, 0.5, "No supported historical curve points", ha="center", va="center"
        )
        axis.set_axis_off()
        return figure
    doses = [point["dose"] for point in points]
    probabilities = [point["dlt_probability"] for point in points]
    intervals = [point.get("posterior_credible_interval_95", {}) for point in points]
    curve_colors = {
        "article_line_weighted": "#0f6e6e",
        "article_posterior_draw_pool": "#2f6da1",
        "power_likelihood": "#b45f06",
        "power_likelihood_random_effects": "#6b4fa1",
        "power_likelihood_random_intercept_slope": "#6b4fa1",
    }
    curve_color = curve_colors.get(method_key, "#444444")
    is_article_posterior_mixture = method_key == "article_posterior_draw_pool"
    axis.plot(
        doses,
        probabilities,
        color=curve_color,
        linewidth=2.2,
        label=(
            "Finite-mixture mean" if is_article_posterior_mixture else "Posterior mean"
        ),
    )
    axis.scatter(doses, probabilities, color=curve_color, s=22)
    if all(
        interval.get("low") is not None and interval.get("high") is not None
        for interval in intervals
    ):
        axis.fill_between(
            doses,
            [interval["low"] for interval in intervals],
            [interval["high"] for interval in intervals],
            color=curve_color,
            alpha=0.14,
            label=(
                "95% similarity-weighted mixture interval"
                if is_article_posterior_mixture
                else "95% credible interval"
            ),
        )
    extrapolated = [
        point
        for point in points
        if float(
            point.get(
                "extrapolation_weight_fraction",
                point.get("extrapolation_power_fraction", 0),
            )
        )
        > 0
    ]
    if extrapolated:
        axis.scatter(
            [point["dose"] for point in extrapolated],
            [point["dlt_probability"] for point in extrapolated],
            color="#b45f06",
            marker="x",
            s=42,
            label="Contains extrapolated historical weight",
        )
    unit = curve_result.get("query", {}).get("target_unit", "")
    axis.set_xlabel(f"Dose ({unit})")
    axis.set_ylabel("Predicted DLT probability")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.18)
    axis.legend(loc="best")
    prior_name = (curve_result.get("bayesian_curve_parameters", {}) or {}).get(
        "prior_name"
    )
    annotate_figure_with_prior_panel(figure, prior_name)
    return figure


def _historical_rows(curve_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = curve_result.get("all_scored_articles")
    if not isinstance(records, list):
        records = curve_result.get("all_scored_study_units", [])
    return [
        {
            "pmcid": article.get("pmcid"),
            "study unit": article.get("study_unit_id"),
            "similarity": article.get("similarity"),
            "curve_weight": article.get("curve_similarity_weight"),
            "available_weight": article.get("available_weight"),
        }
        for article in records
        if isinstance(article, Mapping)
    ]


def _historical_line_rows(curve_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    method = curve_result.get("toxicity_curve", {}).get("article_line_weighted", {})
    rows: list[dict[str, Any]] = []
    for line in method.get("historical_lines", []):
        row = {
            "pmcid": line.get("pmcid"),
            "drug": line.get("drug_name"),
            "arm/cohort": line.get("arm_or_cohort"),
            "line type": line.get("line_type"),
            "similarity": line.get("similarity"),
            "segment weight": line.get("segment_similarity_weight"),
            "reliability": line.get("reliability_weight"),
            "observed dose min": line.get("observed_dose_min"),
            "observed dose max": line.get("observed_dose_max"),
        }
        if line.get("evidence_support_max") is not None:
            row["evidence support max"] = line.get("evidence_support_max")
            row["evidence support policy"] = line.get("evidence_support_policy")
        else:
            row["MAD cap"] = line.get("mad_cap")
        rows.append(row)
    return rows


def _power_likelihood_rows(
    curve_result: Mapping[str, Any],
    method_key: str = "power_likelihood",
) -> list[dict[str, Any]]:
    method = curve_result.get("toxicity_curve", {}).get(method_key, {})
    return [
        {
            "pmcid": line.get("pmcid") or line.get("document_id"),
            "study unit": line.get("study_unit_id"),
            "series": line.get("escalation_series_id"),
            "similarity": line.get("similarity"),
            "reliability": line.get("reliability_weight"),
            "raw power": line.get("raw_likelihood_power"),
            "final power": line.get("likelihood_power"),
            "source N": line.get("source_patient_count"),
            "effective N": line.get("effective_patient_count"),
            "denominator role": ", ".join(line.get("denominator_roles") or []),
        }
        for line in method.get("historical_lines", [])
        if isinstance(line, Mapping)
    ]


def _posterior_mixture_article_rows(
    curve_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Render the document-level quotas of the posterior-component mixture."""

    method = curve_result.get("toxicity_curve", {}).get(
        "article_posterior_draw_pool", {}
    )
    diagnostics = method.get("mixture_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return []
    return [
        {
            "article": allocation.get("pmcid") or allocation.get("document_id"),
            "article similarity": allocation.get("unnormalized_weight"),
            "normalized article weight": allocation.get("normalized_weight"),
            "ideal curves": allocation.get("ideal_curve_count"),
            "allocated curves": allocation.get("allocated_curve_count"),
            "realized curve fraction": allocation.get("realized_curve_fraction"),
            "eligible components": allocation.get("eligible_source_count"),
            "sampled components": allocation.get("allocated_source_count"),
        }
        for allocation in diagnostics.get("article_allocations", [])
        if isinstance(allocation, Mapping)
    ]


def _posterior_mixture_source_rows(
    curve_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Render component weights and exact finite-sample allocations."""

    method = curve_result.get("toxicity_curve", {}).get(
        "article_posterior_draw_pool", {}
    )
    diagnostics = method.get("mixture_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return []
    return [
        {
            "article": allocation.get("pmcid") or allocation.get("document_id"),
            "study unit": allocation.get("study_unit_id"),
            "series": allocation.get("escalation_series_id"),
            "arm/cohort": allocation.get("arm_or_cohort"),
            "component similarity": allocation.get("similarity_weight"),
            "reliability": allocation.get("reliability_weight"),
            "theoretical mixture weight": allocation.get("normalized_weight"),
            "allocated curves": allocation.get("allocated_curve_count"),
            "realized curve fraction": allocation.get("realized_curve_fraction"),
            "available posterior draws": allocation.get(
                "available_posterior_draw_count"
            ),
            "sampled with replacement": allocation.get("sampled_with_replacement"),
            "dose support maximum": allocation.get("support_max"),
        }
        for allocation in diagnostics.get("source_allocations", [])
        if isinstance(allocation, Mapping)
    ]


def render_prediction_results_page(
    st: Any,
    curve_result: Mapping[str, Any] | None,
    probability_lookup: Callable[..., Mapping[str, Any]],
) -> None:
    """Render prediction outputs, historical details, JSON, and diagnostics."""

    st.markdown(
        '<p class="tox-kicker">Step 3 of 4 · Review</p>', unsafe_allow_html=True
    )
    st.markdown("## Prediction results")
    if curve_result is None:
        with st.container(border=True):
            st.markdown("### No prediction yet")
            st.caption(
                "Configure a target trial first. Your result, evidence support, "
                "dose-support diagnostics, and downloadable JSON will appear here."
            )
            if st.button(
                "Go to Target Trial  →", type="primary", use_container_width=True
            ):
                st.session_state["ui_page"] = "Target Trial"
                st.rerun()
        return
    curve_methods = curve_result.get("toxicity_curve", {})
    if not isinstance(curve_methods, Mapping):
        curve_methods = {}
    method_options = [
        key
        for key in _CURVE_METHOD_LABELS
        if isinstance(curve_methods.get(key), Mapping)
    ]
    if not method_options:
        method_options = ["article_line_weighted"]
    selected_method_key = str(
        curve_result.get("selected_curve_method")
        or st.session_state.get("target_curve_method")
        or method_options[0]
    )
    if selected_method_key not in method_options:
        selected_method_key = method_options[0]
    st.markdown("#### Curve synthesis method")
    st.info(_CURVE_METHOD_LABELS.get(selected_method_key, selected_method_key))
    method = curve_methods.get(selected_method_key, {})
    if method.get("status") != "fitted":
        st.warning(
            f"{_CURVE_METHOD_LABELS.get(selected_method_key, selected_method_key)} "
            f"status: {method.get('status', 'unavailable')}"
        )
    if selected_method_key == "power_likelihood":
        st.info(
            "This model assumes all powered historical lines update one shared curve. "
            "Its likelihood powers are information discounts, not probabilities that "
            "a publication is correct. Review the effective-N and conflict diagnostics."
        )
    if selected_method_key == "article_posterior_draw_pool":
        st.info(
            "This is a similarity-weighted finite mixture of historical posteriors. "
            "Each draw copies one complete curve from one article component; it does "
            "not average different articles within a curve and does not multiply or "
            "power their likelihoods. Its interval retains disagreement between "
            "historical sources, but is conditional on the selected heuristic "
            "similarity weights; it is not a calibrated model-probability interval "
            "or formal Bayesian model averaging."
        )
    if selected_method_key == "power_likelihood_random_effects":
        st.info(
            "This curve adds one random intercept per historical study unit. Its main "
            "prediction includes a newly drawn target-study shift, so its interval "
            "includes both parameter uncertainty and between-study variation."
        )
    if selected_method_key == "power_likelihood_random_intercept_slope":
        st.info(
            "Method 3 fits independent study-group random intercepts and latent "
            "slopes. Each target posterior curve draws one new intercept and one "
            "new slope effect and shares them across every dose."
        )

    query = curve_result.get("query", {})
    with st.container(border=True):
        st.caption("Prediction context")
        top = st.columns(4)
        top[0].metric("Drug", query.get("toxicity_curve_drug", "—"))
        top[1].metric("Trial type", query.get("trial_type", "—"))
        top[2].metric("Dose unit", query.get("target_unit", "—"))
        top[3].metric(
            "Bayesian prior",
            (curve_result.get("bayesian_curve_parameters") or {}).get(
                "prior_name", "—"
            ),
        )
        embedding_backend = str(
            curve_result.get("embedding_backend_used") or "local"
        )
        embedding_model = curve_result.get("embedding_model")
        st.caption(
            "Free-text similarity: "
            + (
                f"OpenAI semantic embedding · {embedding_model}"
                if embedding_backend == "openai"
                else "Local text cosine · no API call"
            )
        )

    curve_tab, historical_tab, diagnostics_tab, json_tab = st.tabs(
        ["Curve", "Evidence", "Dose support", "Export"]
    )
    with curve_tab:
        if selected_method_key == "article_posterior_draw_pool":
            st.caption(
                "Finite-mixture mean and 95% similarity-weighted mixture interval "
                "across the requested dose grid. The interval is conditional on the "
                "chosen article-similarity weights."
            )
        else:
            st.caption(
                "Posterior mean and 95% credible interval across the requested dose grid."
            )
        figure = _curve_figure(curve_result, selected_method_key)
        st.pyplot(figure, clear_figure=True)
        points = [
            {
                "dose": point.get("dose"),
                "unit": point.get("unit"),
                "DLT probability": point.get("dlt_probability"),
                "95% interval low": point.get("posterior_credible_interval_95", {}).get(
                    "low"
                ),
                "95% interval high": point.get(
                    "posterior_credible_interval_95", {}
                ).get("high"),
                "prediction target": point.get("prediction_target"),
                "contributors": point.get("contributor_count"),
                "extrapolation weight": point.get(
                    "extrapolation_weight_fraction",
                    point.get("extrapolation_power_fraction"),
                ),
            }
            for point in method.get("curve_points", [])
        ]
        st.dataframe(points, use_container_width=True, hide_index=True)
        fitted_comparison_methods = {
            key: value
            for key, value in curve_methods.items()
            if key in _CURVE_METHOD_LABELS
            and isinstance(value, Mapping)
            and value.get("status") == "fitted"
        }
        if len(fitted_comparison_methods) > 1:
            probabilities_by_method = {
                key: {
                    float(point["dose"]): point.get("dlt_probability")
                    for point in value.get("curve_points", [])
                    if point.get("dose") is not None
                }
                for key, value in fitted_comparison_methods.items()
            }
            all_doses = sorted(
                {dose for values in probabilities_by_method.values() for dose in values}
            )
            comparison = []
            for dose in all_doses:
                row = {"dose": dose}
                for key, values in probabilities_by_method.items():
                    row[_CURVE_METHOD_LABELS[key]] = values.get(dose)
                comparison.append(row)
            with st.expander("Compare all fitted curve methods"):
                st.dataframe(comparison, use_container_width=True, hide_index=True)
        with st.container(border=True):
            st.markdown("#### Check a specific dose")
            st.caption("Evaluate any positive dose using the selected posterior curve.")
            unit = str(query.get("target_unit", ""))
            default_dose = float((query.get("planned_doses") or [1.0])[0])
            lookup_cols = st.columns([3, 1])
            with lookup_cols[0]:
                arbitrary_dose = st.number_input(
                    f"Dose ({unit})",
                    min_value=0.0000001,
                    value=default_dose,
                    format="%.8g",
                )
            with lookup_cols[1]:
                st.write("")
                if st.button(
                    "Calculate",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        lookup = probability_lookup(
                            curve_result,
                            float(arbitrary_dose),
                            method_key=selected_method_key,
                        )
                        probability = lookup.get("dlt_probability")
                        if probability is not None:
                            st.success(
                                f"Predicted DLT probability: {float(probability):.1%}"
                            )
                        with st.expander("Calculation details"):
                            st.dataframe(
                                _readable_diagnostic_rows(dict(lookup)),
                                use_container_width=True,
                                hide_index=True,
                            )
                    except Exception as error:
                        st.error(f"Probability lookup failed: {error}")

    with historical_tab:
        selection_audit = curve_result.get("historical_article_selection") or {}
        if isinstance(selection_audit, Mapping) and selection_audit:
            mode = str(selection_audit.get("mode") or "automatic")
            basis = str(selection_audit.get("candidate_basis") or "threshold")
            st.markdown("#### Article-inclusion decision")
            selection_metrics = st.columns(4)
            selection_metrics[0].metric(
                "Inclusion mode", "Human reviewed" if mode == "manual" else "Automatic"
            )
            selection_metrics[1].metric(
                "Candidate rule",
                (
                    f"Top {int(selection_audit.get('top_n') or 0)}"
                    if basis == "top_n"
                    else f"S ≥ {float(selection_audit.get('similarity_threshold') or 0):.3f}"
                ),
            )
            selection_metrics[2].metric(
                "Articles retained",
                int(selection_audit.get("selected_document_count") or 0),
            )
            selection_metrics[3].metric(
                "Articles excluded by reviewer",
                int(selection_audit.get("excluded_candidate_document_count") or 0),
            )
            if mode == "manual":
                st.success(
                    "All curve methods were refitted from the selected article set. "
                    "The system's similarity/reliability formulas were preserved and "
                    "method-specific weights and ESS were recalculated afterward."
                )
            with st.expander("Article selection audit", expanded=False):
                candidate_details = selection_audit.get("candidates")
                if isinstance(candidate_details, list) and candidate_details:
                    st.dataframe(
                        [
                            {
                                "Rank": item.get("rank"),
                                "Article": item.get("document_id") or item.get("pmcid"),
                                "System similarity": item.get("similarity"),
                                "Available field weight": item.get("available_weight"),
                                "Included": (
                                    item.get("document_id") or item.get("pmcid")
                                )
                                in set(selection_audit.get("selected_document_ids") or []),
                            }
                            for item in candidate_details
                            if isinstance(item, Mapping)
                        ],
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.write(
                        "Selected:",
                        ", ".join(selection_audit.get("selected_document_ids") or [])
                        or "—",
                    )
                    st.write(
                        "Excluded:",
                        ", ".join(
                            selection_audit.get("excluded_candidate_document_ids")
                            or []
                        )
                        or "—",
                    )
                snapshot = selection_audit.get("candidate_snapshot_sha256")
                if snapshot:
                    st.caption(f"Candidate snapshot SHA-256: {snapshot}")
            st.divider()
        st.caption(
            "Historical arms contributing to the curve. The first table shows the "
            "20 highest-positioned stored lines; expand for the full audit trail."
        )
        rows = _historical_rows(curve_result)
        line_rows = _historical_line_rows(curve_result)
        st.dataframe(line_rows[:20], use_container_width=True, hide_index=True)
        with st.expander(f"All historical lines ({len(line_rows)})"):
            st.dataframe(line_rows, use_container_width=True, hide_index=True)
        with st.expander(f"All scored articles ({len(rows)})"):
            st.dataframe(rows, use_container_width=True, hide_index=True)
        mixture_method = curve_methods.get("article_posterior_draw_pool", {})
        mixture_article_rows = _posterior_mixture_article_rows(curve_result)
        mixture_source_rows = _posterior_mixture_source_rows(curve_result)
        if isinstance(mixture_method, Mapping) and mixture_method:
            st.markdown("#### Similarity-weighted posterior-mixture audit")
            st.caption(
                "The article table shows the exact finite-pool quota. An article's "
                "quota is divided among its eligible study/cohort components, then "
                "complete posterior curves are sampled without replacement. This is "
                "component mixing, not power-likelihood borrowing."
            )
            posterior_settings = mixture_method.get("posterior", {})
            mixture_diagnostics = mixture_method.get("mixture_diagnostics", {})
            if not isinstance(posterior_settings, Mapping):
                posterior_settings = {}
            if not isinstance(mixture_diagnostics, Mapping):
                mixture_diagnostics = {}
            mixture_metrics = st.columns(3)
            mixture_metrics[0].metric(
                "Stored mixture curves",
                int(posterior_settings.get("draw_count", 0)),
            )
            mixture_metrics[1].metric(
                "Guaranteed unique-curve capacity",
                int(posterior_settings.get("maximum_unique_draw_count", 0)),
                help=(
                    "Every requested pool size up to this value can satisfy the "
                    "article quotas without reusing a component posterior draw."
                ),
            )
            source_effective_count = mixture_diagnostics.get("source_effective_count")
            mixture_metrics[2].metric(
                "Component weight diversity",
                (
                    f"{float(source_effective_count):.2f}"
                    if source_effective_count is not None
                    else "—"
                ),
                help=(
                    "This is 1 / sum(weight squared), a diversity summary of the "
                    "component weights. It is not a patient effective sample size."
                ),
            )
            if mixture_article_rows:
                st.dataframe(
                    mixture_article_rows, use_container_width=True, hide_index=True
                )
            with st.expander(
                f"Eligible posterior components ({len(mixture_source_rows)})"
            ):
                st.dataframe(
                    mixture_source_rows, use_container_width=True, hide_index=True
                )
            with st.expander("Posterior-mixture construction diagnostics"):
                st.dataframe(
                    _readable_diagnostic_rows(
                        {
                            "interpretation": mixture_method.get("interpretation"),
                            "posterior": posterior_settings,
                            "weight_formula": mixture_diagnostics.get("weight_formula"),
                            "document_posterior_definition": mixture_diagnostics.get(
                                "document_posterior_definition"
                            ),
                            "skipped": mixture_diagnostics.get("skipped_lines"),
                            "skipped_source_details": mixture_diagnostics.get(
                                "skipped_source_details"
                            ),
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        power_rows = _power_likelihood_rows(curve_result)
        if power_rows:
            st.markdown("#### Experimental power-likelihood borrowing audit")
            st.dataframe(power_rows, use_container_width=True, hide_index=True)
            power_method = curve_methods.get("power_likelihood", {})
            power_metrics = st.columns(3)
            diagnostics = power_method.get("weight_diagnostics", {})
            power_metrics[0].metric(
                "Effective historical N",
                f"{float(diagnostics.get('effective_historical_sample_size', 0)):.2f}",
            )
            power_metrics[1].metric(
                "Configured N cap",
                f"{float(diagnostics.get('historical_ess_cap', 0)):.1f}",
            )
            power_metrics[2].metric(
                "Contributing documents",
                int(diagnostics.get("contributor_document_count", 0)),
            )
            with st.expander("Power-likelihood conflict and fit diagnostics"):
                st.dataframe(
                    _readable_diagnostic_rows(
                        {
                            "weight": diagnostics,
                            "heterogeneity": power_method.get(
                                "heterogeneity_diagnostics"
                            ),
                            "fit": power_method.get("fit_diagnostics"),
                            "skipped": power_method.get("skipped_reasons"),
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        random_method = curve_methods.get(
            "power_likelihood_random_intercept_slope", {}
        )
        random_rows = _power_likelihood_rows(
            curve_result, "power_likelihood_random_intercept_slope"
        )
        if isinstance(random_method, Mapping) and random_method:
            st.markdown("#### Study random-effects audit")
            if random_rows:
                st.dataframe(random_rows, use_container_width=True, hide_index=True)
            random_diagnostics = random_method.get("weight_diagnostics", {})
            heterogeneity = (
                random_method.get("heterogeneity_diagnostics", {})
                if isinstance(random_method.get("heterogeneity_diagnostics"), Mapping)
                else {}
            )
            intercept_scale = heterogeneity.get("random_intercept_scale", {})
            slope_scale = heterogeneity.get("random_slope_latent_scale", {})
            random_metrics = st.columns(4)
            random_metrics[0].metric(
                "Independent study units",
                int(random_diagnostics.get("independent_study_group_count", 0)),
            )
            random_metrics[1].metric(
                "Intercept scale",
                (
                    f"{float(intercept_scale.get('posterior_mean')):.3f}"
                    if intercept_scale.get("posterior_mean") is not None
                    else "—"
                ),
            )
            random_metrics[2].metric(
                "Slope latent scale",
                (
                    f"{float(slope_scale.get('posterior_mean')):.3f}"
                    if slope_scale.get("posterior_mean") is not None
                    else "—"
                ),
            )
            random_metrics[3].metric(
                "Effective historical N",
                f"{float(random_diagnostics.get('effective_historical_sample_size', 0)):.2f}",
            )
            with st.expander("Random-effects integration and study-shift diagnostics"):
                st.dataframe(
                    _readable_diagnostic_rows(
                        {
                            "status": random_method.get("status"),
                            "random_effect_prior": random_method.get(
                                "random_effect_prior"
                            ),
                            "integration": random_method.get(
                                "marginal_likelihood_integration"
                            ),
                            "heterogeneity": random_method.get(
                                "heterogeneity_diagnostics"
                            ),
                            "fit": random_method.get("fit_diagnostics"),
                            "skipped": random_method.get("skipped_reasons"),
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        prior = curve_result.get("bayesian_curve_parameters", {})
        model_settings = {
            "bayesian_prior": prior.get("prior_name"),
            "similarity_parameters": curve_result.get("similarity_parameters"),
        }
        with st.expander("Prediction model settings"):
            st.caption(
                "The table follows nested settings from section to field; no model "
                "parameter is omitted."
            )
            st.dataframe(
                _readable_diagnostic_rows(model_settings),
                use_container_width=True,
                hide_index=True,
            )

    with diagnostics_tab:
        st.caption(
            "These diagnostics describe historical dose support. It is not a measure of clinical validity."
        )
        summary = dose_support_summary(curve_result)
        columns = st.columns(3)
        columns[0].metric(
            "Dose support",
            f"{summary['supported_dose_count']}/{summary['requested_dose_count']}",
        )
        columns[1].metric(
            "Historical articles", summary["historical_article_count"] or 0
        )
        columns[2].metric("Historical lines", summary["historical_line_count"] or 0)
        with st.expander("Complete dose-support diagnostics"):
            st.dataframe(
                _readable_diagnostic_rows(summary),
                use_container_width=True,
                hide_index=True,
            )

    with json_tab:
        st.caption(
            "Download the complete reproducible prediction payload, including "
            "parameters, evidence provenance, and diagnostics."
        )
        rendered = prediction_json(curve_result)
        st.download_button(
            "Download complete prediction JSON",
            data=rendered,
            file_name="toxicity_curve_prediction.json",
            mime="application/json",
            use_container_width=True,
        )
        with st.expander("Readable export overview"):
            st.caption(
                "Each row is one section in the downloaded file. The download "
                "retains every value for reproducibility."
            )
            st.dataframe(
                _export_overview_rows(curve_result),
                use_container_width=True,
                hide_index=True,
            )

    if st.button(
        "Continue to OC simulation  →", type="primary", use_container_width=True
    ):
        st.session_state["ui_page"] = "OC Simulation"
        st.rerun()


def _oc_design_label(value: str) -> str:
    return {
        "crm": "CRM",
        "boin": "BOIN",
        "three_plus_three": "3+3",
    }.get(value, value)


ACTIVE_OC_SIMULATION_DESIGNS = ("boin",)
LEGACY_HIDDEN_OC_SIMULATION_DESIGNS = ("crm", "three_plus_three")
ACTIVE_BOIN_TRUTH_CURVE_METHODS = (
    "article_line_weighted",
    "article_posterior_draw_pool",
    "power_likelihood_random_intercept_slope",
)


def _friendly_boin_setting(parameters: Mapping[str, Any]) -> str:
    """Summarize one BOIN candidate without exposing its machine identifier."""

    calibration = str(parameters.get("calibration_label") or "").strip().lower()
    phi1 = parameters.get("resolved_phi1")
    phi2 = parameters.get("resolved_phi2")
    try:
        phi_text = f"φ1={float(phi1):.2f}, φ2={float(phi2):.2f}"
    except (TypeError, ValueError):
        phi_text = "boundary values unavailable"
    calibration_text = (
        f"Automatic ({phi_text})"
        if calibration == "automatic"
        else f"Custom ({phi_text})"
    )
    cutoff = parameters.get("elimination_cutoff")
    minimum_n = parameters.get("elimination_min_n")
    stop_rule = parameters.get("stop_rule_m")
    try:
        elimination_text = f"eliminate at {float(cutoff):.2f} after N≥{int(minimum_n)}"
    except (TypeError, ValueError):
        elimination_text = "elimination rule unavailable"
    stop_text = f"stop m={int(stop_rule)}" if stop_rule is not None else "no stop-m"
    return f"{calibration_text} · {elimination_text} · {stop_text}"


def _boin_ranking_rows(search_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create the compact table used by the active BOIN optimization page."""

    definition = (search_result.get("base_config") or {}).get("mtd_definition")
    selection_label = (
        "Above-target selection"
        if definition != "acceptable_interval"
        else "Overtoxic selection"
    )
    exposure_label = (
        "Patients on above-target doses"
        if definition != "acceptable_interval"
        else "Patients on overtoxic doses"
    )
    acceptable_label = {
        "closest_to_target": "Same-as-correct selection",
        "highest_not_above_target": "At/below-target selection",
    }.get(definition, "Acceptable selection")
    rows: list[dict[str, Any]] = []
    for candidate in search_result.get("candidate_results", []):
        if not isinstance(candidate, Mapping):
            continue
        parameters = candidate.get("parameters") or {}
        metrics = candidate.get("macro_metrics") or {}
        rows.append(
            {
                "Rank": candidate.get("rank"),
                "BOIN setting": _friendly_boin_setting(parameters),
                "Correct decision": metrics.get("correct_decision"),
                selection_label: metrics.get("overtoxic_selection"),
                exposure_label: metrics.get("overtoxic_patient_fraction"),
                "No selection": metrics.get("no_selection"),
                "Mean N": metrics.get("mean_total_sample_size"),
                "Pareto": "★ Pareto" if candidate.get("pareto_efficient") else "—",
                "Candidate": candidate.get("candidate_id"),
                "Calibration": (
                    "Automatic"
                    if str(parameters.get("calibration_label") or "").lower()
                    == "automatic"
                    else "Custom"
                ),
                "phi1": parameters.get("resolved_phi1"),
                "phi2": parameters.get("resolved_phi2"),
                "Escalate boundary": parameters.get("lambda_e"),
                "De-escalate boundary": parameters.get("lambda_d"),
                "Elimination cutoff": parameters.get("elimination_cutoff"),
                "Elimination min N": parameters.get("elimination_min_n"),
                "Stop rule m": parameters.get("stop_rule_m"),
                acceptable_label: metrics.get("acceptable_selection"),
                "Safety stop": metrics.get("safety_stop"),
                "Mean DLT": metrics.get("mean_total_dlt"),
                "Pareto efficient": candidate.get("pareto_efficient"),
            }
        )
    return rows


def render_oc_simulation_page(
    st: Any,
    curve_result: Mapping[str, Any] | None,
    run_oc: Callable[..., Mapping[str, Any]],
) -> None:
    """Render the active BOIN-only parameter-search simulation page.

    ``run_oc`` remains in the public signature so the surrounding application and
    older integrations do not break. The former multi-design page is not shipped
    in this interface.
    """

    del run_oc
    from phase1_prediction.reference.boin_parameter_search import (
        MAX_BOIN_PARAMETER_CANDIDATES,
        BOINCalibration,
        BOINScoreWeights,
        build_boin_parameter_grid,
        parse_calibration_candidates,
        parse_probability_candidates,
        parse_stop_rule_candidates,
        run_boin_parameter_search,
    )
    from phase1_prediction.reference.oc_service import joint_target_posterior_draws
    from phase1_prediction.reference.oc_simulation import MAX_TOTAL_VIRTUAL_TRIALS, OCConfig

    st.markdown(
        '<p class="tox-kicker">Step 4 of 4 · Optimize BOIN</p>',
        unsafe_allow_html=True,
    )
    st.markdown("## BOIN parameter optimization")
    st.markdown(
        '<p class="tox-lead">Compare many BOIN parameter settings under the same '
        "historical posterior-predictive truth ensemble. CRM and 3+3 are temporarily "
        "hidden from this page; their underlying engines remain unchanged.</p>",
        unsafe_allow_html=True,
    )
    if curve_result is None:
        with st.container(border=True):
            st.markdown("### A target-trial dose grid is required")
            st.caption(
                "Enter the actual discrete clinical dose levels on Target Trial and "
                "run the toxicity prediction before optimizing BOIN."
            )
            if st.button("Go to Target Trial  →", type="primary"):
                st.session_state["ui_page"] = "Target Trial"
                st.rerun()
        return

    query = curve_result.get("query", {})
    doses = list(query.get("planned_doses") or [])
    dose_grid = query.get("dose_grid") or {}
    if len(doses) < 2 or dose_grid.get("mode") != "explicit_query_planned_doses":
        st.error(
            "BOIN optimization requires at least two user-entered clinical dose "
            "levels. The automatic display grid cannot be treated as a dose design."
        )
        return
    unit = str(query.get("target_unit") or "")

    with st.container(border=True):
        overview = st.columns(4)
        overview[0].metric("Clinical dose levels", len(doses))
        overview[1].metric("Dose range", f"{doses[0]:g}–{doses[-1]:g} {unit}")
        overview[2].metric("Active design", "BOIN only")
        overview[3].metric("Truth benchmark", "Historical posterior predictive")
        st.caption(
            "Planned doses: " + ", ".join(f"{dose:g}" for dose in doses) + f" {unit}"
        )

    st.markdown("### 1. Trial objective")
    objective_cols = st.columns(3)
    target_dlt = objective_cols[0].number_input(
        "Target DLT probability *",
        min_value=0.01,
        max_value=0.70,
        value=0.25,
        step=0.01,
        format="%.2f",
        key="boin_search_target_dlt",
    )
    mtd_definition = objective_cols[1].selectbox(
        "Correct-dose definition *",
        options=[
            "acceptable_interval",
            "closest_to_target",
            "highest_not_above_target",
        ],
        format_func=lambda value: {
            "acceptable_interval": "Within acceptable interval",
            "closest_to_target": "Closest to target",
            "highest_not_above_target": "Highest not above target",
        }[value],
        key="boin_search_mtd_definition",
    )
    start_index = objective_cols[2].selectbox(
        "Starting dose *",
        options=list(range(len(doses))),
        format_func=lambda index: f"Dose {index + 1}: {doses[index]:g} {unit}",
        key="boin_search_start_index",
    )
    if mtd_definition == "acceptable_interval":
        acceptable_cols = st.columns(2)
        acceptable_low = acceptable_cols[0].number_input(
            "Acceptable lower bound *",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            format="%.2f",
            key="boin_search_acceptable_low",
        )
        acceptable_high = acceptable_cols[1].number_input(
            "Acceptable upper bound *",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
            format="%.2f",
            key="boin_search_acceptable_high",
        )
    else:
        acceptable_low = float(target_dlt)
        acceptable_high = float(target_dlt)
        if mtd_definition == "closest_to_target":
            st.info(
                "Closest-to-target does not use an acceptable interval. Correct "
                "selection means choosing the dose whose true DLT probability is "
                "closest to target; above-target selection/exposure uses target as "
                "the transparent safety threshold."
            )
        else:
            st.info(
                "Highest-not-above-target does not use an acceptable interval. "
                "Correct selection is the highest tested dose whose true DLT "
                "probability does not exceed target."
            )
    objective_error: str | None = None
    if mtd_definition == "acceptable_interval" and not (
        0.0
        <= float(acceptable_low)
        <= float(target_dlt)
        <= float(acceptable_high)
        <= 1.0
    ):
        objective_error = (
            "The acceptable interval must satisfy lower bound ≤ target ≤ upper bound."
        )
        st.error(objective_error)

    design_cols = st.columns(4)
    cohort_size = design_cols[0].number_input(
        "Cohort size *",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
        key="boin_search_cohort_size",
    )
    max_sample_size = design_cols[1].number_input(
        "Maximum sample size *",
        min_value=int(cohort_size),
        max_value=500,
        value=max(30, int(cohort_size)),
        step=int(cohort_size),
        key="boin_search_max_sample_size",
    )
    trials_per_scenario = design_cols[2].number_input(
        "Virtual trials per posterior truth curve *",
        min_value=10,
        max_value=100_000,
        value=100,
        step=50,
        key="boin_search_trials_per_scenario",
    )
    seed = design_cols[3].number_input(
        "Random seed *",
        min_value=0,
        max_value=2_147_483_647,
        value=9,
        step=1,
        key="boin_search_seed",
    )
    st.markdown("### 2. Historical posterior-predictive truth")
    curve_methods = curve_result.get("toxicity_curve", {})
    if not isinstance(curve_methods, Mapping):
        curve_methods = {}
    truth_method_options = [
        key
        for key in ACTIVE_BOIN_TRUTH_CURVE_METHODS
        if isinstance(curve_methods.get(key), Mapping)
        and curve_methods[key].get("status") == "fitted"
    ]
    if not truth_method_options:
        st.error(
            "No fitted historical toxicity-curve method is available for simulation."
        )
        return
    selected_result_method = str(
        curve_result.get("selected_curve_method")
        or st.session_state.get("target_curve_method")
        or truth_method_options[0]
    )
    if selected_result_method not in truth_method_options:
        st.error(
            "The method selected in Step 2 is not available in this prediction. "
            "Return to Target Trial and generate the curve again."
        )
        return
    truth_curve_method = selected_result_method
    truth_cols = st.columns(2)
    truth_cols[0].markdown(
        "**Posterior curve used to generate truth**  \n"
        + _CURVE_METHOD_LABELS.get(truth_curve_method, truth_curve_method)
    )
    truth_preview_error: str | None = None
    available_posterior_draws = 0
    try:
        truth_preview = joint_target_posterior_draws(
            curve_result, curve_method_key=str(truth_curve_method)
        )
        available_posterior_draws = int(
            truth_preview.get(
                "maximum_unique_curve_count",
                truth_preview.get(
                    "maximum_unique_draw_count",
                    len(truth_preview.get("posterior_draws", [])),
                ),
            )
        )
    except Exception as error:
        truth_preview_error = str(error)
    posterior_curve_count_max = max(
        1,
        min(
            10_000,
            available_posterior_draws if available_posterior_draws > 0 else 1,
        ),
    )
    posterior_curve_count_key = "boin_search_posterior_curve_count"
    stored_curve_count = st.session_state.get(posterior_curve_count_key)
    if (
        isinstance(stored_curve_count, (int, float))
        and int(stored_curve_count) > posterior_curve_count_max
    ):
        st.session_state[posterior_curve_count_key] = posterior_curve_count_max
    posterior_curve_count = truth_cols[1].number_input(
        "Posterior truth curves to sample *",
        min_value=1,
        max_value=posterior_curve_count_max,
        value=min(_DEFAULT_BOIN_POSTERIOR_CURVE_COUNT, posterior_curve_count_max),
        step=min(10, posterior_curve_count_max),
        key=posterior_curve_count_key,
        help=(
            "The upper bound is the quota-feasible number of unique complete curve "
            "draws available from the selected truth method."
        ),
    )
    if truth_preview_error:
        st.error(
            f"Selected posterior truth source is unavailable: {truth_preview_error}"
        )
    elif truth_curve_method == "article_posterior_draw_pool":
        st.caption(
            f"Up to {available_posterior_draws:,} quota-feasible unique complete "
            "historical component draws are available. At run time, the requested "
            "pool size is allocated exactly across articles by normalized article "
            "similarity, then within each article across eligible components."
        )
    else:
        st.caption(
            f"{available_posterior_draws:,} complete joint posterior curve draws are "
            "available. The requested rows are sampled without replacement, and one "
            "whole curve remains fixed throughout each virtual trial."
        )
    if truth_curve_method == "article_line_weighted":
        st.info(
            "Truth is generated by the established method: at every posterior draw "
            "index, the accepted article-level posterior curves are combined using "
            "one fixed contributor set and fixed similarity weights across all doses."
        )
    elif truth_curve_method == "article_posterior_draw_pool":
        st.info(
            "Truth is a similarity-weighted finite mixture. The requested number of "
            "curves is first divided among articles, then among eligible components "
            "within each article. Every row comes from one component's posterior and "
            "uses the same alpha/slope draw across all doses. Articles are not "
            "averaged within a row, and no historical likelihood or ESS cap is used."
        )
    elif truth_curve_method == "power_likelihood":
        st.info(
            "Truth is generated from joint posterior draws of the shared powered "
            "likelihood model. Each draw uses one alpha and one monotone slope for "
            "the complete dose grid."
        )
    else:
        st.info(
            "Truth predicts a new target study: each complete curve draw uses one "
            "new-study random intercept and one new-study latent-slope effect, both "
            "held fixed across every dose."
        )
    if truth_curve_method in {
        "power_likelihood",
        "power_likelihood_random_effects",
        "power_likelihood_random_intercept_slope",
    }:
        truth_weight_diagnostics = curve_methods[truth_curve_method].get(
            "weight_diagnostics", {}
        )
        if isinstance(truth_weight_diagnostics, Mapping):
            cap_cols = st.columns(3)
            cap_cols[0].metric(
                "Configured ESS max",
                f"{float(truth_weight_diagnostics.get('historical_ess_cap', 0)):.1f}",
            )
            cap_cols[1].metric(
                "ESS before cap",
                f"{float(truth_weight_diagnostics.get('pre_cap_effective_sample_size', 0)):.2f}",
            )
            cap_cols[2].metric(
                "ESS used by model",
                f"{float(truth_weight_diagnostics.get('effective_historical_sample_size', 0)):.2f}",
            )
            st.caption(
                "ESS max is chosen when the toxicity curve is fitted. To change it, "
                "return to Target Trial, edit the visible ESS-max field, and rerun the "
                "toxicity prediction before this BOIN search."
            )

    st.markdown("### 3. Parameter grid")
    st.caption(
        "Enter comma-separated candidates. These are editable values, not a fixed "
        "menu. Duplicate values are removed. The Cartesian grid must contain at most "
        f"{MAX_BOIN_PARAMETER_CANDIDATES} unique parameter sets."
    )
    calibration_text = st.text_input(
        "Boundary calibrations (automatic or phi1:phi2 pairs) *",
        value=_DEFAULT_BOIN_CALIBRATION_CANDIDATES,
        help="Each custom pair must satisfy 0 < phi1 < target < phi2 < 1.",
        key="boin_search_calibration_text",
    )
    grid_cols = st.columns(2)
    elimination_cutoff_text = grid_cols[0].text_input(
        "Elimination cutoffs *",
        value=_DEFAULT_BOIN_ELIMINATION_CUTOFF_CANDIDATES,
        key="boin_search_elimination_cutoff_text",
    )
    stop_rule_text = grid_cols[1].text_input(
        "Stop-at-m choices *",
        value=_DEFAULT_BOIN_STOP_RULE_CANDIDATES,
        help="Use 'off' to include a candidate with this stopping rule disabled.",
        key="boin_search_stop_rule_text",
    )
    grid_error: str | None = None
    calibrations: tuple[BOINCalibration, ...] = ()
    elimination_cutoffs: tuple[float, ...] = ()
    stop_rule_ms: tuple[int | None, ...] = ()
    try:
        calibrations = parse_calibration_candidates(
            calibration_text, target_dlt=float(target_dlt)
        )
        elimination_cutoffs = parse_probability_candidates(
            elimination_cutoff_text, label="elimination cutoff"
        )
        stop_rule_ms = parse_stop_rule_candidates(
            stop_rule_text,
            cohort_size=int(cohort_size),
            maximum_sample_size=int(max_sample_size),
        )
    except ValueError as error:
        grid_error = str(error)
        st.error(grid_error)

    with st.expander("Settings held fixed across candidates", expanded=False):
        fixed_cols = st.columns(2)
        elimination_min_n = 1
        check_lowest = fixed_cols[0].checkbox(
            "Extra lowest-dose safety check",
            value=True,
            key="boin_search_check_lowest",
        )
        lowest_delta = fixed_cols[1].number_input(
            "Lowest-dose safety delta",
            min_value=0.0,
            max_value=0.50,
            value=0.03,
            step=0.01,
            key="boin_search_lowest_delta",
        )
        fixed_flags = st.columns(3)
        final_bound = fixed_flags[0].checkbox(
            "Final de-escalation bound",
            value=True,
            key="boin_search_final_bound",
        )
        isotonic = fixed_flags[1].checkbox(
            "Isotonic final selection",
            value=True,
            key="boin_search_isotonic",
        )
        modify_1_3 = fixed_flags[2].checkbox(
            "Change a 1/3 de-escalation decision to stay",
            value=False,
            key="boin_search_modify_1_3",
        )

    st.markdown("### 4. Definition of ‘best’")
    st.caption("Choose metric importance, highest first. Later metrics break exact ties only. The winner is conditional on this priority and the simulated scenarios.")
    labels = {"correct_decision": "Correct selection ↑", "overtoxic_selection": "Overtoxic selection ↓", "safety_stop": "Safety early stopping ↓", "mean_total_sample_size": "Mean N ↓"}
    remaining = list(labels)
    ranking_priority = []
    for index, column in enumerate(st.columns(4)):
        key = f"boin_search_priority_{index}"
        if st.session_state.get(key) not in remaining:
            st.session_state.pop(key, None)
        metric = column.selectbox(f"Importance {index + 1}", remaining, format_func=labels.get, key=key)
        ranking_priority.append(metric)
        remaining.remove(metric)

    candidate_count = (
        len(calibrations)
        * len(elimination_cutoffs)
        * len(stop_rule_ms)
    )
    candidate_limit_exceeded = candidate_count > MAX_BOIN_PARAMETER_CANDIDATES
    estimated_trials = (
        candidate_count * int(posterior_curve_count) * int(trials_per_scenario)
    )
    workload_too_large = estimated_trials > MAX_TOTAL_VIRTUAL_TRIALS
    if candidate_count:
        st.info(
            f"Planned search: {candidate_count} parameter sets × "
            f"{int(posterior_curve_count):,} posterior truth curves × "
            f"{int(trials_per_scenario):,} trials = "
            f"{estimated_trials:,} virtual BOIN trials."
        )
    else:
        st.error("Enter at least one valid value in every parameter dimension.")
    if candidate_limit_exceeded:
        st.error(
            f"The Cartesian grid contains {candidate_count} parameter sets; the "
            f"limit is {MAX_BOIN_PARAMETER_CANDIDATES}. Remove one or more values."
        )
    if workload_too_large:
        st.error(
            f"This exceeds the synchronous limit of "
            f"{MAX_TOTAL_VIRTUAL_TRIALS:,} virtual trials. Reduce the grid or repetitions."
        )

    signature_payload = {
        "curve_generated_at": curve_result.get("generated_at_utc"),
        "doses": doses,
        "target": target_dlt,
        "acceptable": [acceptable_low, acceptable_high],
        "start_index": start_index,
        "cohort_size": cohort_size,
        "max_sample_size": max_sample_size,
        "trials_per_scenario": trials_per_scenario,
        "posterior_curve_count": posterior_curve_count,
        "truth_curve_method": truth_curve_method,
        "seed": seed,
        "mtd_definition": mtd_definition,
        "calibrations": [
            [value.label, value.phi1, value.phi2] for value in calibrations
        ],
        "cutoffs": list(elimination_cutoffs),
        "stop_ms": list(stop_rule_ms),
        "fixed": [
            elimination_min_n,
            check_lowest,
            lowest_delta,
            final_bound,
            isotonic,
            modify_1_3,
        ],
        "ranking_priority": ranking_priority,
    }
    current_signature = json.dumps(
        signature_payload, ensure_ascii=False, sort_keys=True, default=str
    )
    run_clicked = st.button(
        "Run BOIN parameter search  →",
        type="primary",
        use_container_width=True,
        disabled=(
            not candidate_count
            or objective_error is not None
            or grid_error is not None
            or truth_preview_error is not None
            or int(posterior_curve_count) > available_posterior_draws
            or candidate_limit_exceeded
            or workload_too_large
        ),
    )
    if (
        truth_preview_error is None
        and int(posterior_curve_count) > available_posterior_draws
    ):
        st.error(
            f"Requested {int(posterior_curve_count):,} posterior truth curves, but "
            f"the selected method contains only {available_posterior_draws:,} "
            "complete joint draws."
        )
    if run_clicked:
        st.session_state.pop("oc_result", None)
        st.session_state.pop("oc_result_signature", None)
        try:
            if int(max_sample_size) % int(cohort_size) != 0:
                raise ValueError(
                    "Maximum sample size must be exactly divisible by cohort size."
                )
            base_config = OCConfig(
                doses=tuple(float(value) for value in doses),
                target_dlt=float(target_dlt),
                acceptable_low=float(acceptable_low),
                acceptable_high=float(acceptable_high),
                start_index=int(start_index),
                cohort_size=int(cohort_size),
                max_sample_size=int(max_sample_size),
                designs=ACTIVE_OC_SIMULATION_DESIGNS,
                trials_per_scenario=int(trials_per_scenario),
                seed=int(seed),
                mtd_definition=str(mtd_definition),
                boin_stop_rule_m=max(int(cohort_size), int(max_sample_size)),
                boin_check_lowest_dose=bool(check_lowest),
                boin_lowest_dose_delta=float(lowest_delta),
                boin_check_final_mtd_deesc_bound=bool(final_bound),
                boin_use_isotonic=bool(isotonic),
                boin_modify_1_3_deesc_to_stay=bool(modify_1_3),
            ).validated()
            grid = build_boin_parameter_grid(
                calibrations=calibrations,
                elimination_cutoffs=tuple(
                    float(value) for value in elimination_cutoffs
                ),
                elimination_min_n=int(elimination_min_n),
                stop_rule_ms=tuple(
                    int(value) if value is not None else None for value in stop_rule_ms
                ),
                base_config=base_config,
            )
            with st.status(
                "Running the same virtual truth set for every BOIN candidate...",
                expanded=True,
            ) as status:
                result = run_boin_parameter_search(
                    curve_result,
                    base_config=base_config,
                    candidates=grid,
                    ranking_priority=ranking_priority,
                    truth_mode="posterior_predictive",
                    posterior_curve_count=int(posterior_curve_count),
                    curve_method_key=str(truth_curve_method),
                )
                status.update(label="BOIN parameter search complete", state="complete")
            st.session_state["oc_result"] = dict(result)
            st.session_state["oc_result_signature"] = current_signature
        except Exception as error:
            st.error(f"BOIN parameter search failed: {error}")

    search_result = st.session_state.get("oc_result")
    if not isinstance(search_result, Mapping):
        st.caption(
            "No BOIN search result yet. Running the search does not change the "
            "historical toxicity curve."
        )
        return
    if st.session_state.get("oc_result_signature") != current_signature:
        st.warning(
            "The BOIN search settings changed after this result was generated. "
            "The stale result is hidden; run the search again."
        )
        return
    if search_result.get("format") != "phase1_boin_parameter_search_v3_priority":
        st.warning(
            "A result from the previous multi-design simulator is stored in this "
            "session. Run the BOIN parameter search to replace it."
        )
        return

    st.markdown("### 5. BOIN optimization results")
    ranking_rows = _boin_ranking_rows(search_result)
    if not ranking_rows:
        st.error("The BOIN search result contains no candidates.")
        return
    best = search_result["candidate_results"][0]
    best_parameters = best["parameters"]
    best_metrics = best["macro_metrics"]
    safety_result_label = (
        "Overtoxic selection"
        if mtd_definition == "acceptable_interval"
        else "Above-target selection"
    )
    exposure_result_label = (
        "Overtoxic exposure"
        if mtd_definition == "acceptable_interval"
        else "Above-target exposure"
    )
    acceptable_result_label = {
        "closest_to_target": "Same-as-correct selection",
        "highest_not_above_target": "At/below-target selection",
    }.get(mtd_definition, "Acceptable selection")
    with st.container(border=True):
        st.markdown("#### Best under the selected metric priorities")
        metrics = st.columns(5)
        metrics[0].metric("Safety early stopping", f"{float(best_metrics['safety_stop']):.1%}")
        metrics[1].metric(
            "Correct decision", f"{float(best_metrics['correct_decision']):.1%}"
        )
        metrics[2].metric(
            safety_result_label,
            f"{float(best_metrics['overtoxic_selection']):.1%}",
        )
        metrics[3].metric(
            exposure_result_label,
            f"{float(best_metrics['overtoxic_patient_fraction']):.1%}",
        )
        metrics[4].metric(
            "Mean total sample size",
            f"{float(best_metrics['mean_total_sample_size']):.1f}",
        )
        st.markdown(
            f"**Recommended setting:** {_friendly_boin_setting(best_parameters)}"
        )
        st.caption(str(search_result.get("best_candidate_interpretation")))

    ranking_tab, scenario_tab, truth_tab, export_tab = st.tabs(
        ["Parameter ranking", "Scenario detail", "Truth scenarios", "Export"]
    )
    with ranking_tab:
        st.caption(
            "Results are ordered by the metric priorities selected above. The compact "
            "table leads with outcomes; expand the technical table for every raw "
            "boundary and the reproducible candidate ID."
        )
        primary_columns = [
            "Rank",
            "BOIN setting",
            "Safety stop",
            "Correct decision",
            safety_result_label,
            exposure_result_label,
            "No selection",
            "Mean N",
            "Pareto",
        ]
        probability_columns = {
            "Correct decision",
            safety_result_label,
            exposure_result_label,
            "No selection",
        }
        ranking_column_config: dict[str, Any] = {
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "BOIN setting": st.column_config.TextColumn("BOIN setting", width="large"),
            "Safety stop": st.column_config.NumberColumn("Safety stop", format="percent", width="small"),
            "Mean N": st.column_config.NumberColumn(
                "Mean N", format="%.1f", width="small"
            ),
            "Pareto": st.column_config.TextColumn("Pareto", width="small"),
        }
        for column in probability_columns:
            ranking_column_config[column] = st.column_config.NumberColumn(
                column,
                format="percent",
                width="small",
            )
        st.dataframe(
            ranking_rows,
            use_container_width=True,
            hide_index=True,
            column_order=primary_columns,
            column_config=ranking_column_config,
            height=min(600, 38 * (len(ranking_rows) + 1)),
        )
        pareto_count = sum(bool(row["Pareto efficient"]) for row in ranking_rows)
        st.caption(
            f"{pareto_count} candidate(s) are Pareto efficient. A Pareto candidate "
            "cannot improve correct selection without worsening at least one of "
            f"{safety_result_label.lower()}, {exposure_result_label.lower()}, or "
            "mean total sample size."
        )
        with st.expander("Complete parameter and metric table"):
            st.caption(
                "This audit table retains the machine candidate ID, resolved BOIN "
                "boundaries, all safety settings, and every reported outcome."
            )
            st.dataframe(
                ranking_rows,
                use_container_width=True,
                hide_index=True,
                column_config=ranking_column_config,
            )
    with scenario_tab:
        candidate_by_id = {
            row["candidate_id"]: row
            for row in search_result.get("candidate_results", [])
        }
        selected_id = st.selectbox(
            "Candidate to inspect",
            options=list(candidate_by_id),
            format_func=lambda candidate_id: (
                f"Rank {candidate_by_id[candidate_id]['rank']}: "
                f"{_friendly_boin_setting(candidate_by_id[candidate_id]['parameters'])}"
            ),
            key="boin_search_detail_candidate",
        )
        selected = candidate_by_id[selected_id]
        scenario_rows = []
        for scenario in selected["simulation"]["design_results"]["boin"][
            "per_scenario"
        ]:
            scenario_rows.append(
                {
                    "Scenario": scenario.get("scenario_name"),
                    "True MTD level": (
                        int(scenario["true_mtd_index"]) + 1
                        if scenario.get("true_mtd_index") is not None
                        else None
                    ),
                    "Correct decision": scenario.get("correct_decision", {}).get(
                        "probability"
                    ),
                    acceptable_result_label: scenario.get(
                        "acceptable_selection", {}
                    ).get("probability"),
                    safety_result_label: scenario.get("overtoxic_selection", {}).get(
                        "probability"
                    ),
                    exposure_result_label: scenario.get("overtoxic_patient_fraction"),
                    "No selection": scenario.get("no_selection", {}).get("probability"),
                    "Mean N": scenario.get("mean_total_sample_size"),
                    "Mean DLT": scenario.get("mean_total_dlt"),
                }
            )
        st.dataframe(scenario_rows, use_container_width=True, hide_index=True)
    with truth_tab:
        truth_rows = []
        for scenario in search_result.get("truth_scenarios", []):
            truth_rows.append(
                {
                    "Scenario": scenario.get("name"),
                    **{
                        f"Dose {index + 1}": probability
                        for index, probability in enumerate(
                            scenario.get("probabilities", [])
                        )
                    },
                }
            )
        st.dataframe(truth_rows, use_container_width=True, hide_index=True)
    with export_tab:
        rendered = json.dumps(search_result, ensure_ascii=False, indent=2)
        st.download_button(
            "Download complete BOIN parameter-search result",
            data=rendered,
            file_name="boin_parameter_search.json",
            mime="application/json",
            use_container_width=True,
        )
