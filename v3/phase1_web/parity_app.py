"""The v1 target/results/BOIN interface on top of the native v3 library."""
from dataclasses import asdict
import json
import streamlit as st

from phase1_prediction.evidence import load_evidence_library
from phase1_prediction.reference_bridge import build_workspace, predict
from phase1_prediction.reference.prediction_contract import PREDICTION_MODEL_VERSION
from phase1_prediction.reference.prediction_service import preview_historical_articles, predict_probability_at_dose
from phase1_prediction.reference.oc_service import run_oc_from_curve
from phase1_web.reference.components import render_target_trial_page, render_prediction_results_page, render_oc_simulation_page
from phase1_web.reference.helpers import load_prior_options
from phase1_web.reference.theme import APP_CSS
from phase1_web.reference.target_trial_draft import (
    TARGET_TRIAL_DRAFT_SKIP_SAVE_KEY, save_target_trial_draft_if_changed, target_trial_values_fingerprint,
)
from phase1_web.reference.article_review import ARTICLE_REVIEW_PREVIEW_KEY, ARTICLE_REVIEW_FORM_FINGERPRINT_KEY


def main(native):
    st.set_page_config(page_title="Phase I Toxicity Curve", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
    if not st.session_state.get(TARGET_TRIAL_DRAFT_SKIP_SAVE_KEY, False):
        try:
            save_target_trial_draft_if_changed(st.session_state)
        except (OSError, TypeError, ValueError) as error:
            st.warning(f"Could not save target draft: {error}")
    settings = st.session_state.setdefault("v3_settings", native._initial_settings())
    roots = native._configured_roots(settings)
    # Reload resolved metadata, not posterior fits, so review edits invalidate a
    # result immediately; a path-only Streamlit cache missed changes in place.
    library = load_evidence_library(roots, evidence_policy=settings["evidence_policy"], review_policy=settings["review_policy"])
    context = native._fingerprint([PREDICTION_MODEL_VERSION, library.fingerprint, library.evidence_policy, library.review_policy, library.source_roots])
    if st.session_state.get("v3_reference_context") != context:
        for key in ("curve_result", "oc_result", "oc_result_signature", ARTICLE_REVIEW_PREVIEW_KEY, "v3_reference_workspace"):
            st.session_state.pop(key, None)
        st.session_state["v3_reference_context"] = context
    st.markdown(f'''<div class="tox-hero"><p class="tox-brand">Phasal Toxicity Curve</p>
        <p class="tox-subtitle">Explore Phase I dose–toxicity curves using a transparent, similarity-weighted historical literature model.</p>
        <div class="tox-hero-meta"><span class="tox-pill">{library.document_count} v3 extracted records</span>
        <span class="tox-pill">v3 evidence · v1 downstream methods</span><span class="tox-pill">Exploratory research use</span></div></div>''', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="tox-sidebar-brand">◈ Phasal</div><div class="tox-sidebar-note">Phase I toxicity workspace</div>', unsafe_allow_html=True)
        st.markdown("### Workflow")
        st.session_state.setdefault("ui_page", "Library")
        for label, display in [("Library", "1  ·  Library"), ("Article comparison", "Article comparison"), ("Target Trial", "2  ·  Target trial"),
                               ("Prediction Results", "3  ·  Results"), ("OC Simulation", "4  ·  OC simulation")]:
            if st.button(display, key=f"nav_{label}", use_container_width=True,
                         type="primary" if st.session_state["ui_page"] == label else "secondary"):
                st.session_state["ui_page"] = label
                st.rerun()
        with st.expander("v3 library and extraction settings"):
            for label in ("Extraction Jobs", "Settings"):
                if st.button(label, key=f"v3_reference_nav_{label}"):
                    st.session_state["ui_page"] = label
                    st.rerun()
        st.caption("The three approved curve methods share the v1 calculation baseline. BOIN uses regular cohorts throughout.")
    page = st.session_state["ui_page"]
    if page == "Library":
        native._library_page(roots, library)
    elif page == "Article comparison":
        from phase1_web.article_comparison import render
        render(st, library)
    elif page == "Extraction Jobs":
        native._jobs_page()
    elif page == "Settings":
        native._settings_page(settings)
    elif page == "Target Trial":
        request = render_target_trial_page(st, load_prior_options(), use_manual_similarity_defaults=True,
            historical_article_preview=st.session_state.get(ARTICLE_REVIEW_PREVIEW_KEY), article_review_context_fingerprint=context)
        if request is None:
            return
        try:
            with st.status("Preparing v3 historical evidence...", expanded=True) as status:
                workspace_key = (context, request.bayesian_prior)
                cached = st.session_state.get("v3_reference_workspace")
                workspace = cached[1] if cached and cached[0] == workspace_key else build_workspace(library, prior_name=request.bayesian_prior)
                st.session_state["v3_reference_workspace"] = (workspace_key, workspace)
                if request.submission_action == "preview":
                    preview = preview_historical_articles(request.query, workspace, field_weights=request.field_weights,
                        similarity_threshold=request.similarity_threshold,
                        embedding_backend=request.embedding_backend,
                        embedding_model=request.embedding_model,
                        candidate_basis=request.historical_article_candidate_basis,
                        top_n=request.historical_article_top_n)
                    # UI detail adapters read v3 artifacts, never legacy compact files.
                    for candidate in preview.get("candidates", []):
                        candidate["v3_source_roots"] = list(library.source_roots)
                    preview["_ui_form_fingerprint"] = target_trial_values_fingerprint(st.session_state)
                    preview["_ui_context_fingerprint"] = context
                    st.session_state[ARTICLE_REVIEW_PREVIEW_KEY] = preview
                    st.session_state[ARTICLE_REVIEW_FORM_FINGERPRINT_KEY] = preview["_ui_form_fingerprint"]
                    status.update(label="Historical article preview ready", state="complete")
                else:
                    options = asdict(request)
                    for key in ("query", "bayesian_prior", "submission_action", "curve_method"):
                        options.pop(key)
                    result = predict(library, request.query, workspace, **options)
                    result["selected_curve_method"] = request.curve_method
                    st.session_state["curve_result"] = result
                    st.session_state.pop("oc_result", None)
                    st.session_state["ui_page"] = "Prediction Results"
                    status.update(label="Prediction complete", state="complete")
            st.rerun()
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            st.error(f"Could not complete prediction: {error}")
    elif page == "Prediction Results":
        render_prediction_results_page(st, st.session_state.get("curve_result"), predict_probability_at_dose)
    elif page == "OC Simulation":
        render_oc_simulation_page(st, st.session_state.get("curve_result"), run_oc_from_curve)
