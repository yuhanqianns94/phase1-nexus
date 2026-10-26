"""User-facing v3 workflow for evidence-based Phase I design.

The application reads existing v3 extraction artifacts and keeps human review in
the separate ``phase1-review-v3`` application.  Article posteriors and target-specific power-prior fits are cached for reuse.
The prospective workflow separates planning, review, exploration and simulation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from phase1_extraction.artifacts import read_json, write_json  # noqa: E402
from phase1_prediction.evidence import load_evidence_library  # noqa: E402
from phase1_project.catalog import build_catalog, default_extraction_roots, discover_batch_jobs  # noqa: E402
from phase1_project.runtime import OUTPUT_ROOT, WORKSPACE_ROOT  # noqa: E402


NAVIGATION = ("Library", "Target Trial", "Prediction Results", "OC Simulation")
STATE_ROOT = Path(os.environ.get("PHASE1_V3_STATE_ROOT", OUTPUT_ROOT)).expanduser()
SETTINGS_PATH = STATE_ROOT / "workflow_settings.json"

EVIDENCE_LABELS = {
    "project_default": "Usable dose–N–DLT records",
    "reported_only": "Directly reported records only",
}
REVIEW_LABELS = {
    "extracted": "Extracted results",
    "reviewed_if_available": "Use human corrections when available",
    "reviewed_only": "Only fully reviewed results",
}


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()




def _initial_settings() -> dict[str, Any]:
    defaults = {
        "extraction_roots": [str(path) for path in default_extraction_roots()],
        "evidence_policy": "project_default",
        "review_policy": "extracted",
    }
    try:
        saved = read_json(SETTINGS_PATH)
        if isinstance(saved, dict):
            defaults.update({key: saved[key] for key in defaults if key in saved})
    except (OSError, ValueError):
        pass
    configured = os.environ.get("PHASE1_V3_EXTRACTION_ROOTS")
    if configured is not None:
        defaults["extraction_roots"] = [value for value in configured.split(os.pathsep) if value]
    defaults["review_policy"] = "extracted"
    return defaults


def _configured_roots(settings: dict[str, Any]) -> list[Path]:
    return [Path(value).expanduser() for value in settings.get("extraction_roots", []) if str(value).strip()]


@st.cache_data(show_spinner=False)
def _load_library_cached(root_values: tuple[str, ...], evidence_policy: str, review_policy: str):
    return load_evidence_library(
        [Path(value) for value in root_values],
        evidence_policy=evidence_policy,
        review_policy=review_policy,
    )




def _json_download(label: str, payload: Any, filename: str) -> None:
    st.download_button(label, json.dumps(payload, ensure_ascii=False, indent=2).encode(), filename, "application/json")




def _invalidate_downstream() -> None:
    for key in (
        "curve_result", "oc_result", "oc_result_signature", "v3_reference_workspace", "historical_article_review_preview",
        "v3_evidence_preview", "v3_prediction", "v3_simulation", "v3_design_comparison",
        "v3_active_method", "v3_include_series", "v3_exclude_series",
    ):
        st.session_state.pop(key, None)
    for key in tuple(st.session_state):
        if str(key).startswith("v3_boin_") or key in {"v3_curve_method_select"}:
            st.session_state.pop(key, None)












































def _article_summaries(library) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for series in library.series:
        row = result.setdefault(series.document_id, {"series": 0, "points": 0, "patients": 0})
        row["series"] += 1
        row["points"] += len(series.points)
        row["patients"] += sum(point.n for point in series.points)
    return result


def _library_page(roots: list[Path], library, *, model_scopes=None) -> None:
    from phase1_web.hosted import is_public
    public = is_public()
    st.title("Literature Library")
    st.caption("Browse the processed evidence in your selected library." if public else "Browse literature already present on disk. This page never starts crawling or model extraction.")
    catalog = build_catalog(extraction_roots=roots, **({"xml_directories": [], "pdf_directories": []} if public else {}))
    from phase1_prediction.multidrug.workflow import phase_status
    if model_scopes is None:
        from phase1_prediction.multidrug.observations import load_model_library
        model_scopes, _ = load_model_library(roots, evidence_policy=library.evidence_policy, review_policy=library.review_policy)
    usable = {}
    phase_by_article = {}
    for item in model_scopes:
        status = phase_status(item.profile)
        phase_by_article.setdefault(item.document_id, set()).add(status)
        if status != 'phase_i':
            continue
        summary = usable.setdefault(item.document_id, {'scopes': 0, 'points': 0})
        summary['scopes'] += 1
        summary['points'] += len(item.observations)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Library articles" if public else "Local articles", catalog["article_count"])
    c2.metric("Articles with v3 output", catalog["ready_extraction_count"])
    c3.metric("Phase I eligible articles", len(usable))
    c4.metric("Phase I model scopes", sum(r["scopes"] for r in usable.values()))
    st.caption("Eligibility follows the Phase I filter used in prediction. Eligible scopes without DLT/N are prior-only. Eligibility does not guarantee a usable fit or a match to your target.")
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search by PMC or PMID", placeholder="PMC3309850 or PMID42718886").strip().casefold()
    scope = f2.selectbox("Show", ["Articles with v3 output", "Phase I eligible articles", "All local articles"])
    status = f3.selectbox("Material filter", ["Any", "PDF available", "XML available"])
    rows = []
    for article in catalog["articles"]:
        document_id = article["document_id"]
        if query and query not in document_id.casefold():
            continue
        if scope == "Articles with v3 output" and article["extraction_status"] != "ready":
            continue
        if scope == "Phase I eligible articles" and document_id not in usable:
            continue
        if status == "PDF available" and not article["pdf_path"]:
            continue
        if status == "XML available" and not article["xml_path"]:
            continue
        summary = usable.get(document_id, {})
        statuses = phase_by_article.get(document_id, set())
        if document_id in usable:
            eligibility = 'Eligible with outcomes' if summary['points'] else 'Eligible: prior only'
        elif 'unknown' in statuses:
            eligibility = 'Excluded: phase not linked or unclear'
        elif 'mixed_requires_separation' in statuses:
            eligibility = 'Excluded: Phase I/II not separated'
        elif 'out_of_scope' in statuses:
            eligibility = 'Excluded: outside Phase I'
        else:
            eligibility = 'No model scope available'
        coverage = "Not extracted"
        if article["extraction_status"] == "ready":
            reported = json.loads(Path(article["reported_extraction"]).read_text())
            coverage = "Complete coverage" if reported.get("extraction_complete") else "Partial coverage"
        rows.append({
            "Article": document_id, "V3 extraction": coverage,
            "Phase I eligibility": eligibility, "Eligible scopes": summary.get("scopes", 0),
            "DLT observations": summary.get("points", 0), "PDF": "Yes" if article["pdf_path"] else "No", "XML": "Yes" if article["xml_path"] else "No",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if any(row["V3 extraction"] == "Partial coverage" for row in rows):
        st.caption("Partial coverage: some source material or extracted information is incomplete. Missing values and model exclusions are retained; coverage is not a manual-review status.")
    if rows:
        selected_id = st.selectbox("Article details", [row["Article"] for row in rows])
        article = next(item for item in catalog["articles"] if item["document_id"] == selected_id)
        if article.get('source_url'):
            st.link_button("Open article", article["source_url"])
        if not public:
            st.write({"Extraction result": article["extraction_root"], "Local PDF": article["pdf_path"], "Local XML": article["xml_path"]})
    with st.expander("Library diagnostics"):
        st.write({"Skipped": library.skipped, "Notices": library.notices, "Duplicate results": library.duplicate_documents})
    _json_download("Download Literature Index", catalog, "v3_literature_index.json")


def _jobs_page() -> None:
    st.title("Extraction Jobs")
    st.caption("Read-only status from local Batch state and collection files. Opening this page does not contact OpenAI.")
    jobs = discover_batch_jobs(WORKSPACE_ROOT / "output")
    if not jobs:
        st.info("No local v3 Batch jobs were found.")
        return
    st.dataframe([
        {
            "Job": row["batch"], "Status": str(row["status"]).title(), "Requests": row["request_count"],
            "Completed": row["completed"], "Failed": row["failed"], "Pending": row["pending"],
            "Estimated cost (USD)": row["estimated_cost_usd"],
        }
        for row in jobs
    ], use_container_width=True, hide_index=True)
    selected = next(row for row in jobs if row["batch"] == st.selectbox("Job details", [row["batch"] for row in jobs]))
    if selected["outcomes"]:
        st.dataframe([
            {"Article": row.get("document_id"), "Status": row.get("status"), "Result folder": row.get("output_root")}
            for row in selected["outcomes"]
        ], use_container_width=True, hide_index=True)
    with st.expander("Local job files"):
        st.code(selected["state_path"])
        if selected["summary_path"]:
            st.code(selected["summary_path"])


def _settings_page(settings: dict[str, Any]) -> None:
    st.title("Settings")
    st.caption("These settings control which existing v3 results enter the prediction library.")
    with st.form("workflow_settings"):
        roots_raw = st.text_area("V3 extraction result roots", value="\n".join(settings["extraction_roots"]), height=120)
        evidence_policy = st.selectbox(
            "Evidence policy", list(EVIDENCE_LABELS), index=list(EVIDENCE_LABELS).index(settings["evidence_policy"]),
            format_func=lambda key: EVIDENCE_LABELS[key],
        )
        review_policy = "extracted"
        st.info("Human review is used only for extraction accuracy statistics. Similarity, curves and simulations use the original extraction.")
        save = st.form_submit_button("Save Settings", type="primary")
    if save:
        value = {
            "extraction_roots": [line.strip() for line in roots_raw.splitlines() if line.strip()],
            "evidence_policy": evidence_policy, "review_policy": review_policy,
        }
        write_json(SETTINGS_PATH, value)
        st.session_state["v3_settings"] = value
        _load_library_cached.clear()
        _invalidate_downstream()
        st.success("Settings saved. Existing target predictions were cleared because their evidence contract changed.")
        st.rerun()
    st.divider()
    st.subheader("Human Review")
    st.write("Human review remains a separate application so validation work cannot be confused with target-trial prediction.")
    st.code("phase1-review-v3 --output-root /path/to/v3/results", language="bash")
    st.subheader("Data behavior")
    st.write("The workflow reads existing v3 literature and extraction results. Crawling and paid extraction remain separate command-line operations. Article posteriors are cached locally. Trial-plan edits save automatically on this computer and are restored when you reopen the website, until you replace or clear the draft. Prediction and simulation results can be saved or downloaded explicitly.")




def main() -> None:
    from phase1_web.multidrug_app import main as multidrug_main
    multidrug_main(sys.modules[__name__])


if __name__ == "__main__":
    main()
