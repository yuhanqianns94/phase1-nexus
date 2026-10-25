"""A ten-page, single-reviewer interface for v3 extraction outputs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import streamlit as st


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from phase1_extraction.artifacts import active_review_root, read_json  # noqa: E402
from phase1_extraction.exposures import exposure_text, exposure_schedule, dose_text, PRESENCE_LABELS, DOSE_STATUS_LABELS
from phase1_extraction.exposure_review import editor_rows, parse_editor_rows
from phase1_extraction.models import DoseExpression
from phase1_extraction.friendly_review import (  # noqa: E402
    CURVE_MODULE_ID,
    MODULES,
    MODULE_LABELS,
    CurvePointAnswer,
    FriendlyArticleReview,
    FriendlyReviewUnit,
    add_missing_curve_point,
    discover_review_articles,
    friendly_review_csv,
    load_current_saved_reviews,
    load_friendly_review_undo,
    load_or_create_friendly_review,
    mark_module_correct,
    restart_friendly_review,
    save_friendly_review,
    summarize_friendly_corpus,
    summarize_friendly_review,
    undo_friendly_review,
)


DECISION_LABELS = {
    "pending": "Not reviewed",
    "correct": "Correct and complete",
    "incorrect": "Needs correction",
    "spurious": "Not present in the source",
    "missing": "Missed by the model",
    "excluded": "Not assessed / not reported",
}

DENOMINATOR_LABELS = {
    "paired_dlt": "Paired DLT denominator",
    "dlt_evaluable": "DLT-evaluable N",
    "treated": "Treated N",
    "equivalent_treated": "Equivalent treated N",
    "design_assumed_cohort": "Inferred from the study design",
    "unknown": "Not specified",
}

MODULE_AUDIT_CATEGORIES = {
    "population": {"population"},
    "eligibility": {"eligibility"},
    "intervention": {"interventions"},
    "dosing": {"dosing"},
    "parts_arms": {"study_structure"},
    "design": {"design"},
    "population_counts": {"population_counts"},
    "dlt_definition": {"dlt_definition"},
    "dlt_window": {"dlt_window"},
    "curve_results": {"dlt_results", "dosing"},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    default_root = Path(os.environ.get("PHASE1_REVIEW_OUTPUT_ROOT", active_review_root()))
    parser.add_argument("--output-root", type=Path, action="append", dest="output_roots")
    args, _ = parser.parse_known_args()
    if not args.output_roots:
        args.output_roots = [default_root]
    return args


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _status_icon(review: FriendlyArticleReview, module_id: str) -> str:
    units = [unit for unit in review.units if unit.module_id == module_id]
    if review.module_complete.get(module_id) and all(unit.decision != "pending" for unit in units):
        return "✅"
    if any(unit.decision in {"incorrect", "spurious", "missing"} for unit in units):
        return "🟠"
    return "⬜"


def _int_or_none(value: str, label: str) -> int | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        parsed = int(cleaned)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} cannot be negative")
    return parsed


def _float_or_none(value: str, label: str) -> float | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric") from exc


def _source_document(paths) -> dict[str, Any]:
    document_ir_path = paths.document_ir
    if not document_ir_path.is_file() and paths.request_summary.is_file():
        summary = read_json(paths.request_summary)
        input_root = summary.get("input_article_root")
        if input_root:
            candidate = Path(input_root) / "document_ir.json"
            if candidate.is_file():
                document_ir_path = candidate
    if not document_ir_path.is_file():
        return {}
    return read_json(document_ir_path)


def _article_links(paths) -> list[tuple[str, str]]:
    identifiers = _source_document(paths).get("identifiers", [])
    pmc = paths.document_id.upper()
    doi = pmid = None
    for item in identifiers:
        name, value = str(item.get("name", "")).casefold(), str(item.get("value", "")).strip()
        if name in {"pmc", "pmcid"} and not re.fullmatch(r"PMC\d+", pmc):
            candidate = value.upper()
            pmc = candidate if candidate.startswith("PMC") else f"PMC{candidate}"
        elif name == "doi" and re.fullmatch(r"10\.\d{4,9}/\S+", value):
            doi = value
        elif name == "pmid" and value.isascii() and value.isdigit():
            pmid = value
    links = []
    if re.fullmatch(r"PMC\d+", pmc):
        links.append(("Open Article (PMC)", f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"))
    if doi:
        links.append(("Publisher / DOI", f"https://doi.org/{quote(doi, safe='/')}"))
    if pmid:
        links.append(("PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
    return links


def _source_index(paths) -> dict[str, dict[str, Any]]:
    raw = _source_document(paths)
    index: dict[str, dict[str, Any]] = {}
    for block in raw.get("blocks", []):
        index[block["block_id"]] = {"kind": "text", "text": block.get("text_raw") or block.get("text_normalized")}
    for page in raw.get("pages", []):
        index[page["page_id"]] = {
            "kind": "image",
            "image": page.get("image_path"),
            "label": f"PDF page {page.get('page_number')}",
        }
    for table in raw.get("tables", []):
        images = [fragment.get("image_path") for fragment in table.get("fragments", []) if fragment.get("image_path")]
        index[table["table_id"]] = {
            "kind": "table",
            "label": table.get("source_label") or table.get("caption") or "Source table",
            "text": table.get("caption"),
            "images": images,
        }
        for fragment in table.get("fragments", []):
            index[fragment["fragment_id"]] = {
                "kind": "image",
                "image": fragment.get("image_path"),
                "label": table.get("source_label") or "Source table",
            }
    return index


def _show_evidence(unit: FriendlyReviewUnit, source_index: dict[str, dict[str, Any]]) -> None:
    shown_text: set[str] = set()
    shown_images: set[str] = set()
    for locator in unit.evidence_text:
        if locator not in shown_text:
            st.caption(locator)
            shown_text.add(locator)
    for source_id in unit.evidence_source_ids:
        source = source_index.get(source_id)
        if not source:
            continue
        source_text = source.get("text")
        if source_text and source_text not in shown_text:
            st.markdown(f"> {source_text}")
            shown_text.add(source_text)
        images = source.get("images", [])
        if source.get("image"):
            images = [source["image"]]
        for image_path in images:
            if image_path and image_path not in shown_images and Path(image_path).is_file():
                st.image(image_path, caption=source.get("label"), use_container_width=True)
                shown_images.add(image_path)
    if not shown_text and not shown_images:
        st.caption("No directly displayable source excerpt is linked to this answer. Open the PDF to verify it.")


def _show_module_source_status(paths, module_id: str) -> None:
    if not paths.extraction_audit.is_file():
        return
    audit = read_json(paths.extraction_audit)
    categories = MODULE_AUDIT_CATEGORIES.get(module_id, set())
    findings = [
        item for item in audit.get("findings", [])
        if item.get("category") in categories or item.get("category") == "materials"
    ]
    if not findings:
        st.caption("Source material is available in the frozen request; clinical correctness still requires review.")
        return
    for item in findings:
        severity = item.get("severity")
        label = (
            "Missing or unread source material"
            if severity in {"blocking", "source_material_required"}
            else "Targeted source reread needed"
        )
        st.warning(f"{label}: {item.get('message')}")


def _save_text_unit(
    paths,
    review: FriendlyArticleReview,
    unit: FriendlyReviewUnit,
    decision: str,
    gold_answer: str,
    note: str,
    completeness_checked: bool,
) -> str | None:
    if decision == "correct":
        if not unit.predicted_answer:
            return "A missing model answer cannot be marked correct. Enter the correct answer and select 'Missed by the model'."
        unit.gold_answer = unit.predicted_answer
    elif decision in {"incorrect", "missing"}:
        if not gold_answer.strip():
            return "Enter the complete answer supported by the source."
        unit.gold_answer = gold_answer.strip()
    else:
        unit.gold_answer = None
    unit.decision = decision
    unit.note = note.strip() or None
    from datetime import datetime, timezone

    unit.updated_at_utc = datetime.now(timezone.utc).isoformat()
    review.module_complete[unit.module_id] = completeness_checked
    save_friendly_review(paths, review)
    return None


def _point_for_edit(unit: FriendlyReviewUnit) -> CurvePointAnswer:
    point = unit.gold_point if unit.decision in {"incorrect", "missing"} and unit.gold_point else unit.predicted_point
    if point is not None:
        return point.model_copy(deep=True)
    return CurvePointAnswer(series_label="", dose_unit="", dose_raw="")


def _save_curve_unit(
    paths,
    review: FriendlyArticleReview,
    unit: FriendlyReviewUnit,
    decision: str,
    point: CurvePointAnswer,
    note: str,
    completeness_checked: bool,
) -> None:
    unit.decision = decision
    if decision == "correct":
        unit.gold_point = unit.predicted_point.model_copy(deep=True) if unit.predicted_point else None
    elif decision in {"incorrect", "missing"}:
        unit.gold_point = point
    else:
        unit.gold_point = None
    unit.note = note.strip() or None
    from datetime import datetime, timezone

    unit.updated_at_utc = datetime.now(timezone.utc).isoformat()
    review.module_complete[CURVE_MODULE_ID] = completeness_checked
    save_friendly_review(paths, review)


def _matrix_data(units: list[FriendlyReviewUnit]) -> tuple[list[tuple[str, list[dict[str, str]]]], list[str]]:
    points: list[tuple[CurvePointAnswer, FriendlyReviewUnit]] = []
    for unit in units:
        if unit.decision == "incorrect" and unit.gold_point is not None:
            point = unit.gold_point
        elif unit.origin == "manual_missing":
            point = unit.gold_point
        else:
            point = unit.predicted_point
        if point is not None and unit.decision != "excluded":
            points.append((point, unit))
    by_unit: dict[str, list[tuple[CurvePointAnswer, FriendlyReviewUnit]]] = {}
    for point, unit in points:
        by_unit.setdefault(point.dose_unit or "Unit not reported", []).append((point, unit))
    tables: list[tuple[str, list[dict[str, str]]]] = []
    notes: list[str] = []
    for dose_unit, grouped in sorted(by_unit.items()):
        doses = sorted({point.dose_value for point, _unit in grouped if point.dose_value is not None})
        raw_doses = sorted({point.dose_raw for point, _unit in grouped if point.dose_value is None})
        columns = [f"{dose:g}" for dose in doses] + raw_doses
        rows: dict[str, dict[str, str]] = {}
        for point, unit in grouped:
            row = rows.setdefault(point.series_label, {"Dose series": point.series_label, **{column: "—" for column in columns}})
            dose_label = f"{point.dose_value:g}" if point.dose_value is not None else point.dose_raw
            marker = "*" if point.value_basis not in {"reported", "reported / reported"} else ""
            if point.curve_status != "included":
                marker += "!"
                notes.append(f"{point.series_label} {dose_label}: excluded from the default prediction curve ({point.curve_status}).")
            if point.association_note:
                notes.append(f"{point.series_label} {dose_label}: {point.association_note}.")
            if unit.origin == "manual_missing":
                marker += "+"
            cell = f"{point.y if point.y is not None else '?'} / {point.n if point.n is not None else '?'}{marker}"
            if row[dose_label] != "—":
                row[dose_label] += f"; {cell}"
            else:
                row[dose_label] = cell
            if point.treated_n is not None and point.n is not None and point.treated_n != point.n:
                notes.append(f"{point.series_label} {dose_label}: N={point.n} is used; treated N={point.treated_n}.")
            normalized = f"{point.dose_value:g} {point.dose_unit}" if point.dose_value is not None else point.dose_unit
            if point.dose_raw.strip() and point.dose_raw.strip().casefold() != normalized.strip().casefold():
                notes.append(f"{point.series_label} {dose_label}: source dose “{point.dose_raw}”; normalized dose {normalized}.")
        tables.append((dose_unit, list(rows.values())))
    return tables, list(dict.fromkeys(notes))


def _widget_key(name: str) -> str:
    scope = st.session_state["review_widget_scope"]
    revision = st.session_state.get(f"review_widget_revision_{scope}", 0)
    return f"review_{scope}_{revision}_{name}"


def _queue_page(document_id: str, module_id: str) -> None:
    st.session_state[_widget_key(f"pending_review_page_{document_id}")] = module_id


def _reload_saved_review(document_id: str, module_id: str, *, message: str | None = None) -> None:
    # A new widget revision loads saved values into every form after undo/reset.
    # The selected page is applied on the next run, before its widget exists.
    scope = st.session_state["review_widget_scope"]
    revision_key = f"review_widget_revision_{scope}"
    st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1
    _queue_page(document_id, module_id)
    if message:
        st.session_state[f"review_notice_{scope}"] = message
    st.rerun()


def _next_module(module_id: str) -> str | None:
    module_ids = [item[0] for item in MODULES]
    index = module_ids.index(module_id)
    return module_ids[index + 1] if index + 1 < len(module_ids) else None


def _render_page_navigation(review: FriendlyArticleReview, module_id: str, *, location: str) -> None:
    module_ids = [item[0] for item in MODULES]
    page_index = module_ids.index(module_id)
    left, center, right = st.columns([1, 3, 1])
    if left.button("← Previous", disabled=page_index == 0, key=_widget_key(f"previous_{location}_{review.document_id}_{module_id}")):
        _queue_page(review.document_id, module_ids[page_index - 1])
        st.rerun()
    center.caption(f"Section {page_index + 1} of {len(module_ids)}")
    if right.button("Next →", disabled=page_index == len(module_ids) - 1, key=_widget_key(f"next_{location}_{review.document_id}_{module_id}")):
        _queue_page(review.document_id, module_ids[page_index + 1])
        st.rerun()


def _save_and_advance(paths, review: FriendlyArticleReview, module_id: str) -> None:
    _reload_saved_review(review.document_id, _next_module(module_id) or module_id)


def _configuration_table(units: list[FriendlyReviewUnit]) -> str:
    """Each observation stays one row; user/source text is always escaped."""
    body = []
    windows: dict[str, str] = {}
    for number, unit in enumerate(units, 1):
        if unit.decision == "excluded":
            continue
        point = _point_for_edit(unit)
        esc = lambda value: html.escape(str(value or ""))
        group = esc(point.series_label or point.arm_label or point.part_label)
        drugs = "".join(
            f'<div class="exposure-drug">{esc(exposure_text(e))}'
            + (f'<div class="exposure-muted">{esc(exposure_schedule(e))}</div>' if exposure_schedule(e) else '')
            + '</div>' for e in point.exposures)
        if not drugs:
            drugs = esc(f"{point.dose_value:g} {point.dose_unit}" if point.dose_value is not None else point.dose_raw)
            drugs += '<div class="exposure-muted">Legacy result: complete drug configuration not extracted</div>'
        status = DECISION_LABELS[unit.decision]
        if unit.decision == "incorrect" and unit.gold_point:
            status = "Correction saved"
        if point.exposure_coverage != "complete" and point.exposures:
            status += f" · exposure coverage: {point.exposure_coverage}"
        if point.exposure_issues:
            status += " · check exposure mapping"
        if point.curve_status != "included":
            status += " · excluded from curve"
        count = f"{point.y if point.y is not None else '?'} / {point.n if point.n is not None else '?'}"
        denominator = DENOMINATOR_LABELS.get(point.denominator_type, point.denominator_type)
        if any(part.strip() != "reported" for part in point.value_basis.split("/")):
            denominator += " · " + point.value_basis.replace("_", " ")
        window = "Not reported"
        if point.window_label:
            window = windows.setdefault(point.window_label, f"W{len(windows) + 1}")
        body.append(f'<tr><td><b>{number}</b><div class="exposure-muted">{group}</div></td>'
                    f'<td>{drugs}</td><td class="exposure-count"><b>{esc(count)}</b>'
                    f'<div class="exposure-muted">{esc(denominator)}</div></td>'
                    f'<td>{esc(window)}</td><td>{esc(status)}</td></tr>')
    if not body:
        return ""
    return '''<style>
.exposure-scroll{overflow-x:auto;margin:12px 0 20px}
.exposure-table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.55;min-width:800px}
.exposure-table th{text-align:left;background:rgba(110,150,150,.12);padding:12px}
.exposure-table td{padding:12px;vertical-align:top;border-bottom:1px solid rgba(120,140,140,.22);overflow-wrap:anywhere}
.exposure-table th:nth-child(1){width:18%}.exposure-table th:nth-child(2){width:47%}
.exposure-table th:nth-child(3){width:12%}.exposure-table th:nth-child(4){width:8%}
.exposure-drug+.exposure-drug{margin-top:7px;padding-top:7px;border-top:1px dashed rgba(120,140,140,.2)}
.exposure-muted{font-size:12px;opacity:.72;margin-top:5px}.exposure-count b{font-size:18px;white-space:nowrap}
</style><div class="exposure-scroll"><table class="exposure-table"><thead><tr>
<th>Cohort / group</th><th>Complete treatment configuration</th><th>DLT / N</th><th>DLT window</th><th>Review</th>
</tr></thead><tbody>''' + "".join(body) + '</tbody></table></div>' + (
        '<details><summary>DLT window definitions</summary>' + ''.join(
            f'<p><b>{html.escape(label)}</b>: {html.escape(value)}</p>' for value, label in windows.items()
        ) + '</details>' if windows else '')


def _exposure_editor(point: CurvePointAnswer, *, key: str):
    st.markdown("**Drugs and concomitant medication for this cohort**")
    st.caption("One row per drug or class. Used + unknown dose is different from not used. "
               "For absence, clear the dose and select not applicable. All rows share the one DLT/N above.")
    rows = editor_rows(point.exposures)
    if not rows:
        rows = [{"Record": "", "Drug / class": "", "Type": "drug", "Role": "escalated", "Use": "unknown",
                 "Dose status": "unknown", "Dose value": None, "Unit": "", "Source dose": "",
                 "Formulation": "", "Schedule": "", "Reporting": "cohort", "Exposed N": None, "Note": ""}]
    return st.data_editor(rows, num_rows="dynamic", hide_index=True, use_container_width=True, key=key,
        column_config={
            "Record": None,
            "Drug / class": st.column_config.TextColumn(width="medium"),
            "Type": st.column_config.SelectboxColumn(options=["drug", "drug_class", "non_drug", "unclear"]),
            "Role": st.column_config.SelectboxColumn(options=["escalated", "fixed_backbone", "combination_partner", "conditioning", "supportive", "concomitant", "radiation", "maintenance", "other", "unclear"]),
            "Use": st.column_config.SelectboxColumn(options=list(PRESENCE_LABELS)),
            "Dose status": st.column_config.SelectboxColumn(options=list(DOSE_STATUS_LABELS)),
            "Dose value": st.column_config.NumberColumn(min_value=0.0),
            "Research order": st.column_config.NumberColumn(min_value=1, step=1, help="Stable order across this regimen; first three receive interaction terms."),
            "Research role": st.column_config.SelectboxColumn(options=["primary", "co_primary", "partner", "background", "unclear"]),
            "Dose interpretation": st.column_config.SelectboxColumn(options=["scalar", "alternative_unknown", "sequential", "subgroups", "patient_conditional", "unknown"]),
            "Exposed N": st.column_config.NumberColumn(min_value=0, step=1),
            "Reporting": st.column_config.SelectboxColumn(options=["cohort", "pooled", "unclear"]),
            "Schedule": st.column_config.TextColumn(width="large"),
        })


def _sync_primary_dose(corrected: CurvePointAnswer, previous: CurvePointAnswer) -> None:
    candidates = [e for e in corrected.exposures if e.component_id and e.component_id == corrected.varying_component_id]
    if not candidates:
        candidates = [e for e in corrected.exposures if e.role == "escalated"]
    if len(candidates) != 1:
        return
    primary = candidates[0]
    if primary.presence == "absent":
        raise ValueError("The primary drug of an administered dose-DLT point cannot be marked absent.")
    scalar_changed = (corrected.dose_value, corrected.dose_unit) != (previous.dose_value, previous.dose_unit)
    previous_exposure = next((e for e in previous.exposures if e.exposure_id == primary.exposure_id), None)
    exposure_changed = previous_exposure is None or primary.dose != previous_exposure.dose
    if scalar_changed:
        if exposure_changed and primary.dose and (primary.dose.value, primary.dose.unit_normalized or primary.dose.unit_raw) != (corrected.dose_value, corrected.dose_unit):
            raise ValueError("Primary dose and the drug exposure row disagree. Enter the same dose in both.")
        primary.dose = DoseExpression(raw_expression=f"{corrected.dose_value:g} {corrected.dose_unit}" if corrected.dose_value is not None else corrected.dose_raw,
            value_kind="scalar" if corrected.dose_value is not None else "unparsed", value=corrected.dose_value,
            unit_raw=corrected.dose_unit, unit_normalized=corrected.dose_unit)
        primary.dose_status = "reported"
    if primary.dose:
        corrected.dose_value = primary.dose.value if primary.dose.value_kind == "scalar" else None
        corrected.dose_unit = primary.dose.unit_normalized or primary.dose.unit_raw or "Unit not reported"
        corrected.dose_raw = primary.dose.raw_expression


def _render_curve_module(paths, review: FriendlyArticleReview, source_index: dict[str, dict[str, Any]]) -> None:
    st.header(f"{_status_icon(review, CURVE_MODULE_ID)} {MODULE_LABELS[CURVE_MODULE_ID]}")
    _show_module_source_status(paths, CURVE_MODULE_ID)
    st.caption("One row = one cohort's complete treatment configuration and one DLT/N. "
               "true = used; false = explicitly not used; unknown = not established. [class] identifies a drug category.")
    curve_units = [unit for unit in review.units if unit.module_id == CURVE_MODULE_ID]
    table_html = _configuration_table(curve_units)
    if table_html:
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.warning("No actual dose-DLT points were extracted. Check the source and add any points the model missed.")
    with st.expander("Dose matrix and denominator notes (summary view)", expanded=False):
        tables, notes = _matrix_data(curve_units)
        st.caption("The complete configuration table above is the review record. Matrix rows are grouped by the original series.")
        for dose_unit, rows in tables:
            st.markdown(f"**Primary dose unit: {dose_unit}**")
            st.dataframe(rows, use_container_width=True, hide_index=True)
        for note in notes:
            st.caption("* " + note)
    if any((unit.predicted_point and unit.predicted_point.value_basis not in {"reported", "reported / reported"}) for unit in curve_units):
        st.caption("* At least one value in this point was inferred or deterministically derived. Review its evidence below.")
    if any((unit.predicted_point and unit.predicted_point.curve_status != "included") for unit in curve_units):
        st.caption("! This actual cohort is currently excluded from the default prediction curve and still requires review.")
    if any(unit.origin == "manual_missing" and unit.decision != "excluded" for unit in curve_units):
        st.caption("+ This point was added manually as a model omission.")

    if st.button("Confirm Entire Table as Correct", type="primary", key=_widget_key(f"curve_all_{review.document_id}")):
        mark_module_correct(review, CURVE_MODULE_ID)
        save_friendly_review(paths, review)
        _reload_saved_review(review.document_id, CURVE_MODULE_ID)

    prediction_units = [unit for unit in curve_units if unit.origin == "prediction"]
    with st.expander("Correct a Point or View Its Evidence", expanded=False):
        if not prediction_units:
            st.info("There are no model-generated points.")
        else:
            selected_id = st.selectbox(
                "Point",
                [unit.review_unit_id for unit in prediction_units],
                format_func=lambda value: next(unit.display_answer() for unit in prediction_units if unit.review_unit_id == value),
                key=_widget_key(f"curve_select_{review.document_id}"),
            )
            unit = next(unit for unit in prediction_units if unit.review_unit_id == selected_id)
            point = _point_for_edit(unit)
            if unit.predicted_point is not None:
                st.caption(
                    f"Source dose: {unit.predicted_point.dose_raw} · "
                    f"Normalized: {unit.predicted_point.dose_value if unit.predicted_point.dose_value is not None else '?'} "
                    f"{unit.predicted_point.dose_unit} · Scope: {unit.predicted_point.series_label}"
                )
            st.caption(f"Saved assessment: {DECISION_LABELS[unit.decision]}")
            for exposure in point.exposures:
                with st.container(border=True):
                    st.write(exposure_text(exposure))
                    st.caption(exposure_schedule(exposure) or "Schedule not reported")
                    if exposure.note:
                        st.caption(exposure.note)
            for problem in point.exposure_issues:
                st.warning(problem)
            with st.form(_widget_key(f"curve_edit_{unit.review_unit_id}")):
                choices = ["pending", "correct", "incorrect", "spurious", "excluded"]
                decision = st.radio(
                    "Assessment",
                    choices,
                    index=choices.index(unit.decision) if unit.decision in choices else 0,
                    format_func=lambda value: DECISION_LABELS[value],
                    horizontal=True,
                )
                c1, c2, c3 = st.columns(3)
                series_label = c1.text_input("Regimen / patient group", value=point.series_label)
                dose_value = c2.text_input("Dose value", value="" if point.dose_value is None else f"{point.dose_value:g}")
                dose_unit = c3.text_input("Dose unit", value=point.dose_unit)
                c4, c5, c6 = st.columns(3)
                n_value = c4.text_input("Denominator N", value="" if point.n is None else str(point.n))
                y_value = c5.text_input("Patients with DLT", value="" if point.y is None else str(point.y))
                denominator_type = c6.selectbox(
                    "Denominator type",
                    list(DENOMINATOR_LABELS),
                    index=list(DENOMINATOR_LABELS).index(point.denominator_type) if point.denominator_type in DENOMINATOR_LABELS else len(DENOMINATOR_LABELS) - 1,
                    format_func=lambda value: DENOMINATOR_LABELS[value],
                )
                window_label = st.text_input("DLT assessment window", value=point.window_label or "")
                exposure_edits = _exposure_editor(point, key=_widget_key(f"exposures_{unit.review_unit_id}"))
                note = st.text_input("Note (optional)", value=unit.note or "")
                complete = st.checkbox("I checked this table for missing and extra points", value=review.module_complete.get(CURVE_MODULE_ID, False))
                submitted = st.form_submit_button("Save This Point", type="primary")
            if submitted:
                try:
                    corrected = point.model_copy(
                        update={
                            "series_label": series_label.strip(),
                            "dose_value": _float_or_none(dose_value, "Dose"),
                            "dose_unit": dose_unit.strip(),
                            "n": _int_or_none(n_value, "Denominator N"),
                            "y": _int_or_none(y_value, "Patients with DLT"),
                            "denominator_type": denominator_type,
                            "window_label": window_label.strip() or None,
                        }
                    )
                    if decision == "incorrect":
                        corrected.exposures = parse_editor_rows(exposure_edits, point.exposures)
                        _sync_primary_dose(corrected, point)
                        corrected.exposure_issues = []
                    if corrected.n is not None and corrected.y is not None and corrected.y > corrected.n:
                        raise ValueError("Patients with DLT cannot exceed denominator N")
                    _save_curve_unit(paths, review, unit, decision, corrected, note, complete)
                    _reload_saved_review(review.document_id, CURVE_MODULE_ID)
                except ValueError as exc:
                    st.error(str(exc))
            st.markdown("**Source evidence for this point**")
            if st.checkbox("Show source evidence", key=_widget_key(f"curve_evidence_{unit.review_unit_id}")):
                _show_evidence(unit, source_index)

    with st.expander("Add a Dose-DLT Point Missed by the Model", expanded=False):
        with st.form(_widget_key(f"add_curve_{review.document_id}"), clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            series_label = c1.text_input("Regimen / patient group", placeholder="Example: CPE -EIASD")
            dose_value = c2.text_input("Dose value", placeholder="Example: 750")
            dose_unit = c3.text_input("Dose unit", placeholder="Example: mg/day")
            c4, c5, c6 = st.columns(3)
            n_value = c4.text_input("Denominator N")
            y_value = c5.text_input("Patients with DLT")
            denominator_type = c6.selectbox("Denominator type", list(DENOMINATOR_LABELS), format_func=lambda value: DENOMINATOR_LABELS[value])
            window_label = st.text_input("DLT assessment window (optional)")
            missing_exposures = _exposure_editor(CurvePointAnswer(series_label="", dose_unit="", dose_raw=""),
                                                 key=_widget_key(f"missing_exposures_{review.document_id}"))
            note = st.text_input("Source location or note (optional)")
            add_clicked = st.form_submit_button("Add Missing Point", type="primary")
        if add_clicked:
            try:
                point = CurvePointAnswer(
                    series_label=series_label.strip(),
                    dose_value=_float_or_none(dose_value, "Dose"),
                    dose_unit=dose_unit.strip(),
                    dose_raw=f"{dose_value.strip()} {dose_unit.strip()}".strip(),
                    n=_int_or_none(n_value, "Denominator N"),
                    y=_int_or_none(y_value, "Patients with DLT"),
                    denominator_type=denominator_type,
                    window_label=window_label.strip() or None,
                    value_basis="Manually added",
                    exposures=parse_editor_rows(missing_exposures, []),
                )
                if not point.series_label or not point.dose_unit or point.dose_value is None or point.n is None or point.y is None:
                    raise ValueError("Enter a patient group, dose, unit, N, and number of patients with DLT")
                if point.y > point.n:
                    raise ValueError("Patients with DLT cannot exceed denominator N")
                add_missing_curve_point(review, point, note=note)
                save_friendly_review(paths, review)
                _reload_saved_review(review.document_id, CURVE_MODULE_ID)
            except ValueError as exc:
                st.error(str(exc))

    manual_units = [unit for unit in curve_units if unit.origin == "manual_missing"]
    if manual_units:
        with st.expander("Manage Manually Added Points", expanded=False):
            selected_missing = st.selectbox(
                "Manually added point",
                [unit.review_unit_id for unit in manual_units],
                format_func=lambda value: next(unit.display_answer(gold=True) for unit in manual_units if unit.review_unit_id == value),
                key=_widget_key("manual_point"),
            )
            if st.button("Delete This Manually Added Point", key=_widget_key("delete_manual_point")):
                review.units = [unit for unit in review.units if unit.review_unit_id != selected_missing]
                save_friendly_review(paths, review)
                _reload_saved_review(review.document_id, CURVE_MODULE_ID)


def _render_text_module(
    paths,
    review: FriendlyArticleReview,
    source_index: dict[str, dict[str, Any]],
    module_id: str,
) -> None:
    unit = next(
        unit
        for unit in review.units
        if unit.module_id == module_id and unit.unit_kind == "text_field" and unit.origin == "prediction"
    )
    st.header(f"{_status_icon(review, module_id)} {MODULE_LABELS[module_id]}")
    _show_module_source_status(paths, module_id)
    st.caption("One page is one major metric. Compare the extracted answer with the source, then save one assessment.")
    heading_metric, heading_scope = st.columns([2, 3])
    heading_metric.markdown(f"**Metric**  \n{MODULE_LABELS[module_id]}")
    heading_scope.markdown("**Scope**  \nCurrent article")
    st.markdown("**Extracted answer**")
    with st.container(border=True):
        if unit.predicted_answer:
            st.markdown(unit.predicted_answer)
        else:
            st.warning("The model did not extract this metric. If the source contains it, enter the complete answer below and select 'Missed by the model'.")

    st.caption(f"Saved assessment: {DECISION_LABELS[unit.decision]}")
    if unit.decision in {"incorrect", "missing"} and unit.gold_answer:
        st.markdown("**Saved human answer**")
        with st.container(border=True):
            st.markdown(unit.gold_answer)
    if unit.note:
        st.caption(f"Saved note: {unit.note}")

    if st.button(
        "Confirm as Correct & Next",
        type="primary",
        key=_widget_key(f"correct_{review.document_id}_{module_id}"),
        disabled=not bool(unit.predicted_answer),
    ):
        mark_module_correct(review, module_id)
        save_friendly_review(paths, review)
        _save_and_advance(paths, review, module_id)

    with st.expander("Correct This Answer or Mark Another Outcome", expanded=unit.decision in {"incorrect", "missing"}):
        choices = ["incorrect", "spurious", "excluded"] if unit.predicted_answer else ["missing", "excluded"]
        current_decision = unit.decision if unit.decision in choices else choices[0]
        with st.form(_widget_key(f"text_form_{review.document_id}_{module_id}")):
            decision = st.radio(
                "Assessment",
                choices,
                index=choices.index(current_decision),
                format_func=lambda value: DECISION_LABELS[value],
                horizontal=True,
            )
            gold = st.text_area(
                "Human correct answer",
                value=unit.gold_answer or "",
                height=220,
                help="Required for 'Needs correction' and 'Missed by the model'. Enter the complete corrected answer for this metric.",
            )
            note = st.text_input("Source location or note (optional)", value=unit.note or "")
            complete = st.checkbox(
                "I checked the source for content the model may have missed",
                value=review.module_complete.get(module_id, False),
            )
            save_column, next_column = st.columns(2)
            stay_submitted = save_column.form_submit_button("Save Review")
            submitted = next_column.form_submit_button("Save Review & Next", type="primary")
        if submitted or stay_submitted:
            error = _save_text_unit(paths, review, unit, decision, gold, note, complete)
            if error:
                st.error(error)
            elif submitted:
                _save_and_advance(paths, review, module_id)
            else:
                _reload_saved_review(review.document_id, module_id, message="Review saved. This replaces the previous assessment for this section.")

    with st.expander("Source Evidence", expanded=False):
        _show_evidence(unit, source_index)


def _render_sidebar_summary(output_roots: list[Path], paths, review: FriendlyArticleReview) -> None:
    summary = summarize_friendly_review(review)
    st.sidebar.progress(
        summary.completed_modules / summary.total_modules,
        text=f"Reviewed {summary.completed_modules}/{summary.total_modules}",
    )
    with st.sidebar.expander("Accuracy and Downloads", expanded=False):
        st.markdown("**Text metrics**")
        st.caption(
            f"Precision {_pct(summary.text_fields.counts.precision)} · "
            f"Recall {_pct(summary.text_fields.counts.recall)} · "
            f"Pending {summary.text_fields.counts.pending}"
        )
        st.markdown("**Dose-DLT points**")
        st.caption(
            f"Precision {_pct(summary.curve_points.counts.precision)} · "
            f"Recall {_pct(summary.curve_points.counts.recall)} · "
            f"Pending {summary.curve_points.counts.pending}"
        )
        rows = [
            {"Metric": label, "TP": counts.tp, "FP": counts.fp, "FN": counts.fn, "Pending": counts.pending}
            for label, counts in summary.by_module.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Review Details (CSV)",
            data=friendly_review_csv(review).encode("utf-8-sig"),
            file_name=f"{review.document_id}_friendly_review_units.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download Accuracy Summary (JSON)",
            data=json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
            file_name=f"{review.document_id}_friendly_review_summary.json",
            mime="application/json",
            use_container_width=True,
        )
        saved_reviews, stale_reviews = load_current_saved_reviews(output_roots)
        corpus = summarize_friendly_corpus(saved_reviews)
        st.caption(f"Saved reviews: {corpus.article_count}; fully complete: {corpus.fully_completed_articles}.")
        if stale_reviews:
            st.warning(f"{stale_reviews} saved review(s) refer to changed extraction results and are excluded from the corpus summary.")
    if paths.friendly_human_review.is_file():
        st.sidebar.caption(f"Saved to: {paths.friendly_human_review}")
    else:
        st.sidebar.caption("No decisions saved yet. The first save creates friendly_human_review.json.")


def _render_review_actions(paths, review: FriendlyArticleReview, module_id: str) -> None:
    st.sidebar.caption("Use Previous or choose any section to revisit a saved answer. Saving again replaces its previous assessment.")
    undo = load_friendly_review_undo(paths)
    if st.sidebar.button(
        "Undo Last Save",
        disabled=undo is None,
        use_container_width=True,
        key=_widget_key("undo"),
        help="Restore the state before the most recent save or restart for this article, including its scores. One level of undo is available.",
    ):
        try:
            _restored, changed_module = undo_friendly_review(paths)
        except ValueError as exc:
            st.sidebar.error(str(exc))
        else:
            _reload_saved_review(review.document_id, changed_module or module_id, message="Last save undone. The previous answers and scores have been restored.")
    with st.sidebar.expander("Restart This Article", expanded=False):
        st.caption("Start all 10 sections again. This clears saved assessments, human answers, notes and manually added points for this article. The review files and scores are overwritten; the extraction stays available.")
        if st.button("Restart Review for This Article", key=_widget_key("restart"), use_container_width=True):
            restart_friendly_review(paths, reviewer=review.reviewer)
            _reload_saved_review(review.document_id, MODULES[0][0], message="Article review restarted at section 1. Previous review results have been cleared. Undo Last Save can restore them until your next save.")


def render_review_page(output_root: Path | list[Path], *, show_title: bool = True) -> None:
    """Render the existing ten-section reviewer inside either Streamlit app."""
    if show_title:
        st.title("PHASE I NEXUS · Evidence Review")
        st.caption("Ten review sections: nine clinical answers and one dose-DLT table. Each page contains one major metric.")
    output_roots = [output_root] if isinstance(output_root, Path) else list(output_root)
    articles = discover_review_articles(output_roots)
    if not articles:
        st.info("No reviewable article was found in the configured extraction output directories.")
        return
    document_counts: dict[str, int] = {}
    for article_paths in articles:
        document_counts[article_paths.document_id] = document_counts.get(article_paths.document_id, 0) + 1
    by_label = {
        (
            article_paths.document_id
            if document_counts[article_paths.document_id] == 1
            else f"{article_paths.document_id} · {article_paths.root.name}"
        ): article_paths
        for article_paths in articles
    }
    selected_label = st.sidebar.selectbox("Article", list(by_label), index=0)
    paths = by_label[selected_label]
    document_id = paths.document_id
    review = load_or_create_friendly_review(paths)
    source_index = _source_index(paths)
    scope = hashlib.sha256(f"{paths.root.resolve()}:{review.source_fingerprint}".encode()).hexdigest()[:16]
    st.session_state["review_widget_scope"] = scope
    notice = st.session_state.pop(f"review_notice_{scope}", None)
    if notice:
        st.success(notice)
    links = _article_links(paths)
    for label, url in links:
        st.sidebar.link_button(label, url, use_container_width=True)
    if not links:
        st.sidebar.caption("No online article identifier is available in the saved source metadata.")

    reviewer = st.sidebar.text_input("Reviewer", value=review.reviewer, placeholder="Name or initials", key=_widget_key("reviewer"))
    if st.sidebar.button("Save Reviewer", use_container_width=True, key=_widget_key("save_reviewer")):
        review.reviewer = reviewer.strip()
        save_friendly_review(paths, review)
        _reload_saved_review(document_id, st.session_state.get(_widget_key(f"review_page_{document_id}"), MODULES[0][0]), message="Reviewer saved.")
    review.reviewer = reviewer.strip()

    module_ids = [item[0] for item in MODULES]
    page_key = _widget_key(f"review_page_{document_id}")
    pending_key = _widget_key(f"pending_review_page_{document_id}")
    pending_page = st.session_state.pop(pending_key, None)
    if pending_page in module_ids:
        st.session_state[page_key] = pending_page
    selected_module = st.sidebar.selectbox(
        "Review section",
        module_ids,
        key=page_key,
        format_func=lambda value: f"{module_ids.index(value) + 1}. {_status_icon(review, value)} {MODULE_LABELS[value]}",
    )
    _render_sidebar_summary(output_roots, paths, review)
    _render_review_actions(paths, review, selected_module)

    _render_page_navigation(review, selected_module, location="top")
    st.divider()
    if selected_module == CURVE_MODULE_ID:
        _render_curve_module(paths, review, source_index)
    else:
        _render_text_module(paths, review, source_index, selected_module)
    st.divider()
    _render_page_navigation(review, selected_module, location="bottom")


def main() -> None:
    st.set_page_config(page_title="PHASE I NEXUS · Evidence Review", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { line-height: 1.65; }
        [data-testid="stMarkdownContainer"] ul { margin-bottom: 1.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    args = _args()
    output_text = st.sidebar.text_area(
        "Extraction output directories",
        value="\n".join(str(path) for path in args.output_roots),
        help="Enter one directory per line. Reviews are saved beside each article's extraction results.",
    )
    output_roots = [Path(line.strip()).expanduser() for line in output_text.splitlines() if line.strip()]
    render_review_page(output_roots)


if __name__ == "__main__":
    main()
