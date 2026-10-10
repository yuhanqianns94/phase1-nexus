"""Human-readable, in-memory exports for target prediction and BOIN results."""

from __future__ import annotations

import csv
import html
import io
from typing import Any, Mapping

from phase1_simulation.service import DEFAULT_RANKING_PRIORITY, SIMULATION_MODEL_VERSION


_RANKING_LABELS = {
    "correct_selection_probability": "Closest-to-target selection (higher)",
    "overtoxic_selection_probability": "Selection above target DLT (lower)",
    "early_stop_probability": "Safety early stopping (lower)",
    "mean_total_n": "Mean sample size (lower)",
}


def prediction_csv(prediction: Mapping[str, Any]) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["method", "dose", "dose_unit", "posterior_mean", "low_95", "high_95"])
    unit = (prediction.get("target") or {}).get("dose_unit", "")
    for method_key, method in (prediction.get("methods") or {}).items():
        if method.get("status") != "fitted":
            continue
        for row in method.get("summary", []):
            writer.writerow([method_key, row.get("dose"), unit, row.get("posterior_mean"), row.get("low_95"), row.get("high_95")])
    return stream.getvalue().encode("utf-8-sig")


def simulation_csv(simulation: Mapping[str, Any]) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["dose", "selection_probability", "probability_closest_to_target", "mean_patients", "mean_dlts"])
    for values in zip(
        simulation.get("doses", []), simulation.get("selection_probability", []),
        simulation.get("true_mtd_probability", []), simulation.get("mean_patients_by_dose", []),
        simulation.get("mean_dlts_by_dose", []), strict=True,
    ):
        writer.writerow(values)
    return stream.getvalue().encode("utf-8-sig")


def analysis_html(bundle: Mapping[str, Any]) -> bytes:
    prediction = bundle.get("prediction") or {}
    target = prediction.get("target") or {}
    simulation = bundle.get("simulation") or {}
    comparison = bundle.get("design_comparison") or {}
    active_method = bundle.get("active_method") or ""
    method = (prediction.get("methods") or {}).get(active_method, {})
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row.get("dose"), f"{100 * row.get('posterior_mean', 0):.1f}%",
            f"{100 * row.get('low_95', 0):.1f}%–{100 * row.get('high_95', 0):.1f}%",
        )) + "</tr>"
        for row in method.get("summary", [])
    )
    simulation_section = ""
    if simulation:
        engine_version = simulation.get("simulation_model_version")
        engine_note = (str(engine_version) if engine_version == SIMULATION_MODEL_VERSION else
                       "Legacy saved engine result; rerun to evaluate the current cohort-only BOIN engine.")
        simulation_section = f"""
        <h2>BOIN simulation</h2>
        <p>Engine: {html.escape(engine_note)}</p>
        <p>Closest-to-target selection: {100 * simulation.get('correct_selection_probability', 0):.1f}% ·
        Selection above target DLT: {100 * simulation.get('overtoxic_selection_probability', 0):.1f}% ·
        No selection: {100 * simulation.get('no_selection_probability', 0):.1f}% ·
        Mean sample size: {simulation.get('mean_total_n', 0):.1f}</p>
        """
    comparison_section = ""
    if comparison:
        comparison_note = (str(comparison["simulation_model_version"])
                           if comparison.get("simulation_model_version") == SIMULATION_MODEL_VERSION else
                           "Legacy saved comparison; rerun with the current cohort-only BOIN engine.")
        comparison_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
                index + 1, row.get("candidate_label"), row.get("cohort_size"), row.get("max_sample_size"),
                f"{100 * row.get('correct_selection_probability', 0):.1f}%",
                f"{100 * row.get('overtoxic_selection_probability', 0):.1f}%",
                f"{100 * row.get('no_selection_probability', 0):.1f}%",
            )) + "</tr>"
            for index, row in enumerate(comparison.get("ranked", []))
        )
        priority = comparison.get("ranking_priority", list(DEFAULT_RANKING_PRIORITY))
        priority_text = " → ".join(_RANKING_LABELS.get(str(metric), str(metric)) for metric in priority)
        comparison_section = f"""
        <h2>BOIN design comparison</h2>
        <p>Engine: {html.escape(comparison_note)}</p>
        <p>Ranking importance: {html.escape(priority_text)}</p>
        <table><thead><tr><th>Rank</th><th>Design</th><th>Cohort</th><th>Maximum N</th>
        <th>Closest-to-target selection</th><th>Selection above target DLT</th><th>No dose selected</th></tr></thead>
        <tbody>{comparison_rows}</tbody></table>
        """
    rendered = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(str(target.get('name', 'v3 analysis')))}</title>
    <style>body{{font-family:system-ui;max-width:900px;margin:40px auto;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style></head><body>
    <h1>{html.escape(str(target.get('name', 'v3 analysis')))}</h1>
    <p><b>Drug:</b> {html.escape(str(target.get('drug', '')))} · <b>Disease:</b> {html.escape(str(target.get('disease', '')))} ·
    <b>Target DLT:</b> {100 * float(target.get('target_dlt', .25)):.1f}%</p>
    <h2>Predicted curve</h2><p>Method: {html.escape(str(method.get('label') or active_method))}</p>
    <table><thead><tr><th>Dose ({html.escape(str(target.get('dose_unit', '')) )})</th><th>Predicted DLT</th><th>95% interval</th></tr></thead><tbody>{rows}</tbody></table>
    {simulation_section}{comparison_section}</body></html>"""
    return rendered.encode("utf-8")
