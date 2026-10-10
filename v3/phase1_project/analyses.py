"""Explicitly saved target analyses. These are not historical article caches."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase1_extraction.artifacts import read_json, write_json


ANALYSIS_FORMAT = "phase1_v3_saved_target_analysis_v1"


def save_analysis(root: Path, payload: dict[str, Any]) -> Path:
    prediction = payload.get("prediction") or {}
    run_id = str(prediction.get("run_id") or "").strip()
    if not run_id.startswith("prediction_"):
        raise ValueError("A fitted target prediction is required before saving an analysis.")
    for key in ("simulation", "design_comparison"):
        attached = payload.get(key)
        if attached and attached.get("prediction_run_id") != run_id:
            raise ValueError(f"{key} belongs to a different prediction; rerun it before saving.")
        if attached and payload.get("active_method") and attached.get("method_key") != payload["active_method"]:
            raise ValueError(f"{key} belongs to a different prediction method.")
    target = prediction.get("target") or {}
    identity = {
        "prediction_run_id": run_id,
        "simulation_run_id": (payload.get("simulation") or {}).get("run_id"),
        "design_comparison_run_id": (payload.get("design_comparison") or {}).get("run_id"),
        "active_method": payload.get("active_method"),
    }
    analysis_id = "analysis_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    record = {
        "format": ANALYSIS_FORMAT,
        "analysis_id": analysis_id,
        "prediction_run_id": run_id,
        "name": str(target.get("name") or run_id),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "draft": payload.get("draft"),
        "evidence_preview": payload.get("evidence_preview"),
        "prediction": prediction,
        "simulation": payload.get("simulation"),
        "design_comparison": payload.get("design_comparison"),
        "active_method": payload.get("active_method"),
    }
    path = Path(root) / "analyses" / analysis_id / "analysis.json"
    write_json(path, record)
    return path


def list_analyses(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((Path(root) / "analyses").glob("*/analysis.json")):
        try:
            record = read_json(path)
        except (OSError, ValueError):
            continue
        if record.get("format") != ANALYSIS_FORMAT:
            continue
        records.append({
            "analysis_id": record.get("analysis_id"), "name": record.get("name"),
            "saved_at_utc": record.get("saved_at_utc"), "path": str(path.resolve()),
            "has_simulation": bool(record.get("simulation")),
            "has_design_comparison": bool(record.get("design_comparison")),
        })
    return sorted(records, key=lambda item: str(item.get("saved_at_utc") or ""), reverse=True)


def load_analysis(path: Path) -> dict[str, Any]:
    record = read_json(Path(path))
    if record.get("format") != ANALYSIS_FORMAT:
        raise ValueError("The selected file is not a saved v3 target analysis.")
    return record
