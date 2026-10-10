from __future__ import annotations

import json
from pathlib import Path

from phase1_extraction.models import ReportedExtraction
from phase1_extraction.pipeline import run_pipeline

from test_resolver import base_extraction


def test_offline_reported_input_runs_without_model(tmp_path: Path) -> None:
    xml = tmp_path / "source.xml"
    xml.write_text(
        "<article><front><article-meta><title-group><article-title>Test</article-title>"
        "</title-group></article-meta></front><body><sec><title>Results</title>"
        "<p>One cohort was treated.</p></sec></body></article>",
        encoding="utf-8",
    )
    extraction = base_extraction(doses=(100,), explicit_no_dlt=True)
    reported = tmp_path / "reported.json"
    reported.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
    paths = run_pipeline(
        document_id="DOC",
        source_paths=[xml],
        output_root=tmp_path / "out",
        reported_input=reported,
    )
    assert paths.document_ir.is_file()
    assert paths.reported_extraction.is_file()
    assert paths.resolved_extraction.is_file()
    curve = json.loads(paths.curve_rows.read_text(encoding="utf-8"))
    assert set(curve) == {"policy_version", "project_default", "reported_only", "all_rows", "model_outcome_rows"}
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"


def test_response_schema_has_no_dose_selection_endpoint_keys() -> None:
    schema = ReportedExtraction.model_json_schema()
    forbidden = {"mtd", "rp2d", "mad", "recommended_dose", "selected_dose"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {item for child in value.values() for item in keys(child)}
        if isinstance(value, list):
            return {item for child in value for item in keys(child)}
        return set()

    assert not (keys(schema) & forbidden)

