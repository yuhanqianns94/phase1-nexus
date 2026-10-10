from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from phase1_extraction import batch, pipeline
from phase1_extraction.batch import run_batch
from phase1_extraction.request_cache import artifact_lock

from test_resolver import base_extraction


def test_batch_isolates_failures_and_resumes_completed_articles(tmp_path: Path) -> None:
    xml = tmp_path / "source.xml"
    xml.write_text(
        "<article><front><article-meta><title-group><article-title>Test</article-title>"
        "</title-group></article-meta></front><body><p>Results.</p></body></article>",
        encoding="utf-8",
    )
    reported = tmp_path / "reported.json"
    reported.write_text(base_extraction(doses=(100,), explicit_no_dlt=True).model_dump_json(), encoding="utf-8")
    manifest = tmp_path / "batch.jsonl"
    manifest.write_text(
        json.dumps({"document_id": "DOC", "sources": [xml.name], "reported_input": reported.name})
        + "\n"
        + json.dumps({"document_id": "MISSING", "sources": ["missing.pdf"]})
        + "\n",
        encoding="utf-8",
    )
    state_path = run_batch(manifest_path=manifest, output_root=tmp_path / "out")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert [item["status"] for item in state["outcomes"]] == ["completed", "failed"]
    run_batch(manifest_path=manifest, output_root=tmp_path / "out")
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert resumed["outcomes"][0]["status"] == "skipped_completed"
    assert resumed["outcomes"][1]["status"] == "failed"


@pytest.fixture
def offline_batch(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "source.xml").write_text(
        "<article><body><p>Results.</p></body></article>", encoding="utf-8",
    )
    (tmp_path / "reported.json").write_text(
        base_extraction(doses=(100,), explicit_no_dlt=True).model_dump_json(), encoding="utf-8",
    )
    manifest = tmp_path / "batch.jsonl"
    manifest.write_text(json.dumps({
        "document_id": "DOC", "sources": ["source.xml"], "reported_input": "reported.json",
    }) + "\n", encoding="utf-8")
    return manifest, tmp_path / "out"


def _run_status(offline_batch: tuple[Path, Path], **kwargs: object) -> str:
    manifest, output = offline_batch
    state_path = run_batch(manifest_path=manifest, output_root=output, **kwargs)
    return json.loads(state_path.read_text(encoding="utf-8"))["outcomes"][0]["status"]


def test_batch_persists_all_skipped_state(offline_batch: tuple[Path, Path]) -> None:
    assert _run_status(offline_batch) == "completed"
    assert _run_status(offline_batch) == "skipped_completed"
    assert _run_status(offline_batch) == "skipped_completed"


@pytest.mark.parametrize("changed", [
    {"model": "different-model"}, {"reasoning_effort": "high"},
    {"max_output_tokens": 1_000}, {"max_packet_chars": 20_000},
])
def test_batch_reprocesses_changed_parameters(offline_batch: tuple[Path, Path], changed: dict[str, object]) -> None:
    assert _run_status(offline_batch) == "completed"
    assert _run_status(offline_batch, **changed) == "completed"
    assert _run_status(offline_batch, **changed) == "skipped_completed"


@pytest.mark.parametrize("part", ["instructions", "text"])
def test_batch_reprocesses_changed_prompt_or_schema(
    offline_batch: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, part: str,
) -> None:
    assert _run_status(offline_batch) == "completed"
    original = batch.build_request_body

    def changed_contract(**kwargs: object) -> dict[str, object]:
        body = original(**kwargs)
        if part == "instructions":
            body[part] += "\nUpdated extraction instructions."
        else:
            body[part]["format"]["schema"]["description"] = "Updated schema."
        return body

    monkeypatch.setattr(batch, "build_request_body", changed_contract)
    assert _run_status(offline_batch) == "completed"
    assert _run_status(offline_batch) == "skipped_completed"


def test_batch_reprocesses_changed_reported_input(offline_batch: tuple[Path, Path]) -> None:
    manifest, output = offline_batch
    assert _run_status(offline_batch) == "completed"
    before = (output / "DOC/reported_extraction.json").read_bytes()
    (manifest.parent / "reported.json").write_text(
        base_extraction(doses=(200,), explicit_no_dlt=True).model_dump_json(), encoding="utf-8",
    )
    assert _run_status(offline_batch) == "completed"
    assert (output / "DOC/reported_extraction.json").read_bytes() != before


@pytest.mark.parametrize("action", ["remove", "change"])
@pytest.mark.parametrize("artifact", ["reported_extraction.json", "curve_rows.json", "extraction_packets.json"])
def test_batch_repairs_missing_or_changed_artifacts(
    offline_batch: tuple[Path, Path], action: str, artifact: str,
) -> None:
    _, output = offline_batch
    assert _run_status(offline_batch) == "completed"
    path = output / "DOC" / artifact
    expected = path.read_bytes()
    if action == "remove":
        path.unlink()
    else:
        path.write_text("{}", encoding="utf-8")
    assert _run_status(offline_batch) == "completed"
    # Reported/resolved facts remain deterministic after local reconstruction.
    assert path.read_bytes() == expected


def test_batch_reuses_completed_article_after_manifest_changes(offline_batch: tuple[Path, Path]) -> None:
    manifest, output = offline_batch
    assert _run_status(offline_batch) == "completed"
    first_state = next(output.glob("batch_*.json"))
    manifest.write_text("# New batch ordering/comment\n" + manifest.read_text(encoding="utf-8"), encoding="utf-8")
    assert _run_status(offline_batch) == "skipped_completed"
    assert len(list(output.glob("batch_*.json"))) == 2
    assert first_state.is_file()


def test_batch_reuses_duplicate_sources_with_ingest_semantics(offline_batch: tuple[Path, Path]) -> None:
    manifest, _ = offline_batch
    entry = json.loads(manifest.read_text(encoding="utf-8"))
    entry["sources"].append("source.xml")
    manifest.write_text(json.dumps(entry), encoding="utf-8")
    assert _run_status(offline_batch) == "completed"
    assert _run_status(offline_batch) == "skipped_completed"


def test_batch_reprocesses_identical_content_from_new_path(offline_batch: tuple[Path, Path]) -> None:
    manifest, output = offline_batch
    assert _run_status(offline_batch) == "completed"
    original = manifest.parent / "source.xml"
    supplement = manifest.parent / "supplement.xml"
    supplement.write_bytes(original.read_bytes())
    entry = json.loads(manifest.read_text(encoding="utf-8"))
    entry["sources"] = [supplement.name]
    manifest.write_text(json.dumps(entry), encoding="utf-8")
    assert _run_status(offline_batch) == "completed"
    materials = json.loads((output / "DOC/material_manifest.json").read_text(encoding="utf-8"))["materials"]
    assert materials[0]["role"] == "supplement"
    assert materials[0]["path"] == str(supplement)


def test_batch_missing_source_fails_instead_of_skipping(offline_batch: tuple[Path, Path]) -> None:
    manifest, _ = offline_batch
    assert _run_status(offline_batch) == "completed"
    entry = json.loads(manifest.read_text(encoding="utf-8"))
    entry["sources"].append("missing.pdf")
    manifest.write_text(json.dumps(entry), encoding="utf-8")
    assert _run_status(offline_batch) == "failed"


def test_batch_reprocesses_changed_source_content(offline_batch: tuple[Path, Path]) -> None:
    manifest, _ = offline_batch
    assert _run_status(offline_batch) == "completed"
    (manifest.parent / "source.xml").write_text(
        "<article><body><p>Updated results.</p></body></article>", encoding="utf-8",
    )
    assert _run_status(offline_batch) == "completed"


@pytest.mark.parametrize("change", ["legacy", "material_hashes"])
def test_batch_reprocesses_unverifiable_manifest(offline_batch: tuple[Path, Path], change: str) -> None:
    _, output = offline_batch
    assert _run_status(offline_batch) == "completed"
    path = output / "DOC/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "legacy":
        manifest.pop("batch_resume")
    else:
        manifest["material_hashes"] = []
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _run_status(offline_batch) == "completed"


def test_batch_never_resumes_dry_run_as_completed(offline_batch: tuple[Path, Path]) -> None:
    manifest, _ = offline_batch
    entry = json.loads(manifest.read_text(encoding="utf-8"))
    entry.pop("reported_input")
    manifest.write_text(json.dumps(entry), encoding="utf-8")
    assert _run_status(offline_batch) == "dry_run"
    assert _run_status(offline_batch) == "dry_run"


def test_batch_ignores_corrupt_batch_state_but_checks_article_state(offline_batch: tuple[Path, Path]) -> None:
    _, output = offline_batch
    assert _run_status(offline_batch) == "completed"
    next(output.glob("batch_*.json")).write_text("{invalid", encoding="utf-8")
    assert _run_status(offline_batch) == "skipped_completed"
    (output / "DOC/manifest.json").write_text("{invalid", encoding="utf-8")
    assert _run_status(offline_batch) == "completed"


def test_batch_no_resume_reprocesses_and_forwards_option(
    offline_batch: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run_status(offline_batch) == "completed"
    original = batch.run_pipeline
    forwarded: list[bool] = []

    def spy_pipeline(**kwargs: object):
        forwarded.append(kwargs["resume"])
        return original(**kwargs)

    monkeypatch.setattr(batch, "run_pipeline", spy_pipeline)
    assert _run_status(offline_batch, resume=False) == "completed"
    assert forwarded == [False]


def test_empty_batch_writes_state(tmp_path: Path) -> None:
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("# No articles yet\n", encoding="utf-8")
    state = run_batch(manifest_path=manifest, output_root=tmp_path / "out")
    assert json.loads(state.read_text(encoding="utf-8"))["outcomes"] == []


def test_batch_does_not_check_or_skip_article_while_another_pipeline_holds_lock(
    offline_batch: tuple[Path, Path],
) -> None:
    _, output = offline_batch
    assert _run_status(offline_batch) == "completed"
    article_manifest = output / "DOC/manifest.json"
    completed_manifest = article_manifest.read_bytes()
    with artifact_lock(output / "DOC/.pipeline.lock"):
        assert _run_status(offline_batch) == "failed"
        assert article_manifest.read_bytes() == completed_manifest
    assert _run_status(offline_batch) == "skipped_completed"


@pytest.mark.parametrize("boundary", ["artifact_hashes", "manifest_stamp"])
def test_batch_holds_pipeline_lock_until_resume_stamp_is_written(
    offline_batch: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    manifest, output = offline_batch
    competing_report = manifest.parent / "competing_report.json"
    competing_report.write_text(
        base_extraction(doses=(900,), explicit_no_dlt=True).model_dump_json(), encoding="utf-8",
    )
    attempts: list[str] = []

    def competing_pipeline() -> None:
        attempts.append(boundary)
        with pytest.raises(RuntimeError, match="Artifacts are in use"):
            pipeline.run_pipeline(
                document_id="DOC", source_paths=[manifest.parent / "source.xml"],
                output_root=output, reported_input=competing_report,
            )

    if boundary == "artifact_hashes":
        original_hashes = batch._artifact_hashes

        def checked_hashes(paths):
            competing_pipeline()
            return original_hashes(paths)

        monkeypatch.setattr(batch, "_artifact_hashes", checked_hashes)
    else:
        original_write = batch.write_json

        def checked_write(path, value):
            if path == output / "DOC/manifest.json" and "batch_resume" in value:
                competing_pipeline()
            original_write(path, value)

        monkeypatch.setattr(batch, "write_json", checked_write)

    assert _run_status(offline_batch) == "completed"
    assert attempts == [boundary]
    assert _run_status(offline_batch) == "skipped_completed"
    assert attempts == [boundary]


@pytest.mark.parametrize("change", ["modified", "deleted", "appeared"])
@pytest.mark.parametrize("relative_href", [True, False])
def test_batch_rechecks_external_jats_graphics_before_skipping(
    offline_batch: tuple[Path, Path], change: str, relative_href: bool,
) -> None:
    manifest, output = offline_batch
    asset = manifest.parent / "external_table.png"
    if change != "appeared":
        Image.new("RGB", (8, 8), "white").save(asset)
    href = asset.name if relative_href else str(asset)
    (manifest.parent / "source.xml").write_text(
        '<article xmlns:xlink="http://www.w3.org/1999/xlink"><body>'
        '<table-wrap id="t1"><label>Table 1</label><caption><p>Dose cohorts and DLTs</p></caption>'
        f'<graphic xlink:href="{href}"/></table-wrap></body></article>', encoding="utf-8",
    )
    assert _run_status(offline_batch) == "completed"
    assert _run_status(offline_batch) == "skipped_completed"
    packets_path = output / "DOC/extraction_packets.json"
    original_packets = packets_path.read_bytes()
    if change == "deleted":
        asset.unlink()
    else:
        Image.new("RGB", (8, 8), "black").save(asset)
    # XML and declared sources are unchanged; only a referenced external image
    # changed. Repacketization must record its current hash/availability.
    assert _run_status(offline_batch) == "completed"
    assert packets_path.read_bytes() != original_packets
    assert _run_status(offline_batch) == "skipped_completed"
