"""Failure-isolated JSONL batch runner with a shared budget guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import (
    ArticlePaths, read_json, register_materials, sha256_file, sha256_json, utc_now, write_json,
)
from .llm import (
    BudgetGuard, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, build_request_body,
)
from .models import DocumentIR
from .packets import DEFAULT_MAX_PACKET_CHARS, packet_image_inventory
# Batch owns the article lock through verification and signature persistence;
# call the unlocked implementation only while that outer lock is held.
from .pipeline import _run_pipeline as run_pipeline
from .request_cache import artifact_lock


def _processing_signature(*, model: str, reasoning_effort: str, max_output_tokens: int, max_packet_chars: int) -> str:
    """Bind reuse to the actual prompt/schema and the processing implementation."""

    package = Path(__file__).parent
    processing_modules = (
        "artifacts", "document_ir", "packets", "llm", "pipeline", "models",
        "quality", "resolver", "drug_ordering", "exposures", "field_normalization",
        "controlled_vocabulary", "dose_units", "request_cache",
    )
    return sha256_json({
        "request_contract": build_request_body(
            packet={}, prior=None, image_paths=[], model=model,
            reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens,
        ),
        "max_packet_chars": max_packet_chars,
        "processing_sources": {name: sha256_file(package / f"{name}.py") for name in processing_modules},
    })


def _entry_signature(entry: dict[str, Any], output_root: Path, processing_signature: str, execute_model: bool) -> str:
    # Use ingest's exact de-duplication and role inference. A missing source must
    # fail this entry, never disappear from a comparison against an older run.
    materials = register_materials(document_id=entry["document_id"], source_paths=entry["sources"])
    reported_input = entry["reported_input"]
    return sha256_json({
        "version": 1,
        "document_id": entry["document_id"],
        "output_root": str(output_root.resolve()),
        "processing_signature": processing_signature,
        "materials": [material.model_dump(mode="json") for material in materials],
        "mode": "reported_input" if reported_input is not None else "model" if execute_model else "dry_run",
        "reported_input": None if reported_input is None else {
            "path": str(Path(reported_input).resolve()),
            "sha256": sha256_file(reported_input),
        },
    })


def _required_artifacts(paths: ArticlePaths) -> list[Path]:
    return [
        paths.material_manifest, paths.document_ir, paths.extraction_packets,
        paths.request_summary, paths.reported_extraction, paths.extraction_audit,
        paths.resolved_extraction, paths.curve_rows, paths.rule_applications, paths.issues,
    ]


def _artifact_hashes(paths: ArticlePaths) -> dict[str, str]:
    artifacts = _required_artifacts(paths)
    for directory in (paths.page_images, paths.table_images, paths.raw_requests, paths.raw_responses, paths.parsed_packets):
        artifacts.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return {str(path.relative_to(paths.root)): sha256_file(path) for path in artifacts}


def _completed_resume_manifest(paths: ArticlePaths, signature: str) -> dict[str, Any] | None:
    try:
        manifest = read_json(paths.manifest)
        resume = manifest.get("batch_resume", {})
        if manifest.get("status") != "completed" or resume.get("signature") != signature:
            return None
        manifest_content = {key: value for key, value in manifest.items() if key != "batch_resume"}
        if resume.get("manifest_sha256") != sha256_json(manifest_content):
            return None
        hashes = resume.get("artifact_sha256", {})
        required_names = {str(path.relative_to(paths.root)) for path in _required_artifacts(paths)}
        if not isinstance(hashes, dict) or not required_names.issubset(hashes):
            return None
        for name, expected in hashes.items():
            artifact = paths.root / name
            if not artifact.resolve().is_relative_to(paths.root.resolve()) or sha256_file(artifact) != expected:
                return None
        # JATS graphics can live beside the XML, outside both manifest.sources
        # and the article output directory. The frozen inventory also records
        # missing images, so newly available visuals invalidate a completed run.
        document = DocumentIR.model_validate(read_json(paths.document_ir))
        material_paths = {material.material_id: Path(material.path) for material in document.materials}
        for table in document.tables:
            for fragment in table.fragments:
                if fragment.source == "jats" and fragment.image_path and not Path(fragment.image_path).is_absolute():
                    # The JATS parser retains relative hrefs when an image was
                    # absent. Resolve one that has appeared since that parse.
                    material_path = material_paths.get(fragment.material_id)
                    if material_path is not None:
                        candidate = (material_path.parent / fragment.image_path).resolve()
                        if candidate.is_file():
                            fragment.image_path = str(candidate)
        for packet in read_json(paths.extraction_packets)["packets"]:
            if packet_image_inventory(packet, document) != packet.get("image_inventory"):
                return None
        return manifest
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        # Corrupt/incomplete local state must not silently suppress work.
        return None


def load_batch_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()
    base = Path(path).resolve().parent
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = json.loads(line)
        document_id = str(item.get("document_id") or "").strip()
        sources = item.get("sources")
        if not document_id or not isinstance(sources, list) or not sources:
            raise ValueError(f"Line {line_number} requires document_id and a non-empty sources list")
        if document_id in seen_document_ids:
            raise ValueError(f"Duplicate document_id {document_id!r} on line {line_number}")
        seen_document_ids.add(document_id)
        resolved_sources = []
        for source in sources:
            candidate = Path(str(source)).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            resolved_sources.append(candidate.resolve())
        # New acquisition manifests bind eligibility to the exact source bytes.
        # Legacy user-authored manifests remain compatible; extraction fields do not change.
        if item.get("screening_record"):
            from phase1_acquisition.screening import validate_screening_gate
            screening_path = Path(item["screening_record"])
            if not screening_path.is_absolute():
                screening_path = base / screening_path
            validate_screening_gate(screening_path, resolved_sources)
        reported_input = item.get("reported_input")
        if reported_input:
            candidate = Path(str(reported_input)).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            reported_input = candidate.resolve()
        entries.append(
            {
                "document_id": document_id,
                "sources": resolved_sources,
                "reported_input": reported_input,
            }
        )
    return entries


def run_batch(
    *,
    manifest_path: Path,
    output_root: Path,
    execute_model: bool = False,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_packet_chars: int = DEFAULT_MAX_PACKET_CHARS,
    budget_guard: BudgetGuard | None = None,
    resume: bool = True,
) -> Path:
    entries = load_batch_manifest(manifest_path)
    output_root = Path(output_root)
    manifest_hash = sha256_file(manifest_path)
    state_path = output_root / f"batch_{manifest_hash[:16]}.json"
    processing_signature = _processing_signature(
        model=model, reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens, max_packet_chars=max_packet_chars,
    )
    outcomes: list[dict[str, Any]] = []

    def save_state() -> None:
        write_json(
            state_path,
            {
                "format": "phase1_extraction_batch_v3",
                "manifest_path": str(Path(manifest_path).resolve()),
                "manifest_sha256": manifest_hash,
                "updated_at_utc": utc_now(),
                "model": model,
                "execute_model": execute_model,
                "estimated_spent_usd": budget_guard.spent_usd if budget_guard else None,
                "outcomes": outcomes,
            },
        )

    for entry in entries:
        document_id = entry["document_id"]
        article_paths = ArticlePaths(output_root, document_id)
        started = utc_now()
        try:
            with artifact_lock(article_paths.root / ".pipeline.lock"):
                signature = _entry_signature(entry, output_root, processing_signature, execute_model)
                completed = _completed_resume_manifest(article_paths, signature) if resume else None
                if completed is not None:
                    outcomes.append({
                        "document_id": document_id,
                        "status": "skipped_completed",
                        "started_at_utc": started,
                        "finished_at_utc": utc_now(),
                        "article_root": str(article_paths.root.resolve()),
                    })
                    save_state()
                    continue
                paths = run_pipeline(
                    document_id=document_id,
                    source_paths=entry["sources"],
                    output_root=output_root,
                    reported_input=entry["reported_input"],
                    execute_model=execute_model,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    max_packet_chars=max_packet_chars,
                    budget_guard=budget_guard,
                    resume=resume,
                )
                article_manifest = read_json(paths.manifest)
                article_status = article_manifest.get("status")
                status = "completed" if article_status == "completed" else "dry_run"
                if status == "completed":
                    if _entry_signature(entry, output_root, processing_signature, execute_model) != signature:
                        raise RuntimeError("Source inputs changed during processing; rerun this article before resuming")
                    article_manifest["batch_resume"] = {
                        "signature": signature,
                        "manifest_sha256": sha256_json({
                            key: value for key, value in article_manifest.items() if key != "batch_resume"
                        }),
                        "artifact_sha256": _artifact_hashes(paths),
                    }
                    write_json(paths.manifest, article_manifest)
                outcome = {
                    "document_id": document_id,
                    "status": status,
                    "started_at_utc": started,
                    "finished_at_utc": utc_now(),
                    "article_root": str(paths.root.resolve()),
                }
        except Exception as error:
            outcome = {
                "document_id": document_id,
                "status": "failed",
                "started_at_utc": started,
                "finished_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        outcomes.append(outcome)
        save_state()
    if not entries:
        save_state()
    return state_path
