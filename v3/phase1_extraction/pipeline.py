"""Stage-addressable v3 extraction orchestration."""

from __future__ import annotations

from functools import wraps
import os
from pathlib import Path
from typing import Any

from .artifacts import (
    ArticlePaths,
    new_run_manifest,
    read_json,
    record_stage,
    register_materials,
    write_json,
)
from .document_ir import build_document_ir
from .llm import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    BudgetGuard,
    extract_packets,
)
from .models import DocumentIR, Material, ReportedExtraction, ResolvedExtraction
from .packets import DEFAULT_MAX_PACKET_CHARS, build_extraction_packets, packet_image_inventory
from .quality import apply_audit_status, audit_reported_extraction
from .resolver import resolve_extraction
from .request_cache import artifact_lock


def ingest(
    *,
    document_id: str,
    source_paths: list[Path],
    output_root: Path,
    material_roles: dict[str, str] | None = None,
) -> ArticlePaths:
    paths = ArticlePaths(output_root=Path(output_root), document_id=document_id)
    paths.ensure()
    materials = register_materials(document_id=document_id, source_paths=source_paths, roles=material_roles)
    manifest_payload = {
        "format": "phase1_material_manifest_v3",
        "document_id": document_id,
        "materials": [item.model_dump(mode="json") for item in materials],
    }
    write_json(paths.material_manifest, manifest_payload)
    run_manifest = new_run_manifest(document_id, materials)
    record_stage(run_manifest, "ingest", status="completed", artifact=paths.material_manifest)
    write_json(paths.manifest, run_manifest)
    return paths


def _load_materials(paths: ArticlePaths) -> list[Material]:
    payload = read_json(paths.material_manifest)
    return [Material.model_validate(item) for item in payload["materials"]]


def parse_sources(
    paths: ArticlePaths, *, render_pages: bool = True, max_asset_bytes: int | None = None,
) -> DocumentIR:
    materials = _load_materials(paths)
    document = build_document_ir(
        document_id=paths.document_id,
        materials=materials,
        page_image_dir=paths.page_images,
        table_image_dir=paths.table_images,
        render_pages=render_pages,
    )
    if max_asset_bytes is not None:
        asset_bytes = sum(path.stat().st_size for folder in (paths.page_images, paths.table_images)
                          for path in folder.rglob("*") if path.is_file())
        if asset_bytes > max_asset_bytes:
            raise ValueError("Parsed assets exceed the upload processing limit.")
    write_json(paths.document_ir, document)
    manifest = read_json(paths.manifest)
    record_stage(
        manifest,
        "parse",
        status="completed" if any(item.status != "failed" for item in document.materials) else "failed",
        artifact=paths.document_ir,
        details={
            "pages": len(document.pages),
            "blocks": len(document.blocks),
            "tables": len(document.tables),
            "issues": len(document.issues),
        },
    )
    write_json(paths.manifest, manifest)
    return document


def prepare_packets(
    paths: ArticlePaths,
    *,
    max_packet_chars: int = DEFAULT_MAX_PACKET_CHARS,
) -> tuple[DocumentIR, list[dict[str, Any]]]:
    document = DocumentIR.model_validate(read_json(paths.document_ir))
    packets = build_extraction_packets(document, max_packet_chars=max_packet_chars)
    included_ids: set[str] = set()
    for packet in packets:
        for unit in packet["source_units"]:
            if unit["unit_kind"] == "text":
                included_ids.update(item["source_id"] for item in unit["items"])
                included_ids.update(item["page_id"] for item in unit["items"] if item.get("page_id"))
            elif unit["unit_kind"] == "table":
                included_ids.add(unit["table"]["source_id"])
                included_ids.update(
                    item["source_id"] for item in unit["table"].get("fragments", [])
                    if item.get("source_id")
                )
                included_ids.update(
                    item["page_id"] for item in unit["table"].get("fragments", [])
                    if item.get("page_id")
                )
            elif unit["unit_kind"] == "page_visual":
                included_ids.add(unit["page"]["source_id"])
        packet["image_inventory"] = packet_image_inventory(packet, document)
    for coverage in document.coverage:
        if coverage.source_id in included_ids:
            coverage.included_in_extraction = True
    relevant_children = [
        item
        for item in document.coverage
        if item.source_kind != "material" and item.potentially_relevant
    ]
    complete_material_ids = {
        item.material_id for item in document.materials if item.status in {"parsed", "partial"}
    }
    for coverage in document.coverage:
        if coverage.source_kind == "material" and coverage.source_id in complete_material_ids:
            coverage.included_in_extraction = True
    material_coverage_complete = all(
        item.included_in_extraction and item.parse_status not in {"failed", "unread"}
        for item in relevant_children
    ) and all(item.status != "failed" for item in document.materials)
    write_json(paths.document_ir, document)
    write_json(
        paths.extraction_packets,
        {
            "packets": packets,
            "material_coverage_complete": material_coverage_complete,
            "not_covered_source_ids": [
                item.source_id
                for item in relevant_children
                if not item.included_in_extraction or item.parse_status in {"failed", "unread"}
            ],
        },
    )
    manifest = read_json(paths.manifest)
    record_stage(
        manifest,
        "packetize",
        status="completed",
        artifact=paths.extraction_packets,
        details={
            "packet_count": len(packets),
            "material_coverage_complete": material_coverage_complete,
        },
    )
    write_json(paths.manifest, manifest)
    return document, packets


def _extract_reported(
    paths: ArticlePaths,
    *,
    reported_input: Path | None = None,
    execute_model: bool = False,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_packet_chars: int = DEFAULT_MAX_PACKET_CHARS,
    budget_guard: BudgetGuard | None = None,
    resume: bool = True,
    client: Any = None,
    sanitize_errors: bool = False,
) -> ReportedExtraction | None:
    public = os.getenv("PHASE1_DEPLOYMENT", "").strip().lower() == "public"
    sanitize_errors = sanitize_errors or public
    if execute_model and public and client is None:
        raise ValueError("Public extraction requires an explicit user API client")
    document, packets = prepare_packets(paths, max_packet_chars=max_packet_chars)
    if reported_input is not None:
        extraction = ReportedExtraction.model_validate(read_json(reported_input))
        if extraction.document_id != paths.document_id:
            raise ValueError("reported_input document_id does not match the selected article")
        summaries = [{"mode": "offline_reported_input", "path": str(Path(reported_input).resolve())}]
    elif not execute_model:
        write_json(
            paths.request_summary,
            {
                "status": "dry_run",
                "document_id": paths.document_id,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "packet_count": len(packets),
                "max_output_tokens_per_packet": max_output_tokens,
                "message": "No model request was sent. Re-run with --execute and an explicit budget/pricing configuration.",
            },
        )
        manifest = read_json(paths.manifest)
        record_stage(manifest, "extract", status="skipped", artifact=paths.request_summary)
        write_json(paths.manifest, manifest)
        return None
    else:
        try:
            extraction, summaries = extract_packets(
                packets=packets,
                document=document,
                raw_response_dir=paths.raw_responses,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                budget_guard=budget_guard,
                raw_request_dir=paths.raw_requests,
                parsed_packet_dir=paths.parsed_packets,
                resume=resume,
                client=client,
                sanitize_errors=sanitize_errors,
            )
        except Exception as error:
            manifest = read_json(paths.manifest)
            record_stage(
                manifest,
                "extract",
                status="failed",
                details={"error_type": type(error).__name__, "message": (
                    "Extraction request failed; provider details are withheld." if sanitize_errors else str(error)
                )},
            )
            write_json(paths.manifest, manifest)
            raise
    packet_manifest = read_json(paths.extraction_packets)
    extraction.material_coverage_complete = bool(packet_manifest.get("material_coverage_complete"))
    if not extraction.material_coverage_complete:
        note = "Programmatic material coverage is incomplete; R2 zero imputation is disabled."
        if note not in extraction.coverage_notes:
            extraction.coverage_notes.append(note)
    from .drug_ordering import apply_drug_ordering
    extraction = apply_drug_ordering(extraction)
    audit = audit_reported_extraction(extraction, document=document)
    apply_audit_status(extraction, audit)
    write_json(paths.extraction_audit, audit)
    write_json(paths.reported_extraction, extraction)
    write_json(paths.request_summary, {"status": "completed", "requests": summaries})
    manifest = read_json(paths.manifest)
    record_stage(
        manifest,
        "extract",
        status="completed",
        artifact=paths.reported_extraction,
        details={"request_count": len(summaries)},
    )
    write_json(paths.manifest, manifest)
    return extraction


def resolve_reported(paths: ArticlePaths) -> ResolvedExtraction:
    extraction = ReportedExtraction.model_validate(read_json(paths.reported_extraction))
    try:
        resolved = resolve_extraction(extraction)
    except Exception as error:
        manifest = read_json(paths.manifest)
        record_stage(
            manifest,
            "resolve",
            status="failed",
            details={"error_type": type(error).__name__, "message": str(error)},
        )
        write_json(paths.manifest, manifest)
        raise
    document = DocumentIR.model_validate(read_json(paths.document_ir)) if paths.document_ir.is_file() else None
    audit = audit_reported_extraction(extraction, document=document, resolved=resolved)
    apply_audit_status(extraction, audit)
    apply_audit_status(resolved.reported_extraction, audit)
    write_json(paths.extraction_audit, audit)
    write_json(paths.reported_extraction, extraction)
    write_json(paths.resolved_extraction, resolved)
    write_json(
        paths.curve_rows,
        {
            "policy_version": resolved.policy_version,
            "project_default": [row.model_dump(mode="json") for row in resolved.curve_rows if row.included_in_default],
            "reported_only": [row.model_dump(mode="json") for row in resolved.curve_rows if row.included_in_reported_only],
            "all_rows": [row.model_dump(mode="json") for row in resolved.curve_rows],
            "model_outcome_rows": [row.model_dump(mode="json") for row in resolved.curve_rows if row.outcome_usable],
        },
    )
    write_json(paths.rule_applications, [item.model_dump(mode="json") for item in resolved.rule_applications])
    write_json(paths.issues, [item.model_dump(mode="json") for item in resolved.issues])
    manifest = read_json(paths.manifest)
    record_stage(
        manifest,
        "resolve",
        status="completed",
        artifact=paths.resolved_extraction,
        details={
            "curve_rows": len(resolved.curve_rows),
            "included_default": sum(row.included_in_default for row in resolved.curve_rows),
            "included_reported_only": sum(row.included_in_reported_only for row in resolved.curve_rows),
            "issues": len(resolved.issues),
        },
    )
    manifest["status"] = "completed"
    write_json(paths.manifest, manifest)
    return resolved


def _run_pipeline(
    *,
    document_id: str,
    source_paths: list[Path],
    output_root: Path,
    reported_input: Path | None = None,
    execute_model: bool = False,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_packet_chars: int = DEFAULT_MAX_PACKET_CHARS,
    budget_guard: BudgetGuard | None = None,
    resume: bool = True,
    client: Any = None,
    sanitize_errors: bool = False,
    max_parsed_asset_bytes: int | None = None,
    material_roles: dict[str, str] | None = None,
) -> ArticlePaths:
    paths = ingest(document_id=document_id, source_paths=source_paths, output_root=output_root,
                   material_roles=material_roles)
    if max_parsed_asset_bytes is None:
        parse_sources(paths)
    else:
        parse_sources(paths, max_asset_bytes=max_parsed_asset_bytes)
    extraction = _extract_reported(
        paths,
        reported_input=reported_input,
        execute_model=execute_model,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        max_packet_chars=max_packet_chars,
        budget_guard=budget_guard,
        resume=resume,
        client=client,
        sanitize_errors=sanitize_errors,
    )
    if extraction is not None:
        resolve_reported(paths)
    return paths


@wraps(_extract_reported)
def extract_reported(paths: ArticlePaths, **kwargs: Any) -> ReportedExtraction | None:
    with artifact_lock(paths.root / ".pipeline.lock"):
        return _extract_reported(paths, **kwargs)


@wraps(_run_pipeline)
def run_pipeline(**kwargs: Any) -> ArticlePaths:
    paths = ArticlePaths(Path(kwargs["output_root"]), kwargs["document_id"])
    with artifact_lock(paths.root / ".pipeline.lock"):
        return _run_pipeline(**kwargs)
