"""Read-only discovery plus optional indexing of already-downloaded materials."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from phase1_extraction.artifacts import read_json, write_json
from phase1_prediction.evidence import discover_article_paths

from .runtime import WORKSPACE_ROOT


@dataclass(frozen=True)
class CatalogArticle:
    document_id: str
    extraction_root: str | None
    reported_extraction: str | None
    curve_rows: str | None
    xml_path: str | None
    pdf_path: str | None
    material_paths: tuple[str, ...]
    source_url: str | None
    extraction_status: str


def default_extraction_roots() -> list[Path]:
    output = WORKSPACE_ROOT / "output"
    values = [output / "phase1_extractions_v3"]
    values.extend(sorted(path / "results" for path in output.glob("v3_batch*") if (path / "results").is_dir()))
    return [path for path in values if path.is_dir()]


def default_xml_directories() -> list[Path]:
    campaigns = sorted((WORKSPACE_ROOT / "v3/output").glob("pubmed*/materials"))
    legacy = [WORKSPACE_ROOT / relative for relative in (
        "v3/output/pmc_xml_no_refs", "v3/output/pmc_xml", "v2/output/pmc_xml_no_refs",
        "v1/output/pmc_xml_no_refs", "output/pmc_xml_no_refs")]
    return [path for path in [*campaigns, *legacy] if path.is_dir()]


def default_pdf_directories() -> list[Path]:
    return [path for path in (WORKSPACE_ROOT / "v3/output/pmc_pdf_assets", WORKSPACE_ROOT / "v2/output/pmc_pdf_assets", WORKSPACE_ROOT / "v1/output/pmc_pdf_assets", WORKSPACE_ROOT / "output/pmc_pdf_assets") if path.is_dir()]


def _pmcid(path: Path) -> str | None:
    match = re.search(r"PMC\d+", str(path), re.I)
    return match.group(0).upper() if match else None


def _file_map(directories: Iterable[Path], suffix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for directory in directories:
        for path in Path(directory).expanduser().resolve().rglob(f"*{suffix}"):
            if path.is_file() and (document_id := _pmcid(path)):
                result.setdefault(document_id, str(path))
    return result


def build_catalog(
    *, extraction_roots: Iterable[Path] | None = None,
    xml_directories: Iterable[Path] | None = None,
    pdf_directories: Iterable[Path] | None = None,
) -> dict:
    roots = list(default_extraction_roots() if extraction_roots is None else extraction_roots)
    selected, duplicates = discover_article_paths(roots)
    xml = _file_map(default_xml_directories() if xml_directories is None else xml_directories, ".xml")
    pdf = _file_map(default_pdf_directories() if pdf_directories is None else pdf_directories, ".pdf")
    document_ids = sorted(set(selected) | set(xml) | set(pdf))
    articles: list[CatalogArticle] = []
    for document_id in document_ids:
        paths = selected.get(document_id)
        material_paths: list[str] = []
        if paths and paths.material_manifest.is_file():
            try:
                material_paths = [str(item["path"]) for item in read_json(paths.material_manifest).get("materials", []) if item.get("path")]
            except (OSError, ValueError, TypeError):
                material_paths = []
        for candidate in (xml.get(document_id), pdf.get(document_id)):
            if candidate and candidate not in material_paths:
                material_paths.append(candidate)
        articles.append(CatalogArticle(
            document_id=document_id,
            extraction_root=str(paths.root) if paths else None,
            reported_extraction=str(paths.reported_extraction) if paths else None,
            curve_rows=str(paths.curve_rows) if paths else None,
            xml_path=xml.get(document_id), pdf_path=pdf.get(document_id),
            material_paths=tuple(material_paths),
            source_url=(f"https://pmc.ncbi.nlm.nih.gov/articles/{document_id}/" if re.fullmatch(r"PMC\d+", document_id)
                        else f"https://pubmed.ncbi.nlm.nih.gov/{document_id[4:]}/" if re.fullmatch(r"PMID\d+", document_id) else None),
            extraction_status="ready" if paths else "awaiting_v3_extraction",
        ))
    identity = [asdict(item) for item in articles]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "format": "phase1_v3_existing_material_catalog_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "crawler_was_run": False,
        "model_extraction_was_run": False,
        "extraction_roots": [str(Path(path).expanduser().resolve()) for path in roots],
        "duplicate_extraction_versions": duplicates,
        "article_count": len(articles),
        "ready_extraction_count": sum(item.extraction_status == "ready" for item in articles),
        "awaiting_extraction_count": sum(item.extraction_status != "ready" for item in articles),
        "fingerprint": fingerprint,
        "articles": identity,
    }


def write_catalog(path: Path, **kwargs) -> dict:
    payload = build_catalog(**kwargs)
    write_json(path, payload)
    return payload


def discover_batch_jobs(search_root: Path | None = None) -> list[dict]:
    """Read local Batch state and collection summaries without contacting OpenAI."""
    root = Path(search_root or (WORKSPACE_ROOT / "output"))
    jobs: list[dict] = []
    for state_path in sorted(root.glob("v3_batch*/batch_workspace/batch_state.json")):
        try:
            state = read_json(state_path)
        except (OSError, ValueError):
            continue
        workspace = state_path.parent
        summary_path = workspace / "collection_summary.json"
        try:
            summary = read_json(summary_path) if summary_path.is_file() else {}
        except (OSError, ValueError):
            summary = {}
        submissions = state.get("submissions") or []
        statuses = [str(item.get("status") or "unknown") for item in submissions]
        status = "completed" if statuses and all(value == "completed" for value in statuses) else (statuses[-1] if statuses else "not submitted")
        counts = {"completed": 0, "failed": 0, "total": 0}
        for item in submissions:
            for key in counts:
                counts[key] += int((item.get("request_counts") or {}).get(key) or 0)
        jobs.append({
            "batch": state_path.parents[1].name,
            "status": status,
            "request_count": int((state.get("validation") or {}).get("request_count") or counts["total"]),
            "completed": int(summary.get("completed_count", counts["completed"])),
            "failed": int(summary.get("failed_count", counts["failed"])),
            "pending": int(summary.get("pending_count", max(counts["total"] - counts["completed"] - counts["failed"], 0))),
            "estimated_cost_usd": (summary.get("usage_cost_estimate") or {}).get("usd"),
            "updated_at_utc": state.get("updated_at_utc"),
            "state_path": str(state_path.resolve()),
            "summary_path": str(summary_path.resolve()) if summary_path.is_file() else None,
            "outcomes": summary.get("outcomes") or [],
        })
    return sorted(jobs, key=lambda item: str(item.get("updated_at_utc") or ""), reverse=True)
