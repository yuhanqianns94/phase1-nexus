"""Pure helpers for reviewing historical articles before curve construction.

The Streamlit page deliberately loads only lightweight XML titles for the
candidate table.  Extraction JSON and PDF bytes are opened only for the one
article selected in the detail pane.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence
from urllib.parse import urlparse

from .helpers import PROJECT_ROOT


ARTICLE_REVIEW_PREVIEW_KEY = "historical_article_review_preview"
ARTICLE_REVIEW_FORM_FINGERPRINT_KEY = "historical_article_review_form_fingerprint"
ARTICLE_REVIEW_CONTEXT_FINGERPRINT_KEY = (
    "historical_article_review_context_fingerprint"
)
ARTICLE_REVIEW_STATE_PREFIX = "historical_article_review_"

_PMCID_RE = re.compile(r"PMC[0-9]+", re.IGNORECASE)
_DOCUMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def normalize_document_id(value: Any) -> str:
    """Return a safe article identifier, including reviewed external documents."""

    text = str(value or "").strip()
    if not _DOCUMENT_ID_RE.fullmatch(text):
        raise ValueError(f"Invalid historical article identifier: {value!r}")
    return text.upper() if _PMCID_RE.fullmatch(text) else text


def normalize_pmcid(value: Any) -> str:
    """Return one safe canonical PMCID or raise for an invalid identifier."""

    text = normalize_document_id(value).upper()
    if not _PMCID_RE.fullmatch(text):
        raise ValueError(f"Invalid PMC article identifier: {value!r}")
    return text


def article_source_url(pmcid: Any) -> str:
    """Build the canonical public PMC article page for one PMCID."""

    return f"https://pmc.ncbi.nlm.nih.gov/articles/{normalize_pmcid(pmcid)}/"


def _safe_http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value.strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_text(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def article_title(
    pmcid: Any,
    *,
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Read an article title from the local PMC XML without loading a PDF."""

    document_id = normalize_document_id(pmcid)
    if not _PMCID_RE.fullmatch(document_id):
        return document_id
    roots = (
        Path(project_root) / "output" / "pmc_xml_no_refs",
        Path(project_root) / "output" / "pmc_xml",
    )
    for root in roots:
        path = root / f"{document_id}.xml"
        if not path.is_file():
            continue
        try:
            xml_root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        for element in xml_root.iter():
            if _local_name(element.tag) == "article-title":
                title = _element_text(element)
                if title:
                    return title
    return document_id


def candidate_table_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Create one readable, article-level row per ranked candidate."""

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        document_id = normalize_document_id(
            candidate.get("document_id") or candidate.get("pmcid")
        )
        source_url = (
            article_source_url(document_id)
            if _PMCID_RE.fullmatch(document_id)
            else _safe_http_url(candidate.get("source_url"))
        )
        normalized_weight = candidate.get("normalized_candidate_weight")
        if normalized_weight is None:
            normalized_weight = candidate.get("normalized_weight")
        rows.append(
            {
                "Include": True,
                "Rank": candidate.get("rank") or candidate.get("article_rank"),
                "Article": document_id,
                "Title": article_title(document_id, project_root=project_root),
                "System similarity": candidate.get("similarity"),
                "Available field weight": candidate.get("available_weight"),
                "Provisional candidate share": normalized_weight,
                "Study units": candidate.get("study_unit_count", 1),
                "Usable curve lines": candidate.get("curve_line_count", 0),
                "Dose–DLT points": candidate.get("curve_source_point_count", 0),
                "Observed dose min": candidate.get("curve_observed_dose_min"),
                "Observed dose max": candidate.get("curve_observed_dose_max"),
                "Original article": source_url,
            }
        )
    return rows


def selected_document_ids(edited_rows: Any) -> tuple[str, ...]:
    """Extract checked PMCIDs from a Streamlit data-editor return value."""

    if hasattr(edited_rows, "to_dict"):
        rows = edited_rows.to_dict(orient="records")
    elif isinstance(edited_rows, list):
        rows = edited_rows
    else:
        rows = []
    selected: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not bool(row.get("Include")):
            continue
        document_id = normalize_document_id(row.get("Article"))
        if document_id not in selected:
            selected.append(document_id)
    return tuple(selected)


def _candidate_extraction_paths(
    candidate: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[Path]:
    raw_paths = candidate.get("extraction_paths")
    if not isinstance(raw_paths, list):
        raw_paths = [
            candidate.get("extraction_path")
            or candidate.get("path")
        ]
    paths: list[Path] = []
    for raw in raw_paths:
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(project_root) / path
        if path not in paths:
            paths.append(path)
    return paths


def load_extracted_content(
    candidate: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load the trusted extraction JSON referenced by one scored candidate."""

    errors: list[str] = []
    if candidate.get("v3_source_roots"):
        from phase1_prediction.evidence import discover_article_paths
        paths, _ = discover_article_paths(Path(root) for root in candidate["v3_source_roots"])
        source = paths.get(normalize_document_id(candidate.get("document_id")))
        if source is not None:
            raw = source.reported_extraction.read_bytes()
            return {"path": str(source.reported_extraction), "payload": json.loads(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    for path in _candidate_extraction_paths(candidate, project_root=project_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        if isinstance(payload, dict):
            return {
                "path": str(path),
                "payload": payload,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        errors.append(f"{path}: JSON root is not an object")
    return {
        "path": None,
        "payload": None,
        "sha256": None,
        "error": "; ".join(errors) or "No extraction JSON path was recorded.",
    }


def extracted_content_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a short human-readable summary while retaining full JSON separately."""

    if payload.get("document_id") and "dose_series" in payload:
        return {"Document": payload["document_id"], "Evidence view": "v3 reported extraction",
                "Studies": len(payload.get("studies", [])), "Dose series": len(payload.get("dose_series", [])),
                "Cohorts": len(payload.get("cohorts", [])), "Regimens": len(payload.get("regimens", []))}
    document = payload.get("document")
    if isinstance(document, Mapping):
        facts = payload.get("facts")
        study_units = payload.get("study_units")
        cohorts = payload.get("cohorts")
        return {
            "Document": document.get("document_id") or document.get("pmcid"),
            "Evidence view": payload.get("view"),
            "Study units": len(study_units) if isinstance(study_units, list) else 0,
            "Cohorts": len(cohorts) if isinstance(cohorts, list) else 0,
            "Extracted facts": len(facts) if isinstance(facts, list) else 0,
        }
    dlt_rows = payload.get("dlt_count_at_each_dose_level")
    if not isinstance(dlt_rows, list):
        dlt_rows = payload.get("DLT")
    if not isinstance(dlt_rows, list):
        dlt_rows = payload.get("dlt")
    if not isinstance(dlt_rows, list):
        dlt_rows = []
    return {
        "Drug / regimen": payload.get("Drug_Name"),
        "Cancer type": payload.get("cancer_type"),
        "Trial type": payload.get("Trial_Type"),
        "Trial design": payload.get("DesignType"),
        "Dose levels": len(payload.get("dose_levels") or []),
        "DLT rows": len(dlt_rows),
    }


def resolve_verified_pdf(
    pmcid: Any,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Resolve and hash-check a local PDF asset without trusting manifest paths."""

    document_id = normalize_document_id(pmcid)
    if not _PMCID_RE.fullmatch(document_id):
        return {
            "status": "unavailable",
            "reason": "No PMC PDF manifest exists for this external document.",
        }
    article_root = (
        Path(project_root) / "output" / "pmc_pdf_assets" / document_id
    ).resolve()
    manifest_path = article_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "unavailable", "reason": "No local PDF manifest."}
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "unavailable", "reason": f"Invalid PDF manifest: {error}"}
    if not isinstance(manifest, Mapping):
        return {"status": "unavailable", "reason": "Invalid PDF manifest object."}
    if str(manifest.get("pmcid") or "").upper() != document_id:
        return {"status": "unavailable", "reason": "PDF manifest PMCID mismatch."}
    if manifest.get("status") != "ready":
        return {
            "status": "unavailable",
            "reason": str(
                manifest.get("availability_reason")
                or manifest.get("error")
                or "PDF is not ready."
            ),
        }
    relative = manifest.get("pdf_path")
    if not isinstance(relative, str) or not relative.strip():
        return {"status": "unavailable", "reason": "PDF path is missing."}
    pdf_path = (article_root / relative).resolve()
    if pdf_path == article_root or article_root not in pdf_path.parents:
        return {"status": "unavailable", "reason": "Unsafe PDF manifest path."}
    if not pdf_path.is_file():
        return {"status": "unavailable", "reason": "Local PDF file is missing."}
    payload = pdf_path.read_bytes()
    expected_bytes = manifest.get("pdf_bytes")
    if expected_bytes is not None:
        try:
            if int(expected_bytes) != len(payload):
                return {
                    "status": "unavailable",
                    "reason": "Local PDF byte count does not match its manifest.",
                }
        except (TypeError, ValueError):
            return {"status": "unavailable", "reason": "Invalid PDF byte count."}
    digest = hashlib.sha256(payload).hexdigest()
    if digest != str(manifest.get("pdf_sha256") or ""):
        return {"status": "unavailable", "reason": "Local PDF SHA-256 mismatch."}
    return {
        "status": "ready",
        "path": str(pdf_path),
        "filename": pdf_path.name,
        "sha256": digest,
        "bytes": len(payload),
        "source_url": manifest.get("source_url"),
        "download_method": manifest.get("download_method"),
    }


def article_review_context_fingerprint(
    *,
    evidence_contract: str,
    extraction_library: str,
    bayesian_cache: str,
    embedding_cache: str,
) -> str:
    """Fingerprint storage/contract settings that make a preview reusable."""

    def path_state(raw_path: str) -> dict[str, Any]:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            return {"path": str(path), "kind": "missing"}
        if path.is_file():
            stat = path.stat()
            return {
                "path": str(path),
                "kind": "file",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        files: list[dict[str, Any]] = []
        for candidate in sorted(path.rglob("*.json")):
            if not candidate.is_file():
                continue
            stat = candidate.stat()
            files.append(
                {
                    "relative_path": str(candidate.relative_to(path)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return {"path": str(path), "kind": "directory", "json_files": files}

    payload = {
        "evidence_contract": evidence_contract,
        "extraction_library": path_state(extraction_library),
        "bayesian_cache": path_state(bayesian_cache),
        "embedding_cache": path_state(embedding_cache),
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def clear_article_review_state(state: MutableMapping[str, Any]) -> None:
    """Drop previews and dynamic editor/detail widgets from one session."""

    for key in list(state):
        if str(key).startswith(ARTICLE_REVIEW_STATE_PREFIX):
            state.pop(key, None)
