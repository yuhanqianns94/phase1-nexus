"""Content-addressed materials and atomic v3 artifact persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import KeyValue, Material


DEFAULT_OUTPUT_ROOT = Path("output/phase1_extractions_v3")


def active_review_root() -> Path:
    """Use a completed, locally activated rerun for the default review page."""
    marker = Path(__file__).resolve().parents[2] / "output/v3_active_extraction.json"
    try:
        root = Path(read_json(marker)["output_root"])
        if root.is_dir() and any(root.glob("*/reported_extraction.json")):
            return root
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return DEFAULT_OUTPUT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArticlePaths:
    output_root: Path
    document_id: str

    @property
    def root(self) -> Path:
        return self.output_root / self.document_id

    @property
    def material_manifest(self) -> Path:
        return self.root / "material_manifest.json"

    @property
    def document_ir(self) -> Path:
        return self.root / "document_ir.json"

    @property
    def extraction_packets(self) -> Path:
        return self.root / "extraction_packets.json"

    @property
    def request_summary(self) -> Path:
        return self.root / "request_summary.json"

    @property
    def reported_extraction(self) -> Path:
        return self.root / "reported_extraction.json"

    @property
    def extraction_audit(self) -> Path:
        return self.root / "extraction_audit.json"

    @property
    def resolved_extraction(self) -> Path:
        return self.root / "resolved_extraction.json"

    @property
    def curve_rows(self) -> Path:
        return self.root / "curve_rows.json"

    @property
    def rule_applications(self) -> Path:
        return self.root / "rule_applications.json"

    @property
    def issues(self) -> Path:
        return self.root / "issues.json"

    @property
    def human_review(self) -> Path:
        return self.root / "human_review.json"

    @property
    def review_units_csv(self) -> Path:
        return self.root / "review_units.csv"

    @property
    def review_summary(self) -> Path:
        return self.root / "review_summary.json"

    @property
    def friendly_human_review(self) -> Path:
        """User-facing review state; the legacy field-level review is preserved."""

        return self.root / "friendly_human_review.json"

    @property
    def friendly_review_units_csv(self) -> Path:
        return self.root / "friendly_review_units.csv"

    @property
    def friendly_review_summary(self) -> Path:
        return self.root / "friendly_review_summary.json"

    @property
    def friendly_review_undo(self) -> Path:
        return self.root / "friendly_review_undo.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def page_images(self) -> Path:
        return self.root / "page_images"

    @property
    def table_images(self) -> Path:
        return self.root / "table_images"

    @property
    def raw_responses(self) -> Path:
        return self.root / "raw_responses"

    @property
    def raw_requests(self) -> Path:
        return self.root / "raw_requests"

    @property
    def parsed_packets(self) -> Path:
        return self.root / "parsed_packets"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


def media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".xml":
        return "application/xml"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "image"
    return "other"


def register_materials(
    *,
    document_id: str,
    source_paths: Iterable[Path],
    roles: dict[str, str] | None = None,
) -> list[Material]:
    materials: list[Material] = []
    seen: set[str] = set()
    for source in source_paths:
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256_file(source)
        if digest in seen:
            continue
        seen.add(digest)
        role = (roles or {}).get(str(source))
        if role is None:
            role = "supplement" if re.search(r"(?:supp|suppl|appendix|\.dc\d)", source.name, re.I) else "main"
        if role not in {"main", "supplement", "table_attachment", "unknown"}:
            raise ValueError(f"Unsupported material role: {role}")
        materials.append(
            Material(
                material_id=f"mat_{digest[:16]}",
                path=str(source),
                media_type=media_type(source),
                role=role,
                sha256=digest,
                byte_count=source.stat().st_size,
                identifiers=[KeyValue(name="document_id", value=document_id)],
            )
        )
    if not materials:
        raise ValueError("At least one source material is required")
    return materials


def new_run_manifest(document_id: str, materials: list[Material]) -> dict[str, Any]:
    now = utc_now()
    return {
        "format": "phase1_extraction_manifest_v3",
        "document_id": document_id,
        "created_at_utc": now,
        "updated_at_utc": now,
        "status": "in_progress",
        "material_hashes": [m.sha256 for m in materials],
        "stages": {},
    }


def record_stage(
    manifest: dict[str, Any],
    stage: str,
    *,
    status: str,
    artifact: Path | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {"status": status, "updated_at_utc": utc_now()}
    if artifact is not None:
        entry["artifact"] = str(artifact)
        if artifact.is_file():
            entry["sha256"] = sha256_file(artifact)
    if details:
        entry["details"] = details
    manifest["stages"][stage] = entry
    manifest["updated_at_utc"] = entry["updated_at_utc"]
    if status == "failed":
        manifest["status"] = "partial"
