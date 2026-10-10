"""Build bounded extraction packets without flattening tables into prose."""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import BlockIR, DocumentIR, ReportedExtraction, TableBundle


DEFAULT_MAX_PACKET_CHARS = 150_000


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _block_payload(block: BlockIR) -> dict[str, Any]:
    return {
        "source_id": block.block_id,
        "kind": block.block_type,
        "page_id": block.page_id,
        "section_path": block.section_path,
        "text": block.text_raw,
        "bbox_normalized": block.bbox_normalized.model_dump() if block.bbox_normalized else None,
        "parser": block.parser,
    }


def _table_payload(table: TableBundle) -> dict[str, Any]:
    return {
        "source_id": table.table_id,
        "source_label": table.source_label,
        "caption": table.caption,
        "section_path": table.section_path,
        "parse_status": table.parse_status,
        "detection_kind": table.detection_kind,
        "visual_required": table.visual_required,
        "relevant_hint": table.relevant_hint,
        "footnotes": table.footnotes,
        "fragments": [
            {
                "source_id": fragment.fragment_id,
                "page_id": fragment.page_id,
                "bbox_normalized": fragment.bbox_normalized.model_dump() if fragment.bbox_normalized else None,
                "image_name": Path(fragment.image_path).name if fragment.image_path else None,
                "source": fragment.source,
                "continued_from_fragment_id": fragment.continued_from_fragment_id,
            }
            for fragment in table.fragments
        ],
        "cells": [
            {
                "source_id": cell.cell_id,
                "row": cell.row_index,
                "column": cell.column_index,
                "row_span": cell.row_span,
                "column_span": cell.column_span,
                "is_header": cell.is_header,
                "text": cell.text_raw,
                "row_header_path": cell.row_header_path,
                "column_header_path": cell.column_header_path,
                "footnote_markers": cell.footnote_markers,
                "bbox_normalized": cell.bbox_normalized.model_dump() if cell.bbox_normalized else None,
            }
            for cell in table.cells
        ],
        "unresolved_notes": table.unresolved_notes,
    }


def _table_units(table: TableBundle, max_packet_chars: int) -> list[dict[str, Any]]:
    payload = _table_payload(table)
    if _json_size(payload) <= max_packet_chars * 0.72 or table.row_count <= 20:
        return [{"unit_kind": "table", "table": payload}]
    header_rows = sorted({cell.row_index for cell in table.cells if cell.is_header})
    data_rows = sorted({cell.row_index for cell in table.cells if cell.row_index not in header_rows})
    if not data_rows:
        return [{"unit_kind": "table", "table": payload}]
    units: list[dict[str, Any]] = []
    step = 19
    for offset in range(0, len(data_rows), step):
        selected_rows = data_rows[max(0, offset - (1 if offset else 0)) : offset + 20]
        selected = set(header_rows + selected_rows)
        sliced = dict(payload)
        sliced["cells"] = [cell for cell in payload["cells"] if cell["row"] in selected]
        sliced["slice"] = {
            "original_row_count": table.row_count,
            "included_header_rows": header_rows,
            "included_data_rows": selected_rows,
            "overlap_rows": selected_rows[:1] if offset else [],
        }
        units.append({"unit_kind": "table", "table": sliced})
    return units


def entity_index(extraction: ReportedExtraction | None) -> dict[str, list[dict[str, str]]]:
    if extraction is None:
        return {}
    result: dict[str, list[dict[str, str]]] = {}
    for field in (
        "studies", "study_parts", "arms", "interventions", "regimens",
        "designs", "dlt_windows", "dose_series", "cohorts", "result_scopes",
    ):
        values = getattr(extraction, field)
        indexed: list[dict[str, str]] = []
        for value in values:
            data = value.model_dump(mode="json")
            id_value = next((str(v) for k, v in data.items() if k.endswith("_id") and v), "")
            label = str(data.get("label") or data.get("raw_name") or data.get("author_name") or "")
            indexed.append({"id": id_value, "label": label})
            if field == "cohorts":
                for exposure in data.get("exposures", []):
                    indexed.append({"id": exposure["exposure_id"],
                                    "label": f"{id_value}: {exposure['name_raw']} ({exposure['presence']})"})
        result[field] = indexed
    return result


def build_extraction_packets(
    document: DocumentIR,
    *,
    max_packet_chars: int = DEFAULT_MAX_PACKET_CHARS,
) -> list[dict[str, Any]]:
    """Create complete source packets; oversized papers split on section/table boundaries."""

    if max_packet_chars < 10_000:
        raise ValueError("max_packet_chars must be at least 10000")
    blocks_by_section: dict[tuple[str, ...], list[BlockIR]] = defaultdict(list)
    for block in document.blocks:
        blocks_by_section[tuple(block.section_path)].append(block)
    units: list[dict[str, Any]] = []
    for section, blocks in blocks_by_section.items():
        current: list[dict[str, Any]] = []
        for block in blocks:
            payload = _block_payload(block)
            if current and _json_size(current + [payload]) > max_packet_chars * 0.72:
                units.append({"unit_kind": "text", "section_path": list(section), "items": current})
                current = []
            current.append(payload)
        if current:
            units.append({"unit_kind": "text", "section_path": list(section), "items": current})
    for table in document.tables:
        units.extend(_table_units(table, max_packet_chars))
    # Every rendered PDF/image page gets a visual route. Native text can omit
    # embedded raster tables, vector glyphs, spatial headers, and footnotes.
    # Attachment de-duplication later ensures a table context page is sent once.
    for page in document.pages:
        if page.image_path:
            units.append(
                {
                    "unit_kind": "page_visual",
                    "page": {
                        "source_id": page.page_id,
                        "page_number": page.page_number,
                        "native_text_status": page.native_text_status,
                        "ocr_status": page.ocr_status,
                        "image_name": Path(page.image_path).name,
                        "visual_reason": "full_page_layout_and_embedded_image_coverage",
                    },
                }
            )

    common = {
        "document_id": document.document_id,
        "title": document.title,
        "identifiers": [item.model_dump(mode="json") for item in document.identifiers],
        "materials": [
            {
                "material_id": item.material_id,
                "media_type": item.media_type,
                "role": item.role,
                "sha256": item.sha256,
                "status": item.status,
            }
            for item in document.materials
        ],
        "table_inventory": [
            {
                "table_id": table.table_id,
                "label": table.source_label,
                "caption": table.caption,
                "pages": [fragment.page_id for fragment in table.fragments if fragment.page_id],
                "parse_status": table.parse_status,
                "detection_kind": table.detection_kind,
                "visual_required": table.visual_required,
            }
            for table in document.tables
        ],
        "material_reference_inventory": [
            item.model_dump(mode="json") for item in document.material_references
        ],
        "document_issues": document.issues,
    }
    packets: list[dict[str, Any]] = []
    current_units: list[dict[str, Any]] = []
    for unit in units:
        candidate = {**common, "source_units": current_units + [unit]}
        if current_units and _json_size(candidate) > max_packet_chars:
            packets.append({**common, "source_units": current_units})
            current_units = []
        current_units.append(unit)
    if current_units or not packets:
        packets.append({**common, "source_units": current_units})
    for index, packet in enumerate(packets, start=1):
        packet["packet_id"] = f"packet_{index:04d}"
        packet["packet_index"] = index
        packet["packet_count"] = len(packets)
        packet["is_final_packet"] = index == len(packets)
    return packets


def packet_image_paths(
    packet: dict[str, Any],
    document: DocumentIR,
    *,
    limit: int | None = None,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    page_by_id = {page.page_id: page for page in document.pages}
    table_by_id = {table.table_id: table for table in document.tables}
    for unit in packet.get("source_units", []):
        if unit.get("unit_kind") == "page_visual":
            page = page_by_id.get((unit.get("page") or {}).get("source_id"))
            if page and page.image_path and Path(page.image_path).is_file() and page.image_path not in seen:
                seen.add(page.image_path)
                paths.append(Path(page.image_path))
            continue
        if unit.get("unit_kind") != "table":
            continue
        table_payload = unit.get("table") or {}
        table = table_by_id.get(table_payload.get("source_id"))
        if table is None:
            continue
        for fragment in table.fragments:
            image_path = fragment.image_path
            candidate = Path(image_path) if image_path else None
            if candidate and candidate.is_file() and str(candidate) not in seen:
                seen.add(str(candidate))
                paths.append(candidate)
            page = page_by_id.get(fragment.page_id or "")
            if page and page.image_path and Path(page.image_path).is_file() and page.image_path not in seen:
                seen.add(page.image_path)
                paths.append(Path(page.image_path))
    return paths if limit is None else paths[:limit]


def packet_image_inventory(packet: dict[str, Any], document: DocumentIR) -> list[dict[str, Any]]:
    """Return every visual dependency, including missing files, in request order."""

    selected = packet_image_paths(packet, document)
    selected_set = {str(path.resolve()) for path in selected}
    candidates: list[tuple[str, str | None, str, bool, str | None]] = []
    page_by_id = {page.page_id: page for page in document.pages}
    table_by_id = {table.table_id: table for table in document.tables}
    for unit in packet.get("source_units", []):
        kind = unit.get("unit_kind")
        if kind == "page_visual":
            page = page_by_id.get((unit.get("page") or {}).get("source_id"))
            if page:
                candidates.append((page.page_id, page.image_path, "full_page_visual", True, None))
        elif kind == "table":
            table = table_by_id.get((unit.get("table") or {}).get("source_id"))
            if table:
                for fragment in table.fragments:
                    candidates.append((
                        fragment.fragment_id,
                        fragment.image_path,
                        "table_region",
                        table.visual_required,
                        page_by_id.get(fragment.page_id or "").image_path
                        if page_by_id.get(fragment.page_id or "") else None,
                    ))
                    page = page_by_id.get(fragment.page_id or "")
                    if page:
                        candidates.append((page.page_id, page.image_path, "table_context_page", True, None))
    inventory: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for source_id, raw_path, role, required, fallback_raw_path in candidates:
        marker = (source_id, raw_path)
        if marker in seen:
            continue
        seen.add(marker)
        path = Path(raw_path).resolve() if raw_path else None
        exists = bool(path and path.is_file())
        fallback_path = Path(fallback_raw_path).resolve() if fallback_raw_path else None
        fallback_exists = bool(
            fallback_path
            and fallback_path.is_file()
            and str(fallback_path) in selected_set
        )
        inventory.append(
            {
                "source_id": source_id,
                "role": role,
                "required": required,
                "path": str(path) if path else None,
                "fallback_path": str(fallback_path) if fallback_path else None,
                "status": (
                    "included" if exists and str(path) in selected_set
                    else "covered_by_context_page" if fallback_exists
                    else "missing" if required
                    else "not_required"
                ),
                "sha256": (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    if exists and path is not None
                    else None
                ),
                "byte_count": path.stat().st_size if exists and path is not None else None,
            }
        )
    return inventory
