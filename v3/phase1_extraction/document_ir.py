"""Build a unified JATS/PDF document representation with table provenance."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image

from .models import (
    BlockIR,
    CoverageRecord,
    DocumentIR,
    KeyValue,
    Material,
    MaterialReference,
    PageIR,
    Rect,
    TableBundle,
    TableCellIR,
    TableFragmentIR,
)


_WS = re.compile(r"\s+")
_TABLE_CUE = re.compile(r"\b(?:table|dose\s+level|cohort)\s*[A-Z0-9IVX.-]*", re.I)
_RELEVANT = re.compile(
    r"\b(?:dose|dosing|cohort|treated|enrolled|evaluable|DLT|dose[- ]limiting|toxicit|patient)\b",
    re.I,
)
_TABLE_LABEL = re.compile(r"\b(Table\s+[A-Z0-9IVX.-]+)", re.I)
_SUPPLEMENT_REFERENCE = re.compile(
    r"\b(?:supplement(?:ary)?\s+(?:appendix|table|figure|methods?|materials?|data|information|file)|eappendix)\s*[A-Z0-9.-]*",
    re.I,
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _WS.sub(" ", "".join(element.itertext())).strip()


def _id(prefix: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _rect(values: Iterable[float]) -> Rect:
    x0, y0, x1, y1 = values
    return Rect(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))


def _normalized_rect(values: Iterable[float], width: float, height: float) -> Rect:
    x0, y0, x1, y1 = (float(value) for value in values)
    return Rect(x0=x0 / width, y0=y0 / height, x1=x1 / width, y1=y1 / height)


def _attr_int(element: ET.Element, name: str, default: int = 1) -> int:
    try:
        return max(1, int(element.attrib.get(name, default)))
    except (TypeError, ValueError):
        return default


def _jats_table(
    wrap: ET.Element,
    *,
    material: Material,
    section_path: list[str],
    ordinal: int,
) -> TableBundle:
    label = _text(next((x for x in wrap if _local(x.tag) == "label"), None)) or None
    caption = _text(next((x for x in wrap if _local(x.tag) == "caption"), None)) or None
    table = next((x for x in wrap.iter() if _local(x.tag) == "table"), None)
    footnote_nodes = [x for x in wrap.iter() if _local(x.tag) == "fn"]
    if not footnote_nodes:
        footnote_nodes = [x for x in wrap.iter() if _local(x.tag) == "table-wrap-foot"]
    footnotes = list(dict.fromkeys(_text(x) for x in footnote_nodes if _text(x)))
    graphic = next((x for x in wrap.iter() if _local(x.tag) == "graphic"), None)
    href = None
    if graphic is not None:
        href = next((v for k, v in graphic.attrib.items() if _local(k) == "href"), None)
        if href:
            local_asset = (Path(material.path).parent / href).resolve()
            if local_asset.is_file():
                href = str(local_asset)
    table_id = _id("tbl", material.material_id, wrap.attrib.get("id", ordinal), label, caption)
    fragments = [
        TableFragmentIR(
            fragment_id=_id("frag", table_id, 0),
            material_id=material.material_id,
            image_path=href,
            source="jats",
        )
    ]
    if table is None:
        return TableBundle(
            table_id=table_id,
            material_id=material.material_id,
            source_label=label,
            caption=caption,
            section_path=section_path,
            fragments=fragments,
            row_count=0,
            column_count=0,
            footnotes=footnotes,
            parse_status="visual_only" if href else "failed",
            relevant_hint=bool(_RELEVANT.search(" ".join(filter(None, [label, caption])))),
            detection_kind="visual_candidate",
            visual_required=True,
            unresolved_notes=["JATS table is image-backed; cells require visual reading"] if href else ["JATS table has no readable grid"],
        )

    rows = [x for x in table.iter() if _local(x.tag) == "tr"]
    occupied: dict[tuple[int, int], str] = {}
    cells: list[TableCellIR] = []
    max_col = 0
    for row_index, row in enumerate(rows):
        col = 0
        for element in list(row):
            if _local(element.tag) not in {"th", "td"}:
                continue
            while (row_index, col) in occupied:
                col += 1
            row_span = _attr_int(element, "rowspan")
            col_span = _attr_int(element, "colspan")
            raw = _text(element)
            cell_id = _id("cell", table_id, row_index, col, raw)
            for rr in range(row_index, row_index + row_span):
                for cc in range(col, col + col_span):
                    occupied[(rr, cc)] = cell_id
            cells.append(
                TableCellIR(
                    cell_id=cell_id,
                    row_index=row_index,
                    column_index=col,
                    row_span=row_span,
                    column_span=col_span,
                    is_header=_local(element.tag) == "th",
                    text_raw=raw,
                    native_text=raw,
                )
            )
            col += col_span
            max_col = max(max_col, col)

    for cell in cells:
        cell.column_header_path = [
            candidate.text_raw
            for candidate in cells
            if candidate.is_header
            and candidate.row_index <= cell.row_index
            and candidate.column_index <= cell.column_index < candidate.column_index + candidate.column_span
            and candidate.text_raw
        ]
        cell.row_header_path = [
            candidate.text_raw
            for candidate in cells
            if candidate.is_header
            and candidate.row_index == cell.row_index
            and candidate.column_index < cell.column_index
            and candidate.text_raw
        ]
    return TableBundle(
        table_id=table_id,
        material_id=material.material_id,
        source_label=label,
        caption=caption,
        section_path=section_path,
        fragments=fragments,
        row_count=len(rows),
        column_count=max_col,
        cells=cells,
        footnotes=footnotes,
        parse_status="complete" if cells else "visual_only",
        relevant_hint=bool(_RELEVANT.search(" ".join(filter(None, [label, caption, _text(table)])))),
        detection_kind="data_table" if cells else "visual_candidate",
        visual_required=not bool(cells),
    )


def _parse_jats(material: Material) -> tuple[str | None, list[KeyValue], list[BlockIR], list[TableBundle], list[str]]:
    root = ET.parse(material.path).getroot()
    title_element = next((x for x in root.iter() if _local(x.tag) == "article-title"), None)
    title = _text(title_element) or None
    identifiers = [
        KeyValue(name=x.attrib.get("pub-id-type", "article-id"), value=_text(x))
        for x in root.iter()
        if _local(x.tag) == "article-id" and _text(x)
    ]
    blocks: list[BlockIR] = []
    tables: list[TableBundle] = []
    order = 0

    def add_block(kind: str, text: str, path: list[str]) -> None:
        nonlocal order
        text = text.strip()
        if not text:
            return
        block_id = _id("src", material.material_id, order, kind, text)
        blocks.append(
            BlockIR(
                block_id=block_id,
                material_id=material.material_id,
                block_type=kind,
                text_raw=text,
                text_normalized=_WS.sub(" ", text),
                section_path=path,
                reading_order=order,
                parser="jats",
            )
        )
        order += 1

    if title:
        add_block("title", title, [])
    for abstract in (x for x in root.iter() if _local(x.tag) == "abstract"):
        for paragraph in (x for x in abstract.iter() if _local(x.tag) == "p"):
            add_block("abstract", _text(paragraph), ["Abstract"])

    body = next((x for x in root.iter() if _local(x.tag) == "body"), None)

    processed_nodes: set[int] = set()
    processed_tables: set[int] = set()

    def walk(parent: ET.Element, path: list[str]) -> None:
        nonlocal order
        for child in list(parent):
            tag = _local(child.tag)
            if tag == "sec":
                heading = _text(next((x for x in child if _local(x.tag) == "title"), None))
                child_path = path + ([heading] if heading else [])
                if heading:
                    add_block("section_title", heading, child_path)
                walk(child, child_path)
            elif tag == "p":
                if id(child) not in processed_nodes:
                    add_block("paragraph", _text(child), path)
                    processed_nodes.add(id(child))
            elif tag in {"list-item", "disp-quote"}:
                if id(child) not in processed_nodes:
                    add_block("list_item", _text(child), path)
                    processed_nodes.add(id(child))
            elif tag == "table-wrap":
                if id(child) in processed_tables:
                    continue
                processed_tables.add(id(child))
                bundle = _jats_table(child, material=material, section_path=path, ordinal=len(tables))
                tables.append(bundle)
                if bundle.caption:
                    add_block("caption", bundle.caption, path)
                for note in bundle.footnotes:
                    add_block("footnote", note, path)
            elif tag == "fig":
                caption = next((x for x in child if _local(x.tag) == "caption"), None)
                if caption is not None:
                    add_block("caption", _text(caption), path)
            elif tag in {"list", "table-wrap-group", "floats-group", "boxed-text", "supplementary-material"}:
                walk(child, path)
            else:
                # JATS permits publisher-specific containers. Recurse so that
                # paragraphs and table-wrap elements are not lost merely
                # because an intermediate tag is unfamiliar.
                if list(child):
                    walk(child, path)

    if body is not None:
        walk(body, [])
    for floats in (x for x in root.iter() if _local(x.tag) == "floats-group"):
        walk(floats, ["Floats"])
    for previous, following in zip(blocks, blocks[1:]):
        previous.next_block_id = following.block_id
        following.previous_block_id = previous.block_id
    issues: list[str] = []
    if body is None:
        issues.append("JATS body is absent; PDF or supplement is required for full extraction")
    return title, identifiers, blocks, tables, issues


def _page_blocks(page: fitz.Page, material: Material, page_id: str, order_start: int) -> list[BlockIR]:
    blocks: list[BlockIR] = []
    for raw in page.get_text("blocks", sort=True):
        if len(raw) < 7 or int(raw[6]) != 0:
            continue
        text = str(raw[4]).strip()
        if not text:
            continue
        bbox = raw[:4]
        block_id = _id("pdf", material.material_id, page.number + 1, *bbox, text)
        blocks.append(
            BlockIR(
                block_id=block_id,
                material_id=material.material_id,
                page_id=page_id,
                block_type="page_text",
                text_raw=text,
                text_normalized=_WS.sub(" ", text),
                reading_order=order_start + len(blocks),
                bbox=_rect(bbox),
                bbox_normalized=_normalized_rect(bbox, page.rect.width, page.rect.height),
                parser="pymupdf",
            )
        )
    return blocks


def _ocr_page(page: fitz.Page) -> str | None:
    executable = shutil.which("tesseract")
    if executable is None:
        return None
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
    process = subprocess.run(
        [executable, "stdin", "stdout", "-l", "eng"],
        input=pixmap.tobytes("png"),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode != 0:
        return None
    text = process.stdout.decode("utf-8", errors="replace").strip()
    return text or None


def _pymupdf_table(
    table: object,
    *,
    material: Material,
    page: fitz.Page,
    page_id: str,
    ordinal: int,
    image_dir: Path,
    source: str,
) -> TableBundle:
    bbox = tuple(float(x) for x in table.bbox)
    table_id = _id("tbl", material.material_id, page.number + 1, ordinal, *bbox)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{table_id}.png"
    pixmap = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=fitz.Rect(bbox), alpha=False)
    pixmap.save(image_path)
    extracted = table.extract()
    caption_candidates: list[tuple[float, str]] = []
    footnotes: list[str] = []
    referenced_by: list[str] = []
    for block in page.get_text("blocks", sort=True):
        if len(block) < 7 or int(block[6]) != 0:
            continue
        text = _WS.sub(" ", str(block[4])).strip()
        if _TABLE_LABEL.search(text) and float(block[3]) <= bbox[1] + 8 and bbox[1] - float(block[3]) <= 180:
            caption_candidates.append((float(block[3]), text))
            referenced_by.append(_id("pdf", material.material_id, page.number + 1, *block[:4], str(block[4]).strip()))
        if (
            float(block[1]) >= bbox[3] - 4
            and float(block[1]) - bbox[3] <= 140
            and re.match(r"^(?:[*†‡]|[a-z][.)]|note\s*:|abbreviations?\s*:)", text, re.I)
        ):
            footnotes.append(text)
            referenced_by.append(_id("pdf", material.material_id, page.number + 1, *block[:4], str(block[4]).strip()))
    caption = max(caption_candidates, default=(0.0, ""))[1] or None
    label_match = _TABLE_LABEL.search(caption or "")
    source_label = label_match.group(1) if label_match else None
    cells: list[TableCellIR] = []
    for row_index, row in enumerate(table.rows):
        for column_index, cell_bbox in enumerate(row.cells):
            raw = ""
            if row_index < len(extracted) and column_index < len(extracted[row_index]):
                raw = extracted[row_index][column_index] or ""
            if cell_bbox is None:
                cells.append(
                    TableCellIR(
                        cell_id=_id("cell", table_id, row_index, column_index, "span"),
                        row_index=row_index,
                        column_index=column_index,
                        text_raw=raw,
                        native_text=raw or None,
                        parse_status="partial",
                    )
                )
                continue
            cells.append(
                TableCellIR(
                    cell_id=_id("cell", table_id, row_index, column_index, raw),
                    row_index=row_index,
                    column_index=column_index,
                    is_header=row_index == 0,
                    text_raw=raw,
                    native_text=raw,
                    footnote_markers=re.findall(r"(?:[*†‡]+|(?<=\d)[a-z])$", raw.strip()),
                    bbox=_rect(cell_bbox),
                    bbox_normalized=_normalized_rect(cell_bbox, page.rect.width, page.rect.height),
                )
            )
    # Preserve physical column indices.  Filtering an empty top-left header and
    # then indexing the compacted list shifts every later header one column.
    headers = {
        cell.column_index: cell.text_raw
        for cell in cells
        if cell.row_index == 0 and cell.text_raw
    }
    first_row = [cell for cell in cells if cell.row_index == 0]
    second_row = [cell for cell in cells if cell.row_index == 1]
    second_numeric = sum(bool(re.search(r"\d", cell.text_raw)) for cell in second_row if cell.text_raw)
    second_nonempty = sum(bool(cell.text_raw.strip()) for cell in second_row)
    multilevel_header = bool(
        first_row
        and second_row
        and (second_nonempty == 0 or second_numeric * 2 < second_nonempty)
        and (
            any(not cell.text_raw.strip() for cell in first_row)
            or len([cell for cell in first_row if cell.text_raw.strip()]) < int(table.col_count)
        )
    )
    if multilevel_header:
        for cell in second_row:
            cell.is_header = True
    for cell in cells:
        if cell.row_index > 0:
            header_path: list[str] = []
            if headers.get(cell.column_index):
                header_path.append(headers[cell.column_index])
            if multilevel_header:
                second = next(
                    (
                        candidate.text_raw
                        for candidate in second_row
                        if candidate.column_index == cell.column_index and candidate.text_raw
                    ),
                    "",
                )
                if second and second not in header_path:
                    header_path.append(second)
            cell.column_header_path = header_path
        if cell.column_index > 0:
            row_header = next(
                (
                    candidate.text_raw
                    for candidate in cells
                    if candidate.row_index == cell.row_index
                    and candidate.column_index < cell.column_index
                    and candidate.text_raw
                ),
                "",
            )
            if row_header:
                cell.row_header_path = [row_header]
    joined = " ".join(cell.text_raw for cell in cells)
    nonempty_cells = [cell for cell in cells if cell.text_raw.strip()]
    numeric_cells = [cell for cell in nonempty_cells if re.search(r"\d", cell.text_raw)]
    average_cell_chars = (
        sum(len(cell.text_raw) for cell in nonempty_cells) / len(nonempty_cells)
        if nonempty_cells else 0.0
    )
    layout_candidate = bool(
        nonempty_cells
        and not numeric_cells
        and average_cell_chars > 80
    )
    unresolved_notes = []
    if layout_candidate:
        unresolved_notes.append(
            "Detected grid resembles page layout rather than a verified numeric data table; retain the page image"
        )
    return TableBundle(
        table_id=table_id,
        material_id=material.material_id,
        source_label=source_label,
        caption=caption,
        fragments=[
            TableFragmentIR(
                fragment_id=_id("frag", table_id, 0),
                material_id=material.material_id,
                page_id=page_id,
                bbox=_rect(bbox),
                bbox_normalized=_normalized_rect(bbox, page.rect.width, page.rect.height),
                image_path=str(image_path.resolve()),
                source=source,
            )
        ],
        row_count=int(table.row_count),
        column_count=int(table.col_count),
        cells=cells,
        footnotes=list(dict.fromkeys(footnotes)),
        referenced_by_block_ids=list(dict.fromkeys(referenced_by)),
        parse_status="partial" if layout_candidate else "complete" if cells else "visual_only",
        relevant_hint=bool(_RELEVANT.search(joined)),
        detection_kind="layout_candidate" if layout_candidate else "data_table" if cells else "visual_candidate",
        visual_required=True,
        unresolved_notes=unresolved_notes,
    )


def _merge_continued_tables(tables: list[TableBundle], page_numbers: dict[str, int]) -> list[TableBundle]:
    merged: list[TableBundle] = []
    for table in tables:
        if not merged:
            merged.append(table)
            continue
        previous = merged[-1]
        previous_page = page_numbers.get(previous.fragments[-1].page_id or "")
        current_page = page_numbers.get(table.fragments[0].page_id or "")
        same_label = (
            previous.source_label
            and table.source_label
            and previous.source_label.casefold() == table.source_label.casefold()
        )
        marked_continued = "continued" in (table.caption or "").casefold()
        compatible_grid = (
            previous.column_count == table.column_count
            or previous.column_count == 0
            or table.column_count == 0
        )
        if previous_page is None or current_page != previous_page + 1 or not compatible_grid or not (same_label or marked_continued):
            merged.append(table)
            continue
        offset = previous.row_count
        previous_headers = [
            cell.text_raw for cell in previous.cells if cell.row_index == 0
        ]
        current_headers = [
            cell.text_raw for cell in table.cells if cell.row_index == 0
        ]
        skip_first = bool(previous_headers and previous_headers == current_headers)
        if skip_first:
            offset -= 1
        for cell in table.cells:
            if skip_first and cell.row_index == 0:
                continue
            cell.row_index += offset
            previous.cells.append(cell)
        fragment = table.fragments[0]
        fragment.continued_from_fragment_id = previous.fragments[-1].fragment_id
        previous.fragments.extend(table.fragments)
        previous.row_count += table.row_count - (1 if skip_first else 0)
        previous.column_count = max(previous.column_count, table.column_count)
        previous.footnotes.extend(note for note in table.footnotes if note not in previous.footnotes)
        previous.unresolved_notes.extend(note for note in table.unresolved_notes if note not in previous.unresolved_notes)
        previous.relevant_hint = previous.relevant_hint or table.relevant_hint
        if previous.parse_status != table.parse_status:
            previous.parse_status = "partial"
    return merged


def _parse_pdf(
    material: Material,
    *,
    page_image_dir: Path,
    table_image_dir: Path,
    render_pages: bool,
) -> tuple[str | None, list[PageIR], list[BlockIR], list[TableBundle], list[str]]:
    document = fitz.open(material.path)
    if document.needs_pass:
        raise ValueError(f"Encrypted PDF requires a password: {material.path}")
    title = str(document.metadata.get("title") or "").strip() or None
    pages: list[PageIR] = []
    blocks: list[BlockIR] = []
    tables: list[TableBundle] = []
    issues: list[str] = []
    if render_pages:
        page_image_dir.mkdir(parents=True, exist_ok=True)
    for page in document:
        page_id = _id("page", material.material_id, page.number + 1)
        native_text = page.get_text("text", sort=True).strip()
        status = "readable" if len(native_text) >= 100 else "sparse" if native_text else "none"
        image_path: Path | None = None
        if render_pages:
            image_path = page_image_dir / f"{page_id}.png"
            page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False).save(image_path)
        page_ir = PageIR(
            page_id=page_id,
            material_id=material.material_id,
            page_number=page.number + 1,
            width=page.rect.width,
            height=page.rect.height,
            rotation=page.rotation,
            native_text_status=status,
            ocr_status="not_needed" if status == "readable" else "not_run",
            image_path=str(image_path.resolve()) if image_path else None,
            text_char_count=len(native_text),
            visual_inventory_status="available" if image_path and image_path.is_file() else "not_rendered",
        )
        pages.append(page_ir)
        page_blocks = _page_blocks(page, material, page_id, len(blocks))
        if status != "readable":
            ocr_text = _ocr_page(page)
            if ocr_text:
                page_ir.ocr_status = "completed"
                page_blocks.append(
                    BlockIR(
                        block_id=_id("ocr", material.material_id, page.number + 1, ocr_text),
                        material_id=material.material_id,
                        page_id=page_id,
                        block_type="page_text",
                        text_raw=ocr_text,
                        text_normalized=_WS.sub(" ", ocr_text),
                        reading_order=len(blocks) + len(page_blocks),
                        bbox=Rect(x0=0, y0=0, x1=page.rect.width, y1=page.rect.height),
                        bbox_normalized=Rect(x0=0, y0=0, x1=1, y1=1),
                        parser="ocr",
                    )
                )
            elif status != "readable":
                page_ir.ocr_status = "failed" if shutil.which("tesseract") else "not_run"
                issues.append(f"PDF page {page.number + 1} has sparse text and no completed OCR")
        blocks.extend(page_blocks)

        detected: list[object] = []
        # Whole-page text strategies frequently turn multi-column prose into a
        # huge false table.  Unruled tables are handled from caption-bounded
        # visual regions below; grid extraction stays line-based here.
        for strategy in ("lines", "lines_strict"):
            try:
                found = page.find_tables(strategy=strategy)
            except Exception as error:
                issues.append(f"Table detection failed on page {page.number + 1} ({strategy}): {error}")
                continue
            for table in found.tables:
                if table.row_count >= 2 and table.col_count >= 2 and all(tuple(table.bbox) != tuple(x.bbox) for x in detected):
                    detected.append(table)
        for table in detected:
            tables.append(
                _pymupdf_table(
                    table,
                    material=material,
                    page=page,
                    page_id=page_id,
                    ordinal=len(tables),
                    image_dir=table_image_dir,
                    source="pymupdf_lines",
                )
            )
        if _TABLE_CUE.search(native_text):
            caption_blocks: list[tuple[str, tuple[float, float, float, float]]] = []
            for block in page.get_text("blocks", sort=True):
                if len(block) < 7 or int(block[6]) != 0:
                    continue
                candidate = _WS.sub(" ", str(block[4])).strip()
                if re.match(r"^Table\s+[A-Z0-9IVX.-]+", candidate, re.I):
                    caption_blocks.append((candidate, tuple(float(value) for value in block[:4])))
            if not caption_blocks:
                caption_blocks = [("", (0.0, 0.0, page.rect.width, page.rect.height))]
            seen_labels: set[str] = set()
            detected_labels = {
                (table.source_label or "").casefold()
                for table in tables
                if table.fragments
                and table.fragments[-1].page_id == page_id
                and table.source_label
            }
            for cue_index, (cue_block, cue_bbox) in enumerate(caption_blocks):
                label_match = _TABLE_LABEL.search(cue_block)
                source_label = label_match.group(1) if label_match else None
                label_key = (source_label or f"unlabeled-{page.number + 1}").casefold()
                if label_key in seen_labels or label_key in detected_labels:
                    continue
                seen_labels.add(label_key)
                table_id = _id("tblvis", material.material_id, page.number + 1, source_label)
                if cue_block:
                    next_y = page.rect.height
                    for _, other_bbox in caption_blocks[cue_index + 1 :]:
                        if other_bbox[1] > cue_bbox[1]:
                            next_y = other_bbox[1] - 6
                            break
                    crop_bbox = (
                        0.0,
                        max(0.0, cue_bbox[1] - 8),
                        page.rect.width,
                        min(page.rect.height, max(cue_bbox[3] + 80, next_y)),
                    )
                    table_image_dir.mkdir(parents=True, exist_ok=True)
                    candidate_image = table_image_dir / f"{table_id}.png"
                    page.get_pixmap(
                        matrix=fitz.Matrix(300 / 72, 300 / 72),
                        clip=fitz.Rect(crop_bbox),
                        alpha=False,
                    ).save(candidate_image)
                else:
                    crop_bbox = (0.0, 0.0, page.rect.width, page.rect.height)
                    candidate_image = image_path
                tables.append(
                    TableBundle(
                        table_id=table_id,
                        material_id=material.material_id,
                        source_label=source_label,
                        caption=cue_block or None,
                        fragments=[
                            TableFragmentIR(
                                fragment_id=_id("frag", table_id, 0),
                                material_id=material.material_id,
                                page_id=page_id,
                                bbox=_rect(crop_bbox),
                                bbox_normalized=_normalized_rect(crop_bbox, page.rect.width, page.rect.height),
                                image_path=str(candidate_image.resolve()) if candidate_image else None,
                                source="visual_candidate",
                            )
                        ],
                        row_count=0,
                        column_count=0,
                        parse_status="visual_only",
                        relevant_hint=bool(_RELEVANT.search(" ".join([cue_block, native_text]))),
                        detection_kind="visual_candidate",
                        visual_required=True,
                        unresolved_notes=["Potential unruled or image table; inspect the page image and PDF layout"],
                    )
                )
    for previous, following in zip(blocks, blocks[1:]):
        previous.next_block_id = following.block_id
        following.previous_block_id = previous.block_id
    tables = _merge_continued_tables(tables, {page.page_id: page.page_number for page in pages})
    return title, pages, blocks, tables, issues


def _parse_image_material(
    material: Material,
) -> tuple[list[PageIR], list[BlockIR], list[TableBundle], list[str]]:
    with Image.open(material.path) as image:
        width, height = image.size
    page_id = _id("page", material.material_id, 1)
    ocr_text: str | None = None
    ocr_status = "not_run"
    executable = shutil.which("tesseract")
    if executable:
        process = subprocess.run(
            [executable, material.path, "stdout", "-l", "eng"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.returncode == 0:
            ocr_text = process.stdout.decode("utf-8", errors="replace").strip() or None
            ocr_status = "completed" if ocr_text else "failed"
        else:
            ocr_status = "failed"
    page = PageIR(
        page_id=page_id,
        material_id=material.material_id,
        page_number=1,
        width=float(width),
        height=float(height),
        native_text_status="none",
        ocr_status=ocr_status,
        image_path=str(Path(material.path).resolve()),
        text_char_count=len(ocr_text or ""),
        visual_inventory_status="available",
    )
    blocks: list[BlockIR] = []
    if ocr_text:
        blocks.append(
            BlockIR(
                block_id=_id("ocr", material.material_id, 1, ocr_text),
                material_id=material.material_id,
                page_id=page_id,
                block_type="table_text",
                text_raw=ocr_text,
                text_normalized=_WS.sub(" ", ocr_text),
                reading_order=0,
                bbox=Rect(x0=0, y0=0, x1=float(width), y1=float(height)),
                bbox_normalized=Rect(x0=0, y0=0, x1=1, y1=1),
                parser="ocr",
            )
        )
    table_id = _id("tblvis", material.material_id, 1)
    table = TableBundle(
        table_id=table_id,
        material_id=material.material_id,
        source_label=Path(material.path).stem,
        fragments=[
            TableFragmentIR(
                fragment_id=_id("frag", table_id, 0),
                material_id=material.material_id,
                page_id=page_id,
                bbox=Rect(x0=0, y0=0, x1=float(width), y1=float(height)),
                bbox_normalized=Rect(x0=0, y0=0, x1=1, y1=1),
                image_path=str(Path(material.path).resolve()),
                source="visual_candidate",
            )
        ],
        row_count=0,
        column_count=0,
        parse_status="visual_only",
        relevant_hint=True,
        detection_kind="visual_candidate",
        visual_required=True,
        unresolved_notes=["Image material requires visual table interpretation"],
    )
    issues = [] if ocr_text else ["Image material has no OCR text; visual extraction is required"]
    return [page], blocks, [table], issues


def _inventory_material_references(
    blocks: list[BlockIR],
    materials: list[Material],
) -> list[MaterialReference]:
    references: list[MaterialReference] = []
    has_supplied_supplement = any(item.role in {"supplement", "table_attachment"} for item in materials)
    seen: set[tuple[str, str]] = set()
    for block in blocks:
        for match in _SUPPLEMENT_REFERENCE.finditer(block.text_raw):
            phrase = _WS.sub(" ", match.group(0)).strip()
            marker = (block.block_id, phrase.casefold())
            if marker in seen:
                continue
            seen.add(marker)
            context = block.text_raw[max(0, match.start() - 180) : match.end() + 180]
            categories: list[str] = []
            if re.search(r"\b(?:DLT|dose[- ]limiting|toxic|safety)\b", context, re.I):
                categories.append("dlt_results")
            if re.search(r"\b(?:cohort|treated|evaluable|patient|enrolled)\b", context, re.I):
                categories.append("population_counts")
            if re.search(r"\b(?:dose|regimen|schedule|infusion)\b", context, re.I):
                categories.append("dose_and_regimen")
            references.append(
                MaterialReference(
                    reference_id=_id("matref", block.block_id, phrase),
                    raw_text=phrase,
                    locator_source_ids=[block.block_id],
                    reference_kind="appendix" if "appendix" in phrase.casefold() else "supplement",
                    availability="supplied" if has_supplied_supplement else "missing",
                    potentially_relevant_categories=list(dict.fromkeys(categories)),
                )
            )
    return references


def build_document_ir(
    *,
    document_id: str,
    materials: list[Material],
    page_image_dir: Path,
    table_image_dir: Path,
    render_pages: bool = True,
) -> DocumentIR:
    pages: list[PageIR] = []
    blocks: list[BlockIR] = []
    tables: list[TableBundle] = []
    coverage: list[CoverageRecord] = []
    identifiers: list[KeyValue] = []
    issues: list[str] = []
    title: str | None = None
    updated_materials: list[Material] = []
    for material in materials:
        current = material.model_copy(deep=True)
        try:
            if current.media_type in {"application/xml", "text/xml"}:
                found_title, found_ids, found_blocks, found_tables, found_issues = _parse_jats(current)
                identifiers.extend(found_ids)
                blocks.extend(found_blocks)
                tables.extend(found_tables)
                issues.extend(found_issues)
                title = found_title or title
                current.status = "parsed" if not found_issues else "partial"
            elif current.media_type == "application/pdf":
                found_title, found_pages, found_blocks, found_tables, found_issues = _parse_pdf(
                    current,
                    page_image_dir=page_image_dir,
                    table_image_dir=table_image_dir,
                    render_pages=render_pages,
                )
                pages.extend(found_pages)
                blocks.extend(found_blocks)
                tables.extend(found_tables)
                issues.extend(found_issues)
                title = title or found_title
                current.page_count = len(found_pages)
                current.status = "parsed" if not found_issues else "partial"
            elif current.media_type == "image":
                found_pages, found_blocks, found_tables, found_issues = _parse_image_material(current)
                pages.extend(found_pages)
                blocks.extend(found_blocks)
                tables.extend(found_tables)
                issues.extend(found_issues)
                current.page_count = 1
                current.status = "parsed" if not found_issues else "partial"
            else:
                current.status = "failed"
                current.error = "Unsupported material type; retained in manifest"
            coverage.append(
                CoverageRecord(
                    source_id=current.material_id,
                    source_kind="material",
                    parse_status=(
                        "complete" if current.status == "parsed"
                        else "failed" if current.status == "failed"
                        else "partial"
                    ),
                    potentially_relevant=True,
                    reason=current.error,
                )
            )
        except Exception as error:
            current.status = "failed"
            current.error = str(error)
            issues.append(f"{current.material_id}: {error}")
            coverage.append(
                CoverageRecord(
                    source_id=current.material_id,
                    source_kind="material",
                    parse_status="failed",
                    potentially_relevant=True,
                    reason=str(error),
                )
            )
        updated_materials.append(current)
    for page in pages:
        coverage.append(
            CoverageRecord(
                source_id=page.page_id,
                source_kind="page",
                parse_status=(
                    "complete"
                    if page.native_text_status == "readable" or page.ocr_status == "completed"
                    else "unread"
                    if page.native_text_status == "none" and page.ocr_status in {"not_run", "failed"}
                    else "partial"
                ),
                potentially_relevant=True,
            )
        )
    for table in tables:
        coverage.append(
            CoverageRecord(
                source_id=table.table_id,
                source_kind="table",
                parse_status="complete" if table.parse_status == "complete" else "partial",
                potentially_relevant=table.relevant_hint,
                reason="; ".join(table.unresolved_notes) or None,
            )
        )
        for fragment in table.fragments:
            coverage.append(
                CoverageRecord(
                    source_id=fragment.fragment_id,
                    source_kind="table_fragment",
                    parse_status=(
                        "complete"
                        if (
                            fragment.image_path and Path(fragment.image_path).is_file()
                        ) or (
                            table.parse_status == "complete"
                            and bool(table.cells)
                            and not table.visual_required
                        )
                        else "unread"
                    ),
                    potentially_relevant=table.relevant_hint,
                    reason=(
                        None
                        if fragment.image_path or (table.parse_status == "complete" and table.cells and not table.visual_required)
                        else "No readable table-region image was produced"
                    ),
                )
            )
    material_references = _inventory_material_references(blocks, updated_materials)
    for reference in material_references:
        dlt_relevant = "dlt_results" in reference.potentially_relevant_categories
        coverage.append(
            CoverageRecord(
                source_id=reference.reference_id,
                source_kind="material_reference",
                parse_status="complete" if reference.availability == "supplied" else "unread",
                included_in_extraction=reference.availability == "supplied",
                potentially_relevant=dlt_relevant,
                reason=(
                    None if reference.availability == "supplied"
                    else f"Referenced material was not supplied: {reference.raw_text}"
                ),
            )
        )
        if reference.availability != "supplied" and dlt_relevant:
            issues.append(
                f"DLT-relevant referenced material was not supplied: {reference.raw_text}"
            )
    unique_identifiers = list(dict.fromkeys((item.name, item.value) for item in identifiers))
    main_material_ids = [item.material_id for item in updated_materials if item.role == "main"]
    relations = []
    if len(main_material_ids) > 1:
        relations.append(KeyValue(name="equivalent_main_materials", value=",".join(main_material_ids)))
    return DocumentIR(
        document_id=document_id,
        title=title,
        identifiers=[KeyValue(name=name, value=value) for name, value in unique_identifiers],
        materials=updated_materials,
        pages=pages,
        blocks=blocks,
        tables=tables,
        coverage=coverage,
        material_references=material_references,
        relations=relations,
        issues=issues,
    )
