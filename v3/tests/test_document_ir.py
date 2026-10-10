from __future__ import annotations

from pathlib import Path

import fitz

from phase1_extraction.artifacts import register_materials
from phase1_extraction.document_ir import build_document_ir
from phase1_extraction.packets import build_extraction_packets, packet_image_inventory, packet_image_paths


def test_jats_table_preserves_spans_headers_and_footnotes(tmp_path: Path) -> None:
    xml = tmp_path / "article.xml"
    xml.write_text(
        """<?xml version="1.0"?>
<article><front><article-meta><article-id pub-id-type="pmcid">PMC1</article-id>
<title-group><article-title>Example phase I study</article-title></title-group>
<abstract><p>Three dose cohorts were studied.</p></abstract></article-meta></front>
<body><sec><title>Results</title><p>Dose-limiting toxicity results are shown in Table 1.</p>
<table-wrap id="t1"><label>Table 1</label><caption><p>Dose cohorts and DLTs</p></caption>
<table><thead><tr><th rowspan="2">Dose</th><th colspan="2">Patients</th></tr>
<tr><th>Treated</th><th>DLT</th></tr></thead><tbody>
<tr><th>50 mg</th><td>3</td><td>0</td></tr><tr><th>100 mg</th><td>6</td><td>1</td></tr>
</tbody></table><table-wrap-foot><fn><p>* Cycle 1 only.</p></fn></table-wrap-foot></table-wrap>
</sec></body></article>""",
        encoding="utf-8",
    )
    materials = register_materials(document_id="PMC1", source_paths=[xml])
    ir = build_document_ir(
        document_id="PMC1",
        materials=materials,
        page_image_dir=tmp_path / "pages",
        table_image_dir=tmp_path / "tables",
    )
    assert ir.title == "Example phase I study"
    assert len(ir.tables) == 1
    table = ir.tables[0]
    assert table.row_count == 4
    assert table.column_count == 3
    assert any(cell.row_span == 2 for cell in table.cells)
    dlt_cell = next(cell for cell in table.cells if cell.text_raw == "1")
    assert "Patients" in dlt_cell.column_header_path
    assert "DLT" in dlt_cell.column_header_path
    assert "100 mg" in dlt_cell.row_header_path
    assert table.footnotes == ["* Cycle 1 only."]


def test_pdf_line_table_creates_grid_and_crop(tmp_path: Path) -> None:
    pdf = tmp_path / "table.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=300)
    for x in (50, 150, 250, 350):
        page.draw_line((x, 80), (x, 180))
    for y in (80, 130, 180):
        page.draw_line((50, y), (350, y))
    page.insert_text((60, 110), "Dose")
    page.insert_text((160, 110), "Treated")
    page.insert_text((260, 110), "DLT")
    page.insert_text((60, 160), "100 mg")
    page.insert_text((160, 160), "6")
    page.insert_text((260, 160), "1")
    document.save(pdf)
    document.close()
    materials = register_materials(document_id="PDF1", source_paths=[pdf])
    ir = build_document_ir(
        document_id="PDF1",
        materials=materials,
        page_image_dir=tmp_path / "pages",
        table_image_dir=tmp_path / "tables",
    )
    parsed = [table for table in ir.tables if table.parse_status == "complete"]
    assert parsed
    assert parsed[0].row_count == 2
    assert parsed[0].column_count == 3
    assert Path(parsed[0].fragments[0].image_path).is_file()


def test_pdf_empty_top_left_header_does_not_shift_later_columns(tmp_path: Path) -> None:
    pdf = tmp_path / "blank-header.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=300)
    for x in (50, 150, 250, 350):
        page.draw_line((x, 80), (x, 180))
    for y in (80, 130, 180):
        page.draw_line((50, y), (350, y))
    page.insert_text((160, 110), "Treated")
    page.insert_text((260, 110), "DLT")
    page.insert_text((60, 160), "100 mg")
    page.insert_text((160, 160), "6")
    page.insert_text((260, 160), "1")
    document.save(pdf)
    document.close()
    ir = build_document_ir(
        document_id="PDF2",
        materials=register_materials(document_id="PDF2", source_paths=[pdf]),
        page_image_dir=tmp_path / "pages",
        table_image_dir=tmp_path / "tables",
    )
    table = next(item for item in ir.tables if item.cells)
    treated = next(item for item in table.cells if item.row_index == 1 and item.column_index == 1)
    dlt = next(item for item in table.cells if item.row_index == 1 and item.column_index == 2)
    assert treated.column_header_path == ["Treated"]
    assert dlt.column_header_path == ["DLT"]


def test_jats_nested_table_group_and_list_are_not_skipped(tmp_path: Path) -> None:
    xml = tmp_path / "nested.xml"
    xml.write_text(
        """<article><body><sec><title>Results</title>
        <list><list-item><p>Three patients were treated.</p></list-item></list>
        <table-wrap-group><table-wrap id="t1"><label>Table 1</label>
        <table><tr><th>Dose</th><th>DLT</th></tr><tr><td>10 mg</td><td>0</td></tr></table>
        </table-wrap></table-wrap-group></sec></body></article>""",
        encoding="utf-8",
    )
    ir = build_document_ir(
        document_id="NESTED",
        materials=register_materials(document_id="NESTED", source_paths=[xml]),
        page_image_dir=tmp_path / "pages",
        table_image_dir=tmp_path / "tables",
    )
    assert any("Three patients" in block.text_raw for block in ir.blocks)
    assert len(ir.tables) == 1
    assert ir.tables[0].parse_status == "complete"
    assert ir.tables[0].visual_required is False


def test_native_text_pdf_still_sends_full_page_visual(tmp_path: Path) -> None:
    pdf = tmp_path / "mixed.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=300)
    page.insert_textbox((30, 20, 370, 120), "Clinical trial results " * 20, fontsize=9)
    # An embedded image can contain a table even when native prose makes the
    # page look text-readable to the parser.
    image_pdf = fitz.open()
    image_page = image_pdf.new_page(width=80, height=40)
    image_page.draw_rect((1, 1, 79, 39), color=(0, 0, 0))
    pix = image_page.get_pixmap()
    page.insert_image((100, 160, 300, 260), stream=pix.tobytes("png"))
    document.save(pdf)
    document.close()
    image_pdf.close()
    ir = build_document_ir(
        document_id="MIXED",
        materials=register_materials(document_id="MIXED", source_paths=[pdf]),
        page_image_dir=tmp_path / "pages",
        table_image_dir=tmp_path / "tables",
    )
    assert ir.pages[0].native_text_status == "readable"
    packet = build_extraction_packets(ir)[0]
    assert any(unit["unit_kind"] == "page_visual" for unit in packet["source_units"])
    assert len(packet_image_paths(packet, ir)) >= 1
    inventory = packet_image_inventory(packet, ir)
    assert any(item["source_id"] == ir.pages[0].page_id and item["status"] == "included" for item in inventory)
