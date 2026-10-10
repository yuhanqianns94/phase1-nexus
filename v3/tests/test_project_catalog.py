from pathlib import Path

from phase1_project.catalog import build_catalog, write_catalog

from test_prediction_pipeline import _article


def test_existing_catalog_indexes_materials_without_crawling_or_extracting(tmp_path: Path) -> None:
    extraction_root = tmp_path / "results"
    paths = _article(extraction_root, "PMC123", rows=[(.1, 3, 0)])
    xml_dir, pdf_dir = tmp_path / "xml", tmp_path / "pdf" / "PMC123"
    xml_dir.mkdir(); pdf_dir.mkdir(parents=True)
    (xml_dir / "PMC123.xml").write_text("<article/>")
    (pdf_dir / "PMC123.pdf").write_bytes(b"%PDF-1.4\n")
    output = tmp_path / "library_index.json"
    catalog = write_catalog(output, extraction_roots=[extraction_root], xml_directories=[xml_dir], pdf_directories=[tmp_path / "pdf"])
    assert catalog["crawler_was_run"] is False
    assert catalog["model_extraction_was_run"] is False
    assert catalog["ready_extraction_count"] == 1
    assert catalog["articles"][0]["reported_extraction"] == str(paths.reported_extraction)
    assert catalog["articles"][0]["xml_path"].endswith("PMC123.xml")
    assert catalog["articles"][0]["pdf_path"].endswith("PMC123.pdf")
    assert output.is_file()
