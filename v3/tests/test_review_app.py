from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from phase1_extraction.artifacts import ArticlePaths, write_json
from phase1_extraction.friendly_review import (
    CURVE_MODULE_ID,
    CurvePointAnswer,
    add_missing_curve_point,
    load_or_create_friendly_review,
    mark_module_correct,
    save_friendly_review,
)
from phase1_extraction.review_app import _article_links
from phase1_extraction.models import (
    Cohort,
    DLTObservation,
    DoseExpression,
    PopulationCount,
    ReportedExtraction,
    Study,
)


def _write_extraction(tmp_path: Path, document_id: str) -> ArticlePaths:
    paths = ArticlePaths(tmp_path, document_id)
    paths.ensure()
    write_json(
        paths.reported_extraction,
        ReportedExtraction(
            document_id=document_id,
            studies=[Study(study_id="study", label="Example phase I study")],
            cohorts=[
                Cohort(
                    cohort_id="cohort_100",
                    series_id="series",
                    label="Drug A 100 mg",
                    actual_status="actual",
                    regimen_id="regimen",
                    dose=DoseExpression(
                        raw_expression="100 mg",
                        value_kind="scalar",
                        value=100,
                        unit_raw="mg",
                        unit_normalized="mg",
                    ),
                )
            ],
            population_counts=[
                PopulationCount(
                    count_id="n100",
                    study_id="study",
                    cohort_id="cohort_100",
                    role="treated",
                    reported_value=3,
                )
            ],
            dlt_observations=[
                DLTObservation(
                    observation_id="o100",
                    study_id="study",
                    result_scope_id="scope",
                    cohort_id="cohort_100",
                    count_kind="dlt_count",
                    reported_count=1,
                    denominator_count_id="n100",
                )
            ],
        ),
    )
    return paths


def _app(tmp_path: Path, monkeypatch) -> AppTest:
    app_path = Path(__file__).parents[1] / "phase1_extraction" / "review_app.py"
    monkeypatch.setenv("PHASE1_REVIEW_OUTPUT_ROOT", str(tmp_path))
    return AppTest.from_file(app_path, default_timeout=10).run()


def test_review_page_renders_ten_clinical_modules_without_writing_on_load(tmp_path: Path, monkeypatch) -> None:
    paths = _write_extraction(tmp_path, "PMC_UI")
    app = _app(tmp_path, monkeypatch)

    assert not app.exception
    assert app.title[0].value == "PHASE I NEXUS · Evidence Review"
    assert [header.value for header in app.header] == ["⬜ Disease and Study Population"]
    assert any(item.value == "Section 1 of 10" for item in app.caption)
    assert not any("Eligibility" in header.value for header in app.header)
    assert "**Study**\n\nExample phase I study" in [item.value for item in app.markdown]
    assert not paths.friendly_human_review.exists()
    assert not paths.human_review.exists()


def test_whole_curve_table_confirmation_scores_one_complete_point(tmp_path: Path, monkeypatch) -> None:
    paths = _write_extraction(tmp_path, "PMC_CURVE")
    app = _app(tmp_path, monkeypatch)
    next(box for box in app.selectbox if box.label == "Review section").set_value("curve_results").run()
    assert [header.value for header in app.header] == ["⬜ Dose-DLT Results"]
    next(button for button in app.button if button.label == "Confirm Entire Table as Correct").click().run()

    assert not app.exception
    saved = json.loads(paths.friendly_review_summary.read_text())
    assert saved["curve_points"]["counts"]["tp"] == 1
    assert saved["curve_points"]["counts"]["pending"] == 0
    assert saved["curve_points"]["recall_is_final"] is True
    assert saved["completed_modules"] == 1
    assert not paths.human_review.exists()


def test_one_click_confirms_a_complete_text_field(tmp_path: Path, monkeypatch) -> None:
    paths = _write_extraction(tmp_path, "PMC_TEXT")
    app = _app(tmp_path, monkeypatch)
    enabled = [button for button in app.button if button.label == "Confirm as Correct & Next" and not button.disabled]
    assert len(enabled) == 1
    enabled[0].click().run()

    assert not app.exception
    assert [header.value for header in app.header] == ["⬜ Eligibility"]
    assert any(item.value == "Section 2 of 10" for item in app.caption)
    saved = json.loads(paths.friendly_review_summary.read_text())
    assert saved["text_fields"]["counts"]["tp"] == 1
    assert saved["completed_modules"] == 1
    assert paths.friendly_review_units_csv.is_file()


def _click(app: AppTest, label: str) -> None:
    next(button for button in app.button if button.label == label and not button.disabled).click().run()
    assert not app.exception


def test_backtracking_replaces_saved_answer_and_undo_restores_scores(tmp_path: Path, monkeypatch) -> None:
    paths = _write_extraction(tmp_path, "PMC3309850")
    app = _app(tmp_path, monkeypatch)
    _click(app, "Confirm as Correct & Next")
    _click(app, "← Previous")
    assert any(caption.value == "Saved assessment: Correct and complete" for caption in app.caption)
    next(area for area in app.text_area if area.label == "Human correct answer").input("Adults with recurrent disease")
    _click(app, "Save Review")
    assert [header.value for header in app.header] == ["✅ Disease and Study Population"]
    assert "Adults with recurrent disease" in [item.value for item in app.markdown]
    summary = json.loads(paths.friendly_review_summary.read_text())
    assert (summary["text_fields"]["counts"]["tp"], summary["text_fields"]["counts"]["fp"], summary["text_fields"]["counts"]["fn"]) == (0, 1, 1)
    _click(app, "Next →")
    _click(app, "← Previous")
    assert app.text_area[0].value == "Adults with recurrent disease"
    _click(app, "Undo Last Save")
    assert any(caption.value == "Saved assessment: Correct and complete" for caption in app.caption)
    assert app.text_area[0].value == "**Study**\n\nExample phase I study"
    assert json.loads(paths.friendly_review_summary.read_text())["text_fields"]["counts"]["tp"] == 1
    assert next(button for button in app.button if button.label == "Undo Last Save").disabled
    assert any(item.proto.url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC3309850/" for item in app.get("link_button"))


def test_restart_clears_forms_missing_points_and_can_restore_previous_review(tmp_path: Path, monkeypatch) -> None:
    paths = _write_extraction(tmp_path, "PMC3309850")
    review = load_or_create_friendly_review(paths)
    review.reviewer = "AB"
    mark_module_correct(review, "population")
    mark_module_correct(review, CURVE_MODULE_ID)
    add_missing_curve_point(review, CurvePointAnswer(
        series_label="Additional series", dose_value=200, dose_unit="mg", dose_raw="200 mg", n=6, y=2,
    ))
    save_friendly_review(paths, review)
    original = paths.reported_extraction.read_bytes()
    app = _app(tmp_path, monkeypatch)
    next(box for box in app.selectbox if box.label == "Review section").set_value(CURVE_MODULE_ID).run()
    _click(app, "Restart Review for This Article")
    assert [header.value for header in app.header] == ["⬜ Disease and Study Population"]
    assert app.text_area[0].value == ""
    assert not app.checkbox[0].value
    assert json.loads(paths.friendly_review_summary.read_text())["completed_modules"] == 0
    assert paths.reported_extraction.read_bytes() == original
    saved = load_or_create_friendly_review(paths)
    assert not any(unit.origin == "manual_missing" for unit in saved.units)
    _click(app, "Undo Last Save")
    assert json.loads(paths.friendly_review_summary.read_text())["completed_modules"] == 2
    assert any(unit.origin == "manual_missing" for unit in load_or_create_friendly_review(paths).units)
    assert next(field for field in app.text_input if field.label == "Reviewer").value == "AB"

    # A deliberate restart followed by a fresh assessment replaces the old run.
    _click(app, "Restart Review for This Article")
    next(area for area in app.text_area if area.label == "Human correct answer").input("New review answer")
    _click(app, "Save Review")
    assert load_or_create_friendly_review(paths).units[0].gold_answer == "New review answer"
    assert not any(unit.origin == "manual_missing" for unit in load_or_create_friendly_review(paths).units)
    _click(app, "Undo Last Save")
    assert all(unit.decision == "pending" for unit in load_or_create_friendly_review(paths).units)


def test_switching_articles_or_output_directories_does_not_reuse_form_values(tmp_path: Path, monkeypatch) -> None:
    paths = _write_extraction(tmp_path, "PMC3309850")
    second = _write_extraction(tmp_path, "PMC3945293")
    other_root = tmp_path / "another_run"
    other = _write_extraction(other_root, "PMC3945293")
    app = _app(tmp_path, monkeypatch)
    next(area for area in app.text_area if area.label == "Human correct answer").input("First article correction")
    _click(app, "Save Review")
    next(box for box in app.selectbox if box.label == "Article").set_value(second.document_id).run()
    assert not app.exception
    assert app.text_area[0].value == ""
    next(area for area in app.text_area if area.label == "Human correct answer").input("Second article correction")
    _click(app, "Save Review")
    next(field for field in app.text_area if field.label == "Extraction output directories").input(str(other_root)).run()
    assert not app.exception
    assert app.text_area[0].value == ""
    assert not other.friendly_human_review.exists()
    _click(app, "Restart Review for This Article")
    assert load_or_create_friendly_review(paths).units[0].gold_answer == "First article correction"
    assert load_or_create_friendly_review(second).units[0].gold_answer == "Second article correction"


def test_review_page_combines_multiple_output_roots(tmp_path: Path, monkeypatch) -> None:
    _write_extraction(tmp_path / "regular", "PMC3309850")
    _write_extraction(tmp_path / "batch", "PMC3945293")
    app = _app(tmp_path / "regular", monkeypatch)
    next(field for field in app.text_area if field.label == "Extraction output directories").input(
        f"{tmp_path / 'regular'}\n{tmp_path / 'batch'}"
    ).run()

    assert not app.exception
    article = next(box for box in app.selectbox if box.label == "Article")
    assert article.options == ["PMC3309850", "PMC3945293"]
    article.set_value("PMC3945293").run()
    assert not app.exception
    assert any(item.proto.url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC3945293/" for item in app.get("link_button"))


def test_article_links_use_source_identifiers_including_batch_input_fallback(tmp_path: Path) -> None:
    paths = _write_extraction(tmp_path, "local_article")
    source_root = tmp_path / "prepared_article"
    write_json(paths.request_summary, {"input_article_root": str(source_root)})
    write_json(source_root / "document_ir.json", {"identifiers": [
        {"name": "pmc", "value": "3309850"},
        {"name": "doi", "value": "10.1234/example"},
        {"name": "pmid", "value": "123456"},
    ]})
    assert _article_links(paths) == [
        ("Open Article (PMC)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3309850/"),
        ("Publisher / DOI", "https://doi.org/10.1234/example"),
        ("PubMed", "https://pubmed.ncbi.nlm.nih.gov/123456/"),
    ]
    write_json(source_root / "document_ir.json", {"identifiers": [{"name": "pmc", "value": "not_an_id"}]})
    assert _article_links(paths) == []
