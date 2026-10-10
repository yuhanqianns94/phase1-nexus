from __future__ import annotations

from pathlib import Path

import pytest

from phase1_extraction.artifacts import ArticlePaths, read_json, write_json
from phase1_extraction.friendly_review import (
    CURVE_MODULE_ID,
    CurvePointAnswer,
    add_missing_curve_point,
    load_friendly_review_undo,
    load_or_create_friendly_review,
    mark_module_correct,
    restart_friendly_review,
    save_friendly_review,
    summarize_friendly_review,
    undo_friendly_review,
)
from phase1_extraction.models import (
    Cohort,
    DLTObservation,
    DoseExpression,
    PopulationCount,
    ReportedExtraction,
    Study,
)


def _paths(tmp_path: Path, *, treated_n: int = 3, dlt_n: int = 1) -> ArticlePaths:
    paths = ArticlePaths(tmp_path, "PMC_FRIENDLY")
    paths.ensure()
    write_json(
        paths.reported_extraction,
        ReportedExtraction(
            document_id="PMC_FRIENDLY",
            studies=[Study(study_id="study", label="A phase I study")],
            cohorts=[
                Cohort(
                    cohort_id="c1",
                    series_id="s1",
                    label="Drug 100 mg",
                    actual_status="actual",
                    regimen_id="r1",
                    dose=DoseExpression(raw_expression="100 mg", value_kind="scalar", value=100, unit_raw="mg", unit_normalized="mg"),
                )
            ],
            population_counts=[
                PopulationCount(count_id="n1", study_id="study", cohort_id="c1", role="treated", reported_value=treated_n)
            ],
            dlt_observations=[
                DLTObservation(
                    observation_id="o1",
                    study_id="study",
                    result_scope_id="scope",
                    cohort_id="c1",
                    count_kind="dlt_count",
                    reported_count=dlt_n,
                    denominator_count_id="n1",
                )
            ],
        ),
    )
    return paths


def test_review_has_nine_text_fields_and_one_curve_point(tmp_path: Path) -> None:
    review = load_or_create_friendly_review(_paths(tmp_path))
    assert sum(unit.unit_kind == "text_field" for unit in review.units) == 9
    assert sum(unit.unit_kind == "curve_point" for unit in review.units) == 1
    curve = next(unit for unit in review.units if unit.unit_kind == "curve_point")
    assert (curve.predicted_point.dose_value, curve.predicted_point.n, curve.predicted_point.y) == (100, 3, 1)
    population = next(unit for unit in review.units if unit.module_id == "population")
    assert population.module_label == "Disease and Study Population"
    assert population.predicted_answer == "**Study**\n\nA phase I study"


def test_text_and_curve_scores_are_separate_and_f1_is_zero_when_all_wrong(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    review = load_or_create_friendly_review(paths)
    population = next(unit for unit in review.units if unit.module_id == "population")
    population.decision = "incorrect"
    population.gold_answer = "Correct population"
    review.module_complete["population"] = True
    curve = next(unit for unit in review.units if unit.unit_kind == "curve_point")
    curve.decision = "spurious"
    review.module_complete[CURVE_MODULE_ID] = True

    summary = summarize_friendly_review(review)
    assert (summary.text_fields.counts.fp, summary.text_fields.counts.fn) == (1, 1)
    assert summary.text_fields.counts.f1 == 0
    assert (summary.curve_points.counts.fp, summary.curve_points.counts.fn) == (1, 0)
    assert summary.curve_points.counts.f1 == 0


def test_curve_whole_table_and_missing_point_produce_auditable_prf(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    review = load_or_create_friendly_review(paths)
    mark_module_correct(review, CURVE_MODULE_ID)
    add_missing_curve_point(
        review,
        CurvePointAnswer(
            series_label="Drug series",
            dose_value=200,
            dose_unit="mg",
            dose_raw="200 mg",
            n=3,
            y=2,
            denominator_type="treated",
        ),
    )
    summary = save_friendly_review(paths, review)

    assert (summary.curve_points.counts.tp, summary.curve_points.counts.fn) == (1, 1)
    assert summary.curve_points.recall_is_final
    assert paths.friendly_human_review.is_file()
    assert not paths.human_review.exists()


def test_changed_curve_value_resets_only_curve_module(tmp_path: Path) -> None:
    paths = _paths(tmp_path, treated_n=3)
    review = load_or_create_friendly_review(paths)
    mark_module_correct(review, "population")
    mark_module_correct(review, CURVE_MODULE_ID)
    save_friendly_review(paths, review)

    _paths(tmp_path, treated_n=6)
    synced = load_or_create_friendly_review(paths)
    population = next(unit for unit in synced.units if unit.module_id == "population")
    curve = next(unit for unit in synced.units if unit.unit_kind == "curve_point")
    assert population.decision == "correct"
    assert synced.module_complete["population"] is True
    assert curve.decision == "pending"
    assert synced.module_complete[CURVE_MODULE_ID] is False


def test_restart_overwrites_all_review_artifacts_and_can_be_undone(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    extraction_before = paths.reported_extraction.read_bytes()
    other_paths = _paths(tmp_path / "other_run")
    other_review = load_or_create_friendly_review(other_paths)
    mark_module_correct(other_review, "population")
    save_friendly_review(other_paths, other_review)
    other_before = other_paths.friendly_human_review.read_bytes()
    review = load_or_create_friendly_review(paths)
    review.reviewer = "AB"
    population = next(unit for unit in review.units if unit.module_id == "population")
    population.decision = "incorrect"
    population.gold_answer = "Corrected population"
    population.note = "Page 2"
    review.module_complete["population"] = True
    mark_module_correct(review, CURVE_MODULE_ID)
    add_missing_curve_point(review, CurvePointAnswer(
        series_label="Additional series", dose_value=200, dose_unit="mg", dose_raw="200 mg", n=6, y=2,
    ))
    old_summary = save_friendly_review(paths, review)
    old_csv = paths.friendly_review_units_csv.read_bytes()

    restarted = restart_friendly_review(paths)

    assert restarted.reviewer == "AB"
    assert all(unit.decision == "pending" for unit in restarted.units)
    assert all(unit.gold_answer is None and unit.gold_point is None and unit.note is None for unit in restarted.units)
    assert not any(unit.origin == "manual_missing" for unit in restarted.units)
    assert not any(restarted.module_complete.values())
    assert paths.reported_extraction.read_bytes() == extraction_before
    assert other_paths.friendly_human_review.read_bytes() == other_before
    assert read_json(paths.friendly_review_summary)["completed_modules"] == 0
    assert read_json(paths.friendly_review_summary)["curve_points"]["counts"]["fn"] == 0
    assert "Corrected population" not in paths.friendly_review_units_csv.read_text()
    assert load_friendly_review_undo(paths) is not None

    restored, module_id = undo_friendly_review(paths)
    assert module_id == "population"
    assert restored.reviewer == "AB"
    assert next(unit for unit in restored.units if unit.module_id == "population").gold_answer == "Corrected population"
    assert paths.friendly_review_units_csv.read_bytes() == old_csv
    assert summarize_friendly_review(restored).curve_points == old_summary.curve_points
    assert load_friendly_review_undo(paths) is None
    with pytest.raises(ValueError, match="No matching saved review"):
        undo_friendly_review(paths)


def test_undo_survives_reload_and_rejects_changed_extraction_or_review(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    review = load_or_create_friendly_review(paths)
    mark_module_correct(review, "population")
    save_friendly_review(paths, review)
    assert load_friendly_review_undo(paths).module_id == "population"
    # A no-op save must not replace the useful undo state.
    save_friendly_review(paths, load_or_create_friendly_review(paths))
    restored, _ = undo_friendly_review(paths)
    assert all(unit.decision == "pending" for unit in restored.units)
    assert read_json(paths.friendly_review_summary)["text_fields"]["counts"]["tp"] == 0

    mark_module_correct(restored, "population")
    save_friendly_review(paths, restored)
    _paths(tmp_path, treated_n=6)
    assert load_friendly_review_undo(paths) is None
    with pytest.raises(ValueError, match="No matching saved review"):
        undo_friendly_review(paths)

    # A newer external review must not be overwritten by an older undo record.
    current = load_or_create_friendly_review(paths)
    current.reviewer = "Second session"
    write_json(paths.friendly_human_review, current)
    assert load_friendly_review_undo(paths) is None
