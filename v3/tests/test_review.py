from __future__ import annotations

import csv
import io

from phase1_extraction.models import (
    Cohort,
    DLTObservation,
    DoseExpression,
    PopulationCount,
    ReportedExtraction,
)
from phase1_extraction.review import (
    add_missing_unit,
    build_review_units,
    corpus_review_csv,
    new_review,
    review_csv,
    summarize_corpus,
    summarize_review,
    sync_review,
)


def extraction(*, treated_n: int = 6) -> ReportedExtraction:
    dose = DoseExpression(
        raw_expression="100 mg/m2",
        value_kind="scalar",
        value=100,
        unit_raw="mg/m2",
        unit_normalized="mg/m2",
    )
    return ReportedExtraction(
        document_id="PMC_TEST",
        cohorts=[
            Cohort(
                cohort_id="cohort_100",
                series_id="series",
                label="100 mg/m2",
                actual_status="actual",
                regimen_id="regimen",
                dose=dose,
            )
        ],
        population_counts=[
            PopulationCount(
                count_id="treated_100",
                study_id="study",
                cohort_id="cohort_100",
                role="treated",
                reported_value=treated_n,
            ),
            PopulationCount(
                count_id="evaluable_100",
                study_id="study",
                cohort_id="cohort_100",
                role="dlt_evaluable",
                reported_value=5,
            ),
        ],
        dlt_observations=[
            DLTObservation(
                observation_id="dlt_100",
                study_id="study",
                result_scope_id="scope",
                cohort_id="cohort_100",
                count_kind="patients",
                reported_count=2,
            )
        ],
    )


def test_each_count_and_dose_is_a_separate_review_answer() -> None:
    units = build_review_units(extraction())
    answers = {(unit.metric_name, unit.predicted_answer) for unit in units}
    assert ("cohort.dose", "100 mg/m2") in answers
    assert ("population_count.treated", "6") in answers
    assert ("population_count.dlt_evaluable", "5") in answers
    assert ("dlt.count.patients", "2") in answers


def test_precision_recall_policy() -> None:
    review = new_review(extraction(), "hash")
    for unit in review.units:
        unit.decision = "excluded"
    scored = review.units[:3]
    scored[0].decision = "correct"
    scored[0].gold_answer = scored[0].predicted_answer
    scored[1].decision = "incorrect"
    scored[1].gold_answer = "corrected"
    scored[2].decision = "spurious"
    add_missing_unit(
        review,
        metric_group="DLT Results",
        metric_name="dlt.toxicity",
        metric_label="DLT manifestation",
        scope="100 mg/m2",
        gold_answer="grade 3 fever",
    )
    summary = summarize_review(review)
    assert (summary.overall.tp, summary.overall.fp, summary.overall.fn) == (1, 2, 2)
    assert summary.overall.precision == 1 / 3
    assert summary.overall.recall == 1 / 3
    assert summary.overall.f1 == 1 / 3
    assert not summary.recall_is_final
    review.source_review_complete = True
    assert summarize_review(review).recall_is_final


def test_changed_prediction_resets_only_changed_unit() -> None:
    original = extraction(treated_n=6)
    review = new_review(original, "old")
    review.source_review_complete = True
    treated = next(unit for unit in review.units if unit.metric_name == "population_count.treated")
    treated.decision = "correct"
    treated.gold_answer = "6"
    dlt = next(unit for unit in review.units if unit.metric_name == "dlt.count.patients")
    dlt.decision = "correct"
    dlt.gold_answer = "2"

    synced = sync_review(review, extraction(treated_n=7), "new")
    changed = next(unit for unit in synced.units if unit.metric_name == "population_count.treated")
    unchanged = next(unit for unit in synced.units if unit.metric_name == "dlt.count.patients")
    assert (changed.predicted_answer, changed.decision, changed.gold_answer) == ("7", "pending", None)
    assert (unchanged.decision, unchanged.gold_answer) == ("correct", "2")
    assert not synced.source_review_complete


def test_csv_has_direct_tp_fp_fn_columns() -> None:
    review = new_review(extraction(), "hash")
    first = review.units[0]
    first.decision = "correct"
    first.gold_answer = first.predicted_answer
    rows = list(csv.DictReader(io.StringIO(review_csv(review))))
    assert rows[0]["tp"] == "1"
    assert rows[0]["fp"] == "0"
    assert rows[0]["fn"] == "0"


def test_corpus_summary_pools_articles_and_exports_document_id() -> None:
    first = new_review(extraction(), "hash-1")
    second = new_review(extraction(treated_n=7), "hash-2")
    for review in (first, second):
        for unit in review.units:
            unit.decision = "correct"
            unit.gold_answer = unit.predicted_answer
        review.source_review_complete = True

    summary = summarize_corpus([first, second])
    assert summary.article_count == 2
    assert summary.overall.fp == summary.overall.fn == 0
    assert summary.overall.tp == len(first.units) + len(second.units)
    assert summary.recall_is_final_for_saved_reviews
    rows = list(csv.DictReader(io.StringIO(corpus_review_csv([first, second]))))
    assert len(rows) == summary.overall.tp
    assert {row["document_id"] for row in rows} == {"PMC_TEST"}
