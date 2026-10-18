from __future__ import annotations

import pytest

from phase1_extraction.friendly_review import _build_text_units
from phase1_extraction.models import DLTObservation, EvidenceLink, ReportedExtraction, ValueProvenance


def _extraction(status: str, *, count: int | None = None) -> ReportedExtraction:
    return ReportedExtraction(
        document_id="PMC_DLT_LABEL",
        dlt_observations=[
            DLTObservation(
                observation_id="observation",
                study_id="study",
                result_scope_id="scope",
                count_kind="named_toxicities_only",
                reporting_status=status,
                reported_count=count,
                toxicities_raw=["Grade 3 abdominal pain after a second-cycle procedural complication"],
                provenance=ValueProvenance(
                    note="The event was excluded from DLT assessment on procedural grounds."
                ),
                evidence=[
                    EvidenceLink(
                        source_ids=["source_safety"],
                        source_text="The grade 3 event was excluded from DLT assessment.",
                    )
                ],
            )
        ],
    )


def test_non_dlt_event_retains_its_facts_and_evidence_without_an_observed_dlt_label() -> None:
    extraction = _extraction("not_applicable")
    original = extraction.model_dump(mode="json")

    unit = next(unit for unit in _build_text_units(extraction) if unit.module_id == "dlt_definition")

    assert "Toxicity notes — DLT not applicable" in unit.predicted_answer
    assert "Observed DLTs" not in unit.predicted_answer
    assert "Grade 3 abdominal pain" in unit.predicted_answer
    assert "excluded from DLT assessment on procedural grounds" in unit.predicted_answer
    assert unit.source_record_ids == ["observation"]
    assert unit.evidence_source_ids == ["source_safety"]
    assert "The grade 3 event was excluded from DLT assessment." in unit.evidence_text
    assert extraction.model_dump(mode="json") == original


@pytest.mark.parametrize(
    ("status", "expected_label"),
    [
        ("not_reported", "DLT result not reported"),
        ("explicitly_unreported", "DLT result explicitly unreported"),
        ("unclear", "DLT status unclear"),
        ("explicitly_not_evaluable", "DLT not evaluable"),
        ("conflicting", "conflicting DLT status"),
    ],
)
def test_unconfirmed_observations_do_not_appear_as_observed_dlts(status: str, expected_label: str) -> None:
    unit = next(unit for unit in _build_text_units(_extraction(status)) if unit.module_id == "dlt_definition")
    assert expected_label in unit.predicted_answer
    assert "Observed DLTs" not in unit.predicted_answer


def test_reported_dlt_label_remains_available_for_confirmed_observations() -> None:
    extraction = _extraction("reported", count=1)
    extraction.dlt_observations[0].toxicities_raw = ["Grade 3 neutropenia"]
    extraction.dlt_observations[0].provenance.note = None

    unit = next(unit for unit in _build_text_units(extraction) if unit.module_id == "dlt_definition")

    assert "**Observed DLTs (Cohort not identified)**" in unit.predicted_answer
    assert "Grade 3 neutropenia" in unit.predicted_answer
    assert "Toxicity notes" not in unit.predicted_answer
