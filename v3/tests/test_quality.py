from phase1_extraction.models import (
    Arm,
    Cohort,
    DoseExpression,
    DoseSeries,
    ExtractionCoverage,
    Regimen,
    ReportedExtraction,
    ResultScope,
    Study,
)
from phase1_extraction.quality import (
    EXPECTED_COVERAGE_CATEGORIES,
    apply_audit_status,
    audit_reported_extraction,
)


def _base() -> ReportedExtraction:
    extraction = ReportedExtraction(
        document_id="DOC",
        studies=[Study(study_id="s", label="study")],
        arms=[Arm(arm_id="a", study_id="s", label="arm", regimen_id="r")],
        regimens=[Regimen(regimen_id="r", label="regimen")],
        dose_series=[
            DoseSeries(
                series_id="ser",
                study_id="s",
                arm_id="a",
                label="series",
                regimen_id="r",
                varying_component_id="component",
            )
        ],
        cohorts=[
            Cohort(
                cohort_id="c",
                series_id="ser",
                label="dose",
                actual_status="actual",
                regimen_id="r",
                dose=DoseExpression(raw_expression="1 mg", value_kind="scalar", value=1, unit_raw="mg"),
            )
        ],
        result_scopes=[
            ResultScope(
                result_scope_id="scope",
                study_id="s",
                series_id="ser",
                cohort_ids=["c"],
                dlt_results_coverage="complete",
            )
        ],
        extraction_coverage=[
            ExtractionCoverage(category=category, status="not_reported")
            for category in EXPECTED_COVERAGE_CATEGORIES
        ],
        material_coverage_complete=True,
        extraction_complete=True,
    )
    extraction.extraction_coverage[-1].status = "extracted"
    return extraction


def test_audit_accepts_complete_referential_structure() -> None:
    audit = audit_reported_extraction(_base())
    assert audit["blocking_count"] == 0
    assert audit["targeted_reread_count"] == 0


def test_audit_blocks_unknown_scope_series_and_invalidates_complete_flag() -> None:
    extraction = _base()
    extraction.result_scopes[0].series_id = "missing"
    audit = audit_reported_extraction(extraction)
    assert any(
        item["code"] == "unknown_entity_reference" and "series_id" in item["message"]
        for item in audit["findings"]
    )
    apply_audit_status(extraction, audit)
    assert extraction.extraction_complete is False


def test_audit_requests_targeted_reread_for_unscoped_actual_cohort() -> None:
    extraction = _base()
    extraction.result_scopes.clear()
    audit = audit_reported_extraction(extraction)
    assert audit["requires_targeted_reread"] is True
    assert any(item["code"] == "actual_cohorts_without_result_scope" for item in audit["findings"])
