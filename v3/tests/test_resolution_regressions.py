"""Patient/event totals and nested scopes from the two-paper Batch pilot."""
import pytest

from phase1_extraction.models import DLTObservation, PopulationCount
from phase1_extraction.quality import _audit_resolved_rows
from phase1_extraction.resolver import resolve_extraction
from test_resolver import base_extraction


def reconciled_extraction():
    e = base_extraction(doses=(850, 1000, 2000), design=True)
    e.population_counts = [
        PopulationCount(count_id=f"n{i}", study_id="study", cohort_id=f"c{i}",
                        role="treated", reported_value=n)
        for i, n in enumerate((3, 6, 5), 1)
    ]
    e.dlt_observations = [
        DLTObservation(observation_id="total", study_id="study", result_scope_id="scope",
                       count_kind="dlt_count", reported_count=4, explicit_patient_count=3,
                       same_patient_repeated_events=True),
        DLTObservation(observation_id="one", study_id="study", result_scope_id="scope", cohort_id="c2",
                       count_kind="patients", reported_count=1, explicit_patient_count=1,
                       same_patient_repeated_events=False),
        DLTObservation(observation_id="two", study_id="study", result_scope_id="scope", cohort_id="c3",
                       count_kind="dlt_count", reported_count=3, explicit_patient_count=2,
                       same_patient_repeated_events=True),
    ]
    return e


def nested_extraction():
    e = reconciled_extraction()
    cohort = e.cohorts[0].model_copy(deep=True, update={
        "cohort_id": "expansion", "label": "1800 aggregate", "cohort_role": "expansion",
        "expansion_status": "completed_expansion", "result_scope_ids": ["scope", "nested"],
    })
    cohort.dose.value = 1800
    e.cohorts.append(cohort)
    e.result_scopes[0].cohort_ids.append("expansion")
    e.result_scopes.append(e.result_scopes[0].model_copy(deep=True, update={
        "result_scope_id": "nested", "cohort_ids": ["expansion"], "explicit_no_dlt_in_scope": True,
    }))
    e.population_counts.append(PopulationCount(
        count_id="n8", study_id="study", cohort_id="expansion", result_scope_id="nested",
        role="treated", reported_value=8,
    ))
    e.dlt_observations.append(DLTObservation(
        observation_id="zero", study_id="study", cohort_id="expansion", result_scope_id="nested",
        count_kind="patients", reported_count=0, explicit_patient_count=0, denominator_count_id="n8",
    ))
    return e


def test_patient_and_event_totals_reconcile_without_changing_source_facts():
    e = reconciled_extraction()
    before = e.model_dump()
    resolved = resolve_extraction(e)
    assert [(r.y, r.n) for r in resolved.curve_rows] == [(0, 3), (1, 6), (2, 5)]
    assert resolved.curve_rows[0].rule_ids == ["R2"]
    assert e.model_dump() == before
    assert _audit_resolved_rows(e, resolved) == []
    assert "total" in next(a for a in resolved.rule_applications if a.rule_id == "R2").input_record_ids


@pytest.mark.parametrize("change", ["patients", "events", "duplicate", "window", "study", "unknown_patients", "conflict", "outside_cohort"])
def test_unproven_allocation_still_blocks_zero(change):
    e = reconciled_extraction()
    if change == "patients":
        e.dlt_observations[0].explicit_patient_count = 4
    elif change == "events":
        e.dlt_observations[0].reported_count = 5
    elif change == "duplicate":
        e.dlt_observations.append(e.dlt_observations[1].model_copy(update={"observation_id": "duplicate"}))
    elif change == "window":
        e.dlt_observations[2].window_id = "other_cycle"
    elif change == "study":
        e.dlt_observations[2].study_id = "other_study"
    elif change == "unknown_patients":
        e.dlt_observations[2].explicit_patient_count = None
    elif change == "outside_cohort":
        e.dlt_observations[2].cohort_id = "unlisted"
    else:
        e.dlt_observations[2].reporting_status = "conflicting"
    row = resolve_extraction(e).curve_rows[0]
    assert row.y is None and "R2" not in row.rule_ids


def test_nested_report_projects_one_eight_patient_expansion_row():
    e = nested_extraction()
    before = e.model_dump()
    resolved = resolve_extraction(e)
    rows = [r for r in resolved.curve_rows if r.cohort_id == "expansion"]
    assert len(rows) == 1
    assert (rows[0].n, rows[0].y, rows[0].result_scope_id) == (8, 0, "nested")
    assert not any(r.n == 6 and r.cohort_id == "expansion" for r in resolved.curve_rows)
    assert any(a.target_field == "result_scope_id" and a.output_value == "nested" for a in resolved.rule_applications)
    assert _audit_resolved_rows(e, resolved) == []
    assert e.model_dump() == before


@pytest.mark.parametrize("field", ["window_id", "data_cutoff_raw", "part_id", "arm_id"])
def test_distinct_contexts_are_not_collapsed(field):
    e = nested_extraction()
    setattr(e.result_scopes[1], field, "different")
    rows = [r for r in resolve_extraction(e).curve_rows if r.cohort_id == "expansion"]
    assert len(rows) == 2
    assert rows[0].n is None  # An expansion population is not automatically six.


def test_broad_reported_conflict_is_preserved_and_audited():
    e = nested_extraction()
    e.dlt_observations.append(DLTObservation(
        observation_id="broad-positive", study_id="study", result_scope_id="scope", cohort_id="expansion",
        count_kind="patients", reported_count=1,
    ))
    resolved = resolve_extraction(e)
    assert len([r for r in resolved.curve_rows if r.cohort_id == "expansion"]) == 2
    assert "overlapping_cohort_projections" in {f["code"] for f in _audit_resolved_rows(e, resolved)}


def test_post_resolution_audit_catches_old_phantom_n_and_missing_zero():
    e = nested_extraction()
    resolved = resolve_extraction(e)
    zero = resolved.curve_rows[0]
    zero.y = None
    expansion = next(r for r in resolved.curve_rows if r.cohort_id == "expansion")
    resolved.curve_rows.append(expansion.model_copy(update={
        "curve_row_id": "bad", "result_scope_id": "scope", "n": 6,
        "denominator_basis": "design_inferred", "y": None,
    }))
    codes = {f["code"] for f in _audit_resolved_rows(e, resolved)}
    assert {"eligible_zero_not_resolved", "overlapping_cohort_projections",
            "design_denominator_conflicts_with_reported_count"} <= codes
