from __future__ import annotations

from phase1_extraction.models import (
    Cohort,
    CountEquation,
    DoseExpression,
    DoseSeries,
    DLTObservation,
    ExtractionCoverage,
    PopulationCount,
    ReportedExtraction,
    ResultScope,
    TrialDesign,
    ValueProvenance,
)
from phase1_extraction.resolver import resolve_extraction


def dose(value: float) -> DoseExpression:
    return DoseExpression(
        raw_expression=f"{value:g} mg",
        value_kind="scalar",
        value=value,
        unit_raw="mg",
        unit_normalized="mg",
    )


def base_extraction(
    *,
    doses: tuple[float, ...] = (50, 100, 150),
    design: bool = False,
    explicit_no_dlt: bool = False,
) -> ReportedExtraction:
    cohorts = [
        Cohort(
            cohort_id=f"c{i}",
            series_id="series",
            label=f"DL{i}",
            actual_status="actual",
            cohort_role="escalation",
            regimen_id="regimen",
            dose=dose(value),
            result_scope_ids=["scope"],
        )
        for i, value in enumerate(doses, start=1)
    ]
    return ReportedExtraction(
        document_id="DOC",
        material_coverage_complete=True,
        dose_series=[
            DoseSeries(
                series_id="series",
                study_id="study",
                label="dose escalation",
                regimen_id="regimen",
                varying_component_id="drug",
                design_id="design" if design else None,
                comparable_unit="mg",
            )
        ],
        cohorts=cohorts,
        designs=(
            [
                TrialDesign(
                    design_id="design",
                    study_id="study",
                    raw_name="standard 3+3",
                    normalized_type="classic_3plus3",
                )
            ]
            if design
            else []
        ),
        result_scopes=[
            ResultScope(
                result_scope_id="scope",
                study_id="study",
                series_id="series",
                cohort_ids=[item.cohort_id for item in cohorts],
                dlt_results_coverage="complete",
                has_observed_dlt_results=not explicit_no_dlt,
                explicit_no_dlt_in_scope=explicit_no_dlt,
            )
        ],
    )


def test_r1_and_r2_form_default_curve() -> None:
    extraction = base_extraction()
    extraction.population_counts = [
        PopulationCount(count_id=f"n{i}", study_id="study", cohort_id=f"c{i}", role="treated", reported_value=n)
        for i, n in enumerate((3, 6, 6), start=1)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c2",
            count_kind="dlt_count",
            reported_count=2,
        )
    ]
    result = resolve_extraction(extraction)
    assert [(row.y, row.n) for row in result.curve_rows] == [(0, 3), (2, 6), (0, 6)]
    assert result.curve_rows[1].numerator_basis == "inferred_patient_count"
    assert set(result.curve_rows[0].rule_ids) == {"R2"}
    assert set(result.curve_rows[1].rule_ids) == {"R1"}
    assert all(row.included_in_default for row in result.curve_rows)
    assert not any(row.included_in_reported_only for row in result.curve_rows)


def test_explicit_zero_dlt_count_is_reported_patient_zero_without_r1() -> None:
    extraction = base_extraction(doses=(100,))
    extraction.population_counts = [
        PopulationCount(count_id="n", study_id="study", cohort_id="c1", role="treated", reported_value=3)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="zero",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c1",
            count_kind="dlt_count",
            reported_count=0,
        )
    ]

    row = resolve_extraction(extraction).curve_rows[0]
    assert (row.y, row.n) == (0, 3)
    assert row.numerator_basis == "reported"
    assert "R1" not in row.rule_ids
    assert row.included_in_reported_only


def test_explicit_same_patient_mapping_overrides_r1() -> None:
    extraction = base_extraction(doses=(100,))
    extraction.population_counts = [
        PopulationCount(count_id="n", study_id="study", cohort_id="c1", role="treated", reported_value=6)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c1",
            count_kind="events",
            reported_count=2,
            explicit_patient_count=1,
            same_patient_repeated_events=True,
        )
    ]
    row = resolve_extraction(extraction).curve_rows[0]
    assert (row.y, row.n, row.reported_dlt_count) == (1, 6, 2)
    assert row.numerator_basis == "reported"
    assert "R1" not in row.rule_ids
    assert row.included_in_reported_only


def test_r3_then_r2_gives_three_six_three() -> None:
    extraction = base_extraction(design=True)
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c2",
            count_kind="patients",
            reported_count=1,
        )
    ]
    rows = resolve_extraction(extraction).curve_rows
    assert [(row.y, row.n) for row in rows] == [(0, 3), (1, 6), (0, 3)]
    assert rows[0].rule_ids == ["R3", "R2"]
    assert rows[1].rule_ids == ["R3"]
    assert rows[2].rule_ids == ["R3", "R2"]
    assert all(row.denominator_basis == "design_inferred" for row in rows)


def test_d0_then_r3_produces_zero_rows() -> None:
    extraction = base_extraction(design=True, explicit_no_dlt=True)
    rows = resolve_extraction(extraction).curve_rows
    assert [(row.y, row.n) for row in rows] == [(0, 3), (0, 3), (0, 3)]
    assert all(row.rule_ids == ["D0", "R3"] for row in rows)
    assert not any(row.included_in_reported_only for row in rows)


def test_dlt_evaluable_denominator_precedes_treated() -> None:
    extraction = base_extraction(doses=(70,))
    extraction.population_counts = [
        PopulationCount(count_id="treated", study_id="study", cohort_id="c1", role="treated", reported_value=9),
        PopulationCount(count_id="eval", study_id="study", cohort_id="c1", role="dlt_evaluable", reported_value=6),
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c1",
            count_kind="patients",
            reported_count=1,
            denominator_count_id="eval",
        )
    ]
    row = resolve_extraction(extraction).curve_rows[0]
    assert row.n == 6
    assert row.reported_treated_n == 9
    assert row.reported_dlt_evaluable_n == 6
    assert row.denominator_role == "paired_dlt"


def test_explicit_not_evaluable_prevents_r2() -> None:
    extraction = base_extraction(doses=(50, 100))
    extraction.population_counts = [
        PopulationCount(count_id=f"n{i}", study_id="study", cohort_id=f"c{i}", role="treated", reported_value=3)
        for i in (1, 2)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="positive",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c1",
            count_kind="patients",
            reported_count=1,
        ),
        DLTObservation(
            observation_id="ne",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c2",
            reporting_status="explicitly_not_evaluable",
        ),
    ]
    row = resolve_extraction(extraction).curve_rows[1]
    assert row.y is None
    assert not row.included_in_default
    assert "R2" not in row.rule_ids


def test_r3_two_dlts_defaults_to_three_and_preserves_candidates() -> None:
    extraction = base_extraction(doses=(100,), design=True)
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c1",
            count_kind="dlt_count",
            reported_count=2,
        )
    ]
    row = resolve_extraction(extraction).curve_rows[0]
    assert (row.y, row.n) == (2, 3)
    assert row.design_candidate_ns == [3, 6]
    assert row.rule_ids == ["R1", "R3"]


def test_direct_nonstandard_n_overrides_three_plus_three() -> None:
    extraction = base_extraction(doses=(100,), design=True)
    extraction.population_counts = [
        PopulationCount(count_id="n", study_id="study", cohort_id="c1", role="treated", reported_value=7)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c1",
            count_kind="patients",
            reported_count=1,
        )
    ]
    row = resolve_extraction(extraction).curve_rows[0]
    assert row.n == 7
    assert row.denominator_basis == "reported"
    assert "R3" not in row.rule_ids


def test_explicit_early_stop_keeps_n_three_when_y_one() -> None:
    extraction = base_extraction(doses=(100,), design=True)
    extraction.cohorts[0].expansion_status = "stopped_before_expansion"
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c1",
            count_kind="patients",
            reported_count=1,
        )
    ]
    row = resolve_extraction(extraction).curve_rows[0]
    assert (row.y, row.n) == (1, 3)


def test_unassigned_positive_dlt_prevents_zero_imputation() -> None:
    extraction = base_extraction(doses=(50, 100))
    extraction.result_scopes[0].unresolved_positive_dlt = True
    extraction.population_counts = [
        PopulationCount(count_id=f"n{i}", study_id="study", cohort_id=f"c{i}", role="treated", reported_value=3)
        for i in (1, 2)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="scope-positive",
            study_id="study",
            result_scope_id="scope",
            cohort_id=None,
            count_kind="patients",
            reported_count=1,
        )
    ]
    rows = resolve_extraction(extraction).curve_rows
    assert all(row.y is None for row in rows)
    assert not any("R2" in row.rule_ids for row in rows)


def test_unique_dlt_remainder_runs_before_r2_and_inherits_r1() -> None:
    extraction = base_extraction(doses=(50, 100))
    extraction.population_counts = [
        PopulationCount(count_id=f"n{i}", study_id="study", cohort_id=f"c{i}", role="treated", reported_value=6)
        for i in (1, 2)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="total",
            study_id="study",
            result_scope_id="scope",
            cohort_id=None,
            count_kind="dlt_count",
            reported_count=4,
        ),
        DLTObservation(
            observation_id="known",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c1",
            count_kind="dlt_count",
            reported_count=2,
        ),
        DLTObservation(
            observation_id="missing",
            study_id="study",
            result_scope_id="scope",
            cohort_id="c2",
            count_kind="patients",
            reporting_status="not_reported",
        ),
    ]
    extraction.count_equations = [
        CountEquation(
            equation_id="eq",
            value_type="dlt_patient_count",
            total_record_id="total",
            component_record_ids=["known", "missing"],
            mutually_exclusive=True,
            exhaustive=True,
        )
    ]
    rows = resolve_extraction(extraction).curve_rows
    assert [(row.y, row.n) for row in rows] == [(2, 6), (2, 6)]
    assert rows[1].numerator_basis == "derived"
    assert "R2" not in rows[1].rule_ids
    assert "R1" in rows[1].rule_ids


def test_shared_scope_preserves_each_cohorts_series_identity() -> None:
    extraction = base_extraction(doses=(50, 100))
    extraction.dose_series.append(
        DoseSeries(
            series_id="series_b",
            study_id="study",
            part_id="part_b",
            arm_id="arm_b",
            label="TC6 combination",
            regimen_id="regimen_b",
            varying_component_id="drug_b",
        )
    )
    extraction.cohorts[1].series_id = "series_b"
    extraction.cohorts[1].regimen_id = "regimen_b"
    extraction.population_counts = [
        PopulationCount(count_id="n1", study_id="study", cohort_id="c1", role="treated", reported_value=3),
        PopulationCount(
            count_id="n2", study_id="study", part_id="part_b", arm_id="arm_b",
            cohort_id="c2", role="treated", reported_value=3,
        ),
    ]
    extraction.result_scopes[0].has_observed_dlt_results = True
    rows = resolve_extraction(extraction).curve_rows
    assert [(row.series_id, row.regimen_id, row.part_id, row.arm_id) for row in rows] == [
        ("series", "regimen", None, None),
        ("series_b", "regimen_b", "part_b", "arm_b"),
    ]


def test_conflicting_denominator_blocks_r3_and_the_curve_row() -> None:
    extraction = base_extraction(doses=(100,), design=True)
    extraction.population_counts = [
        PopulationCount(count_id="n_a", study_id="study", cohort_id="c1", role="treated", reported_value=3),
        PopulationCount(count_id="n_b", study_id="study", cohort_id="c1", role="treated", reported_value=6),
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs", study_id="study", result_scope_id="scope",
            cohort_id="c1", count_kind="patients", reported_count=1,
        )
    ]
    result = resolve_extraction(extraction)
    row = result.curve_rows[0]
    assert row.n is None
    assert "R3" not in row.rule_ids
    assert not row.included_in_default
    assert any(issue.code == "conflicting_denominators" for issue in result.issues)


def test_positive_named_dlt_with_unknown_patient_count_cannot_become_zero() -> None:
    extraction = base_extraction(doses=(100,))
    extraction.population_counts = [
        PopulationCount(count_id="n", study_id="study", cohort_id="c1", role="treated", reported_value=3)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="positive", study_id="study", result_scope_id="scope",
            cohort_id="c1", count_kind="named_toxicities_only",
            toxicities_raw=["grade 3 toxicity"],
        )
    ]
    result = resolve_extraction(extraction)
    row = result.curve_rows[0]
    assert row.y is None
    assert "R2" not in row.rule_ids
    assert not row.included_in_default
    assert any(issue.code == "positive_dlt_patient_count_unresolved" for issue in result.issues)


def test_incomplete_programmatic_material_coverage_disables_r2() -> None:
    extraction = base_extraction(doses=(50, 100))
    extraction.material_coverage_complete = False
    extraction.population_counts = [
        PopulationCount(count_id=f"n{i}", study_id="study", cohort_id=f"c{i}", role="treated", reported_value=3)
        for i in (1, 2)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs", study_id="study", result_scope_id="scope",
            cohort_id="c2", count_kind="patients", reported_count=1,
        )
    ]
    rows = resolve_extraction(extraction).curve_rows
    assert rows[0].y is None
    assert "R2" not in rows[0].rule_ids


def test_model_reported_dlt_material_gap_disables_r2() -> None:
    extraction = base_extraction(doses=(50, 100))
    extraction.material_coverage_complete = True
    extraction.extraction_coverage = [
        ExtractionCoverage(
            category="dlt_results",
            status="material_not_covered",
            note="The referenced supplement was not supplied.",
        )
    ]
    extraction.population_counts = [
        PopulationCount(count_id=f"n{i}", study_id="study", cohort_id=f"c{i}", role="treated", reported_value=3)
        for i in (1, 2)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs", study_id="study", result_scope_id="scope",
            cohort_id="c2", count_kind="patients", reported_count=1,
        )
    ]
    rows = resolve_extraction(extraction).curve_rows
    assert rows[0].y is None
    assert "R2" not in rows[0].rule_ids


def test_unassigned_positive_patient_count_blocks_r2_without_manual_flag() -> None:
    extraction = base_extraction(doses=(50, 100))
    extraction.population_counts = [
        PopulationCount(count_id=f"n{i}", study_id="study", cohort_id=f"c{i}", role="treated", reported_value=3)
        for i in (1, 2)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="unassigned", study_id="study", result_scope_id="scope",
            cohort_id=None, count_kind="patients", reported_count=1,
        )
    ]
    rows = resolve_extraction(extraction).curve_rows
    assert all(row.y is None for row in rows)
    assert not any("R2" in row.rule_ids for row in rows)


def test_explicit_not_evaluable_overrides_scope_zero() -> None:
    extraction = base_extraction(doses=(100,), design=True, explicit_no_dlt=True)
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="ne", study_id="study", result_scope_id="scope",
            cohort_id="c1", reporting_status="explicitly_not_evaluable",
        )
    ]
    result = resolve_extraction(extraction)
    row = result.curve_rows[0]
    assert row.y is None
    assert not row.included_in_default
    assert "D0" not in row.rule_ids
    assert any(issue.code == "unusable_dlt_status" for issue in result.issues)


def test_unrelated_efficacy_zero_does_not_block_dlt_denominator() -> None:
    extraction = base_extraction(doses=(100,))
    extraction.population_counts = [
        PopulationCount(count_id="treated", study_id="study", cohort_id="c1", role="treated", reported_value=3),
        PopulationCount(count_id="efficacy", study_id="study", cohort_id="c1", role="efficacy_evaluable", reported_value=0),
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs", study_id="study", result_scope_id="scope",
            cohort_id="c1", count_kind="patients", reported_count=1,
        )
    ]
    result = resolve_extraction(extraction)
    assert (result.curve_rows[0].y, result.curve_rows[0].n) == (1, 3)
    assert result.curve_rows[0].included_in_default
    assert not any(issue.code == "nonpositive_denominator" for issue in result.issues)


def test_observation_window_mismatch_blocks_row() -> None:
    extraction = base_extraction(doses=(100,))
    extraction.result_scopes[0].window_id = "cycle_1"
    extraction.population_counts = [
        PopulationCount(count_id="n", study_id="study", cohort_id="c1", role="treated", reported_value=3)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="later", study_id="study", result_scope_id="scope",
            cohort_id="c1", window_id="cycle_2", count_kind="patients", reported_count=1,
        )
    ]
    result = resolve_extraction(extraction)
    row = result.curve_rows[0]
    assert row.y is None
    assert not row.included_in_default
    assert any(issue.code == "dlt_scope_mismatch" for issue in result.issues)


def test_mass_unit_conversion_scales_value() -> None:
    extraction = base_extraction(doses=(0.05,))
    extraction.cohorts[0].dose.unit_raw = "g"
    extraction.cohorts[0].dose.unit_normalized = "mg"
    extraction.population_counts = [
        PopulationCount(count_id="n", study_id="study", cohort_id="c1", role="treated", reported_value=3)
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs", study_id="study", result_scope_id="scope",
            cohort_id="c1", count_kind="patients", reported_count=0,
        )
    ]
    row = resolve_extraction(extraction).curve_rows[0]
    assert row.dose_value == 50
    assert row.dose_unit == "mg"


def test_arithmetic_provenance_sets_derived_denominator_basis() -> None:
    extraction = base_extraction(doses=(100,))
    extraction.population_counts = [
        PopulationCount(
            count_id="n", study_id="study", cohort_id="c1", role="dlt_evaluable", reported_value=3,
            provenance=ValueProvenance(
                basis="arithmetic_derived", input_record_ids=["treated", "replaced"], expression="4 - 1",
            ),
        )
    ]
    extraction.dlt_observations = [
        DLTObservation(
            observation_id="obs", study_id="study", result_scope_id="scope",
            cohort_id="c1", count_kind="patients", reported_count=0,
        )
    ]
    row = resolve_extraction(extraction).curve_rows[0]
    assert row.n == 3
    assert row.denominator_basis == "derived"
