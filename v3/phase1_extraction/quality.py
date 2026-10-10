"""Programmatic coverage and referential-integrity audit for model output.

This audit does not decide whether a clinical statement is true. It detects
concrete gaps that can be established without re-reading the source: missing
coverage declarations, unread request material, broken entity references, and
actual cohorts that cannot be connected to a result scope.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .models import DocumentIR, ReportedExtraction, ResolvedExtraction
from .exposures import exposure_problems


EXPECTED_COVERAGE_CATEGORIES = (
    "study_structure",
    "population",
    "eligibility",
    "interventions",
    "dosing",
    "design",
    "population_counts",
    "dlt_definition",
    "dlt_window",
    "dlt_results",
)


def audit_reported_extraction(
    extraction: ReportedExtraction,
    *,
    document: DocumentIR | None = None,
    resolved: ResolvedExtraction | None = None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []

    def add(
        code: str,
        severity: str,
        category: str,
        message: str,
        affected_ids: list[str] | None = None,
    ) -> None:
        findings.append(
            {
                "code": code,
                "severity": severity,
                "category": category,
                "message": message,
                "affected_ids": affected_ids or [],
            }
        )

    coverage_counts = Counter(item.category for item in extraction.extraction_coverage)
    for category in EXPECTED_COVERAGE_CATEGORIES:
        count = coverage_counts.get(category, 0)
        if count == 0:
            add(
                "missing_coverage_declaration",
                "targeted_reread",
                category,
                f"No extraction coverage status was returned for {category}.",
            )
        elif count > 1:
            add(
                "duplicate_coverage_declaration",
                "targeted_reread",
                category,
                f"{count} extraction coverage records were returned for {category}.",
            )
    for item in extraction.extraction_coverage:
        if item.status in {"material_not_covered", "conflict"}:
            severity = "source_material_required" if item.status == "material_not_covered" else "targeted_reread"
            add(
                f"coverage_{item.status}",
                severity,
                item.category,
                item.note or f"Coverage status is {item.status}.",
                item.source_ids,
            )

    if extraction.material_coverage_complete is not True:
        add(
            "material_coverage_incomplete",
            "blocking",
            "materials",
            "Programmatic material coverage is incomplete; absence-based inference must remain disabled.",
        )
    if document is not None:
        unread = sorted(
            item.source_id
            for item in document.coverage
            if item.potentially_relevant
            and (not item.included_in_extraction or item.parse_status in {"failed", "unread"})
        )
        if unread:
            add(
                "relevant_material_not_read",
                "blocking",
                "materials",
                f"{len(unread)} potentially relevant source units were not readable and included.",
                unread,
            )

    study_ids = {item.study_id for item in extraction.studies}
    part_ids = {item.part_id for item in extraction.study_parts}
    arm_ids = {item.arm_id for item in extraction.arms}
    regimen_ids = {item.regimen_id for item in extraction.regimens}
    series_ids = {item.series_id for item in extraction.dose_series}
    cohort_ids = {item.cohort_id for item in extraction.cohorts}
    scope_ids = {item.result_scope_id for item in extraction.result_scopes}
    window_ids = {item.window_id for item in extraction.dlt_windows}
    analysis_population_ids = {item.analysis_population_id for item in extraction.analysis_populations}
    count_ids = {item.count_id for item in extraction.population_counts}

    def check_ref(
        owner_id: str,
        value: str | None,
        valid: set[str],
        field: str,
        category: str,
    ) -> None:
        if value is not None and value not in valid:
            add(
                "unknown_entity_reference",
                "blocking",
                category,
                f"{owner_id}.{field} refers to unknown id {value}.",
                [owner_id, value],
            )

    for item in extraction.study_parts:
        check_ref(item.part_id, item.study_id, study_ids, "study_id", "study_structure")
    for item in extraction.arms:
        check_ref(item.arm_id, item.study_id, study_ids, "study_id", "study_structure")
        check_ref(item.arm_id, item.part_id, part_ids, "part_id", "study_structure")
        check_ref(item.arm_id, item.regimen_id, regimen_ids, "regimen_id", "study_structure")
    for item in extraction.dose_series:
        check_ref(item.series_id, item.study_id, study_ids, "study_id", "study_structure")
        check_ref(item.series_id, item.part_id, part_ids, "part_id", "study_structure")
        check_ref(item.series_id, item.arm_id, arm_ids, "arm_id", "study_structure")
        check_ref(item.series_id, item.regimen_id, regimen_ids, "regimen_id", "dosing")
        check_ref(item.series_id, item.window_id, window_ids, "window_id", "dlt_window")
    for item in extraction.cohorts:
        if extraction.exposure_contract and item.actual_status == "actual":
            if not item.exposures or item.exposure_coverage != "complete":
                add("incomplete_cohort_exposures", "targeted_reread", "dosing",
                    "Actual cohort lacks a complete, explicitly linked drug exposure configuration.", [item.cohort_id])
        series = next((s for s in extraction.dose_series if s.series_id == item.series_id), None)
        for problem in exposure_problems(item, series.varying_component_id if series else None):
            add("cohort_exposure_conflict", "targeted_reread", "dosing", problem, [item.cohort_id])
        for exposure in item.exposures:
            check_ref(exposure.exposure_id, exposure.intervention_id,
                      {x.intervention_id for x in extraction.interventions}, "intervention_id", "dosing")
            regimen = next((r for r in extraction.regimens if r.regimen_id == item.regimen_id), None)
            check_ref(exposure.exposure_id, exposure.component_id,
                      {c.component_id for c in regimen.components} if regimen else set(), "component_id", "dosing")
            check_ref(exposure.exposure_id, exposure.window_id, window_ids, "window_id", "dlt_window")
            if exposure.presence in {"present", "absent", "mixed"} and not exposure.evidence:
                add("ungrounded_exposure_status", "targeted_reread", "dosing",
                    f"{exposure.name_raw}: exposure status lacks source evidence.", [item.cohort_id, exposure.exposure_id])
        check_ref(item.cohort_id, item.series_id, series_ids, "series_id", "study_structure")
        check_ref(item.cohort_id, item.regimen_id, regimen_ids, "regimen_id", "dosing")
        for scope_id in item.result_scope_ids:
            check_ref(item.cohort_id, scope_id, scope_ids, "result_scope_ids", "dlt_results")
    scoped_cohorts: set[str] = set()
    for item in extraction.result_scopes:
        check_ref(item.result_scope_id, item.study_id, study_ids, "study_id", "dlt_results")
        check_ref(item.result_scope_id, item.part_id, part_ids, "part_id", "dlt_results")
        check_ref(item.result_scope_id, item.arm_id, arm_ids, "arm_id", "dlt_results")
        check_ref(item.result_scope_id, item.series_id, series_ids, "series_id", "dlt_results")
        check_ref(item.result_scope_id, item.window_id, window_ids, "window_id", "dlt_results")
        for cohort_id in item.cohort_ids:
            scoped_cohorts.add(cohort_id)
            check_ref(item.result_scope_id, cohort_id, cohort_ids, "cohort_ids", "dlt_results")
    for item in extraction.population_counts:
        check_ref(item.count_id, item.study_id, study_ids, "study_id", "population_counts")
        check_ref(item.count_id, item.part_id, part_ids, "part_id", "population_counts")
        check_ref(item.count_id, item.arm_id, arm_ids, "arm_id", "population_counts")
        check_ref(item.count_id, item.cohort_id, cohort_ids, "cohort_id", "population_counts")
        check_ref(item.count_id, item.result_scope_id, scope_ids, "result_scope_id", "population_counts")
        check_ref(
            item.count_id,
            item.analysis_population_id,
            analysis_population_ids,
            "analysis_population_id",
            "population_counts",
        )
        check_ref(item.count_id, item.window_id, window_ids, "window_id", "population_counts")
    for item in extraction.dlt_observations:
        check_ref(item.observation_id, item.result_scope_id, scope_ids, "result_scope_id", "dlt_results")
        check_ref(item.observation_id, item.cohort_id, cohort_ids, "cohort_id", "dlt_results")
        check_ref(item.observation_id, item.window_id, window_ids, "window_id", "dlt_results")
        check_ref(item.observation_id, item.denominator_count_id, count_ids, "denominator_count_id", "dlt_results")

    dlt_coverage = next(
        (item for item in extraction.extraction_coverage if item.category == "dlt_results"),
        None,
    )
    if dlt_coverage is not None and dlt_coverage.status == "extracted":
        unscoped = sorted(
            item.cohort_id
            for item in extraction.cohorts
            if item.actual_status == "actual"
            and item.cohort_id not in scoped_cohorts
            and not any(scope_id in scope_ids for scope_id in item.result_scope_ids)
        )
        if unscoped:
            add(
                "actual_cohorts_without_result_scope",
                "targeted_reread",
                "dlt_results",
                f"{len(unscoped)} actual cohorts cannot be connected to a DLT result scope.",
                unscoped,
            )

    if resolved is not None:
        findings.extend(_audit_resolved_rows(extraction, resolved))

    return {
        "format": "phase1_extraction_audit_v3.1",
        "document_id": extraction.document_id,
        "source_truth_checked": False,
        "blocking_count": sum(item["severity"] == "blocking" for item in findings),
        "targeted_reread_count": sum(item["severity"] == "targeted_reread" for item in findings),
        "source_material_required_count": sum(
            item["severity"] == "source_material_required" for item in findings
        ),
        "requires_targeted_reread": any(item["severity"] == "targeted_reread" for item in findings),
        "requires_source_material": any(item["severity"] == "source_material_required" for item in findings),
        "findings": findings,
    }


def _audit_resolved_rows(extraction: ReportedExtraction, resolved: ResolvedExtraction) -> list[dict[str, Any]]:
    """Check projections against the preserved facts, including excluded rows."""
    from .resolver import _Value, _r2_allowed, _scope_identity

    findings: list[dict[str, Any]] = []
    scopes = {s.result_scope_id: s for s in extraction.result_scopes}
    cohorts = {c.cohort_id: c for c in extraction.cohorts}
    coverage = {s.result_scope_id: s.status for s in resolved.scope_coverage}
    grouped: dict[tuple, list] = {}

    def add(code: str, message: str, ids: list[str]) -> None:
        findings.append(dict(code=code, severity="blocking", category="resolved_results",
                             message=message, affected_ids=ids))

    for row in resolved.curve_rows:
        scope, cohort = scopes.get(row.result_scope_id), cohorts.get(row.cohort_id)
        if scope is None or cohort is None:
            add("unknown_resolved_reference", "Resolved row has an unknown cohort or scope", [row.curve_row_id])
            continue
        grouped.setdefault((row.cohort_id, _scope_identity(scope)), []).append(row)
        if row.denominator_basis == "design_inferred":
            reported = [c for c in extraction.population_counts
                        if c.cohort_id == row.cohort_id and c.study_id == row.study_id
                        and c.role in {"treated", "dlt_evaluable", "paired_dlt_denominator"}
                        and c.reporting_status == "reported" and c.reported_value is not None
                        and c.result_scope_id in scopes
                        and _scope_identity(scopes[c.result_scope_id]) == _scope_identity(scope)
                        and c.window_id in {None, row.window_id}]
            if any(c.reported_value != row.n for c in reported):
                add("design_denominator_conflicts_with_reported_count",
                    "Design-inferred N disagrees with a reported count for this cohort in the same scope context",
                    [row.curve_row_id, *[c.count_id for c in reported]])
        observations = [o for o in extraction.dlt_observations if o.result_scope_id == scope.result_scope_id]
        if row.y is None and _r2_allowed(
            cohort=cohort, scope=scope, observations=[o for o in observations if o.cohort_id == row.cohort_id],
            all_scope_observations=observations, denominator=_Value(row.n, row.denominator_basis),
            scope_coverage_status=coverage.get(scope.result_scope_id, "unknown"),
        ):
            add("eligible_zero_not_resolved", "A fully covered cohort eligible for R2 still has an unresolved numerator",
                [row.curve_row_id, row.cohort_id, row.result_scope_id])
    for rows in grouped.values():
        if len(rows) > 1:
            add("overlapping_cohort_projections",
                "The same cohort is projected more than once in the same study, series, window and cutoff; resolve overlap before use",
                [r.curve_row_id for r in rows])
    return findings


def apply_audit_status(extraction: ReportedExtraction, audit: dict[str, Any]) -> None:
    if (
        audit["blocking_count"]
        or audit["targeted_reread_count"]
        or audit["source_material_required_count"]
    ):
        extraction.extraction_complete = False
        note = (
            "Programmatic extraction audit found unresolved structural or coverage gaps; "
            "see extraction_audit.json."
        )
        if note not in extraction.coverage_notes:
            extraction.coverage_notes.append(note)
