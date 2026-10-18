"""Deterministically apply denominator selection, D0, R1, R3, and R2."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from .exposures import exposure_problems

from .models import (
    Cohort,
    CurveRow,
    DLTObservation,
    Issue,
    POLICY_VERSION,
    PopulationCount,
    ReportedExtraction,
    ResolvedExtraction,
    ResultScope,
    RuleApplication,
    ScopeCoverageAssessment,
    TrialDesign,
)


RULE_VERSION = "3.1.1"


_MASS_TO_MG = {"ug": 0.001, "mcg": 0.001, "mg": 1.0, "g": 1000.0}


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _source_ids(records: Iterable[object]) -> list[str]:
    found: list[str] = []
    for record in records:
        for link in getattr(record, "evidence", []):
            for source_id in link.source_ids:
                if source_id not in found:
                    found.append(source_id)
    return found


@dataclass
class _Value:
    value: int | None
    basis: str
    role: str = "unknown"
    input_ids: list[str] | None = None
    candidates: list[int] | None = None
    reason: str | None = None
    dependencies: list[str] | None = None


def _application(
    rule_id: str,
    target_id: str,
    target_field: str,
    value: int | float | str,
    input_ids: list[str],
    reason: str,
) -> RuleApplication:
    return RuleApplication(
        rule_application_id=_stable_id("rule", rule_id, target_id, target_field, value, *input_ids),
        rule_id=rule_id,
        version=RULE_VERSION,
        target_id=target_id,
        target_field=target_field,
        input_record_ids=input_ids,
        output_value=value,
        reason=reason,
    )


def _issue(
    severity: str,
    code: str,
    affected: list[str],
    message: str,
    *,
    field: str | None = None,
    source_ids: list[str] | None = None,
) -> Issue:
    return Issue(
        issue_id=_stable_id("issue", severity, code, *affected, message),
        severity=severity,
        code=code,
        affected_record_ids=affected,
        field=field,
        message=message,
        source_ids=source_ids or [],
    )


def _reported_patient_value(observation: DLTObservation) -> _Value:
    """Resolve an observed DLT record before any zero or design inference."""

    if observation.reporting_status != "reported":
        return _Value(None, "missing")
    if observation.explicit_patient_count is not None:
        return _Value(
            observation.explicit_patient_count,
            "reported",
            input_ids=[observation.observation_id],
            reason="The source explicitly reports the number of patients with DLT",
        )
    if observation.count_kind == "patients" and observation.reported_count is not None:
        return _Value(
            observation.reported_count,
            "reported",
            input_ids=[observation.observation_id],
            reason="The reported DLT count is explicitly patient-based",
        )
    if observation.reported_count == 0:
        return _Value(
            0,
            "reported",
            input_ids=[observation.observation_id],
            reason="A source-reported zero DLT count unambiguously means zero affected patients",
        )
    if (
        observation.reported_count is not None
        and observation.count_kind in {"dlt_count", "events", "episodes", "unclear"}
        and observation.same_patient_repeated_events is not True
    ):
        return _Value(
            observation.reported_count,
            "inferred_patient_count",
            input_ids=[observation.observation_id],
            reason="R1 interprets a summarized DLT count as affected patients because no contrary patient-event mapping is reported",
            dependencies=["R1"],
        )
    return _Value(None, "missing")


def _derive_equations(
    extraction: ReportedExtraction,
) -> tuple[dict[str, int], dict[str, _Value], list[RuleApplication]]:
    count_values = {
        item.count_id: item.reported_value
        for item in extraction.population_counts
        if item.reported_value is not None and item.reporting_status == "reported"
    }
    observation_values: dict[str, _Value] = {}
    directly_resolved_observations: set[str] = set()
    for item in extraction.dlt_observations:
        value = _reported_patient_value(item)
        if value.value is not None:
            observation_values[item.observation_id] = value
            directly_resolved_observations.add(item.observation_id)
    directly_reported_counts = set(count_values)
    applications: list[RuleApplication] = []
    applied_equations: set[str] = set()
    for _ in range(len(extraction.count_equations) + 1):
        progress = False
        for equation in extraction.count_equations:
            if equation.equation_id in applied_equations or not equation.mutually_exclusive or not equation.exhaustive:
                continue
            values: dict[str, object] = count_values if equation.value_type == "population_count" else observation_values
            if equation.total_record_id not in values:
                continue
            missing = [record_id for record_id in equation.component_record_ids if record_id not in values]
            if len(missing) != 1:
                continue
            if equation.value_type == "population_count":
                total = int(count_values[equation.total_record_id])
                known = [count_values[record_id] for record_id in equation.component_record_ids if record_id in count_values]
                dependencies: list[str] = []
            else:
                total_value = observation_values[equation.total_record_id]
                total = int(total_value.value or 0)
                known_values = [observation_values[record_id] for record_id in equation.component_record_ids if record_id in observation_values]
                known = [int(value.value or 0) for value in known_values]
                dependencies = list(
                    dict.fromkeys(
                        (total_value.dependencies or [])
                        + [dep for value in known_values for dep in (value.dependencies or [])]
                    )
                )
            derived = total - sum(known)
            if derived < 0:
                continue
            target = missing[0]
            if equation.value_type == "population_count":
                count_values[target] = derived
            else:
                observation_values[target] = _Value(
                    derived,
                    "derived",
                    input_ids=[equation.equation_id, equation.total_record_id, *equation.component_record_ids],
                    reason="Derived as the unique remainder of an explicit DLT patient total",
                    dependencies=dependencies,
                )
            applications.append(
                _application(
                    "DETERMINISTIC_DERIVATION",
                    target,
                    "reported_value" if equation.value_type == "population_count" else "resolved_patient_count",
                    derived,
                    [equation.equation_id, equation.total_record_id, *equation.component_record_ids],
                    "Unique remainder from an explicitly exhaustive, mutually exclusive total",
                )
            )
            applied_equations.add(equation.equation_id)
            progress = True
        if not progress:
            break
    return (
        {key: value for key, value in count_values.items() if key not in directly_reported_counts},
        {key: value for key, value in observation_values.items() if key not in directly_resolved_observations},
        applications,
    )


def _denominator_candidate(
    item: PopulationCount,
    analysis_roles: dict[str, str],
) -> bool:
    """Return whether a count is allowed to influence a DLT denominator."""

    return bool(
        item.role in {"paired_dlt_denominator", "dlt_evaluable", "treated"}
        or item.equivalent_to_role in {"dlt_evaluable", "treated"}
        or analysis_roles.get(item.analysis_population_id or "") in {"dlt_evaluable", "treated"}
    )


def _normalized_curve_dose(cohort: Cohort) -> tuple[float | None, str | None, str | None]:
    """Normalize only conversions whose mass dimension and denominator agree.

    The extractor may preserve a source value in grams while also proposing a
    normalized milligram unit. Relabelling without scaling changes the dose, so
    this function either performs a known conversion or retains the raw unit.
    """

    dose = cohort.dose
    if dose.value_kind != "scalar" or dose.value is None:
        return None, dose.unit_normalized or dose.unit_raw, None
    raw_unit = (dose.unit_raw or "").strip()
    normalized_unit = (dose.unit_normalized or "").strip()
    if not normalized_unit:
        return float(dose.value), raw_unit or None, None
    if not raw_unit:
        return float(dose.value), normalized_unit, None

    def split_unit(value: str) -> tuple[str, str] | None:
        canonical = value.casefold().replace("µ", "u").replace("μ", "u")
        canonical = canonical.translate(str.maketrans({"²": "2", "¹": "1", "⁻": "-", "−": "-"}))
        canonical = re.sub(r"\s+", "", canonical)
        match = re.fullmatch(r"(mcg|ug|mg|g)(.*)", canonical)
        if match is None:
            return None
        # Reciprocal exponents are notation changes, not body-size or schedule
        # conversions. Preserve unsupported powers and every other suffix.
        reciprocal_units = {("kg", "1"): "kg", ("m", "2"): "m2",
                            ("day", "1"): "day", ("d", "1"): "d", ("h", "1"): "h"}

        def reciprocal(part: re.Match[str]) -> str:
            denominator = reciprocal_units.get((part.group(1), part.group(2)))
            return f"/{denominator}" if denominator else part.group(0)

        suffix = re.sub(r"[·⋅*]?(kg|m|day|d|h)\^?-(1|2)(?!\d)", reciprocal, match.group(2))
        return match.group(1), suffix

    raw_parts = split_unit(raw_unit)
    normalized_parts = split_unit(normalized_unit)
    if raw_parts and normalized_parts and raw_parts[1] == normalized_parts[1]:
        scale = _MASS_TO_MG[raw_parts[0]] / _MASS_TO_MG[normalized_parts[0]]
        return float(dose.value) * scale, normalized_unit, None
    if raw_unit.casefold() == normalized_unit.casefold():
        return float(dose.value), normalized_unit, None
    return (
        float(dose.value),
        raw_unit,
        f"The proposed normalized unit {normalized_unit!r} was not dimensionally compatible with {raw_unit!r}; the source unit was retained",
    )


def _counts_for_cohort(
    counts: list[PopulationCount],
    cohort: Cohort,
    scope: ResultScope,
    *,
    study_id: str,
    part_id: str | None,
    arm_id: str | None,
    window_id: str | None,
    analysis_windows: dict[str, str | None],
) -> list[PopulationCount]:
    matches: list[PopulationCount] = []
    for count in counts:
        if count.study_id != study_id:
            continue
        if count.part_id is not None and part_id is not None and count.part_id != part_id:
            continue
        if count.arm_id is not None and arm_id is not None and count.arm_id != arm_id:
            continue
        if count.result_scope_id is not None and count.result_scope_id != scope.result_scope_id:
            continue
        count_window = count.window_id or analysis_windows.get(count.analysis_population_id or "")
        if count_window is not None and window_id is not None and count_window != window_id:
            continue
        if count.data_cutoff_raw and scope.data_cutoff_raw and count.data_cutoff_raw != scope.data_cutoff_raw:
            continue
        if count.cohort_id == cohort.cohort_id or (
            count.cohort_id is None
            and count.result_scope_id == scope.result_scope_id
            and len(scope.cohort_ids) == 1
        ):
            matches.append(count)
    return matches


def _select_denominator(
    *,
    cohort: Cohort,
    scope: ResultScope,
    observations: list[DLTObservation],
    counts: list[PopulationCount],
    analysis_roles: dict[str, str],
    analysis_windows: dict[str, str | None],
    derived_counts: dict[str, int],
    study_id: str,
    part_id: str | None,
    arm_id: str | None,
    window_id: str | None,
) -> tuple[_Value, int | None, int | None, list[Issue], bool]:
    issues: list[Issue] = []
    candidates = [
        item for item in _counts_for_cohort(
        counts,
        cohort,
        scope,
        study_id=study_id,
        part_id=part_id,
        arm_id=arm_id,
        window_id=window_id,
        analysis_windows=analysis_windows,
        )
        if _denominator_candidate(item, analysis_roles)
    ]
    linked_ids = {item.denominator_count_id for item in observations if item.denominator_count_id}
    candidate_ids = {item.count_id for item in candidates}
    applicable_by_id = {
        item.count_id: item
        for item in _counts_for_cohort(
            counts,
            cohort,
            scope,
            study_id=study_id,
            part_id=part_id,
            arm_id=arm_id,
            window_id=window_id,
            analysis_windows=analysis_windows,
        )
        if _denominator_candidate(item, analysis_roles)
    }
    candidates.extend(
        applicable_by_id[count_id]
        for count_id in linked_ids
        if count_id in applicable_by_id and count_id not in candidate_ids
    )

    def effective_value(item: PopulationCount) -> tuple[int | None, str]:
        if item.count_id in derived_counts:
            value = derived_counts[item.count_id]
            return (value, "derived") if value > 0 else (None, "missing")
        if item.reporting_status == "reported":
            value = item.reported_value
            if value is None or value <= 0:
                return None, "missing"
            return (
                value,
                "derived"
                if item.derived_from_count_ids or item.provenance.basis == "arithmetic_derived"
                else "reported",
            )
        return None, "missing"

    invalid_counts = [
        item
        for item in candidates
        if (
            item.reporting_status == "reported"
            and item.reported_value is not None
            and item.reported_value <= 0
        )
        or (item.count_id in derived_counts and derived_counts[item.count_id] <= 0)
    ]
    if invalid_counts:
        issues.append(
            _issue(
                "row_blocker",
                "nonpositive_denominator",
                [cohort.cohort_id, *[item.count_id for item in invalid_counts]],
                "A candidate population denominator is zero or negative",
                field="n",
                source_ids=_source_ids(invalid_counts),
            )
        )

    unavailable_counts = [
        item for item in candidates
        if item.reporting_status in {"conflicting", "explicitly_not_evaluable"}
    ]
    if unavailable_counts:
        issues.append(
            _issue(
                "row_blocker",
                "unusable_denominator_status",
                [cohort.cohort_id, *[item.count_id for item in unavailable_counts]],
                "A candidate denominator is conflicting or explicitly not evaluable",
                field="n",
                source_ids=_source_ids(unavailable_counts),
            )
        )

    blocked = bool(invalid_counts or unavailable_counts)

    treated_values = [
        effective_value(item)[0]
        for item in candidates
        if item.role == "treated" and effective_value(item)[0] is not None
    ]
    evaluable_values = [
        effective_value(item)[0]
        for item in candidates
        if (
            item.role in {"dlt_evaluable", "paired_dlt_denominator"}
            or analysis_roles.get(item.analysis_population_id or "") == "dlt_evaluable"
            or item.equivalent_to_role == "dlt_evaluable"
        )
        and effective_value(item)[0] is not None
    ]
    reported_treated = treated_values[0] if treated_values and len(set(treated_values)) == 1 else None
    reported_evaluable = evaluable_values[0] if evaluable_values and len(set(evaluable_values)) == 1 else None

    ranked: list[tuple[int, PopulationCount, str]] = []
    for item in candidates:
        value, _ = effective_value(item)
        if value is None:
            continue
        if item.count_id in linked_ids or item.role == "paired_dlt_denominator":
            rank, role = 0, "paired_dlt"
        elif (
            item.role == "dlt_evaluable"
            or analysis_roles.get(item.analysis_population_id or "") == "dlt_evaluable"
            or item.equivalent_to_role == "dlt_evaluable"
        ):
            rank, role = 1, "dlt_evaluable"
        elif item.role == "treated":
            rank, role = 2, "treated"
        elif item.equivalent_to_role == "treated" or analysis_roles.get(item.analysis_population_id or "") == "treated":
            rank, role = 3, "equivalent_treated"
        else:
            continue
        ranked.append((rank, item, role))
    if not ranked:
        return _Value(None, "missing"), reported_treated, reported_evaluable, issues, blocked
    best_rank = min(item[0] for item in ranked)
    best = [item for item in ranked if item[0] == best_rank]
    values = {effective_value(item[1])[0] for item in best}
    if len(values) != 1:
        issues.append(
            _issue(
                "row_blocker",
                "conflicting_denominators",
                [cohort.cohort_id, *[item[1].count_id for item in best]],
                "Counts with the same denominator priority disagree in the same cohort and result scope",
                field="n",
                source_ids=_source_ids([item[1] for item in best]),
            )
        )
        return _Value(None, "missing"), reported_treated, reported_evaluable, issues, True
    selected = best[0]
    value, basis = effective_value(selected[1])
    return (
        _Value(
            value,
            basis,
            role=selected[2],
            input_ids=[selected[1].count_id],
            reason=f"Selected {selected[2]} using the predefined denominator priority",
        ),
        reported_treated,
        reported_evaluable,
        issues,
        blocked,
    )


def _r3_denominator(
    cohort: Cohort,
    design: TrialDesign | None,
    numerator: _Value,
) -> _Value:
    if design is None or design.normalized_type not in {"classic_3plus3", "modified_3plus3"}:
        return _Value(None, "missing")
    # A dose-expansion/confirmation population need not have the escalation
    # design's size (e.g. an initial three plus five additional patients).
    if cohort.cohort_role in {"expansion", "confirmation"}:
        return _Value(None, "missing")
    initial = design.initial_cohort_n or 3
    added = design.added_cohort_n or 3
    expanded = design.expanded_total_n or initial + added
    candidates: list[int] = []
    if cohort.expansion_status in {"completed_expansion", "completed_to_six", "added_three"}:
        value = expanded
        assumption = "The source reports completion of the design expansion"
    elif cohort.expansion_status in {"initial_three_only", "stopped_before_expansion"}:
        value = initial
        assumption = "The source limits the cohort to the initial design stage"
    elif design.normalized_type == "modified_3plus3" and (
        design.initial_cohort_n is not None
        or design.added_cohort_n is not None
        or design.expand_when_y_equals
        or design.stop_when_y_at_least is not None
    ):
        y = numerator.value
        if y is not None and y in design.expand_when_y_equals:
            value = expanded
            assumption = "The source-specific modified 3+3 expansion rule applies"
        elif y is not None and design.stop_when_y_at_least is not None and y >= design.stop_when_y_at_least:
            value = initial
            assumption = "The source-specific modified 3+3 stopping rule applies"
        elif y is None or y == 0:
            value = initial
            assumption = "No source-specific expansion trigger is observed; use the initial modified-design cohort"
        else:
            return _Value(None, "missing")
    else:
        y = numerator.value
        if y == 1:
            value = 6
            assumption = "Classic 3+3 assumption: one DLT leads to a completed additional cohort"
        elif y is None or y == 0:
            value = 3
            assumption = "Classic 3+3 assumption: use the initial cohort when expansion is not indicated"
        elif y in {2, 3}:
            value = 3
            candidates = [3, 6]
            assumption = "Classic 3+3 assumption: the initial cohort reached the stopping condition"
        elif y is not None and 4 <= y <= 6:
            value = 6
            assumption = "The adopted patient numerator cannot fit in a three-patient cohort"
        else:
            return _Value(None, "missing")
    return _Value(
        value,
        "design_inferred",
        role="design_assumed_cohort",
        input_ids=[design.design_id, cohort.cohort_id, *(numerator.input_ids or [])],
        candidates=candidates,
        reason=assumption,
    )


def _resolve_numerator(
    observations: list[DLTObservation],
    scope: ResultScope,
    cohort: Cohort,
    derived_observations: dict[str, _Value],
) -> tuple[_Value, int | None, list[Issue]]:
    issues: list[Issue] = []
    unavailable = [
        item for item in observations
        if item.reporting_status in {"conflicting", "explicitly_not_evaluable"}
    ]
    if unavailable:
        issues.append(
            _issue(
                "row_blocker",
                "unusable_dlt_status",
                [cohort.cohort_id, *[item.observation_id for item in unavailable]],
                "The cohort-specific DLT result is conflicting or explicitly not evaluable",
                field="y",
                source_ids=_source_ids(unavailable),
            )
        )
        return _Value(None, "missing"), None, issues
    reported = [item for item in observations if item.reporting_status == "reported"]
    values: list[_Value] = []
    raw_counts: list[int] = []
    for item in observations:
        if item.reporting_status != "reported" and item.observation_id not in derived_observations:
            continue
        if (
            (item.reported_count is not None and item.reported_count < 0)
            or (item.explicit_patient_count is not None and item.explicit_patient_count < 0)
        ):
            issues.append(
                _issue(
                    "row_blocker",
                    "negative_dlt_count",
                    [cohort.cohort_id, item.observation_id],
                    "A reported DLT count is negative",
                    field="y",
                    source_ids=_source_ids([item]),
                )
            )
            continue
        if item.reported_count is not None:
            raw_counts.append(item.reported_count)
        if item.observation_id in derived_observations:
            values.append(derived_observations[item.observation_id])
        else:
            value = _reported_patient_value(item)
            if value.value is not None:
                values.append(value)
    unique = {item.value for item in values}
    if len(unique) > 1:
        issues.append(
            _issue(
                "row_blocker",
                "conflicting_dlt_patient_counts",
                [cohort.cohort_id, *[item.observation_id for item in reported]],
                "Multiple DLT observations resolve to different patient counts in the same cohort and scope",
                field="y",
                source_ids=_source_ids(reported),
            )
        )
        return _Value(None, "missing"), raw_counts[0] if raw_counts else None, issues
    if values:
        priority = {"reported": 0, "derived": 1, "inferred_patient_count": 2}
        preferred = sorted(values, key=lambda item: priority.get(item.basis, 9))[0]
        return preferred, raw_counts[0] if raw_counts else None, issues
    unresolved_positive = [
        item
        for item in reported
        if (item.reported_count or 0) > 0
        or (item.explicit_patient_count or 0) > 0
        or bool(item.toxicities_raw)
    ]
    if unresolved_positive:
        issues.append(
            _issue(
                "row_blocker",
                "positive_dlt_patient_count_unresolved",
                [cohort.cohort_id, *[item.observation_id for item in unresolved_positive]],
                "Positive DLT evidence exists, but the number of affected patients cannot be resolved",
                field="y",
                source_ids=_source_ids(unresolved_positive),
            )
        )
        return _Value(None, "missing"), raw_counts[0] if raw_counts else None, issues
    if scope.explicit_no_dlt_in_scope:
        return (
            _Value(
                0,
                "derived",
                input_ids=[scope.result_scope_id],
                reason="D0 propagates an explicit scope-wide statement of no DLT to this actual cohort",
            ),
            raw_counts[0] if raw_counts else None,
            issues,
        )
    return _Value(None, "missing"), raw_counts[0] if raw_counts else None, issues


def _aggregate_is_allocated(
    total: DLTObservation,
    scope: ResultScope,
    observations: list[DLTObservation],
) -> bool:
    """Recognize a redundant total, never interpret event totals as patients.

    Require one unambiguous patient observation per named cohort, within the
    exact study/scope/window. Duplicate records and incompatible observations
    remain unresolved instead of being summed or silently discarded.
    """
    target = _reported_patient_value(total)
    if target.basis != "reported" or target.value is None or target.value <= 0:
        return False
    if (total.study_id != scope.study_id or total.window_id != scope.window_id
        or total.result_scope_id != scope.result_scope_id):
        return False
    by_cohort: dict[str, DLTObservation] = {}
    for item in observations:
        if item.cohort_id is None:
            continue
        if item.reporting_status in {"conflicting", "explicitly_not_evaluable", "explicitly_unreported"}:
            return False
        if item.reporting_status != "reported":
            if item.reported_count is not None or item.explicit_patient_count is not None or item.toxicities_raw:
                return False
            continue
        if (item.study_id, item.result_scope_id, item.window_id) != (
            total.study_id, total.result_scope_id, total.window_id
        ) or item.cohort_id not in scope.cohort_ids:
            return False
        value = _reported_patient_value(item)
        if value.basis != "reported" or value.value is None or value.value < 0:
            return False
        if item.cohort_id in by_cohort:
            return False
        by_cohort[item.cohort_id] = item
    parts = list(by_cohort.values())
    if not parts or sum(_reported_patient_value(item).value or 0 for item in parts) != target.value:
        return False
    # If the total also reports events, those must reconcile separately.
    if total.count_kind != "patients" and total.reported_count is not None:
        if total.count_kind not in {"events", "episodes", "dlt_count"}:
            return False
        event_counts: list[int] = []
        for item in parts:
            if item.count_kind not in {"patients", "events", "episodes", "dlt_count"}:
                return False
            if item.count_kind == "patients" and item.same_patient_repeated_events is not False:
                return False
            if item.reported_count is None or item.reported_count < 0:
                return False
            event_counts.append(item.reported_count)
        if sum(event_counts) != total.reported_count:
            return False
    return True


def _r2_allowed(
    *,
    cohort: Cohort,
    scope: ResultScope,
    observations: list[DLTObservation],
    all_scope_observations: list[DLTObservation],
    denominator: _Value,
    scope_coverage_status: str,
) -> bool:
    if (
        denominator.value is None
        or scope.dlt_results_coverage != "complete"
        or scope.unresolved_positive_dlt
        or scope_coverage_status != "ready"
    ):
        return False
    if any(
        item.reporting_status in {"explicitly_unreported", "explicitly_not_evaluable", "conflicting"}
        for item in observations
    ):
        return False
    def positive_signal(item: DLTObservation) -> bool:
        return item.reporting_status == "reported" and (
            (item.reported_count or 0) > 0
            or (item.explicit_patient_count or 0) > 0
            or bool(item.toxicities_raw)
        )

    if any(
        positive_signal(item)
        and (
            (item.cohort_id is None and not _aggregate_is_allocated(item, scope, all_scope_observations))
            or (
                item.cohort_id == cohort.cohort_id
                and _reported_patient_value(item).value is None
            )
        )
        for item in all_scope_observations
    ):
        return False
    has_result = scope.has_observed_dlt_results or any(
        item.reporting_status == "reported"
        and (item.reported_count is not None or item.explicit_patient_count is not None)
        for item in all_scope_observations
    )
    return has_result and cohort.actual_status == "actual"


def _scope_identity(scope: ResultScope) -> tuple:
    """Unknown context is not interchangeable with a specified context."""
    return (scope.study_id, scope.part_id, scope.arm_id, scope.series_id,
            scope.window_id, scope.data_cutoff_raw)


def _specific_reported_scope(
    extraction: ReportedExtraction, broad: ResultScope, cohort: Cohort,
) -> ResultScope | None:
    """Prefer an explicit nested result only when the broad row has no facts.

    Preserve distinct windows/cutoffs and conflicting or separately reported
    observations. An explicit nested zero can resolve an empty broad projection
    even when missing material keeps both scopes partial. Coverage flags remain
    unchanged, and any design denominator still uses the existing R3 rule.
    The original scopes and evidence are never changed.
    """
    if broad.dlt_results_coverage not in {"complete", "partial"} or broad.explicit_no_dlt_in_scope or broad.unresolved_positive_dlt:
        return None
    own_observations = [o for o in extraction.dlt_observations
                        if o.result_scope_id == broad.result_scope_id and o.cohort_id == cohort.cohort_id]
    if any(o.reporting_status not in {"not_reported", "not_applicable"}
           or o.reported_count is not None or o.explicit_patient_count is not None or o.toxicities_raw
           for o in own_observations):
        return None
    if any(c.result_scope_id == broad.result_scope_id and c.cohort_id == cohort.cohort_id
           and (c.reported_value is not None or c.reporting_status == "conflicting")
           for c in extraction.population_counts):
        return None
    candidates: list[ResultScope] = []
    for narrow in extraction.result_scopes:
        if not (set(narrow.cohort_ids) < set(broad.cohort_ids)) or cohort.cohort_id not in narrow.cohort_ids:
            continue
        if _scope_identity(narrow) != _scope_identity(broad) or narrow.unresolved_positive_dlt:
            continue
        if narrow.dlt_results_coverage != "complete" and not (
            narrow.dlt_results_coverage == "partial" and narrow.explicit_no_dlt_in_scope
        ):
            continue
        observations = [o for o in extraction.dlt_observations
                        if o.result_scope_id == narrow.result_scope_id and o.cohort_id == cohort.cohort_id]
        if any(o.study_id != narrow.study_id or o.window_id != narrow.window_id for o in observations):
            continue
        numerator, _, issues = _resolve_numerator(observations, narrow, cohort, {})
        if narrow.explicit_no_dlt_in_scope and numerator.value != 0:
            continue
        denominator, _, _, n_issues, blocked = _select_denominator(
            cohort=cohort, scope=narrow, observations=observations,
            counts=extraction.population_counts,
            analysis_roles={p.analysis_population_id: p.interpreted_role for p in extraction.analysis_populations},
            analysis_windows={p.analysis_population_id: p.window_id for p in extraction.analysis_populations},
            derived_counts={}, study_id=narrow.study_id, part_id=narrow.part_id,
            arm_id=narrow.arm_id, window_id=narrow.window_id,
        )
        if denominator.value is None and not blocked and not n_issues and narrow.explicit_no_dlt_in_scope:
            series = next((s for s in extraction.dose_series if s.series_id == cohort.series_id), None)
            design = next((d for d in extraction.designs if series is not None and d.design_id == series.design_id), None)
            denominator = _r3_denominator(cohort, design, numerator)
        if (not issues and not n_issues and not blocked and denominator.basis in {"reported", "design_inferred"}
            and denominator.value is not None and numerator.value is not None
            and numerator.basis in {"reported", "derived"} and 0 <= numerator.value <= denominator.value):
            candidates.append(narrow)
    return candidates[0] if len(candidates) == 1 else None


def _scope_coverage_assessments(
    extraction: ReportedExtraction,
) -> dict[str, ScopeCoverageAssessment]:
    dlt_coverage = [item for item in extraction.extraction_coverage if item.category == "dlt_results"]
    assessments: dict[str, ScopeCoverageAssessment] = {}
    for scope in extraction.result_scopes:
        reasons: list[str] = []
        source_ids: list[str] = []
        statuses = {item.status for item in dlt_coverage}
        for item in dlt_coverage:
            source_ids.extend(item.source_ids)
            if item.note:
                reasons.append(item.note)
        if "conflict" in statuses:
            status = "conflict"
        elif "material_not_covered" in statuses or extraction.material_coverage_complete is False:
            status = "missing"
            if extraction.material_coverage_complete is False:
                reasons.append("Programmatic request inventory contains DLT-relevant material gaps.")
        elif scope.dlt_results_coverage == "unread":
            status = "unreadable"
            reasons.append("The result scope reports unreadable DLT material.")
        elif scope.dlt_results_coverage in {"partial", "unclear"}:
            status = "unknown"
            reasons.append(f"DLT result coverage for this scope is {scope.dlt_results_coverage}.")
        elif extraction.material_coverage_complete is True and scope.dlt_results_coverage == "complete":
            status = "ready"
        else:
            status = "unknown"
            reasons.append("Programmatic material coverage has not been established.")
        assessments[scope.result_scope_id] = ScopeCoverageAssessment(
            result_scope_id=scope.result_scope_id,
            status=status,
            reasons=list(dict.fromkeys(reasons)),
            affected_source_ids=list(dict.fromkeys(source_ids)),
            basis="mixed" if dlt_coverage else "programmatic",
        )
    return assessments


def resolve_extraction(extraction: ReportedExtraction) -> ResolvedExtraction:
    """Return policy-resolved curve rows without changing reported source facts."""

    from .drug_ordering import apply_drug_ordering
    extraction = apply_drug_ordering(extraction)
    cohort_by_id = {item.cohort_id: item for item in extraction.cohorts}
    series_by_id = {item.series_id: item for item in extraction.dose_series}
    design_by_id = {item.design_id: item for item in extraction.designs}
    analysis_roles = {item.analysis_population_id: item.interpreted_role for item in extraction.analysis_populations}
    analysis_windows = {item.analysis_population_id: item.window_id for item in extraction.analysis_populations}
    observations_by_scope: dict[str, list[DLTObservation]] = {}
    for observation in extraction.dlt_observations:
        observations_by_scope.setdefault(observation.result_scope_id, []).append(observation)
    derived_counts, derived_observations, applications = _derive_equations(extraction)
    scope_coverage = _scope_coverage_assessments(extraction)
    rows: list[CurveRow] = []
    issues: list[Issue] = []
    known_regimens = {item.regimen_id for item in extraction.regimens}
    known_interventions = {item.intervention_id for item in extraction.interventions}
    known_components = {component.component_id for regimen in extraction.regimens for component in regimen.components}
    for series in extraction.dose_series:
        if series.regimen_id not in known_regimens:
            issues.append(
                _issue(
                    "warning",
                    "unknown_regimen_reference",
                    [series.series_id, series.regimen_id],
                    "Dose series references a regimen that is absent from the reported extraction",
                )
            )
        if series.varying_component_id not in known_interventions | known_components:
            issues.append(
                _issue(
                    "warning",
                    "unknown_varying_component_reference",
                    [series.series_id, series.varying_component_id],
                    "Dose series references a varying intervention that is absent from the reported extraction",
                )
            )
    for scope in extraction.result_scopes:
        scope_series = series_by_id.get(scope.series_id)
        if scope_series is None:
            issues.append(_issue("row_blocker", "missing_series", [scope.result_scope_id], "Result scope references an unknown dose series"))
            continue
        scope_observations = observations_by_scope.get(scope.result_scope_id, [])
        for cohort_id in scope.cohort_ids:
            cohort = cohort_by_id.get(cohort_id)
            if cohort is None:
                issues.append(_issue("row_blocker", "missing_cohort", [scope.result_scope_id, cohort_id], "Result scope references an unknown cohort"))
                continue
            if cohort.actual_status != "actual":
                continue
            specific = _specific_reported_scope(extraction, scope, cohort)
            if specific is not None:
                applications.append(_application(
                    "DETERMINISTIC_DERIVATION", cohort_id, "result_scope_id", specific.result_scope_id,
                    [scope.result_scope_id, specific.result_scope_id, cohort_id],
                    "Use the nested reported cohort result instead of inventing another row from its broad summary",
                ))
                continue
            series = series_by_id.get(cohort.series_id)
            if series is None:
                issues.append(
                    _issue(
                        "row_blocker",
                        "missing_cohort_series",
                        [scope.result_scope_id, cohort_id, cohort.series_id],
                        "Cohort references an unknown dose series",
                    )
                )
                continue
            if cohort.series_id != scope.series_id:
                issues.append(
                    _issue(
                        "warning",
                        "result_scope_spans_dose_series",
                        [scope.result_scope_id, cohort_id, scope.series_id, cohort.series_id],
                        "The result scope covers cohorts from more than one dose series; the cohort's own series identity was retained",
                    )
                )
            curve_row_id = _stable_id("curve", extraction.document_id, scope.result_scope_id, cohort_id)
            row_study_id = series.study_id
            row_part_id = series.part_id
            row_arm_id = series.arm_id
            row_window_id = scope.window_id or series.window_id
            assigned_observations = [item for item in scope_observations if item.cohort_id == cohort_id]
            incompatible_observations = [
                item for item in assigned_observations
                if item.study_id != row_study_id
                or (
                    item.window_id is not None
                    and row_window_id is not None
                    and item.window_id != row_window_id
                )
            ]
            cohort_observations = [item for item in assigned_observations if item not in incompatible_observations]
            observation_scope_issues: list[Issue] = []
            if scope.study_id != row_study_id or incompatible_observations:
                observation_scope_issues.append(
                    _issue(
                        "row_blocker",
                        "dlt_scope_mismatch",
                        [scope.result_scope_id, cohort_id, *[item.observation_id for item in incompatible_observations]],
                        "A DLT observation or result scope is assigned to a different study or observation window",
                        field="y",
                        source_ids=_source_ids([scope, *incompatible_observations]),
                    )
                )
            numerator, reported_dlt_count, numerator_issues = _resolve_numerator(
                cohort_observations, scope, cohort, derived_observations
            )
            numerator_issues = [*observation_scope_issues, *numerator_issues]
            issues.extend(numerator_issues)
            denominator, treated_n, evaluable_n, denominator_issues, denominator_blocked = _select_denominator(
                cohort=cohort,
                scope=scope,
                observations=cohort_observations,
                counts=extraction.population_counts,
                analysis_roles=analysis_roles,
                analysis_windows=analysis_windows,
                derived_counts=derived_counts,
                study_id=row_study_id,
                part_id=row_part_id,
                arm_id=row_arm_id,
                window_id=row_window_id,
            )
            issues.extend(denominator_issues)
            numerator_blocked = any(item.severity == "row_blocker" for item in numerator_issues)
            row_rule_ids: list[str] = list(numerator.dependencies or [])
            if numerator.basis == "inferred_patient_count" and numerator.value is not None:
                if "R1" not in row_rule_ids:
                    row_rule_ids.append("R1")
                applications.append(_application("R1", curve_row_id, "y", numerator.value, numerator.input_ids or [], numerator.reason or "R1"))
            elif "R1" in row_rule_ids and numerator.value is not None:
                applications.append(
                    _application(
                        "R1",
                        curve_row_id,
                        "y_dependency",
                        numerator.value,
                        numerator.input_ids or [],
                        "The deterministic numerator inherits an R1 patient-count interpretation from its inputs",
                    )
                )
            elif numerator.basis == "derived" and numerator.value == 0 and scope.explicit_no_dlt_in_scope:
                row_rule_ids.append("D0")
                applications.append(_application("D0", curve_row_id, "y", 0, numerator.input_ids or [], numerator.reason or "D0"))
            if denominator.value is None and not denominator_blocked:
                active_design = design_by_id.get(series.design_id or "")
                denominator = _r3_denominator(cohort, active_design, numerator)
                if denominator.value is not None:
                    row_rule_ids.append("R3")
                    applications.append(_application("R3", curve_row_id, "n", denominator.value, denominator.input_ids or [], denominator.reason or "R3"))
                    if (
                        active_design is not None
                        and active_design.normalized_type == "classic_3plus3"
                        and numerator.value is not None
                        and 4 <= numerator.value <= 6
                    ):
                        issues.append(
                            _issue(
                                "warning",
                                "classic_3plus3_protocol_deviation_candidate",
                                [cohort_id, active_design.design_id],
                                "The resolved numerator requires six patients but is unusual for a completed classic 3+3 cohort; retain the row and inspect replacements or deviations",
                            )
                        )
            elif denominator.value is not None:
                applications.append(
                    _application(
                        "DENOMINATOR_SELECTION", curve_row_id, "n", denominator.value,
                        denominator.input_ids or [], denominator.reason or "Selected denominator",
                    )
                )
            if scope.explicit_no_dlt_in_scope and any(
                value.value not in {None, 0}
                for value in (_reported_patient_value(item) for item in cohort_observations)
            ):
                issues.append(
                    _issue(
                        "warning",
                        "scope_zero_conflicts_with_positive_dlt",
                        [scope.result_scope_id, cohort_id],
                        "The scope is marked as explicitly no DLT but contains a positive cohort observation; the positive observation was retained",
                        field="y",
                        source_ids=_source_ids([scope, *cohort_observations]),
                    )
                )
            if numerator.value is None and not numerator_blocked and _r2_allowed(
                cohort=cohort,
                scope=scope,
                observations=cohort_observations,
                all_scope_observations=scope_observations,
                denominator=denominator,
                scope_coverage_status=scope_coverage[scope.result_scope_id].status,
            ):
                numerator = _Value(
                    0,
                    "imputed_zero",
                    input_ids=[scope.result_scope_id, cohort_id, *(denominator.input_ids or []),
                               *[o.observation_id for o in scope_observations if o.reporting_status == "reported"]],
                    reason="R2 imputes zero because this actual dose is unmentioned within a complete scope that reports DLT results at sibling doses",
                )
                row_rule_ids.append("R2")
                applications.append(_application("R2", curve_row_id, "y", 0, numerator.input_ids or [], numerator.reason or "R2"))

            exclusion_reasons: list[str] = []
            exposure_warnings = exposure_problems(cohort, series.varying_component_id)
            for warning in exposure_warnings:
                issues.append(_issue("warning", "cohort_exposure_conflict", [cohort_id], warning, field="exposures"))
            primary_exposures = [e for e in cohort.exposures if e.component_id == series.varying_component_id]
            if any(e.dose_status == "conflicting" or e.presence == "absent" for e in primary_exposures) or any(
                "disagrees with cohort primary dose" in warning for warning in exposure_warnings
            ):
                exclusion_reasons.append("primary drug exposure has an unresolved source conflict")
            dose_value, dose_unit, dose_note = _normalized_curve_dose(cohort)
            if dose_note:
                issues.append(
                    _issue(
                        "warning",
                        "dose_unit_normalization_mismatch",
                        [cohort_id],
                        dose_note,
                        field="dose",
                        source_ids=_source_ids([cohort]),
                    )
                )
            if dose_value is None or dose_value <= 0:
                exclusion_reasons.append("dose is not a resolved scalar")
                dose_value = None
            if not dose_unit:
                exclusion_reasons.append("dose unit is missing")
            if denominator.value is None:
                exclusion_reasons.append("no direct, derived, or design-inferred denominator")
            if numerator.value is None:
                exclusion_reasons.append("DLT patient numerator is unresolved")
            if denominator_blocked:
                exclusion_reasons.append("denominator has a source conflict or invalid value")
            if numerator_blocked:
                exclusion_reasons.append("DLT numerator has a source conflict or invalid value")
            if denominator.value is not None and numerator.value is not None and numerator.value > denominator.value:
                exclusion_reasons.append("DLT patient numerator exceeds denominator")
                issues.append(
                    _issue(
                        "row_blocker", "y_exceeds_n", [cohort_id, scope.result_scope_id],
                        f"Resolved y={numerator.value} exceeds n={denominator.value}", field="y",
                    )
                )
            included = not exclusion_reasons
            outcome_exclusions = [reason for reason in exclusion_reasons if any(
                word in reason.casefold() for word in ("denominator", "numerator", "not actual", "not used", "overlap", "not evaluable"))]
            outcome_usable = (not outcome_exclusions and denominator.value is not None
                              and numerator.value is not None and denominator.value > 0
                              and 0 <= numerator.value <= denominator.value)
            basis_map = {
                "reported": "reported",
                "derived": "derived",
                "inferred_patient_count": "inferred_patient_count",
                "imputed_zero": "imputed_zero",
                "missing": "missing",
            }
            denominator_basis = denominator.basis if denominator.basis in {"reported", "derived", "design_inferred"} else "missing"
            evidence_records: list[object] = [cohort, scope, series, *cohort_observations]
            evidence_records.extend(
                _counts_for_cohort(
                    extraction.population_counts,
                    cohort,
                    scope,
                    study_id=row_study_id,
                    part_id=row_part_id,
                    arm_id=row_arm_id,
                    window_id=row_window_id,
                    analysis_windows=analysis_windows,
                )
            )
            design = design_by_id.get(series.design_id or "")
            if design:
                evidence_records.append(design)
            rows.append(
                CurveRow(
                    curve_row_id=curve_row_id,
                    document_id=extraction.document_id,
                    study_id=row_study_id,
                    part_id=row_part_id,
                    arm_id=row_arm_id,
                    cohort_id=cohort_id,
                    series_id=series.series_id,
                    result_scope_id=scope.result_scope_id,
                    regimen_id=cohort.regimen_id or series.regimen_id,
                    varying_component_id=series.varying_component_id,
                    dose_value=float(dose_value) if dose_value is not None else None,
                    dose_unit=dose_unit or "unknown",
                    dose_raw=cohort.dose.raw_expression,
                    exposures=[item.model_copy(deep=True) for item in cohort.exposures],
                    exposure_coverage=cohort.exposure_coverage,
                    exposure_issues=exposure_warnings,
                    window_id=row_window_id,
                    n=denominator.value,
                    y=numerator.value,
                    rate=(numerator.value / denominator.value) if included and denominator.value else None,
                    reported_treated_n=treated_n,
                    reported_dlt_evaluable_n=evaluable_n,
                    reported_dlt_count=reported_dlt_count,
                    numerator_basis=basis_map[numerator.basis],
                    denominator_basis=denominator_basis,
                    denominator_role=denominator.role if denominator.role in {
                        "paired_dlt", "dlt_evaluable", "treated", "equivalent_treated", "design_assumed_cohort"
                    } else "unknown",
                    rule_ids=row_rule_ids,
                    design_candidate_ns=denominator.candidates or [],
                    included_in_default=included,
                    outcome_usable=outcome_usable,
                    outcome_exclusion_reasons=outcome_exclusions,
                    included_in_reported_only=included and not any(rule in {"R1", "R2", "R3"} for rule in row_rule_ids),
                    exclusion_reasons=exclusion_reasons,
                    evidence_source_ids=_source_ids(evidence_records),
                )
            )
    scoped_cohorts = {cohort_id for scope in extraction.result_scopes for cohort_id in scope.cohort_ids}
    for cohort in extraction.cohorts:
        if cohort.actual_status == "actual" and cohort.cohort_id not in scoped_cohorts:
            issues.append(
                _issue(
                    "warning", "actual_cohort_without_result_scope", [cohort.cohort_id],
                    "Actual dose cohort is preserved but has no DLT result scope, so no curve row was projected",
                )
            )
    return ResolvedExtraction(
        document_id=extraction.document_id,
        reported_extraction=extraction,
        curve_rows=rows,
        scope_coverage=list(scope_coverage.values()),
        rule_applications=applications,
        issues=issues,
        policy_version=POLICY_VERSION,
    )
