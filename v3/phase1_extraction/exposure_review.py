"""Round-trip editable cohort exposures without silently losing source fields."""
from __future__ import annotations
import math
from typing import Any
from .models import Cohort, CohortExposure, DoseExpression
from .exposures import dose_text, exposure_problems, exposure_schedule


def editor_rows(exposures: list[CohortExposure]) -> list[dict[str, Any]]:
    return [{
        'Record': e.exposure_id, 'Drug / class': e.name_raw, 'Type': e.entity_kind,
        'Role': e.role, 'Use': e.presence, 'Dose status': e.dose_status,
        'Research order': e.research_priority, 'Research role': e.research_role,
        'Dose interpretation': e.dose_semantics,
        'Dose value': e.dose.value if e.dose and e.dose.value_kind == 'scalar' else None,
        'Unit': (e.dose.unit_normalized or e.dose.unit_raw or '') if e.dose else '',
        'Source dose': e.dose.raw_expression if e.dose else '',
        'Formulation': e.formulation_raw or '', 'Schedule': exposure_schedule(e),
        'Reporting': e.reporting_scope, 'Exposed N': e.exposed_n, 'Note': e.note or '',
    } for e in exposures]


def _text(value: Any) -> str:
    return '' if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()


def parse_editor_rows(rows: Any, original: list[CohortExposure]) -> list[CohortExposure]:
    rows = rows.to_dict(orient='records') if hasattr(rows, 'to_dict') else rows
    original_by_id = {e.exposure_id: e for e in original}
    original_rows = {r['Record']: r for r in editor_rows(original)}
    result = []
    for i, row in enumerate(rows):
        name = _text(row.get('Drug / class'))
        if not name:
            defaults = {'Type': 'drug', 'Role': 'escalated', 'Use': 'unknown', 'Dose status': 'unknown', 'Reporting': 'cohort'}
            if any(_text(v) and _text(v) != defaults.get(k) for k, v in row.items() if k != 'Record'):
                raise ValueError('Each exposure row needs a drug or class name; delete an unused row.')
            continue
        record = _text(row.get('Record')) or f'human_exposure_{i + 1}'
        old = original_by_id.get(record)
        e = old.model_copy(deep=True) if old else CohortExposure(exposure_id=record, name_raw=name)
        previous = original_rows.get(record, {})
        e.name_raw = name
        e.entity_kind = _text(row.get('Type')) or 'drug'
        if old and (name != old.name_raw or e.entity_kind != old.entity_kind):
            # A replacement drug must not inherit another entity's references.
            e.intervention_id = None
            e.component_id = None
        e.role = _text(row.get('Role')) or 'unclear'
        if 'Research order' in row:
            order = _text(row.get('Research order'))
            number = float(order) if order else None
            if number is not None and (not math.isfinite(number) or number < 1 or not number.is_integer()):
                raise ValueError(f'{name}: research order must be a positive integer')
            e.research_priority = int(number) if number else None
            e.research_role = _text(row.get('Research role')) or 'unclear'
            if order != _text(previous.get('Research order')) or e.research_role != previous.get('Research role', 'unclear'):
                e.priority_basis = 'human_confirmed'
                e.priority_reason = 'Confirmed in the exposure review editor'
        e.presence = _text(row.get('Use')) or 'unknown'
        e.dose_status = _text(row.get('Dose status')) or 'unknown'
        e.dose_semantics = _text(row.get('Dose interpretation')) or e.dose_semantics
        e.reporting_scope = _text(row.get('Reporting')) or 'cohort'
        e.formulation_raw = _text(row.get('Formulation')) or None
        e.note = _text(row.get('Note')) or None
        exposed = _text(row.get('Exposed N'))
        if exposed:
            numeric = float(exposed)
            if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                raise ValueError(f'{name}: exposed N must be a nonnegative integer')
            e.exposed_n = int(numeric)
        else:
            e.exposed_n = None
        dose_columns = ('Dose value', 'Unit', 'Source dose')
        changed = any(_text(row.get(k)) != _text(previous.get(k)) for k in dose_columns)
        if changed or not old:
            e.dose_subgroups = []
            value, unit, raw = (_text(row.get(k)) for k in dose_columns)
            if value:
                number = float(value)
                if not math.isfinite(number) or number <= 0:
                    raise ValueError(f'{name}: dose must be positive; use absent instead of zero')
                # A changed numerical dose must not retain the previous raw dose.
                if old and (value != _text(previous.get('Dose value')) or unit != _text(previous.get('Unit'))) and raw == _text(previous.get('Source dose')):
                    raw = f'{number:g} {unit}'.strip()
                e.dose = DoseExpression(raw_expression=raw or f'{number:g} {unit}'.strip(),
                    value_kind='scalar', value=number, unit_raw=unit or None, unit_normalized=unit or None)
            elif raw:
                e.dose = DoseExpression(raw_expression=raw, value_kind='unparsed', unit_raw=unit or None)
            else:
                e.dose = None
        schedule = _text(row.get('Schedule'))
        if schedule != _text(previous.get('Schedule')):
            e.schedule_raw = schedule or None
            e.dosing_phases = []  # Human replacement must not leave contradictory structured phases.
        result.append(CohortExposure.model_validate(e.model_dump()))
    probe = Cohort(cohort_id='review', series_id='review', label='review', actual_status='actual',
                   regimen_id='review', dose=DoseExpression(raw_expression='', value_kind='unparsed'), exposures=result)
    problems = exposure_problems(probe)
    if problems:
        raise ValueError('; '.join(problems))
    return result
