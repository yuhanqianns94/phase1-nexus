"""Equivalent unit notation must not change dose values or dimensions."""

import pytest

from phase1_extraction.models import Cohort, DoseExpression
from phase1_extraction.resolver import _normalized_curve_dose


def cohort(raw_unit, normalized_unit, value=2.5):
    return Cohort(
        cohort_id="cohort", series_id="series", regimen_id="regimen", label="Dose cohort",
        actual_status="actual", cohort_role="escalation",
        dose=DoseExpression(
            raw_expression=f"{value} {raw_unit}", value_kind="scalar", value=value,
            unit_raw=raw_unit, unit_normalized=normalized_unit,
        ),
    )


@pytest.mark.parametrize("raw,normalized", [
    ("mg kg−1", "mg/kg"),
    ("mg kg-1", "mg/kg"),
    ("mg kg^-1", "mg/kg"),
    ("mg kg⁻¹", "mg/kg"),
    ("mg·kg⁻¹", "mg/kg"),
    ("mg⋅kg−1", "mg/kg"),
    ("mg*kg^-1", "mg/kg"),
    ("mg\u00a0kg⁻¹", "mg/kg"),
    ("mg m−2", "mg/m2"),
    ("mg m^-2", "mg/m²"),
    ("mg m⁻²", "mg/m2"),
    ("mg m−2 day−1", "mg/m2/day"),
    ("mg·m⁻²·day⁻¹", "mg/m²/day"),
    ("mg kg−1/day", "mg/kg/day"),
    ("mg kg⁻¹ h⁻¹", "mg/kg/h"),
    ("mg/kg", "mg kg⁻¹"),
])
def test_reciprocal_notation_retains_the_same_dose(raw, normalized):
    source = cohort(raw, normalized)
    before = source.model_dump()
    assert _normalized_curve_dose(source) == (2.5, normalized, None)
    assert source.model_dump() == before


@pytest.mark.parametrize("raw,normalized,expected", [
    ("µg kg⁻¹", "mg/kg", 0.0025),
    ("g m−2", "mg/m2", 2500),
    ("mg/kg", "µg kg⁻¹", 2500),
])
def test_mass_scaling_is_preserved_for_equivalent_reciprocal_units(raw, normalized, expected):
    assert _normalized_curve_dose(cohort(raw, normalized)) == (expected, normalized, None)


@pytest.mark.parametrize("raw,normalized", [
    ("mg kg−1", "mg/m2"),
    ("mg m−2", "mg/kg"),
    ("mg kg−1", "mg"),
    ("mg kg−1 day−1", "mg/kg"),
    ("mg kg−1", "mg/kg/day"),
    ("mg kg−1 day−1", "mg/kg/h"),
    ("mg kg+1", "mg/kg"),
    ("mg m²", "mg/m2"),
    ("mg kg−2", "mg/kg"),
    ("mg m−1", "mg/m2"),
    ("mg kg−10", "mg/kg"),
    ("mg/kg−1", "mg/kg"),
])
def test_different_dimensions_and_unsupported_exponents_retain_source_units(raw, normalized):
    value, unit, note = _normalized_curve_dose(cohort(raw, normalized))
    assert (value, unit) == (2.5, raw)
    assert "not dimensionally compatible" in note
