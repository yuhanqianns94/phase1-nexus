"""Method 3 reports observed marginal support without inventing exposures."""
import numpy as np
import pytest

from phase1_prediction.multidrug.contracts import (
    DrugSlot, Exposure, ModelScope, Observation, RegimenPoint, TargetDrug, TargetRegimen,
)
from phase1_prediction.multidrug.power_prior import _support


def support(exposures, coordinates, *, background=False, power=1.):
    mode = 'background' if background else 'numeric'
    reference = None if background else 1.
    unit = None if background else 'relative dose'
    scope = ModelScope(document_id='paper', scope_id='scope', label='A',
        drugs=[DrugSlot(model_drug_id='A', name='A', research_priority=1,
            research_role='primary', mode=mode, reference_dose=reference, unit=unit)],
        observations=[Observation(observation_id=str(i), cohort_id=str(i), y=0, n=10,
            exposures=[Exposure(drug_id='A', **exposure)]) for i, exposure in enumerate(exposures)])
    target = TargetRegimen(name='target', drugs=[TargetDrug(drug_id='A', name='A',
        research_priority=1, mode=mode, reference_dose=reference, unit=unit)],
        points=[RegimenPoint(level_id=str(i), doses={'A': None if background and value else value})
            for i, value in enumerate(coordinates)])
    prepared = {'sources': [{'power': power, 'scope': scope.model_dump()}]}
    return _support(prepared, target, np.asarray(coordinates)[:, None])


def assert_counts(rows, outside, missing):
    assert [row['outside_source_axes'] for row in rows] == outside
    assert [row['missing_source_axes'] for row in rows] == missing


def test_known_dose_allocations_supply_observed_marginal_range():
    rows = support([{'presence': 'present', 'dose_subgroups': [(50., 5), (200., 5)]}],
        [25., 50., 100., 200., 250.])
    assert_counts(rows, [1, 0, 0, 0, 1], [0] * 5)
    assert 'joint support not established' in rows[2]['support']


@pytest.mark.parametrize(('count', 'outside'), [(0, [0, 1]), (5, [0, 0]), (10, [1, 0])])
def test_known_mixed_counts_supply_only_observed_use_states(count, outside):
    rows = support([{'presence': 'mixed', 'exposed_n': count}], [0., 1.], background=True)
    assert_counts(rows, outside, [0, 0])


def test_known_mixed_numeric_exposure_supports_its_dose_and_absence():
    rows = support([{'presence': 'mixed', 'exposed_n': 5, 'dose': 100.}], [0., 50., 100., 200.])
    assert_counts(rows, [0, 1, 0, 1], [0] * 4)


@pytest.mark.parametrize('presence', ['unknown', 'mixed'])
def test_unknown_use_fraction_does_not_imply_observed_use_or_nonuse(presence):
    rows = support([{'presence': presence}], [0., 1.], background=True)
    assert_counts(rows, [0, 0], [1, 1])
    assert all('information incomplete' in row['support'] for row in rows)


@pytest.mark.parametrize('uncertain_dose', [{'candidates': [50., 200.]}, {'lower': 50., 'upper': 200.}, {}])
def test_uncertain_doses_are_not_treated_as_observed_support(uncertain_dose):
    rows = support([{'presence': 'present', **uncertain_dose}], [50., 100., 200.])
    assert_counts(rows, [0] * 3, [1] * 3)


def test_unknown_cohort_dose_prevents_definite_outside_label():
    rows = support([{'presence': 'present', 'dose': 100.}, {'presence': 'present'}], [100., 200.])
    assert_counts(rows, [0, 0], [0, 1])


def test_absent_only_source_does_not_claim_numeric_dose_information():
    rows = support([{'presence': 'absent'}], [0., 100.])
    assert_counts(rows, [0, 1], [0, 0])


def test_zero_power_source_does_not_supply_support():
    rows = support([{'presence': 'present', 'dose': 100.}], [100.], power=0.)
    assert_counts(rows, [0], [0])
    assert rows[0]['support'] == 'No historical outcome borrowing'
