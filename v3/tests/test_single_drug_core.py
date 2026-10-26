"""Single focal dose model: generative context, outcome preservation and priors."""
import copy

import jax
import numpy as np
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density
from scipy.special import expit, logsumexp
from scipy.stats import binom, halfnorm, norm

from phase1_prediction.multidrug.contracts import PriorSpec, SamplerSpec, TargetRegimen
from phase1_prediction.multidrug.fitting import fit_source, source_model
from phase1_prediction.multidrug.math import single_drug_logistic_risk
from phase1_prediction.multidrug.prior_presets import canonical_prior
from phase1_prediction.multidrug.single_drug import CONTEXT_ID, project_scope, project_target
from phase1_prediction.multidrug.store import source_fit_id
from test_multidrug_power_prior import article

BACKEND = 'single_drug_logistic'


def fixture(k=2):
    row, target = article(k=k)
    for point in target.points:
        for drug in target.drugs[1:]:
            point.doses[drug.drug_id] = 50.
    return row[0], target


def test_target_keeps_fixed_companions_and_explicit_absence():
    _, target = fixture(3)
    target.profile['single_drug_analysis'] = {'focal_drug_id': 'A'}
    snapshot = target.model_dump()
    projected = project_target(target)
    assert [d.drug_id for d in projected.drugs] == ['A', CONTEXT_ID]
    assert all(p.doses[CONTEXT_ID] is None for p in projected.points)
    assert projected.profile['single_drug_projection']['companion_doses'] == {'B': 50., 'C': 50.}
    assert target.model_dump() == snapshot
    target.points[0].doses['A'] = 0.
    assert project_target(target).points[0].doses['A'] == 0
    for point in target.points:
        point.doses.update(B=0., C=0.)
    assert all(p.doses[CONTEXT_ID] == 0 for p in project_target(target).points)
    target.points[-1].doses['B'] = 20.
    with pytest.raises(ValueError, match='fixed'):
        project_target(target)


def test_dose_unknown_focal_is_never_created_and_focal_selection_is_explicit():
    _, target = fixture()
    target.profile['single_drug_analysis'] = {'focal_drug_id': 'B'}
    # Fix A and vary B to exercise a non-first focal selection.
    for i, point in enumerate(target.points):
        point.doses['B'] = float(25*(i+1))
        point.doses['A'] = 50.
    assert project_target(target).drugs[0].drug_id == 'B'
    target.points[0].doses['B'] = None
    with pytest.raises(ValueError, match='numeric dose'):
        project_target(target)
    scope, _ = fixture()
    scope.drugs[0].mode = 'background'
    with pytest.raises(ValueError, match='numeric source'):
        project_scope(scope, 'A')


def test_scope_splits_background_dose_signatures_without_relabelling_outcomes():
    scope, _ = fixture()
    snapshot = scope.model_dump()
    scopes = project_scope(scope, 'A')
    assert len(scopes) == 3  # B is 50, 100 and 200 in these original cohorts.
    observed = [o for s in scopes for o in s.observations]
    assert sorted((o.cohort_id, o.y, o.n) for o in observed) == sorted((o.cohort_id, o.y, o.n) for o in scope.observations)
    assert sum(o.n for o in observed) == 30
    assert len({s.scope_id for s in scopes}) == 3
    assert all(o.exposures[1].presence == 'present' for o in observed)
    assert all(o.provenance['single_drug_projection']['whole_regimen_outcomes_retained'] for o in observed)
    assert scope.model_dump() == snapshot
    assert all(s.lineage['focal_source_id'] == 'A' for s in scopes)
    assert project_scope(scope, 'B')[0].scope_id != scopes[0].scope_id


def test_unknown_and_mixed_companions_keep_uncertainty_and_unsupported_joints_are_excluded():
    scope, _ = fixture()
    scope.observations[0].exposures[1].presence = 'unknown'
    scope.observations[1].exposures[1].presence = 'mixed'
    scope.observations[1].exposures[1].exposed_n = 4
    scope.observations[2].exposures[1].presence = 'absent'
    projected = project_scope(scope, 'A')
    contexts = {o.cohort_id: o.exposures[1] for s in projected for o in s.observations}
    assert contexts['cohort_0'].presence == 'unknown'
    assert contexts['cohort_1'].presence == 'mixed' and contexts['cohort_1'].exposed_n == 4
    assert contexts['cohort_2'].presence == 'absent'
    multiple, _ = fixture(3)
    for exposure in multiple.observations[0].exposures[1:]:
        exposure.presence = 'unknown'
    rejected = []
    parts = project_scope(multiple, 'A', exclusions=rejected)
    assert len(rejected) == 1 and 'Joint use allocation' in rejected[0]['reason']
    assert sum(len(s.observations) for s in parts) == 2


def test_duplicate_cohorts_are_rejected_before_projection():
    scope, _ = fixture()
    scope.observations.append(scope.observations[0].model_copy(deep=True))
    with pytest.raises(ValueError, match='same patients twice'):
        project_scope(scope, 'A')


def test_formula_has_zero_dose_baseline_monotonic_focal_curve_and_nonnegative_context():
    r = np.array([0., .5, 1., 2.])
    mono = np.asarray(single_drug_logistic_risk(-3., np.log(.7), r, .4, 0))
    combo = np.asarray(single_drug_logistic_risk(-3., np.log(.7), r, .4, 1))
    np.testing.assert_allclose(mono, expit(-3.+.7*r))
    np.testing.assert_allclose(combo, expit(-3.+.7*r+.4))
    assert mono[0] == pytest.approx(expit(-3.))
    assert np.all(np.diff(mono) > 0) and np.all(combo > mono)


def test_source_likelihood_contains_context_shift_before_observing_outcomes():
    scope, _ = fixture()
    for observation in scope.observations:
        observation.exposures[1].dose = 50.
    source = project_scope(scope, 'A')[0]
    prior = PriorSpec(intercept_mean=-3., intercept_sd=1.2, log_slope_mean=-.2,
                      log_slope_sd=.4, combination_sd=.6)
    fixed = {'alpha': np.array(-2.5), 'ell': np.array(-.3), 'delta': np.array(.4)}
    actual, trace = log_density(source_model, (source, BACKEND, prior), {}, fixed)
    expected = norm.logpdf(-2.5, -3., 1.2)+norm.logpdf(-.3, -.2, .4)+halfnorm.logpdf(.4, scale=.6)
    for observation in source.observations:
        r = observation.exposures[0].dose / source.drugs[0].reference_dose
        expected += binom.logpmf(observation.y, observation.n, expit(-2.5+np.exp(-.3)*r+.4))
    assert float(actual) == pytest.approx(expected)
    assert trace['eta']['value'].shape == (0,)
    np.testing.assert_allclose(trace['log_b']['value'], [-.3, 0.])
    assert not any(name.startswith('a[') for name in trace)


def test_unknown_context_is_marginalized_as_cohort_uncertainty():
    scope, _ = fixture()
    scope.observations = scope.observations[:1]
    scope.observations[0].exposures[1].presence = 'unknown'
    source = project_scope(scope, 'A')[0]
    fixed = {'alpha': -3., 'ell': -.2, 'delta': .7}
    trace = handlers.trace(handlers.substitute(source_model, data=fixed)).get_trace(source, BACKEND, PriorSpec())
    o = source.observations[0]
    risk = expit(-3.+np.exp(-.2)*o.exposures[0].dose/source.drugs[0].reference_dose+np.array([0., .7]))
    expected = logsumexp(np.log(.5)+binom.logpmf(o.y, o.n, risk))
    assert float(trace['cohort_0']['fn'].log_factor) == pytest.approx(expected)


def test_prior_arrays_have_one_slope_no_interaction_and_distinct_cache_identity():
    scope, _ = fixture()
    source = project_scope(scope, 'A')[0]
    source.observations = []
    sampler = SamplerSpec(warmup=1, samples=4)
    arrays, diagnostics = fit_source(source, BACKEND, PriorSpec(), sampler)
    assert diagnostics['status'] == 'prior_only'
    assert arrays['alpha'].shape == arrays['ell'].shape == arrays['delta'].shape == (2, 4)
    assert arrays['eta'].shape == (2, 4, 0)
    assert arrays['log_b'].shape == (2, 4, 2)
    assert (arrays['delta'] >= 0).all()
    np.testing.assert_allclose(arrays['log_b'][..., 1], 0)
    assert source_fit_id(source, BACKEND, PriorSpec(), sampler) != source_fit_id(source, 'logistic', PriorSpec(), sampler)
    old = canonical_prior({'pair_sd': 9.}, BACKEND)
    assert old == canonical_prior({}, BACKEND)
    assert canonical_prior({'combination_sd': .9}, BACKEND) != old


def test_mixed_companion_keeps_aggregate_dlt_likelihood_without_assigning_events():
    scope, _ = fixture()
    scope.observations = scope.observations[:1]
    scope.observations[0].exposures[1].presence = 'mixed'
    scope.observations[0].exposures[1].exposed_n = 4
    source = project_scope(scope, 'A')[0]
    trace = handlers.trace(handlers.substitute(source_model, data={'alpha': -3., 'ell': -.2, 'delta': .7})).get_trace(
        source, BACKEND, PriorSpec())
    o = source.observations[0]
    risk = expit(-3.+np.exp(-.2)*o.exposures[0].dose/source.drugs[0].reference_dose+np.array([0., .7]))
    probability = sum(binom.pmf(k, 4, risk[1])*binom.pmf(o.y-k, 6, risk[0]) for k in range(o.y+1))
    assert float(trace['cohort_0']['fn'].log_factor) == pytest.approx(np.log(probability))


def test_direct_source_cache_projects_all_reported_numeric_axes(tmp_path, monkeypatch):
    from phase1_prediction.multidrug import prior_presets, service
    scope, _ = fixture()
    status = prior_presets.source_cache_status([scope], backend=BACKEND, store_root=tmp_path)
    assert status['counts']['missing'] == 6
    assert not status['projection_notices']
    captured = []
    def fitting(scopes, **kwargs):
        captured.extend(scopes)
        assert kwargs['backends'] == (BACKEND,)
        return [], []
    monkeypatch.setattr(service, 'fit_library', fitting)
    prepared = prior_presets.prepare_source_cache([scope], backend=BACKEND, store_root=tmp_path)
    assert len(captured) == 6
    assert {source.lineage['focal_source_id'] for source in captured} == {'A', 'B'}
    assert prepared['counts']['missing'] == 6
    assert not prepared['errors']
