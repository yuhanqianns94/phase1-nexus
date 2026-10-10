"""Method 3: target-only, fixed-power historical likelihood with heterogeneity.

Article-local fits are neither read nor overwritten. Each selected article
contributes one complete, compatible scope and each cohort contributes once.
"""
from __future__ import annotations

import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import Predictive
from pydantic import Field

from .contracts import Contract, ModelScope, PriorSpec, SamplerSpec, fingerprint
from .fitting import sample_curve_parameters, curve_risk_function, observe_scope, run_nuts
from . import validate_backend
from .mapping import mapped_relative_dose
from .math import component_risks, logistic_risk, product_logistic_risk
from .transport import _base_result, _summary, flat_samples
from .workflow import relevance_multiplier

METHOD = 'target_power_prior'
POWER_PRIOR_VERSION = 'fixed_partial_power_prior_v3'


class PowerPriorSpec(Contract):
    """Fixed discounting controls, not probabilities of article correctness."""
    strength: float = Field(default=1., ge=0, le=1)
    patient_budget: float | None = Field(default=30., gt=0)
    heterogeneity_scale: float = Field(default=.5, gt=0)


def target_signature(target):
    """Freeze clinical matching and axis meaning, not evaluated coordinates.

    Relative analogy may acquire actual target units/references for BOIN. That
    changes the displayed scale, not the already fitted dimensionless model.
    Same-entity transport references are part of the fitted coordinate system.
    """
    profile = copy.deepcopy(target.profile)
    profile.pop('planning', None)
    drugs = []
    for drug in target.drugs:
        item = drug.model_dump(mode='json')
        if drug.transport_mode == 'relative_dose_analogy':
            item.pop('unit', None)
            item.pop('reference_dose', None)
        drugs.append(item)
    return {'drugs': drugs, 'profile': profile,
            'mapping_overrides': target.mapping_overrides,
            'minimum_drug_similarity': target.minimum_drug_similarity}


def _mapped_scope(scope, target, mapping):
    if mapping['unmatched_target_ids'] or mapping['removed_source_ids']:
        raise ValueError('Method 3 requires complete one-to-one regimen correspondence; no source drug may be removed.')
    by_id = {d.model_drug_id: d for d in scope.drugs}
    drugs, scales = [], {}
    for td in target.drugs:
        sd = by_id[mapping['pairs'][td.drug_id]]
        if sd.mode != td.mode:
            raise ValueError('Method 3 requires corresponding numeric or dose-unknown axes; background-to-numeric anchoring is not supported.')
        if td.mode == 'numeric':
            # Source actual dose / scale is its coordinate in the target model.
            # Analogy -> source reference. Same entity -> target reference in source units.
            scale = sd.reference_dose * mapped_relative_dose(td, sd, td.reference_dose)
        else:
            scale = 1.
        scales[sd.model_drug_id] = scale
        drug = sd.model_copy(deep=True)
        drug.model_drug_id = td.drug_id
        drug.name = td.name
        drug.research_priority = td.research_priority
        drug.research_role = td.research_role
        drug.reference_dose = 1. if td.mode == 'numeric' else None
        drug.unit = 'relative dose' if td.mode == 'numeric' else None
        drugs.append(drug)
    observations, cohorts, rows = [], set(), set()
    for original in scope.observations:
        if original.cohort_id in cohorts or rows.intersection(original.source_row_ids):
            raise ValueError('Overlapping cohort identities within this scope; Method 3 will not count patients twice.')
        cohorts.add(original.cohort_id)
        rows.update(original.source_row_ids)
        exposures = {e.drug_id: e for e in original.exposures}
        mapped = []
        for td in target.drugs:
            source_id = mapping['pairs'][td.drug_id]
            exposure = exposures[source_id].model_copy(deep=True)
            scale = scales[source_id]
            exposure.drug_id = td.drug_id
            for field in ('dose', 'lower', 'upper'):
                value = getattr(exposure, field)
                if value is not None:
                    setattr(exposure, field, value / scale)
            exposure.candidates = [x / scale for x in exposure.candidates]
            exposure.dose_subgroups = [(x / scale, n) for x, n in exposure.dose_subgroups]
            if td.mode == 'numeric':
                # r_target = r_source * source_reference / scale. Preserve the
                # latent physical-dose prior as well as its reported bounds.
                exposure.log_relative_dose_prior_mean += float(np.log(
                    by_id[source_id].reference_dose / scale))
            mapped.append(exposure)
        observation = original.model_copy(deep=True)
        observation.exposures = mapped
        observations.append(observation)
    return ModelScope(document_id=scope.document_id, scope_id=scope.scope_id, label=scope.label,
                      drugs=drugs, observations=observations)


def prepare_power_prior(scored, target, spec, relevance_decisions=None):
    """Select one strongest applicable scope per article, before relevance review.

    The patient cap is fixed before applying Include/Uncertain/Exclude. Removed
    information is not redistributed to the remaining articles.
    """
    by_document, excluded = {}, []
    for scope, mapping, strength in scored:
        try:
            if not scope.observations:
                raise ValueError('No legal DLT/N observations.')
            if not np.isfinite(strength) or not 0 < strength <= 1:
                raise ValueError('No positive similarity support.')
            mapped = _mapped_scope(scope, target, mapping)
        except ValueError as error:
            excluded.append({'document_id': scope.document_id, 'scope_id': scope.scope_id,
                             'reason': str(error)})
            continue
        item = {'document_id': scope.document_id, 'scope_id': scope.scope_id,
                'scope': mapped.model_dump(mode='json'),
                'source_scope': scope.model_dump(mode='json'), 'mapping': mapping,
                'strength': float(strength), 'patient_count': sum(o.n for o in mapped.observations)}
        by_document.setdefault(scope.document_id, []).append(item)
    selected = []
    for doc in sorted(by_document):
        candidates = sorted(by_document[doc], key=lambda x: (-x['strength'], x['scope_id']))
        selected.append(candidates[0])
        excluded.extend({'document_id': doc, 'scope_id': c['scope_id'],
                         'reason': 'Another complete scope from this article has higher priority; one scope per article in Method 3.'}
                        for c in candidates[1:])
    base_patients = sum(c['strength'] * c['patient_count'] for c in selected)
    cap = min(1., spec.patient_budget / base_patients) if spec.patient_budget is not None and base_patients else 1.
    for item in selected:
        item['review_multiplier'] = relevance_multiplier(item['document_id'], relevance_decisions)
        item['power'] = spec.strength * cap * item['strength'] * item['review_multiplier']
    return {'version': POWER_PRIOR_VERSION, 'target_signature': target_signature(target),
            'spec': spec.model_dump(), 'sources': selected, 'excluded': excluded,
            'cap_factor': cap, 'pre_review_weighted_patients': base_patients,
            'discounted_patient_count': sum(c['power'] * c['patient_count'] for c in selected)}


def power_prior_model(scopes, powers, backend, prior, spec):
    """Partial discounting: only outcome factors are powered, once per cohort."""
    parameters = sample_curve_parameters(scopes[0].drugs, backend, prior)
    tau = numpyro.sample('tau', dist.HalfNormal(spec.heterogeneity_scale))
    active = [(i, scope, power) for i, (scope, power) in enumerate(zip(scopes, powers)) if power > 0]
    if active:
        z = numpyro.sample('study_z', dist.Normal(0., 1.).expand((len(active),)).to_event(1))
        for j, (i, scope, power) in enumerate(active):
            a, log_b, eta = parameters
            shifted = (a + tau * z[j], log_b, eta)
            risk = curve_risk_function(scope.drugs, backend, shifted)
            with handlers.scope(prefix=f'article_{i}'):
                observe_scope(scope, risk, prior, likelihood_power=power)


def fit_power_prior(prepared, backend, prior, sampler):
    scopes = [ModelScope.model_validate(c['scope']) for c in prepared['sources']]
    powers = [c['power'] for c in prepared['sources']]
    spec = PowerPriorSpec.model_validate(prepared['spec'])
    args = (scopes, powers, backend, prior, spec)
    if not any(powers):
        samples = Predictive(power_prior_model, num_samples=sampler.chains * sampler.samples)(
            jax.random.PRNGKey(sampler.seed), *args)
        arrays = {key: np.asarray(value).reshape((sampler.chains, sampler.samples, *value.shape[1:]))
                  for key, value in samples.items()}
        return arrays, {'status': 'fitted', 'inference': 'explicit_prior_predictive',
                        'attempts': [], 'reason': 'All historical powers are zero; no outcome information is borrowed.'}
    arrays, diagnostics = run_nuts(power_prior_model, args, sampler)
    return arrays, {**diagnostics, 'inference': 'historical_power_prior_predictive'}


def _target_coordinates(target):
    values = []
    for point in target.points:
        row = []
        for drug in target.drugs:
            dose = point.doses[drug.drug_id]
            if drug.mode == 'numeric':
                if dose is None:
                    raise ValueError('Method 3 needs an explicit evaluation coordinate for each numeric axis. Use relative-dose exploration if planned doses are unknown.')
                row.append(dose / drug.reference_dose)
            else:
                row.append(0. if dose == 0 else 1.)
        values.append(row)
    return np.asarray(values)


def _observed_axis_support(scope, axis):
    """Observed marginal support; latent candidates are not observed patients."""
    states, doses = set(), []
    unknown_use = unknown_dose = False
    for observation in scope.observations:
        exposure = observation.exposures[axis]
        if exposure.presence == 'absent':
            states.add('absent')
            continue
        if exposure.presence == 'present':
            used = True
            states.add('present')
        elif exposure.presence == 'mixed' and exposure.exposed_n is not None:
            used = exposure.exposed_n > 0
            if used:
                states.add('present')
            if exposure.exposed_n < observation.n:
                states.add('absent')
        elif exposure.presence == 'mixed' and exposure.dose_subgroups:
            # Positive-dose allocations partition the entire evaluable cohort.
            used = True
            states.add('present')
        else:
            unknown_use = True
            continue
        if not used:
            continue
        if exposure.dose_subgroups:
            doses.extend(dose for dose, count in exposure.dose_subgroups if count > 0)
        elif exposure.dose is not None:
            doses.append(exposure.dose)
        else:
            # A candidate list or bounds describe uncertainty, not actual doses.
            unknown_dose = True
    return states, doses, unknown_use, unknown_dose


def _support(prepared, target, coordinates):
    rows = []
    active = [c for c in prepared['sources'] if c['power'] > 0]
    for i, point in enumerate(target.points):
        outside = missing = 0
        for item in active:
            scope = ModelScope.model_validate(item['scope'])
            for j, drug in enumerate(scope.drugs):
                states, doses, unknown_use, unknown_dose = _observed_axis_support(scope, j)
                r = coordinates[i, j]
                if drug.mode == 'background' or r == 0:
                    expected = 'absent' if r == 0 else 'present'
                    if expected not in states:
                        if unknown_use:
                            missing += 1
                        else:
                            outside += 1
                    continue
                if not doses:
                    if 'present' in states or unknown_use:
                        missing += 1
                    else:
                        outside += 1
                elif r < min(doses) - 1e-9 or r > max(doses) + 1e-9:
                    if unknown_use or unknown_dose:
                        missing += 1
                    else:
                        outside += 1
        status = ('No historical outcome borrowing' if not active else
                  'Outside at least one source dose/use-state range' if outside else
                  'Source dose/use-state information incomplete' if missing else
                  'Within marginal dose/use-state ranges; joint support not established')
        rows.append({'level_id': point.level_id, 'support': status,
                     'outside_source_axes': int(outside), 'missing_source_axes': missing})
    return rows


def evaluate_power_prior(prepared, arrays, manifest, target, *, draw_count, seed):
    if target_signature(target) != prepared['target_signature']:
        raise ValueError('The new target changes the frozen Power prior model or evidence context. Generate a new prediction.')
    r = _target_coordinates(target)
    backend = manifest['backend']
    result = _base_result(target, backend, METHOD, seed)
    result.update(target_fit_id=manifest['target_fit_id'], power_prior=prepared['spec'],
                  discounted_patient_count=prepared['discounted_patient_count'],
                  cap_factor=prepared['cap_factor'], inference=manifest.get('inference'),
                  sampling_diagnostics={k: manifest.get(k) for k in ('status', 'attempts', 'reason', 'elapsed_seconds')},
                  excluded=copy.deepcopy(prepared['excluded']))
    result['contributors'] = [{k: copy.deepcopy(c[k]) for k in
        ('document_id', 'scope_id', 'mapping', 'strength', 'patient_count', 'review_multiplier', 'power')}
        for c in prepared['sources']]
    result['coordinate_support'] = _support(prepared, target, r)
    if manifest['status'] != 'fitted':
        result['status'] = manifest['status']
        return result
    flat = flat_samples(arrays)
    rng = np.random.default_rng(seed)
    intercept_key = 'alpha' if backend == 'logistic_product' else 'a'
    ids = rng.integers(0, len(flat[intercept_key]), draw_count)
    target_z = rng.normal(size=draw_count)
    u_target = flat['tau'][ids] * target_z
    a = (flat['alpha'][ids] + u_target if backend == 'logistic_product'
         else flat['a'][ids] + u_target[:, None])
    log_b = flat['log_b'][ids]
    numeric = jnp.array([d.mode == 'numeric' for d in target.drugs])
    doses = jnp.asarray(r)
    def one(intercepts, slopes, eta):
        if backend == 'logistic_product':
            return product_logistic_risk(intercepts, slopes, doses, eta)
        q = component_risks(intercepts, slopes, doses, numeric, doses > 0)
        return logistic_risk(q, doses, eta)
    values = jax.jit(jax.vmap(one))(a, log_b, flat['eta'][ids])
    matrix = np.asarray(values)
    if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError('Invalid Power prior predictive risks; no fallback used.')
    result.update(status='fitted', posterior_draws=matrix.tolist(), summary=_summary(matrix, target),
                  provenance=[{'target_fit_id': manifest['target_fit_id'], 'flat_draw': int(index),
                               'target_random_intercept': float(u)} for index, u in zip(ids, u_target)],
                  aggregation_note='Target prior-predictive curves from discounted historical likelihoods. One new-study intercept per whole curve; source article posteriors are unchanged.')
    result['prediction_id'] = fingerprint(result)
    return result


class PowerPriorStore:
    """Separate, locked, checksummed target cache; never writes source fits."""
    def __init__(self, source_store_root):
        self.root = Path(source_store_root) / 'target_power_priors'

    @staticmethod
    def identity(prepared, backend, prior, sampler):
        import hashlib
        from .store import numerical_dependencies
        from .prior_presets import canonical_prior
        prior = canonical_prior(prior, backend)
        here = Path(__file__).parent
        engine = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in
                  ('power_prior.py', 'fitting.py', 'math.py', 'contracts.py', 'mapping.py')}
        return 'power_' + fingerprint({'prepared': prepared, 'backend': backend, 'prior': prior.model_dump(),
                                      'sampler': sampler.model_dump(), 'engine': engine,
                                      'dependencies': numerical_dependencies()})[:32]

    def load(self, fit_id):
        import hashlib
        import json
        if not fit_id.startswith('power_') or Path(fit_id).name != fit_id:
            raise ValueError('Invalid Power prior cache identity')
        directory = self.root / fit_id
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['target_fit_id'] != fit_id or manifest['version'] != POWER_PRIOR_VERSION:
            raise ValueError('Incompatible Power prior cache')
        validate_backend(manifest['backend'])
        for name in ('inputs.json', 'draws.npz'):
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest['checksums'][name]:
                raise ValueError('Power prior cache checksum mismatch')
        prepared = json.loads((directory / 'inputs.json').read_text())
        with np.load(directory / 'draws.npz', allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        return prepared, arrays, manifest

    def fit(self, prepared, backend, prior, sampler, progress=None):
        import fcntl
        import hashlib
        import json
        import shutil
        import tempfile
        from datetime import datetime, timezone
        from .prior_presets import canonical_prior
        prior = canonical_prior(prior, backend)
        fit_id = self.identity(prepared, backend, prior, sampler)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / (fit_id + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            final = self.root / fit_id
            if (final / 'manifest.json').is_file():
                return self.load(fit_id)
            if progress:
                progress(0, 1, 'Method 3 target model', backend)
            arrays, diagnostics = fit_power_prior(prepared, backend, prior, sampler)
            staging = Path(tempfile.mkdtemp(prefix='.writing_', dir=self.root))
            try:
                (staging / 'inputs.json').write_text(json.dumps(prepared, ensure_ascii=False, allow_nan=False))
                np.savez_compressed(staging / 'draws.npz', **arrays)
                manifest = {**diagnostics, 'target_fit_id': fit_id, 'version': POWER_PRIOR_VERSION,
                            'backend': backend, 'prior': prior.model_dump(), 'sampler': sampler.model_dump(),
                            'created_at_utc': datetime.now(timezone.utc).isoformat(),
                            'checksums': {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                                          for name in ('inputs.json', 'draws.npz')}}
                (staging / 'manifest.json').write_text(json.dumps(manifest, allow_nan=False, indent=2))
                staging.rename(final)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            return self.load(fit_id)


def predict_power_prior(scored, target, *, backend, prior, sampler, spec, store_root,
                        draw_count, seed, relevance_decisions=None, progress=None):
    _target_coordinates(target)  # Fail before fitting if evaluation inputs are incomplete.
    prepared = prepare_power_prior(scored, target, spec, relevance_decisions)
    if not prepared['sources'] or not any(c['review_multiplier'] > 0 for c in prepared['sources']):
        result = _base_result(target, backend, METHOD, seed)
        result.update(excluded=prepared['excluded'], power_prior=spec.model_dump(), coordinate_support=[])
        return result
    prepared, arrays, manifest = PowerPriorStore(store_root).fit(prepared, backend, prior, sampler, progress)
    return evaluate_power_prior(prepared, arrays, manifest, target, draw_count=draw_count, seed=seed)
