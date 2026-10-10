"""Named sensitivity choices and reusable posterior-cache preparation.

Preset names describe parameter regularization, not expected clinical toxicity.
The numerical specification, rather than its display label, identifies a prior.
"""
from __future__ import annotations

from .contracts import PriorSpec, SamplerSpec, fingerprint
from . import validate_backend


RISK_PRESETS = {
    'balanced': {
        'label': 'Balanced (recommended / default)',
        'description': 'The existing default prior. A starting point for sensitivity comparisons, not a clinically calibrated toxicity prior.',
        'values': {},
    },
    'regularized': {
        'label': 'More regularized',
        'description': 'Smaller parameter variation and interactions.',
        'values': {'intercept_sd': 1.5, 'log_slope_sd': .5, 'pair_sd': .5,
                   'triple_sd': .25, 'missing_log_dose_sd': .75},
    },
    'broad': {
        'label': 'Broader parameter variation',
        'description': 'Allows more varied slopes and interactions. Broader does not mean noninformative or more toxic.',
        'values': {'intercept_sd': 3., 'log_slope_sd': 1.25, 'pair_sd': 1.5,
                   'triple_sd': .75, 'missing_log_dose_sd': 1.25},
    },
}

POWER_PRESETS = {
    'default': {
        'label': 'Default borrowing (recommended / default)',
        'description': 'The existing Method 3 defaults: strength 1, patient budget 30, heterogeneity scale 0.5.',
        'values': {'strength': 1., 'patient_budget': 30., 'heterogeneity_scale': .5},
    },
    'cautious': {
        'label': 'Less historical borrowing',
        'description': 'Strength 0.5 and patient budget 15 reduce historical outcome information; heterogeneity scale is 0.75.',
        'values': {'strength': .5, 'patient_budget': 15., 'heterogeneity_scale': .75},
    },
    'flexible': {
        'label': 'More between-study variation',
        'description': 'Keep strength 1 and patient budget 30; allow a larger heterogeneity scale of 1.0.',
        'values': {'strength': 1., 'patient_budget': 30., 'heterogeneity_scale': 1.},
    },
}

COMMON_PRIOR_FIELDS = ('intercept_mean', 'intercept_sd', 'log_slope_mean', 'log_slope_sd',
                       'missing_log_dose_sd', 'unknown_presence_probability')


def active_prior_fields(backend):
    validate_backend(backend)
    return (*COMMON_PRIOR_FIELDS, 'pair_sd', 'triple_sd')


def canonical_prior(prior, backend):
    """Erase display labels from numerical cache identity.

    A custom specification equal to a preset resolves to the same prior.
    """
    original = prior if isinstance(prior, PriorSpec) else PriorSpec.model_validate(prior or {})
    active = {field: getattr(original, field) for field in active_prior_fields(backend)}
    # Canonicalize signed zero, which has identical numerical meaning.
    active = {field: 0. if value == 0 else value for field, value in active.items()}
    values = PriorSpec().model_dump()
    values.update(active)
    defaults = PriorSpec()
    values['prior_id'] = (defaults.prior_id if all(value == getattr(defaults, field)
                                                 for field, value in active.items()) else
                          'prior_' + fingerprint({'backend': backend, 'parameters': active})[:24])
    return PriorSpec.model_validate(values)


def risk_prior_preset(name, backend):
    if name not in RISK_PRESETS:
        raise ValueError('Unknown risk-prior preset')
    return canonical_prior(RISK_PRESETS[name]['values'], backend)


def power_prior_preset(name):
    from .power_prior import PowerPriorSpec
    if name not in POWER_PRESETS:
        raise ValueError('Unknown Power prior borrowing preset')
    return PowerPriorSpec.model_validate(POWER_PRESETS[name]['values'])


def _cache_entry(store, fit_id):
    """Cheap manifest inspection; full integrity checks happen on actual reuse."""
    import json
    directory = store.root / fit_id
    if not (directory / 'manifest.json').is_file():
        return {'cache_status': 'missing', 'fit_status': None, 'fit_id': fit_id}
    try:
        manifest = json.loads((directory / 'manifest.json').read_text())
        if not isinstance(manifest, dict):
            raise ValueError('The cache manifest must be an object.')
        if manifest.get('target_fit_id', manifest.get('source_fit_id')) != fit_id:
            raise ValueError('Cached identity does not match the requested model.')
        required = ('draws.npz', 'inputs.json' if fit_id.startswith('power_') else 'scope.json')
        if not all((directory / name).is_file() for name in required):
            raise ValueError('The cache is incomplete.')
        status = manifest.get('status', 'unknown')
        return {'cache_status': 'ready' if status == 'fitted' else 'cached_failure',
                'fit_status': status, 'fit_id': fit_id,
                'reason': manifest.get('reason'),
                'diagnostics': {key: manifest[key] for key in ('status', 'reason', 'attempts') if key in manifest}}
    except (ValueError, OSError, KeyError, EOFError) as error:
        return {'cache_status': 'invalid_cache', 'fit_status': None,
                'fit_id': fit_id, 'reason': str(error)}


def source_cache_status(scopes, *, backend='logistic', prior=None, sampler=None, store_root=None):
    """Inspect article-local caches shared by Method 1 and Method 2 targets."""
    from .service import DEFAULT_STORE
    from .store import PosteriorStore, source_fit_id
    from .workflow import phase_status
    prior, sampler = canonical_prior(prior, backend), sampler or SamplerSpec()
    store = PosteriorStore(DEFAULT_STORE if store_root is None else store_root)
    entries = []
    for scope in scopes:
        identity = {'document_id': scope.document_id, 'scope_id': scope.scope_id}
        phase = phase_status(scope.profile)
        if phase != 'phase_i':
            entries.append({**identity, 'cache_status': 'phase_excluded', 'fit_status': None,
                            'fit_id': None, 'reason': phase})
            continue
        if not scope.observations:
            entries.append({**identity, 'cache_status': 'no_outcomes', 'fit_status': None,
                            'fit_id': None, 'reason': 'No DLT/N observations; not precomputed.'})
            continue
        entries.append({**identity, **_cache_entry(store, source_fit_id(scope, backend, prior, sampler))})
    counts = {status: sum(x['cache_status'] == status for x in entries) for status in
              ('ready', 'missing', 'cached_failure', 'invalid_cache', 'phase_excluded', 'no_outcomes')}
    return {'kind': 'source_posteriors', 'backend': backend, 'prior': prior.model_dump(),
            'sampler': sampler.model_dump(), 'entries': entries, 'counts': counts,
            'ready': [x for x in entries if x['cache_status'] == 'ready'],
            'pending': [x for x in entries if x['cache_status'] == 'missing'],
            'failed': [x for x in entries if x['cache_status'] in ('cached_failure', 'invalid_cache')],
            'skipped': [x for x in entries if x['cache_status'] in ('phase_excluded', 'no_outcomes')]}


def prepare_source_cache(scopes, *, backend='logistic', prior=None, sampler=None,
                         store_root=None, progress=None):
    """Fit missing article-local models with the normal pipeline's policy."""
    from .service import DEFAULT_STORE, fit_library
    from .workflow import phase_status
    prior, sampler = canonical_prior(prior, backend), sampler or SamplerSpec()
    scopes = list(scopes)
    root = DEFAULT_STORE if store_root is None else store_root
    eligible = [s for s in scopes if s.observations and phase_status(s.profile) == 'phase_i']
    _, errors = fit_library(eligible, backends=(backend,), store_root=root,
                            prior=prior, sampler=sampler, progress=progress)
    return {**source_cache_status(scopes, backend=backend, prior=prior, sampler=sampler, store_root=root),
            'errors': errors}


def _target_cache_inputs(library, target, backend, prior, sampler, power_prior,
                         weights, threshold, top_articles, include_series_keys,
                         exclude_series_keys, relevance_decisions, store_root):
    from .power_prior import PowerPriorSpec, PowerPriorStore, prepare_power_prior, _target_coordinates
    from .service import _select, native_target, DEFAULT_STORE
    from .workflow import relevance_multiplier
    target = native_target(target)
    _target_coordinates(target)
    prior, sampler = canonical_prior(prior, backend), sampler or SamplerSpec()
    spec = power_prior if isinstance(power_prior, PowerPriorSpec) else PowerPriorSpec.model_validate(power_prior or {})
    scored, diagnostics = _select(library, target, weights, threshold or 0., top_articles,
                                  include_series_keys, exclude_series_keys)
    if not any(relevance_multiplier(scope.document_id, relevance_decisions) > 0 for scope, _, _ in scored):
        raise ValueError('No included Phase I evidence remains. Revise the candidate pool or relevance decisions.')
    prepared = prepare_power_prior(scored, target, spec, relevance_decisions)
    store = PowerPriorStore(DEFAULT_STORE if store_root is None else store_root)
    return prepared, store, prior, sampler, diagnostics


def _target_status(prepared, store, backend, prior, sampler, diagnostics):
    supported = bool(prepared['sources'] and any(s['review_multiplier'] > 0 for s in prepared['sources']))
    entry = (_cache_entry(store, store.identity(prepared, backend, prior, sampler)) if supported else
             {'cache_status': 'no_eligible_evidence', 'fit_status': None, 'fit_id': None})
    return {'kind': 'target_power_prior', 'backend': backend, 'prior': prior.model_dump(),
            'sampler': sampler.model_dump(), 'power_prior': prepared['spec'], **entry,
            'article_count': len(prepared['sources']),
            'discounted_patient_count': prepared['discounted_patient_count'],
            'excluded': prepared['excluded'], 'diagnostics': diagnostics}


def prepared_target_cache_status(prepared, *, backend='logistic', prior=None, sampler=None,
                                 store_root=None):
    """Inspect already-selected evidence; does not score text or call an API."""
    from .service import DEFAULT_STORE
    from .power_prior import PowerPriorStore
    prior, sampler = canonical_prior(prior, backend), sampler or SamplerSpec()
    store = PowerPriorStore(DEFAULT_STORE if store_root is None else store_root)
    return _target_status(prepared, store, backend, prior, sampler, {})


def prepare_prepared_target_cache(prepared, *, backend='logistic', prior=None, sampler=None,
                                  store_root=None, progress=None):
    """Cache an already-selected Method 3 recipe without repeating selection."""
    from .service import DEFAULT_STORE
    from .power_prior import PowerPriorStore
    prior, sampler = canonical_prior(prior, backend), sampler or SamplerSpec()
    store = PowerPriorStore(DEFAULT_STORE if store_root is None else store_root)
    status = _target_status(prepared, store, backend, prior, sampler, {})
    if status['cache_status'] == 'no_eligible_evidence':
        return status
    store.fit(prepared, backend, prior, sampler, progress)
    return _target_status(prepared, store, backend, prior, sampler, {})


from phase1_prediction.similarity_engine import with_similarity


@with_similarity
def target_cache_status(library, target, *, backend='logistic', prior=None, sampler=None,
                        power_prior=None, weights=None, threshold=0., top_articles=None,
                        include_series_keys=(), exclude_series_keys=(), relevance_decisions=None,
                        store_root=None):
    """Inspect the exact Method 3 target/evidence cache without fitting."""
    prepared, store, prior, sampler, diagnostics = _target_cache_inputs(
        library, target, backend, prior, sampler, power_prior, weights, threshold, top_articles,
        include_series_keys, exclude_series_keys, relevance_decisions, store_root)
    return _target_status(prepared, store, backend, prior, sampler, diagnostics)


@with_similarity
def prepare_target_cache(library, target, *, backend='logistic', prior=None, sampler=None,
                         power_prior=None, weights=None, threshold=0., top_articles=None,
                         include_series_keys=(), exclude_series_keys=(), relevance_decisions=None,
                         store_root=None, progress=None):
    """Fit the selected target recipe only; never populate source-article fits."""
    prepared, store, prior, sampler, diagnostics = _target_cache_inputs(
        library, target, backend, prior, sampler, power_prior, weights, threshold, top_articles,
        include_series_keys, exclude_series_keys, relevance_decisions, store_root)
    status = _target_status(prepared, store, backend, prior, sampler, diagnostics)
    if status['cache_status'] == 'no_eligible_evidence':
        return status
    store.fit(prepared, backend, prior, sampler, progress)
    return _target_status(prepared, store, backend, prior, sampler, diagnostics)
