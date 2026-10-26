"""Explicit, resumable preparation of candidate Bayesian posterior caches."""
from phase1_prediction.multidrug import prior_presets as presets


MODEL_LABELS = {'logistic': 'Component Logistic', 'logistic_product': 'Product-interaction Logistic',
                'single_drug_logistic': 'Single-drug Bayesian Logistic'}
STATUS_LABELS = {
    'ready': 'Cached', 'missing': 'Pending', 'cached_failure': 'Sampling needs attention',
    'invalid_cache': 'Cache needs attention', 'no_eligible_evidence': 'No eligible evidence',
}


def _risk_choices(st, backends, selected_prior, key):
    """Preset labels never enter a numerical cache identity."""
    choices = {name: item['label'] for name, item in presets.RISK_PRESETS.items()}
    if selected_prior is not None and any(
        all(presets.canonical_prior(selected_prior, backend) != presets.risk_prior_preset(name, backend)
            for name in presets.RISK_PRESETS) for backend in backends
    ):
        choices['current'] = 'Current custom prior'
    names = st.multiselect('Risk priors to cache', list(choices), default=list(choices),
                           format_func=choices.__getitem__, key=key + '_priors')
    def resolve(name, backend):
        return (presets.canonical_prior(selected_prior, backend) if name == 'current'
                else presets.risk_prior_preset(name, backend))
    return [(name, choices[name], resolve) for name in names]


def _source_rows(jobs):
    return [{'Model': MODEL_LABELS[backend], 'Risk prior': label,
             'Cached': result['counts'].get('ready', 0),
             'Pending': result['counts'].get('missing', 0),
             'Needs attention': sum(result['counts'].get(k, 0)
                                    for k in ('cached_failure', 'invalid_cache')),
             'Skipped': sum(result['counts'].get(k, 0)
                            for k in ('phase_excluded', 'no_outcomes'))}
            for backend, _, label, result in jobs]


def _show_source_details(st, jobs):
    failures = [{**row, 'model': backend, 'risk_prior': label}
                for backend, _, label, result in jobs for row in result.get('failed', [])]
    errors = [{**row, 'model': backend, 'risk_prior': label}
              for backend, _, label, result in jobs for row in result.get('errors', [])]
    if failures or errors:
        st.warning('Some cache entries need attention. They are not counted as usable posteriors. '
                   'Changing the prior or sampling settings creates a separate fit; cached sampling failures remain available for diagnosis.')
        with st.expander('Cache failures and diagnostics'):
            st.json(failures + errors)


def render_source_cache(st, scopes, *, backends=('logistic',),
                        store_root=None, key='source_cache', allow_backend_selection=True,
                        selected_prior=None, target=None):
    """Show article-cache availability, fitting only after an explicit click.

    Posterior averages and mixtures share these article-local posterior fits. No target
    prediction or source outcomes are changed by warming the cache.
    """
    scopes = list(scopes)
    projected_scopes = []
    if 'single_drug_logistic' in backends:
        from phase1_prediction.multidrug.single_drug import project_scope
        if target is None:
            # Library preparation stores one conditional model per possible
            # numeric focal drug; no target data enter an article fit.
            for scope in scopes:
                for drug in scope.drugs:
                    if drug.mode == 'numeric':
                        try:
                            projected_scopes.extend(project_scope(scope, drug.model_drug_id))
                        except ValueError:
                            continue
        else:
            from phase1_prediction.multidrug.mapping import match_target
            from phase1_prediction.multidrug.service import prepare_single_drug_evidence, _focal_matching_target
            focal_target = _focal_matching_target(target)
            scored = [(scope, match_target(focal_target, scope), 0.) for scope in scopes]
            projected, _, notices = prepare_single_drug_evidence(scored, target)
            projected_scopes = [scope for scope, _, _ in projected]
            if notices:
                with st.expander('Single-drug source preparation notes'):
                    st.json(notices)
    models = list(backends)
    if allow_backend_selection and len(models) > 1:
        models = st.multiselect('Models to cache', models, default=models,
                                format_func=MODEL_LABELS.__getitem__, key=key + '_models')
    else:
        st.caption('Selected probability model: ' + ', '.join(MODEL_LABELS[b] for b in models))
    candidates = _risk_choices(st, models, selected_prior, key)
    st.caption('Posterior averages and mixtures reuse the same article posteriors. Each fit uses only that article\'s '
               'Phase I DLT/N data. Matching cache entries are reused; changing the target or curve aggregation '
               'does not require refitting them. The first preparation can take time.')
    if not models or not candidates:
        st.info('Select at least one probability model and one risk prior to inspect its cache.')
        return
    jobs = []
    for backend in models:
        for name, label, resolve in candidates:
            prior = resolve(name, backend)
            try:
                model_scopes = projected_scopes if backend == 'single_drug_logistic' else scopes
                result = presets.source_cache_status(model_scopes, backend=backend, prior=prior, store_root=store_root)
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f'{MODEL_LABELS[backend]} · {label}: {error}')
                continue
            jobs.append((backend, prior, label, result))
    table = st.empty()
    table.dataframe(_source_rows(jobs), hide_index=True, use_container_width=True)
    pending = sum(j[3]['counts'].get('missing', 0) for j in jobs)
    if st.button('Prepare or resume article posterior caches', key=key + '_prepare', disabled=not pending):
        progress = st.empty()
        total = sum(sum(job[3]['counts'].get(status, 0)
                        for status in ('ready', 'missing', 'cached_failure', 'invalid_cache')) for job in jobs)
        completed = 0
        results = []
        for backend, prior, label, before in jobs:
            def update(done, _total, document, model):
                progress.info(f'{min(completed + done, total)}/{total} · {MODEL_LABELS[model]} · {label} · {document}')
            try:
                model_scopes = projected_scopes if backend == 'single_drug_logistic' else scopes
                result = presets.prepare_source_cache(model_scopes, backend=backend, prior=prior,
                                                      store_root=store_root, progress=update)
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f'{MODEL_LABELS[backend]} · {label}: {error}')
                result = {**before, 'errors': [{'reason': str(error)}]}
            results.append((backend, prior, label, result))
            completed += sum(before['counts'].get(status, 0)
                             for status in ('ready', 'missing', 'cached_failure', 'invalid_cache'))
        jobs = results
        rows = _source_rows(jobs)
        table.dataframe(rows, hide_index=True, use_container_width=True)
        ready = sum(row['Cached'] for row in rows)
        unfinished = sum(row['Pending'] + row['Needs attention'] for row in rows)
        if unfinished or any(job[3].get('errors') for job in jobs):
            progress.warning(f'{ready} usable cached fits; {unfinished} entries pending or needing attention.')
        else:
            progress.success(f'All {ready} selected article posterior fits are cached.')
    _show_source_details(st, jobs)


def _target_rows(jobs):
    return [{'Risk prior': label, 'Status': STATUS_LABELS.get(result['cache_status'], result['cache_status']),
             'Articles': result.get('article_count', 0),
             'Discounted patients': result.get('discounted_patient_count', 0.),
             'Cache ID': result.get('fit_id')}
            for _, label, result in jobs]


def render_target_cache(st, library, target, preview, included, relevance_decisions,
                        weights, threshold, top_articles, backend, power_settings, review_ready,
                        *, store_root=None, key='target_cache', selected_prior=None, scopes=None):
    """Prepare Historical-likelihood candidates for the exact previewed evidence recipe.

    This reuses the supplied evidence preview rather than recomputing embeddings.
    Target posterior preparation never modifies a displayed prediction.
    """
    st.caption('Historical-likelihood has a target-specific cache. It depends on the target, selected articles, drug '
               'correspondence, review decisions, risk prior and borrowing settings. This prepares the selected '
               'risk priors using the current borrowing settings; it does not try every borrowing-setting combination.')
    candidates = _risk_choices(st, (backend,), selected_prior, key)
    if not preview:
        st.info('Preview articles and drug correspondence before preparing Historical-likelihood caches.')
        return
    if not included:
        st.info('Include at least one source model scope before preparing Historical-likelihood caches.')
        return
    if not review_ready:
        st.info('Complete the selected evidence review before preparing Historical-likelihood caches.')
        return
    if not candidates:
        st.info('Select at least one risk prior to inspect its Historical-likelihood cache.')
        return
    from phase1_prediction.multidrug.power_prior import PowerPriorSpec, prepare_power_prior
    from phase1_prediction.multidrug.observations import load_model_library
    try:
        if scopes is None:
            scopes, _ = load_model_library(library.source_roots, evidence_policy=library.evidence_policy,
                                           review_policy=library.review_policy)
        by_id = {scope.scope_id: scope for scope in scopes}
        rows = [row for row in preview['selected_series'] if row['series_key'] in set(included)]
        if set(included) - {row['series_key'] for row in rows} or any(row['series_key'] not in by_id for row in rows):
            raise ValueError('The library or selected scopes changed. Preview the evidence again.')
        scored = [(by_id[row['series_key']], row['mapping'],
                   row.get('similarity') or 0.) for row in rows]
        spec = (power_settings if isinstance(power_settings, PowerPriorSpec)
                else PowerPriorSpec.model_validate(power_settings or {}))
        if backend == 'single_drug_logistic':
            from phase1_prediction.multidrug.service import prepare_single_drug_evidence
            scored, target, _ = prepare_single_drug_evidence(scored, target)
        prepared = prepare_power_prior(scored, target, spec, relevance_decisions)
    except (ValueError, RuntimeError, OSError) as error:
        st.error(str(error))
        return
    options = dict(backend=backend, store_root=store_root)
    jobs = []
    for name, label, resolve in candidates:
        prior = resolve(name, backend)
        try:
            status = presets.prepared_target_cache_status(prepared, prior=prior, **options)
        except (ValueError, RuntimeError, OSError) as error:
            st.error(f'{label}: {error}')
            continue
        jobs.append((prior, label, status))
    table = st.empty()
    table.dataframe(_target_rows(jobs), hide_index=True, use_container_width=True)
    pending = [job for job in jobs if job[2]['cache_status'] == 'missing']
    if st.button('Prepare or resume Historical-likelihood target caches', key=key + '_prepare', disabled=not pending):
        progress = st.empty()
        results = []
        for index, (prior, label, before) in enumerate(jobs, 1):
            def update(*details):
                progress.info(f'{index}/{len(jobs)} prior candidates · {MODEL_LABELS[backend]} · {label}'
                              + (' · ' + ' · '.join(map(str, details)) if details else ''))
            update()
            try:
                result = presets.prepare_prepared_target_cache(prepared, prior=prior, progress=update, **options)
            except (ValueError, RuntimeError, OSError) as error:
                result = {**before, 'cache_status': 'invalid_cache', 'reason': str(error)}
            results.append((prior, label, result))
        jobs = results
        table.dataframe(_target_rows(jobs), hide_index=True, use_container_width=True)
        ready = sum(job[2]['cache_status'] == 'ready' for job in jobs)
        if ready == len(jobs):
            progress.success(f'All {ready} selected Historical-likelihood target fits are cached.')
        else:
            progress.warning(f'{ready}/{len(jobs)} usable target fits; inspect the remaining entries below.')
    failures = [{'risk_prior': label, **result} for _, label, result in jobs
                if result['cache_status'] in ('cached_failure', 'invalid_cache', 'no_eligible_evidence')]
    if failures:
        with st.expander('Target cache exclusions and diagnostics'):
            st.json(failures)
