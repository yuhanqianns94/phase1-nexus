"""Public Bayesian pipeline: immutable source fits, then target transport."""
from datetime import datetime, timezone
from pathlib import Path
import math

from phase1_prediction.models import PredictionResult
from . import VERSION, TARGET_PREDICTION_VERSION, BACKENDS
from .contracts import TargetRegimen, TargetDrug, RegimenPoint, PriorSpec, SamplerSpec, fingerprint
from .observations import load_model_library
from .mapping import MATCHING_POLICY, match_target
from .store import PosteriorStore
from .transport import (transport_joint_posteriors, assemble_components,
                        average_joint_posteriors, average_assembled_components)

from phase1_prediction.similarity_engine import with_similarity
from .workflow import phase_status, relevance_multiplier, article_budgets, coordinate_support

DEFAULT_STORE = Path(__file__).resolve().parents[2] / 'output' / 'source_posteriors'
SYNTHESIS_MODES = ('same_article_joint_posterior_mixture', 'cross_article_component_assembly',
                   'same_article_joint_posterior_average', 'target_power_prior',
                   'cross_article_component_assembly_average')


def _synthesis_function(mode):
    """Explicit dispatch, shared by initial prediction and frozen redraw."""
    methods = {
        'same_article_joint_posterior_mixture': transport_joint_posteriors,
        'cross_article_component_assembly': assemble_components,
        'cross_article_component_assembly_average': average_assembled_components,
        'same_article_joint_posterior_average': average_joint_posteriors,
    }
    if mode not in methods:
        raise ValueError('Unknown synthesis mode')
    return methods[mode]


def _evidence_distribution(method):
    """Include unsampled candidates when checking a frozen evidence recipe."""
    return {
        'contributors': method.get('contributors', []),
        'component_pools': method.get('component_pools', []),
        'recipe_bank': method.get('recipe_bank', []),
        'assembly_computation': method.get('assembly_computation', {}),
        'excluded': method.get('excluded', []),
    }


def native_target(target):
    if isinstance(target, TargetRegimen):
        return target
    if isinstance(target, dict):
        return TargetRegimen.model_validate(target)
    from phase1_prediction.compact_adapter import target_query
    from phase1_prediction.profiles import query_profile
    query = target_query(target)
    profile = query_profile(query)
    configs = profile.get('configurations', [])
    profiles = (configs[0].get('drugs', []) if configs else []) or [{'name': target.drug}]
    profiles = sorted(profiles, key=lambda p: p.get('name') != target.drug)
    exposures = [e for e in query.get('_target_exposures', []) if e['scenario'] == query.get('_selected_scenario', 'Main')]
    main_doses = list(query.get('planned_doses') or target.planned_doses)
    if not main_doses:
        main_doses = [None]
    drugs, vectors = [], []
    for i, profile_drug in enumerate(profiles):
        name = profile_drug['name']; rows = [e for e in exposures if e['name'] == name]
        grids = {tuple(e['doses']) for e in rows if e['doses']}
        if len(grids) > 1:
            raise ValueError('Multiple dosing phases require an explicit target regimen coordinate')
        values = list(next(iter(grids))) if grids else (main_doses if i == 0 else [None])
        if profile_drug.get('presence') == 'absent':
            values = [0.]
        if len(values) == 1:
            values *= len(main_doses)
        if len(values) != len(main_doses):
            raise ValueError('Partner dose vectors must be fixed or paired with every primary level; use explicit regimen points for a grid')
        positive = [v for v in values if v is not None and v > 0]
        unit = next((e['unit'] for e in rows if e.get('unit')), target.dose_unit if i == 0 else None)
        drugs.append(TargetDrug(drug_id=f'drug_{i+1}', name=name, research_priority=i+1,
            research_role='primary' if i == 0 else 'partner', profile=profile_drug,
            mode='numeric' if positive else 'background', unit=unit,
            reference_dose=math.exp(sum(map(math.log, positive))/len(positive)) if positive else None))
        vectors.append(values)
    points = [RegimenPoint(level_id=f'L{i+1}', doses={d.drug_id: v[i] for d, v in zip(drugs, vectors)}) for i in range(len(main_doses))]
    return TargetRegimen(name=target.name, drugs=drugs, points=points, path_level_ids=[p.level_id for p in points],
                         profile=profile, target_dlt=target.target_dlt)


def fit_library(scopes, *, backends=('logistic', 'logistic_product'), store_root=DEFAULT_STORE, prior=None, sampler=None, progress=None):
    from .prior_presets import canonical_prior
    store = PosteriorStore(store_root)
    entries, errors = [], []
    scopes = list(scopes)
    jobs = []
    for backend in backends:
        for scope in scopes:
            if backend != 'single_drug_logistic' or scope.lineage.get('single_drug_projection'):
                jobs.append((backend, scope))
                continue
            from .single_drug import project_scope
            numeric = [drug for drug in scope.drugs if drug.mode == 'numeric']
            if not numeric:
                errors.append({'document_id': scope.document_id, 'scope_id': scope.scope_id,
                    'backend': backend, 'status': 'projection_error', 'reason': 'No numeric focal source axis.'})
            for drug in numeric:
                try:
                    notices = []
                    jobs.extend((backend, projected) for projected in project_scope(scope, drug.model_drug_id, exclusions=notices))
                    errors.extend({**notice, 'backend': backend, 'status': 'projection_exclusion'} for notice in notices)
                except ValueError as error:
                    errors.append({'document_id': scope.document_id, 'scope_id': scope.scope_id,
                        'backend': backend, 'status': 'projection_error', 'reason': str(error)})
    total = len(jobs) + len(errors)
    for backend, scope in jobs:
        if backend not in BACKENDS:
            raise ValueError('Unknown Bayesian backend')
        backend_prior = canonical_prior(prior, backend)
        if phase_status(scope.profile)!='phase_i':
            errors.append({'document_id':scope.document_id,'scope_id':scope.scope_id,'backend':backend,'status':'phase_excluded','reason':phase_status(scope.profile)})
            continue
        if progress:
            progress(len(entries)+len(errors), total, scope.document_id, backend)
        try:
            entry = store.fit(scope, backend, prior=backend_prior, sampler=sampler)
            entries.append(entry)
        except (ValueError, RuntimeError, OSError) as error:
            errors.append({'document_id': scope.document_id, 'scope_id': scope.scope_id,
                           'backend': backend, 'status': 'fit_error', 'reason': str(error)})
    return entries, errors


def _focal_matching_target(target):
    """Match the selected dose axis; companion identities remain clinical context."""
    from types import SimpleNamespace
    from .mapping import coverage_target_profile
    metadata = target.profile.get('single_drug_analysis', {})
    selected = metadata.get('focal_drug_id')
    focal = next((d for d in target.drugs if d.drug_id == selected), None)
    if selected and focal is None:
        raise ValueError('The selected focal drug is missing from the trial plan.')
    focal = focal or next((d for d in target.drugs if getattr(d, 'mode', None) == 'numeric'), target.drugs[0])
    overrides = {key: {k: v for k, v in pairs.items() if k == focal.drug_id}
                 for key, pairs in target.mapping_overrides.items()}
    return SimpleNamespace(drugs=[focal], profile=target.profile,
        mapping_overrides=overrides, minimum_drug_similarity=target.minimum_drug_similarity,
        minimum_clinical_coverage=getattr(target, 'minimum_clinical_coverage', 0.),
        coverage_profile=coverage_target_profile(target))


def prepare_single_drug_evidence(scored, target):
    """Preserve original outcomes while projecting the focal axis and context."""
    import copy
    from .single_drug import project_target, project_scope
    projected_target = project_target(target)
    focal_id = projected_target.drugs[0].drug_id
    projected, notices = [], []
    for scope, mapping, strength in scored:
        source_id = mapping['pairs'].get(focal_id)
        if not source_id:
            notices.append({'document_id': scope.document_id, 'scope_id': scope.scope_id,
                            'reason': 'No corresponding source for the focal drug.'})
            continue
        try:
            scopes = project_scope(scope, source_id, exclusions=notices)
            for source in scopes:
                # Dose compatibility is checked before fitting/cache preparation.
                from .mapping import mapped_relative_dose
                for point in projected_target.points:
                    mapped_relative_dose(projected_target.drugs[0], source.drugs[0], point.doses[focal_id])
                projected_mapping = copy.deepcopy(mapping)
                projected_mapping.update(scope_id=source.scope_id,
                    pairs={t.drug_id: d.model_drug_id for t, d in zip(projected_target.drugs, source.drugs)},
                    unmatched_target_ids=[], removed_source_ids=[],
                    original_scope_id=scope.scope_id, original_pairs=copy.deepcopy(mapping['pairs']))
                projected_mapping['mapping_id'] = fingerprint(projected_mapping)
                projected.append((source, projected_mapping, strength))
        except ValueError as error:
            notices.append({'document_id': scope.document_id, 'scope_id': scope.scope_id, 'reason': str(error)})
    return projected, projected_target, notices


def _single_signature(target):
    from .power_prior import target_signature
    focal = _focal_matching_target(target).drugs[0].drug_id
    values = {d.drug_id: sorted({str(p.doses[d.drug_id]) for p in target.points})
              for d in target.drugs if d.drug_id != focal}
    return {'clinical': target_signature(target), 'focal_drug_id': focal, 'fixed_companions': values}


def _restore_single_target(method, target):
    """Expose original regimen coordinates to plots, exports and BOIN."""
    method['computational_target'] = method['target']
    method['target'] = target.model_dump(mode='json')
    by_level = {p.level_id: p.doses for p in target.points}
    for row in method.get('summary', []):
        row['doses'] = dict(by_level[row['level_id']])
    method['evidence_construction'] = 'single_drug_conditional_regimen'
    method['prediction_id'] = fingerprint(method)
    return method


def _select(library, target, weights, threshold, top_articles, included=(), excluded=(), *, focal_only=False):
    if phase_status(target.profile) in {'out_of_scope','mixed_requires_separation'}:
        raise ValueError('Only Phase I or an explicitly separated Phase I component is supported.')
    scopes, diagnostics = load_model_library(library.source_roots, evidence_policy=library.evidence_policy, review_policy=library.review_policy)
    included, excluded = set(included), set(excluded)
    known = {s.scope_id for s in scopes}
    if set(target.mapping_overrides) - known:
        raise ValueError('Manual mapping refers to a missing or regrouped source scope; preview the current library and update that mapping first.')
    if included & excluded or (included | excluded) - known:
        raise ValueError('Invalid or conflicting model scope selection')
    if not 0 <= threshold <= 1:
        raise ValueError('Similarity threshold must be in [0,1]')
    if top_articles is not None and (isinstance(top_articles, bool) or not isinstance(top_articles, int) or top_articles < 1):
        raise ValueError('top_articles must be a positive integer')
    scored = []
    for scope in scopes:
        phase=phase_status(scope.profile)
        if phase!='phase_i':
            diagnostics['notices'].append({'document_id':scope.document_id,'scope_id':scope.scope_id,'reason':'phase_filter','phase_status':phase})
            continue
        mapping = match_target(_focal_matching_target(target) if focal_only else target, scope, weights=weights)
        if mapping.get('comparison_skipped'):
            diagnostics['notices'].append({'document_id': scope.document_id, 'scope_id': scope.scope_id,
                'reason': mapping['comparison_skipped'],
                'coverage': mapping['clinical_coverage']['coverage'],
                'minimum_coverage': mapping['minimum_clinical_coverage']})
            continue
        score = mapping['score']; similarity = score['similarity'] or 0.
        if similarity >= threshold and (not included or scope.scope_id in included) and scope.scope_id not in excluded:
            scored.append((scope, mapping, similarity))
    scored.sort(key=lambda x: (-x[2], x[0].document_id, x[0].scope_id))
    docs = list(dict.fromkeys(s.document_id for s, _, _ in scored))
    if top_articles is not None:
        scored = [x for x in scored if x[0].document_id in docs[:top_articles]]
    return scored, diagnostics


@with_similarity
def preview(library, target, *, threshold=None, top_articles=None, weights=None, include_series_keys=(), exclude_series_keys=()):
    target = native_target(target)
    scored, diagnostics = _select(library, target, weights, threshold or 0., top_articles, include_series_keys, exclude_series_keys, focal_only=True)
    rows = [{'series_key': s.scope_id, 'document_id': s.document_id, 'label': s.label,
             'similarity': m['score']['similarity'], 'mapping': m,
             'point_count': len(s.observations), 'patient_count': sum(o.n for o in s.observations),
             'prior_only': not s.observations} for s, m, _ in scored]
    return {'format': 'multidrug_evidence_preview_v1', 'target': target.model_dump(mode='json'),
            'selected_series': rows, 'excluded_series': diagnostics['notices'], 'diagnostics': diagnostics,
            'selected_article_count': len({r['document_id'] for r in rows}), 'selected_series_count': len(rows),
            'selected_point_count': sum(r['point_count'] for r in rows)}


@with_similarity
def predict(library, target, *, backend='logistic', synthesis=SYNTHESIS_MODES[0], prior=None, sampler=None,
            store_root=DEFAULT_STORE, draw_count=1000, seed=20260919, weights=None, threshold=None,
            top_articles=None, include_series_keys=(), exclude_series_keys=(), progress=None, relevance_decisions=None,
            power_prior=None, **legacy_options):
    if legacy_options:
        raise ValueError('Legacy cross-article fitting options are unsupported; use an explicit PriorSpec/SamplerSpec: ' + ', '.join(legacy_options))
    if backend not in BACKENDS or synthesis not in SYNTHESIS_MODES:
        raise ValueError('Unknown backend or synthesis mode')
    if backend == 'logistic_product' and synthesis in {
            'cross_article_component_assembly', 'cross_article_component_assembly_average'}:
        raise ValueError('Product-interaction Logistic supports complete-regimen borrowing (Methods 1 and 2) and Power prior (Method 3). Component assembly requires Component Logistic because no cross-article shared-intercept rule is defined.')
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 1:
        raise ValueError('draw_count must be positive')
    from .prior_presets import canonical_prior
    target, prior, sampler = native_target(target), canonical_prior(prior, backend), sampler or SamplerSpec()
    if power_prior is not None and synthesis != 'target_power_prior':
        raise ValueError('Power prior settings apply only to Method 3.')
    scored, diagnostics = _select(library, target, weights, threshold or 0., top_articles, include_series_keys, exclude_series_keys, **({"focal_only": True} if backend == "single_drug_logistic" else {}))
    original_target = target
    original_scored = scored
    projection_notices = []
    if backend == "single_drug_logistic":
        if synthesis not in {"same_article_joint_posterior_average", "same_article_joint_posterior_mixture", "target_power_prior"}:
            raise ValueError("The single-drug model supports posterior average, posterior mixture and historical likelihood.")
        scored, target, projection_notices = prepare_single_drug_evidence(scored, target)
    review_weights=article_budgets([{'document_id':s.document_id,**m['score']} for s,m,_ in scored],relevance_decisions)
    included_scored=[x for x in scored if relevance_multiplier(x[0].document_id,relevance_decisions)>0]
    if not included_scored:raise ValueError('No included Phase I evidence remains. Revise the candidate pool or relevance decisions.')
    entries, errors = [], []
    if synthesis == 'target_power_prior':
        from .power_prior import PowerPriorSpec, predict_power_prior
        spec = power_prior if isinstance(power_prior, PowerPriorSpec) else PowerPriorSpec.model_validate(power_prior or {})
        # Keep excluded articles in the pre-review cap; exclusion cannot increase
        # the powers of articles that remain. No article posterior is re-fitted.
        result = predict_power_prior(scored, target, backend=backend, prior=prior, sampler=sampler,
            spec=spec, store_root=store_root, draw_count=draw_count, seed=seed,
            relevance_decisions=relevance_decisions, progress=progress)
    else:
        entries, errors = fit_library([s for s, _, _ in included_scored], backends=(backend,), store_root=store_root,
                                     prior=prior, sampler=sampler, progress=progress)
        fn = _synthesis_function(synthesis)
        transport_options = ({"mappings": {s.scope_id: m for s, m, _ in scored}}
                             if backend == "single_drug_logistic" else {})
        result = fn(entries, target, backend=backend, draw_count=draw_count, seed=seed, prior=prior, weights=weights, relevance_decisions=relevance_decisions, **transport_options)
        result['coordinate_support']=coordinate_support(entries,target,result)
    failures = [{'source_fit_id': m['source_fit_id'], 'status': m['status'], 'diagnostics': m}
                for _, _, m in entries if m['status'] != 'fitted']
    contract = {'version': VERSION, 'backend': backend, 'synthesis': synthesis, 'matching_policy': MATCHING_POLICY,
                'prediction_engine_version': TARGET_PREDICTION_VERSION,
                'source_fit_ids': [m['source_fit_id'] for _, _, m in entries], 'prior': prior.model_dump(),
                'sampler': sampler.model_dump(), 'library_diagnostics': diagnostics, 'fit_errors': errors,
                'source_statuses': failures, 'partial_library': bool(errors or diagnostics['notices'] or failures),
                'article_fits_use_only_own_data': True, 'relevance_decisions':relevance_decisions, 'review_weights':review_weights, 'store_root':str(store_root), 'weights': weights, 'threshold': threshold, 'top_articles': top_articles}
    if synthesis == 'target_power_prior':
        contract.update(target_fit_id=result.get('target_fit_id'), power_prior=spec.model_dump(),
                        target_inference='fixed_discounted_historical_likelihood',
                        review_weight_interpretation='Normalized review weights are preview-only. Method 3 uses the unnormalized contributor powers.',
                        partial_library=bool(diagnostics['notices'] or result['excluded'] or result['status'] != 'fitted'))
    if backend == 'single_drug_logistic':
        result['excluded'].extend(projection_notices)
        result = _restore_single_target(result, original_target)
        contract.update(single_drug_signature=_single_signature(original_target),
            single_drug_mappings={s.scope_id: m for s, m, _ in scored},
            projection_notices=projection_notices)
        target = original_target
    payload = {'target': target.model_dump(mode='json'), 'contract': contract, 'method': result}
    return PredictionResult(format='phase1_v3_multidrug_prediction_v1', model_version=VERSION,
        run_id='prediction_' + fingerprint(payload)[:24], target=payload['target'], evidence_contract=contract,
        selected_series=result['contributors'], excluded_series=result['excluded']+errors,
        methods={synthesis: result}, generated_at_utc=datetime.now(timezone.utc).isoformat())


@with_similarity
def preview_planned(library, draft, *, weights=None, threshold=0., top_articles=None):
    """Clinical matching before dose selection, without inventing exposures."""
    from types import SimpleNamespace
    drugs=[SimpleNamespace(drug_id=d['drug_id'],name=d['name'],
        research_priority=i+1,research_role=d.get('research_role','unclear'),
        profile=d.get('profile',{})) for i,d in enumerate(draft.get('drugs',[])) if d.get('name','').strip()]
    if not drugs:raise ValueError('Enter at least one drug name or development code to compare articles.')
    target=SimpleNamespace(drugs=drugs,profile=draft.get('profile',{}),
        mapping_overrides=draft.get('mapping_overrides',{}),
        minimum_drug_similarity=draft.get('minimum_drug_similarity',.2),
        minimum_clinical_coverage=draft.get('minimum_clinical_coverage',0.))
    scored,diagnostics=_select(library,target,weights,threshold,top_articles,focal_only=True)
    return {'format':'planned_clinical_preview_v1','prediction_ready':False,
            'selected_series':[{'series_key':s.scope_id,'document_id':s.document_id,'label':s.label,
                'similarity':m['score']['similarity'],'mapping':m}
                for s,m,_ in scored], 'diagnostics':diagnostics}


def evaluate_frozen_prediction(prediction, target):
    """Re-evaluate saved posterior recipes. Never fit or reselect the library.

    A changed eligibility set is rejected instead of silently changing weights.
    """
    import copy
    from phase1_prediction.similarity_engine import similarity_context, DEFAULT_MODEL
    target=native_target(target)
    old=next(iter(prediction['methods'].values()))
    contract=prediction['evidence_contract']
    if old.get('prediction_engine_version')!=TARGET_PREDICTION_VERSION:
        raise ValueError('Generate a current-version prediction before exploring new coordinates.')
    original_target = target
    is_single = contract['backend'] == 'single_drug_logistic'
    if is_single:
        from .single_drug import project_target
        if _single_signature(target) != contract.get('single_drug_signature'):
            raise ValueError('The focal drug or fixed companion context changed; generate a new prediction.')
        target = project_target(target)
    if contract['synthesis'] == 'target_power_prior':
        from .power_prior import PowerPriorStore, evaluate_power_prior
        if not contract.get('target_fit_id') or old['status'] != 'fitted':
            raise ValueError('Generate a fitted Method 3 prediction before exploring new coordinates.')
        prepared, arrays, manifest = PowerPriorStore(contract.get('store_root',DEFAULT_STORE)).load(contract['target_fit_id'])
        method = evaluate_power_prior(prepared, arrays, manifest, target,
            draw_count=len(old['posterior_draws']), seed=old['seed'])
        if method['provenance'] != old['provenance'] or method['contributors'] != old['contributors']:
            raise ValueError('The frozen Power prior recipe changed; generate a new prediction.')
        if is_single:
            method['excluded'].extend(contract.get('projection_notices', []))
            method = _restore_single_target(method, original_target)
            target = original_target
        result = copy.deepcopy(prediction)
        result['target'] = target.model_dump(mode='json')
        result['methods'] = {contract['synthesis']: method}
        result['evidence_contract']['regridded_from'] = prediction['run_id']
        result['run_id'] = 'prediction_' + fingerprint([result['target'], result['evidence_contract'], method])[:24]
        result['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
        return result
    store=PosteriorStore(contract.get('store_root',DEFAULT_STORE))
    entries=[store.load(k) for k in contract['source_fit_ids']]
    fn=_synthesis_function(contract['synthesis'])
    engine=contract.get('similarity_engine',{})
    with similarity_context(engine.get('engine','local'),engine.get('model') or DEFAULT_MODEL):
        method=fn(entries,target,backend=contract['backend'],draw_count=len(old['posterior_draws']),seed=old['seed'],
                  prior=PriorSpec.model_validate(contract['prior']),weights=contract.get('weights'),
                  relevance_decisions=contract.get('relevance_decisions'),
                  **({'mappings': contract['single_drug_mappings']} if is_single else {}))
    def recipes(value):
        value=copy.deepcopy(value)
        def walk(x):
            if isinstance(x,dict):
                x.pop('target_reference_dose',None)
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(value)
        return value
    if is_single:
        method['excluded'].extend(contract.get('projection_notices', []))
    if _evidence_distribution(method) != _evidence_distribution(old):
        raise ValueError('The new coordinates change the frozen evidence distribution. Generate a new prediction explicitly; saved-curve exploration cannot change contributors or weights.')
    if method['status']!='fitted' or recipes(method['provenance'])!=recipes(old['provenance']):
        raise ValueError('The new coordinates change source eligibility or posterior recipes. Generate a new prediction explicitly; this is not a redraw of the saved curve.')
    method['coordinate_support']=coordinate_support(entries,target,method)
    if is_single:
        method = _restore_single_target(method, original_target)
        target = original_target
    result=copy.deepcopy(prediction)
    result['target']=target.model_dump(mode='json')
    result['methods']={contract['synthesis']:method}
    result['selected_series']=method['contributors']
    result['excluded_series']=method['excluded']+contract.get('fit_errors',[])
    result['evidence_contract']['regridded_from']=prediction['run_id']
    result['run_id']='prediction_'+fingerprint([result['target'],result['evidence_contract'],method])[:24]
    result['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    return result
