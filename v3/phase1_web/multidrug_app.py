"""Native full-regimen Bayesian workspace; historical analyses remain readable."""
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from phase1_prediction.evidence import load_evidence_library
from phase1_prediction.multidrug.contracts import fingerprint
from phase1_prediction.multidrug import TARGET_PREDICTION_VERSION, VERSION
from phase1_prediction.multidrug.mapping import MATCHING_POLICY
from phase1_prediction.profile_similarity import VERSION as SIMILARITY_VERSION
from phase1_prediction.multidrug.workflow import phase_status
from phase1_prediction.multidrug.service import predict, preview_planned, BACKENDS
from phase1_prediction.multidrug.power_prior import POWER_PRIOR_VERSION
from phase1_prediction.multidrug.observations import load_model_library
from phase1_simulation.service import BoinSettings, search_boin_parameters
from phase1_simulation.multidrug import simulate_regimen_path
from phase1_project.analyses import save_analysis, list_analyses, load_analysis
from phase1_web.multidrug_state import activate_prediction, clear_dependents, discard_stale_dependents, invalidate_prediction


def example_target():
    return {'version':'target_regimen_v1', 'name':'Target trial', 'target_dlt':.25,
        'drugs':[{'drug_id':'D','name':'D','research_priority':1,'research_role':'primary',
                  'mode':'numeric','reference_dose':100.,'unit':'mg',
                  'profile':{'mechanisms':[], 'targets':[], 'classes':[]}}],
        'points':[{'level_id':f'L{i+1}', 'doses':{'D':d}} for i,d in enumerate([50.,100.,200.])],
        'path_level_ids':['L1','L2','L3'], 'profile':{}}


def prediction_result_key(backend, synthesis, power_settings=None, prior=None):
    """Exact settings identify a result; rounded values are display text only."""
    label = backend + ' · ' + synthesis
    if prior is not None:
        from phase1_prediction.multidrug.prior_presets import RISK_PRESETS, canonical_prior, risk_prior_preset
        prior = canonical_prior(prior, backend)
        matching = next((item['label'] for name, item in RISK_PRESETS.items()
                         if risk_prior_preset(name, backend) == prior), 'Custom risk prior')
        label += ' · ' + matching
        if power_settings is not None:
            label += (f" · borrowing {power_settings['strength']:g}, budget {power_settings['patient_budget']}, "
                      f"variation {power_settings['heterogeneity_scale']:g}")
            if backend == 'single_drug_logistic':
                label += f", log-slope variation {power_settings.get('slope_heterogeneity_scale', .5):g}"
        return label + ' | settings=' + fingerprint({'prior': prior.model_dump(), 'power_prior': power_settings})
    if power_settings is None:
        return label
    label += (f" · borrowing {power_settings['strength']:g}, budget {power_settings['patient_budget']}, "
              f"variation {power_settings['heterogeneity_scale']:g}")
    if backend == 'single_drug_logistic':
        label += f", log-slope variation {power_settings.get('slope_heterogeneity_scale', .5):g}"
    return label + ' | settings=' + fingerprint(power_settings)


def prediction_result_label(key):
    label=key.split(' | settings=', 1)[0]
    for value,name in {'same_article_joint_posterior_average':'Posterior average',
                       'same_article_joint_posterior_mixture':'Posterior mixture',
                       'target_power_prior':'Historical likelihood',
                       'cross_article_component_assembly_average':'Component posterior average',
                       'cross_article_component_assembly':'Component posterior mixture',
                       'single_drug_logistic':'Single-drug Bayesian Logistic',
                       'logistic_product':'Product-interaction Logistic',
                       'logistic':'Component Logistic'}.items():
        label=label.replace(value,name)
    return label



def probability_model_controls(st, drugs=None, focal_drug_id=None, *, combination=False):
    """One focal dose curve with a nonnegative combination-background effect."""
    from phase1_web.model_formulas import render_model
    render_model(st, drugs or [{'name': 'Drug'}], focal_drug_id, combination=combination)
    return 'single_drug_logistic', 'Focal drug with combination background'


def _validate_draft_shape(draft, *, single_drug_only=False):
    """Check editable JSON containers without requiring optional target doses."""
    if not isinstance(draft,dict):
        raise ValueError('The target JSON must be an object.')
    for field in ('drugs','points'):
        value=draft.get(field,[])
        if not isinstance(value,list) or any(not isinstance(row,dict) for row in value):
            raise ValueError(f'Target {field} must be a list of objects.')
    for profile in [draft.get('profile',{}),*[d.get('profile',{}) for d in draft.get('drugs',[])]]:
        if not isinstance(profile,dict):
            raise ValueError('Target and drug profiles must be objects.')
    for point in draft.get('points',[]):
        if not isinstance(point.get('doses',{}),dict):
            raise ValueError('Candidate regimen doses must be an object.')
    if single_drug_only and len(draft.get('drugs', [])) > 1:
        raise ValueError('Enter only one target drug. Use the combination-treatment setting for other treatment.')
    if single_drug_only:
        from phase1_prediction.multidrug.single_drug import COMBINATION_PRIOR_SCALES
        analysis = draft.get('profile', {}).get('single_drug_analysis', {})
        if not isinstance(analysis, dict):
            raise ValueError('The single-drug settings must be an object.')
        if analysis.get('combination_present') is not None and not isinstance(analysis['combination_present'], bool):
            raise ValueError('Combination treatment must be true or false.')
        if analysis.get('combination_prior_level', 'medium') not in COMBINATION_PRIOR_SCALES:
            raise ValueError('Choose Low, Medium or High for the combination toxicity prior.')


def _result(result, native):
    if not result.get('methods'):
        return
    method = next(iter(result['methods'].values()))
    if method.get('prediction_engine_version') != TARGET_PREDICTION_VERSION:
        st.warning('This saved result uses an older calculation version. Recompute prediction and simulation before reuse; the historical record is unchanged.')
    st.caption(prediction_result_label(method['backend']+' · '+method['synthesis'])+' · '+method['status'])
    from phase1_web.result_exports import notices
    if method['status'] != 'fitted':
        st.warning('No usable posterior is available for this target. Review the selected evidence and analysis settings, then generate a new prediction.')
        return
    from phase1_web.curve_explorer import render_chart
    render_chart(st, method, key='result_' + result['run_id'])
    for message in notices(result):
        st.info(message)
    diagnostics = method.get('sampling_diagnostics') or {}
    attempts = diagnostics.get('attempts', [])
    if attempts and not attempts[-1].get('passed', False):
        st.warning('Sampling checks did not pass. Recompute this prediction before using it.')
    st.caption('Figures, numerical results and calculation settings are available in Download results.')



def _prediction_output(result, native, comparisons=None):
    comparisons=comparisons or {}
    _result(result,native)
    from phase1_web.results_view import render_comparison
    render_comparison(st, result, comparisons)
    if st.button('Continue to trial simulation'):
        from phase1_web.guided_workflow import go_to
        go_to(st,'OC Simulation')
    if st.button('Download without trial simulation'):
        from phase1_web.guided_workflow import go_to
        go_to(st, 'Download results')
    if st.button('Save analysis',disabled=not result.get('methods')):
        try:
            saved=save_analysis(native.STATE_ROOT,{'prediction':result,'simulation':st.session_state.get('multidrug_simulation'),
                'design_comparison':st.session_state.get('multidrug_search'),
                'draft':result['target'],'active_method':next(iter(result['methods']))})
            st.success('Analysis saved in your private session.' if getattr(native, 'PUBLIC', False) else str(saved))
        except (ValueError,RuntimeError,OSError) as error:
            st.error(str(error))

def main(native):
    st.set_page_config(page_title='PHASE I NEXUS', layout='wide')
    from phase1_web.hosted import public_session, is_public
    native = public_session(native, st)
    public = is_public()
    # Keep BOIN inputs when its widgets are absent on another workflow page.
    for key in list(st.session_state):
        keep_candidate_input=key.startswith('boin_candidates_') and key.endswith(('_source','_edit_units','_unit','_ref','_custom_ref'))
        if key.startswith(('boin_design_','simulation_path_', 'power_prior_', 'prior_settings_', 'explore_', 'workflow_', 'multidrug_weight_', 'threshold_', 'relevance_', 'multidrug_included_'  )) or keep_candidate_input:
            st.session_state[key]=st.session_state[key]
    settings = st.session_state.setdefault('v3_settings', native._initial_settings())
    settings['review_policy'] = 'extracted'
    roots = native._configured_roots(settings)
    library = load_evidence_library(roots, evidence_policy=settings['evidence_policy'], review_policy=settings['review_policy'])
    scopes, diagnostics = load_model_library(roots, evidence_policy=library.evidence_policy, review_policy=library.review_policy)
    context = fingerprint(['prospective_protocol_workflow_1', MATCHING_POLICY, SIMILARITY_VERSION, TARGET_PREDICTION_VERSION, [s.model_dump(mode='json') for s in scopes]])
    if st.session_state.get('multidrug_library_context') != context:
        for key in ('multidrug_prediction', 'multidrug_simulation', 'multidrug_preview', 'multidrug_results', 'multidrug_search', 'multidrug_review_state'):
            st.session_state.pop(key, None)
        st.session_state['multidrug_library_context'] = context
    discard_stale_dependents(st.session_state)
    st.title('PHASE I NEXUS')
    st.caption(f"{diagnostics['document_count']} historical articles available · Plan a trial, review evidence and explore dose–toxicity risks.")
    from phase1_web.guided_workflow import (PAGE_LABELS, initialize_draft, go_to,
        evidence_settings, sync_evidence_selection, evidence_state, render_evidence_controls,
        enforce_automatic_correspondence)
    initialize_draft(st, native.STATE_ROOT)
    from phase1_web.single_drug_plan import migrate_saved_plan, single_drug_plan
    migrate_saved_plan(st, native.STATE_ROOT)
    enforce_automatic_correspondence(st, native.STATE_ROOT)
    pending=st.session_state.pop('workflow_next_page',None)
    if pending in PAGE_LABELS:
        st.session_state['workflow_page']=pending
    pages = {key: value for key, value in PAGE_LABELS.items()
             if not public or key not in {'Settings', 'Extraction Jobs'}}
    if public and native.LIBRARY_SOURCE == 'personal':
        pages['Upload literature'] = 'Upload and extract literature'
    if st.session_state.get('workflow_page') not in pages:
        st.session_state['workflow_page'] = next(iter(pages))
    page = st.sidebar.radio('Workflow', list(pages), format_func=pages.__getitem__, key='workflow_page')
    if page == 'Upload literature':
        from phase1_web.user_library import render_user_library
        render_user_library(st, native.WORKSPACE)
        return
    st.sidebar.caption('Your trial draft saves automatically. Simulation is optional.')
    previous_page=st.session_state.get('multidrug_previous_page')
    st.session_state['multidrug_previous_page']=page
    if page == 'Library':
        native._library_page(roots, library, model_scopes=scopes)
        st.subheader('Historical regimen records')
        st.dataframe([{'article':s.document_id,'scope':s.scope_id,'regimen':s.label,
                       'drugs':' + '.join(d.name for d in s.drugs),'observations':len(s.observations),
                       'source_arms':', '.join(a or 'Ungrouped' for a in s.lineage.get('arm_ids',[s.lineage['scope_key'][2]])),
                       'phase_eligibility':phase_status(s.profile),
                       'status':'has outcomes' if s.observations else 'prior only'} for s in scopes], hide_index=True)
        with st.expander('Input diagnostics and exclusions'):
            st.json({'diagnostics':diagnostics,'excluded':[{'scope':s.scope_id,'records':s.exclusions} for s in scopes if s.exclusions],
                     'grouping':[{'scope':s.scope_id,**s.lineage['grouping']} for s in scopes if s.lineage.get('grouping')]})
        if not public:
            with st.expander('Precompute and cache Phase I article posteriors'):
                from phase1_web.prior_cache import render_source_cache
                render_source_cache(st, scopes, backends=('single_drug_logistic',),
                                    allow_backend_selection=False, key='library_prior_cache')
    elif page == 'Article comparison':
        from phase1_web.article_comparison import render
        render(st, library)
    elif page == 'Extraction Jobs': native._jobs_page()
    elif page == 'Settings': native._settings_page(settings)
    elif page == 'Target Trial':
        from phase1_web.plan_navigation import REVIEW_REQUEST_KEY, STEP_KEY, STEPS
        review_requested = st.session_state.pop(REVIEW_REQUEST_KEY, False)
        st.subheader('Plan a new trial')
        st.caption('Describe a new Phase I trial. Doses and unknown clinical details can be left blank; historical outcomes are not requested.')
        path = native.STATE_ROOT / 'multidrug_target_draft.json'
        from phase1_web.multidrug_form import render_draft, blank_target
        from phase1_web.multidrug_examples import fixed_example_scope, target_from_scope
        example_scope = fixed_example_scope(scopes)
        new_trial, load_example = st.columns(2)
        replacement = None
        if new_trial.button('Start new trial', width='stretch'):
            replacement = blank_target()
        if load_example.button('Load example trial', width='stretch', disabled=example_scope is None,
                help='Load the extracted terameprocol trial from PMC3309850.' if example_scope else
                     'The fixed example (PMC3309850) is not available in this library. Use the Shared research library.'):
            try:
                replacement = single_drug_plan(target_from_scope(example_scope).model_dump(mode='json'))
                replacement['name'] = 'Example trial · PMC3309850 · terameprocol'
            except ValueError as error:
                st.error(str(error))
        if replacement is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = st.session_state['multidrug_target_text']
            st.session_state['multidrug_previous_target_text'] = previous
            path.with_name('multidrug_target_draft.previous.json').write_text(previous)
            st.session_state['multidrug_target_text'] = json.dumps(replacement, ensure_ascii=False, indent=2)
            st.session_state['multidrug_reset_editor'] = True
            st.session_state['multidrug_advanced_editor'] = False
            st.rerun()
        if st.session_state.get('multidrug_previous_target_text') and st.button('Restore previous draft', type='tertiary'):
            st.session_state['multidrug_target_text'] = st.session_state.pop('multidrug_previous_target_text')
            path.with_name('multidrug_target_draft.previous.json').unlink(missing_ok=True)
            st.session_state['multidrug_reset_editor'] = True
            st.rerun()
        advanced = st.checkbox('Advanced JSON editor', False, key='multidrug_advanced_editor')
        mode_changed=advanced!=st.session_state.get('multidrug_advanced_previous',False)
        requested_reset=st.session_state.pop('multidrug_reset_editor',False)
        requested_rebase=st.session_state.pop('multidrug_rebase_editor',False)
        rebase=mode_changed or previous_page!='Target Trial' or requested_reset or requested_rebase
        if requested_reset:
            invalidate_prediction(st.session_state)
            st.session_state.pop('multidrug_review_state', None)
            st.session_state.pop('trial_plan_step',None)
            st.session_state.pop(REVIEW_REQUEST_KEY, None)
            review_requested = False
            for key in list(st.session_state):
                if key.startswith('trial_plan_rule_'):st.session_state.pop(key,None)
            for key in ('workflow_target_dlt','workflow_min_drug_similarity','workflow_min_clinical_coverage'):
                st.session_state.pop(key,None)
        if rebase or 'multidrug_form_base' not in st.session_state:
            # Editor deltas always apply to a stable base until a mode/load change.
            for key in list(st.session_state):
                if (requested_reset and key.startswith('explore_')) or key.startswith('plan_') or key=='multidrug_drug_editor' or key.startswith(('multidrug_phases_','multidrug_adjustments_','multidrug_schedule_','regimen_points_')):
                    del st.session_state[key]
            st.session_state['multidrug_form_base']=st.session_state['multidrug_target_text']
            st.session_state['multidrug_target_json_editor']=st.session_state['multidrug_target_text']
        st.session_state['multidrug_advanced_previous']=advanced
        try:
            if advanced:
                text = st.text_area('Protocol and drug attributes (JSON)', key='multidrug_target_json_editor', height=420)
                draft=json.loads(text)
            else:
                draft=render_draft(json.loads(st.session_state['multidrug_form_base']), single_drug_only=True)
                text=json.dumps(draft,ensure_ascii=False,indent=2)
            _validate_draft_shape(draft, single_drug_only=True)
            if draft.get('mapping_overrides'):
                enforce_automatic_correspondence(st, native.STATE_ROOT, draft)
                text=json.dumps(draft,ensure_ascii=False,indent=2)
            from phase1_extraction.clinical_protocol import planned_issues
            issues=planned_issues(draft,require_doses=False)
            previous_draft=json.loads(st.session_state['multidrug_target_text'])
            for field,key in [('target_dlt','workflow_target_dlt'),('minimum_clinical_coverage','workflow_min_clinical_coverage')]:
                if draft.get(field) != previous_draft.get(field):
                    st.session_state.pop(key,None)
            st.session_state['multidrug_target_text']=text
        except (ValueError,TypeError,AttributeError,KeyError) as error:
            invalidate_prediction(st.session_state)
            st.error(str(error));return
        path.parent.mkdir(parents=True,exist_ok=True)
        if st.session_state.get('workflow_saved_draft') != fingerprint(draft):
            activate_prediction(st.session_state,None)
            st.session_state['multidrug_results']={}
            st.session_state['workflow_saved_draft']=fingerprint(draft)
        if not path.exists() or path.read_text()!=text:path.write_text(text)
        if issues:
            for issue in issues:st.info(issue)
            invalidate_prediction(st.session_state)
            return
        source = draft.get('profile', {}).get('example_source', {})
        if source.get('document_id'):
            st.caption('Example source: ' + source['document_id'] + ' · Extracted clinical information and doses; editable as a new trial plan.')
        if review_requested:
            go_to(st,'Review evidence')
        if advanced or st.session_state.get(STEP_KEY) != STEPS[-1]:
            st.caption('Draft saved. You can review evidence now or continue adding plan details.')
            if st.button('Review evidence with current plan',type='primary' if advanced else 'secondary'):
                go_to(st,'Review evidence')
    elif page == 'Review evidence':
        st.subheader('2. Review evidence')
        st.write('Choose the historical evidence to use for your planned trial. Doses are not required for this step.')
        if st.button('Back to trial plan'):
            go_to(st,'Target Trial')
        draft=json.loads(st.session_state['multidrug_target_text'])
        if not any(d.get('name','').strip() for d in draft.get('drugs',[])):
            st.info('Add a drug name or development code in Trial plan first.')
            if st.button('Go to trial plan'):go_to(st,'Target Trial')
            return
        config=render_evidence_controls(st,draft)
        text=json.dumps(draft,ensure_ascii=False,indent=2)
        st.session_state['multidrug_target_text']=text
        (native.STATE_ROOT/'multidrug_target_draft.json').write_text(text)
        st.session_state['workflow_saved_draft']=fingerprint(draft)
        if sum(config['weights'].values())<=0:
            invalidate_prediction(st.session_state)
            st.error('At least one indicator weight must be positive.');return
        selection_spec=sync_evidence_selection(st,draft,config,context)
        if st.button('Find matching articles',type='primary'):
            try:
                with st.spinner('Comparing your plan with the Phase I library'):
                    if public:
                        from phase1_web.jobs import run_task
                        st.session_state['multidrug_preview'] = run_task('preview', {
                            'roots': list(library.source_roots), 'target': draft,
                            'options': config['query']}, native.WORKSPACE)
                    else:
                        st.session_state['multidrug_preview']=preview_planned(library,draft,**config['query'])
            except (ValueError,RuntimeError,OSError) as error:st.error(str(error))
        if st.session_state.get('multidrug_preview') is None:
            st.info('Find matching articles to review their original sources.')
            return
        candidates=st.session_state['multidrug_preview']['selected_series']
        coverage_skips = [row for row in st.session_state['multidrug_preview'].get('diagnostics', {}).get('notices', [])
                          if row.get('reason') == 'insufficient_clinical_coverage']
        if coverage_skips:
            st.caption(f"{len({row['document_id'] for row in coverage_skips})} article(s) have evidence groups below the coverage threshold; those groups were skipped before similarity comparison.")
        if not candidates:
            st.info('No articles meet these filters. Add missing clinical details or adjust the coverage threshold and candidate pool.')
        from phase1_web.evidence_review import render_article_selection, render_review
        included=render_article_selection(st,candidates,scopes,selection_spec)
        decisions,review_ready=render_review(st,candidates,scopes,native.STATE_ROOT,selection_spec,config['review_mode'],included_scope_ids=included,
            target_drug_names={d['drug_id']:d['name'] for d in draft['drugs']},
            library_source=native.LIBRARY_SOURCE if public else 'local',
            workspace=native.WORKSPACE if public else None)
        new_review={'identity':selection_spec,'included':included,'decisions':decisions,'ready':review_ready}
        if new_review != st.session_state.get('multidrug_review_state'):
            activate_prediction(st.session_state,None)
            st.session_state['multidrug_results']={}
        st.session_state['multidrug_review_state']=new_review
        if st.button('Continue to risk curves',type='primary',disabled=not review_ready):
            go_to(st,'Prediction Results')
    elif page == 'Prediction Results':
        st.subheader('Explore risks')
        existing=st.session_state.get('multidrug_prediction')
        previous_method = next(iter((existing or {}).get('methods', {}).values()), {})
        if (previous_method.get('backend') == 'single_drug_logistic'
                and previous_method.get('synthesis') == 'target_power_prior'
                and previous_method.get('power_prior_version') != POWER_PRIOR_VERSION):
            activate_prediction(st.session_state, None)
            st.session_state['multidrug_results'] = {}
            existing = None
            st.info('Historical likelihood now includes random intercepts and slopes. Generate a new prediction with your saved plan and evidence.')
        if existing and next(iter(existing.get('methods', {}).values()), {}).get('backend',
                existing.get('evidence_contract', {}).get('backend')) not in (None, 'single_drug_logistic'):
            activate_prediction(st.session_state, None)
            st.session_state['multidrug_results'] = {}
            st.session_state.pop('multidrug_comparison_request', None)
            existing = None
        if existing and not st.session_state.get('multidrug_comparison_request'):
            _prediction_output(existing,native)
            if st.button('Set up new risk exploration'):
                activate_prediction(st.session_state,None)
                st.rerun()
            return
        draft=json.loads(st.session_state['multidrug_target_text'])
        if not any(d.get('name','').strip() for d in draft.get('drugs',[])):
            st.info('Add a drug in Trial plan before exploring risks.')
            if st.button('Go to trial plan'):go_to(st,'Target Trial')
            return
        from phase1_web.curve_explorer import coordinate_editor
        from phase1_web.guided_workflow import sync_exploration_units
        sync_exploration_units(st,draft)
        target=coordinate_editor(st,draft)
        with st.container():
            st.subheader('Analysis settings')
            target_probability=st.number_input('Target DLT probability',min_value=.01,max_value=.99,value=float(draft.get('target_dlt',.25)),step=.01,key='workflow_target_dlt')
            if target is not None:target=target.model_copy(update={'target_dlt':target_probability})
            if draft.get('target_dlt') != target_probability:
                draft['target_dlt']=target_probability
                text=json.dumps(draft,ensure_ascii=False,indent=2)
                st.session_state['multidrug_target_text']=text
                (native.STATE_ROOT/'multidrug_target_draft.json').write_text(text)
                st.session_state['workflow_saved_draft']=fingerprint(draft)
            st.caption('The desired toxicity rate for selecting a dose. This value carries into BOIN without re-entry.')
            model_drugs = target.drugs if target is not None else draft['drugs']
            focal_id = (target.profile.get('single_drug_analysis', {}).get('focal_drug_id') if target is not None
                        else (st.session_state.get('explore_axes') or [None])[0])
            from phase1_web.model_formulas import render_aggregation, target_combination
            combination = (target_combination(target) if target is not None else
                           draft.get('profile', {}).get('single_drug_analysis', {}).get('combination_present', False if len(model_drugs) == 1 else None))
            backend, construction = probability_model_controls(st, model_drugs, focal_id, combination=combination)
            synthesis = render_aggregation(st, combination=combination)
            from phase1_web.prior_settings import render_prior_settings
            try:
                selected_prior, power_settings, _prior_choice = render_prior_settings(
                    st, backend, synthesis, native.STATE_ROOT, drug_count=len(model_drugs),
                    drug_modes=[getattr(d, 'mode', None) or d.get('mode', 'numeric') for d in model_drugs],
                    combination=combination,
                    combination_prior_level=draft.get('profile', {}).get('single_drug_analysis', {}).get('combination_prior_level', 'medium') if combination else None)
            except (ValueError, OSError) as error:
                invalidate_prediction(st.session_state)
                st.error('Check the prior settings: ' + str(error))
                return
            draw_count=st.number_input('Target joint draws',100,5000,1000,100,key='workflow_draw_count')
        config=evidence_settings(st)
        selection_spec=sync_evidence_selection(st,draft,config,context)
        included,decisions,review_ready=evidence_state(st,selection_spec,config['review_mode'])
        threshold=config['query']['threshold']; weights=config['weights']; top=config['query']['top_articles']
        engine=config['query']['similarity_engine']; embedding_model=config['query']['embedding_model']
        if st.session_state.get('multidrug_preview') is not None:
            rows=st.session_state['multidrug_preview']['selected_series']
            doc_ids={r['document_id'] for r in rows if r['series_key'] in included and (decisions or {}).get(r['document_id'])!='Exclude'}
            st.caption(f"Evidence selected: {len(doc_ids)} articles · " + ('Abstract review' if config['review_mode'] else 'Automatic inclusion'))
        else:
            st.info('No evidence list has been previewed. Automatic inclusion will find articles using your saved filters when you generate a prediction.' if not config['review_mode'] else 'Complete article review before generating a prediction.')
        if target is None:
            activate_prediction(st.session_state,None)
            st.session_state['multidrug_results']={}
            return
        if public:
            st.caption('Compatible published article posteriors are reused automatically. Missing fits are queued; target-specific calculations stay private to this session.')
        else:
            with st.expander('Prior cache'):
                from phase1_web.prior_cache import render_source_cache, render_target_cache
                if synthesis == 'target_power_prior':
                    render_target_cache(st, library, target, st.session_state.get('multidrug_preview'),
                        included, decisions, weights, threshold, top, backend, power_settings,
                        review_ready, key='target_prior_cache_' + backend, selected_prior=selected_prior, scopes=scopes)
                else:
                    render_source_cache(st, scopes, backends=(backend,), allow_backend_selection=False,
                        key='target_source_cache_' + backend, selected_prior=selected_prior, target=target)
        comparison_spec = fingerprint([selection_spec,target.model_dump(mode='json'),draw_count,sorted(included),decisions])
        if comparison_spec != st.session_state.get('multidrug_comparison_request'):
            st.session_state['multidrug_results']={}
            activate_prediction(st.session_state,None)
            st.session_state['multidrug_comparison_request']=comparison_spec
        result_key=prediction_result_key(backend,synthesis,power_settings,selected_prior)
        spec = fingerprint([comparison_spec,result_key])
        if spec != st.session_state.get('multidrug_request'):
            activate_prediction(st.session_state,st.session_state.get('multidrug_results',{}).get(result_key))
            st.session_state['multidrug_request'] = spec
        if st.button('Generate prediction', type='primary',disabled=not review_ready):
            progress = st.empty()
            def update(done,total,article,model):
                progress.info(f'{done}/{total} · {article} · {model} · checking cache / fitting if needed')
            try:
                if st.session_state.get('multidrug_preview') and not included:
                    raise ValueError('Select at least one article or find matching articles again.')
                options = dict(backend=backend,synthesis=synthesis,threshold=threshold,weights=weights,
                    top_articles=top,include_series_keys=included,draw_count=int(draw_count),
                    relevance_decisions=decisions,power_prior=power_settings,prior=selected_prior,
                    similarity_engine=engine,embedding_model=embedding_model)
                if public:
                    from phase1_web.jobs import run_task
                    progress.info('Queued calculations run one at a time so other visitors can keep using the website.')
                    result = run_task('predict', {'roots':list(library.source_roots),
                        'target':target.model_dump(mode='json'),
                        'options':{**options, 'prior':selected_prior.model_dump(mode='json')},
                        'shared_posterior_root':str(native.SHARED_POSTERIOR_ROOT) if native.SHARED_POSTERIOR_ROOT else None},
                        native.WORKSPACE)
                else:
                    result = predict(library,target,progress=update,**options).to_dict()
                activate_prediction(st.session_state,result,recomputed=True)
                st.session_state['multidrug_result_backend'] = result_key
                st.session_state.setdefault('multidrug_results',{})[result_key]=result
                progress.success('Calculation complete. Your risk curves are shown below.')
            except (ValueError, RuntimeError, OSError) as error: st.error(str(error))
        result=st.session_state.get('multidrug_prediction')
        if result:
            _prediction_output(result,native,st.session_state.get('multidrug_results',{}))
    elif page == 'OC Simulation':
        result = st.session_state.get('multidrug_prediction')
        if not result: st.info('Generate a target prediction first.'); return
        st.subheader('Simulate a trial')
        st.write('BOIN uses your candidate doses in increasing order. Each virtual trial uses one complete posterior risk curve.')
        from phase1_web.multidrug_state import simulation_view, automatic_boin_path
        with st.container(border=True):
            st.markdown('**Candidate doses**')
            from phase1_web.curve_explorer import boin_candidate_editor
            try:
                candidates=boin_candidate_editor(st,result,rebase=previous_page!='OC Simulation')
                if candidates is None:return
                current_points=result['target']['points']
                needs_evaluation=(candidates.model_dump(mode='json')['points']!=current_points or
                    candidates.model_dump(mode='json')['drugs']!=result['target']['drugs'] or
                    result['target'].get('profile',{}).get('planning',{}).get('exploration',{}).get('relative',False))
                if needs_evaluation:
                    if st.button('Evaluate BOIN candidate regimens'):
                        from phase1_web.jobs import evaluate_for_ui
                        new=evaluate_for_ui(result,candidates)
                        activate_prediction(st.session_state,new,recomputed=True)
                        st.session_state['multidrug_results']={new['evidence_contract']['backend']:new}
                        st.rerun()
                    st.info('Evaluate these candidate doses before running a simulation.');return
                st.caption('These candidate doses are already evaluated.')
            except (ValueError,OSError,RuntimeError) as error:st.info(str(error));return
        target=result['target']
        if target.get('profile',{}).get('planning',{}).get('exploration',{}).get('relative'):
            st.info('Enter and evaluate actual candidate doses above before BOIN.');return
        try:
            path_ids=automatic_boin_path(target)
        except ValueError as error:
            st.info(str(error));return
        by_id={point['level_id']:point for point in target['points']}
        numeric=[d for d in target['drugs'] if d['mode']=='numeric']
        focal_id=target.get('profile',{}).get('single_drug_analysis',{}).get('focal_drug_id')
        focal=next((d for d in numeric if d['drug_id']==focal_id), None)
        focal=focal or next((d for d in numeric if len({p['doses'][d['drug_id']] for p in target['points']})>1), numeric[0])
        dose_order=' → '.join(f"{by_id[level]['doses'][focal['drug_id']]:g}" for level in path_ids)
        st.markdown('**Dose order:** ' + dose_order + ' ' + focal['unit'])
        path_key=fingerprint([result['run_id'],path_ids])
        if st.session_state.get('simulation_path_identity')!=path_key:
            clear_dependents(st.session_state)
            st.session_state['simulation_path_identity']=path_key
        simulation_input=simulation_view(result,path_ids)
        mode=st.radio('Simulation task', ['Simulate one design', 'Compare multiple designs'],
                      horizontal=True, key='boin_design_task')
        st.caption('One design uses one sample size and cohort size. Multiple designs compare combinations of these settings using the same simulated risk scenarios.')
        st.subheader('Simulation settings')
        if public:
            st.caption('Public calculations allow up to 20,000 trials per design and 50,000 across a design comparison.')
        st.caption(f"Applies to Run BOIN and Compare designs. Target DLT probability: {result['target']['target_dlt']:.2f}, inherited from the trial plan.")
        m=st.number_input('Posterior scenarios',1,1000,100,key='boin_design_scenarios')
        repetitions=st.number_input('Trials per scenario',1,1000,100,key='boin_design_repetitions')
        with st.container(border=True):
            st.markdown('**BOIN decision and stopping rules**')
            elimination=st.number_input('Overtoxicity elimination probability',.5,.999,.95,.01,key='boin_design_elimination')
            stopping=st.number_input('Patients at one regimen before stopping',3,100,12,3,key='boin_design_stopping')
            if st.session_state.get('boin_design_start',1)>len(path_ids):
                st.session_state['boin_design_start']=len(path_ids)
            start_index=st.number_input('Starting dose level (1 = lowest dose)',1,len(path_ids),1,key='boin_design_start')
            low=st.number_input('Acceptable DLT probability: lower bound',.0,.99,.20,.01,key='boin_design_low')
            high=st.number_input('Acceptable DLT probability: upper bound',.01,1.,.35,.01,key='boin_design_high')
        common_request=fingerprint([path_key, elimination, stopping, start_index, low, high, m, repetitions])
        previous_common=st.session_state.get('boin_design_common_request')
        if previous_common is not None and previous_common!=common_request:
            clear_dependents(st.session_state)
        st.session_state['boin_design_common_request']=common_request
        if mode=='Simulate one design':
            st.subheader('Run one BOIN design')
            nmax=st.number_input('Maximum sample size',6,300,30,3,key='boin_design_nmax')
            cohort=st.number_input('Patients per cohort',1,12,3,key='boin_design_cohort')
            boin_settings=BoinSettings(target_dlt=result['target']['target_dlt'],max_sample_size=int(nmax),cohort_size=int(cohort),
                elimination_cutoff=elimination,stop_rule_m=int(stopping),start_index=int(start_index)-1,
                acceptable_low=low,acceptable_high=high)
            simulation_request=fingerprint([path_key,asdict(boin_settings),m,repetitions])
            if st.session_state.get('multidrug_simulation_request')!=simulation_request:
                st.session_state.pop('multidrug_simulation',None)
                st.session_state['multidrug_simulation_request']=simulation_request
            if st.button('Run BOIN'):
                try:
                    with st.spinner('Simulating'):
                        options = dict(method_key=next(iter(result['methods'])), settings=boin_settings,
                            posterior_curve_count=int(m), trials_per_curve=int(repetitions))
                        if public:
                            from phase1_web.jobs import run_task
                            sim = run_task('simulate', {'prediction':simulation_input,
                                'options':{**options, 'settings':asdict(boin_settings)}}, native.WORKSPACE)
                        else:
                            sim=simulate_regimen_path(simulation_input,**options)
                    st.session_state['multidrug_simulation']=sim
                except (ValueError, RuntimeError) as error: st.error(str(error))
            if st.session_state.get('multidrug_simulation'):
                sim=st.session_state['multidrug_simulation']
                from phase1_web.results_view import render_simulation
                render_simulation(st, result, sim)
        else:
            st.subheader('Compare BOIN designs')
            st.caption('Choose two or more combinations of sample size and cohort size to compare.')
            cohorts=st.multiselect('Candidate cohort sizes',[1,2,3,4,6],[3],key='boin_design_search_cohorts')
            sizes=st.multiselect('Candidate maximum sample sizes',[12,18,24,30,36,42,48,60],[30,36],key='boin_design_search_sizes')
            search_request=fingerprint([path_key,cohorts,sizes,elimination,stopping,start_index,low,high,m,repetitions])
            if st.session_state.get('multidrug_search_request')!=search_request:
                st.session_state.pop('multidrug_search',None)
                st.session_state['multidrug_search_request']=search_request
            design_count=len(cohorts)*len(sizes)
            st.caption(f'{design_count} designs selected.')
            if st.button('Compare designs', disabled=design_count<2):
                try:
                    with st.spinner('Comparing designs using the same posterior scenarios and random seed'):
                        options = dict(method_key=next(iter(result['methods'])),
                            target_dlt=[result['target']['target_dlt']],cohort_size=cohorts,max_sample_size=sizes,
                            elimination_cutoff=[elimination],stop_rule_m=[int(stopping)],
                            base_settings=BoinSettings(start_index=int(start_index)-1,acceptable_low=low,acceptable_high=high),
                            posterior_curve_count=int(m),trials_per_curve=int(repetitions))
                        if public:
                            from phase1_web.jobs import run_task
                            search = run_task('search', {'prediction':simulation_input,
                                'options':{**options, 'base_settings':asdict(options['base_settings'])}}, native.WORKSPACE)
                        else:
                            search=search_boin_parameters(simulation_input,**options)
                    st.session_state['multidrug_search']=search
                except (ValueError,RuntimeError) as error:st.error(str(error))
            if st.session_state.get('multidrug_search'):
                search=st.session_state['multidrug_search']
                from phase1_web.results_view import render_design_comparison
                render_design_comparison(st, result, search)
        if st.button('Continue to downloads', type='primary'):
            go_to(st, 'Download results')
    elif page == 'Download results':
        from phase1_web.results_view import render_downloads
        render_downloads(st, st.session_state.get('multidrug_prediction'),
                         st.session_state.get('multidrug_simulation'), st.session_state.get('multidrug_search'))
    elif page == 'Saved analyses':
        records=list_analyses(native.STATE_ROOT)
        if not records: st.info('No saved analyses yet.'); return
        st.dataframe(records, hide_index=True)
        st.caption('Historical analyses retain their original results and calculation versions.')
        chosen=st.selectbox('Saved analysis',list(range(len(records))),format_func=lambda i:records[i]['name']+' · '+records[i]['saved_at_utc'])
        record=load_analysis(Path(records[chosen]['path']))
        if record.get('prediction',{}).get('model_version') in {'v3_multidrug_bayes_1', VERSION}:
            _result(record['prediction'],native)
            from phase1_web.results_view import render_downloads
            render_downloads(st, record['prediction'], record.get('simulation'), record.get('design_comparison'),
                             key='saved_results_' + record['analysis_id'])
        else:
            st.info('This older analysis format is retained. Generate a new prediction to use the current report export.')
