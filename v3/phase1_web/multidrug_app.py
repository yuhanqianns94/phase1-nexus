"""Native full-regimen Bayesian workspace; historical analyses remain readable."""
import json
import copy
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
        return label + ' | settings=' + fingerprint({'prior': prior.model_dump(), 'power_prior': power_settings})
    if power_settings is None:
        return label
    label += (f" · borrowing {power_settings['strength']:g}, budget {power_settings['patient_budget']}, "
              f"variation {power_settings['heterogeneity_scale']:g}")
    return label + ' | settings=' + fingerprint(power_settings)


def prediction_result_label(key):
    label=key.split(' | settings=', 1)[0]
    for value,name in {'same_article_joint_posterior_average':'Method 1: weighted average',
                       'same_article_joint_posterior_mixture':'Method 2: curve mixture',
                       'target_power_prior':'Method 3: power prior',
                       'cross_article_component_assembly_average':'Method 1: component assembly average',
                       'cross_article_component_assembly':'Method 2: component assembly mixture',
                       'logistic_product':'Product-interaction Logistic',
                       'logistic':'Component Logistic'}.items():
        label=label.replace(value,name)
    return label



def probability_model_controls(st):
    """Keep model selection distinct from evidence construction and aggregation."""
    from phase1_web.prior_cache import MODEL_LABELS
    backend = st.selectbox('Probability model', list(BACKENDS),
        format_func=MODEL_LABELS.__getitem__, key='workflow_probability_model')
    if backend == 'logistic_product':
        st.latex(r'p(\mathbf r)=\sigma\left(\alpha+\sum_j e^{\ell_j}r_j+\sum_{S\in\mathcal I}\eta_S\prod_{j\in S}r_j\right)')
        st.caption('One common intercept, positive linear relative-dose effects, and product interactions among the first three drugs. Negative interactions can produce decreasing conditional curves. At all-zero doses the risk is sigmoid(alpha).')
        construction = st.selectbox('Evidence construction', ['Complete-regimen borrowing'],
            key='workflow_product_construction')
        st.caption('Component assembly is available with Component Logistic. Separate ordinary Logistic source fits do not define one shared baseline intercept for an assembled target model.')
    else:
        st.caption('Each drug has a dose-risk component; the full risk combines these components with bounded interaction effects.')
        construction = st.selectbox('Evidence construction',
            ['Complete-regimen borrowing', 'Component assembly (advanced)'], key='workflow_construction')
    return backend, construction


def _validate_draft_shape(draft):
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


def _result(result, native):
    if not result.get('methods'):
        return
    method = next(iter(result['methods'].values()))
    if method.get('prediction_engine_version') != TARGET_PREDICTION_VERSION:
        st.warning('This saved result uses an older calculation version. Recompute prediction and simulation before reuse; the historical record is unchanged.')
    st.caption(prediction_result_label(method['backend']+' · '+method['synthesis'])+' · '+method['status'])
    if result.get('evidence_contract', {}).get('prior'):
        from phase1_web.prior_settings import prior_parameter_rows
        with st.expander('Applied parameter priors'):
            st.dataframe(prior_parameter_rows(result['evidence_contract']['prior'], method['backend']),
                         hide_index=True, use_container_width=True)
    if method['synthesis'] == 'target_power_prior':
        st.caption('Method 3: historical information forms a prior for a new trial. Each curve includes a new-study risk shift; no target patient outcomes have been observed.')
        if method.get('inference') == 'explicit_prior_predictive':
            st.warning('All historical powers are zero. These curves come entirely from the specified priors.')
        st.write('Discounted patient count (not an exact effective sample size):', method.get('discounted_patient_count', 0.))
        with st.expander('Power prior settings and sampling diagnostics'):
            st.json({'settings': method.get('power_prior'), 'diagnostics': method.get('sampling_diagnostics')})
    if method.get('evidence_construction') == 'component_assembly':
        st.caption(method['aggregation_note'])
        st.caption('Weights are normalized separately for each target drug. Historical single-drug studies can supply components of a combination.')
        if any(row['fixed_prior_fraction'] > 0 for row in method.get('interaction_support', [])):
            st.info('Some assembled interactions have no shared source fit. Their coefficients are drawn from the selected interaction prior; drug similarity does not estimate them.')
        with st.expander('Component sources and interaction origins'):
            st.dataframe([{'target_drug': pool['target_id'], **c}
                          for pool in method.get('component_pools', []) for c in pool['candidates']],
                         hide_index=True, use_container_width=True)
            st.dataframe(method.get('interaction_support', []), hide_index=True)
            st.caption('Fractions describe average recipe weights for Method 1 and sampled recipe frequencies for Method 2. A source joint draw preserves its fitted interaction; it does not establish that the interaction is well identified.')
        computation = method.get('assembly_computation', {})
        if computation.get('evaluation') == 'monte_carlo_recipe_average':
            st.info(f"Method 1 approximates the large assembly space using {computation['recipe_sample_count']} sampled recipes ({computation['evaluated_recipe_count']} distinct). The same recipe bank is reused across doses and saved-curve exploration. Intervals do not include recipe-integration error.")
    if method['status'] != 'fitted':
        st.warning('No usable posterior for this regimen. Check drug correspondence, data and sampling status below.')
    else:
        frame = pd.DataFrame(method['summary'])
        st.dataframe(frame, hide_index=True, use_container_width=True)
        with st.expander('Dose support and extrapolation'):
            st.dataframe(method.get('coordinate_support',[]),hide_index=True)
        from phase1_web.curve_explorer import render_chart
        render_chart(st,method,key='result_'+result['run_id'])
    with st.expander('Drug correspondence, source weights and exclusions'):
        st.json({'contributors':method['contributors'], 'excluded':method['excluded'],
                 'selection_frequencies':method.get('selection_frequencies'), 'evidence':result['evidence_contract']})
    native._json_download('Download prediction and joint draws', result, result['run_id']+'.json')



def _prediction_output(result, native, comparisons=None):
    comparisons=comparisons or {}
    _result(result,native)
    method = next(iter(result.get('methods', {}).values()), {})
    backend = method.get('backend', result.get('evidence_contract', {}).get('backend'))
    if backend == 'logistic_product':
        st.caption('The common intercept, main effects and interactions are estimated jointly. Limited dose variation can make their individual estimates strongly prior-sensitive.')
    else:
        st.caption('Component risks inferred from combination data are model-derived. With fixed doses or no monotherapy data, slopes and decompositions may depend strongly on priors.')
    if len(comparisons)>1:
        rows=[{'backend':prediction_result_label(b),**row} for b,r in comparisons.items() for row in next(iter(r.get('methods',{}).values()),{}).get('summary',[])]
        st.subheader('Model comparison for the same target')
        st.dataframe(rows,hide_index=True)
    if st.button('Continue to trial simulation'):
        from phase1_web.guided_workflow import go_to
        go_to(st,'OC Simulation')
    if st.button('Save analysis',disabled=not result.get('methods')):
        try:
            saved=save_analysis(native.STATE_ROOT,{'prediction':result,'simulation':st.session_state.get('multidrug_simulation'),
                'design_comparison':st.session_state.get('multidrug_search'),
                'draft':result['target'],'active_method':next(iter(result['methods']))})
            st.success('Analysis saved in your private session.' if getattr(native, 'PUBLIC', False) else str(saved))
        except (ValueError,RuntimeError,OSError) as error:
            st.error(str(error))

def main(native):
    st.set_page_config(page_title='Phasal V3 Bayesian', layout='wide')
    from phase1_web.hosted import public_session, is_public
    native = public_session(native, st)
    public = is_public()
    # Keep BOIN inputs when its widgets are absent on another workflow page.
    for key in list(st.session_state):
        keep_candidate_input=key.startswith('boin_candidates_') and key.endswith(('_source','_edit_units','_unit','_ref'))
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
    st.title('Phasal · V3 Bayesian')
    st.caption(f"{diagnostics['document_count']} historical articles available · Plan a trial, review evidence and explore dose–toxicity risks.")
    from phase1_web.guided_workflow import (PAGE_LABELS, initialize_draft, go_to,
        evidence_settings, sync_evidence_selection, evidence_state, render_evidence_controls)
    initialize_draft(st, native.STATE_ROOT)
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
        st.subheader('Complete-regimen model inputs')
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
                render_source_cache(st, scopes, backends=BACKENDS, key='library_prior_cache')
    elif page == 'Article comparison':
        from phase1_web.article_comparison import render
        render(st, library)
    elif page == 'Extraction Jobs': native._jobs_page()
    elif page == 'Settings': native._settings_page(settings)
    elif page == 'Target Trial':
        st.subheader('Plan a new trial')
        st.caption('Describe a new Phase I trial. Doses and unknown clinical details can be left blank; historical outcomes are not requested.')
        path = native.STATE_ROOT / 'multidrug_target_draft.json'
        from phase1_web.multidrug_form import render_draft, blank_target
        with st.popover('Draft tools'):
            if st.session_state.get('multidrug_previous_target_text') and st.button('Restore previous draft'):
                st.session_state['multidrug_target_text']=st.session_state.pop('multidrug_previous_target_text')
                st.session_state['multidrug_reset_editor']=True
            if st.button('Start a new trial'):
                st.session_state['multidrug_previous_target_text']=st.session_state['multidrug_target_text']
                st.session_state['multidrug_target_text']=json.dumps(blank_target())
                st.session_state['multidrug_reset_editor']=True
                st.rerun()
            advanced = st.checkbox('Advanced JSON editor',False)
        mode_changed=advanced!=st.session_state.get('multidrug_advanced_previous',False)
        requested_reset=st.session_state.pop('multidrug_reset_editor',False)
        requested_rebase=st.session_state.pop('multidrug_rebase_editor',False)
        rebase=mode_changed or previous_page!='Target Trial' or requested_reset or requested_rebase
        if requested_reset:
            st.session_state.pop('trial_plan_step',None)
            for key in list(st.session_state):
                if key.startswith('trial_plan_rule_'):st.session_state.pop(key,None)
            for key in ('workflow_target_dlt','workflow_min_drug_similarity'):
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
                draft=render_draft(json.loads(st.session_state['multidrug_form_base']))
                text=json.dumps(draft,ensure_ascii=False,indent=2)
            _validate_draft_shape(draft)
            from phase1_extraction.clinical_protocol import planned_issues
            issues=planned_issues(draft,require_doses=False)
            previous_draft=json.loads(st.session_state['multidrug_target_text'])
            for field,key in [('target_dlt','workflow_target_dlt'),('minimum_drug_similarity','workflow_min_drug_similarity')]:
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
        st.caption('Draft saved. Doses are optional for reviewing matching articles.')
        if st.button('Continue to matching articles',type='primary'):
            go_to(st,'Review evidence')
    elif page == 'Review evidence':
        st.subheader('Review matching articles')
        st.write('Choose the historical evidence to use for your planned trial. Doses are not required for this step.')
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
            st.info('Find matching articles to see drug correspondence and abstracts.')
            return
        candidates=st.session_state['multidrug_preview']['selected_series']
        if not candidates:
            st.info('No articles meet these filters. Broaden the candidate pool or add drug mechanism information.')
        st.caption(f"{len({r['document_id'] for r in candidates})} candidate articles · {len(candidates)} study groups")
        all_ids=[row['series_key'] for row in candidates]
        scope_by_id={scope.scope_id:scope for scope in scopes}
        scope_labels={r['series_key']:r['document_id']+' · '+r['label']+' · '+r['series_key'][-6:] for r in candidates}
        with st.expander('Choose individual study groups (advanced)'):
            included=st.multiselect('Included source model scopes',all_ids,default=all_ids,format_func=scope_labels.__getitem__,key='multidrug_included_'+selection_spec[:12])
        from phase1_web.evidence_review import render_review
        decisions,review_ready=render_review(st,candidates,scopes,native.STATE_ROOT,selection_spec,config['review_mode'],included_scope_ids=included,
            target_drug_names={d['drug_id']:d['name'] for d in draft['drugs']})
        new_review={'identity':selection_spec,'included':included,'decisions':decisions,'ready':review_ready}
        if new_review != st.session_state.get('multidrug_review_state'):
            activate_prediction(st.session_state,None)
            st.session_state['multidrug_results']={}
        st.session_state['multidrug_review_state']=new_review
        with st.expander('Adjust drug correspondence (advanced)'):
            st.caption('Automatic correspondence is the default. An explicit choice will also be used in prediction.')
            if all_ids:
                scope_id=st.selectbox('Source regimen to map',all_ids,format_func=scope_labels.__getitem__)
                source=scope_by_id[scope_id]
                names={d.model_drug_id:d.name for d in source.drugs}
                old_pairs=draft.get('mapping_overrides',{}).get(scope_id,{})
                chosen_pairs={}
                for drug in draft['drugs']:
                    options=[None,*names]
                    old=old_pairs.get(drug['drug_id'])
                    chosen=st.selectbox('Source for '+drug['name'],options,index=options.index(old) if old in options else 0,
                        format_func=lambda value:'Automatic correspondence' if value is None else names[value],
                        key='manual_'+scope_id+'_'+drug['drug_id']+'_'+selection_spec[:8])
                    if chosen is not None:chosen_pairs[drug['drug_id']]=chosen
                if st.button('Apply correspondence'):
                    if len(set(chosen_pairs.values()))!=len(chosen_pairs):
                        st.error('A source drug cannot be assigned to two target drugs.')
                    else:
                        data=copy.deepcopy(draft)
                        data.setdefault('mapping_overrides',{})
                        if chosen_pairs:data['mapping_overrides'][scope_id]=chosen_pairs
                        else:data['mapping_overrides'].pop(scope_id,None)
                        text=json.dumps(data,indent=2)
                        st.session_state['multidrug_target_text']=text
                        (native.STATE_ROOT/'multidrug_target_draft.json').write_text(text)
                        st.session_state['multidrug_rebase_editor']=True
                        invalidate_prediction(st.session_state)
                        st.session_state.pop('multidrug_review_state',None)
                        st.rerun()
        with st.expander('Detailed comparison evidence'):
            st.json(st.session_state['multidrug_preview'],expanded=False)
        if st.button('Continue to risk curves',type='primary',disabled=not review_ready):
            go_to(st,'Prediction Results')
    elif page == 'Prediction Results':
        st.subheader('Explore risks')
        existing=st.session_state.get('multidrug_prediction')
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
        with st.expander('Analysis settings (optional)'):
            target_probability=st.number_input('Target DLT probability',min_value=.01,max_value=.99,value=float(draft.get('target_dlt',.25)),step=.01,key='workflow_target_dlt')
            if target is not None:target=target.model_copy(update={'target_dlt':target_probability})
            if draft.get('target_dlt') != target_probability:
                draft['target_dlt']=target_probability
                text=json.dumps(draft,ensure_ascii=False,indent=2)
                st.session_state['multidrug_target_text']=text
                (native.STATE_ROOT/'multidrug_target_draft.json').write_text(text)
                st.session_state['workflow_saved_draft']=fingerprint(draft)
            st.caption('The desired toxicity rate for selecting a dose. This value carries into BOIN without re-entry.')
            backend, construction = probability_model_controls(st)
            power_settings = None
            if construction=='Complete-regimen borrowing':
                methods = {
                    'Average of article posterior curves (Method 1)': 'same_article_joint_posterior_average',
                    'Mixture of complete article posterior curves (Method 2)': 'same_article_joint_posterior_mixture',
                    'Power prior from discounted historical data (Method 3)': 'target_power_prior',
                }
                aggregation=st.selectbox('Curve aggregation',list(methods),index=1,key='workflow_aggregation')
                synthesis=methods[aggregation]
            else:
                methods = {
                    'Average of assembled component curves (Method 1)': 'cross_article_component_assembly_average',
                    'Mixture of assembled component curves (Method 2)': 'cross_article_component_assembly',
                }
                aggregation=st.selectbox('Curve aggregation',list(methods),index=1,key='workflow_component_aggregation')
                synthesis=methods[aggregation]
                st.caption('Each target drug can borrow a mapped component from a different article, including a single-drug study. A source combination contributes its jointly fitted components; its raw DLT counts are never treated as single-drug outcomes.')
                st.caption('Interactions come from a shared source joint draw when available. Otherwise, they come from your selected prior. Both methods support a curve with one drug fixed or a two-drug surface.')
                if synthesis == 'cross_article_component_assembly_average':
                    st.caption('Method 1 averages complete assembled curves, keeping each source draw and each missing-interaction prior draw shared across recipes. Its intervals describe the average. Large recipe spaces use an explicitly reported 256-recipe approximation.')
                else:
                    st.caption('Method 2 samples one complete assembly per posterior curve, retaining variation between alternative assemblies. The same assembly is used at every dose on that curve.')
            if synthesis == 'same_article_joint_posterior_average':
                st.caption('Averages a complete posterior curve from each contributing article using fixed article weights. Its intervals describe the weighted average; they do not retain all between-article variation as the curve mixture does.')
            if synthesis == 'target_power_prior':
                st.caption('Fits a separate target model from original cohort DLT/N. Uses one best complete scope per article, fixed similarity discounts and between-article variation. Source article fits remain unchanged. The first target fit can take time; identical settings and new plot coordinates reuse its cache.')
            from phase1_web.prior_settings import render_prior_settings
            try:
                selected_prior, power_settings, _prior_choice = render_prior_settings(st, backend, synthesis, native.STATE_ROOT)
            except (ValueError, OSError) as error:
                invalidate_prediction(st.session_state)
                st.error('Check the prior settings: ' + str(error))
                return
            draw_count=st.number_input('Target joint draws',100,5000,1000,100,key='workflow_draw_count')
        st.caption('Curve method: '+prediction_result_label(prediction_result_key(backend,synthesis,prior=selected_prior)))
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
        if st.button('Change articles or matching settings'):go_to(st,'Review evidence')
        if target is None:
            activate_prediction(st.session_state,None)
            st.session_state['multidrug_results']={}
            return
        if synthesis=='target_power_prior' and st.session_state.get('multidrug_preview') and review_ready:
            from phase1_web.evidence_review import power_prior_review_preview
            from phase1_prediction.multidrug.power_prior import PowerPriorSpec
            try:
                rows=[r for r in st.session_state['multidrug_preview']['selected_series'] if r['series_key'] in included]
                checked=power_prior_review_preview(rows,scopes,target,PowerPriorSpec.model_validate(power_settings),decisions)
                review_ready=review_ready and checked['ready']
                with st.expander('Method 3 historical contributions'):
                    st.dataframe(checked['rows'],hide_index=True)
                    st.caption('Method 3 uses absolute likelihood powers, not normalized article weights. Unsure halves an article contribution; Exclude removes it without redistributing its pre-review budget.')
                    st.caption(f"Discounted patient count: {checked['discounted_patient_count']:.4g} (a borrowing budget, not an exact effective sample size).")
                    if checked['excluded']:st.dataframe(checked['excluded'],hide_index=True)
                if not checked['ready']:st.info('No selected article is usable for Method 3 at these settings. Review its exclusions above.')
            except (ValueError,RuntimeError) as error:
                st.error(str(error));review_ready=False
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
                        key='target_source_cache_' + backend, selected_prior=selected_prior)
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
                    raise ValueError('Select at least one source scope or preview the candidates again.')
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
        st.write('Choose an ordered regimen path for BOIN. Each virtual trial uses one complete joint posterior risk curve.')
        from phase1_web.multidrug_state import simulation_view
        with st.expander('Actual candidate doses for BOIN',expanded=True):
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
                    st.info('Evaluate these candidate doses before selecting the BOIN path.');return
                st.caption('These candidate regimens are already evaluated. Choose the BOIN path below; editing doses is optional.')
            except (ValueError,OSError,RuntimeError) as error:st.info(str(error));return
        target=result['target']
        if target.get('profile',{}).get('planning',{}).get('exploration',{}).get('relative'):
            st.info('Enter and evaluate actual candidate doses above before BOIN.');return
        names={d['drug_id']:d['name'] for d in target['drugs']}
        labels={p['level_id']:p['level_id']+' · '+', '.join(f"{names[k]}: {v if v is not None else 'used, dose unknown'}" for k,v in p['doses'].items()) for p in target['points']}
        path_ids=st.multiselect('BOIN path: select regimens in exploration order',list(labels),default=target.get('path_level_ids',[]),format_func=lambda x:labels[x],key='simulation_path_'+result['run_id'])
        st.caption('Selection order defines the path. No toxicity ordering is inferred from a two-dimensional grid.')
        path_key=fingerprint([result['run_id'],path_ids])
        if st.session_state.get('simulation_path_identity')!=path_key:
            clear_dependents(st.session_state)
            st.session_state['simulation_path_identity']=path_key
        if len(path_ids)<2:
            st.info('Select at least two candidate regimens to configure BOIN. Prediction does not require a path.')
            return
        simulation_input=simulation_view(result,path_ids)
        st.subheader('Shared simulation settings')
        if public:
            st.caption('Public calculations allow up to 20,000 trials per design and 50,000 across a design comparison.')
        st.caption(f"Applies to Run BOIN and Compare designs. Target DLT probability: {result['target']['target_dlt']:.2f}, inherited from the trial plan.")
        m=st.number_input('Posterior scenarios',1,1000,100,key='boin_design_scenarios')
        repetitions=st.number_input('Trials per scenario',1,1000,100,key='boin_design_repetitions')
        with st.expander('Shared BOIN decision and stopping rules',expanded=True):
            elimination=st.number_input('Overtoxicity elimination probability',.5,.999,.95,.01,key='boin_design_elimination')
            stopping=st.number_input('Patients at one regimen before stopping',3,100,12,3,key='boin_design_stopping')
            if st.session_state.get('boin_design_start',1)>len(path_ids):
                st.session_state['boin_design_start']=len(path_ids)
            start_index=st.number_input('Starting path position (1-based)',1,len(path_ids),1,key='boin_design_start')
            low=st.number_input('Acceptable DLT probability: lower bound',.0,.99,.20,.01,key='boin_design_low')
            high=st.number_input('Acceptable DLT probability: upper bound',.01,1.,.35,.01,key='boin_design_high')
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
            st.json({k:v for k,v in sim.items() if k not in {'trials','draw_provenance','per_scenario'}}, expanded=False)
            native._json_download('Download simulation',sim,sim['run_id']+'.json')
        with st.expander('Compare BOIN designs'):
            st.caption('Choose sample sizes and cohort sizes to compare using the shared settings above.')
            cohorts=st.multiselect('Candidate cohort sizes',[1,2,3,4,6],[3],key='boin_design_search_cohorts')
            sizes=st.multiselect('Candidate maximum sample sizes',[12,18,24,30,36,42,48,60],[30,36],key='boin_design_search_sizes')
            search_request=fingerprint([path_key,cohorts,sizes,elimination,stopping,start_index,low,high,m,repetitions])
            if st.session_state.get('multidrug_search_request')!=search_request:
                st.session_state.pop('multidrug_search',None)
                st.session_state['multidrug_search_request']=search_request
            if st.button('Compare designs'):
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
                st.dataframe([{k:v for k,v in r.items() if k!='result'} for r in search['ranked']],hide_index=True)
                native._json_download('Download design comparison',search,'boin_comparison.json')
    elif page == 'Saved analyses':
        records=list_analyses(native.STATE_ROOT)
        if not records: st.info('No saved analyses yet.'); return
        st.dataframe(records, hide_index=True)
        st.caption('Historical analyses retain their original results and calculation versions.')
        chosen=st.selectbox('Saved analysis',list(range(len(records))),format_func=lambda i:records[i]['name']+' · '+records[i]['saved_at_utc'])
        record=load_analysis(Path(records[chosen]['path']))
        if record.get('prediction',{}).get('model_version') in {'v3_multidrug_bayes_1', VERSION}:
            _result(record['prediction'],native)
        else:
            st.json(record,expanded=False)
        native._json_download('Download saved analysis',record,record['analysis_id']+'.json')
