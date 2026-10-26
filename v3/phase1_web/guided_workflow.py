"""Presentation state for the prospective trial-planning workflow.

Matching has no dose-grid dependency. Review decisions are usable only for the
exact clinical plan, library and filters that produced the candidate list.
"""
import copy
import json

from phase1_prediction.multidrug.contracts import fingerprint
from phase1_prediction.profile_similarity import DEFAULT_WEIGHTS, LABELS
from phase1_web.multidrug_state import invalidate_prediction


PAGE_LABELS = {
    'Target Trial': '1. Trial plan',
    'Review evidence': '2. Review evidence',
    'Prediction Results': '3. Explore risks',
    'OC Simulation': '4. Simulate a trial (optional)',
    'Download results': '5. Download results',
    'Library': 'Library',
    'Article comparison': 'Compare articles',
    'Saved analyses': 'Saved analyses',
    'Extraction Jobs': 'Extraction jobs',
    'Settings': 'Settings',
}


def go_to(st, page):
    st.session_state['workflow_next_page'] = page
    st.rerun()


def initialize_draft(st, state_root):
    if 'multidrug_target_text' in st.session_state:
        return
    from phase1_web.multidrug_form import blank_target
    path = state_root / 'multidrug_target_draft.json'
    initial = blank_target()
    legacy = state_root / 'toxicity_ui' / 'target_trial_draft.json'
    if not path.exists() and legacy.exists():
        try:
            from phase1_web.multidrug_draft import migrate_legacy_draft
            initial = migrate_legacy_draft(json.loads(legacy.read_text())).model_dump(mode='json')
        except (ValueError, KeyError, TypeError):
            st.warning('The older draft could not be loaded. Its original file is retained.')
    st.session_state['multidrug_target_text'] = (path.read_text() if path.exists()
        else json.dumps(initial, ensure_ascii=False, indent=2))
    backup = state_root / 'multidrug_target_draft.previous.json'
    if backup.exists():
        st.session_state['multidrug_previous_target_text'] = backup.read_text()


def enforce_automatic_correspondence(st, state_root, draft=None):
    """Remove legacy/manual web overrides, preserving the original draft locally."""
    if draft is None:
        try:
            draft = json.loads(st.session_state['multidrug_target_text'])
        except (ValueError, KeyError, TypeError):
            return
    if not isinstance(draft, dict) or not draft.get('mapping_overrides'):
        return
    state_root.mkdir(parents=True, exist_ok=True)
    backup = state_root / ('multidrug_target_draft.before_automatic_' + fingerprint(draft)[:16] + '.json')
    if not backup.exists():
        backup.write_text(json.dumps(draft, ensure_ascii=False, indent=2))
    draft['mapping_overrides'] = {}
    text = json.dumps(draft, ensure_ascii=False, indent=2)
    st.session_state['multidrug_target_text'] = text
    (state_root / 'multidrug_target_draft.json').write_text(text)
    st.session_state['multidrug_rebase_editor'] = True
    invalidate_prediction(st.session_state)
    st.session_state.pop('multidrug_review_state', None)


def evidence_settings(st):
    saved = st.session_state.get('workflow_evidence', {})
    engine = saved.get('engine', 'local')
    weights = saved.get('weights', dict(DEFAULT_WEIGHTS))
    query = {
        'threshold': saved.get('threshold', 0.),
        'top_articles': saved.get('top_articles'),
        'weights': weights,
        'similarity_engine': engine,
        'embedding_model': saved.get('embedding_model', 'text-embedding-3-small'),
    }
    return {'query': query, 'weights': weights, 'review_mode': saved.get('review_mode', False)}


def render_evidence_controls(st, draft):
    from phase1_web.hosted import is_public
    engine = st.selectbox('Similarity engine', ['local'] if is_public() else ['local', 'openai'],
        format_func=lambda x: 'Local comparison' if x == 'local' else 'OpenAI embeddings',
        key='workflow_similarity_engine')
    model = 'text-embedding-3-small'
    if engine == 'openai':
        model = st.selectbox('Embedding model', ['text-embedding-3-small', 'text-embedding-3-large'],
                             key='workflow_embedding_model')
        st.caption('Uncached clinical text is sent to the configured OpenAI API. Numeric rules are still compared structurally. An API error stops matching.')
    else:
        st.caption('Compares structured clinical fields and text locally. No embedding API is called.')
    review = st.radio('Evidence screening', ['Automatic inclusion', 'Review article abstracts'],
                      key='workflow_screening', horizontal=True) == 'Review article abstracts'
    with st.expander('Matching filters (optional)'):
        pool = st.selectbox('Candidate pool', ['Similarity threshold', 'Top N articles', 'Threshold then Top N'],
                            key='workflow_pool_mode')
        threshold = (st.slider('Minimum article similarity', 0., 1., 0., .01,
                              key='threshold_' + engine) if pool != 'Top N articles' else 0.)
        top = (int(st.number_input('Top N distinct articles', 1, 10000, 10,
                                   key='workflow_top_articles')) if pool != 'Similarity threshold' else None)
        from phase1_prediction.clinical_coverage import DEFAULT_MIN_CLINICAL_COVERAGE
        old = float(draft.get('minimum_clinical_coverage', DEFAULT_MIN_CLINICAL_COVERAGE))
        minimum = st.number_input('Minimum clinical-information coverage', 0., 1., old, .05,
                                   key='workflow_min_clinical_coverage',
                                   help='0.20 means 20% coverage. Evidence below this threshold is skipped before similarity comparison.')
        draft['minimum_clinical_coverage'] = minimum
        # Retain the old JSON field for compatibility, without a hidden .20
        # similarity cutoff behind the new coverage control.
        draft['minimum_drug_similarity'] = 0.
        st.caption('Check information coverage first, then compare similarity. Coverage uses disease and population, eligibility, drugs, administration, DLT criteria and the assessment window.')
        with st.expander('How coverage is calculated'):
            st.latex(r'c_g=\frac{|F_{g,\mathrm{plan}}\cap F_{g,\mathrm{article}}|}{|F_{g,\mathrm{plan}}\cup F_{g,\mathrm{article}}|},\qquad C=\sum_g w_g c_g')
            st.caption('F lists reported clinical field types in a category. A field counts as shared when both sides provide it; values may differ. A category missing on both sides contributes zero. The category weights below sum to 1.')
            st.caption('Only evidence with C at or above your threshold is compared. Coverage is a gate, not an extra similarity or borrowing-weight multiplier. Setting the threshold to 0 disables the gate.')
        with st.expander('Importance of each clinical field'):
            weights = {key: st.number_input(LABELS[key], min_value=0., max_value=100.,
                value=float(value), key='multidrug_weight_' + key) for key, value in DEFAULT_WEIGHTS.items() if key != 'structure'}
            weights['structure'] = 0.
        st.caption('Top N counts distinct articles that pass the coverage gate. Ranking uses article similarity.')
    st.session_state['workflow_evidence'] = {'engine': engine, 'embedding_model': model,
        'weights': weights, 'threshold': threshold, 'top_articles': top, 'review_mode': review}
    return evidence_settings(st)


def sync_evidence_selection(st, draft, config, context):
    # Plot coordinates and design names do not alter a clinical correspondence.
    matching = {key: copy.deepcopy(draft.get(key)) for key in
                ('drugs', 'profile', 'mapping_overrides', 'minimum_drug_similarity', 'minimum_clinical_coverage')}
    identity = fingerprint([matching, config, context])
    if identity != st.session_state.get('multidrug_selection_request'):
        invalidate_prediction(st.session_state)
        st.session_state.pop('multidrug_review_state', None)
        st.session_state['multidrug_selection_request'] = identity
    return identity


def evidence_state(st, identity, review_mode):
    stored = st.session_state.get('multidrug_review_state')
    if stored and stored.get('identity') == identity:
        return stored['included'], stored['decisions'], stored['ready']
    preview = st.session_state.get('multidrug_preview')
    if preview is not None:
        included = [row['series_key'] for row in preview['selected_series']]
        return included, None, bool(included) and not review_mode
    return (), None, not review_mode


def sync_exploration_units(st, draft, *, key='explore'):
    """A changed plan rebases inherited units; independent plot inputs remain."""
    current = {d['drug_id']: (d.get('unit'), d.get('reference_dose'))
               for d in draft.get('drugs', [])}
    saved_key = 'workflow_coordinate_plan_' + key
    previous = st.session_state.get(saved_key, {})
    for drug_id, values in current.items():
        if drug_id in previous and previous[drug_id] != values:
            for suffix in ('_unit', '_ref'):
                st.session_state.pop(key + '_' + drug_id + suffix, None)
    st.session_state[saved_key] = current
