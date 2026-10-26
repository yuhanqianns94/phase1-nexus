"""Adapt older trial plans to one named drug plus a combination background."""
import copy
import json

from phase1_prediction.multidrug.contracts import fingerprint
from phase1_web.multidrug_state import invalidate_prediction


def combination_settings(st, analysis):
    from phase1_prediction.multidrug.single_drug import COMBINATION_PRIOR_SCALES
    if 'plan_combination_prior' in st.session_state:
        st.session_state['plan_combination_prior'] = st.session_state['plan_combination_prior']
        analysis['combination_prior_level'] = st.session_state['plan_combination_prior']
    present = analysis.get('combination_present', False)
    selection = st.radio('Will this drug be used in combination?', ['No', 'Yes'],
        index=None if present is None else int(present), horizontal=True, key='plan_combination_use')
    analysis['combination_present'] = None if selection is None else selection == 'Yes'
    if selection == 'Yes':
        prior_levels = list(COMBINATION_PRIOR_SCALES)
        saved = analysis.get('combination_prior_level', 'medium')
        level = st.radio('Combination toxicity prior', prior_levels,
            index=prior_levels.index(saved) if saved in prior_levels else 1,
            format_func=str.title, horizontal=True, key='plan_combination_prior')
        analysis['combination_prior_level'] = level
        st.latex(rf'\delta\sim\operatorname{{HalfNormal}}({COMBINATION_PRIOR_SCALES[level]:g}),\quad\delta\geq0')
        st.caption('Additional toxicity on the log-odds scale. Low, medium and high describe the prior increase, not an absolute DLT rate.')


def single_drug_plan(initial):
    draft = copy.deepcopy(initial)
    drugs = draft.get('drugs', [])
    analysis = draft.setdefault('profile', {}).setdefault('single_drug_analysis', {})
    analysis.setdefault('combination_prior_level', 'medium')
    if not drugs:
        analysis.setdefault('combination_present', False)
        analysis.pop('focal_drug_id', None)
        return draft
    focal = next((d for d in drugs if d['drug_id'] == analysis.get('focal_drug_id')), drugs[0])
    did = focal['drug_id']
    companions = [d for d in drugs if d['drug_id'] != did]
    if companions:
        states = draft['profile'].get('planning', {}).get('exposure_states', {})
        contexts = []
        for point in draft.get('points') or [None]:
            use = []
            for drug in companions:
                if point is not None:
                    state = states.get(point['level_id'], {}).get(drug['drug_id'])
                    doses = point.get('doses', {})
                    value = doses.get(drug['drug_id'])
                    use.append(None if drug['drug_id'] not in doses or state in {'Not entered', 'Undecided', 'Not decided'}
                               else value is None or value > 0)
                else:
                    presence = drug.get('profile', {}).get('presence')
                    use.append(True if presence == 'present' else False if presence == 'absent' else None)
            contexts.append(True if True in use else None if None in use else False)
        analysis['combination_present'] = contexts[0] if all(c is contexts[0] for c in contexts) else None
    else:
        analysis.setdefault('combination_present', False)
    analysis['focal_drug_id'] = did
    focal['research_priority'] = 1
    draft['drugs'] = [focal]
    for point in draft.get('points', []):
        point['doses'] = {did: point['doses'][did]} if did in point.get('doses', {}) else {}
    planning = draft['profile'].get('planning', {})
    for level, states in planning.get('exposure_states', {}).items():
        planning['exposure_states'][level] = {did: states[did]} if did in states else {}
    for config in draft['profile'].get('configurations', []):
        if 'drugs' in config:
            config['drugs'] = [copy.deepcopy(focal.get('profile', {}))]
    draft['mapping_overrides'] = {}
    return draft


def migrate_saved_plan(st, state_root):
    """Keep the previous full plan on disk before simplifying its input fields."""
    text = st.session_state.get('multidrug_target_text', '')
    try:
        original = json.loads(text)
        if not isinstance(original, dict) or not isinstance(original.get('drugs'), list) or len(original['drugs']) <= 1:
            return
        converted = single_drug_plan(original)
    except (ValueError, KeyError, TypeError, AttributeError):
        return
    state_root.mkdir(parents=True, exist_ok=True)
    backup = state_root / ('multidrug_target_draft.before_single_drug_' + fingerprint(original)[:16] + '.json')
    if not backup.exists():
        backup.write_text(text)
    st.session_state['multidrug_target_text'] = json.dumps(converted, ensure_ascii=False, indent=2)
    st.session_state['multidrug_reset_editor'] = True
    invalidate_prediction(st.session_state)
    st.session_state.pop('multidrug_review_state', None)
