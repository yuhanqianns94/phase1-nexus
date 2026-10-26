"""A visible, non-destructive sequence for prospective clinical inputs."""
import streamlit as st

STEPS = ('1. Drugs', '2. Administration', '3. Population', '4. DLT',
         '5. Doses (optional)', '6. Summary')
STEP_KEY = 'trial_plan_step'
TAB_KEY = 'trial_plan_selected_tab'
REVIEW_REQUEST_KEY = 'trial_plan_review_requested'


def _tab_selected():
    """Commit direct tab navigation before rendering the progress/buttons."""
    selected=st.session_state.get(TAB_KEY)
    if selected in STEPS:st.session_state[STEP_KEY]=selected


def show_steps():
    for key in (STEP_KEY, TAB_KEY):
        if st.session_state.get(key) == '6. Review':
            st.session_state[key] = STEPS[-1]
    if st.session_state.get(STEP_KEY) not in STEPS:
        st.session_state[STEP_KEY] = STEPS[0]
        st.session_state[TAB_KEY] = STEPS[0]
    elif st.session_state.get(TAB_KEY) not in STEPS:
        st.session_state[TAB_KEY] = st.session_state[STEP_KEY]
    header=st.container()
    # Register the tab widget before reading its state for the header. Its
    # explicit callback also synchronizes direct clicks with Previous / Next.
    # All containers still render: switching tabs must not discard edits.
    pages=st.tabs(STEPS, key=TAB_KEY, on_change=_tab_selected)
    with header:
        index = STEPS.index(st.session_state[STEP_KEY])
        st.caption(f'Trial plan · Section {index + 1} of {len(STEPS)} · Changes save automatically. '
                   'Select a section to edit your plan.')
        if index < len(STEPS) - 1:
            show_step_buttons(position='top')
    return pages


def _select_step(step):
    st.session_state[STEP_KEY] = step
    st.session_state[TAB_KEY] = step


def _request_evidence_review():
    # The workspace saves and validates the latest edits before navigating.
    st.session_state[REVIEW_REQUEST_KEY] = True


def show_step_buttons(*, position='bottom', review_ready=True):
    index = STEPS.index(st.session_state[STEP_KEY])
    suffix = '_top' if position == 'top' else ''
    previous, following = st.columns(2)
    previous.button('Previous', key='trial_plan_previous'+suffix, disabled=index == 0,
                    on_click=_select_step, args=(STEPS[max(0, index - 1)],))
    if index == len(STEPS) - 1:
        following.button('Next: Review evidence', key='trial_plan_review_evidence',
                         disabled=not review_ready, type='primary',
                         on_click=_request_evidence_review)
    else:
        following.button('Next: ' + STEPS[index + 1][3:],
                         key='trial_plan_next'+suffix, type='primary',
                         on_click=_select_step, args=(STEPS[index + 1],))
