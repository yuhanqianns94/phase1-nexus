"""A visible, non-destructive sequence for prospective clinical inputs."""
import streamlit as st

STEPS = ('1. Drugs', '2. Administration', '3. Population', '4. DLT',
         '5. Doses (optional)', '6. Review')
STEP_KEY = 'trial_plan_step'
TAB_KEY = 'trial_plan_selected_tab'


def _tab_selected():
    """Commit direct tab navigation before rendering the progress/buttons."""
    selected=st.session_state.get(TAB_KEY)
    if selected in STEPS:st.session_state[STEP_KEY]=selected


def show_steps():
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
        st.caption(f'Step {index + 1} of {len(STEPS)} · Changes are saved automatically. '
                   'Use the tabs or Previous / Next; you can skip optional details.')
        show_step_buttons(position='top')
    return pages


def _select_step(step):
    st.session_state[STEP_KEY] = step
    st.session_state[TAB_KEY] = step


def show_step_buttons(*, position='bottom'):
    index = STEPS.index(st.session_state[STEP_KEY])
    suffix = '_top' if position == 'top' else ''
    previous, following = st.columns(2)
    previous.button('Previous', key='trial_plan_previous'+suffix, disabled=index == 0,
                    on_click=_select_step, args=(STEPS[max(0, index - 1)],))
    following.button('Review complete' if index == len(STEPS) - 1 else 'Next: ' + STEPS[index + 1][3:],
                     key='trial_plan_next'+suffix, disabled=index == len(STEPS) - 1,
                     type='primary', on_click=_select_step,
                     args=(STEPS[min(index + 1, len(STEPS) - 1)],))
