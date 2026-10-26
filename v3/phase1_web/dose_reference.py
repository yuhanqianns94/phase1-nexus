"""Automatic, stable computational dose scales; never clinical dose advice."""
import math


def geometric_reference(values):
    """Use distinct positive doses, independent of row order or multiplicity."""
    positive = sorted({float(v) for v in values if v is not None
                       and math.isfinite(float(v)) and float(v) > 0})
    return math.exp(math.fsum(map(math.log, positive)) / len(positive)) if positive else None


def reference_control(st, drug, values, *, key, saved_drug=None, basis='candidate doses'):
    """Retain an inherited/first automatic scale across range and grid edits.

    Only an explicit advanced override changes it. Relative coordinate 1 is not
    an actual dose reference when acquiring units for BOIN.
    """
    unit = drug.get('unit')
    if not unit or unit == 'relative dose':
        drug['reference_dose'] = None
        return
    state_key = key + '_reference_state'
    state = st.session_state.get(state_key)
    if state and state['unit'] != unit:
        state = None
        for suffix in ('_ref', '_custom_ref'):
            st.session_state.pop(key + suffix, None)
    if state is None:
        saved = saved_drug or drug
        inherited = saved.get('reference_dose') if saved.get('unit') == unit else None
        # Keep a reference already entered in an open pre-update session.
        legacy = st.session_state.get(key + '_ref')
        reference = legacy or inherited or geometric_reference(values)
        if reference is None:
            drug['reference_dose'] = None
            st.caption('The reference dose will be calculated after you enter positive doses.')
            return
        state = {'unit': unit, 'value': float(reference),
                 'basis': 'saved analysis or plan' if legacy or inherited else basis}
        st.session_state[state_key] = state
    description = st.empty()
    custom_key = key + '_custom_ref'
    for suffix in ('_custom_ref', '_ref'):
        if key + suffix in st.session_state:
            st.session_state[key + suffix] = st.session_state[key + suffix]
    custom = st.checkbox('Use a custom reference dose (advanced)', key=custom_key,
                         help='Only needed for a prespecified dose scale or a dose-mapping sensitivity analysis.')
    reference = state['value']
    if custom:
        reference = st.number_input('Custom reference dose', min_value=.00000001,
                                    value=reference, key=key + '_ref')
        st.caption('Changing this reference changes historical dose mapping and may change predictions; '
                   'it does not rescale actual doses already entered.')
    drug['reference_dose'] = reference
    if reference is None:
        st.info('Enter a positive custom reference dose, or turn off the custom option.')
        return
    description.caption(f"Reference dose: {reference:.6g} {unit} · "
                        f"{'custom scale' if custom else state['basis']}. "
                        'Kept fixed when doses or the number of calculation points change.')
    st.caption(r'Relative dose is calculated automatically: $r=d/d_{\mathrm{ref}}$. '
               'The reference is a model scale, not a recommended dose.')
