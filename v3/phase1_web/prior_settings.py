"""Editable Bayesian prior choices, separate from the planned clinical protocol."""
from __future__ import annotations

import json
import copy
from pathlib import Path

from phase1_extraction.artifacts import write_json
from phase1_prediction.multidrug.contracts import PriorSpec
from phase1_prediction.multidrug.power_prior import PowerPriorSpec
from phase1_prediction.multidrug.prior_presets import (
    RISK_PRESETS, POWER_PRESETS, active_prior_fields, canonical_prior,
    risk_prior_preset, power_prior_preset,
)


_FIELDS = {
    'intercept_mean': ('Reference-dose component log-odds: mean',
        'The center for each drug component at its reference dose. This is on the log-odds scale, not a DLT probability.'),
    'intercept_sd': ('Reference-dose component log-odds: standard deviation',
        'How widely the reference-dose component log odds may vary before observing outcomes.'),
    'log_slope_mean': ('Log dose slope: mean',
        'The dose slope is exp(this normally distributed parameter), so the component slope stays positive.'),
    'log_slope_sd': ('Log dose slope: standard deviation',
        'How much the positive component dose slopes may differ. Used only for numerical-dose axes.'),
    'pair_sd': ('Two-drug interaction: standard deviation',
        'The AB, AC and BC effects have zero-centered Normal priors. Larger values allow stronger effects in either direction.'),
    'triple_sd': ('Three-drug interaction: standard deviation',
        'The ABC effect has a zero-centered Normal prior. It is used only when the first three drugs are present.'),
    'missing_log_dose_sd': ('Unknown numerical dose: log-scale standard deviation',
        'Uncertainty when an article has a numerical dose axis but a cohort dose is missing. A wholly dose-unknown drug keeps its use-state axis.'),
    'unknown_presence_probability': ('Unknown drug use: prior probability of use',
        'Used only when drug use itself is unreported. It does not replace an explicitly reported used/not-used state.'),
}


_PRODUCT_FIELDS = {
    'intercept_mean': ('Baseline log-odds at all-zero doses: mean',
        'The common intercept alpha. Unlike Component Logistic, this is the log odds when every drug is absent.'),
    'intercept_sd': ('Baseline log-odds at all-zero doses: standard deviation',
        'Prior uncertainty for the single common baseline intercept alpha.'),
    'log_slope_mean': ('Log linear relative-dose slope: mean',
        'The main effect is exp(this normally distributed parameter) times relative dose, not log relative dose.'),
    'log_slope_sd': ('Log linear relative-dose slope: standard deviation',
        'Uncertainty in each positive main-effect coefficient, including use-state effects. Negative interactions can still produce a decreasing curve.'),
}


def prior_parameter_rows(prior, backend):
    prior = canonical_prior(prior, backend)
    rows = [
        {'Parameter': ('Common baseline log odds at all-zero doses' if backend == 'logistic_product'
                       else 'Component log odds at reference dose'),
         'Prior distribution': f'Normal(mean={prior.intercept_mean:g}, SD={prior.intercept_sd:g})'},
        {'Parameter': ('Log of positive linear relative-dose effect' if backend == 'logistic_product'
                       else 'Log of positive component dose slope'),
         'Prior distribution': f'Normal(mean={prior.log_slope_mean:g}, SD={prior.log_slope_sd:g})'},
    ]
    rows.extend([
        {'Parameter': 'Two-drug interactions (first three drugs)',
         'Prior distribution': f'Normal(mean=0, SD={prior.pair_sd:g})'},
        {'Parameter': 'Three-drug interaction',
         'Prior distribution': f'Normal(mean=0, SD={prior.triple_sd:g})'},
    ])
    rows.extend([
        {'Parameter': 'Missing numerical dose on the source relative scale',
         'Prior distribution': f'LogNormal(log mean=0, log SD={prior.missing_log_dose_sd:g}); reported bounds retained'},
        {'Parameter': 'Unknown drug-use state',
         'Prior distribution': f'Probability of use={prior.unknown_presence_probability:g}'},
    ])
    return rows


def _read_preferences(path, st):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get('backends', {}), dict):
            raise ValueError('Invalid prior preference format')
        return value
    except (OSError, ValueError) as error:
        st.warning('Saved prior preferences could not be read. Select settings below: ' + str(error))
        return {}


def _select(st, label, catalog, namespace, saved, default):
    choices = [*catalog, 'custom']
    initial = saved if saved in choices else default
    return st.selectbox(label, choices, index=choices.index(initial),
        format_func=lambda name: 'Custom values' if name == 'custom' else catalog[name]['label'],
        key=namespace + 'preset')


def _custom_risk(st, backend, namespace, seed):
    values = PriorSpec.model_validate(seed).model_dump()
    for field in active_prior_fields(backend):
        label, help_text = (_PRODUCT_FIELDS.get(field, _FIELDS[field])
                            if backend == 'logistic_product' else _FIELDS[field])
        key = namespace + field
        if key not in st.session_state:
            st.session_state[key] = float(values[field])
        minimum = None if field.endswith('_mean') else .000001
        maximum = .999999 if field == 'unknown_presence_probability' else None
        values[field] = st.number_input(label, min_value=minimum, max_value=maximum,
            step=.05 if field == 'unknown_presence_probability' else .1,
            format='%.6f', key=key, help=help_text)
    return canonical_prior(values, backend)


def render_prior_settings(st, backend, synthesis, state_root):
    """Return exact risk/borrowing specifications and readable choice metadata."""
    path = Path(state_root) / 'bayesian_prior_preferences.json'
    preferences = _read_preferences(path, st)
    previous_preferences = copy.deepcopy(preferences)
    stored = preferences.setdefault('backends', {}).setdefault(backend, {})
    risk_namespace = f'prior_settings_{backend}_risk_'
    with st.expander('Parameter priors', expanded=True):
        choice = _select(st, 'Risk-parameter prior', RISK_PRESETS, risk_namespace,
                         stored.get('risk_choice'), 'balanced')
        if choice == 'custom':
            seed = stored.get('custom_risk') or st.session_state.get(risk_namespace + 'last_values') or PriorSpec().model_dump()
            prior = _custom_risk(st, backend, risk_namespace, seed)
            stored['custom_risk'] = prior.model_dump()
            label = 'Custom risk prior'
        else:
            prior = risk_prior_preset(choice, backend)
            label = RISK_PRESETS[choice]['label']
            st.caption(RISK_PRESETS[choice]['description'])
        stored['risk_choice'] = choice
        st.session_state[risk_namespace + 'last_values'] = prior.model_dump()
        st.dataframe(prior_parameter_rows(prior, backend), hide_index=True, use_container_width=True)
        if backend == 'logistic_product':
            st.caption('The intercept is the all-zero-dose baseline, and slopes multiply relative dose directly. Identical numerical prior settings therefore imply different risk distributions from Component Logistic. The priors have not been clinically calibrated.')
        st.caption('Recommended/default preserves the existing settings. Presets describe parameter uncertainty; '
                   'they are starting points for comparison, not clinically validated risk categories. '
                   'Method 1 and Method 2 share article posterior caches for identical settings.')

    power = None
    power_choice = None
    power_label = None
    if synthesis == 'target_power_prior':
        namespace = f'prior_settings_{backend}_power_'
        # Retain Method 3 values entered in the previous numeric-only UI.
        if 'power_choice' not in stored and 'power_prior_strength' in st.session_state:
            legacy = PowerPriorSpec(
                strength=st.session_state.get('power_prior_strength', 1.),
                patient_budget=st.session_state.get('power_prior_budget', 30.)
                    if st.session_state.get('power_prior_use_cap', True) else None,
                heterogeneity_scale=st.session_state.get('power_prior_heterogeneity', .5))
            if legacy != PowerPriorSpec():
                stored['power_choice'] = 'custom'
                stored['custom_power'] = legacy.model_dump()
        with st.expander('Power prior settings', expanded=True):
            power_choice = _select(st, 'Historical borrowing settings', POWER_PRESETS, namespace,
                                   stored.get('power_choice'), 'default')
            if power_choice == 'custom':
                seed = PowerPriorSpec.model_validate(stored.get('custom_power') or
                    st.session_state.get(namespace + 'last_values') or {})
                for field, value in (('strength', seed.strength), ('use_cap', seed.patient_budget is not None),
                                     ('budget', seed.patient_budget or 30.), ('heterogeneity', seed.heterogeneity_scale)):
                    if namespace + field not in st.session_state:
                        st.session_state[namespace + field] = value
                strength = st.number_input('Historical borrowing strength', min_value=0., max_value=1.,
                    step=.1, format='%.6f', key=namespace + 'strength',
                    help='Multiplies every historical likelihood power. Zero uses the specified priors only.')
                capped = st.checkbox('Limit discounted historical patient count', key=namespace + 'use_cap')
                budget = st.number_input('Discounted historical patient budget', min_value=.000001,
                    step=5., format='%.6f', key=namespace + 'budget') if capped else None
                scale = st.number_input('Between-article variation prior scale', min_value=.000001,
                    step=.1, format='%.6f', key=namespace + 'heterogeneity',
                    help=('Half-normal scale for an article-specific shift of the common baseline log odds.'
                          if backend == 'logistic_product' else 'Half-normal scale for the shared component log-odds shift within an article. Larger values allow greater between-article variation.'))
                power = PowerPriorSpec(strength=strength, patient_budget=budget, heterogeneity_scale=scale)
                stored['custom_power'] = power.model_dump()
                power_label = 'Custom borrowing settings'
            else:
                power = power_prior_preset(power_choice)
                power_label = POWER_PRESETS[power_choice]['label']
                st.caption(POWER_PRESETS[power_choice]['description'])
            stored['power_choice'] = power_choice
            st.session_state[namespace + 'last_values'] = power.model_dump()
            st.dataframe([
                {'Setting': 'Historical borrowing strength', 'Value': f'{power.strength:g}'},
                {'Setting': 'Discounted historical patient budget', 'Value': 'No cap' if power.patient_budget is None else f'{power.patient_budget:g}'},
                {'Setting': 'Between-article shift', 'Value': f'HalfNormal(scale={power.heterogeneity_scale:g})'},
            ], hide_index=True, use_container_width=True)
            st.caption('Uncertain halves a historical likelihood contribution; Exclude removes it without reallocating its share. '
                       'The patient budget limits borrowing and is not an exact Bayesian effective sample size. '
                       'Method 3 caches also depend on the current target, drug correspondence and selected evidence.')
    preferences['version'] = 1
    if not path.exists() or previous_preferences != preferences:
        write_json(path, preferences)
    return prior, power.model_dump() if power else None, {
        'risk_preset': choice, 'risk_label': label,
        'borrowing_preset': power_choice, 'borrowing_label': power_label,
    }
