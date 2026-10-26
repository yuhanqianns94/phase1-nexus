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
    'combination_sd': ('Combination effect: half-normal scale',
        'A nonnegative log-odds effect when companion treatment is used. Larger values allow a greater increase in combination risk.'),
    'pair_sd': ('Two-drug interaction: standard deviation',
        'Each pairwise interaction has a zero-centered Normal prior. Larger values allow stronger effects in either direction.'),
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


_SINGLE_FIELDS = {
    'intercept_mean': ('Baseline log-odds: mean', 'Center of the focal model baseline at zero focal dose without companion treatment.'),
    'intercept_sd': ('Baseline log-odds: standard deviation', 'Prior uncertainty for the common baseline alpha.'),
    'log_slope_mean': ('Log focal dose slope: mean', 'The focal effect is exp(ell) times relative dose.'),
    'log_slope_sd': ('Log focal dose slope: standard deviation', 'Uncertainty in the positive focal dose effect.'),
}


def prior_parameter_rows(prior, backend):
    prior = canonical_prior(prior, backend)
    if backend == 'single_drug_logistic':
        return [
            {'Parameter': 'Baseline log odds without companion treatment',
             'Prior distribution': f'Normal(mean={prior.intercept_mean:g}, SD={prior.intercept_sd:g})'},
            {'Parameter': 'Log of positive focal dose slope',
             'Prior distribution': f'Normal(mean={prior.log_slope_mean:g}, SD={prior.log_slope_sd:g})'},
            {'Parameter': 'Nonnegative combination effect',
             'Prior distribution': f'HalfNormal(scale={prior.combination_sd:g})'},
        ]
    rows = [
        {'Parameter': ('Common baseline log odds at all-zero doses' if backend == 'logistic_product'
                       else 'Component log odds at reference dose'),
         'Prior distribution': f'Normal(mean={prior.intercept_mean:g}, SD={prior.intercept_sd:g})'},
        {'Parameter': ('Log of positive linear relative-dose effect' if backend == 'logistic_product'
                       else 'Log of positive component dose slope'),
         'Prior distribution': f'Normal(mean={prior.log_slope_mean:g}, SD={prior.log_slope_sd:g})'},
    ]
    rows.extend([
        {'Parameter': 'Pairwise interactions',
         'Prior distribution': f'Normal(mean=0, SD={prior.pair_sd:g})'},
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


def _custom_risk(st, backend, namespace, seed, drug_count=2, drug_modes=None, combination=None):
    values = PriorSpec.model_validate(seed).model_dump()
    for field in active_prior_fields(backend):
        if (field == 'triple_sd' or (field == 'pair_sd' and drug_count == 1) or
                (field.startswith('log_slope') and drug_modes and 'numeric' not in drug_modes)):
            continue
        overrides = _SINGLE_FIELDS if backend == 'single_drug_logistic' else _PRODUCT_FIELDS if backend == 'logistic_product' else {}
        label, help_text = overrides.get(field, _FIELDS[field])
        if field == 'combination_sd' and combination is False:
            label = 'Historical combination adjustment: half-normal scale'
            help_text = 'The target is monotherapy. This parameter can still adjust historical combination outcomes during fitting.'
        key = namespace + field
        if key not in st.session_state:
            st.session_state[key] = float(values[field])
        minimum = None if field.endswith('_mean') else .000001
        maximum = .999999 if field == 'unknown_presence_probability' else None
        values[field] = st.number_input(label, min_value=minimum, max_value=maximum,
            step=.05 if field == 'unknown_presence_probability' else .1,
            format='%.6f', key=key, help=help_text)
    return canonical_prior(values, backend)


def prior_formulas(prior, backend='logistic', drug_count=1, drug_modes=None, *, combination=None):
    prior = canonical_prior(prior, backend)
    index = '' if drug_count == 1 else '_j'
    if backend == 'single_drug_logistic':
        return [
            rf'\alpha\sim\mathcal{{N}}({prior.intercept_mean:g},\,{prior.intercept_sd:g}^2)',
            rf'\ell=\log\beta\sim\mathcal{{N}}({prior.log_slope_mean:g},\,{prior.log_slope_sd:g}^2)',
        ] + ([rf'\delta\sim\operatorname{{HalfNormal}}({prior.combination_sd:g})'] if combination is not False else [])
    intercept = r'\alpha' if backend == 'logistic_product' else 'a' + index
    formulas = [
        rf'{intercept}\sim\mathcal{{N}}({prior.intercept_mean:g},\,{prior.intercept_sd:g}^2)',
    ]
    numeric = [str(i + 1) for i, mode in enumerate(drug_modes or ['numeric'] * drug_count) if mode == 'numeric']
    if numeric:
        suffix = (r',\quad j\in\{' + ','.join(numeric) + r'\}') if drug_count > 1 else ''
        formulas.append(rf'\ell{index}=\log b{index}\sim\mathcal{{N}}({prior.log_slope_mean:g},\,{prior.log_slope_sd:g}^2)' + suffix)
    if drug_count > 1:
        pair = r'\eta_{12}' if drug_count == 2 else r'\eta_{jk}\ (j<k)'
        formulas.append(rf'{pair}\sim\mathcal{{N}}(0,\,{prior.pair_sd:g}^2)')
    return formulas


def render_prior_formulas(st, prior, backend='logistic', drug_count=1, drug_modes=None, *, combination=None):
    if backend == 'single_drug_logistic':
        prior = canonical_prior(prior, backend)
        for label, equation in zip(['Baseline log odds', 'Focal dose slope (log scale)', 'Combination effect (nonnegative)'],
                                   prior_formulas(prior, backend, combination=combination)):
            st.caption(label)
            st.latex(equation)
        if combination is False:
            st.caption('Historical combination adjustment')
            st.latex(rf'\delta\sim\operatorname{{HalfNormal}}({prior.combination_sd:g})')
            st.caption(r'$\delta$ adjusts historical combination outcomes during fitting; it does not enter the monotherapy target curve.')
        st.caption(r'$\mathcal N(\mu,\sigma^2)$: Normal prior (mean, variance). $\operatorname{HalfNormal}(\sigma)$: nonnegative prior (scale).')
        return
    labels = ['Baseline log odds' if backend == 'logistic_product' else
              ('Drug-use log odds' if drug_modes and all(mode == 'background' for mode in drug_modes) else
               'Reference-dose log odds')]
    if not drug_modes or 'numeric' in drug_modes:
        labels.append('Dose slope (log scale)')
    if drug_count > 1:
        labels.append('Pairwise interaction')
    for label, equation in zip(labels, prior_formulas(prior, backend, drug_count, drug_modes)):
        st.caption(label)
        st.latex(equation)
    st.caption(r'$\mathcal N(\mu,\sigma^2)$ denotes a Normal distribution with mean $\mu$ and variance $\sigma^2$.')


def render_prior_settings(st, backend, synthesis, state_root, drug_count=1, drug_modes=None, *, combination=None):
    """Return exact prior settings; render the active mathematical distributions."""
    path = Path(state_root) / 'bayesian_prior_preferences.json'
    preferences = _read_preferences(path, st)
    previous_preferences = copy.deepcopy(preferences)
    backends = preferences.setdefault('backends', {})
    if backend == 'single_drug_logistic' and backend not in backends:
        # Preserve the user's selected uncertainty scale, not obsolete interaction values.
        old = backends.get('logistic', {})
        backends[backend] = {k: copy.deepcopy(v) for k, v in old.items() if k != 'custom_risk'}
        if old.get('custom_risk'):
            fields = set(active_prior_fields(backend))
            backends[backend]['custom_risk'] = {k: v for k, v in old['custom_risk'].items() if k in fields}
    stored = backends.setdefault(backend, {})
    risk_namespace = f'prior_settings_{backend}_risk_'
    with st.container(border=True):
        st.markdown('### Parameter distributions')
        choices = [*RISK_PRESETS, 'custom']
        saved = stored.get('risk_choice', 'balanced')
        def formula_label(name):
            if name == 'custom':
                return 'Custom distributions'
            spec = risk_prior_preset(name, backend)
            numeric = not drug_modes or 'numeric' in drug_modes
            if backend == 'single_drug_logistic':
                symbols = r'\sigma_\alpha,\sigma_\ell' + (r',\sigma_\delta' if combination is not False else '')
                values = f'{spec.intercept_sd:g},{spec.log_slope_sd:g}' + (f',{spec.combination_sd:g}' if combination is not False else '')
            else:
                symbols = r'\sigma_a' + (r',\sigma_\ell' if numeric else '') + (r',\sigma_\eta' if drug_count > 1 else '')
                values = f'{spec.intercept_sd:g}' + (f',{spec.log_slope_sd:g}' if numeric else '') + (f',{spec.pair_sd:g}' if drug_count > 1 else '')
            description = {'balanced': 'Balanced / default', 'regularized': 'More regularized',
                           'broad': 'Broader variation'}[name]
            return description + ' · $(' + symbols + ')=(' + values + ')$'
        choice = st.radio('Prior scales', choices, index=choices.index(saved) if saved in choices else 0,
            format_func=formula_label, key=risk_namespace + 'preset', horizontal=True)
        if choice == 'custom':
            seed = stored.get('custom_risk') or st.session_state.get(risk_namespace + 'last_values') or PriorSpec().model_dump()
            prior = _custom_risk(st, backend, risk_namespace, seed, drug_count, drug_modes, combination)
            stored['custom_risk'] = prior.model_dump()
            label = 'Custom risk prior'
        else:
            prior = risk_prior_preset(choice, backend)
            label = RISK_PRESETS[choice]['label']
        stored['risk_choice'] = choice
        st.session_state[risk_namespace + 'last_values'] = prior.model_dump()
        render_prior_formulas(st, prior, backend, drug_count, drug_modes, combination=combination)
        with st.container():
            st.markdown('**Unreported historical values**')
            st.caption('Missing numerical dose')
            st.latex(rf'\log r_{{\mathrm{{missing}}}}\sim\mathcal{{N}}(\mu_{{\mathrm{{dose}}}},\,{prior.missing_log_dose_sd:g}^2)')
            st.caption('Unknown drug use')
            st.latex(rf'U_{{\mathrm{{unknown}}}}\sim\operatorname{{Bernoulli}}({prior.unknown_presence_probability:g})')
            st.caption(r'$\mu_{\mathrm{dose}}=0$ on the original historical relative-dose scale and shifts with dose-coordinate conversion. Reported bounds truncate this distribution. $U$: unreported drug use. Missing-dose modeling requires an existing numerical dose axis.')

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
        with st.container(border=True):
            st.markdown('### Historical borrowing settings')
            power_choice = _select(st, 'Historical borrowing settings', POWER_PRESETS, namespace,
                                   stored.get('power_choice'), 'default')
            if power_choice == 'custom':
                seed = PowerPriorSpec.model_validate(stored.get('custom_power') or
                    st.session_state.get(namespace + 'last_values') or {})
                for field, value in (('strength', seed.strength), ('use_cap', seed.patient_budget is not None),
                                     ('budget', seed.patient_budget or 30.), ('heterogeneity', seed.heterogeneity_scale),
                                     ('slope_heterogeneity', seed.slope_heterogeneity_scale)):
                    if namespace + field not in st.session_state:
                        st.session_state[namespace + field] = value
                strength = st.number_input('Historical borrowing strength', min_value=0., max_value=1.,
                    step=.1, format='%.6f', key=namespace + 'strength',
                    help='Multiplies every historical likelihood power. Zero uses the specified priors only.')
                capped = st.checkbox('Limit discounted historical patient count', key=namespace + 'use_cap')
                budget = st.number_input('Discounted historical patient budget', min_value=.000001,
                    step=5., format='%.6f', key=namespace + 'budget') if capped else None
                scale = st.number_input('Between-article intercept prior scale' if backend == 'single_drug_logistic'
                    else 'Between-article variation prior scale', min_value=.000001,
                    step=.1, format='%.6f', key=namespace + 'heterogeneity',
                    help=('Half-normal scale for an article-specific shift of the common baseline log odds.'
                          if backend in ('logistic_product', 'single_drug_logistic') else 'Half-normal scale for the shared component log-odds shift within an article. Larger values allow greater between-article variation.'))
                slope_scale = (st.number_input('Between-article log-slope prior scale', min_value=.000001,
                    step=.1, format='%.6f', key=namespace + 'slope_heterogeneity',
                    help='Half-normal scale for variation in log dose slopes. Each article has a positive slope exp(ℓ + v).')
                    if backend == 'single_drug_logistic' else seed.slope_heterogeneity_scale)
                power = PowerPriorSpec(strength=strength, patient_budget=budget, heterogeneity_scale=scale,
                                       slope_heterogeneity_scale=slope_scale)
                stored['custom_power'] = power.model_dump()
                power_label = 'Custom borrowing settings'
            else:
                power = power_prior_preset(power_choice)
                power_label = POWER_PRESETS[power_choice]['label']
                st.caption(POWER_PRESETS[power_choice]['description'])
            stored['power_choice'] = power_choice
            st.session_state[namespace + 'last_values'] = power.model_dump()
            st.latex(rf'a_i={power.strength:g}\,c\,s_i\,m_i')
            if power.patient_budget is not None:
                st.latex(rf'c=\min\!\left(1,\frac{{{power.patient_budget:g}}}{{\sum_i s_i n_i}}\right)')
            else:
                st.latex(r'c=1')
            if backend == 'single_drug_logistic':
                st.caption('Random intercept')
                st.latex(rf'\tau_\alpha\sim\operatorname{{HalfNormal}}({power.heterogeneity_scale:g}),\quad u_i\sim\mathcal{{N}}(0,\tau_\alpha^2)')
                st.caption('Random log slope')
                st.latex(rf'\tau_\ell\sim\operatorname{{HalfNormal}}({power.slope_heterogeneity_scale:g}),\quad v_i\sim\mathcal{{N}}(0,\tau_\ell^2)')
                st.latex(r'\beta_i=e^{\ell+v_i}>0,\qquad u_i\perp v_i\mid\tau_\alpha,\tau_\ell')
                st.caption(r'$s_i$: similarity · $m_i$: review multiplier · $n_i$: historical patients. Article intercept and log-slope effects are independent given their scales. A new trial draws one pair for its entire curve.')
            else:
                st.latex(rf'\tau\sim\operatorname{{HalfNormal}}({power.heterogeneity_scale:g}),\quad u_i\sim\mathcal{{N}}(0,\tau^2)')
                st.caption(r'$s_i$: similarity · $m_i$: review multiplier · $n_i$: historical patients · $u_i$: article baseline shift.')
            st.caption('Uncertain halves a historical likelihood contribution; Exclude removes it without reallocating its share. '
                       'The patient budget limits borrowing and is not an exact Bayesian effective sample size. '
                       'Historical-likelihood caches also depend on the current target, drug correspondence and selected evidence.')
    preferences['version'] = 1
    if not path.exists() or previous_preferences != preferences:
        write_json(path, preferences)
    return prior, power.model_dump() if power else None, {
        'risk_preset': choice, 'risk_label': label,
        'borrowing_preset': power_choice, 'borrowing_label': power_label,
    }
