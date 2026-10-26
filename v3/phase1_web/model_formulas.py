"""Readable equations for the actual target context and three evidence methods."""
from __future__ import annotations


ASSEMBLY_CHOICES = {
    'average': 'same_article_joint_posterior_average',
    'mixture': 'same_article_joint_posterior_mixture',
    'historical': 'target_power_prior',
}


def target_combination(target):
    """Use the computational projection, including explicit unused companions."""
    if target is None:
        return None
    from phase1_prediction.multidrug.single_drug import project_target
    return project_target(target).profile['single_drug_projection']['combination_present']


def _context_term(combination):
    return r'+\delta' if combination is True else '' if combination is False else r'+\delta C'


def model_formulas(*, combination=False, hierarchical=False):
    term = _context_term(combination)
    curve = (rf'p_{{\mathrm{{new}}}}(r)=\sigma\!\left(\alpha+u_{{\mathrm{{new}}}}+e^{{\ell+v_{{\mathrm{{new}}}}}}r{term}\right)'
             if hierarchical else rf'p_{{\mathrm{{source}}}}(r)=\sigma\!\left(\alpha+e^\ell r{term}\right)')
    return [r'r=d/d_{\mathrm{ref}}', curve]


def render_model(st, drugs, focal_drug_id=None, *, combination=False):
    def get(drug, field):
        return drug.get(field) if isinstance(drug, dict) else getattr(drug, field, None)
    focal = next((drug for drug in drugs if get(drug, 'drug_id') == focal_drug_id), drugs[0] if drugs else {})
    hierarchical = st.session_state.get('workflow_formula_aggregation', 'mixture') == 'historical'
    with st.container(border=True):
        st.markdown('### Dose–risk model')
        context = 'Monotherapy' if combination is False else 'Fixed combination background' if combination else 'Choose companion use'
        st.markdown('**' + context + ' · one focal dose axis**')
        st.caption('Focal drug: ' + (get(focal, 'name') or 'Choose a drug'))
        for equation in model_formulas(combination=combination, hierarchical=hierarchical):
            st.latex(equation)
        st.latex(r'\sigma(x)=\frac{1}{1+e^{-x}}')
        st.caption(r'$r$: relative dose. $r=0.5$ is half the reference; $r=2$ is twice the reference.')
        st.caption(r'$\alpha$: baseline log odds · $\ell$: log dose slope. The slope $e^\ell$ is positive.')
        if combination is True:
            st.caption(r'$\delta\geq0$: combination-risk increase. Companion doses stay fixed; only the focal drug dose varies.')
        elif combination is None:
            st.caption(r'$C=0$: no companion · $C=1$: at least one companion. Set companion use to simplify the formula.')
        if hierarchical:
            st.caption(r'$u_{\mathrm{new}}$: new-trial baseline shift · $v_{\mathrm{new}}$: log-slope shift. One pair is sampled for the entire curve.')
        st.markdown('**Dose and probability notation**')
        st.latex(r'\operatorname{logit}(p)=\log\!\left(\frac{p}{1-p}\right)')
        st.caption(r'$p$: DLT probability. The logistic function $\sigma$ converts log odds to a probability between $0$ and $1$. The reference dose defines a scale, not a recommended dose.')


def render_aggregation(st, *, combination=False):
    st.markdown('### Historical evidence')
    choice = st.radio('Curve distribution', list(ASSEMBLY_CHOICES), index=1, horizontal=True,
        format_func=lambda value: {'average': 'Posterior average', 'mixture': 'Posterior mixture',
                                  'historical': 'Historical likelihood'}[value],
        key='workflow_formula_aggregation')
    with st.container(border=True):
        if choice == 'average':
            st.markdown('**Posterior average**')
            st.caption('Blend a curve from every article using fixed article weights.')
            st.latex(r'\bar p^{(s)}(r)=\sum_i w_i\,p_i^{(s)}(r)')
            st.caption(r'Example at one dose: $0.7\times0.10+0.3\times0.30=0.16$ for one draw.')
        elif choice == 'mixture':
            st.markdown('**Posterior mixture**')
            st.caption('Choose one article curve at a time. Higher-weight articles are chosen more often.')
            st.latex(r'\Pr(I_s=i)=w_i,\qquad p^{(s)}(r)=p_{I_s}^{(s)}(r)')
            st.caption('With weights 0.7 and 0.3, 70% of draws use the first article and 30% use the second, on average. The chosen curve is kept across all doses.')
        else:
            st.markdown('**Historical likelihood**')
            st.caption('Fit one model to discounted historical outcomes, with article-specific intercepts and dose slopes.')
            st.latex(model_formulas(combination=combination, hierarchical=True)[1])
            st.caption('The new trial receives its own baseline and slope shifts. Their distributions are shown under Parameter distributions.')
            st.markdown('**Historical likelihood equations**')
            st.caption('For a cohort with known common exposure:')
            st.latex(r'Y_{ik}\sim\operatorname{Binomial}(n_{ik},p_{ik})')
            st.latex(r'\operatorname{logit}(p_{ik})=\alpha+u_i+e^{\ell+v_i}r_{ik}+\delta C_{ik}')
            st.caption(r'$i$: article · $k$: cohort · $Y_{ik}$: DLT patients · $n_{ik}$: evaluable patients. $u_i$: baseline shift · $v_i$: log-slope shift.')
            st.caption(r'$C_{ik}$ records companion use in each historical cohort, including when the target is monotherapy. Mixed or unreported exposures are integrated over allowed states or reported patient subgroups.')
            st.latex(r'\pi(\Theta\mid\mathcal D)\propto\pi_0(\Theta)\prod_i L_i(\Theta)^{a_i}')
            st.caption(r'$\Theta$: model parameters, article effects, heterogeneity scales and latent exposures. $\pi_0$: joint prior · $\pi$: posterior given historical data $\mathcal D$.')
            st.caption(r'$L_i$: article outcome likelihood · $a_i$: borrowing power. Missing-exposure priors remain unpowered.')
            st.caption(r'$a_i=1$: full likelihood · $a_i=0.5$: half the log-likelihood contribution · $a_i=0$: no outcome information. These powers do not need to sum to $1$.')
    if choice == 'historical':
        st.caption('Selected: one historical-likelihood model with random intercepts and random slopes. The borrowing settings below control how much historical information is used.')
    else:
        st.latex(r'\sum_iw_i=1,\qquad w_i\geq0')
        st.caption(r'$i$: article · $w_i$: article weight · $s$: posterior draw. Each draw is one plausible curve given the data and priors.')
        st.markdown('**Article weights and curves**')
        st.latex(r'w_i=\frac{m_i\,\max_h q_{ih}}{\sum_j m_j\,\max_h q_{jh}}')
        st.latex(r'\Pr(H_i^{(s)}=h)=\frac{q_{ih}}{\sum_g q_{ig}}')
        st.caption(r'$q_{ih}$: similarity of eligible group $h$ in article $i$. $m_i$: review multiplier (Include $=1$, Unsure $=0.5$, Exclude $=0$).')
        st.caption('Each article receives one weight. One eligible group is drawn per article and retained across the whole curve; repeated groups do not give the article extra weight.')
    return ASSEMBLY_CHOICES[choice]
