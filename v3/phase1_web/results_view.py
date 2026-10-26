"""Patient-risk and trial-design results without implementation payloads."""
import altair as alt
import pandas as pd

from phase1_web.result_exports import (method_of, attached_result, build_results_bundle,
    simulation_doses, design_rows, design_doses, PATIENT_PERCENT, TOXICITY_STOP_PERCENT,
    ALLOCATION_NOTE, TOXICITY_STOP_NOTE)


def render_comparison(st, current, comparisons):
    from phase1_web.multidrug_app import prediction_result_label
    valid = [(label, result) for label, result in comparisons.items()
             if result.get('target') == current.get('target') and method_of(result).get('status') == 'fitted']
    if len(valid) < 2:
        return
    st.subheader('Compare analysis settings')
    st.caption('Compare completed analyses for the same target and selected evidence. Curves show posterior medians under different borrowing methods or prior settings; this does not rank their accuracy.')
    target = current['target']
    rows = []
    levels = [p['level_id'] for p in target['points']]
    varying = [d for d in target['drugs'] if d.get('mode') == 'numeric'
               and len({p['doses'].get(d['drug_id']) for p in target['points']}) > 1]
    numeric = varying[0] if len(varying) == 1 else None
    if numeric:
        # A single varying dose may still have several backgrounds. Use dose-level
        # categories in that case so separate regimens are never joined as one curve.
        others = [d['drug_id'] for d in target['drugs'] if d['drug_id'] != numeric['drug_id']]
        if len({tuple(p['doses'].get(did) for did in others) for p in target['points']}) > 1:
            numeric = None
    for i, (_, result) in enumerate(valid, 1):
        for row in method_of(result).get('summary', []):
            rows.append({'Analysis': f'Analysis {i}', 'Dose level': row['level_id'],
                         'Dose': row['doses'].get(numeric['drug_id']) if numeric else row['level_id'],
                         'Median DLT probability': row['posterior_median']})
    axis = (alt.X('Dose:Q', title=numeric['name'] + ' (' + str(numeric.get('unit') or 'relative dose') + ')')
            if numeric else alt.X('Dose:O', sort=levels, title='Dose level'))
    chart = alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
        x=axis, y=alt.Y('Median DLT probability:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Analysis:N', scale=alt.Scale(range=['#0b7770', '#bd6137', '#5369a4', '#9a5b89', '#758b35'])),
        strokeDash='Analysis:N', tooltip=['Analysis', 'Dose level', 'Dose', 'Median DLT probability'])
    st.altair_chart(chart, use_container_width=True)
    for i, (label, _) in enumerate(valid, 1):
        st.caption(f'Analysis {i}: {prediction_result_label(label)}')


def render_simulation(st, result, sim):
    st.subheader('Simulation results')
    columns = list(st.columns(3)) + list(st.columns(2))
    stop = sim.get('early_stop_probability')
    for col, label, value in zip(columns,
        ['Correct selection', 'Overtoxic selection', 'No dose selected', 'Early stop for toxicity', 'Mean patients'],
        [f"{sim['correct_selection_probability']:.1%}", f"{sim['overtoxic_selection_probability']:.1%}",
         f"{sim['no_selection_probability']:.1%}", f'{stop:.1%}' if stop is not None else '—', f"{sim['mean_total_n']:.1f}"]):
        col.metric(label, value)
    st.caption(TOXICITY_STOP_NOTE)
    rows = simulation_doses(result, sim)
    if rows:
        chart_rows = [{**row, 'Measure': label, 'Percentage': value}
                      for row in rows for label, value in [
                          ('Trials selecting dose', 100 * row['Selection probability']),
                          ('Patients treated', row[PATIENT_PERCENT])]]
        st.markdown('**Dose selection and patient allocation**')
        chart = alt.Chart(pd.DataFrame(chart_rows)).encode(
            x=alt.X('Dose level:N', sort=[r['Dose level'] for r in rows], title='BOIN path dose level'),
            xOffset=alt.XOffset('Measure:N', sort=['Trials selecting dose', 'Patients treated']),
            y=alt.Y('Percentage:Q', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelExpr="datum.value + '%'")),
            color=alt.Color('Measure:N', scale=alt.Scale(domain=['Trials selecting dose', 'Patients treated'],
                range=['#0b7770', '#bd6137']), legend=alt.Legend(title=None, orient='top')),
            tooltip=['Dose level', 'Measure', alt.Tooltip('Percentage:Q', format='.1f'), 'Mean patients', 'Mean DLT patients'])
        labels = chart.transform_calculate(Label="format(datum.Percentage, '.1f') + '%'").mark_text(
            dy=-7, color='#193740').encode(text='Label:N', color=alt.value('#193740'))
        st.altair_chart(chart.mark_bar() + labels, use_container_width=True)
        st.caption(ALLOCATION_NOTE)
    st.caption(f"{sim['total_trials']:,} virtual trials across {sim['posterior_curve_count']} posterior scenarios. Results describe this design under the sampled risk curves.")


def render_design_comparison(st, result, search):
    summaries = design_rows(search)
    columns = ['Design', TOXICITY_STOP_PERCENT] + [key for key in summaries[0] if key not in ('Design', TOXICITY_STOP_PERCENT)] if summaries else []
    st.dataframe(summaries, hide_index=True, use_container_width=True, column_order=columns,
                 column_config={TOXICITY_STOP_PERCENT: st.column_config.NumberColumn(format='%.1f%%')})
    st.caption(TOXICITY_STOP_NOTE)
    rows = design_doses(result, search)
    if rows:
        st.markdown('**Patient allocation by design**')
        chart = alt.Chart(pd.DataFrame(rows)).encode(
            x=alt.X('Dose level:N', sort=list(dict.fromkeys(r['Dose level'] for r in rows)), title='BOIN path dose level'),
            xOffset='Design:N', color=alt.Color('Design:N', legend=alt.Legend(orient='top')),
            y=alt.Y(PATIENT_PERCENT + ':Q', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelExpr="datum.value + '%'")),
            tooltip=['Design', 'Dose level', alt.Tooltip(PATIENT_PERCENT + ':Q', format='.1f'), 'Mean patients'])
        labels = chart.transform_calculate(Label="format(datum['Patients treated (%)'], '.1f') + '%'").mark_text(
            dy=-7).encode(text='Label:N', color=alt.value('#193740'))
        st.altair_chart(chart.mark_bar() + labels, use_container_width=True)
        st.caption(ALLOCATION_NOTE)


def render_downloads(st, result, simulation=None, search=None, *, key='current_results'):
    st.subheader('Download results')
    if not result or method_of(result).get('status') != 'fitted':
        st.info('Generate a fitted prediction to download your results. Trial simulation is optional.')
        return
    simulation = attached_result(result, simulation)
    search = attached_result(result, search)
    st.write('Download a report with figures, posterior summaries and complete posterior risk draws. CSV files open in Excel or other analysis software.')
    st.caption('The package includes simulation results when available. If simulation has not been run for this prediction, its report section and result files remain blank.')
    try:
        with st.spinner('Preparing figures and results'):
            data = build_results_bundle(result, simulation, search)
        st.download_button('Download report, figures and data (ZIP)', data,
                           file_name='phase1_nexus_results.zip', mime='application/zip',
                           key=key, type='primary', on_click='ignore')
    except (ValueError, TypeError, KeyError) as error:
        st.error('This result could not be exported: ' + str(error))
