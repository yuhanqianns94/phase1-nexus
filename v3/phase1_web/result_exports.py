"""Readable, portable results. Exports never rerun inference or simulation."""
import base64
import csv
from html import escape
from io import BytesIO, StringIO
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import PercentFormatter


RISK_COLUMNS = {'posterior_mean': 'Mean DLT probability', 'posterior_median': 'Median DLT probability',
                'low_95': '95% credible interval lower', 'high_95': '95% credible interval upper',
                'probability_above_target': 'Probability above target DLT rate'}
PATIENT_PERCENT = 'Patients treated (%)'
TOXICITY_STOP_PERCENT = 'Early stop for toxicity (%)'
ALLOCATION_NOTE = 'Patient percentages use all patients actually enrolled across the simulated trials, including trials that stopped early.'
TOXICITY_STOP_NOTE = 'Early stop for toxicity counts trials stopped by toxicity safety rules; it is separate from no dose selected and from stopping after enough patients at one dose.'
SIM_METRICS = {'total_trials': 'Simulated trials', 'posterior_curve_count': 'Posterior scenarios',
               'trials_per_curve': 'Trials per scenario', 'correct_selection_probability': 'Correct selection probability',
               'overtoxic_selection_probability': 'Overtoxic selection probability',
               'no_selection_probability': 'No selection probability', 'early_stop_probability': TOXICITY_STOP_PERCENT,
               'mean_total_n': 'Mean patients per trial', 'mean_total_dlt': 'Mean DLT patients per trial',
               'pooled_overtoxic_patient_fraction': 'Fraction of patients treated above toxicity threshold',
               'overtoxic_threshold': 'Overtoxicity threshold', 'correct_dose_definition': 'Correct dose definition'}


def method_of(result):
    return next(iter(result.get('methods', {}).values()), {})


def attached_result(result, attached):
    """Never include simulation from another prediction, even for saved analyses."""
    return attached if attached and result.get('run_id') and attached.get('prediction_run_id') == result['run_id'] else None


def dose_columns(target, doses):
    return {d['name'] + (' (' + str(d.get('unit') or 'relative dose') + ')' if d.get('mode') == 'numeric' else ' use'):
            doses.get(d['drug_id']) if doses.get(d['drug_id']) is not None else 'Used; dose unknown'
            for d in target['drugs']}


def summary_rows(result):
    method = method_of(result)
    target = method.get('target', result.get('target', {}))
    return [{'Dose level': r['level_id'], **dose_columns(target, r['doses']),
             **{label: r.get(key) for key, label in RISK_COLUMNS.items()}}
            for r in method.get('summary', [])]


def simulation_metrics(sim):
    return [{'Measure': label, 'Value': 100 * sim[key] if key == 'early_stop_probability' else sim[key]}
            for key, label in SIM_METRICS.items() if key in sim]


def simulation_doses(result, sim):
    rows = []
    total = sim.get('mean_total_n')
    for i, point in enumerate(sim.get('path', [])):
        row = {'Dose level': point['level_id'], **dose_columns(result['target'], point['doses'])}
        for key, label in [('selection_probability', 'Selection probability'), ('true_mtd_probability', 'True MTD probability'),
                           ('mean_patients_by_dose', 'Mean patients'), ('mean_dlts_by_dose', 'Mean DLT patients')]:
            if i < len(sim.get(key, [])):
                row[label] = sim[key][i]
        # Ratio of means equals the pooled patient fraction because both means
        # use the same trial count. Do not average per-trial fractions or divide
        # by the maximum planned sample size when trials can stop early.
        mean = row.get('Mean patients')
        row[PATIENT_PERCENT] = 100 * mean / total if mean is not None and total is not None and total > 0 else None
        rows.append(row)
    return rows


def design_rows(search):
    keys = {'candidate_label': 'Design', 'max_sample_size': 'Maximum patients', 'cohort_size': 'Cohort size',
            'target_dlt': 'Target DLT probability', 'elimination_cutoff': 'Elimination probability',
            'stop_rule_m': 'Patients at one dose before stopping',
            **{k: SIM_METRICS[k] for k in ['correct_selection_probability', 'overtoxic_selection_probability',
                                         'no_selection_probability', 'mean_total_n']}}
    rows = []
    for row in search.get('ranked', []):
        values = {label: row.get(key) for key, label in keys.items()}
        stop = row.get('early_stop_probability', row.get('result', {}).get('early_stop_probability'))
        values[TOXICITY_STOP_PERCENT] = 100 * stop if stop is not None else None
        rows.append(values)
    return rows


def design_doses(result, search):
    return [{'Design': candidate['candidate_label'], **row}
            for candidate in search.get('ranked', [])
            for row in simulation_doses(result, candidate.get('result', {}))]


def notices(result):
    method = method_of(result)
    messages = []
    if method.get('inference') == 'explicit_prior_predictive':
        messages.append('Historical borrowing is zero: these curves use only the specified priors.')
    support = method.get('coordinate_support', [])
    outside = sum('Outside' in row.get('support', '') for row in support)
    incomplete = sum('incomplete' in row.get('support', '').lower() for row in support)
    if outside:
        messages.append(f'{outside} of {len(support)} evaluated dose levels extend beyond at least one historical source range.')
    if incomplete:
        messages.append('Some historical dose or use-state information is incomplete.')
    return messages


def _csv(rows, headers=None):
    headers = headers or list(dict.fromkeys(k for row in rows for k in row))
    text = StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=headers)
    writer.writeheader()
    # Escape spreadsheet formulas in untrusted article/drug names, preserving numbers.
    for row in rows:
        writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v
                         for k, v in row.items()})
    return text.getvalue().encode('utf-8-sig')


def _png(fig):
    output = BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return output.getvalue()


def prediction_plot(result):
    method = method_of(result)
    if method.get('status') != 'fitted' or not method.get('summary'):
        return None
    target = method.get('target', result['target'])
    rows = method['summary']
    varying = [d for d in target['drugs'] if d.get('mode') == 'numeric'
               and len({r['doses'].get(d['drug_id']) for r in rows}) > 1]
    fig = Figure(figsize=(9, 5.4), dpi=170, layout='constrained')
    ax = fig.subplots()
    if len(varying) == 1:
        drug = varying[0]; did = drug['drug_id']
        # Separate fixed backgrounds rather than connecting different regimens.
        groups = {}
        for row in rows:
            group = tuple((d['name'], row['doses'].get(d['drug_id'])) for d in target['drugs'] if d['drug_id'] != did)
            groups.setdefault(group, []).append(row)
        for group, points in groups.items():
            points = sorted(points, key=lambda r: r['doses'].get(did) or 0)
            xs = [r['doses'].get(did) or 0 for r in points]
            line, = ax.plot(xs, [r['posterior_median'] for r in points], linewidth=2,
                            label='Posterior median' if len(groups) == 1 else ', '.join(f'{name}: {dose}' for name, dose in group))
            ax.fill_between(xs, [r['low_95'] for r in points], [r['high_95'] for r in points], color=line.get_color(), alpha=.17,
                            label='95% credible interval' if len(groups) == 1 else None)
        ax.set_xlabel(drug['name'] + ' (' + str(drug.get('unit') or 'relative dose') + ')')
    else:
        xs = np.arange(len(rows)); med = np.array([r['posterior_median'] for r in rows])
        ax.errorbar(xs, med, yerr=[med - [r['low_95'] for r in rows], np.array([r['high_95'] for r in rows]) - med],
                    fmt='o', color='#0b7770', capsize=3, label='Median and 95% credible interval')
        ax.set_xticks(xs, [r['level_id'] for r in rows], rotation=45)
        ax.set_xlabel('Dose level (see posterior summary for doses)')
    ax.axhline(target['target_dlt'], linestyle='--', color='#b65b38', label='Target DLT probability')
    ax.set(ylim=(0, 1), ylabel='DLT probability', title=target.get('name', 'Dose–toxicity prediction'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15); ax.legend(frameon=False, fontsize=9)
    return _png(fig)


def simulation_plot(result, sim):
    rows = simulation_doses(result, sim)
    if not rows:
        return None
    fig = Figure(figsize=(9, 5.4), dpi=170, layout='constrained')
    ax = fig.subplots()
    xs = np.arange(len(rows))
    selection_bars = ax.bar(xs - .2, [100 * r.get('Selection probability', 0) for r in rows], width=.4,
           color='#0b7770', label='Trials selecting dose')
    patient_bars = ax.bar(xs + .2, [r[PATIENT_PERCENT] if r[PATIENT_PERCENT] is not None else np.nan for r in rows],
           width=.4, color='#bd6137', label='Patients treated')
    for bars in (selection_bars, patient_bars):
        ax.bar_label(bars, labels=[f'{bar.get_height():.1f}%' if np.isfinite(bar.get_height()) else '' for bar in bars],
                     padding=3, fontsize=8)
    ax.set_xticks(xs, [r['Dose level'] for r in rows], rotation=45)
    ax.set(ylim=(0, 100), ylabel='Percentage', xlabel='BOIN path dose level', title='Dose selection and patient allocation')
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    stop = sim.get('early_stop_probability')
    stop_text = f'{stop:.1%}' if stop is not None else 'Unavailable'
    ax.text(.99, .98, f"Early stop for toxicity: {stop_text}\nNo dose selected: {sim.get('no_selection_probability', 0):.1%}",
            transform=ax.transAxes, ha='right', va='top')
    return _png(fig)


def _table(rows, headers=None):
    headers = headers or list(dict.fromkeys(k for row in rows for k in row))
    def cell(value, percentage=False):
        if percentage and isinstance(value, (int, float)):
            return f'{value:.1f}%'
        return escape(f'{value:.6g}' if isinstance(value, float) else str(value if value is not None else ''))
    return '<table><thead><tr>' + ''.join('<th>' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + cell(row.get(h), h.endswith('(%)') or
            (h == 'Value' and row.get('Measure', '').endswith('(%)'))) + '</td>' for h in headers) + '</tr>' for row in rows) + '</tbody></table>'


def build_results_bundle(result, simulation=None, search=None):
    """One ZIP: self-contained report, PNGs and numeric CSVs; no raw JSON."""
    method = method_of(result)
    if method.get('status') != 'fitted':
        raise ValueError('Generate a fitted prediction before downloading results.')
    simulation = attached_result(result, simulation)
    search = attached_result(result, search)
    target = method.get('target', result['target'])
    summaries = summary_rows(result)
    levels = [p['level_id'] for p in target['points']]
    matrix = np.asarray(method.get('posterior_draws', []), dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(levels) or not len(matrix) or not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError('Posterior risk draws are missing or invalid; regenerate this prediction.')
    from phase1_web.multidrug_app import prediction_result_label
    label = prediction_result_label(method['backend'] + ' · ' + method['synthesis'])
    settings = [{'Setting': 'Prediction run', 'Value': result.get('run_id')},
                {'Setting': 'Calculation version', 'Value': method.get('prediction_engine_version')},
                {'Setting': 'Analysis', 'Value': label}, {'Setting': 'Target DLT probability', 'Value': target['target_dlt']}]
    for group, data in [('Parameter prior', result.get('evidence_contract', {}).get('prior', {})),
                        ('Historical borrowing', method.get('power_prior') or {}),
                        ('Simulation', (simulation or {}).get('settings', {}))]:
        settings.extend({'Setting': group + ': ' + k.replace('_', ' '), 'Value': v}
                        for k, v in data.items() if not isinstance(v, (dict, list))
                        and k != 'prior_id' and not (group == 'Parameter prior' and k == 'pair_sd' and method['backend'] == 'single_drug_logistic'))
    if simulation:
        settings.extend({'Setting': 'Simulation: ' + k.replace('_', ' '), 'Value': simulation.get(k)}
                        for k in ['seed', 'simulation_model_version', 'posterior_curve_count', 'trials_per_curve'])
    settings.extend({'Setting': d['name'] + ': reference dose (' + str(d.get('unit') or 'relative dose') + ')',
                     'Value': d.get('reference_dose')} for d in target['drugs'] if d.get('mode') == 'numeric')
    diagnostics = [{k.replace('_', ' ').capitalize(): v for k, v in attempt.items() if not isinstance(v, (dict, list))}
                   for attempt in (method.get('sampling_diagnostics') or {}).get('attempts', [])]
    support = [{'Dose level': row['level_id'], 'Historical dose support': row['support']} for row in method.get('coordinate_support', [])]
    metrics = simulation_metrics(simulation) if simulation else []
    sim_doses = simulation_doses(result, simulation) if simulation else []
    pred_png = prediction_plot(result)
    sim_png = simulation_plot(result, simulation) if simulation else None
    files = {'posterior_summary.csv': _csv(summaries),
             'posterior_risk_draws.csv': _csv([{'Draw': i + 1, **dict(zip(levels, values))} for i, values in enumerate(matrix)]),
             'analysis_settings.csv': _csv(settings), 'dose_support.csv': _csv(support, ['Dose level', 'Historical dose support']),
             'sampling_checks.csv': _csv(diagnostics, list(diagnostics[0]) if diagnostics else ['Attempt', 'Passed']),
             'simulation_summary.csv': _csv(metrics, ['Measure', 'Value']),
             'simulation_by_dose.csv': _csv(sim_doses, list(sim_doses[0]) if sim_doses else ['Dose level', 'Selection probability', 'Mean patients', 'Mean DLT patients', PATIENT_PERCENT])}
    if simulation:
        trials = []
        for trial in simulation.get('trials', []):
            selected = trial.get('selected_index')
            row = {'Scenario': trial['scenario_index'] + 1, 'Trial': trial['trial_index'] + 1,
                   'Posterior draw': trial['source_draw_index'] + 1,
                   'Selected dose level': simulation['path'][selected]['level_id'] if selected is not None else '',
                   'Patients': trial['total_n'], 'DLT patients': trial['total_dlt'], 'Early stopped for toxicity': trial['early_stopped'],
                   'Stop reason': trial.get('stop_reason', '')}
            for i, point in enumerate(simulation['path']):
                row[point['level_id'] + ' patients'] = trial['treated_by_dose'][i]
                row[point['level_id'] + ' DLT patients'] = trial['dlt_by_dose'][i]
            trials.append(row)
        files['simulation_trials.csv'] = _csv(trials)
        scenarios = []
        for i, scenario in enumerate(simulation.get('per_scenario', []), 1):
            true_index = scenario.get('true_mtd_index')
            scenarios.append({'Scenario': i, 'Posterior draw': scenario['posterior_draw_index'] + 1,
                              'True MTD': simulation['path'][true_index]['level_id'] if true_index is not None else '',
                              'Correct selection probability': scenario['correct_selection_probability'],
                              **{point['level_id'] + ' DLT probability': risk for point, risk in zip(simulation['path'], scenario['truth'])}})
        files['simulation_scenarios.csv'] = _csv(scenarios)
    if search:
        files['design_comparison.csv'] = _csv(design_rows(search))
        files['design_comparison_by_dose.csv'] = _csv(design_doses(result, search),
            ['Design', 'Dose level', *dose_columns(result['target'], {}), 'Selection probability',
             'True MTD probability', 'Mean patients', 'Mean DLT patients', PATIENT_PERCENT])
    def img(data):
        return '<img src="data:image/png;base64,' + base64.b64encode(data).decode() + '" alt="Result chart">' if data else ''
    report = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>PHASE I NEXUS results</title><style>'
              'body{font:16px system-ui;color:#193740;max-width:1050px;margin:48px auto;padding:0 24px;line-height:1.6}'
              'h1,h2{color:#0b7770}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}'
              'th,td{text-align:left;padding:9px;border-bottom:1px solid #dbe6e6}th{background:#edf5f4}'
              'img{width:100%;height:auto}section{overflow-x:auto}p{color:#506b73}</style><body>'
              '<h1>PHASE I NEXUS · Results</h1><p>' + escape(target.get('name', 'Target trial')) + ' · ' + escape(label) + '</p>'
              '<p>Predictions use historical evidence and model assumptions; target-trial patient outcomes have not been observed. '
              'Posterior risk draws retain each sampled curve across all dose levels. They are risk probabilities, not regression-parameter samples.</p>'
              + ''.join('<p>' + escape(note) + '</p>' for note in notices(result))
              + '<h2>Dose–toxicity prediction</h2>' + img(pred_png) + '<section>' + _table(summaries) + '</section>'
              '<h2>Trial simulation</h2>' + img(sim_png) + '<section>' + _table(metrics, ['Measure', 'Value']) + _table(sim_doses) + '</section>'
              + ('<p>' + escape(ALLOCATION_NOTE) + ' ' + escape(TOXICITY_STOP_NOTE) + '</p>' if simulation else '')
              + ('<h2>BOIN design comparison</h2><section>' + _table(design_rows(search)) + _table(design_doses(result, search)) + '</section>'
                 '<p>' + escape(ALLOCATION_NOTE) + ' ' + escape(TOXICITY_STOP_NOTE) + '</p>' if search else '')
              + '<h2>Analysis settings</h2>' + _table(settings)
              + '<h2>Historical dose support</h2>' + _table(support)
              + '<h2>Sampling checks</h2>' + _table(diagnostics) + '</body></html>')
    files['report.html'] = report.encode()
    files['README.txt'] = ('Open report.html for the illustrated results. CSV files open in spreadsheet software.\n'
        'Probabilities are on the 0–1 scale; fields labelled (%) are percentages on the 0–100 scale. Normal prior settings use mean and standard deviation (sd); charts use 95% credible intervals.\n'
        + ALLOCATION_NOTE + '\n' + TOXICITY_STOP_NOTE + '\n'
        'simulation_by_dose.csv and design_comparison_by_dose.csv include patient allocation percentages; simulation_summary.csv and design_comparison.csv include toxicity early-stopping percentages.\n'
        'posterior_risk_draws.csv: each row is one entire risk curve; columns are dose levels in posterior_summary.csv. These are not coefficient draws.\n'
        'Simulation files and the report simulation section are blank when no simulation is attached to this prediction.\n'
        'Analysis settings describe the calculated result, not subsequent edits to inputs.\n').encode()
    if pred_png:
        files['prediction.png'] = pred_png
    if sim_png:
        files['simulation.png'] = sim_png
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    return output.getvalue()
