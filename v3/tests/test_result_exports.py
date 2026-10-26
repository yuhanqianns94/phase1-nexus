"""Exports preserve fitted risk draws and only their own simulation results."""
import copy
import csv
from io import BytesIO, StringIO
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from phase1_prediction.multidrug.contracts import PriorSpec
from phase1_prediction.multidrug.transport import transport_joint_posteriors
from phase1_simulation.multidrug import simulate_regimen_path
from phase1_simulation.service import BoinSettings
from phase1_web.result_exports import (build_results_bundle, simulation_doses, simulation_metrics,
    PATIENT_PERCENT, TOXICITY_STOP_PERCENT)
from test_multidrug_transport import fixture


def prediction():
    scope, arrays, manifest, target = fixture(1)
    method = transport_joint_posteriors([(scope, arrays, manifest)], target, draw_count=4)
    method['backend'] = 'single_drug_logistic'
    method['coordinate_support'] = [{'level_id': target.points[0].level_id, 'support': 'Outside at least one source dose/use-state range'}]
    return {'run_id': 'export-test', 'target': target.model_dump(mode='json'),
            'methods': {method['synthesis']: method}, 'evidence_contract': {'prior': PriorSpec().model_dump()}}


def rows(bundle, filename):
    return list(csv.DictReader(StringIO(bundle.read(filename).decode('utf-8-sig'))))


def test_export_preserves_entire_draws_and_leaves_missing_simulation_blank():
    result = prediction(); before = copy.deepcopy(result)
    with ZipFile(BytesIO(build_results_bundle(result))) as bundle:
        assert not any(name.endswith('.json') for name in bundle.namelist())
        summary = rows(bundle, 'posterior_summary.csv')
        assert 'A (mg)' in summary[0] and 'doses' not in summary[0]
        draws = rows(bundle, 'posterior_risk_draws.csv')
        levels = [p['level_id'] for p in result['target']['points']]
        actual = np.array([[float(row[level]) for level in levels] for row in draws])
        np.testing.assert_array_equal(actual, next(iter(result['methods'].values()))['posterior_draws'])
        assert rows(bundle, 'simulation_summary.csv') == []
        assert rows(bundle, 'simulation_by_dose.csv') == []
        assert 'simulation.png' not in bundle.namelist()
        Image.open(BytesIO(bundle.read('prediction.png'))).verify()
        report = bundle.read('report.html').decode()
        assert '<h2>Trial simulation</h2>' in report
        assert 'extend beyond' in report
        assert 'base64,' in report and 'intercept sd' in report
    assert result == before


def test_export_includes_actual_simulation_and_rejects_stale_simulation():
    result = prediction()
    sim = simulate_regimen_path(result, method_key=next(iter(result['methods'])),
        settings=BoinSettings(max_sample_size=6), posterior_curve_count=2, trials_per_curve=2)
    with ZipFile(BytesIO(build_results_bundle(result, sim))) as bundle:
        assert len(rows(bundle, 'simulation_trials.csv')) == 4
        stats = {r['Measure']: r['Value'] for r in rows(bundle, 'simulation_summary.csv')}
        assert float(stats['Correct selection probability']) == sim['correct_selection_probability']
        assert float(stats['Mean patients per trial']) == sim['mean_total_n']
        safety_stops = sum(t['early_stopped'] for t in sim['trials'])
        assert float(stats[TOXICITY_STOP_PERCENT]) == pytest.approx(100 * safety_stops / len(sim['trials']))
        by_dose = rows(bundle, 'simulation_by_dose.csv')
        counts = np.array([t['treated_by_dose'] for t in sim['trials']]).sum(axis=0)
        np.testing.assert_allclose([float(r[PATIENT_PERCENT]) for r in by_dose], 100 * counts / counts.sum())
        assert PATIENT_PERCENT in bundle.read('report.html').decode()
        assert TOXICITY_STOP_PERCENT in bundle.read('report.html').decode()
        Image.open(BytesIO(bundle.read('simulation.png'))).verify()
    sim['prediction_run_id'] = 'another-prediction'
    with ZipFile(BytesIO(build_results_bundle(result, sim))) as bundle:
        assert not rows(bundle, 'simulation_summary.csv')
        assert 'simulation.png' not in bundle.namelist()
        assert 'simulation_trials.csv' not in bundle.namelist()


def test_invalid_posterior_cannot_be_exported():
    result = prediction()
    next(iter(result['methods'].values()))['posterior_draws'] = [[float('nan')]]
    with pytest.raises(ValueError, match='missing or invalid'):
        build_results_bundle(result)


def test_report_escapes_names_and_csv_formulas():
    result = prediction()
    result['target']['name'] = '<script>alert(1)</script>'
    next(iter(result['methods'].values()))['target']['name'] = result['target']['name']
    next(iter(result['methods'].values()))['summary'][0]['level_id'] = '=1+1'
    with ZipFile(BytesIO(build_results_bundle(result))) as bundle:
        assert '<script>' not in bundle.read('report.html').decode()
        assert rows(bundle, 'posterior_summary.csv')[0]['Dose level'] == "'=1+1"


def test_download_workflow_without_simulation_and_simplified_results(tmp_path, monkeypatch):
    from test_multidrug_ui_state_audit import app_with_library
    app = app_with_library(tmp_path, monkeypatch)
    app.sidebar.radio[0].set_value('Download results').run()
    assert not app.exception
    assert 'Generate a fitted prediction' in app.info[0].value
    result = prediction()
    app.session_state['multidrug_prediction'] = result
    app.session_state['multidrug_results'] = {'single_drug_logistic': result}
    app.run()
    assert not app.exception and not app.error
    assert app.get('download_button')[0].label == 'Download report, figures and data (ZIP)'
    app.sidebar.radio[0].set_value('Prediction Results').run()
    assert not app.exception and not app.error and not app.json
    assert not any(x.label in ('Applied parameter priors', 'Dose support and extrapolation',
        'Power prior settings and sampling diagnostics', 'Drug correspondence, source weights and exclusions') for x in app.expander)
    next(b for b in app.button if b.label == 'Download without trial simulation').click().run()
    assert app.sidebar.radio[0].value == 'Download results'
    options = app.sidebar.radio[0].options
    assert options.index('5. Download results') == options.index('4. Simulate a trial (optional)') + 1
    assert not app.exception


def test_design_comparison_is_readable_and_attached_to_prediction():
    result = prediction()
    search = {'prediction_run_id': result['run_id'], 'ranked': [{'candidate_label': 'Design 1',
        'max_sample_size': 30, 'cohort_size': 3, 'correct_selection_probability': .42,
        'result': {'internal': 'payload'}}]}
    with ZipFile(BytesIO(build_results_bundle(result, search=search))) as bundle:
        designs = rows(bundle, 'design_comparison.csv')
        assert designs[0]['Maximum patients'] == '30'
        assert float(designs[0]['Correct selection probability']) == .42
        assert 'result' not in designs[0]
        assert rows(bundle, 'simulation_summary.csv') == []
    search['prediction_run_id'] = 'stale'
    with ZipFile(BytesIO(build_results_bundle(result, search=search))) as bundle:
        assert 'design_comparison.csv' not in bundle.namelist()


def unequal_length_simulation(result):
    # Four trials enroll 3, 9, 12 and 6 patients. One stops for toxicity;
    # another finishes without an admissible recommendation.
    return {'prediction_run_id': result['run_id'], 'path': result['target']['points'],
            'total_trials': 4, 'posterior_curve_count': 2, 'trials_per_curve': 2,
            'mean_patients_by_dose': [1.5, 2.25, 3.75], 'mean_total_n': 7.5,
            'mean_dlts_by_dose': [.5, .5, .5], 'mean_total_dlt': 1.5,
            'selection_probability': [0, .25, .25], 'true_mtd_probability': [0, .5, .5],
            'no_selection_probability': .5, 'early_stop_probability': .25,
            'correct_selection_probability': .5, 'overtoxic_selection_probability': 0}


def test_patient_percentages_pool_actual_patients_and_keep_safety_distinct(tmp_path):
    result = prediction()
    sim = unequal_length_simulation(result)
    before = copy.deepcopy(sim)
    allocations = simulation_doses(result, sim)
    assert [row[PATIENT_PERCENT] for row in allocations] == [20, 30, 50]
    metrics = {r['Measure']: r['Value'] for r in simulation_metrics(sim)}
    assert metrics[TOXICITY_STOP_PERCENT] == 25
    assert metrics['No selection probability'] == .5
    # Missing/empty historical results must not report a fictitious zero percent.
    assert all(r[PATIENT_PERCENT] is None for r in simulation_doses(result, {**sim, 'mean_total_n': 0}))
    assert sim == before

    from streamlit.testing.v1 import AppTest
    def show(result, sim):
        import streamlit as st
        from phase1_web.results_view import render_simulation
        render_simulation(st, result, sim)
    app = AppTest.from_function(show, args=(result, sim)).run()
    assert not app.exception
    values = {metric.label: metric.value for metric in app.metric}
    assert values['Early stop for toxicity'] == '25.0%'
    assert values['No dose selected'] == '50.0%'
    import pyarrow as pa
    chart = app.get('vega_lite_chart')[0].proto
    data = pa.ipc.open_stream(chart.datasets[0].data.data).read_all().to_pandas()
    assert list(data.loc[data['Measure'] == 'Patients treated', 'Percentage']) == [20, 30, 50]


def test_design_comparison_and_exports_include_patient_percentages():
    result = prediction()
    sim = unequal_length_simulation(result)
    search = {'prediction_run_id': result['run_id'], 'ranked': [{
        'candidate_label': 'Design 1', 'max_sample_size': 12, 'cohort_size': 3,
        'early_stop_probability': .25, 'result': sim}]}
    with ZipFile(BytesIO(build_results_bundle(result, sim, search))) as bundle:
        assert float(rows(bundle, 'design_comparison.csv')[0][TOXICITY_STOP_PERCENT]) == 25
        allocations = rows(bundle, 'design_comparison_by_dose.csv')
        assert [float(row[PATIENT_PERCENT]) for row in allocations] == [20, 30, 50]
        assert all(row['Design'] == 'Design 1' for row in allocations)
        stats = {r['Measure']: r['Value'] for r in rows(bundle, 'simulation_summary.csv')}
        assert float(stats[TOXICITY_STOP_PERCENT]) == 25
        assert '0–100' in bundle.read('README.txt').decode()
    from streamlit.testing.v1 import AppTest
    def show(result, search):
        import streamlit as st
        from phase1_web.results_view import render_design_comparison
        render_design_comparison(st, result, search)
    app = AppTest.from_function(show, args=(result, search)).run()
    assert not app.exception
    assert app.dataframe[0].value[TOXICITY_STOP_PERCENT].tolist() == [25]
    assert len(app.get('vega_lite_chart')) == 1
