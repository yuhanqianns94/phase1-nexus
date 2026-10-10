"""BOIN operating characteristics on an explicit full-regimen path.

No source fitting, risk sorting, truth isotonic projection or dose-grid flattening.
"""
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from phase1_prediction.multidrug.contracts import TargetRegimen, fingerprint
from .boin import simulate_boin_trial
from .reference_runner import config_from_settings
from phase1_prediction.reference.oc_simulation import true_mtd_index

REGIMEN_PATH_SIMULATION_VERSION = 'regimen_path_boin_v2'


def path_matrix(prediction, method_key=None):
    from phase1_prediction.multidrug import TARGET_PREDICTION_VERSION
    method = prediction if prediction.get('format') == 'multidrug_target_prediction_v1' else prediction.get('methods', {}).get(method_key, {})
    if method.get('status') != 'fitted':
        raise ValueError('The selected multidrug prediction is not fitted')
    if method.get('prediction_engine_version') != TARGET_PREDICTION_VERSION:
        raise ValueError('This prediction uses an older calculation version; recompute the target prediction before simulation')
    target = TargetRegimen.model_validate(method['target'])
    from phase1_prediction.multidrug.workflow import validate_boin_target
    validate_boin_target(target)
    if len(target.path_level_ids) < 2:
        raise ValueError('BOIN requires an explicitly ordered regimen path with at least two levels; a grid is not a path')
    index = {p.level_id: i for i, p in enumerate(target.points)}
    raw = np.asarray(method['posterior_draws'], dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(target.points) or not len(raw) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError('Invalid joint posterior risk matrix')
    columns = [index[x] for x in target.path_level_ids]
    return raw[:, columns].copy(), [target.points[i].model_dump() for i in columns], method


def simulate_regimen_path(prediction, *, method_key=None, settings, posterior_curve_count=100, trials_per_curve=100, seed=9):
    matrix, points, method = path_matrix(prediction, method_key)
    for name, value in [('posterior_curve_count', posterior_curve_count), ('trials_per_curve', trials_per_curve)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if posterior_curve_count > len(matrix) or posterior_curve_count * trials_per_curve > 250000:
        raise ValueError('Simulation exceeds available joint draws or the 250,000 trial limit')
    config = config_from_settings(settings, tuple(range(1, len(points) + 1)), trials_per_curve, seed)
    overtoxic_threshold = (config.acceptable_high if config.mtd_definition == 'acceptable_interval'
                          else config.target_dlt)
    ids = np.random.default_rng(seed).choice(len(matrix), posterior_curve_count, replace=False)
    truth = matrix[ids]
    streams = np.random.SeedSequence(seed).spawn(posterior_curve_count * trials_per_curve)
    options = asdict(settings)
    for key in ('mtd_definition', 'acceptable_low', 'acceptable_high'):
        options.pop(key)
    counts = np.zeros(len(points)); treated = np.zeros(len(points)); dlts = np.zeros(len(points))
    correct = over = no_selection = early = total_n = total_y = over_patients = 0
    true_counts = np.zeros(len(points))
    trials, per_scenario = [], []
    for mi, probabilities in enumerate(truth):
        true_index = true_mtd_index(probabilities, config)
        if true_index is not None:
            true_counts[true_index] += 1
        scenario_correct = 0
        for ri in range(trials_per_curve):
            trial_number = mi * trials_per_curve + ri
            trial = simulate_boin_trial(probabilities, rng=np.random.default_rng(streams[trial_number]),
                                        allow_nonmonotone_truth=True, **options)
            selected = trial['selected_index']
            ok = selected == true_index
            scenario_correct += ok; correct += ok
            no_selection += selected is None
            if selected is not None:
                counts[selected] += 1
                over += probabilities[selected] > overtoxic_threshold
            ns = np.asarray(trial['treated_by_dose'])
            treated += ns; dlts += trial['dlt_by_dose']
            total_n += trial['total_n']; total_y += trial['total_dlt']
            over_patients += int(ns[probabilities > overtoxic_threshold].sum())
            early += trial['early_stopped']
            trials.append({'scenario_index': mi, 'trial_index': ri, 'source_draw_index': int(ids[mi]), **trial})
        per_scenario.append({'posterior_draw_index': int(ids[mi]), 'truth': probabilities.tolist(),
                             'true_mtd_index': true_index, 'correct_selection_probability': scenario_correct / trials_per_curve})
    n_trials = len(trials)
    result = {'format': 'phase1_v3_multidrug_boin_v1', 'simulation_model_version': REGIMEN_PATH_SIMULATION_VERSION,
              'prediction_run_id': prediction.get('run_id') or method.get('prediction_id'),
              'method_key': method_key, 'backend': method['backend'], 'synthesis': method['synthesis'],
              'path': points, 'doses': list(range(1, len(points) + 1)), 'settings': asdict(settings),
              'seed': seed, 'posterior_curve_count': posterior_curve_count, 'trials_per_curve': trials_per_curve,
              'total_trials': n_trials, 'selection_probability': (counts / n_trials).tolist(),
              'no_selection_probability': no_selection / n_trials, 'true_mtd_probability': (true_counts / len(truth)).tolist(),
              'correct_dose_definition': settings.mtd_definition, 'correct_selection_probability': correct / n_trials,
              'overtoxic_threshold': overtoxic_threshold,
              'overtoxic_selection_probability': over / n_trials, 'early_stop_probability': early / n_trials,
              'pooled_overtoxic_patient_fraction': over_patients / total_n if total_n else 0.,
              'mean_patients_by_dose': (treated / n_trials).tolist(), 'mean_dlts_by_dose': (dlts / n_trials).tolist(),
              'mean_total_n': total_n / n_trials, 'mean_total_dlt': total_y / n_trials,
              'nonmonotone_truth_fraction': float(np.mean(np.any(np.diff(truth, axis=1) < -1e-10, axis=1))),
              'truth_was_projected_or_sorted': False, 'per_scenario': per_scenario,
              'draw_provenance': [method.get('provenance', [])[i] for i in ids] if method.get('provenance') else [],
              'trials': trials}
    result['input_fingerprint'] = fingerprint({'simulation_model_version': result['simulation_model_version'],
                                               'prediction': method.get('prediction_id'), 'truth': truth.tolist(),
                                               'path': points, 'settings': asdict(settings), 'seed': seed,
                                               'trials_per_curve': trials_per_curve})
    result['run_id'] = 'simulation_' + result['input_fingerprint'][:20]
    result['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    return result
