"""Keep prediction, simulation and design comparisons attached to one run."""


def clear_dependents(state):
    for key in ('multidrug_simulation', 'multidrug_search'):
        state.pop(key, None)


def invalidate_prediction(state):
    """An invalid draft or settings cannot retain an active evidence result."""
    state['multidrug_prediction'] = None
    state['multidrug_results'] = {}
    state.pop('multidrug_preview', None)
    clear_dependents(state)


def activate_prediction(state, result, *, recomputed=False):
    old_id = (state.get('multidrug_prediction') or {}).get('run_id')
    new_id = (result or {}).get('run_id')
    if recomputed or old_id != new_id:
        clear_dependents(state)
    state['multidrug_prediction'] = result
    discard_stale_dependents(state)


def discard_stale_dependents(state):
    run_id = (state.get('multidrug_prediction') or {}).get('run_id')
    for key in ('multidrug_simulation', 'multidrug_search'):
        attached = state.get(key)
        if attached and (not run_id or attached.get('prediction_run_id') != run_id):
            state.pop(key, None)


def simulation_view(prediction, path_ids):
    """Select columns for simulation without changing a prediction or its draws."""
    import copy
    target=prediction['target']
    ids={p['level_id'] for p in target['points']}
    if len(path_ids)<2 or len(set(path_ids))!=len(path_ids) or not set(path_ids)<=ids:
        raise ValueError('Select at least two distinct regimens from this prediction.')
    result=copy.deepcopy(prediction)
    result['target']['path_level_ids']=list(path_ids)
    for method in result['methods'].values():
        method['target']['path_level_ids']=list(path_ids)
    return result


def automatic_boin_path(target):
    """Order one focal dose axis without altering fitted columns or probabilities."""
    from phase1_prediction.multidrug.contracts import TargetRegimen
    from phase1_prediction.multidrug.workflow import validate_boin_target
    target = TargetRegimen.model_validate(target)
    numeric = [d.drug_id for d in target.drugs if d.mode == 'numeric']
    focal = target.profile.get('single_drug_analysis', {}).get('focal_drug_id')
    if not focal:
        varying = [did for did in numeric if len({p.doses[did] for p in target.points}) > 1]
        focal = varying[0] if len(varying) == 1 else numeric[0] if len(numeric) == 1 else None
    if focal not in numeric:
        raise ValueError('Choose one focal drug in Explore risks before simulating a trial.')
    for drug in target.drugs:
        if drug.drug_id != focal and len({p.doses[drug.drug_id] for p in target.points}) > 1:
            raise ValueError('Keep ' + drug.name + ' fixed across candidate doses. Only the focal drug may vary.')
    doses = [p.doses[focal] for p in target.points]
    if any(d is None for d in doses):
        raise ValueError('Enter a numeric dose for the focal drug at every level.')
    if len(set(doses)) != len(doses):
        raise ValueError('Each candidate dose must be different. Remove duplicate dose rows.')
    if len(doses) < 2:
        raise ValueError('Enter at least two different candidate doses for BOIN.')
    path = [p.level_id for p in sorted(target.points, key=lambda p: p.doses[focal])]
    validate_boin_target(target.model_copy(update={'path_level_ids': path}))
    return path
