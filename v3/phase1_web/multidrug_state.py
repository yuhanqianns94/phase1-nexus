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
