"""Dose-coordinate controls and consistent posterior slice displays."""
import copy
import numpy as np
import pandas as pd
import altair as alt
from phase1_prediction.multidrug.workflow import exploration_target
from phase1_prediction.multidrug.contracts import TargetRegimen, fingerprint
from phase1_web.dose_reference import reference_control


def _companion_state(drug, draft):
    """An unfinished plan is different from confirmed use at an unknown dose."""
    did = drug['drug_id']
    points = draft.get('points', [])
    states = draft.get('profile', {}).get('planning', {}).get('exposure_states', {})
    if any(states.get(p['level_id'], {}).get(did) in {'Not entered', 'Undecided', 'Not decided'} for p in points):
        return 'Not decided'
    if points and all(did in p['doses'] and p['doses'][did] == 0 for p in points):
        return 'Not used'
    if points and drug.get('mode') == 'background':
        if all(did in p['doses'] and p['doses'][did] is None for p in points):
            return 'Used, dose unknown'
        return None
    presence = drug.get('profile', {}).get('presence')
    if not points and presence == 'absent':
        return 'Not used'
    if not points and presence == 'mixed':
        return None
    if drug.get('mode') == 'background':
        return 'Used, dose unknown' if presence == 'present' else 'Not decided'
    return 'Numeric dose'


def _actual_values(drug, draft):
    """Only prefill physical doses from actual, explicitly specified inputs."""
    if drug.get('unit') == 'relative dose' or draft.get('profile', {}).get('planning', {}).get('exploration', {}).get('relative'):
        return []
    did = drug['drug_id']
    states = draft.get('profile', {}).get('planning', {}).get('exposure_states', {})
    return [float(p['doses'][did]) for p in draft.get('points', [])
            if states.get(p['level_id'], {}).get(did) not in {'Not entered', 'Undecided', 'Not decided'}
            and p['doses'].get(did) is not None and p['doses'][did] > 0]


def _fixed_coordinate(drug, draft, *, relative):
    values = _actual_values(drug, draft)
    if values and all(v == values[0] for v in values):
        return values[0] / (drug.get('reference_dose') or 1.) if relative else values[0]
    return 1. if relative else None


def dose_mapping_controls(st, data, *, key, relative=False, references=False):
    """Edit computational assumptions on an exploration copy, never the plan."""
    for drug in data['drugs']:
        drug['allow_background_anchor'] = False
        st.session_state.pop(key + '_' + drug['drug_id'] + '_anchor', None)
        if relative:
            drug['transport_mode'] = 'relative_dose_analogy'
    if relative:
        return
    with st.container(border=True):
        st.markdown('**Historical dose mapping**')
        st.caption('These settings control how target doses map to historical doses. '
                   'They are analysis assumptions, not clinical facts to extract.')
        modes={'relative_dose_analogy':'Relative-dose analogy',
               'same_entity_unit_conversion':'Same-drug unit conversion'}
        for drug in data['drugs']:
            sub=key+'_'+drug['drug_id']
            st.markdown('**'+drug['name']+'**')
            options=list(modes)
            drug['transport_mode']=st.selectbox('Mapping assumption',options,
                index=options.index(drug.get('transport_mode','relative_dose_analogy')),
                format_func=modes.get,key=sub+'_transport')
            if references and drug.get('mode')=='numeric':
                reference_control(st, drug, _actual_values(drug, data), key=sub)


def _focal_target(data, focal_id):
    """Record the modeled drug and validate one fixed combination context."""
    data = copy.deepcopy(data)
    drugs = data.get('drugs', [])
    focal = next((drug for drug in drugs if drug['drug_id'] == focal_id), None)
    if not focal or focal.get('mode') != 'numeric':
        raise ValueError('Choose one focal drug with a numerical dose axis.')
    for drug in drugs:
        did = drug['drug_id']
        if did == focal_id:
            continue
        if len({point['doses'].get(did) for point in data.get('points', [])}) > 1:
            raise ValueError('Keep the dose and use state of ' + drug['name'] + ' fixed across all candidate regimens. Only the focal drug may vary.')
    data.setdefault('profile', {}).setdefault('single_drug_analysis', {})['focal_drug_id'] = focal_id
    return TargetRegimen.model_validate(data)


def coordinate_editor(st, draft, *, key='explore'):
    """Choose the scientific question first, then its dose coordinates.

    Relative coordinates are exploratory assumptions. Missing actual values and
    undecided companion use never become administered doses or absence.
    """
    drugs = draft.get('drugs', [])
    if not drugs:
        st.info('Add a target drug to your trial plan before exploring risk.')
        return None
    if draft.get('profile', {}).get('single_drug_analysis', {}).get('combination_present', False) is None:
        st.info('Choose whether the drug will be used in combination in your trial plan.')
        return None
    saved_keys = [key + suffix for suffix in ('_goal', '_mode', '_axes')]
    saved_keys += [key + '_' + d['drug_id'] + '_' + suffix for d in drugs
                  for suffix in ('lo', 'hi', 'n', 'fixed', 'actual_lo', 'actual_hi', 'actual_n',
                                 'actual_fixed', 'unit', 'ref', 'custom_ref', 'state', 'transport', 'anchor')]
    for saved_key in saved_keys:
        if saved_key in st.session_state:
            st.session_state[saved_key] = st.session_state[saved_key]

    names = {d['drug_id']: d['name'] for d in drugs}
    # Confirmed dose-unknown exposures can be fixed, but cannot supply a numeric
    # axis. New drugs with no dosing decision can be explored relatively.
    unknown_ids = {d['drug_id'] for d in drugs if d.get('mode') == 'background'
                   and _companion_state(d, draft) in {'Used, dose unknown', 'Not used', None}}
    axis_ids = [did for did in names if did not in unknown_ids]
    goals = ['Explore a dose range', 'Compare candidate doses']
    legacy_goals = {'Vary one drug': goals[0], 'Vary one drug with fixed companions': goals[0],
                    'Compare planned regimens': goals[1]}
    if st.session_state.get(key + '_goal') in legacy_goals:
        st.session_state[key + '_goal'] = legacy_goals[st.session_state[key + '_goal']]
    if st.session_state.get(key + '_goal') not in goals:
        st.session_state[key + '_goal'] = goals[0]
    goal = st.radio('What would you like to explore?', goals, key=key + '_goal')
    context_note = st.empty()

    def describe_context(target):
        from phase1_web.model_formulas import target_combination
        if target_combination(target):
            context_note.caption('Combination treatment is included as a fixed background. Only the target drug dose varies.')
        return target
    if not axis_ids:
        st.info('Choose a focal drug with a numerical dose axis in the trial plan.')
        return None
    previous_axes = st.session_state.get(key + '_axes', [])
    saved_focal = draft.get('profile', {}).get('single_drug_analysis', {}).get('focal_drug_id')
    valid_axes = [did for did in previous_axes if did in axis_ids][:1]
    if not valid_axes:
        valid_axes = [saved_focal] if saved_focal in axis_ids else axis_ids[:1]
    if key + '_axes' not in st.session_state or previous_axes != valid_axes:
        st.session_state[key + '_axes'] = valid_axes
    if len(drugs) == 1:
        axes = axis_ids[:1]
        st.caption('Target drug: ' + names[axes[0]])
    else:
        axes = st.multiselect('Focal drug', axis_ids, format_func=lambda did: names[did],
                              max_selections=1, key=key + '_axes')
    if len(axes) != 1:
        st.info('Choose exactly one focal drug. Other drugs remain fixed background.')
        return None
    if goal == 'Compare candidate doses':
        if not draft.get('points'):
            st.info('Add candidate doses to your trial plan, or choose Explore a dose range.')
            return None
        data=copy.deepcopy(draft)
        dose_mapping_controls(st,data,key=key,references=True)
        try:
            return describe_context(_focal_target(data, axes[0]))
        except ValueError as error:
            st.info('Check the candidate doses: ' + str(error))
            return None

    modes = ['Relative-dose exploration', 'Actual-dose exploration']
    if st.session_state.get(key + '_mode') not in modes:
        actual_ready = all(d.get('unit') not in (None, '', 'relative dose')
                           for d in drugs if d['drug_id'] in axis_ids)
        st.session_state[key + '_mode'] = modes[1] if draft.get('points') and actual_ready else modes[0]
    mode = st.radio('Have you chosen actual doses?', modes,
                    format_func=lambda value: 'Not yet — explore relative doses' if value == modes[0] else 'Yes — enter actual doses',
                    key=key + '_mode')
    relative = mode == modes[0]
    if relative:
        st.caption('Compare lower and higher relative doses before choosing mg or other units. A value of 1 is the model reference, not a recommended dose. Actual doses can be entered later for BOIN.')
    else:
        st.caption('Enter the dose range to examine. These are exploration settings; the saved trial plan stays unchanged.')
    axis_count = 1

    data = copy.deepcopy(draft)
    coordinates = {}
    complete = True
    for d in data['drugs']:
        did = d['drug_id']
        sub = key + '_' + did
        coordinate_key = sub if relative else sub + '_actual'
        with st.container(border=True):
            st.markdown('**' + d['name'] + (' · dose range' if did in axes else ' · fixed companion') + '**')
            default_state = _companion_state(d, draft)
            states = ['Numeric dose', 'Used, dose unknown', 'Not used', 'Not decided']
            if default_state is None:
                states.insert(0, 'Choose companion use state')
            labels = {'Numeric dose': 'Set a fixed relative dose' if relative else 'Set a fixed actual dose',
                      'Used, dose unknown': 'Will be used, but its dose is unavailable',
                      'Not used': 'Will not be used', 'Not decided': 'Use or dose not decided',
                      'Choose companion use state': 'Choose how this companion will be used'}
            if did in axes:
                state = 'Numeric dose'
            else:
                if st.session_state.get(sub + '_state') not in states:
                    st.session_state[sub + '_state'] = default_state or states[0]
                state = st.selectbox('How is ' + d['name'] + ' used?', states,
                                     format_func=lambda value: labels[value], key=sub + '_state')
            if state in {'Not decided', 'Choose companion use state'}:
                st.info('Choose a fixed dose or an explicit use state for this curve. An undecided plan does not mean the drug is absent or already used at an unknown dose.')
                complete = False
                continue
            if state != 'Numeric dose':
                coordinates[did] = [None if state == 'Used, dose unknown' else 0.]
                continue

            if relative:
                st.caption('Relative dose: 0.5 means half the reference; 2 means twice the reference.')
            else:
                saved_drug = copy.deepcopy(d)
                saved_unit = d.get('unit') if d.get('unit') != 'relative dose' else None
                d['unit'] = st.text_input('Dose unit', saved_unit or '', key=sub + '_unit').strip()
                if not d['unit']:
                    complete = False
            units = 'relative dose' if relative else d.get('unit') or 'unit not specified'
            values = _actual_values(d, draft)
            if did in axes:
                low_default = .25 if relative else min(values) if values else None
                high_default = 2. if relative else max(values) if len(set(values)) > 1 else None
                lo = st.number_input('From (' + units + ')', min_value=0., value=low_default, key=coordinate_key + '_lo')
                hi = st.number_input('To (' + units + ')', min_value=.00000001, value=high_default, key=coordinate_key + '_hi')
                n = st.number_input('Points to calculate', min_value=2, max_value=40,
                                    value=20 if axis_count == 1 else 10, key=coordinate_key + '_n')
                if lo is None or hi is None:
                    st.info('Enter both ends of the dose range.')
                    complete = False
                elif hi <= lo:
                    st.info('The upper dose must be greater than the lower dose.')
                    complete = False
                else:
                    coordinates[did] = np.linspace(lo, hi, int(n)).tolist()
            else:
                fixed = st.number_input('Fixed dose (' + units + ')', min_value=.00000001,
                                        value=_fixed_coordinate(d, draft, relative=relative), key=coordinate_key + '_fixed')
                if fixed is None:
                    st.info('Enter the fixed companion dose for this curve.')
                    complete = False
                else:
                    coordinates[did] = [fixed]
            if not relative:
                # Use only the endpoints, never the density of the plotting grid.
                range_values = ([lo, hi] if lo is not None and hi is not None and 0 < lo < hi else []) if did in axes else ([fixed] if fixed else [])
                reference_control(st, d, values or range_values, key=sub, saved_drug=saved_drug,
                                  basis='geometric mean of candidate doses' if values else 'geometric mean of the initial dose range' if did in axes else 'fixed dose')
                if not d.get('reference_dose'):
                    complete = False
    dose_mapping_controls(st,data,key=key,relative=relative)
    if not complete:
        return None
    try:
        explored = exploration_target(data, coordinates, axes, relative=relative)
        return describe_context(_focal_target(explored.model_dump(), axes[0]))
    except ValueError as error:
        st.info(str(error))
        return None


def render_chart(st,method,*,key='curve'):
    target=method['target'];summary=method['summary'];drugs=target['drugs']
    rows=[{**r,**{'dose_'+k:v for k,v in r['doses'].items()}} for r in summary]
    numeric=[d for d in drugs if d['mode']=='numeric' and len({r['dose_'+d['drug_id']] for r in rows if r['dose_'+d['drug_id']] is not None})>1]
    if not numeric:
        from phase1_web.result_exports import summary_rows
        st.dataframe(summary_rows({'target': target, 'methods': {'current': method}}), hide_index=True, use_container_width=True)
        return
    ids=[d['drug_id'] for d in numeric];names={d['drug_id']:d['name'] for d in drugs}
    preferred=target.get('profile',{}).get('planning',{}).get('exploration',{}).get('axes',[])
    ids=list(dict.fromkeys([did for did in preferred if did in ids]+ids))
    if st.session_state.get(key+'_x') not in ids:st.session_state[key+'_x']=ids[0]
    x=st.selectbox('Horizontal dose axis',ids,format_func=lambda k:names[k],key=key+'_x')
    views=['Conditional curve','Two-drug heatmap','Three-dimensional surface'] if len(ids)>1 else ['Conditional curve']
    if st.session_state.get(key+'_view') not in views:
        st.session_state[key+'_view']='Two-drug heatmap' if len(preferred)==2 and len(ids)>1 else views[0]
    view=st.radio('Display',views,horizontal=True,key=key+'_view')
    y=None
    if view!='Conditional curve':
        y_ids=[did for did in ids if did!=x]
        if st.session_state.get(key+'_y') not in y_ids:st.session_state[key+'_y']=y_ids[0]
        y=st.selectbox('Second dose axis',y_ids,format_func=lambda k:names[k],key=key+'_y')
    fixed={}
    for d in drugs:
        did=d['drug_id']
        if did in (x,y):continue
        vals=list(dict.fromkeys(r['dose_'+did] for r in rows))
        fixed[did]=st.selectbox('Fix '+d['name'],vals,format_func=lambda v:'Used, dose unknown' if v is None else 'Not used' if v==0 else str(v),key=key+'_fix_'+did)
    frame=pd.DataFrame([r for r in rows if all(r['dose_'+did]==v for did,v in fixed.items())])
    if frame.empty:st.info('This fixed combination is not in the evaluated grid. Use the explorer to evaluate it.');return
    dx=next(d for d in drugs if d['drug_id']==x)
    title=names[x]+' ('+str(dx.get('unit') or 'relative dose')+')'
    chart=alt.Chart(frame).encode(x=alt.X('dose_'+x+':Q',title=title),tooltip=['level_id','posterior_median','low_95','high_95'])
    if y:
        chart=alt.Chart(frame).mark_rect().encode(x=alt.X('dose_'+x+':O',title=title),y=alt.Y('dose_'+y+':O',title=names[y]),color=alt.Color('posterior_median:Q',scale=alt.Scale(domain=[0,1]),title='DLT probability'),tooltip=['posterior_median','low_95','high_95'])
    else:
        chart=chart.mark_area(opacity=.2).encode(y=alt.Y('low_95:Q',scale=alt.Scale(domain=[0,1]),title='DLT probability'),y2='high_95:Q')+chart.mark_line(point=True).encode(y='posterior_median:Q')
    if view=='Three-dimensional surface':
        from matplotlib.figure import Figure
        grid=frame.pivot_table(index='dose_'+y,columns='dose_'+x,values='posterior_median')
        xx,yy=np.meshgrid(grid.columns.to_numpy(),grid.index.to_numpy())
        fig=Figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
        ax.plot_surface(xx,yy,grid.to_numpy(),cmap='viridis',vmin=0,vmax=1)
        ax.set(xlabel=title,ylabel=names[y],zlabel='DLT probability',zlim=(0,1))
        st.pyplot(fig)
    else:st.altair_chart(chart,use_container_width=True)
    st.caption('The shaded interval is a 95% posterior credible interval. Dose analogies and predictions outside observed source dose combinations are model-dependent; fixed or missing source doses do not establish a measured surface.')


def boin_candidate_editor(st,result,*,rebase=False):
    """Explicit actual regimens, independently of the exploration grid."""
    data=copy.deepcopy(result['target'])
    exploration=data.get('profile',{}).get('planning',{}).get('exploration',{})
    relative=exploration.get('relative',False)
    prefix='boin_candidates_'+result['run_id']
    reuse=not relative
    if reuse and exploration.get('auto_grid'):
        reuse=st.radio('Candidate regimen source',['Use evaluated actual-dose regimens','Enter new actual-dose regimens'],
            horizontal=True,key=prefix+'_source')=='Use evaluated actual-dose regimens'
        if reuse:
            st.caption(f"Use the {len(data['points'])} evaluated doses, or edit the rows below to change your candidate doses.")
    st.caption('Enter one candidate dose per row. Dose-level IDs are generated automatically; BOIN follows increasing target-drug doses.')
    rows=[dict(p['doses'],Regimen=p['level_id']) for p in data['points']] if reuse else []
    numeric=[d for d in data['drugs'] if d['mode']=='numeric']
    saved_drugs={d['drug_id']:copy.deepcopy(d) for d in data['drugs']}
    edit_units=relative
    if numeric and not relative:
        edit_units=st.checkbox('Edit dose units',key=prefix+'_edit_units')
    elif numeric:
        st.caption('Enter the actual dose unit and candidate doses. The reference dose is calculated automatically.')
    # Preserve existing IDs internally so an unchanged prediction can be reused.
    # New rows receive IDs below; users only edit doses and companion use states.
    columns={'Regimen':None}
    for d in data['drugs']:
        did=d['drug_id']
        if d['mode']=='numeric':
            if edit_units:
                d['unit']=st.text_input(d['name']+' actual dose unit',value='' if relative else d['unit'],key=prefix+did+'_unit')
            columns[did]=st.column_config.NumberColumn(d['name']+' ('+d['unit']+')',min_value=0.,required=True)
        else:
            columns[did]=st.column_config.SelectboxColumn(d['name'],options=['Used, dose unknown','Not used'],required=True)
            for r in rows:r[did]='Not used' if r.get(did)==0 else 'Used, dose unknown'
    editor_key=prefix+('_new' if not reuse else '')
    drafts=st.session_state.setdefault('multidrug_boin_candidate_drafts',{})
    table=drafts.setdefault(editor_key,{'base':copy.deepcopy(rows),'edited':copy.deepcopy(rows)})
    schema=fingerprint([(d['drug_id'],d['mode'],d.get('unit')) for d in data['drugs']])
    if rebase or editor_key not in st.session_state or table.get('schema')!=schema:
        # On return, start from the last edited table with no old widget deltas.
        # During ordinary reruns keep the same base, so added rows apply once.
        table['base']=copy.deepcopy(table['edited'])
        st.session_state.pop(editor_key,None)
    table['schema']=schema
    edited=st.data_editor(pd.DataFrame(table['base'],columns=['Regimen',*[d['drug_id'] for d in data['drugs']]]),num_rows='dynamic',column_config=columns,hide_index=True,key=editor_key)
    table['edited']=edited.to_dict('records')
    used_ids={str(row['Regimen']) for row in table['edited']
              if not pd.isna(row.get('Regimen')) and str(row['Regimen']).strip()}
    next_id=1
    for row in table['edited']:
        if pd.isna(row.get('Regimen')) or not str(row.get('Regimen', '')).strip():
            while f'L{next_id}' in used_ids:
                next_id+=1
            row['Regimen']=f'L{next_id}'
            used_ids.add(row['Regimen'])
    if edited.empty:
        st.info('Enter actual candidate regimens to evaluate for BOIN.');return None
    data['points']=[];data['path_level_ids']=[]
    for row in table['edited']:
        values={}
        for d in data['drugs']:
            v=row[d['drug_id']]
            if pd.isna(v):raise ValueError('Complete every candidate dose or use state.')
            values[d['drug_id']]=(0. if v=='Not used' else None) if d['mode']=='background' else float(v)
        data['points'].append({'level_id':str(row['Regimen']),'doses':values})
    for d in numeric:
        if not d.get('unit') or d['unit'] == 'relative dose':
            st.info('Enter an actual dose unit for ' + d['name'] + '.');return None
        reference_control(st, d, [p['doses'][d['drug_id']] for p in data['points']],
                          key=prefix+d['drug_id'], saved_drug=saved_drugs[d['drug_id']],
                          basis='geometric mean of candidate doses')
        if not d.get('reference_dose'):
            return None
    planning=data.setdefault('profile',{}).setdefault('planning',{})
    planning.pop('exploration',None);planning['exposure_states']={}
    focal = data.get('profile', {}).get('single_drug_analysis', {}).get('focal_drug_id')
    return _focal_target(data, focal) if focal else TargetRegimen.model_validate(data)
