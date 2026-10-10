"""Native, editable target drug/scenario contract shared with extraction profiles."""
import hashlib
import copy
import json
import math
import re
from phase1_prediction.profiles import query_profile
from phase1_prediction.profile_similarity import condition, unique

COLUMNS=['scenario','drug','kind','role','presence','doses','unit','phase','route','form','frequency','administration_days','cycle_days','duration','infusion_duration','infusion_rate','start','end','aliases','classes','targets','mechanisms','adjustment_conditions']
ROLES=['escalated','fixed_backbone','combination_partner','conditioning','supportive','concomitant','radiation','maintenance','other','unclear']


from phase1_prediction.target_configurations import parse_rows


def render(st,drug,dose_unit):
    with st.expander('Drug configurations and additional comparison fields',expanded=True):
        st.caption('One row per drug and dosing phase. Drug classes use the same rows. Present with an empty dose means used, dose unknown; absent means explicitly not used. Give each allowed fixed-background combination its own scenario name.')
        seed={k:'' for k in COLUMNS};seed.update(scenario='Main',drug=drug,kind='drug',role='escalated',presence='present',unit=dose_unit,phase='treatment')
        try:rows=json.loads(st.session_state.get('native_target_rows_json','[]'))
        except (ValueError,TypeError):rows=[]
        if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows):rows=[]
        if not rows:rows=[seed]
        old_axis=st.session_state.get('_native_axis_name','')
        for r in rows:
            if r.get('role')=='escalated' and r.get('drug','') in ('',old_axis):
                r['drug']=drug
                if not r.get('unit'):r['unit']=dose_unit
        if old_axis!=drug and 'native_target_rows_editor' in st.session_state:
            del st.session_state['native_target_rows_editor']
        st.session_state['_native_axis_name']=drug
        def editor_base(key,initial):
            cache=st.session_state.setdefault('_native_editor_bases',{})
            if key not in st.session_state or key not in cache:cache[key]=copy.deepcopy(initial)
            return cache[key]
        previous_by_identity={(r.get('scenario'),r.get('drug'),r.get('phase')):r for r in rows}

        drug_tab,schedule_tab,mechanism_tab=st.tabs(['Drugs and doses','Administration by drug','Mechanism and aliases'])
        with drug_tab:
            edited=st.data_editor(editor_base('native_target_rows_editor',rows),num_rows='dynamic',hide_index=True,key='native_target_rows_editor',
                column_order=COLUMNS[:8],
                column_config={'kind':st.column_config.SelectboxColumn('Entity type',options=['drug','drug_class','non_drug','unclear']),
                               'role':st.column_config.SelectboxColumn('Role',options=ROLES),
                               'presence':st.column_config.SelectboxColumn('Use',options=['present','absent','unknown','mixed'])})
        identity=['scenario','drug','phase']
        for r in edited:
            previous=previous_by_identity.get(tuple(r.get(k) for k in identity),{})
            for k in COLUMNS[8:]:
                if k in previous:r[k]=previous[k]
        identity_key=hashlib.sha256(json.dumps([[r.get(k) for k in identity] for r in edited],sort_keys=True).encode()).hexdigest()[:12]
        schedule_key='native_schedule_editor_'+identity_key
        mechanism_key='native_mechanism_editor_'+identity_key
        with schedule_tab:
            schedule=st.data_editor(editor_base(schedule_key,[{k:r.get(k,'') for k in [*identity,*COLUMNS[8:18]]} for r in edited]),
                disabled=identity,hide_index=True,key=schedule_key)
        with mechanism_tab:
            mechanism=st.data_editor(editor_base(mechanism_key,[{k:r.get(k,'') for k in [*identity,*COLUMNS[18:]]} for r in edited]),
                disabled=identity,hide_index=True,key=mechanism_key)
        for r,s,m in zip(edited,schedule,mechanism):
            r.update({k:v for k,v in s.items() if k not in identity})
            r.update({k:v for k,v in m.items() if k not in identity})
        st.session_state['native_target_rows_json']=json.dumps(edited,ensure_ascii=False)
        scenarios=sorted({str(r.get('scenario') or 'Main') for r in edited if r.get('drug')}) or ['Main']
        if st.session_state.get('native_target_scenario') not in scenarios:st.session_state['native_target_scenario']=scenarios[0]
        selected=st.selectbox('Fixed-background scenario for this curve',scenarios,key='native_target_scenario')
        st.caption('The curve varies the primary drug on its stated dose grid. Partner-dose effects are not estimated by the current one-dimensional curve methods.')
        cols=st.columns(2)
        with cols[0]:
            st.text_input('Study purpose',key='native_purpose')
            st.text_input('Disease modifiers',key='native_modifiers')
            st.text_input('Study phase',key='native_study_phase')
            st.selectbox('Study-part role',['unclear','escalation','expansion','lead_in','maintenance','confirmation','other'],key='native_study_role')
            st.text_area('DLT exceptions (one complete condition per line)',key='native_dlt_exceptions')
            st.text_input('Toxicity grading system and version',key='native_grading')
            st.text_input('DLT definition version',key='native_dlt_version')
        with cols[1]:
            st.selectbox('Window type',['formal_dlt','safety_monitoring','unclear'],key='native_window_kind')
            st.text_input('Window start',key='native_window_start')
            st.text_input('Window end',key='native_window_end')
            st.number_input('Actual cycle length for DLT window (days)',min_value=0.,step=1.,key='native_window_cycle_days',help='Leave 0 if unknown. No default cycle length is assumed.')
            st.text_area('Dose reduction / interruption conditions',key='native_adjustments')
    return edited,selected


def apply_extras(query,rows,selected,state):
    query={**query,'_target_phase':state.get('native_study_phase'),'_target_role':state.get('native_study_role'),
           'dlt_window_kind':state.get('native_window_kind','unclear'),'dlt_window_start':state.get('native_window_start'),'dlt_window_end':state.get('native_window_end')}
    window=query.get('DLT_EVA',{}).get('value',{})
    cycle=state.get('native_window_cycle_days') or None;cycles=window.get('dlt_eval_cycle_number');duration=window.get('dlt_eval_duration_days')
    if cycle and cycles:
        derived=float(cycle)*float(cycles)
        if duration and not math.isclose(duration,derived):raise ValueError('DLT days conflict with the specified cycles × actual cycle length.')
        window['dlt_eval_duration_days']=derived;window['normalization_status']='user_cycle_length_derived'
    query=parse_rows(rows,query,selected);p=query.setdefault('_comparison_profile',query_profile(query))
    p['population'].update(purpose=state.get('native_purpose'),modifiers=[v.strip() for v in str(state.get('native_modifiers') or '').split(';') if v.strip()])
    from phase1_prediction.profiles import grading_version
    p['dlt']['grading_version']=grading_version(state.get('native_grading'))
    p['dlt'].update(grading=state.get('native_grading'),version=state.get('native_dlt_version'),exceptions=[condition(v) for v in str(state.get('native_dlt_exceptions') or '').splitlines() if v.strip()])
    for c in p['configurations']:c['adjustments']=[condition(v) for v in str(state.get('native_adjustments') or '').splitlines() if v.strip()]
    return query
