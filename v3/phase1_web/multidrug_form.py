"""Human-readable full-regimen target editor; untouched structured facts survive."""
import copy
import json

import pandas as pd
import streamlit as st
from phase1_prediction.multidrug.contracts import TargetRegimen
from phase1_prediction.profile_similarity import condition, DEFAULT_MIN_DRUG_SIMILARITY
from phase1_prediction.profiles import grading_version
from phase1_prediction.evidence_relevance import irrelevant_eligibility_indices
from phase1_extraction.controlled_vocabulary import choices, labels, canonical_value
from phase1_web.plan_navigation import show_steps, show_step_buttons


def _list(value):
    return [x.strip() for x in str(value or '').split(';') if x.strip()]


def _text(value):
    if value is None:
        return ''
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _raw(values):
    return '\n'.join(v.get('raw', v.get('raw_text', '')) if isinstance(v, dict) else str(v) for v in values)


def merge_conditions(original, text):
    """A no-edit round trip must not reparse away timing/logic/unit metadata."""
    if text == _raw(original):
        return copy.deepcopy(original)
    by_raw = {v.get('raw', v.get('raw_text', '')): v for v in original if isinstance(v, dict)}
    return [copy.deepcopy(by_raw[x]) if x in by_raw else condition(x)
            for x in text.splitlines() if x.strip()]


def _set_text(record, key, value):
    if value != _text(record.get(key)):
        record[key] = value or None


PHASE_FIELDS = {'phase':'Stage name', 'route':'Route', 'form':'Dosage form', 'frequency':'Frequency',
    'administration_days':'Administration days', 'cycle':'Cycle length', 'duration':'Treatment duration',
    'infusion_duration':'Infusion duration', 'infusion_rate':'Infusion rate', 'start':'Stage start', 'end':'Stage end'}


def phase_rows(phases):
    return [{'__row_id':str(i), **{label:_text(p.get(key)) for key,label in PHASE_FIELDS.items()}}
            for i,p in enumerate(phases)]


def merge_phases(original, rows):
    result=[]
    used=set()
    for row in rows:
        identity=row.get('__row_id')
        if identity is not None and str(identity) in used:
            raise ValueError('Duplicate administration stage ID')
        if identity is not None:
            used.add(str(identity))
        old=original[int(identity)] if identity is not None and str(identity).isdigit() and int(identity)<len(original) else {}
        if not old and not any(row.get(label) for label in PHASE_FIELDS.values()):
            continue
        item=copy.deepcopy(old)
        for key,label in PHASE_FIELDS.items():
            value=_text(row.get(label))
            _set_text(item,key,value)
        result.append(item)
    return result


def prepare_drug_rows(rows, original):
    """Existing IDs are immutable; new table rows receive stable free IDs."""
    known_ids={d['drug_id'] for d in original}
    used=set(known_ids)
    result=[]
    for row in rows:
        row=dict(row)
        if not row.get('ID') and not row.get('Drug name'):continue
        original_id=row.get('__original_id')
        if original_id:
            if original_id not in known_ids or row.get('ID')!=original_id:
                raise ValueError('Internal drug IDs cannot be changed. Edit the display name; doses and stages are retained.')
        elif not row.get('ID'):
            n=1
            while f'drug_{n}' in used:n+=1
            row['ID']=f'drug_{n}'
        used.add(row['ID'])
        result.append(row)
    return result


# All controls below edit planned facts. Internal IDs never serve as labels.
import math
import re
import uuid
from phase1_extraction.clinical_protocol import normalized_phase, planned_issues, PROTOCOL_VERSION

USE_STATES=['Not entered','Undecided','Use: dose specified','Use: dose unknown','Do not use']


def _input_key(key):
    key='plan_'+key
    st.session_state.setdefault('multidrug_plan_input_keys',set()).add(key)
    return key


def _mark_edited(key):
    st.session_state[key+'_edited']=True


def choice(label, options, value, key, *, help=None):
    # Imported prose must never silently expand a controlled vocabulary.
    options=list(dict.fromkeys(options))
    if value not in options:
        value=next((option for option in ('Other','other','Undecided','Not specified') if option in options),options[0])
    widget_key=_input_key(key)
    if widget_key in st.session_state and st.session_state[widget_key] not in options:
        st.session_state[widget_key]=value
    display={'escalated':'Dose exploration','fixed_backbone':'Fixed backbone','combination_partner':'Combination partner','supportive':'Supportive treatment','conditioning':'Conditioning','maintenance':'Maintenance','concomitant':'Concomitant medication','lead_in':'Lead-in','escalation':'Dose escalation','expansion':'Dose expansion'}
    display.update(per_administration='Per administration',daily_total='Daily total',infusion_rate='Infusion rate',other='Other')
    return st.selectbox(label,options,index=options.index(value) if value in options else 0,key=widget_key,format_func=lambda x:display.get(x,x),help=help,on_change=_mark_edited,args=(widget_key,))


def tags(label, values, options, key):
    """Open scientific identifiers are prose, never a misleading closed list."""
    if isinstance(values,str):values=[values]
    old='\n'.join(str(value) for value in values)
    widget_key=_input_key(key)
    if isinstance(st.session_state.get(widget_key),(list,tuple)):
        st.session_state[widget_key]='\n'.join(str(value) for value in st.session_state[widget_key])
    text=st.text_area(label,old,key=widget_key,help='Enter one item per line. Names and scientific descriptions are open text.')
    return values if text==old else [line.strip() for line in text.splitlines() if line.strip()]


def vocabulary_choice(label, vocabulary, value, key, *, missing='Undecided', code=None, other_label=None, help=None):
    """A fixed menu with optional separate prose; return value, code, edited."""
    names=labels(vocabulary)
    codes=[item for item in choices(vocabulary) if item!='unclear']
    known=canonical_value(vocabulary,code) if code is not None else None
    if known is None:known=canonical_value(vocabulary,value)
    if known=='unclear':known=None
    missing_value=value in (None,'') or str(value).strip().lower() in {'unclear','unknown','undecided','not specified','not reported'}
    default=known or ('other' if not missing_value and 'other' in codes else None)
    displayed={item:('Other' if item=='other' else item if vocabulary in {'route','dosage_form','dose_basis','treatment_role','study_part_role'} else names.get(item,item)) for item in codes}
    options=[missing,*displayed.values()]
    selected=choice(label,options,displayed.get(default,missing),key,help=help)
    selected_code=next((item for item,text in displayed.items() if text==selected),None)
    old_other=_text(value) if default=='other' and str(value).lower()!='other' else ''
    other=old_other
    if selected_code=='other':
        other=st.text_input(other_label or 'Describe other '+label.lower(),old_other,key=_input_key(key+'_custom'))
    elif value not in (None,'') and known is None:
        st.caption('Unclassified source wording retained: '+_text(value))
    changed=selected_code!=default or (selected_code=='other' and other!=old_other) or st.session_state.get('plan_'+key+'_edited',False)
    if not changed:return value,known or default,False
    result=(other or 'other') if selected_code=='other' else displayed.get(selected_code)
    return result,selected_code,True


def vocabulary_field(record, field, vocabulary, label, key, *, code_field=None, missing='Undecided', help=None):
    code_field=code_field or field+'_code'
    value,code,changed=vocabulary_choice(label,vocabulary,record.get(field),key,
        missing=missing,code=record.get(code_field),help=help)
    if changed:
        record[field]=value
        if code is None:record.pop(code_field,None)
        else:record[code_field]=code
    return value


def vocabulary_tags(label, values, vocabulary, key):
    names=labels(vocabulary)
    values=[values] if isinstance(values,str) else values
    known=[canonical_value(vocabulary,value) for value in values]
    default=list(dict.fromkeys(names[code] for code in known if code))
    unknown=[str(value) for value,code in zip(values,known) if code is None]
    if unknown and 'other' in names and names['other'] not in default:default.append(names['other'])
    widget_key=_input_key(key)
    if widget_key in st.session_state and any(value not in names.values() for value in st.session_state[widget_key]):
        st.session_state[widget_key]=default
    selected=st.multiselect(label,list(names.values()),default=default,key=widget_key)
    old_other='\n'.join(unknown)
    other=old_other
    if 'other' in names and names['other'] in selected:
        other=st.text_area('Other '+label.lower(),old_other,key=_input_key(key+'_other'),help='Describe characteristics that are not in the fixed list, one per line.')
    elif unknown:st.caption('Unclassified source wording retained: '+ '; '.join(unknown))
    if selected==default and other==old_other:return values
    return [text for text in selected if text!=names.get('other')]+([line.strip() for line in other.splitlines() if line.strip()] if names.get('other') in selected else [])


def optional_number(label,value,key,min_value=0.):
    return st.number_input(label,min_value=min_value,value=float(value) if value is not None else None,key=_input_key(key))


def text_field(label, record, field, key, *, area=False):
    old=_text(record.get(field)); value=(st.text_area if area else st.text_input)(label,old,key=_input_key(key))
    if value!=old:record[field]=value or None
    return value


def _merge_rule_controls(rule, text, controls):
    """Parse edited prose once; only changed structure controls override it."""
    result=copy.deepcopy(rule)
    if text!=rule.get('raw',rule.get('raw_text','')):
        parsed=condition(text)
        for field in ('metric_code','comparator_code','unit_code'):
            result.pop(field,None)
            if parsed.get(field) is not None:result[field]=parsed[field]
        for field in ('raw','topic','operator','threshold','upper','unit','logic_tokens'):
            result[field]=parsed.get(field)
    for field,value in controls.items():
        if value!=rule.get(field):result[field]=value
    if result!=rule:
        result['clause_id']=rule.get('clause_id') or uuid.uuid4().hex
        result.pop('raw_text',None)
    return result


def _rule_statement(topic, operator, threshold, upper, unit, timing='', negated=None):
    """Generate wording only for rules created with the structured controls."""
    parts=[topic,operator if operator!='Not specified' else '',
           f'{threshold:g}' if threshold is not None else '',
           f'to {upper:g}' if operator=='between' and upper is not None else '',
           unit if unit!='Not specified' else '',timing]
    statement=' '.join(str(value) for value in parts if value not in ('',None))
    return f'NOT ({statement})' if negated else statement


def _select_rule(key, index):
    st.session_state[key]=index


def rules_editor(label, values, key, *, hidden_indices=()):
    """Stable rule identity; untouched source/JSON predicates remain intact."""
    result=copy.deepcopy(values)
    visible=[i for i in range(len(result)) if i not in hidden_indices]
    with st.container():
        st.markdown('#### '+label)
        st.caption('Use the fields for a simple requirement, or keep the full statement for exceptions and complex conditions.')
        if key=='elig_inclusion':st.caption('Add requirements beyond the disease, age and sex eligibility already entered above.')
        remove=None
        page_key='trial_plan_rule_'+key
        page=min(max(0,st.session_state.get(page_key,0)),max(0,len(visible)-1))
        st.session_state[page_key]=page
        if len(visible)>1:
            st.dataframe([{'Rule':n+1,'Requirement':result[i].get('raw',result[i].get('raw_text','')) if isinstance(result[i],dict) else str(result[i])} for n,i in enumerate(visible)],hide_index=True,height=min(220,36+35*len(visible)))
            previous,count,following=st.columns([1,2,1])
            previous.button('Previous rule',key='plan_'+key+'_previous_rule',disabled=page==0,
                            on_click=_select_rule,args=(page_key,page-1))
            count.caption(f'Editing requirement {page+1} of {len(visible)}')
            following.button('Next rule',key='plan_'+key+'_next_rule',disabled=page==len(visible)-1,
                             on_click=_select_rule,args=(page_key,page+1))
        for i in ([visible[page]] if visible else []):
            item=result[i]
            rule=item if isinstance(item,dict) else condition(item)
            raw=rule.get('raw',rule.get('raw_text',''))
            stem=f'{key}_{i}'
            st.markdown(f'**Requirement {page+1}**')
            with st.container(border=True):
                measure,comparison=st.columns([2,1])
                with measure:
                    topic,topic_code,topic_changed=vocabulary_choice('Clinical measure / event','clinical_metric',rule.get('topic'),stem+'_topic',missing='Not specified',code=rule.get('metric_code'),other_label='Specify clinical measure / event')
                    topic=topic or ''
                with comparison:
                    operator,operator_code,operator_changed=vocabulary_choice('Comparison','comparison',rule.get('operator'),stem+'_op',missing='Not specified',code=rule.get('comparator_code'))
                    operator=operator or 'Not specified'
                number,units=st.columns([1,1])
                with number:threshold=optional_number('Threshold',rule.get('threshold'),stem+'_threshold')
                with units:
                    unit,unit_code,unit_changed=vocabulary_choice('Threshold unit','condition_unit',rule.get('unit'),stem+'_unit',missing='Not specified',code=rule.get('unit_code'),other_label='Custom threshold unit')
                    unit=unit or 'Not specified'
                upper=optional_number('Upper threshold (range only)',rule.get('upper'),stem+'_upper') if operator=='between' else rule.get('upper')
                if threshold is not None:
                    interpreted=' '.join(str(value) for value in (topic,operator if operator!='Not specified' else '',threshold,('to '+str(upper)) if operator=='between' and upper is not None else '',unit if unit!='Not specified' else '') if value!='')
                    st.caption('Structured interpretation: '+interpreted)
                if operator=='between' and threshold is not None and upper is not None and upper<threshold:
                    st.warning('The upper threshold must be at least the lower threshold.')
            with st.container():
                value=st.text_area(f'Rule {page+1}',raw,key=_input_key(stem+'_raw'),help='Retain all exceptions, timing and conditions. Existing source wording is kept when you edit structured fields.')
                if raw and not rule.get('_generated_statement'):
                    st.caption('Source wording is retained. Check it if you change the structured interpretation.')
            with st.container():
                timing=st.text_input('Rule-specific timing',rule.get('timing') or '',key=_input_key(stem+'_timing'),help='For DLT rules, enter timing only when it differs from the assessment window below. Otherwise retain any source timing as given.')
                negation_labels={None:'Not specified',False:'As stated',True:'NOT (the condition is negated)'}
                negation=choice('Condition meaning',list(negation_labels.values()),negation_labels.get(rule.get('negated'),'Not specified'),stem+'_negated',help='Whether the condition itself is negated, for example no prior therapy. This is separate from choosing the inclusion or exclusion section.')
                negated=next(value for value,label in negation_labels.items() if label==negation)
                linkage=rule.get('logic') or 'Not specified'
                if rule.get('logic'):
                    with st.container():
                        st.caption('Relationship to preceding rule: '+str(rule['logic'])+'. Use the section AND / OR groups to combine new requirements.')
                        if st.checkbox('Edit imported relationship',key=_input_key(stem+'_edit_logic')):
                            linkage=choice('Relationship to preceding rule',['Not specified','AND','OR'],linkage,stem+'_logic')
            generated=bool(topic.strip()) and (not value.strip() or (rule.get('_generated_statement') and value==raw))
            if generated:value=_rule_statement(topic,operator,threshold,upper,unit,timing,negated)
            if (value,topic,operator,threshold,upper,unit,timing,linkage,negated)!=(raw,rule.get('topic') or '',rule.get('operator') or 'Not specified',rule.get('threshold'),rule.get('upper'),rule.get('unit') or 'Not specified',rule.get('timing') or '',rule.get('logic') or 'Not specified',rule.get('negated')):
                controls={
                      'topic':topic or None,'operator':None if operator=='Not specified' else operator,
                      'threshold':threshold,'upper':upper,'unit':None if unit=='Not specified' else unit,
                      'timing':timing or None,'logic':None if linkage=='Not specified' else linkage,'negated':negated}
                for edited,field,code in ((topic_changed,'metric_code',topic_code),(operator_changed,'comparator_code',operator_code),(unit_changed,'unit_code',unit_code)):
                    if edited:controls[field]=code
                rule=_merge_rule_controls(rule,value,controls)
                if generated:
                    rule.update(controls)
                    rule['_generated_statement']=True
                elif value!=raw:rule.pop('_generated_statement',None)
                result[i]=rule
            if key.endswith('_rules'):
                action,action_code,action_changed=vocabulary_choice('Dose-adjustment action','adjustment_action',rule.get('action_raw') if rule.get('action')=='other' else rule.get('action'),stem+'_action',missing='Not specified',code=rule.get('action'),other_label='Other dose-adjustment action')
                if action_changed:
                    rule=copy.deepcopy(rule)
                    rule['action']=action_code
                    if action_code=='other':rule['action_raw']=action
                    result[i]=rule
            if st.button(f'Remove rule {page+1}',key='plan_'+stem+'_remove'):remove=i
        if remove is not None:
            result.pop(remove)
            st.session_state[page_key]=min(page,max(0,len(visible)-2))
        if st.button('Add requirement',key='plan_'+key+'_add'):
            st.session_state[page_key]=len(visible)
            result.append(condition(''))
    return result


def logic_editor(section, fields, key, *, hidden_clause_ids=(), hidden_rules=()):
    """Nested AND/OR groups use stable rule IDs, never list-position links."""
    changed=False
    with st.container():
        st.markdown('#### Combine requirements (optional)')
        groups=section.get('logic_groups',[])
        rules=[c for field in fields for c in section.get(field,[]) if isinstance(c,dict)]
        hidden=set(hidden_clause_ids)
        # Hide wholly administrative groups too, retaining their IDs and source
        # metadata in the original section. Only AND projections are hidden.
        while True:
            new={g['group_id'] for g in groups if g.get('children') and set(g['children'])<=hidden}
            if new<=hidden:break
            hidden.update(new)
        visible_rules=[c for c in rules if c.get('clause_id') not in hidden and
                       not any(c is rule for rule in hidden_rules)]
        st.caption('Use groups to express how requirements combine. A single rule does not need a group.')
        if any(c.get('logic') for c in rules):st.caption('Imported per-rule relationships are retained under each rule. Keep new group relationships consistent with those source relationships.')
        if groups:
            for c in visible_rules:
                if not c.get('clause_id'):c['clause_id']=uuid.uuid4().hex;changed=True
            labels={c['clause_id']:c.get('raw','')[:100] or 'Untitled requirement' for c in visible_rules}
            labels.update({g['group_id']:f"Rule group {i+1}" for i,g in enumerate(groups) if g['group_id'] not in hidden})
            remove=None
            for i,g in enumerate(groups):
                if g['group_id'] in hidden:continue
                with st.container(border=True):
                    st.markdown(f'**Rule group {i+1}**')
                    retained=[child for child in g['children'] if child in hidden]
                    if retained:
                        op=g['operator']
                        st.caption('AND — source administrative requirements are retained separately. Add a nested group for clinical alternatives.')
                    else:op=choice('Relationship',['AND','OR'],g['operator'],key+f'_group_{i}_op')
                    options=[x for x in labels if x!=g['group_id']]
                    displayed=[x for x in g['children'] if x in options]
                    selected=st.multiselect('Requirements or nested groups',options,default=displayed,format_func=lambda x:labels[x],key=f'plan_{key}_group_{i}_children')
                    if op!=g['operator'] or selected!=displayed:g.update(operator=op,children=retained+selected);changed=True
                    if st.button('Remove group',key=f'plan_{key}_group_{i}_remove'):remove=i
            if remove is not None:
                retained=[child for child in groups[remove]['children'] if child in hidden]
                if retained:groups[remove]['children']=retained
                else:
                    removed=groups.pop(remove)['group_id']
                    for g in groups:g['children']=[x for x in g['children'] if x!=removed]
                changed=True
        if st.button('Add rule group',key='plan_'+key+'_group_add',disabled=not visible_rules):
            for c in visible_rules:
                if not c.get('clause_id'):c['clause_id']=uuid.uuid4().hex
            groups.append({'group_id':'group_'+uuid.uuid4().hex[:8],'operator':'AND','children':[c['clause_id'] for c in visible_rules]})
            section['logic_groups']=groups;changed=True
        if groups:
            from phase1_extraction.clinical_protocol import validate_logic
            try:validate_logic(groups,{c['clause_id'] for c in rules if c.get('clause_id')})
            except ValueError as error:st.warning(str(error))
    return changed


def _cycle_day_count(phase):
    from phase1_prediction.profile_similarity import days
    length=days(normalized_phase(phase).get('cycle'))
    if length is None or not math.isfinite(length) or length<1 or not float(length).is_integer():return None
    return int(length)


def _dosing_days_editor(phase, key):
    """Offer shortcuts without interpreting unknown or complex source schedules."""
    from phase1_prediction.profile_similarity import administration_days
    original=normalized_phase(phase).get('administration_days')
    known=administration_days(original) or []
    if any(not isinstance(day,(int,float)) or not math.isfinite(day) or day<1 or not float(day).is_integer() for day in known):known=[]
    else:known=[int(day) for day in known]
    cycle_days=_cycle_day_count(phase)
    all_days=list(range(1,cycle_days+1)) if cycle_days and cycle_days<=366 else None
    if not original:default='Not specified'
    elif known and all_days==known:default='Every day'
    elif known and len(known)>1 and known==list(range(known[0],known[-1]+1)):default='Consecutive days'
    elif known:default='Specific days'
    else:default='Custom description'
    mode=choice('Dosing days',['Not specified','Every day','Consecutive days','Specific days','Custom description'],default,key+'_days_mode')
    selected=original
    if mode=='Not specified':
        if mode!=default:selected=None
    elif mode=='Every day':
        if all_days is not None:
            st.caption(f'Every day in the {cycle_days}-day cycle (days 1–{cycle_days}).')
            if known!=all_days:selected=all_days
        else:st.info('Enter a whole-day cycle length (up to 366 days) to use the every-day shortcut. Longer or non-cyclic schedules can use a custom description.')
    elif mode=='Consecutive days':
        left,right=st.columns(2)
        with left:start=optional_number('First dosing day',known[0] if known else 1.,key+'_days_start',1.)
        with right:end=optional_number('Last dosing day',known[-1] if known else None,key+'_days_end',1.)
        if start is not None and end is not None:
            if not start.is_integer() or not end.is_integer() or end<start:
                st.warning('Enter whole day numbers, with the last day on or after the first.')
            elif cycle_days and end>cycle_days:
                st.warning(f'The last dosing day exceeds this {cycle_days}-day cycle. Check the day or the cycle length.')
            elif end-start>365:
                st.info('For more than 366 consecutive dosing days, use a custom description.')
            else:
                proposed=list(range(int(start),int(end)+1))
                if proposed!=known:selected=proposed
    elif mode=='Specific days':
        if all_days is not None:
            proposed=st.multiselect('Days within each cycle',sorted(set(all_days)|set(known)),default=known,key=_input_key(key+'_days'))
            if any(day>cycle_days for day in proposed):st.warning(f'Some retained dosing days exceed this {cycle_days}-day cycle. Check the schedule before continuing.')
            if sorted(proposed)!=known:selected=sorted(proposed) or None
        else:
            entered=st.text_input('Dosing day numbers',', '.join(str(day) for day in known),key=_input_key(key+'_day_numbers'),placeholder='For example: 1, 8, 15 or 1–5')
            parsed=administration_days(entered)
            if entered and (not parsed or any(day<1 for day in parsed)):
                st.warning('Use positive day numbers or ranges, such as 1, 8, 15 or 1–5.')
            elif parsed!=known and (parsed or entered==''):selected=parsed
    elif mode=='Custom description':
        original_text=original if isinstance(original,str) else ', '.join(str(day) for day in known) if known else _text(original)
        entered=st.text_area('Dosing-day description',original_text or '',key=_input_key(key+'_days_description'),placeholder='Describe the complete schedule, including breaks and exceptions.')
        if entered!=(original_text or ''):selected=entered or None
    if selected!=original:
        phase['administration_days']=selected
        structured=phase.get('structured')
        if isinstance(structured,dict):structured.pop('administration_days',None)
    if original and not known and mode!='Custom description':st.caption('Retained source schedule: '+str(original))
    elif phase.get('administration_days') and phase['administration_days']!=original and selected==original:
        st.caption('Retained source schedule: '+_text(phase['administration_days']))


def phase_editor(original,key):
    p=copy.deepcopy(original)
    vocabulary_field(p,'route','route','Route',key+'_route')
    old=p.get('frequency')
    structured=copy.deepcopy(p.get('structured') or {})
    normalized=normalized_phase(p).get('frequency')
    common=['once daily','twice daily','three times daily','every 7 days','every 14 days','every 21 days','every 28 days']
    old_pattern=structured.get('frequency_pattern') or p.get('frequency_pattern')
    inferred_pattern=canonical_value('frequency_pattern',old) or ('intermittent' if normalized in common or structured.get('frequency_count') is not None else None)
    pattern_value,pattern,pattern_changed=vocabulary_choice('Frequency pattern','frequency_pattern',old_pattern or inferred_pattern,key+'_frequency_pattern')
    changed=False
    if pattern_changed:
        if pattern is None:structured.pop('frequency_pattern',None)
        else:structured['frequency_pattern']=pattern
    if pattern in ('continuous','single_dose','as_needed'):
        if pattern_changed:
            p['frequency']={'continuous':'Continuous infusion','single_dose':'Single dose','as_needed':'As needed'}[pattern]
            for field in ('frequency_count','interval_value','interval_unit'):structured.pop(field,None)
    else:
        default=normalized if normalized in common else 'Custom interval' if structured.get('frequency_count') is not None else 'Other' if old else 'Undecided'
        f=choice('Frequency',['Undecided',*common,'Custom interval','Other'],default,key+'_freq',help='Choose a shortcut or enter the administration count and interval. Dosing days within the cycle are entered below.')
        if f=='Custom interval':
            n=optional_number('Administrations per interval',structured.get('frequency_count'),key+'_freq_n',1.)
            interval=optional_number('Interval',structured.get('interval_value'),key+'_interval',.001)
            u=choice('Interval unit',list(choices('interval_unit')),structured.get('interval_unit','days'),key+'_interval_unit')
            if n and interval and (n,interval,u)!=(structured.get('frequency_count'),structured.get('interval_value'),structured.get('interval_unit')):
                structured.update(frequency_count=n,interval_value=interval,interval_unit=u,frequency_pattern='intermittent');changed=True
        elif f=='Other':
            f=st.text_area('Describe frequency',old or '',key=_input_key(key+'_freq_other'))
            if f!=(old or ''):
                p['frequency']=f or None
                for field in ('frequency_count','interval_value','interval_unit'):structured.pop(field,None)
                structured['frequency_pattern']='other'
        elif f!=default:
            p['frequency']=None if f=='Undecided' else f
            for field in ('frequency_count','interval_value','interval_unit'):structured.pop(field,None)
            if f in common:
                count={'once daily':1,'twice daily':2,'three times daily':3}.get(f,1)
                interval=1 if f.endswith('daily') else int(f.split()[1])
                structured.update(frequency_count=count,interval_value=interval,interval_unit='days',frequency_pattern='intermittent')
            else:structured.pop('frequency_pattern',None)
    for field,label,units in [('cycle','Cycle length',['days','weeks']),('infusion_duration','Infusion duration',['minutes','hours'])]:
        old=p.get(field)
        current_form=st.session_state.get('plan_'+key+'_form',p.get('form',''))
        if current_form=='Other':current_form=st.session_state.get('plan_'+key+'_form_custom',current_form)
        if field=='infusion_duration' and not old and not any(x in str(p.get('route','')).lower()+' '+str(current_form).lower()+' '+str(p.get('frequency','')).lower() for x in ('intravenous','infusion','iv')):continue
        match=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(days?|weeks?|hours?|minutes?)',str(old or ''))
        oldn=float(match[1]) if match else float(old) if isinstance(old,(int,float)) else None
        oldu=match[2].rstrip('s')+'s' if match else units[0]
        c1,c2=st.columns([2,1])
        with c1:n=optional_number(label,oldn,key+'_'+field,.000001)
        with c2:u=choice(label+' unit',units,oldu,key+'_'+field+'_unit')
        if n!=oldn or u!=oldu:
            p[field]=f'{n:g} {u}' if n is not None else None
            # Explicit editing replaces any previous normalized representation.
            for k in (('cycle_value','cycle_unit') if field=='cycle' else ('infusion_value','infusion_unit')):structured.pop(k,None)
        if oldn is None and old:st.caption(f'Additional schedule detail retained: {old}')
    # Use current cycle edits when a source also supplies normalized fields.
    if structured:p['structured']=structured
    elif p.get('structured'):p.pop('structured')
    _dosing_days_editor(p,key)
    if p.get('administration_days')!=original.get('administration_days'):structured.pop('administration_days',None)
    with st.container():
        st.markdown('**Stage details (optional)**')
        vocabulary_field(p,'form','dosage_form','Dosage form',key+'_form')
        for field,label in [('start','Stage start event / condition'),('end','Stage end event / transition'),('duration','Planned elapsed treatment duration (if separately specified)'),('infusion_rate','Infusion rate')]:
            text_field(label,p,field,key+'_'+field)
        st.caption('Product / formulation is entered once on the Drugs page. Leave stage details blank if undecided.')
    if structured!=original.get('structured',{}):
        if structured:p['structured']=structured
        else:p.pop('structured',None)
    if changed:p=normalized_phase(p)
    return p


def blank_target():
    return {'version':'target_regimen_v1','name':'New trial','target_dlt':.25,'drugs':[],
            'minimum_clinical_coverage':.2, 'minimum_drug_similarity':0.,
            'points':[],'path_level_ids':[],'profile':{'planning':{'version':PROTOCOL_VERSION}}}


def _assessment_cycle_length(draft, planning, window, *, prefix=''):
    """Select a known stage cycle or retain an independently entered length."""
    from phase1_prediction.profile_similarity import days
    sources={}
    for drug in draft.get('drugs',[]):
        for index,stage in enumerate(drug.get('profile',{}).get('phases',[])):
            length=days(normalized_phase(stage).get('cycle'))
            if length is None or not math.isfinite(length) or length<1:continue
            source=json.dumps([drug['drug_id'],index])
            sources[source]=(f"{drug['name'] or 'Unnamed drug'} · stage {index+1}: {stage.get('phase') or 'treatment'} ({length:g} days)",length)
    previous_length=planning.get('window_cycle_days')
    if previous_length is None and window.get('days'):
        try:reported_cycles=float(planning.get('window_cycles') or window.get('cycles'))
        except (TypeError,ValueError):reported_cycles=None
        if reported_cycles and math.isfinite(reported_cycles) and reported_cycles>0:previous_length=window['days']/reported_cycles
    custom_length=st.session_state.get('plan_'+prefix+'window_cycle_days',planning.get('window_cycle_custom_days',previous_length))
    previous_source=planning.get('window_cycle_source')
    if previous_source and previous_source!='custom' and previous_source not in sources:planning['window_cycle_source_needs_choice']=True
    needs_choice=planning.get('window_cycle_source_needs_choice',False)
    default=previous_source if previous_source in sources or previous_source=='custom' else 'custom' if previous_length is not None and not needs_choice else None
    options=[None,*sources,'custom']
    key=_input_key(prefix+'window_cycle_source')
    if key in st.session_state and st.session_state[key] not in options:st.session_state[key]=default
    source=st.selectbox('Assessment cycle length',options,index=options.index(default),
                        format_func=lambda x:'Choose a treatment stage or enter a custom length' if x is None else 'Enter a separate assessment-cycle length' if x=='custom' else sources[x][0],key=key)
    if source=='custom':
        length=optional_number('Length of each assessment cycle (days)',custom_length,prefix+'window_cycle_days',1.)
        planning['window_cycle_custom_days']=length
    elif source is not None:
        length=sources[source][1]
        if custom_length is not None:planning['window_cycle_custom_days']=custom_length
        st.caption(f'Using the selected administration cycle: {length:g} days. No need to enter that length again.')
    else:
        length=None
        if custom_length is not None:planning['window_cycle_custom_days']=custom_length
        if needs_choice:st.info('The selected administration stage no longer has an available cycle. Choose a stage or a separate assessment-cycle length.')
    if source is not None:planning.pop('window_cycle_source_needs_choice',None)
    planning['window_cycle_source']=source
    return length


def _remove_cycle_source(planning, drug_id, stage_index=None):
    for extra in planning.get('additional_window_plans',[]):
        _remove_cycle_source(extra,drug_id,stage_index)
    source=planning.get('window_cycle_source')
    if not source or source=='custom':return
    try:source_drug,source_index=json.loads(source)
    except (TypeError,ValueError):return
    if source_drug!=drug_id:return
    if stage_index is None or source_index==stage_index:
        planning['window_cycle_source']=None
        planning['window_cycle_source_needs_choice']=True
        if planning.get('window_cycle_custom_days') is None:planning['window_cycle_custom_days']=planning.get('window_cycle_days')
    elif source_index>stage_index:
        planning['window_cycle_source']=json.dumps([source_drug,source_index-1])


def _window_editor(draft, planning, w, *, prefix=''):
    kind=choice('Window purpose',['Undecided','Formal DLT assessment','Safety monitoring'],{'formal_dlt':'Formal DLT assessment','safety_monitoring':'Safety monitoring'}.get(w.get('kind'),'Undecided'),prefix+'window_kind')
    oldkind=w.get('kind','unclear');newkind={'Undecided':'unclear','Formal DLT assessment':'formal_dlt','Safety monitoring':'safety_monitoring'}[kind]
    if oldkind!=newkind:w['kind']=newkind
    anchor_names=labels('window_anchor')
    old_start=w.get('start')
    source_start=planning.get('window_start_event') or w.get('start_anchor') or old_start
    start_code=canonical_value('window_anchor',source_start)
    default_start=anchor_names.get(start_code,'Other' if old_start else 'Undecided')
    start=choice('Observation starts at',['Undecided',*anchor_names.values()],default_start,prefix+'window_start')
    selected_start_code=next((code for code,label in anchor_names.items() if label==start),None)
    planning['window_start_event']=start
    if start=='First dose of a specified drug':
        names={d['drug_id']:d['name'] or 'Unnamed drug' for d in draft.get('drugs',[])}
        options=[None,*names]
        selected=st.selectbox('Drug that starts the assessment window',options,index=options.index(planning.get('window_start_drug')) if planning.get('window_start_drug') in options else 0,format_func=lambda x:'Undecided' if x is None else names[x],key=_input_key(prefix+'window_start_drug'))
        planning['window_start_drug']=selected
        start='First dose of '+names[selected] if selected else old_start if start_code=='first_specified_drug_dose' and old_start else 'Undecided'
    elif start=='Start of a specified stage':
        stage=st.text_input('Stage that starts the assessment window',planning.get('window_start_stage') or '',key=_input_key(prefix+'window_start_stage'))
        planning['window_start_stage']=stage
        start='Start of '+stage if stage else old_start if start_code=='start_of_stage' and old_start else 'Undecided'
    elif start=='Other':start=st.text_input('Start event',w.get('start') or '',key=_input_key(prefix+'window_customstart'))
    if start!=default_start and start!=(w.get('start') or 'Undecided'):
        w['start']=None if start=='Undecided' else start
        w['start_anchor']=selected_start_code

    previous_mode=planning.get('window_mode') or ('Number of days' if w.get('days') is not None else 'Undecided')
    mode=choice('Observation duration',['Undecided','Number of days','Number of cycles','Until an event'],planning.get('window_mode') or ('Number of days' if w.get('days') is not None else 'Undecided'),prefix+'window_mode')
    planning['window_mode']=mode
    if mode=='Undecided' and previous_mode!='Undecided':
        if w.get('raw'):w['source_raw']=w['raw']
        w.update(days=None,cycles=None,end=None,basis='unclear',raw=None)
        w.pop('end_anchor',None)
    if mode=='Number of days':
        n=optional_number('Assessment duration (days)',w.get('days'),prefix+'window_days',1.)
        if n!=w.get('days'):
            w.update(days=n,basis='reported',end=f'After {n:g} days' if n else None,cycles=None,raw=None)
            w.pop('end_anchor',None)
    elif mode=='Number of cycles':
        initial_cycles=planning.get('window_cycles')
        if initial_cycles is None:
            try:initial_cycles=float(w.get('cycles'))
            except (TypeError,ValueError):pass
            if initial_cycles is not None and (not math.isfinite(initial_cycles) or initial_cycles<1):initial_cycles=None
        n=optional_number('Number of assessment cycles',initial_cycles,prefix+'window_cycles',1.)
        cycle=_assessment_cycle_length(draft,planning,w,prefix=prefix)
        planning.update(window_cycles=n,window_cycle_days=cycle)
        w.update(days=n*cycle if n and cycle else None,basis='derived' if n and cycle else 'unclear',cycles=str(n) if n else None,end=f'After {n:g} cycles' if n else None,raw=None)
        w['end_anchor']='end_of_first_cycle' if n==1 else 'end_of_specified_cycle' if n else None
    elif mode=='Until an event':
        vocabulary_field(w,'end','window_anchor','Assessment end event',prefix+'window_end',code_field='end_anchor')
        w['days']=None;w['basis']='unclear'
    if mode=='Undecided':
        text_field('Window description (if structured timing is unavailable)',w,'raw',prefix+'window_raw',area=True)
    if w.get('raw') or w.get('source_raw'):
        st.caption('Retained assessment wording: '+str(w.get('raw') or w['source_raw']))
    return w


def _sex_eligibility(population):
    original=population.get('sex',[])
    value=original[0] if isinstance(original,list) and len(original)==1 else original if isinstance(original,str) else None
    code=canonical_value('sex_eligibility',value)
    names=labels('sex_eligibility')
    options=['Undecided',*[names[item] for item in choices('sex_eligibility')]]
    default=names.get(code,'Undecided')
    selected=choice('Sex eligibility',options,default,'pop_sex',help='Female, Male and No restriction are distinct reported or planned rules. Undecided means missing information, not unrestricted eligibility.')
    if original and code is None:st.caption('Unclassified source eligibility retained: '+_text(original))
    if selected!=default or st.session_state.get('plan_pop_sex_edited',False):population['sex']=[] if selected=='Undecided' else [selected]


def _grading_editor(dlt):
    """System and version have separate closed pools; unknown wording survives."""
    original=dlt.get('grading')
    raw=_text(original)
    matched=re.fullmatch(r'(CTCAE|NCI[ -]?CTC|WHO|RTOG/EORTC)(?:\s+(?:version|v)\s*|\s+)?(\d+(?:\.\d+)?)?',raw,re.I)
    inferred_system=canonical_value('grading_system',matched[1]) if matched else canonical_value('grading_system',raw)
    source_system=dlt.get('grading_system') or dlt.get('grading_code') or inferred_system or raw
    system,system_code,system_changed=vocabulary_choice('Toxicity grading system','grading_system',source_system,'grading',other_label='Other grading system / source wording')
    old_version=dlt.get('grading_version') or (matched[2] if matched else None)
    version,version_code,version_changed=vocabulary_choice('Grading version','grading_version',old_version,'grading_version',other_label='Other grading version')
    if system_changed or version_changed:
        system_label=labels('grading_system').get(system_code,system) if system_code!='other' else system
        parts=[system_label or '',('version '+version) if version else '']
        dlt['grading']=' '.join(part for part in parts if part) or None
        dlt['grading_system']=system_code
        dlt['grading_version']=version_code if version_code not in (None,'other') else version


def _summary_dose(point, drug, state):
    if state == 'Use: dose specified':
        dose = point.get('doses', {}).get(drug['drug_id'])
        if dose is None:
            return 'Dose not entered'
        return f"{dose:g} {drug.get('unit') or '(unit not selected)'}"
    return {'Use: dose unknown': 'Used; dose unknown',
            'Do not use': 'Not used', 'Undecided': 'Use not decided'}.get(state, state)


def render_draft(initial):
    # Conditional controls must retain in-progress values while hidden. Streamlit
    # otherwise removes their widget state at the end of the rerun; the stable
    # form base would then restore the earlier value when they are shown again.
    # Explicit draft replacements and structural rebases clear these keys in
    # multidrug_app before this function runs.
    for key in st.session_state.get('multidrug_plan_input_keys',set()):
        if key in st.session_state:
            st.session_state[key]=st.session_state[key]
    draft=copy.deepcopy(initial)
    profile=draft.setdefault('profile',{})
    planning=profile.setdefault('planning',{'version':PROTOCOL_VERSION})
    actions=[]
    pages=show_steps()
    with pages[0]:
        st.markdown('### Which drugs will be studied?')
        st.caption('Enter each planned drug or drug class. Doses can stay undecided while you review evidence and explore risks.')
        if not draft.get('drugs'):
            first_name=st.text_input('Drug name or development code','',key=_input_key('first_drug_name'),placeholder='Type a drug name, then press Enter')
            if first_name.strip():actions.append(('first_drug',first_name.strip()))
        for i,drug in enumerate(draft.get('drugs',[])):
            did=drug['drug_id'];p=drug.setdefault('profile',{})
            with st.container(border=True):
                st.markdown(f"#### {i+1}. {drug.get('name') or 'New drug'}")
                name_column,unit_column=st.columns([2,1])
                with name_column:drug['name']=st.text_input('Drug name or development code',drug.get('name',''),key='plan_'+did+'_name')
                p['name']=drug['name']
                with unit_column:
                    unit,unit_code,unit_changed=vocabulary_choice('Dose unit (optional)','dose_unit',drug.get('unit'),did+'_unit',code=p.get('dose_unit_code'),help='Needed when you enter actual doses. Leave undecided for relative-dose exploration.')
                    if unit_changed:
                        drug['unit']=unit
                        p['dose_unit_code']=unit_code
                with st.container():
                    st.markdown('**Drug identity and matching details (optional)**')
                    kinds={'drug':'Named drug','drug_class':'Drug class','non_drug':'Non-drug intervention','unclear':'Undecided'}
                    oldkind=p.get('kind','drug')
                    kind=choice('Type of intervention',list(kinds.values()),kinds.get(oldkind,'Undecided'),did+'_kind')
                    selected_kind=next(k for k,v in kinds.items() if v==kind)
                    if selected_kind!=oldkind:p['kind']=selected_kind
                    vocabulary_field(p,'dose_basis','dose_basis','Dose represents',did+'_basis')
                    for field,label in [('aliases','Aliases'),('mechanisms','Mechanisms'),('targets','Molecular targets'),('classes','Drug classes'),('pathways','Pathways')]:
                        old=p.get(field,[]);v=tags(label,old,{'classes':['small molecule','monoclonal antibody','cytotoxic chemotherapy','antibody-drug conjugate'],'mechanisms':['kinase inhibition','immune checkpoint blockade','DNA damage','transcription inhibition']}.get(field,[]),did+'_'+field)
                        if v!=old:p[field]=v
                    text_field('Product / formulation',p,'formulation',did+'_formulation')
                with st.container():
                    st.caption('Order also sets research priority for analyses that use a limited number of interacting drugs.')
                    a,b,c=st.columns(3)
                    if a.button('Move up',key='plan_'+did+'_up',disabled=i==0):actions.append(('up',did))
                    if b.button('Move down',key='plan_'+did+'_down',disabled=i==len(draft['drugs'])-1):actions.append(('down',did))
                    if c.button('Remove drug',key='plan_'+did+'_remove'):actions.append(('remove_drug',did))
        if st.button('Add drug',key='plan_add_drug'):actions.append(('add_drug',None))
    with pages[1]:
        st.markdown('### How will each drug be given?')
        st.caption('Add one schedule per treatment stage. All fields are optional; no schedule is assumed.')
        if not draft.get('drugs'):st.info('Add a drug on the Drugs page first.')
        for drug in draft.get('drugs',[]):
            did=drug['drug_id'];p=drug.setdefault('profile',{})
            with st.container():
                st.markdown('#### '+(drug.get('name') or 'Unnamed drug'))
                phases=p.get('phases',[]);edited=[]
                for j,ph in enumerate(phases):
                    summary=' · '.join(str(ph[field]) for field in ('phase','route','frequency','cycle') if ph.get(field))
                    with st.container(border=True):
                        st.markdown(f"**Schedule {j+1}: {summary or ph.get('phase') or 'Add schedule details'}**")
                        edited.append(phase_editor(ph,f'{did}_phase_{j}'))
                        duplicate,remove=st.columns(2)
                        if duplicate.button('Duplicate schedule',key=f'plan_{did}_duplicate_phase_{j}'):actions.append(('duplicate_phase',(did,j)))
                        if remove.button('Remove stage',key=f'plan_{did}_remove_phase_{j}'):actions.append(('remove_phase',(did,j)))
                if edited!=phases:p['phases']=edited
                if st.button('Add dosing schedule',key='plan_'+did+'_phase_add'):actions.append(('add_phase',did))
                if not p.get('phases'):
                    text_field('Schedule description (if a structured schedule is unavailable)',p,'schedule',did+'_schedule',area=True)
                elif p.get('schedule'):
                    with st.popover('Edit retained schedule wording'):
                        st.caption('Used only when the matched article lacks structured stages. You do not need to repeat a structured schedule here.')
                        text_field('Retained schedule description',p,'schedule',did+'_schedule',area=True)
                rules=p.get('adjustments',[]);new=rules_editor('Dose interruption, reduction and restart rules (optional)',rules,did+'_rules')
                if new!=rules:p['adjustments']=new;actions.append(('rebase',None))
        cfgs=profile.get('configurations') or [{}]
        profile['configurations']=cfgs
        cfg=cfgs[0]
        whole=cfg.get('adjustments',[]);edited_whole=rules_editor('Whole-regimen adjustment rules (optional)',whole,'whole_rules')
        if edited_whole!=whole:cfg['adjustments']=edited_whole;actions.append(('rebase',None))
    with pages[2]:
        st.markdown('### Who can join the trial?')
        cfgs=profile.get('configurations') or [{}];cfg=cfgs[0]
        with st.container():
            original_phase=cfg.get('phase')
            allowed={'phase_i':'I','phase_ia':'Ia','phase_ib':'Ib'}
            phase_code=canonical_value('study_phase',original_phase)
            default=allowed.get(phase_code,'I component of I/II' if original_phase=='I component of I/II' else 'Undecided')
            selected=choice('Planned study phase',['Undecided','I','Ia','Ib','I component of I/II'],default,'cfg_phase')
            if selected!=default:cfg['phase']=None if selected=='Undecided' else selected
            if original_phase and default=='Undecided':st.caption('Source phase retained: '+str(original_phase)+'. Only Phase I or its explicitly separated component is supported.')
        profile['configurations']=cfgs

        st.caption('Who is the trial for, and who can or cannot join? Leave undecided details blank.')
        pop=profile.setdefault('population',{})
        for field,label,opts in [('disease','Disease / cancer type',[]),('modifiers','Disease characteristics',['advanced','metastatic','recurrent','refractory'])]:
            values=pop.get(field,[])
            v=vocabulary_tags(label,values,'disease_modifier','pop_'+field) if field=='modifiers' else tags(label,values,opts,'pop_'+field)
            if v!=values:pop[field]=v
        _sex_eligibility(pop)
        a,b=st.columns(2)
        for container,field,label in [(a,'age_min','Minimum age (years)'),(b,'age_max','Maximum age (years)')]:
            with container:
                v=optional_number(label,pop.get(field),field)
                if v!=pop.get(field):pop[field]=v
                if v is not None:
                    operator_field=field+'_operator'
                    allowed=['Undecided',*choices('age_min_comparison' if field=='age_min' else 'age_max_comparison')]
                    old_operator=pop.get(operator_field)
                    selected=choice('Lower age comparison' if field=='age_min' else 'Upper age comparison',allowed,old_operator or 'Undecided',operator_field)
                    if selected!=(old_operator or 'Undecided'):pop[operator_field]=None if selected=='Undecided' else selected
        eligibility=profile.setdefault('eligibility',{})
        hidden=irrelevant_eligibility_indices(eligibility)
        if any(hidden.values()):
            st.caption('Administrative source requirements are retained in the saved record; only clinical requirements are edited here.')
        eligibility_fields=[('inclusion','Additional inclusion and prior-treatment requirements'),('exclusion','Exclusion requirements')]
        if eligibility.get('unclassified'):
            st.caption('These additional source conditions were not explicitly assigned to inclusion or exclusion. Their original wording and relationships are retained.')
            eligibility_fields.append(('unclassified','Source conditions with unspecified inclusion/exclusion'))
        if eligibility.get('unresolved_source_logic'):
            st.info('Some source rule relationships cannot be fully resolved for this study part and arm. The complete source records are retained for review; only fully applicable relationships appear below.')
        for field,label in eligibility_fields:
            values=eligibility.get(field,[]);new=rules_editor(label,values,'elig_'+field,hidden_indices=hidden[field])
            if new!=values:eligibility[field]=new;actions.append(('rebase',None))
        hidden=irrelevant_eligibility_indices(eligibility)
        hidden_ids={rule['clause_id'] for field,indices in hidden.items()
                    for i,rule in enumerate(eligibility.get(field,[]))
                    if i in indices and isinstance(rule,dict) and rule.get('clause_id')}
        hidden_rules=[rule for field,indices in hidden.items() for i,rule in enumerate(eligibility.get(field,[])) if i in indices]
        if logic_editor(eligibility,[field for field,_ in eligibility_fields],'elig',hidden_clause_ids=hidden_ids,hidden_rules=hidden_rules):actions.append(('rebase',None))
    with pages[3]:
        st.markdown('### What counts as a DLT, and when?')
        st.caption('Define which side effects count as DLT and when they will be counted. Enter planned rules, not observed results.')
        dlt=profile.setdefault('dlt',{})
        _grading_editor(dlt)
        for field,label in [('criteria','DLT criteria'),('exceptions','DLT exceptions')]:
            values=dlt.get(field,[]);new=rules_editor(label,values,'dlt_'+field)
            if new!=values:dlt[field]=new;actions.append(('rebase',None))
        if logic_editor(dlt,['criteria','exceptions'],'dlt'):actions.append(('rebase',None))
        st.markdown('#### Assessment windows')
        windows=profile.get('windows') or [{}]
        extra_plans=planning.setdefault('additional_window_plans',[])
        while len(extra_plans)<len(windows)-1:extra_plans.append({})
        for index,w in enumerate(windows):
            with st.container(border=True):
                st.markdown(f'**Assessment window {index+1}**')
                state=planning if index==0 else extra_plans[index-1]
                _window_editor(draft,state,w,prefix='' if index==0 else f'window_{index}_')
                if index and st.button('Remove assessment window',key=f'plan_window_{index}_remove'):
                    actions.append(('remove_window',index))
        profile['windows']=windows
        if st.button('Add assessment window',key='plan_add_window'):actions.append(('add_window',None))
    with pages[4]:
        st.markdown('### Have you chosen candidate doses? (optional)')
        st.caption('You can review evidence and explore relative-dose curves without candidate doses. Complete actual candidate regimens before running BOIN.')
        st.caption('Each row is a complete planned regimen. Use explicit choices; an unanswered field never means that a drug is used.')
        states=planning.setdefault('exposure_states',{})
        for point in draft.get('points',[]):
            lid=point['level_id'];state_row=states.setdefault(lid,{})
            with st.container(border=True):
                st.markdown(f'**Regimen {lid}**')
                for drug in draft.get('drugs',[]):
                    did=drug['drug_id'];old=point['doses'].get(did)
                    default=state_row.get(did) or ('Do not use' if old==0 else 'Use: dose specified' if old is not None else 'Use: dose unknown' if did in point['doses'] else 'Not entered')
                    left,right=st.columns(2)
                    with left:state=choice(drug['name'] or 'Unnamed drug',USE_STATES,default,lid+'_'+did+'_state')
                    state_row[did]=state
                    if state=='Use: dose specified':
                        with right:value=optional_number(f"Dose ({drug.get('unit') or 'choose a unit'})",old if old and old>0 else None,lid+'_'+did+'_dose',.00000001)
                    else:value=0. if state=='Do not use' else None
                    point['doses'][did]=value
                if st.button('Remove regimen',key='plan_'+lid+'_remove'):actions.append(('remove_point',lid))
        if st.button('Add candidate regimen',key='plan_add_point'):actions.append(('add_point',None))
        with st.container(border=True):
            st.markdown('**Generate candidates with fixed companion drugs**')
            st.caption('Choose one drug to vary. Each new regimen copies a selected template, including explicit use states. Existing regimens are preserved.')
            if draft.get('drugs') and draft.get('points'):
                ids=[d['drug_id'] for d in draft['drugs']];names={d['drug_id']:d['name'] for d in draft['drugs']}
                did=st.selectbox('Drug with candidate doses',ids,format_func=lambda x:names[x],key='plan_generate_drug')
                template=st.selectbox('Template regimen',[p['level_id'] for p in draft['points']],key='plan_generate_template')
                values=st.text_input('Candidate doses (comma separated)',key='plan_generate_values')
                if st.button('Generate candidate regimens',key='plan_generate'):
                    try:
                        doses=[float(x.strip()) for x in values.split(',')]
                        if not doses or any(not math.isfinite(x) or x<=0 for x in doses):raise ValueError()
                        actions.append(('generate',(did,template,doses)))
                    except ValueError:st.warning('Enter positive numeric doses separated by commas.')
    # Determine model coordinates only from explicit user choices.
    for i,d in enumerate(draft.get('drugs',[])):
        d['research_priority']=i+1
        numeric=[p['doses'][d['drug_id']] for p in draft.get('points',[]) if p['doses'].get(d['drug_id']) is not None and p['doses'][d['drug_id']]>0]
        if numeric:
            d['mode']='numeric'
            if not d.get('reference_dose'):
                d['reference_dose']=math.exp(sum(math.log(v) for v in numeric)/len(numeric))
                actions.append(('rebase',None))
        elif draft.get('points') and all(states.get(p['level_id'],{}).get(d['drug_id']) in {'Do not use','Use: dose unknown'} for p in draft.get('points',[])):
            d['mode']='background'
        elif d.get('mode')!='numeric' or not d.get('reference_dose'):d['mode']='background'
    # Apply actions after all current widget edits have been captured.
    for action,value in actions:
        if action in {'add_drug','first_drug'}:
            did='drug_'+uuid.uuid4().hex[:8]
            draft.setdefault('drugs',[]).append({'drug_id':did,'name':value if action=='first_drug' else '','research_priority':len(draft.get('drugs',[]))+1,'research_role':'unclear','mode':'background','profile':{}})
            for point in draft.get('points',[]):states[point['level_id']][did]='Not entered'
        elif action in {'up','down','remove_drug','add_phase','remove_phase','duplicate_phase'}:
            did=value[0] if action in {'remove_phase','duplicate_phase'} else value
            index=next(i for i,d in enumerate(draft['drugs']) if d['drug_id']==did)
            d=draft['drugs'][index]
            if action=='remove_drug':
                _remove_cycle_source(planning,did)
                draft['drugs'].pop(index)
                for p in draft['points']:p['doses'].pop(did,None);states.get(p['level_id'],{}).pop(did,None)
                for pairs in draft.get('mapping_overrides',{}).values():pairs.pop(did,None)
            elif action in {'up','down'}:
                j=index+(-1 if action=='up' else 1);draft['drugs'][index],draft['drugs'][j]=draft['drugs'][j],d
            elif action=='add_phase':d['profile'].setdefault('phases',[]).append({})
            elif action=='remove_phase':
                _remove_cycle_source(planning,did,value[1])
                d['profile']['phases'].pop(value[1])
            elif action=='duplicate_phase':d['profile']['phases'].append(copy.deepcopy(d['profile']['phases'][value[1]]))
        elif action=='add_window':
            profile['windows'].append({})
        elif action=='remove_window':
            profile['windows'].pop(value)
            planning['additional_window_plans'].pop(value-1)
        elif action=='remove_point':
            draft['points']=[p for p in draft['points'] if p['level_id']!=value];states.pop(value,None)
            draft['path_level_ids']=[x for x in draft.get('path_level_ids',[]) if x!=value]
        elif action in {'add_point','generate'}:
            existing={p['level_id'] for p in draft.get('points',[])}
            for dose in value[2] if action=='generate' else [None]:
                n=1
                while f'L{n}' in existing:n+=1
                lid=f'L{n}';existing.add(lid)
                if action=='generate':
                    did,template,_=value
                    point=copy.deepcopy(next(p for p in draft['points'] if p['level_id']==template));point['level_id']=lid;point['doses'][did]=dose
                    states[lid]={**states[template],did:'Use: dose specified'}
                else:
                    point={'level_id':lid,'doses':{}}
                    states[lid]={d['drug_id']:'Not entered' for d in draft['drugs']}
                draft.setdefault('points',[]).append(point)
    for i,d in enumerate(draft.get('drugs',[])):d['research_priority']=i+1
    if draft.get('name') in (None,'','New trial') or planning.get('automatic_name'):
        trial_names=[d['name'] for d in draft.get('drugs',[]) if d.get('name')]
        draft['name']=' + '.join(trial_names)+' trial' if trial_names else 'New trial'
        planning['automatic_name']=True
    if actions:
        st.session_state['multidrug_target_text']=json.dumps(draft,ensure_ascii=False,indent=2)
        st.session_state['multidrug_rebase_editor']=True
        # This rerun ends before the explorer is rendered. Keep its input state
        # through Streamlit's unused-widget cleanup, just as for hidden inputs.
        for key in list(st.session_state):
            if key.startswith('explore_'):
                st.session_state[key]=st.session_state[key]
        st.rerun()
    with pages[5]:
        st.markdown('### Trial plan summary')
        st.write(f"**{draft['name']}**")
        population=profile.get('population',{})
        st.write('Disease: '+('; '.join(population.get('disease',[])) or 'Not specified'))
        for drug in draft.get('drugs',[]):
            descriptions=[]
            for phase in drug.get('profile',{}).get('phases',[]):
                descriptions.append('; '.join(str(phase[k]) for k in ('route','frequency','administration_days','cycle') if phase.get(k)))
            if descriptions:st.write(f"{drug['name']} — "+' / '.join(descriptions))
        window=(profile.get('windows') or [{}])[0]
        if window.get('days') is not None:st.write(f"Assessment duration: {window['days']:g} days; start: {window.get('start') or 'not specified'}.")
        missing=[]
        if not population.get('disease'):missing.append('disease')
        if not profile.get('dlt',{}).get('criteria'):missing.append('DLT criteria')
        if window.get('kind') not in {'formal_dlt','safety_monitoring'}:missing.append('assessment-window purpose')
        if missing:st.caption('Details you can add later: '+', '.join(missing)+'. Matching will use the clinical information provided.')
        if draft.get('points'):
            st.dataframe([{'Regimen':p['level_id'],**{d['name'] or d['drug_id']:_summary_dose(p,d,states.get(p['level_id'],{}).get(d['drug_id'],'Not entered')) for d in draft['drugs']}} for p in draft['points']],hide_index=True)
        issues=planned_issues(draft,require_doses=False)
        if issues:st.info('Before reviewing evidence: '+' '.join(issues))
        else:
            st.success('Plan saved. Next, review matching articles.')
            st.caption('Compare your plan with historical studies, set similarity filters and choose the evidence to use.')
    show_step_buttons(review_ready=not issues)
    return draft


def render(initial):
    return TargetRegimen.model_validate(render_draft(initial))
