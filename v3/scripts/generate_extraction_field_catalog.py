"""Render the English/Chinese extraction inventory from the current Pydantic contract.

This is documentation only: no source extraction or target draft is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase1_extraction.models import ReportedExtraction
from phase1_extraction.controlled_vocabulary import CATALOG


MODEL_NAMES = {
    'ReportedExtraction': ('Article extraction', '整篇文献提取结果'),
    'Study': ('Study identity and objective', '研究身份与目的'),
    'StudyPart': ('Study parts', '研究部分'),
    'Arm': ('Study arms', '研究分组'),
    'PopulationDescription': ('Population and eligibility', '人群与入排标准'),
    'Intervention': ('Drugs and interventions', '药物与干预'),
    'ConditionClause': ('Eligibility / toxicity conditions', '入排或毒性条件'),
    'RuleGroup': ('Condition logic groups', '条件逻辑分组'),
    'Regimen': ('Complete regimens', '完整用药方案'),
    'RegimenComponent': ('Drugs within a regimen', '方案内的药物'),
    'DosingPhase': ('Administration stages', '给药阶段'),
    'StructuredSchedule': ('Structured administration schedule', '结构化给药安排'),
    'DoseAdjustmentRule': ('Dose adjustment rules', '减量、暂停和恢复规则'),
    'DoseExpression': ('Dose expressions', '剂量表达式'),
    'TrialDesign': ('Reported trial design', '文献报告的试验设计'),
    'DLTWindow': ('DLT assessment windows', 'DLT 评价窗口'),
    'DLTDefinition': ('DLT definitions', 'DLT 定义'),
    'DoseSeries': ('Dose series', '剂量系列'),
    'Cohort': ('Dose cohorts', '剂量队列'),
    'CohortExposure': ('Each cohort’s drug exposure', '每个队列的逐药暴露'),
    'DoseSubgroup': ('Reported dose subgroups', '文献报告的剂量亚组'),
    'AnalysisPopulation': ('Analysis population definitions', '分析人群定义'),
    'PopulationCount': ('Patient counts', '患者人数'),
    'ResultScope': ('Result reporting scope', '结果的报告范围'),
    'DLTObservation': ('Observed DLT results', '实际 DLT 结果'),
    'CountEquation': ('Reported / derived count relationships', '人数之间的加总关系'),
    'EvidenceLink': ('Source evidence links', '来源证据定位'),
    'ValueProvenance': ('Value provenance', '数值来源'),
    'UnresolvedItem': ('Unresolved extraction issues', '未解决的提取问题'),
    'ExtractionCoverage': ('Extraction coverage', '提取覆盖情况'),
}

ORDER = list(MODEL_NAMES)
FIELD_ZH = {
    'schema_version':'提取格式版本','exposure_contract':'逐药暴露格式版本','document_id':'文献标识','title':'文章标题',
    'studies':'研究列表','study_parts':'研究部分列表','arms':'研究分组列表','populations':'人群列表',
    'interventions':'干预列表','regimens':'用药方案列表','designs':'试验设计列表','dlt_windows':'DLT 窗口列表',
    'dlt_definitions':'DLT 定义列表','dose_series':'剂量系列列表','cohorts':'剂量队列列表',
    'analysis_populations':'分析人群定义列表','population_counts':'人数记录列表','result_scopes':'结果报告范围列表',
    'dlt_observations':'DLT 观察结果列表','count_equations':'人数关系列表','unresolved_items':'未解决问题列表',
    'extraction_coverage':'提取覆盖记录','material_coverage_complete':'是否覆盖全部提供的材料',
    'extraction_complete':'是否完成提取','coverage_notes':'覆盖情况备注',
    'study_id':'研究标识','label':'来源中的名称或标签','registration_ids':'研究注册号','is_current_study':'是否为本文当前研究',
    'purpose_summary':'研究目的','evidence':'支持该记录的证据','part_id':'研究部分标识','phase_raw':'研究期别原文',
    'phase':'规范化研究期别','role':'类别或作用','sequence_index':'先后顺序','arm_id':'研究组标识',
    'population_description':'人群描述','regimen_id':'用药方案标识','population_id':'人群标识','disease_raw':'疾病原文',
    'disease_core':'疾病核心名称','modifiers':'疾病状态与限定条件','age_min':'最低年龄','age_min_comparator':'最低年龄比较符',
    'age_max':'最高年龄','age_max_comparator':'最高年龄比较符','age_unit':'年龄单位','sex_eligibility_raw':'性别入组要求原文',
    'sex_eligibility':'规范化性别入组要求','inclusion_criteria':'入组条件原文','exclusion_criteria':'排除条件原文',
    'prior_treatment_requirements':'既往治疗要求','clinical_requirements':'器官功能等临床要求','baseline_summary':'已入组人群基线描述',
    'eligibility_conditions':'结构化入排条件','eligibility_logic_groups':'入排条件逻辑分组','intervention_id':'干预标识',
    'name_raw':'药物或干预名称原文','aliases':'别名或开发代号','modality':'干预类型','product_or_formulation':'产品或制剂说明',
    'drug_class_raw':'药物类别原文','mechanism_raw':'作用机制原文','targets_raw':'靶点原文','pathways_raw':'通路原文',
    'raw_expression':'剂量原文','value_kind':'剂量表达类型','comparator':'比较符','value':'剂量数值',
    'range_min':'范围下限','range_max':'范围上限','discrete_values':'离散剂量值','coefficient':'科学计数法系数','exponent':'科学计数法指数',
    'unit_raw':'单位原文','unit_normalized':'规范化剂量单位','denominator_basis':'剂量分母基准','dose_cap_raw':'剂量上限原文','formula_raw':'计算公式原文',
    'dosing_phase_id':'给药阶段标识','structured':'结构化给药安排','phase_label':'给药阶段名称','dose':'剂量表达',
    'route_raw':'给药途径原文','route':'规范化给药途径','dose_form_raw':'剂型原文','dose_form':'规范化剂型',
    'frequency_raw':'给药频率原文','frequency':'规范化频率','administration_days_raw':'给药日原文','cycle_length_raw':'周期长度原文',
    'duration_raw':'疗程时长原文','infusion_duration_raw':'输注时长原文','infusion_rate_raw':'输注速率原文',
    'start_condition_raw':'阶段开始条件原文','end_condition_raw':'阶段结束条件原文','schedule_completeness':'给药信息完整程度',
    'research_priority':'主要研究药物排序','research_role':'研究角色','priority_basis':'排序依据类别','priority_reason':'排序理由',
    'priority_evidence':'排序依据证据','component_id':'方案成分标识','participates_in_dose_finding':'是否参与剂量探索',
    'dosing_phases':'逐阶段给药安排','composition':'方案组成类型','components':'方案成分列表','shared_schedule_raw':'全方案共用给药说明',
    'dose_adjustment_rules':'减量等调整规则','design_id':'试验设计标识','raw_name':'设计名称原文','normalized_type':'规范化试验设计类型',
    'initial_cohort_n':'初始队列人数','added_cohort_n':'追加队列人数','expanded_total_n':'扩展后总人数','replacement_count':'替换人数',
    'nonevaluable_count':'不可评价人数','cohort_addition_count':'追加队列次数','expand_when_y_equals':'触发扩展的 DLT 人数',
    'stop_when_y_at_least':'触发停止的最低 DLT 人数','decision_rules_raw':'决策规则原文','amendments_raw':'设计修订原文',
    'window_kind':'窗口类型','window_id':'窗口标识','raw_definition':'窗口定义原文','start_raw':'窗口起点原文','end_raw':'窗口终点原文',
    'cycle_numbers_raw':'涉及周期原文','normalized_days':'可确定的窗口天数','basis':'来源或推导依据','definition_id':'DLT 定义标识',
    'version_label':'定义版本或标签','criteria_raw':'DLT 判定条件原文','exceptions_raw':'DLT 判定例外原文','logic_raw':'条件逻辑原文',
    'grading_system_raw':'毒性分级体系原文','grading_system':'规范化毒性分级体系','grading_version':'毒性分级体系版本',
    'criteria_structured':'结构化 DLT 判定条件','criteria_logic_groups':'DLT 条件逻辑分组','series_id':'剂量系列标识',
    'varying_component_id':'发生剂量变化的成分标识','fixed_component_ids':'固定成分标识','comparable_unit':'该系列可比较的剂量单位',
    'n':'人数','population_basis':'该人数的人群定义','exposure_id':'暴露记录标识','entity_kind':'药物实体类型','presence':'是否使用',
    'reporting_scope':'报告范围','exposed_n':'使用该药的人数','dose_status':'剂量报告状态','dose_conflict_details':'剂量冲突详情',
    'dose_semantics':'多剂量记录含义','dose_subgroups':'实际剂量亚组','formulation_raw':'制剂原文','schedule_raw':'给药安排原文',
    'note':'备注','cohort_id':'队列标识','actual_status':'队列实际使用状态','cohort_role':'队列作用','exposures':'逐药暴露记录',
    'exposure_coverage':'逐药暴露提取完整程度','expansion_status':'队列扩展状态','result_scope_ids':'所属结果范围标识',
    'analysis_population_id':'分析人群标识','author_name':'作者给出的人群名称','interpreted_role':'规范化分析人群类别',
    'definition_raw':'分析人群定义原文','count_id':'人数记录标识','result_scope_id':'结果范围标识','author_role_raw':'作者所述人数类别',
    'reported_value':'文献报告的人数','reporting_status':'报告状态','timepoint_raw':'时间点原文','data_cutoff_raw':'数据截止时间原文',
    'is_cumulative':'是否为累计人数','equivalent_to_role':'明确等同的人群类别','derived_from_count_ids':'推导引用的人数记录',
    'provenance':'数值来源说明','cohort_ids':'该结果覆盖的队列','dlt_results_coverage':'DLT 结果覆盖完整程度',
    'has_observed_dlt_results':'是否有实际 DLT 观察结果','explicit_no_dlt_in_scope':'是否明确报告范围内无 DLT',
    'unresolved_positive_dlt':'是否仍有未分配的阳性 DLT','observation_id':'DLT 观察记录标识','count_kind':'报告计数的对象',
    'reported_count':'报告的计数','explicit_patient_count':'明确报告的 DLT 患者数','same_patient_repeated_events':'是否同一患者重复事件',
    'denominator_count_id':'对应分母人数标识','toxicities_raw':'观察到的毒性名称原文','equation_id':'人数关系标识',
    'value_type':'人数关系所指类型','total_record_id':'合计记录标识','component_record_ids':'组成记录标识',
    'mutually_exclusive':'分组是否互斥','exhaustive':'分组是否穷尽总人群','scope_description':'人数关系范围说明',
    'clause_id':'条件标识','metric':'规范化临床指标','metric_other':'候选池以外的指标说明','category':'条件类别',
    'raw_text':'完整原文','applies_to_part_ids':'适用研究部分','applies_to_arm_ids':'适用研究组','threshold':'阈值或下界',
    'threshold_upper':'阈值上界','unit':'规范化单位','unit_other':'候选池以外的单位说明','timing_raw':'条件时间要求原文',
    'connective_to_previous':'与前一条件的逻辑关系','negated':'是否否定该条件','group_id':'逻辑组标识','operator':'逻辑运算符',
    'children':'组成条件或嵌套组标识','frequency_count':'每个间隔内给药次数','interval_value':'给药间隔数值',
    'interval_unit':'给药间隔单位','cycle_value':'周期长度数值','cycle_unit':'周期单位','administration_days':'周期内给药日序号',
    'infusion_value':'输注时长数值','infusion_unit':'输注时长单位','dose_basis':'剂量所代表的时间基准','rule_id':'调整规则标识',
    'action':'调整动作','condition_raw':'触发条件原文','action_raw':'调整动作详情原文','applies_to_component_ids':'适用药物成分',
    'source_ids':'来源材料位置标识','source_text':'支持该事实的原文摘录','locator':'页码或表格等定位','support_type':'证据支持形式',
    'input_record_ids':'引用的输入记录','expression':'数值推导表达式','item_id':'问题标识','description':'问题描述',
    'affected_ids':'受影响记录','status':'覆盖状态',
    'age_unit_code':'规范化年龄单位','age_min_operator':'规范化最低年龄比较符','age_max_operator':'规范化最高年龄比较符',
    'disease_modifier_codes':'规范化疾病状态','metric_code':'临床指标代码','category_code':'条件类别代码',
    'comparator_code':'规范化比较符','unit_code':'规范化单位','basis_code':'规范化剂量分母基准',
    'phase_kind':'规范化给药阶段类别','frequency_pattern':'给药频率模式','start_anchor':'规范化窗口开始事件',
    'end_anchor':'规范化窗口结束事件','comparable_unit_code':'规范化可比较剂量单位',
    'condition_structured':'结构化调整触发条件',
}

POOL_ZH = {
    'sex_eligibility':'性别入组要求','study_phase':'研究期别','clinical_metric':'临床指标或事件','comparison':'比较符',
    'condition_category':'临床条件类别','condition_unit':'条件阈值单位','age_unit':'年龄单位','route':'给药途径',
    'dosage_form':'剂型','treatment_stage':'给药阶段','frequency_pattern':'给药频率模式','dose_unit':'剂量单位',
    'dose_denominator':'剂量分母基准','grading_system':'毒性分级体系','grading_version':'毒性分级版本',
    'window_anchor':'DLT 窗口锚点','disease_modifier':'疾病状态','research_role':'研究角色','treatment_role':'治疗角色',
    'study_part_role':'研究部分角色','dose_basis':'剂量时间基准','interval_unit':'给药间隔单位','cycle_unit':'周期单位',
    'infusion_unit':'输注时长单位',
    'adjustment_action':'给药调整动作',
}

SOURCE_ONLY = {
    'Arm','TrialDesign','DoseSeries','Cohort','DoseSubgroup','AnalysisPopulation','PopulationCount',
    'ResultScope','DLTObservation','CountEquation','EvidenceLink','ValueProvenance','UnresolvedItem','ExtractionCoverage',
}

LEGACY_NORMALIZED = {
    ('StudyPart','phase_raw'):'phase',
    ('PopulationDescription','sex_eligibility_raw'):'sex_eligibility',
    ('PopulationDescription','age_unit'):'age_unit_code',
    ('PopulationDescription','age_min_comparator'):'age_min_operator',
    ('PopulationDescription','age_max_comparator'):'age_max_operator',
    ('PopulationDescription','modifiers'):'disease_modifier_codes',
    ('ConditionClause','metric'):'metric_code',
    ('ConditionClause','category'):'category_code',
    ('ConditionClause','comparator'):'comparator_code',
    ('ConditionClause','unit_raw'):'unit_code',
    ('DoseExpression','comparator'):'comparator_code',
    ('DoseExpression','unit_raw'):'unit_code',
    ('DoseExpression','unit_normalized'):'unit_code',
    ('DoseExpression','denominator_basis'):'basis_code',
    ('DosingPhase','route_raw'):'route',
    ('DosingPhase','dose_form_raw'):'dose_form',
    ('DosingPhase','phase_label'):'phase_kind',
    ('DLTWindow','start_raw'):'start_anchor',
    ('DLTWindow','end_raw'):'end_anchor',
    ('DLTDefinition','grading_system_raw'):'grading_system + grading_version',
    ('DoseSeries','comparable_unit'):'comparable_unit_code',
}


def scalar(s):
    if 'anyOf' in s:
        values=[x for x in s['anyOf'] if x.get('type') != 'null']
        return scalar(values[0]) if len(values)==1 else s
    return s


def fmt(s, zh=False):
    nullable=any(x.get('type')=='null' for x in s.get('anyOf', []))
    s=scalar(s)
    suffix=('；未报告可为空' if zh else '; nullable when unreported') if nullable else ''
    if '$ref' in s:
        result=('结构化记录：' if zh else 'Structured record: ')+s['$ref'].split('/')[-1]
    elif 'const' in s:
        result=('固定值：' if zh else 'Fixed value: ')+f"`{s['const']}`"
    elif 'enum' in s:
        result=('固定候选：' if zh else 'Closed choices: ')+', '.join(f'`{v}`' for v in s['enum'])
    elif s.get('type')=='array':
        result=('列表；元素为 ' if zh else 'List; each item is ')+fmt(s.get('items', {}), zh)
    elif s.get('type')=='boolean':
        result=('结构化判断：' if zh else 'Structured boolean: ')+'`true`, `false`'
    elif s.get('type') in {'integer','number'}:
        result=('整数填写' if zh else 'Integer entry') if s['type']=='integer' else ('数值填写' if zh else 'Numeric entry')
        for key,symbol in [('minimum','≥'),('maximum','≤'),('exclusiveMinimum','>'),('exclusiveMaximum','<')]:
            if key in s:result+=f' ({symbol} {s[key]})'
    else:
        result='文字 / 原文' if zh else 'Text / verbatim wording'
    return result+suffix


def refs(s):
    if '$ref' in s:yield s['$ref'].split('/')[-1]
    for x in s.get('anyOf',[]):yield from refs(x)
    if s.get('type')=='array':yield from refs(s['items'])


def usage(model,field,s,zh):
    if model=='ConditionClause' and field in {'category','category_code'}:
        return '提取用的粗分类标签；具体临床指标和完整条件参与比较，不要求用户重复填写类别' if zh else 'Extraction classification metadata; compare the clinical measure and full condition, without a duplicate category input'
    if model=='DoseExpression' and field in {'basis_code','denominator_basis'}:
        return '剂量分母分类；目标端由所选剂量单位表达，不再单独重复填写' if zh else 'Dose denominator classification; represented by the chosen target dose unit, without a duplicate field'
    if model=='PopulationDescription' and field in {'age_unit','age_unit_code'}:
        return '目标网页统一按年填写；来源月份、周或天在比较视图中换算成年，原文不变' if zh else 'Target age is entered in years; source months/weeks/days convert to years in the comparison view without changing source wording'
    system=field.endswith('_id') or field.endswith('_ids') or field in {
        'schema_version','exposure_contract','evidence','priority_evidence','priority_basis','priority_reason','sequence_index',
        'research_priority','document_id','title','registration_ids','label','is_current_study',
    }
    if system:return '来源定位 / 系统维护；不作为临床填写项' if zh else 'Source identity / system metadata; no clinical entry'
    if model=='ReportedExtraction':return '列表由提取流程组织；不逐项填写' if zh else 'Extraction container; not a clinical input'
    if model in SOURCE_ONLY:return '历史文献记录；新试验 Plan 不填写' if zh else 'Historical record; not entered in a new-trial Plan'
    if (model,field) in LEGACY_NORMALIZED:
        target=LEGACY_NORMALIZED[(model,field)]
        return (f'保留原文 / 旧字段；目标分类使用 `{target}` 对应选项，额外说明单独填写' if zh else f'Retained source / legacy field; target categories use `{target}` selections, with separate text for details')
    if field in {'baseline_summary','schedule_completeness','exposure_coverage','reporting_scope','exposed_n','dose_status','dose_conflict_details','dose_semantics','dose_subgroups','note'}:
        return '历史来源描述或完整性状态；Plan 不填写' if zh else 'Historical description / completeness status; no Plan entry'
    if field.endswith('_raw') or field in {'raw_expression','raw_text','raw_definition'}:
        return '保留来源原文；目标端只有相应的文字说明项允许自由输入' if zh else 'Preserve source wording; free text only in the corresponding target description'
    s=scalar(s)
    if 'enum' in s or 'const' in s or s.get('type')=='boolean':
        return '如属目标临床事实，使用固定选项；缺失不猜测' if zh else 'Closed selection for corresponding target clinical facts; never guess missing facts'
    if s.get('type') in {'number','integer'}:return '对应目标数值使用数字框；不提供任意候选数值' if zh else 'Corresponding target value uses a numeric control, not an arbitrary choice list'
    if s.get('type')=='array' and ('$ref' in scalar(s.get('items',{}))):return '逐条或逐阶段填写；子字段见相应表' if zh else 'Repeated entries / stages; see the nested-record table'
    if '$ref' in s:return '对应子字段见相应表' if zh else 'See the nested-record table'
    return '对应目标自由文字或列表；没有假设完整的候选池' if zh else 'Corresponding target text / list; no presumed exhaustive vocabulary'


def build(zh=False):
    schema=ReportedExtraction.model_json_schema(); definitions=schema['$defs']; definitions['ReportedExtraction']=schema
    paths=defaultdict(set);paths['ReportedExtraction'].add('$')
    def walk(name,path,seen=()):
        if name in seen:return
        for field,s in definitions[name].get('properties',{}).items():
            for ref in refs(s):
                new=path+'.'+field+('[]' if scalar(s).get('type')=='array' else '')
                paths[ref].add(new);walk(ref,new,(*seen,name))
    walk('ReportedExtraction','$')
    lines=intro(zh)
    lines+=['',('## 共享固定候选池：提取模型只能选择这些值' if zh else '## Shared closed vocabularies: exact choices available to extraction'),'',
            ('下表把候选池集中列出。网页显示英文名称，保存的是括号内代码。原文没有说明时允许缺失，不要求硬选。研究期别保留非一期分类是为了识别、排除不适用的历史数据；这不代表 Plan 允许填写 Phase II。布尔量与文献记录状态等其它固定值在后面的逐字段表中完整列出。' if zh else 'The pools below are shared with the form. English display labels are followed by stored codes. Missing source facts remain missing. Non-Phase-I study codes exist to classify and exclude unsuitable historical data; their presence does not make Phase II available in Plan. Booleans and additional historical-record status enums appear in the complete field tables below.'),'',
            '| '+('候选池 | 全部允许值 |' if zh else 'Pool | All allowed choices |'), '| --- | --- |']
    for key,entry in CATALOG.items():
        title=POOL_ZH.get(key,entry['label']) if zh else entry['label']
        options=[]
        for value in entry['options']:
            label=entry['labels'][value]
            options.append(f'{label} (`{value}`)' if label != value else f'`{value}`')
        lines.append('| '+title+f' (`{key}`) | '+ '; '.join(options)+' |')
    for i,name in enumerate([x for x in ORDER if x in paths]+sorted(set(paths)-set(ORDER)),1):
        title=MODEL_NAMES.get(name,(name,name))[int(zh)]
        lines+=['',f'## {i}. {title} — `{name}`','']
        # Several records (doses, evidence) are deeply reused. Document each distinct
        # schema rather than repeating identical tables for every nesting path.
        locations=sorted(paths[name])
        lines.append(('出现位置：' if zh else 'Locations: ')+', '.join(f'`{p}`' for p in locations))
        lines+=['', '| '+('字段 | 指标含义 | 提取格式 / 允许值 | 网页填写关系 |' if zh else 'Field | Meaning | Extraction format / allowed values | Plan relationship |'),
                '| --- | --- | --- | --- |']
        for field,s in definitions[name].get('properties',{}).items():
            label=FIELD_ZH.get(field,field.replace('_',' ')) if zh else field.replace('_',' ')
            values=[f'`{field}`',label,fmt(s,zh),usage(name,field,s,zh)]
            lines.append('| '+' | '.join(x.replace('|','\\|').replace('\n',' ') for x in values)+' |')
    lines+=['', ('## 与来源材料、数学模型的边界' if zh else '## Boundaries with source materials and mathematical models'),'',
            ('本文完整列出 `ReportedExtraction` 及其引用记录的字段。PDF/XML 的页面坐标、单元格几何、OCR 中间状态等属于材料解析层，不是临床提取指标，不在此逐项展开。曲线参数、先验、相似度门槛、文章纳入决策、参考剂量映射和 BOIN 模拟设置是分析设置，不要求 LLM 从文章中伪造，也不混入上述临床候选池。' if zh else
             'This catalog covers every field in ReportedExtraction and its referenced record types. PDF/XML page geometry, cell coordinates and OCR intermediate states belong to document parsing and are not listed as clinical indicators. Curve parameters, priors, similarity cutoffs, article inclusion decisions, reference-dose mappings and BOIN settings are analysis controls, not facts to invent during extraction.'),'',
            ('生成来源：`phase1_extraction/models.py` 的 Pydantic JSON Schema。重新生成命令：' if zh else 'Generated from the Pydantic JSON Schema in phase1_extraction/models.py. Regenerate with:'),'',
            '```sh','v3/.venv/bin/python v3/scripts/generate_extraction_field_catalog.py','```','']
    return '\n'.join(lines)


def intro(zh):
    if zh:return [
        '# V3 提取指标、结构化候选池与网页填写对应表','',
        '这份目录回答三个问题：文章中提取什么；模型只能从哪些候选值中选择；新试验网页应当选择、填写数值还是输入文字。下面的字段表直接读取当前提取数据契约生成，固定候选值与代码一致。','',
        '## 填写和提取遵循的规则','',
        '- **有限分类用固定选项。** 同一字段的提取和网页使用同一候选池；不能把一段文章原文当成新选项塞进下拉列表。',
        '- **连续量用数字填写。** 年龄、剂量、阈值、间隔、周期长度和输注时长都是结构化数值，不适合列出所有可能数值。单位和比较符另外选择。',
        '- **开放内容保留文字。** 药物名称、疾病名称、机制、靶点、通路、制剂细节、完整入排条件、例外和复杂给药规则没有一个可靠且穷尽的短候选池，继续保留文字或段落。',
        '- **未报告不等于无限制。** 性别的临床候选为 female、male、no_restriction；来源未报告时为空，网页可以显示 Not specified。它不是第四种性别类别，更不能默认成 No restriction。',
        '- **Other 只是固定分类。** 选中 Other 后，具体说明单独保存为文字；说明本身不会扩展所有用户共用的候选池。',
        '- **原文与规范化结果并存。** 原文用于溯源；规范化字段用于比较。明确的结构化比较符、阈值和单位优先；没有明确比较表达或分级陈述时，不从任意数字猜测阈值。',
        '- **新试验只填写事前已知的信息。** 历史研究的实际人数、DLT 人数、实际毒性、设计执行情况、证据位置和提取状态不会变成 Plan 必填项。',
        '- **本次不自动重新提取旧文献。** 已保存的来源提取文件保留；新契约用于后续提取。兼容读取或显示时的规范化不覆盖来源资料。','',
        '## 如何阅读后面的表','',
        '`[]` 表示可重复记录；例如 `$.cohorts[].exposures[]` 是每个队列的每一种药物。`null` 或空列表表示没有相应信息；`unclear`、`unknown` 等只在该字段确实列出时才是合法值。字段是否必须出现在 JSON 中与文章是否报告该事实是两回事：允许为空的字段不能因为格式要求而编造。','',
        '“网页填写关系”描述对应原则。历史记录及来源证据只供提取、核验和建模使用；它们不要求计划一个新试验的用户重新填写。',
    ]
    return [
        '# V3 extraction fields, closed choices and Plan controls','',
        'This catalog lists what is extracted, the exact choices available to the extraction model, and whether the corresponding target fact needs a selection, a number or prose. The tables are generated from the current extraction data contract.','',
        '## Extraction and entry rules','',
        '- **Finite categories use closed choices.** Extraction and Plan share a vocabulary. An article sentence must never become a new dropdown option.',
        '- **Continuous quantities use numbers.** Age, dose, thresholds, intervals, cycle lengths and infusion durations are structured numeric values. Units and comparison operators are selected separately.',
        '- **Open concepts remain text.** Drug and disease names, mechanisms, targets, pathways, formulation details, full eligibility statements, exceptions and complex administration rules do not have a reliable exhaustive short vocabulary.',
        '- **Unreported does not mean unrestricted.** The clinical sex-eligibility choices are female, male and no_restriction. An unreported source value is null and the UI may show Not specified; it is not an additional sex category or an assumption of no restriction.',
        '- **Other is a closed category.** Its detail is separate text; that detail never expands the global choice pool.',
        '- **Source wording and normalized facts coexist.** Source text provides traceability. Explicit structured operators, thresholds and units take precedence over parsed defaults. A number without an explicit comparison or grade statement is not guessed to be a threshold.',
        '- **A new trial needs pretrial facts.** Historical sample sizes, DLT counts, observed toxicities, design execution, evidence locations and extraction statuses are not Plan requirements.',
        '- **Existing articles are not re-extracted automatically.** Saved source extractions remain intact. The new contract applies to subsequent extraction; compatibility normalization does not overwrite source evidence.','',
        '## Reading the tables','',
        '`[]` denotes repeated records: $.cohorts[].exposures[] contains each drug in each cohort. null or an empty list means the corresponding information is absent. unclear and unknown are legal only where explicitly listed. A property being required in a JSON response does not authorize inventing its value when the source is silent.','',
        'The Plan relationship column describes the input boundary. Historical results and provenance support extraction, review and modeling; the user planning a new trial does not re-enter them.',
    ]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    for zh,suffix in [(False,''),(True,'_ZH')]:
        path=ROOT/'docs'/f'EXTRACTION_FIELD_CATALOG{suffix}.md';content=build(zh)
        if args.check:
            if not path.exists() or path.read_text()!=content:raise SystemExit(f'Catalog is stale: {path}')
        else:path.write_text(content)


if __name__=='__main__':main()
