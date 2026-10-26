"""Render the English extraction inventory from the current Pydantic contract.

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


MODEL_NAMES = {'ReportedExtraction': 'Article extraction', 'Study': 'Study identity and objective', 'StudyPart': 'Study parts', 'Arm': 'Study arms', 'PopulationDescription': 'Population and eligibility', 'Intervention': 'Drugs and interventions', 'ConditionClause': 'Eligibility / toxicity conditions', 'RuleGroup': 'Condition logic groups', 'Regimen': 'Complete regimens', 'RegimenComponent': 'Drugs within a regimen', 'DosingPhase': 'Administration stages', 'StructuredSchedule': 'Structured administration schedule', 'DoseAdjustmentRule': 'Dose adjustment rules', 'DoseExpression': 'Dose expressions', 'TrialDesign': 'Reported trial design', 'DLTWindow': 'DLT assessment windows', 'DLTDefinition': 'DLT definitions', 'DoseSeries': 'Dose series', 'Cohort': 'Dose cohorts', 'CohortExposure': 'Each cohort’s drug exposure', 'DoseSubgroup': 'Reported dose subgroups', 'AnalysisPopulation': 'Analysis population definitions', 'PopulationCount': 'Patient counts', 'ResultScope': 'Result reporting scope', 'DLTObservation': 'Observed DLT results', 'CountEquation': 'Reported / derived count relationships', 'EvidenceLink': 'Source evidence links', 'ValueProvenance': 'Value provenance', 'UnresolvedItem': 'Unresolved extraction issues', 'ExtractionCoverage': 'Extraction coverage'}

ORDER = list(MODEL_NAMES)




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


def fmt(s):
    nullable=any(x.get('type')=='null' for x in s.get('anyOf', []))
    s=scalar(s)
    suffix=('; nullable when unreported') if nullable else ''
    if '$ref' in s:
        result=('Structured record: ')+s['$ref'].split('/')[-1]
    elif 'const' in s:
        result=('Fixed value: ')+f"`{s['const']}`"
    elif 'enum' in s:
        result=('Closed choices: ')+', '.join(f'`{v}`' for v in s['enum'])
    elif s.get('type')=='array':
        result=('List; each item is ')+fmt(s.get('items', {}))
    elif s.get('type')=='boolean':
        result=('Structured boolean: ')+'`true`, `false`'
    elif s.get('type') in {'integer','number'}:
        result=('Integer entry') if s['type']=='integer' else ('Numeric entry')
        for key,symbol in [('minimum','≥'),('maximum','≤'),('exclusiveMinimum','>'),('exclusiveMaximum','<')]:
            if key in s:result+=f' ({symbol} {s[key]})'
    else:
        result='Text / verbatim wording'
    return result+suffix


def refs(s):
    if '$ref' in s:yield s['$ref'].split('/')[-1]
    for x in s.get('anyOf',[]):yield from refs(x)
    if s.get('type')=='array':yield from refs(s['items'])


def usage(model,field,s):
    if model=='ConditionClause' and field in {'category','category_code'}:
        return 'Extraction classification metadata; compare the clinical measure and full condition, without a duplicate category input'
    if model=='DoseExpression' and field in {'basis_code','denominator_basis'}:
        return 'Dose denominator classification; represented by the chosen target dose unit, without a duplicate field'
    if model=='PopulationDescription' and field in {'age_unit','age_unit_code'}:
        return 'Target age is entered in years; source months/weeks/days convert to years in the comparison view without changing source wording'
    system=field.endswith('_id') or field.endswith('_ids') or field in {
        'schema_version','exposure_contract','evidence','priority_evidence','priority_basis','priority_reason','sequence_index',
        'research_priority','document_id','title','registration_ids','label','is_current_study',
    }
    if system:return 'Source identity / system metadata; no clinical entry'
    if model=='ReportedExtraction':return 'Extraction container; not a clinical input'
    if model in SOURCE_ONLY:return 'Historical record; not entered in a new-trial Plan'
    if (model,field) in LEGACY_NORMALIZED:
        target=LEGACY_NORMALIZED[(model,field)]
        return (f'Retained source / legacy field; target categories use `{target}` selections, with separate text for details')
    if field in {'baseline_summary','schedule_completeness','exposure_coverage','reporting_scope','exposed_n','dose_status','dose_conflict_details','dose_semantics','dose_subgroups','note'}:
        return 'Historical description / completeness status; no Plan entry'
    if field.endswith('_raw') or field in {'raw_expression','raw_text','raw_definition'}:
        return 'Preserve source wording; free text only in the corresponding target description'
    s=scalar(s)
    if 'enum' in s or 'const' in s or s.get('type')=='boolean':
        return 'Closed selection for corresponding target clinical facts; never guess missing facts'
    if s.get('type') in {'number','integer'}:return 'Corresponding target value uses a numeric control, not an arbitrary choice list'
    if s.get('type')=='array' and ('$ref' in scalar(s.get('items',{}))):return 'Repeated entries / stages; see the nested-record table'
    if '$ref' in s:return 'See the nested-record table'
    return 'Corresponding target text / list; no presumed exhaustive vocabulary'


def build():
    schema=ReportedExtraction.model_json_schema(); definitions=schema['$defs']; definitions['ReportedExtraction']=schema
    paths=defaultdict(set);paths['ReportedExtraction'].add('$')
    def walk(name,path,seen=()):
        if name in seen:return
        for field,s in definitions[name].get('properties',{}).items():
            for ref in refs(s):
                new=path+'.'+field+('[]' if scalar(s).get('type')=='array' else '')
                paths[ref].add(new);walk(ref,new,(*seen,name))
    walk('ReportedExtraction','$')
    lines=intro()
    lines+=['',('## Shared closed vocabularies: exact choices available to extraction'),'',
            ('The pools below are shared with the form. English display labels are followed by stored codes. Missing source facts remain missing. Non-Phase-I study codes exist to classify and exclude unsuitable historical data; their presence does not make Phase II available in Plan. Booleans and additional historical-record status enums appear in the complete field tables below.'),'',
            '| '+('Pool | All allowed choices |'), '| --- | --- |']
    for key,entry in CATALOG.items():
        title=entry['label']
        options=[]
        for value in entry['options']:
            label=entry['labels'][value]
            options.append(f'{label} (`{value}`)' if label != value else f'`{value}`')
        lines.append('| '+title+f' (`{key}`) | '+ '; '.join(options)+' |')
    for i,name in enumerate([x for x in ORDER if x in paths]+sorted(set(paths)-set(ORDER)),1):
        title=MODEL_NAMES.get(name, name)
        lines+=['',f'## {i}. {title} — `{name}`','']
        # Several records (doses, evidence) are deeply reused. Document each distinct
        # schema rather than repeating identical tables for every nesting path.
        locations=sorted(paths[name])
        lines.append(('Locations: ')+', '.join(f'`{p}`' for p in locations))
        lines+=['', '| '+('Field | Meaning | Extraction format / allowed values | Plan relationship |'),
                '| --- | --- | --- | --- |']
        for field,s in definitions[name].get('properties',{}).items():
            label=field.replace('_',' ')
            values=[f'`{field}`',label,fmt(s),usage(name,field,s)]
            lines.append('| '+' | '.join(x.replace('|','\\|').replace('\n',' ') for x in values)+' |')
    lines+=['', ('## Boundaries with source materials and mathematical models'),'',
            ('This catalog covers every field in ReportedExtraction and its referenced record types. PDF/XML page geometry, cell coordinates and OCR intermediate states belong to document parsing and are not listed as clinical indicators. Curve parameters, priors, similarity cutoffs, article inclusion decisions, reference-dose mappings and BOIN settings are analysis controls, not facts to invent during extraction.'),'',
            ('Generated from the Pydantic JSON Schema in phase1_extraction/models.py. Regenerate with:'),'',
            '```sh','v3/.venv/bin/python v3/scripts/generate_extraction_field_catalog.py','```','']
    return '\n'.join(lines)


def intro():

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
        'R3 is retired. Legacy `r3_design_inferred` provenance remains readable for historical audit only; it cannot supply a denominator or arithmetic input in current extraction.','',
        '## Reading the tables','',
        '`[]` denotes repeated records: $.cohorts[].exposures[] contains each drug in each cohort. null or an empty list means the corresponding information is absent. unclear and unknown are legal only where explicitly listed. A property being required in a JSON response does not authorize inventing its value when the source is silent.','',
        'The Plan relationship column describes the input boundary. Historical results and provenance support extraction, review and modeling; the user planning a new trial does not re-enter them.',
    ]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    # Keep the old _ZH document path as an English compatibility copy.
    for suffix in ['', '_ZH']:
        path=ROOT/'docs'/f'EXTRACTION_FIELD_CATALOG{suffix}.md';content=build()
        if args.check:
            if not path.exists() or path.read_text()!=content:raise SystemExit(f'Catalog is stale: {path}')
        else:path.write_text(content)


if __name__=='__main__':main()
