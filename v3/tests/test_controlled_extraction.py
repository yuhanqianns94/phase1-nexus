"""The extraction schema and UI catalog share finite, lossless choices."""
import copy
import pytest
from pydantic import ValidationError

from phase1_extraction.controlled_vocabulary import CATALOG, canonical_value, choices, labels
from phase1_extraction.field_normalization import normalize_fields
from phase1_extraction.llm import build_request_body, load_prompt
from phase1_extraction.models import (
    SCHEMA_VERSION, ReportedExtraction, PopulationDescription, ConditionClause,
    DosingPhase, DoseExpression, StudyPart, DLTDefinition, DLTWindow,
    Regimen, RegimenComponent, DoseAdjustmentRule, DoseSeries,
)


def test_catalog_is_closed_and_aliases_cannot_extend_it():
    for name, entry in CATALOG.items():
        assert choices(name) == tuple(entry['options'])
        assert set(labels(name)) == set(choices(name))
        assert set(entry['aliases'].values()) <= set(choices(name))
        for code in choices(name):
            assert canonical_value(name, code) == code
        for code, label in labels(name).items():
            assert canonical_value(name, label) == code
    assert choices('sex_eligibility') == ('female', 'male', 'no_restriction')
    assert canonical_value('sex_eligibility', 'Female only') == 'female'
    assert canonical_value('sex_eligibility', 'not reported') is None
    assert canonical_value('clinical_metric', 'If a patient developed a DLT then reduce to 20 mg') is None
    assert canonical_value('clinical_metric', 'ECOG') == 'ecog'
    assert canonical_value('route', 'IV infusion') == 'intravenous'
    assert canonical_value('dose_unit', 'μg/kg') == 'micrograms/kg'


def test_json_schema_rejects_generated_sentences_in_category_slots():
    with pytest.raises(ValidationError):
        ConditionClause(clause_id='r', raw_text='If DLT resolves, resume', metric_code='If DLT resolves, resume')
    with pytest.raises(ValidationError):
        PopulationDescription(population_id='p', study_id='s', sex_eligibility='unknown')
    with pytest.raises(ValidationError):
        DosingPhase(dosing_phase_id='p', phase_label='Treatment', route='custom route')
    body = build_request_body(packet={'document_id':'D'}, prior=None, image_paths=[], model='test', reasoning_effort='low', max_output_tokens=1000)
    schema = body['text']['format']['schema']
    assert schema['properties']['schema_version']['enum'] == [SCHEMA_VERSION]
    sex = schema['$defs']['PopulationDescription']['properties']['sex_eligibility']['anyOf']
    assert sex[0]['enum'] == list(choices('sex_eligibility')) and sex[1] == {'type':'null'}
    route = schema['$defs']['DosingPhase']['properties']['route']['anyOf'][0]['enum']
    assert route == list(choices('route'))
    assert 'SHARED CLOSED CHOICE CATALOG' in body['instructions']


def test_missing_sex_and_complex_sex_are_not_no_restriction():
    source = ReportedExtraction(document_id='D', populations=[
        PopulationDescription(population_id='p', study_id='s'),
        PopulationDescription(population_id='q', study_id='s', sex_eligibility_raw='Not reported'),
        PopulationDescription(population_id='r', study_id='s', sex_eligibility_raw='Men or postmenopausal women'),
        PopulationDescription(population_id='t', study_id='s', sex_eligibility_raw='Both sexes'),
    ])
    enriched, _ = normalize_fields(source)
    assert [p.sex_eligibility for p in enriched.populations] == [None, None, None, 'no_restriction']
    assert enriched.populations[2].sex_eligibility_raw == 'Men or postmenopausal women'


def test_known_aliases_enrich_a_copy_without_editing_raw_or_guessing_numbers():
    source = ReportedExtraction(document_id='D', populations=[PopulationDescription(
        population_id='p', study_id='s', sex_eligibility_raw='Women only', age_min=18,
        age_min_comparator='≥', age_unit='yrs', modifiers=['metastatic', 'a newly described phenotype'],
        eligibility_conditions=[ConditionClause(clause_id='c', metric='ECOG',
            raw_text='Of 35 patients, all had good performance status.', category='performance status')])],
        study_parts=[StudyPart(part_id='i', study_id='s', label='Part one', phase_raw='Phase 1')])
    before = copy.deepcopy(source.model_dump())
    enriched, changes = normalize_fields(source)
    assert source.model_dump() == before and changes
    assert enriched.populations[0].sex_eligibility == 'female'
    assert enriched.populations[0].age_min_operator == '>='
    assert enriched.populations[0].age_unit_code == 'years'
    assert enriched.populations[0].disease_modifier_codes == ['metastatic','other']
    clause = enriched.populations[0].eligibility_conditions[0]
    assert clause.metric_code == 'ecog'
    assert clause.threshold is None and clause.comparator_code is None
    assert clause.raw_text == before['populations'][0]['eligibility_conditions'][0]['raw_text']
    assert enriched.study_parts[0].phase == 'phase_i'
    assert normalize_fields(enriched)[1] == []


def test_explicit_structured_codes_win_without_rewriting_the_source():
    source = ReportedExtraction(document_id='D', populations=[PopulationDescription(
        population_id='p', study_id='s', sex_eligibility_raw='Women', sex_eligibility='male',
        eligibility_conditions=[ConditionClause(clause_id='c', raw_text='Hb above 9 g/dL',
            metric='Hb', metric_code='hemoglobin', comparator='>', comparator_code='>=',
            threshold=10, unit_raw='g/dL', unit_code='g/L')])])
    enriched, _ = normalize_fields(source)
    condition = enriched.populations[0].eligibility_conditions[0]
    assert condition.comparator_code == '>=' and condition.threshold == 10 and condition.unit_code == 'g/L'
    assert condition.raw_text == 'Hb above 9 g/dL'
    assert enriched.populations[0].sex_eligibility == 'male'
    assert enriched.populations[0].sex_eligibility_raw == 'Women'


def test_dose_aliases_never_scale_numeric_values():
    dose = DoseExpression(raw_expression='500 μg/kg', value_kind='scalar', value=500,
        unit_raw='μg/kg', unit_normalized='mg/kg', denominator_basis='kg')
    source = ReportedExtraction(document_id='D', regimens=[Regimen(regimen_id='r', label='R', components=[
        RegimenComponent(component_id='a', intervention_id='a', dosing_phases=[
            DosingPhase(dosing_phase_id='p', phase_label='lead-in', dose=dose, route_raw='IV infusion', dose_form_raw='solution for infusion')])])],
        dose_series=[DoseSeries(series_id='s',study_id='s', label='s', regimen_id='r', varying_component_id='a', comparable_unit='mg/m²')])
    enriched, _ = normalize_fields(source)
    phase = enriched.regimens[0].components[0].dosing_phases[0]
    assert phase.phase_kind == 'lead_in' and phase.route == 'intravenous' and phase.dose_form == 'solution'
    assert phase.dose.unit_code == 'micrograms/kg' and phase.dose.value == 500
    assert phase.dose.unit_raw == 'μg/kg' and phase.dose.unit_normalized == 'mg/kg'
    assert phase.dose.basis_code == 'body_weight'
    assert enriched.dose_series[0].comparable_unit_code == 'mg/m2'


def test_window_grading_and_adjustment_trigger_preserve_full_statements():
    rule = DoseAdjustmentRule(rule_id='r', action='resume', condition_raw='If a DLT resolves, resume after 14 days',
        condition_structured=ConditionClause(clause_id='c', metric='DLT', raw_text='If a DLT resolves, resume after 14 days'))
    source = ReportedExtraction(document_id='D', regimens=[Regimen(regimen_id='r',label='R',dose_adjustment_rules=[rule])],
        dlt_windows=[DLTWindow(window_id='w',study_id='s',raw_definition='First cycle', start_raw='start of therapy', end_raw='end of cycle 1')],
        dlt_definitions=[DLTDefinition(definition_id='d', study_id='s', grading_system_raw='CTCAE version 5.0'),
                         DLTDefinition(definition_id='e', study_id='s', grading_system_raw='Publication 2020; Grade 3 toxicity')])
    enriched, _ = normalize_fields(source)
    assert enriched.dlt_definitions[0].grading_system == 'ctcae' and enriched.dlt_definitions[0].grading_version == '5.0'
    assert enriched.dlt_definitions[1].grading_system == 'other' and enriched.dlt_definitions[1].grading_version is None
    assert enriched.dlt_windows[0].start_anchor == 'first_treatment_dose' and enriched.dlt_windows[0].end_anchor == 'end_of_first_cycle'
    structured = enriched.regimens[0].dose_adjustment_rules[0].condition_structured
    assert structured.metric_code == 'dlt' and structured.threshold is None
    assert structured.raw_text == rule.condition_raw


def test_legacy_artifacts_still_parse_but_new_outputs_are_versioned():
    for version in ['phase1_extraction_v3','phase1_extraction_v3.2']:
        old = ReportedExtraction.model_validate({'schema_version':version,'document_id':'D'})
        assert old.schema_version == version
    assert ReportedExtraction(document_id='D').schema_version == 'phase1_extraction_v3.3'
    assert 'Set threshold' in load_prompt()


def test_age_comparisons_are_directional_without_breaking_legacy_prose():
    with pytest.raises(ValidationError):
        PopulationDescription(population_id='p', study_id='s', age_min_operator='<=')
    with pytest.raises(ValidationError):
        PopulationDescription(population_id='p', study_id='s', age_max_operator='>')
    source = ReportedExtraction(document_id='D', populations=[PopulationDescription(
        population_id='p', study_id='s', age_min=18, age_min_comparator='=', age_max=18, age_max_comparator='=')])
    normalized, _ = normalize_fields(source)
    assert normalized.populations[0].age_min_operator is None
    assert normalized.populations[0].age_max_operator is None
    assert normalized.populations[0].age_min_comparator == '='
    assert normalized.populations[0].age_max_comparator == '='
