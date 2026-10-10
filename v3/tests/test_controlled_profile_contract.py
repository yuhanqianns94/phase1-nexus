"""Closed extraction categories and Plan values describe the same evidence."""
import copy

import pytest

from phase1_extraction.models import (ReportedExtraction, PopulationDescription,
    DosingPhase, DoseExpression, CohortExposure, DoseAdjustmentRule, ConditionClause)
from phase1_prediction.profiles import extraction_profile, phase, adjustment
from phase1_prediction.profile_similarity import (condition, conditions,
    compare_profiles, phase_pair, window_pair)
from phase1_prediction.multidrug.observations import _numbers
from phase1_prediction.multidrug.mapping import _unit_scale


def test_explicit_categories_win_without_mutating_source_words():
    source=ReportedExtraction(document_id='x',populations=[PopulationDescription(
        population_id='p',study_id='s',sex_eligibility_raw='Male',sex_eligibility='female',
        age_min=24,age_unit='years',age_unit_code='months',age_min_operator='>',
        modifiers=['metastatic'],disease_modifier_codes=['recurrent'])])
    before=copy.deepcopy(source.model_dump())
    result=extraction_profile(source)['population']
    assert result['sex']==['Female']
    assert result['age_min']==2
    assert result['age_min_operator']=='>'
    assert result['modifiers']==['Recurrent']
    assert source.model_dump()==before


def test_missing_sex_does_not_imply_unrestricted_eligibility():
    source=ReportedExtraction(document_id='x',populations=[PopulationDescription(population_id='p',study_id='s')])
    assert extraction_profile(source)['population']['sex']==[]
    left={'population':{'sex':['No restriction']}}
    unknown={'population':{'sex':[]}}
    r=compare_profiles(left,unknown,weights={'population':1})
    assert r['similarity'] is None
    assert compare_profiles(left,{'population':{'sex':['both sexes']}},weights={'population':1})['similarity']==1


def test_route_stage_and_form_codes_reach_same_comparison_as_plan():
    source=DosingPhase(dosing_phase_id='p',phase_label='initial induction period',
        phase_kind='induction',route_raw='infusion through a peripheral line',
        route='intravenous',dose_form_raw='lyophilized powder',dose_form='powder')
    before=source.model_dump()
    actual=phase(source)
    planned={'phase':'Induction','route':'IV','form':'Powder'}
    assert phase_pair(actual,planned)['score']==1
    assert source.model_dump()==before


def test_other_topics_remain_distinct_prose_not_an_identical_other_bucket():
    first=condition({'raw_text':'rare condition A','metric_code':'other','metric':'rare condition A'})
    second=condition({'raw_text':'different illness B','metric_code':'other','metric':'different illness B'})
    assert first['topic']=='rare condition A'
    result=conditions([first],[second])
    assert result['score'] is None and 'coverage' not in result


def test_closed_predicates_override_prose_and_do_not_invent_thresholds():
    item=condition({'raw_text':'ANC < 500 cells/uL','metric_code':'hemoglobin',
        'comparator_code':'>=','threshold':9.,'unit_code':'g/dL'})
    assert (item['topic'],item['operator'],item['threshold'],item['unit'])==('Hemoglobin','>=',9.,'g/dL')
    prose=condition({'raw_text':'A history of 2 previous regimens; study 2014'})
    assert prose['threshold'] is None and prose['operator'] is None


def test_structured_adjustment_trigger_and_action_both_reach_comparison():
    rule=DoseAdjustmentRule(rule_id='r',action='reduce',condition_raw='DLT occurs',
        action_raw='Reduce the dose',condition_structured=ConditionClause(
            clause_id='c',raw_text='DLT occurs',metric_code='dlt'))
    source=adjustment(rule)
    assert source['topic']=='DLT' and source['action']=='reduce'
    other={**source,'action':'resume'}
    assert conditions([source],[other])['score']==0


def test_age_comparators_are_compared_without_fabricated_defaults():
    a={'population':{'age_min':18.,'age_min_operator':'>='}}
    b={'population':{'age_min':18.,'age_min_operator':'>'}}
    assert compare_profiles(a,b,weights={'population':1})['similarity']<1


def test_window_codes_normalize_known_anchors():
    a={'kind':'formal_dlt','start':'from therapy commencement','start_anchor':'first_treatment_dose','days':28.}
    b={'kind':'formal_dlt','start':'First treatment dose','days':28.}
    assert window_pair(a,b)['score']==1


@pytest.mark.parametrize('anchor,a,b',[
    ('first_specified_drug_dose','First dose of drug A','First dose of drug B'),
    ('start_of_stage','Start of induction','Start of maintenance'),
    ('end_of_specified_cycle','End of cycle 1','End of cycle 3')])
def test_window_anchor_category_does_not_erase_qualifiers(anchor,a,b):
    left={'kind':'formal_dlt','start':a,'start_anchor':anchor,'days':28.}
    right={**left,'start':b}
    assert window_pair(left,right)['score']<window_pair(left,left)['score']


def test_other_grading_versions_do_not_collapse_to_one_category():
    from phase1_extraction.models import DLTDefinition
    def profile(version):
        return extraction_profile(ReportedExtraction(document_id='x',dlt_definitions=[DLTDefinition(
            definition_id='d',study_id='s',grading_system='ctcae',grading_version='other',
            grading_system_raw='CTCAE version '+version)]))
    first,second=profile('7.1'),profile('8.0')
    assert first['dlt']['grading_version']=='7.1'
    assert compare_profiles(first,second,weights={'dlt':1})['similarity']<1


@pytest.mark.parametrize('unit,code',[('µg','micrograms'),('ug',None),('micrograms','micrograms')])
def test_source_units_remain_bound_to_the_reported_number(unit,code):
    dose=DoseExpression(raw_expression='1000 micrograms',value_kind='scalar',value=1000.,
                        unit_raw=unit,unit_code=code,unit_normalized='mg')
    exposure=CohortExposure(exposure_id='e',name_raw='A',presence='present',dose_status='reported',dose=dose)
    value,converted,*_=_numbers(exposure)
    assert (value,converted)==(1.,'mg')
    assert _unit_scale('mg','micrograms')==.001


def test_conflicting_closed_dose_unit_cannot_relabel_source_number():
    dose=DoseExpression(raw_expression='1000 micrograms',value_kind='scalar',value=1000.,unit_raw='ug',unit_code='mg')
    exposure=CohortExposure(exposure_id='e',name_raw='A',presence='present',dose_status='reported',dose=dose)
    with pytest.raises(ValueError,match='conflicts'):_numbers(exposure)


def test_supported_cell_count_units_compare_in_the_same_dimension():
    from phase1_prediction.profile_similarity import canonical_threshold
    assert canonical_threshold(1.5,'10^9/L')==canonical_threshold(1500.,'/uL')


def test_same_entity_dose_conversion_rejects_conflicting_frequency_patterns():
    from types import SimpleNamespace
    from phase1_prediction.multidrug.mapping import mapped_relative_dose
    def drug(pattern):
        return SimpleNamespace(name='A',profile={'name':'A','phases':[{'structured':{'frequency_pattern':pattern}}]},
            transport_mode='same_entity_unit_conversion',mode='numeric',allow_background_anchor=False,
            unit='mg',reference_dose=10.)
    with pytest.raises(ValueError,match='frequency patterns'):
        mapped_relative_dose(drug('continuous'),drug('intermittent'),10.)
