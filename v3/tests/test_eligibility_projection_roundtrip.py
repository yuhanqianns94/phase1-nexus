"""Source eligibility predicates and scoped Boolean groups survive projection."""
import copy
import json
from types import SimpleNamespace

import pytest

from phase1_extraction.clinical_protocol import validate_logic
from phase1_prediction.evidence_relevance import toxicity_eligibility
from phase1_prediction.multidrug.contracts import TargetRegimen
from phase1_prediction.multidrug.mapping import match_target
from phase1_prediction.multidrug.observations import load_model_library
from phase1_prediction.profiles import extraction_profile
from phase1_prediction.profile_similarity import all_texts, condition, conditions, compare_profiles
from phase1_web.multidrug_examples import target_from_scope
from test_friendly_clinical_form import _draft, _form_app
from test_prediction_pipeline import _article


def _population():
    return {
        'population_id':'pop','study_id':'study',
        'inclusion_criteria':['Adequate kidney function'],
        'exclusion_criteria':['Prior treatment with an HDAC inhibitor'],
        'eligibility_conditions':[
            {'clause_id':'renal','raw_text':'Creatinine clearance >= 60 mL/min',
             'metric':'Creatinine clearance','comparator':'>=','threshold':60.,'unit_raw':'mL/min'},
            {'clause_id':'prior','raw_text':'Patients were excluded if previously treated with an HDAC inhibitor',
             'metric':'Prior therapy','negated':True,'timing_raw':'before enrollment'},
        ],
        'eligibility_logic_groups':[
            {'group_id':'requirements','operator':'AND','children':['renal','prior']},
        ],
    }


def _clauses(section):
    return [clause for field in ('inclusion','exclusion','unclassified') for clause in section.get(field,[])]


def test_nonmatching_structured_statements_keep_ids_predicates_and_unknown_polarity():
    source={'populations':[_population()]}
    before=copy.deepcopy(source)
    section=extraction_profile(source)['eligibility']
    assert [c['raw'] for c in section['inclusion']]==['Adequate kidney function']
    assert [c['raw'] for c in section['exclusion']]==['Prior treatment with an HDAC inhibitor']
    assert [c['clause_id'] for c in section['unclassified']]==['renal','prior']
    assert section['unclassified'][0]['threshold']==60.
    assert section['unclassified'][1]['negated'] is True
    assert section['unclassified'][1]['timing']=='before enrollment'
    assert section['logic_groups']==source['populations'][0]['eligibility_logic_groups']
    validate_logic(section['logic_groups'],{c['clause_id'] for c in _clauses(section) if c.get('clause_id')})
    assert source==before


def test_only_unique_exact_source_sections_assign_inclusion_or_exclusion():
    pop=_population()
    pop['inclusion_criteria']=[pop['eligibility_conditions'][0]['raw_text']]
    pop['exclusion_criteria']=[pop['eligibility_conditions'][1]['raw_text']]
    section=extraction_profile({'populations':[pop]})['eligibility']
    assert 'unclassified' not in section
    assert section['inclusion'][0]['clause_id']=='renal'
    assert section['exclusion'][0]['clause_id']=='prior'
    pop['exclusion_criteria'].append(pop['inclusion_criteria'][0])
    ambiguous=extraction_profile({'populations':[pop]})['eligibility']
    assert [c['clause_id'] for c in ambiguous['unclassified']]==['renal']
    assert not ambiguous['inclusion'][0]['clause_id']
    assert not ambiguous['exclusion'][1]['clause_id']


@pytest.mark.parametrize('scope_field,excluded',[
    ('applies_to_part_ids','other_part'),('applies_to_arm_ids','other_arm'),
])
@pytest.mark.parametrize('operator',['AND','OR'])
def test_partial_nested_scope_keeps_whole_source_tree_outside_active_logic(scope_field,excluded,operator):
    pop=_population()
    pop['eligibility_conditions'][1][scope_field]=[excluded]
    pop['eligibility_logic_groups']=[
        {'group_id':'nested','operator':operator,'children':['renal','prior']},
        {'group_id':'root','operator':'AND','children':['nested','renal']},
        {'group_id':'global','operator':'AND','children':['renal']},
    ]
    source={'populations':[pop]}
    before=copy.deepcopy(source)
    row=SimpleNamespace(study_id='study',part_id='part_i',arm_id='arm')
    section=extraction_profile(source,row)['eligibility']
    assert [c['clause_id'] for c in section['unclassified']]==['renal']
    assert section['logic_groups']==[pop['eligibility_logic_groups'][2]]
    preserved=section['unresolved_source_logic'][0]
    assert preserved['logic_groups']==pop['eligibility_logic_groups'][:2]
    assert preserved['conditions']==pop['eligibility_conditions']
    validate_logic(section['logic_groups'],{'renal'})
    global_section=extraction_profile(source)['eligibility']
    assert global_section['logic_groups']==pop['eligibility_logic_groups']
    assert 'unresolved_source_logic' not in global_section
    assert source==before


@pytest.mark.parametrize('partial_scope',[False,True])
def test_extracted_example_validates_and_matches_itself_after_form_roundtrip(tmp_path,partial_scope):
    paths=_article(tmp_path,'PMC123',rows=[(.1,3,0),(.3,6,2)])
    extraction=json.loads(paths.reported_extraction.read_text())
    extraction['populations'][0].update(_population())
    if partial_scope:
        extraction['populations'][0]['eligibility_conditions'][1]['applies_to_part_ids']=['other_part']
    paths.reported_extraction.write_text(json.dumps(extraction))
    original=paths.reported_extraction.read_bytes()
    scopes,_=load_model_library([tmp_path])
    scope=scopes[0]
    target=target_from_scope(scope)
    expected=copy.deepcopy(target.profile['eligibility'])
    form=_form_app(target.model_dump(mode='json'))
    result=TargetRegimen.model_validate(form.session_state['result'])
    assert result.profile['eligibility']['unclassified']==expected['unclassified']
    assert result.profile['eligibility']['logic_groups']==expected['logic_groups']
    if partial_scope:
        assert result.profile['eligibility']['unresolved_source_logic']==expected['unresolved_source_logic']
        assert any('cannot be fully resolved' in info.value for info in form.info)
    assert all(value==1. for value in match_target(result,scope)['score']['field_scores'].values())
    form.number_input(key='plan_elig_unclassified_0_threshold').set_value(75.).run()
    edited=TargetRegimen.model_validate(form.session_state['result'])
    assert edited.profile['eligibility']['unclassified'][0]['clause_id']=='renal'
    assert edited.profile['eligibility']['unclassified'][0]['threshold']==75.
    assert match_target(edited,scope)['score']['field_scores']['eligibility']<1.
    assert paths.reported_extraction.read_bytes()==original


def test_neutral_administrative_condition_is_filtered_without_losing_clinical_logic():
    source={'unclassified':[
        {'clause_id':'consent','raw':'Written informed consent'},
        {'clause_id':'renal','raw':'Creatinine clearance >= 60 mL/min'},
    ],'logic_groups':[{'group_id':'all','operator':'AND','children':['consent','renal']}]}
    before=copy.deepcopy(source)
    assert toxicity_eligibility(source)=={'unclassified':[source['unclassified'][1]],'logic_groups':[]}
    assert source==before


def test_unresolved_source_metadata_never_requests_embeddings():
    profile={'eligibility':{'unclassified':[{'raw':'Active renal requirement'}],
             'unresolved_source_logic':[{'conditions':[{'raw_text':'Unrelated historical source phrase'}],
                                         'reason':'Excluded study part metadata'}]}}
    assert all_texts([profile])==['active renal requirement']


def test_legacy_clause_without_id_is_preserved_without_inventing_a_group_link():
    pop=_population()
    pop['eligibility_conditions'][0].pop('clause_id')
    section=extraction_profile({'populations':[pop]})['eligibility']
    assert section['unclassified'][0]['clause_id'] is None
    assert section['unclassified'][0]['threshold']==60.
    assert section['logic_groups']==[]
    assert section['unresolved_source_logic'][0]['logic_groups']==pop['eligibility_logic_groups']


def _repeated_topic_exceptions():
    return [condition('Febrile neutropenia or infection was excepted from the grade 3 criterion.'),
            condition('Febrile neutropenia/infection was excepted from the grade 4-or-higher criterion.')]


def test_repeated_topic_assignment_ignores_ids_and_input_order():
    source=_repeated_topic_exceptions()
    before=copy.deepcopy(source)
    identified=[dict(source[0],clause_id='z'),dict(source[1],clause_id='a')]
    for reordered in (identified,list(reversed(identified))):
        assert conditions(source,reordered)['score']==1.
        assert conditions(reordered,source)['score']==1.
    assert source==before
    assert [c['clause_id'] for c in identified]==['z','a']


def test_repeated_topic_assignment_ignores_equivalent_structured_annotations():
    source=[condition('ANC >= 1500 /uL'),condition('ANC >= 500 /uL')]
    annotated=copy.deepcopy(source)
    annotated[1]['metric_code']='anc'
    before=copy.deepcopy(annotated)
    for reordered in (annotated,list(reversed(annotated))):
        assert conditions(source,reordered)['score']==1.
        assert conditions(reordered,source)['score']==1.
    assert annotated==before


def test_form_generated_ids_do_not_change_dlt_correspondence():
    draft=_draft()
    draft['profile']['dlt']={
        'criteria':[dict(condition('Grade 3 renal toxicity'),clause_id='renal')],
        'exceptions':_repeated_topic_exceptions(),
        'logic_groups':[{'group_id':'dlt','operator':'AND','children':['renal']}],
    }
    before=copy.deepcopy(draft)
    form=_form_app(draft)
    result=form.session_state['result']
    assert all(c['clause_id'] for c in result['profile']['dlt']['exceptions'])
    assert result['profile']['dlt']['logic_groups']==draft['profile']['dlt']['logic_groups']
    assert compare_profiles(draft['profile'],result['profile'])['field_scores']['dlt']==1.
    assert draft==before
