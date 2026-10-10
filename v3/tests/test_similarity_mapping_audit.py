"""Regression cases from the current similarity/mapping implementation audit."""
import copy
import json

import pytest

from phase1_prediction.profile_similarity import (
    _TEXT_VECTORS, admin_pair, compare_profiles, condition, condition_pair,
    conditions, text_score,
)
from phase1_prediction.similarity_engine import DEFAULT_MODEL, NORMALIZATION_VERSION, similarity_context


@pytest.mark.parametrize('stage_name', ['treatment', None])
def test_repeated_or_unnamed_stages_align_by_administration(stage_name):
    left={'phases':[{'phase':stage_name,'route':'oral','cycle':'21 days'},
                    {'phase':stage_name,'route':'intravenous','cycle':'28 days'}]}
    right=copy.deepcopy(left)
    right['phases'][0]['cycle']='35 days'
    result=admin_pair(left,right)
    pairs=[r for r in result['details'] if r['indicator']=='Dosing phase']
    assert all(next(d for d in row['details'] if d['indicator']=='route')['score']==1. for row in pairs)
    assert result['score']>.8
    assert result['score']==admin_pair(right,left)['score']
    right['phases'].reverse()
    assert result['score']==admin_pair(left,right)['score']


def test_explicit_stage_names_still_take_priority_over_schedule_affinity():
    left={'phases':[{'phase':'induction','route':'oral'}, {'phase':'maintenance','route':'intravenous'}]}
    right={'phases':[{'phase':'induction','route':'intravenous'}, {'phase':'maintenance','route':'oral'}]}
    pairs=[r for r in admin_pair(left,right)['details'] if r['indicator']=='Dosing phase']
    # Names still determine which clinical schedules correspond, but are not
    # a separate bonus for toxicity similarity or information coverage.
    assert all(not any(d['indicator']=='phase' for d in row['details']) for row in pairs)
    assert all(next(d for d in row['details'] if d['indicator']=='route')['score']==0. for row in pairs)


def test_raw_only_conditions_are_normalized_without_mutating_the_profile():
    profile={'eligibility':{'inclusion':[{'raw':'Age >= 18 years','source_id':'s1'}]},
             'dlt':{'criteria':['Grade 3 toxicity']}}
    original=copy.deepcopy(profile)
    result=compare_profiles(profile,profile)
    assert result['groups']['eligibility']['score']==1.
    assert result['groups']['dlt']['score']==1.
    assert profile==original


@pytest.mark.parametrize('raw', ['grade 3 toxicity','grade >= 3 toxicity'])
def test_grade_parsing_preserves_explicit_structured_values(raw):
    result=condition({'raw':raw,'threshold':4.,'operator':'>','unit':'grade','timing':'cycle 1'})
    assert (result['threshold'],result['operator'],result['unit'],result['timing'])==(4.,'>','grade','cycle 1')
    assert condition(raw)['unit']=='grade'


@pytest.mark.parametrize('left,right', [
    ('Grade 3 toxicity (CTCAE)','Grade 3 toxicity CTCAE'),
    ('Grade 3 toxicity (CTCAE) and persistent fever','Grade 3 toxicity CTCAE and persistent fever'),
])
def test_parenthesized_annotations_do_not_become_boolean_conflicts(left,right):
    result=conditions([left],[right])
    assert result['score']==1.


def test_actual_boolean_grouping_conflicts_are_still_rejected():
    assert conditions(['(A and B) or C'],['A and (B or C)'])['score']==0.


@pytest.mark.parametrize('left,right,duplicate', [
    ('grade 3 toxicity','grade 4 toxicity','Grade'),
    ('toxicity > 7 days','toxicity > 14 days','Duration'),
])
def test_one_numeric_predicate_is_not_counted_twice(left,right,duplicate):
    result=condition_pair(condition(left),condition(right))
    assert duplicate not in [row['indicator'] for row in result['details']]
    assert next(row for row in result['details'] if row['indicator']=='threshold')['score']<1.


def test_grade_and_duration_remain_distinct_predicates():
    a=condition('grade >= 3 toxicity > 7 days')
    b=condition('grade >= 3 toxicity > 14 days')
    result=condition_pair(a,b)
    assert next(row for row in result['details'] if row['indicator']=='threshold')['score']==1.
    assert next(row for row in result['details'] if row['indicator']=='Duration')['score']<1.


@pytest.mark.parametrize('vector', [[float('nan'),1.], [float('inf'),1.], [0.,0.], [], [[1.,0.]], [1.,0.,0.]])
def test_invalid_embedding_vectors_fail_instead_of_matching_perfectly(vector):
    token=_TEXT_VECTORS.set({'alpha':vector,'beta':[1.,0.]})
    try:
        with pytest.raises(ValueError,match='no local fallback'):text_score('alpha','beta')
    finally:_TEXT_VECTORS.reset(token)


def test_invalid_api_vectors_are_not_saved(tmp_path,monkeypatch):
    from phase1_prediction.reference import toxicity_similarity
    monkeypatch.setattr(toxicity_similarity,'embed_texts_openai',lambda texts,model:[[float('nan'),1.] for _ in texts])
    cache=tmp_path/'embeddings.json'
    with pytest.raises(ValueError,match='no local fallback'):
        with similarity_context('openai',cache=cache):text_score('alpha','beta')
    assert not cache.exists()
    assert text_score('alpha','beta')==0.


def test_invalid_cached_vector_fails_without_requesting_external_api(tmp_path,monkeypatch):
    from phase1_prediction.reference import toxicity_similarity
    cache=tmp_path/'embeddings.json'
    items={f'{NORMALIZATION_VERSION}:{DEFAULT_MODEL}:{toxicity_similarity.hash_text(text)}':
           {'vector':vector,'model':DEFAULT_MODEL} for text,vector in [('alpha',[0.,0.]),('beta',[1.,0.])]}
    cache.write_text(json.dumps({'items':items}))
    original=cache.read_bytes()
    monkeypatch.setattr(toxicity_similarity,'embed_texts_openai',lambda *args:pytest.fail('A cached-vector validation must not call the API'))
    with pytest.raises(ValueError,match='no local fallback'):
        with similarity_context('openai',cache=cache):text_score('alpha','beta')
    assert cache.read_bytes()==original
