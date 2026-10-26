import copy
from types import SimpleNamespace

import pytest

from phase1_prediction.clinical_coverage import clinical_coverage, clinical_fields
from phase1_prediction.multidrug import mapping, service
from phase1_prediction.multidrug.power_prior import target_signature
from test_multidrug_transport import fixture


def full_profile():
    return {'population': {'disease': ['glioma']},
            'eligibility': {'inclusion': ['ECOG <= 2']},
            'configurations': [{'phase': 'Phase I', 'drugs': [
                {'name': 'A', 'mechanisms': ['kinase inhibition'],
                 'phases': [{'route': 'oral', 'frequency': 'daily'}]}]}],
            'dlt': {'criteria': ['Grade >= 3 neutropenia']},
            'windows': [{'kind': 'formal_dlt', 'days': 28}]}


def test_coverage_uses_shared_field_types_and_all_six_category_weights():
    left = full_profile()
    before = copy.deepcopy(left)
    assert clinical_coverage(left, left)['coverage'] == pytest.approx(1.)
    right = copy.deepcopy(left)
    right['population']['age_min'] = 18
    result = clinical_coverage(left, right)
    assert result['groups']['population']['coverage'] == .5
    assert result['coverage'] == pytest.approx(11/12)
    assert clinical_coverage(right, left) == result
    right['population']['disease'] = ['lymphoma']
    assert clinical_coverage(left, right) == result  # Values do not score at this stage.
    assert clinical_coverage({}, {})['coverage'] == 0.
    assert clinical_coverage(left, right, weights={'population': 1})['coverage'] == .5
    assert left == before


def test_outcomes_provenance_and_placeholders_cannot_supply_coverage():
    profile = {'population': {'baseline': ['all patients responded'], 'disease': ['unknown']},
        'eligibility': {'inclusion': [{'clause_id': 'x', 'raw_text': 'not reported'}]},
        'dlt': {'observed': ['Grade 3 toxicity'], 'criteria': [{'evidence': 'source text'}]},
        'configurations': [{'label': 'Arm A', 'phase': 'Phase I', 'drugs': [
            {'model_drug_id': 'A', 'role': 'primary', 'dose': 100,
             'mechanisms': ['unknown'], 'phases': [{'phase': 'Induction', 'dose': 100}]}]}]}
    assert all(not fields for fields in clinical_fields(profile).values())
    assert clinical_coverage(profile, profile)['coverage'] == 0.
    consent = {'eligibility': {'inclusion': ['Written informed consent']}}
    assert clinical_coverage(consent, consent)['coverage'] == 0.


def test_gate_precedes_similarity_and_embeddings_even_with_manual_mapping(monkeypatch):
    scope, _, _, target = fixture(1)
    target.minimum_clinical_coverage = .20
    target.mapping_overrides = {scope.scope_id: {'A': 'A'}}
    def forbidden(*args, **kwargs):
        raise AssertionError('Similarity must not run below coverage threshold')
    monkeypatch.setattr(mapping, 'drug_affinity', forbidden)
    monkeypatch.setattr(mapping, 'drug_correspondence', forbidden)
    monkeypatch.setattr(mapping, 'compare_mapped_profiles', forbidden)
    result = mapping.match_target(target, scope)
    assert result['comparison_skipped'] == 'insufficient_clinical_coverage'
    assert result['clinical_coverage']['coverage'] == pytest.approx(1/6)
    assert result['score']['similarity'] is None and not result['pairs']


def test_gate_boundary_and_threshold_are_part_of_frozen_signature():
    scope, _, _, target = fixture(1)
    target.minimum_clinical_coverage = 1/6
    signature = target_signature(target)
    accepted = mapping.match_target(target, scope)
    assert not accepted.get('comparison_skipped')
    assert accepted['pairs'] == {'A': 'A'}
    assert accepted['score']['similarity'] == pytest.approx(1.)
    target.minimum_clinical_coverage = .2
    assert target_signature(target) != signature
    assert mapping.match_target(target, scope)['comparison_skipped']


def test_selection_skips_insufficient_information_before_top_n(monkeypatch):
    sparse, _, _, target = fixture(1)
    sparse.document_id = 'first_sparse'
    complete = sparse.model_copy(deep=True)
    complete.document_id, complete.scope_id = 'second_complete', 'complete'
    complete.profile = full_profile()
    complete.drugs[0].profile = complete.profile['configurations'][0]['drugs'][0]
    target.profile = full_profile()
    target.drugs[0].profile = copy.deepcopy(complete.drugs[0].profile)
    target.minimum_clinical_coverage = .8
    target.minimum_drug_similarity = 0.
    monkeypatch.setattr(service, 'load_model_library', lambda *args, **kwargs:
        ([sparse, complete], {'notices': []}))
    library = SimpleNamespace(source_roots=[], evidence_policy='project_default', review_policy='extracted')
    result = service.preview_planned(library, target.model_dump(), top_articles=1)
    assert [row['document_id'] for row in result['selected_series']] == ['second_complete']
    assert result['selected_series'][0]['similarity'] == pytest.approx(1.)
    assert result['diagnostics']['notices'][0]['reason'] == 'insufficient_clinical_coverage'


def test_focal_matching_keeps_companion_information_for_coverage():
    scope, _, _, target = fixture(2)
    target.drugs[0].profile.pop('mechanisms')
    expected = clinical_coverage(mapping.target_profile(target), scope.profile)
    focal = service._focal_matching_target(target)
    assert clinical_coverage(mapping.coverage_target_profile(focal), scope.profile) == expected
