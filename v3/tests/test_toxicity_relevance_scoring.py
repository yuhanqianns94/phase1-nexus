"""Administrative metadata must not increase evidence borrowed for toxicity.

Exercise both article comparison and the fixed drug mapping used for risk
transport. Preserve clinical differences and the unmodified source records.
"""
import copy

import pytest

from phase1_prediction.profile_similarity import (
    compare_mapped_profiles,
    compare_profiles,
    condition,
    condition_pair,
    drug_correspondence,
    drug_identity,
    normalize_weights,
    phase_pair,
)


def clinical_profile():
    return {
        "population": {
            "disease": ["advanced solid tumors"],
            "modifiers": ["advanced"],
            "sex": ["No restriction"],
            "age_min": 18.,
            "age_min_operator": ">=",
        },
        "eligibility": {
            "inclusion": [{"topic": "Creatinine clearance", "operator": ">=",
                           "threshold": 60., "unit": "mL/min"}],
            "exclusion": ["Uncontrolled infection"],
        },
        "configurations": [{
            "label": "Regimen A",
            "drugs": [{
                "name": "Drug A", "model_drug_id": "A", "kind": "drug",
                "mechanisms": ["EGFR inhibition"], "targets": ["EGFR"],
                "classes": ["kinase inhibitor"],
                "dose_basis": "daily_total", "formulation": "immediate release",
                "phases": [{"route": "oral", "frequency": "once daily",
                            "administration_days": [1, 2, 3, 4, 5],
                            "cycle": "28 days"}],
            }],
        }],
        "dlt": {
            "grading": "CTCAE", "grading_version": "5.0",
            "criteria": [{"topic": "Nonhematologic toxicity", "operator": ">=",
                          "threshold": 3., "unit": "grade"}],
        },
        "windows": [{"kind": "formal_dlt", "start": "first treatment dose", "days": 28.}],
    }


def contribution(result):
    """Factors used for borrowing, excluding display-only descriptions."""
    return result["similarity"]


def with_metadata(profile, variant):
    result = copy.deepcopy(profile)
    result["population"]["purpose"] = variant
    result["dlt"]["version"] = variant
    configuration = result["configurations"][0]
    configuration.update(label=variant, phase=variant, role=variant, status=variant)
    drug = configuration["drugs"][0]
    drug.update(research_role=variant, role=variant)
    drug["phases"][0]["phase"] = variant
    return result


@pytest.mark.parametrize("mode", ["pretrial", "full"])
def test_metadata_presence_or_agreement_cannot_change_borrowing(mode):
    base = clinical_profile()
    first, other = (with_metadata(base, value) for value in ("Study information A", "Study information B"))
    expected = contribution(compare_profiles(base, base, mode=mode))
    for left, right in [(base, first), (first, first), (first, other)]:
        assert contribution(compare_profiles(left, right, mode=mode)) == pytest.approx(expected)


def test_fixed_drug_mapping_also_ignores_metadata_and_preserves_sources():
    left, right = with_metadata(clinical_profile(), "First label"), with_metadata(clinical_profile(), "Second label")
    before = copy.deepcopy((left, right))
    result = compare_mapped_profiles(left, right, {"A": "A"})
    expected = compare_mapped_profiles(clinical_profile(), clinical_profile(), {"A": "A"})
    assert contribution(result) == pytest.approx(contribution(expected))
    assert (left, right) == before


def test_entity_kind_is_identity_qualification_not_a_second_similarity_bonus():
    left, right = clinical_profile(), clinical_profile()
    right["configurations"][0]["drugs"][0].pop("kind")
    assert contribution(compare_profiles(left, right)) == pytest.approx(contribution(compare_profiles(left, left)))
    assert drug_identity({"name": "X", "kind": "drug"}, {"name": "X", "kind": "drug_class"}) == 0.


def test_metadata_never_establishes_a_drug_match():
    left = {"name": "A", "research_role": "primary", "role": "escalated"}
    right = {"name": "B", "research_role": "primary", "role": "escalated"}
    assert not drug_correspondence(left, right)["eligible"]
    assert drug_correspondence(left, {"name": "A", "research_role": "concomitant"})["eligible"]
    left["targets"] = right["targets"] = ["EGFR"]
    assert drug_correspondence(left, right)["eligible"]


@pytest.mark.parametrize("metadata", [None, "Treatment", "Unrelated stage name"])
def test_stage_name_does_not_increase_phase_score(metadata):
    left = {"route": "oral", "frequency": "once daily", "cycle": "28 days"}
    right = {**left, "cycle": "21 days"}
    expected = phase_pair(left, right)
    actual = phase_pair({**left, "phase": metadata}, {**right, "phase": "Treatment"})
    assert actual["score"] == pytest.approx(expected["score"])
    assert actual["score"] < 1.


def test_structure_weight_cannot_be_enabled_through_custom_or_legacy_weights():
    assert normalize_weights({"population": 1., "structure": 1000.})["structure"] == 0.
    assert normalize_weights({"population": 1., "structure": 1000.})["population"] == 1.
    legacy = normalize_weights({"cancer_type": 1., "trial_type": 1000., "schema_embedding": 1000.})
    assert legacy["structure"] == 0. and legacy["population"] == 1.
    left, right = clinical_profile(), with_metadata(clinical_profile(), "Different structure")
    expected = compare_profiles(left, right, weights={"population": 1.})
    actual = compare_profiles(left, right, weights={"population": 1., "structure": 1000.})
    assert contribution(actual) == pytest.approx(contribution(expected))


@pytest.mark.parametrize("administrative", [
    "Ability to provide written informed consent",
    "Written informed consent must be obtained before enrollment",
])
def test_pure_consent_requirement_does_not_change_similarity(administrative):
    left, right = clinical_profile(), clinical_profile()
    expected = contribution(compare_profiles(left, right))
    right["eligibility"]["inclusion"].append(administrative)
    assert contribution(compare_profiles(left, right)) == pytest.approx(expected)
    left["eligibility"]["inclusion"].append(administrative)
    assert contribution(compare_profiles(left, right)) == pytest.approx(expected)


def test_mixed_consent_and_clinical_requirement_is_not_discarded():
    left = {"eligibility": {"inclusion": [{
        "raw": "Written informed consent and creatinine clearance >= 60 mL/min",
        "topic": "Creatinine clearance", "operator": ">=", "threshold": 60., "unit": "mL/min",
    }]}}
    right = copy.deepcopy(left)
    right["eligibility"]["inclusion"][0].update(
        raw="Written informed consent and creatinine clearance >= 90 mL/min", threshold=90.)
    same = compare_profiles(left, left, weights={"eligibility": 1.})
    different = compare_profiles(left, right, weights={"eligibility": 1.})
    assert same["similarity"] == 1.
    assert different["similarity"] is not None and different["similarity"] < 1.


def test_administrative_and_wrapper_does_not_change_clinical_eligibility():
    left = {"eligibility": {"inclusion": [{
        "clause_id": "renal", "topic": "Creatinine clearance", "operator": ">=",
        "threshold": 60., "unit": "mL/min",
    }]}}
    right = copy.deepcopy(left)
    right["eligibility"]["inclusion"].append({
        "clause_id": "consent", "raw": "Ability to provide written informed consent",
    })
    right["eligibility"]["logic_groups"] = [{
        "group_id": "entry", "operator": "AND", "children": ["renal", "consent"],
    }]
    before = copy.deepcopy(right)
    expected = contribution(compare_profiles(left, left, weights={"eligibility": 1.}))
    assert contribution(compare_profiles(left, right, weights={"eligibility": 1.})) == pytest.approx(expected)
    assert right == before


def test_simple_raw_predicate_is_not_an_additional_score_source():
    left = {"topic": "ANC", "operator": ">=", "threshold": 1500., "unit": "/uL"}
    right = {**left, "threshold": 2000.}
    expected = condition_pair(condition(left), condition(right))
    actual = condition_pair(condition({**left, "raw": "ANC >= 1500 /uL"}),
                            condition({**right, "raw": "ANC >= 2000 /uL"}))
    assert actual["score"] == pytest.approx(expected["score"])
    assert actual["score"] < 1.


def test_clinical_exceptions_in_raw_text_are_not_removed_with_redundant_words():
    base = {"topic": "Nonhematologic toxicity", "operator": ">=", "threshold": 3., "unit": "grade"}
    left = condition({**base, "raw": "Grade >= 3 nonhematologic toxicity except reversible nausea"})
    right = condition({**base, "raw": "Grade >= 3 nonhematologic toxicity except reversible diarrhea"})
    assert condition_pair(left, right)["score"] < condition_pair(left, left)["score"]


@pytest.mark.parametrize("clinical_change,category", [
    ("renal", "eligibility"), ("schedule", "administration"),
    ("grading", "dlt"), ("window", "window"),
])
def test_clinically_relevant_differences_still_reduce_similarity(clinical_change, category):
    left, right = clinical_profile(), clinical_profile()
    if clinical_change == "renal":
        right["eligibility"]["inclusion"][0]["threshold"] = 90.
    elif clinical_change == "schedule":
        right["configurations"][0]["drugs"][0]["phases"][0]["frequency"] = "twice daily"
    elif clinical_change == "grading":
        right["dlt"]["grading_version"] = "4.03"
    else:
        right["windows"][0]["days"] = 7.
    assert compare_profiles(left, right)["groups"][category]["score"] < compare_profiles(left, left)["groups"][category]["score"]
