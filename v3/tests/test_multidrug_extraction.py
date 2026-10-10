from phase1_extraction.drug_ordering import apply_drug_ordering, clinical_payload, ordered_catalog
from phase1_extraction.exposures import exposure_identity
from phase1_extraction.models import DoseExpression
from phase1_extraction.resolver import resolve_extraction
from phase1_prediction.multidrug.observations import build_article_scopes
from test_cohort_exposures import extraction, exposure, write_article


def test_rank_only_does_not_change_clinical_identity():
    e = extraction()
    ordered = apply_drug_ordering(e)
    assert clinical_payload(ordered) == clinical_payload(e)
    assert exposure_identity(ordered.cohorts[0].exposures) == exposure_identity(e.cohorts[0].exposures)
    assert ordered.cohorts[0].exposures[0].research_priority == 1
    assert ordered.cohorts[1].exposures[1].research_priority == 2


def test_order_is_regimen_wide_and_permutation_invariant():
    e = extraction()
    e.cohorts[0].exposures.append(exposure('Fourth', component='fourth', value=5))
    catalog = ordered_catalog(e, 'regimen')
    for c in e.cohorts:
        c.exposures.reverse()
    e.cohorts.reverse()
    assert ordered_catalog(e, 'regimen') == catalog
    ordered=apply_drug_ordering(e)
    assert [x.research_priority for x in ordered.cohorts[0].exposures] == sorted(x.research_priority for x in ordered.cohorts[0].exposures)
    assert clinical_payload(ordered)==clinical_payload(e)
    assert apply_drug_ordering(ordered)==ordered


def test_unknown_main_dose_does_not_drop_valid_outcome(tmp_path):
    e = extraction()
    c = e.cohorts[0]
    c.dose = DoseExpression(raw_expression='not reported', value_kind='unparsed')
    c.exposures[0].dose = None
    c.exposures[0].dose_status = 'not_reported'
    resolved = resolve_extraction(e)
    assert not resolved.curve_rows[0].included_in_default
    assert resolved.curve_rows[0].outcome_usable
    scopes, notices = build_article_scopes(write_article(tmp_path, e))
    assert not notices
    assert sum(len(s.observations) for s in scopes) == 3
    assert sum(o.n for s in scopes for o in s.observations) == 9
    assert len(scopes) == 1  # B present/absent/unknown belong to the same model.


def test_class_rename_uses_same_model_rules(tmp_path):
    e = extraction()
    first, _ = build_article_scopes(write_article(tmp_path / 'a', e))
    for c in e.cohorts:
        c.exposures[1].name_raw = 'Another reported group'
    second, _ = build_article_scopes(write_article(tmp_path / 'b', e))
    assert [d.mode for d in first[0].drugs] == [d.mode for d in second[0].drugs]
    assert [(o.y, o.n) for o in first[0].observations] == [(o.y, o.n) for o in second[0].observations]
