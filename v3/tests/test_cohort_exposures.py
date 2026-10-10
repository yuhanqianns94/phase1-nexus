from __future__ import annotations
import pytest
from phase1_extraction.artifacts import ArticlePaths, write_json
from phase1_extraction.models import CohortExposure, DosingPhase, DoseExpression, EvidenceLink, PopulationCount, DLTObservation
from phase1_extraction.exposures import exposure_identity, exposure_text
from phase1_extraction.exposure_review import editor_rows, parse_editor_rows
from phase1_extraction.friendly_review import load_or_create_friendly_review, mark_module_correct, save_friendly_review, CURVE_MODULE_ID, _curve_semantic_key
from phase1_extraction.llm import _merge_models, build_request_body
from phase1_extraction.quality import audit_reported_extraction
from phase1_extraction.resolver import resolve_extraction
from phase1_extraction.review_app import _configuration_table, _sync_primary_dose
from test_resolver import base_extraction, dose


def exposure(name='A', *, presence='present', value=750, kind='drug', component='drug'):
    return CohortExposure(exposure_id='ex_' + name, name_raw=name, entity_kind=kind,
        component_id=component, role='escalated' if name == 'A' else 'concomitant', presence=presence,
        dose=dose(value) if value is not None else None,
        dose_status='reported' if value is not None else ('not_applicable' if presence == 'absent' else 'not_reported'),
        evidence=[EvidenceLink(source_ids=['table1'], source_text=f'{name}: {presence}')])


def extraction():
    e = base_extraction(doses=(750, 750, 2200))
    e.exposure_contract = 'cohort_exposures_v1'
    for i, c in enumerate(e.cohorts):
        partner = exposure('EIASD', presence=['present', 'absent', 'unknown'][i], value=None, kind='drug_class', component=None)
        if i == 2:
            partner.reporting_scope = 'pooled'
        c.exposures = [exposure(value=c.dose.value), partner]
        c.exposure_coverage = 'complete'
        e.population_counts.append(PopulationCount(count_id=f'n{i}', study_id='study', cohort_id=c.cohort_id, role='paired_dlt_denominator', reported_value=[3,3,3][i]))
        e.dlt_observations.append(DLTObservation(observation_id=f'o{i}', study_id='study', result_scope_id='scope', cohort_id=c.cohort_id,
            count_kind='patients', explicit_patient_count=[0,0,2][i], denominator_count_id=f'n{i}'))
    return e


def test_resolver_keeps_one_outcome_per_configuration_and_pooled_result():
    e = extraction()
    rows = resolve_extraction(e).curve_rows
    assert len(rows) == 3
    assert [(r.y, r.n) for r in rows] == [(0,3), (0,3), (2,3)]
    assert sum(r.n for r in rows) == 9  # Not doubled for two drugs.
    assert [r.exposures[1].presence for r in rows] == ['present', 'absent', 'unknown']
    assert rows[-1].exposures[1].reporting_scope == 'pooled'
    assert rows[0].exposures[1].dose is None
    assert rows[1].exposures[1].dose_status == 'not_applicable'
    assert rows[0].exposures is not e.cohorts[0].exposures


def test_merge_does_not_collapse_same_primary_dose_with_different_partner():
    left = extraction(); left.cohorts = left.cohorts[:1]
    right = left.model_copy(deep=True)
    right.cohorts[0].cohort_id = 'another_cohort'
    right.cohorts[0].exposures[1] = exposure('EIASD', presence='absent', value=None, kind='drug_class', component=None)
    merged = _merge_models(left, right)
    assert len(merged.cohorts) == 2
    assert merged.exposure_contract == 'cohort_exposures_v1'


def test_same_cohort_conflict_is_visible_not_silently_accepted():
    left = extraction(); right = left.model_copy(deep=True)
    right.cohorts[0].exposures[0].dose = dose(999)
    merged = _merge_models(left, right)
    assert merged.unresolved_items
    assert merged.cohorts[0].exposures[0].dose_status == 'conflicting'
    audit = audit_reported_extraction(merged)
    assert any(f['code'] == 'cohort_exposure_conflict' for f in audit['findings'])


def test_editor_preserves_complex_dose_and_phases_on_unchanged_round_trip():
    e = exposure()
    e.dose = DoseExpression(raw_expression='100–200 mg/m2', value_kind='range', range_min=100, range_max=200, unit_raw='mg/m2')
    e.dosing_phases = [DosingPhase(dosing_phase_id='p', phase_label='cycle 1', administration_days_raw='days 1–5', cycle_length_raw='28 days')]
    assert parse_editor_rows(editor_rows([e]), [e]) == [e]
    rows = editor_rows([e]); rows[0]['Schedule'] = 'days 1 and 8 every 21 days'
    changed = parse_editor_rows(rows, [e])[0]
    assert changed.schedule_raw == rows[0]['Schedule']
    assert changed.dosing_phases == []
    assert changed.dose == e.dose


def test_editor_distinguishes_unknown_absent_and_rejects_false_with_dose():
    e = exposure('B', value=None, component=None)
    rows = editor_rows([e]); rows[0]['Use'] = 'absent'; rows[0]['Dose status'] = 'not_applicable'
    assert parse_editor_rows(rows, [e])[0].presence == 'absent'
    rows[0]['Dose value'] = 10
    with pytest.raises(ValueError, match='absent exposure'):
        parse_editor_rows(rows, [e])
    assert parse_editor_rows([{'Drug / class':'', 'Type':'drug', 'Role':'escalated', 'Use':'unknown', 'Dose status':'unknown', 'Reporting':'cohort'}], []) == []


def write_article(tmp_path, e):
    paths = ArticlePaths(tmp_path, e.document_id); paths.ensure()
    write_json(paths.reported_extraction, e)
    write_json(paths.curve_rows, {'all_rows':[r.model_dump(mode='json') for r in resolve_extraction(e).curve_rows]})
    return paths


def test_review_table_and_fingerprint_include_exposure_and_evidence(tmp_path):
    e = extraction(); paths = write_article(tmp_path, e)
    review = load_or_create_friendly_review(paths)
    units = [u for u in review.units if u.unit_kind == 'curve_point']
    assert len(units) == 3
    assert 'table1' in units[0].evidence_source_ids
    assert _curve_semantic_key(units[0].predicted_point) != _curve_semantic_key(units[1].predicted_point)
    html = _configuration_table(units)
    assert 'EIASD [class]: true (used)' in html
    assert 'false (not used)' in html
    assert 'pooled / not stratified' in html
    mark_module_correct(review, CURVE_MODULE_ID); save_friendly_review(paths, review)
    e.cohorts[0].exposures[1].name_raw = 'Specific drug B'
    write_article(tmp_path, e)
    updated = load_or_create_friendly_review(paths)
    updated_units = [u for u in updated.units if u.unit_kind == 'curve_point']
    assert updated_units[0].decision == 'pending'
    assert updated_units[1].decision == 'correct'
    assert not updated.module_complete[CURVE_MODULE_ID]


def test_safe_html_and_primary_dose_edit(tmp_path):
    e = extraction(); e.cohorts[0].exposures[0].name_raw = '<script>alert(1)</script>'
    review = load_or_create_friendly_review(write_article(tmp_path, e))
    units = [u for u in review.units if u.unit_kind == 'curve_point']
    assert '<script>' not in _configuration_table(units)
    previous = units[0].predicted_point; corrected = previous.model_copy(deep=True)
    corrected.exposures[0].dose = dose(800)
    _sync_primary_dose(corrected, previous)
    assert corrected.dose_value == 800
    assert corrected.dose_raw == '800 mg'


def test_exposure_identity_ignores_order_and_record_ids():
    a, b = exposure(), exposure('B', value=None, component=None)
    changed = a.model_copy(update={'exposure_id':'different'})
    assert exposure_identity([a,b]) == exposure_identity([b,changed])
    assert 'dose not reported' in exposure_text(b)


def test_strict_request_schema_contains_exposure_contract_and_fields():
    body = build_request_body(packet={'document_id':'D'}, prior=None, image_paths=[], model='test', reasoning_effort='low', max_output_tokens=1000)
    schema = body['text']['format']['schema']
    exposure_schema = schema['$defs']['CohortExposure']
    assert set(exposure_schema['properties']) == set(exposure_schema['required'])
    assert 'presence' in exposure_schema['properties']
    assert 'exposure_contract' in schema['properties']
    assert 'Never split pooled y/n' in body['instructions']


def test_primary_exposure_mismatch_is_audited_and_replacement_clears_references():
    e = extraction()
    e.cohorts[0].exposures[0].dose = dose(999)
    audit = audit_reported_extraction(e)
    assert any('disagrees with cohort primary dose' in str(f) for f in audit['findings'])
    assert not resolve_extraction(e).curve_rows[0].included_in_default
    original = exposure()
    rows = editor_rows([original]); rows[0]['Drug / class'] = 'Replacement'
    replaced = parse_editor_rows(rows, [original])[0]
    assert replaced.component_id is None
    assert replaced.intervention_id is None


def test_confirm_visible_table_preserves_saved_correction(tmp_path):
    review = load_or_create_friendly_review(write_article(tmp_path, extraction()))
    unit = next(u for u in review.units if u.predicted_point)
    unit.decision = 'incorrect'
    unit.gold_point = unit.predicted_point.model_copy(update={'n': 2})
    mark_module_correct(review, CURVE_MODULE_ID)
    assert unit.decision == 'incorrect'
    assert unit.gold_point.n == 2
    assert review.module_complete[CURVE_MODULE_ID]
