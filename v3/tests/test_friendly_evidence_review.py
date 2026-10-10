"""Readable review cards preserve article decisions and borrowing contracts."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phase1_web.evidence_review import abstract_for, _mapping_rows


def _source(root, title='A study of source medicines', year='2021'):
    root.mkdir(parents=True, exist_ok=True)
    xml = root / 'paper.xml'
    xml.write_text(
        '<article xmlns="urn:jats"><front><article-meta>'
        f'<title-group><article-title>{title}</article-title></title-group>'
        f'<pub-date pub-type="epub"><year>{year}</year></pub-date>'
        '<abstract><sec><title>Methods</title><p>Original abstract wording.</p></sec></abstract>'
        '<abstract abstract-type="keypoints"><p>Additional source wording.</p></abstract>'
        '</article-meta></front></article>', encoding='utf-8')
    (root / 'document_ir.json').write_text(json.dumps({
        'materials': [{'path': str(xml)}], 'blocks': []}), encoding='utf-8')
    return root


def _review_app(source_root, state_root, method3=False):
    """Independent review page: no model fitting or candidate doses required."""
    import streamlit as st
    from types import SimpleNamespace
    from phase1_web.evidence_review import render_review

    scopes = [SimpleNamespace(
        document_id=doc, scope_id=doc + '_scope', label='Phase I escalation',
        lineage={'root': source_root}, drugs=[SimpleNamespace(
            model_drug_id='source_a', name='Source medicine A')])
        for doc in ('PMC123', 'PMC456')]
    rows = [{
        'document_id': scope.document_id, 'series_key': scope.scope_id,
        'label': scope.label, 'similarity': .8,
        'point_count': 2, 'prior_only': False,
        'mapping': {'pairs': {'target_a': 'source_a'}, 'details': [{
            'target_id': 'target_a', 'role_rule': 'manual_override',
            'comparison': {'details': [{'indicator': 'Identity',
                                       'left': 'Planned medicine A', 'right': 'Source medicine A'}]}}],
            'score': {'groups': {'drugs': {'score': .8}}}},
    } for scope in scopes]
    decisions, ready = render_review(st, rows, scopes, state_root, 'planned-target', True,
                                    power_prior_settings={} if method3 else None)
    st.session_state['review_result'] = {'decisions': decisions, 'ready': ready}


def test_original_xml_metadata_and_abstract_are_read_without_llm(tmp_path):
    root = _source(tmp_path / 'paper')
    result = abstract_for(root)
    assert result['title'] == 'A study of source medicines'
    assert result['year'] == '2021'
    assert result['source'] == 'Original JATS XML abstract'
    assert result['main'] == [{'heading': 'Methods'}, {'text': 'Original abstract wording.'}]
    assert result['additional'][0]['sections'] == [{'text': 'Additional source wording.'}]
    # Publication-year metadata does not reset existing decisions.
    _source(root, year='2022')
    assert abstract_for(root)['content_id'] == result['content_id']


def test_mapping_display_uses_frozen_assignment_and_human_drug_names():
    from types import SimpleNamespace
    scope = SimpleNamespace(drugs=[SimpleNamespace(model_drug_id='source_b', name='B')])
    row = {'mapping': {'pairs': {'target_a': 'source_b'}, 'details': [{
        'target_id': 'target_a', 'role_rule': 'manual_override',
        'comparison': {'details': [{'indicator': 'Identity', 'left': 'A', 'right': 'B'}]}}]}}
    assert _mapping_rows(row, scope) == [
        {'Your drug': 'A', 'Article drug': 'B', 'Match': 'Chosen by you'}]
    assert row['mapping']['pairs'] == {'target_a': 'source_b'}


def test_cards_keep_pending_distinct_and_save_canonical_uncertain(tmp_path):
    root = _source(tmp_path / 'paper')
    state = tmp_path / 'state'
    app = AppTest.from_function(_review_app, args=(str(root), str(state))).run()
    assert not app.exception
    assert len(app.subheader) == 2
    assert all('A study of source medicines' in item.value for item in app.subheader)
    assert any('2021' in item.value for item in app.caption)
    assert not any('coverage' in item.value.lower() or 'Information available' in item.value for item in app.caption)
    assert not any('coverage' in str(column).lower() or column == 'Information available'
                   for frame in app.dataframe for column in frame.value.columns)
    assert any('Planned medicine A → Source medicine A' in item.value for item in app.markdown)
    assert not app.session_state['review_result']['ready']
    assert app.radio[0].options == ['Not reviewed', 'Include', 'Exclude', 'Unsure']
    app.radio[0].set_value('Include').run()
    assert not app.session_state['review_result']['ready']
    app.radio[1].set_value('Uncertain').run()
    result = app.session_state['review_result']
    assert result['ready']
    assert result['decisions'] == {'PMC123': 'Include', 'PMC456': 'Uncertain'}
    saved = json.loads(next((state / 'relevance_reviews').glob('*.json')).read_text())
    assert saved['decisions'] == result['decisions']
    weights = next(frame.value for frame in app.dataframe if 'normalized_weight' in frame.value.columns)
    assert weights['normalized_weight'].tolist() == pytest.approx([2/3, 1/3])
    app.radio[0].set_value('Exclude').run()
    app.radio[1].set_value('Exclude').run()
    assert not app.session_state['review_result']['ready']
    reopened = AppTest.from_function(_review_app, args=(str(root), str(state))).run()
    assert [radio.value for radio in reopened.radio] == ['Exclude', 'Exclude']


def test_dose_free_method3_review_defers_exact_power_checks(tmp_path):
    root = _source(tmp_path / 'paper')
    app = AppTest.from_function(_review_app, args=(str(root), str(tmp_path / 'state'), True)).run()
    app.radio[0].set_value('Include').run()
    app.radio[1].set_value('Uncertain').run()
    assert not app.exception and not app.error
    assert app.session_state['review_result']['ready']
    assert any('after you choose curve coordinates' in item.value for item in app.info)
    assert not any('Likelihood power' in frame.value.columns for frame in app.dataframe)


def test_unmatched_drugs_display_names_without_inventing_correspondence():
    from types import SimpleNamespace
    row={'mapping':{'pairs':{},'unmatched_target_ids':['internal_19']}}
    result=_mapping_rows(row,SimpleNamespace(drugs=[]),{'internal_19':'New drug A'})
    assert result==[{'Your drug':'New drug A','Article drug':'No suitable match','Match':'Unmatched'}]
    assert row['mapping']['pairs']=={}
