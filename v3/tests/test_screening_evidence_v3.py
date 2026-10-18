import json

from phase1_acquisition.screening import (
    CURRENT_PROMPT, LEGACY_POLICY_VERSION, POLICY_VERSION, REFERENCE_POLICY_VERSION,
    PROMPT, REFERENCE_PROMPT, SUPPORTIVE_CARE_POLICY_VERSION, SUPPORTIVE_CARE_PROMPT,
    build_document, make_request,
)


def test_float_tables_and_captions_are_evidence_without_bibliography(tmp_path):
    source = tmp_path / 'article.xml'
    source.write_text('''<article xmlns="urn:jats"><front><article-meta><title-group>
      <article-title>Phase I antitumor treatment</article-title></title-group></article-meta></front>
      <body><sec><p>Patients received study treatment.</p><table-wrap id="body-table">
      <table><tr><td>Body unique value 14 mg</td></tr></table></table-wrap></sec></body>
      <floats-group><table-wrap id="dose"><label>Table 1</label><caption><p>Actual doses</p></caption>
      <table><tr><td>10 mg</td><td>0/6 DLT</td></tr></table><table-wrap-foot><p>First cycle</p></table-wrap-foot>
      </table-wrap><fig><caption><p>Figure observed response</p></caption></fig></floats-group>
      <back><app-group><app><p>Appendix patients</p></app></app-group><floats-group>
      <table-wrap><table><tr><td>Back floating 20 mg</td></tr></table></table-wrap></floats-group>
      <ref-list><ref>Unrelated dose 999 mg</ref></ref-list></back>
      <sub-article><body><p>Unrelated nested 888 mg</p></body></sub-article></article>''')
    doc = build_document({'document_id': 'example'}, [source])
    text = ' '.join(b['text'] for b in doc['blocks'])
    for expected in ['10 mg', '0/6 DLT', 'First cycle', 'Figure observed response',
                     'Appendix patients', 'Back floating 20 mg']:
        assert expected in text
    assert text.count('Body unique value 14 mg') == 1
    assert '999 mg' not in text and '888 mg' not in text
    assert doc['fulltext_available']
    assert doc['policy_version'] == POLICY_VERSION


def test_supportive_care_clarification_keeps_old_policy_requests_immutable():
    doc = build_document({'document_id': 'PMID1', 'title': 'Example', 'abstract': 'Patients treated.'}, [])
    for policy, expected in [(LEGACY_POLICY_VERSION, PROMPT),
                             (REFERENCE_POLICY_VERSION, REFERENCE_PROMPT),
                             (SUPPORTIVE_CARE_POLICY_VERSION, SUPPORTIVE_CARE_PROMPT)]:
        doc['policy_version'] = policy
        request = make_request(doc)
        assert request['instructions'] == expected
        schema = request['text']['format']['schema']
        assert schema['properties']['decision']['enum'] == ['\u4fdd\u7559', '\u4e0d\u4fdd\u7559']
    assert 'Stem-cell/CD34+ mobilization' in SUPPORTIVE_CARE_PROMPT
    assert 'identifiable cancer subgroup' in SUPPORTIVE_CARE_PROMPT
    assert 'Stem-cell/CD34+ mobilization' not in REFERENCE_PROMPT


def test_title_only_xml_does_not_displace_indexed_abstract(tmp_path):
    source = tmp_path / 'stub.xml'
    source.write_text('<article><front><article-meta><title-group><article-title>Trial title</article-title>'
                      '</title-group></article-meta></front></article>')
    doc = build_document({'document_id': 'PMID1', 'abstract': 'Six patients received 15 mg; no DLT.'}, [source])
    assert not doc['fulltext_available']
    assert any(b['source_id'].startswith('metadata:abstract:') and '15 mg' in b['text'] for b in doc['blocks'])
