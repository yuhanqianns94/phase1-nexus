"""Library availability uses the same Phase I scope filter as prediction."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from test_prediction_pipeline import _article


def test_library_does_not_advertise_phase_excluded_articles_as_ready(tmp_path,monkeypatch):
    for doc,phase in [('PMC101','Phase I'),('PMC102','Phase II'),('PMC103',None)]:
        paths=_article(tmp_path,doc,rows=[(.1,3,0)])
        value=json.loads(paths.reported_extraction.read_text())
        value['study_parts'][0]['phase_raw']=phase
        paths.reported_extraction.write_text(json.dumps(value))
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS',str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT',str(tmp_path/'state'))
    app=AppTest.from_file(Path(__file__).parents[1]/'phase1_web/app.py',default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Library').run()
    metrics={x.label:x.value for x in app.metric}
    assert metrics['Phase I eligible articles']=='1'
    rows=app.dataframe[0].value.set_index('Article')
    assert rows.loc['PMC101','Phase I eligibility']=='Eligible with outcomes'
    assert rows.loc['PMC102','Phase I eligibility']=='Excluded: outside Phase I'
    assert rows.loc['PMC103','Phase I eligibility']=='Excluded: phase not linked or unclear'
    next(s for s in app.selectbox if s.label=='Show').set_value('Phase I eligible articles').run()
    assert app.dataframe[0].value['Article'].tolist()==['PMC101']
