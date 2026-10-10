"""An unfinished prospective trial can match evidence before choosing doses."""
import json
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest
from test_prediction_pipeline import _article


APP = Path(__file__).parents[1] / 'phase1_web/app.py'


def button(app, label):
    return next(item for item in app.button if item.label == label)


def new_app(tmp_path, monkeypatch):
    _article(tmp_path, 'PMC123', rows=[(.1, 3, 0), (.3, 6, 2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS', str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT', str(tmp_path / 'state'))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input(key='plan_first_drug_name').set_value('IPH2101').run()
    assert not app.exception
    return app


def review_one(app):
    button(app, 'Continue to matching articles').click().run()
    assert app.sidebar.radio[0].value == 'Review evidence'
    app.radio(key='workflow_screening').set_value('Review article abstracts').run()
    button(app, 'Find matching articles').click().run()
    decision = next(item for item in app.radio if item.label == 'Relevance decision')
    assert decision.value == 'Not reviewed'
    assert button(app, 'Continue to risk curves').disabled
    decision.set_value('Include').run()
    assert not button(app, 'Continue to risk curves').disabled
    return app


def test_no_dose_prospective_plan_can_review_and_predict_relative_curve(tmp_path, monkeypatch):
    import phase1_web.multidrug_app as workspace
    calls = []
    def predict(library, target, **kwargs):
        calls.append((target, kwargs))
        return SimpleNamespace(to_dict=lambda: {'run_id': 'guided-test', 'target': target.model_dump(),
            'methods': {}, 'evidence_contract': {'prior': kwargs['prior'].model_dump()}})
    monkeypatch.setattr(workspace, 'predict', predict)
    app = new_app(tmp_path, monkeypatch)
    original = json.loads(app.session_state['multidrug_target_text'])
    assert not original['points']
    assert original['drugs'][0].get('reference_dose') is None
    review_one(app)
    button(app, 'Continue to risk curves').click().run()
    assert app.sidebar.radio[0].value == 'Prediction Results'
    assert app.radio(key='explore_mode').value == 'Relative-dose exploration'
    assert not button(app, 'Generate prediction').disabled
    button(app, 'Generate prediction').click().run()
    target, options = calls[-1]
    assert target.profile['planning']['exploration']['relative']
    assert len(target.points) > 1
    assert options['relevance_decisions'] == {'PMC123': 'Include'}
    assert options['include_series_keys']
    assert not json.loads(app.session_state['multidrug_target_text'])['points']
    assert not app.exception and not app.error


def test_reading_trial_plan_does_not_erase_screening_or_filters(tmp_path, monkeypatch):
    app = review_one(new_app(tmp_path, monkeypatch))
    before = app.session_state['multidrug_review_state']
    app.sidebar.radio[0].set_value('Target Trial').run()
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert app.session_state['multidrug_review_state'] == before
    assert next(item for item in app.radio if item.label == 'Relevance decision').value == 'Include'
    app.selectbox(key='workflow_pool_mode').set_value('Top N articles').run()
    app.number_input(key='workflow_top_articles').set_value(5).run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert app.number_input(key='workflow_top_articles').value == 5
    assert not app.exception and not app.error
