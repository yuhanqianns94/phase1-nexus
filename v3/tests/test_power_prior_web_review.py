"""Method 3 review shows the actual likelihood recipe, not mixture weights."""
import copy
from pathlib import Path

import pytest

from phase1_prediction.multidrug.mapping import match_target
from phase1_prediction.multidrug.power_prior import PowerPriorSpec, prepare_power_prior
from phase1_web.evidence_review import power_prior_review_preview
from phase1_web.multidrug_app import prediction_result_key, prediction_result_label
from test_multidrug_power_prior import article


def preview_rows(scored):
    return [{'series_key': scope.scope_id, 'document_id': scope.document_id,
             'mapping': mapping, 'similarity': strength}
            for scope, mapping, strength in scored]


def test_precise_power_settings_never_reactivate_another_configuration():
    settings = {'strength': .12345671, 'patient_budget': 30., 'heterogeneity_scale': .5}
    other = {**settings, 'strength': .12345672}
    first = prediction_result_key('logistic', 'target_power_prior', settings)
    second = prediction_result_key('logistic', 'target_power_prior', other)
    assert first != second
    # The previous six-significant-digit labels collided; display can still use
    # those compact labels without allowing them to identify a stored result.
    assert prediction_result_label(first) == prediction_result_label(second)
    assert 'settings=' not in prediction_result_label(first)
    assert prediction_result_key('logistic', 'same_article_joint_posterior_average') == 'logistic · same_article_joint_posterior_average'


def test_review_preview_selects_the_same_eligible_scope_and_absolute_powers_as_prediction():
    first, target = article(doc='one')
    weaker = copy.deepcopy(first)
    weaker[0].scope_id = 'one_weaker'
    second, _ = article(doc='two', strength=.4)
    incomplete, _ = article(k=3, doc='one')
    incomplete[0].scope_id = 'one_incomplete'
    incomplete = (incomplete[0], match_target(target, incomplete[0]), .99)
    scored = [first, (weaker[0], weaker[1], .3), second, incomplete]
    rows, scopes = preview_rows(scored), [row[0] for row in scored]
    spec = PowerPriorSpec(patient_budget=18.)
    include = power_prior_review_preview(rows, scopes, target, spec)
    uncertain = power_prior_review_preview(rows, scopes, target, spec, {'one': 'Uncertain', 'two': 'Uncertain'})
    decisions = {'one': 'Include', 'two': 'Exclude'}
    excluded = power_prior_review_preview(rows, scopes, target, spec, decisions)
    actual = prepare_power_prior(scored, target, spec, decisions)
    assert [row['Selected scope'] for row in excluded['rows']] == [item['scope_id'] for item in actual['sources']]
    assert [row['Likelihood power'] for row in excluded['rows']] == pytest.approx([item['power'] for item in actual['sources']])
    assert excluded['excluded'] == actual['excluded']
    assert include['discounted_patient_count'] == pytest.approx(18.)
    assert uncertain['discounted_patient_count'] == pytest.approx(9.)
    assert excluded['discounted_patient_count'] == pytest.approx(12.)
    assert include['cap_factor'] == uncertain['cap_factor'] == excluded['cap_factor'] == .5
    assert sum(row['Likelihood power'] for row in include['rows']) == pytest.approx(.6)
    assert not power_prior_review_preview(rows, scopes, target, spec, {'one': 'Exclude', 'two': 'Exclude'})['ready']
    prior_only = power_prior_review_preview(rows, scopes, target, PowerPriorSpec(strength=0))
    assert prior_only['ready'] and prior_only['discounted_patient_count'] == 0


def test_method3_abstract_review_displays_absolute_discounted_counts(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_prediction_pipeline import _article
    _article(tmp_path, 'PMC123', rows=[(.1, 3, 0), (.3, 6, 2)])
    monkeypatch.setenv('PHASE1_V3_EXTRACTION_ROOTS', str(tmp_path))
    monkeypatch.setenv('PHASE1_V3_STATE_ROOT', str(tmp_path / 'state'))
    app = AppTest.from_file(Path(__file__).parents[1] / 'phase1_web/app.py', default_timeout=30).run()
    app.sidebar.radio[0].set_value('Target Trial').run()
    from test_multidrug_ui_state_audit import seed_test_plan
    seed_test_plan(app,tmp_path)
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(x for x in app.radio if x.label == 'Evidence screening').set_value('Review article abstracts').run()
    next(b for b in app.button if b.label == 'Find matching articles').click().run()
    assert next(b for b in app.button if b.label == 'Continue to risk curves').disabled
    app.sidebar.radio[0].set_value('Prediction Results').run()
    next(x for x in app.selectbox if x.label == 'Curve aggregation').set_value('Power prior from discounted historical data (Method 3)').run()
    assert next(b for b in app.button if b.label == 'Generate prediction').disabled
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(x for x in app.radio if x.label == 'Relevance decision').set_value('Include').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    included = next(frame.value for frame in app.dataframe if 'Likelihood power' in frame.value.columns)
    app.sidebar.radio[0].set_value('Review evidence').run()
    next(x for x in app.radio if x.label == 'Relevance decision').set_value('Uncertain').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    uncertain = next(frame.value for frame in app.dataframe if 'Likelihood power' in frame.value.columns)
    assert not app.exception and not app.error
    assert uncertain['Discounted patients'].sum() == pytest.approx(included['Discounted patients'].sum() * .5)
    assert not next(b for b in app.button if b.label == 'Generate prediction').disabled
    captions = ' '.join(x.value for x in app.caption)
    assert 'absolute likelihood powers' in captions
    assert 'does not halve patient counts' not in captions
    next(x for x in app.selectbox if x.label == 'Curve aggregation').set_value('Average of article posterior curves (Method 1)').run()
    app.sidebar.radio[0].set_value('Review evidence').run()
    assert any('does not halve patient counts' in x.value for x in app.caption)
    assert any('normalized_weight' in frame.value.columns for frame in app.dataframe)
