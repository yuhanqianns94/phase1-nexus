"""One named target, explicit combination context and independent prior strength."""
import copy
import json
from types import SimpleNamespace

import pytest

from phase1_prediction.multidrug.single_drug import COMBINATION_PRIOR_SCALES, CONTEXT_ID, project_target
from phase1_prediction.multidrug.service import _single_signature
from phase1_web.single_drug_plan import single_drug_plan
from phase1_web.multidrug_app import _validate_draft_shape
from test_single_drug_core import fixture
from test_multidrug_ui_state_audit import target_app


def test_old_plan_preserves_focal_doses_and_explicit_combination_without_mutating_source():
    _, old = fixture(3)
    snapshot = old.model_dump(mode='json')
    draft = single_drug_plan(snapshot)
    assert len(draft['drugs']) == 1
    assert draft['profile']['single_drug_analysis']['combination_present'] is True
    assert [p['doses']['A'] for p in draft['points']] == [p.doses['A'] for p in old.points]
    assert old.model_dump(mode='json') == snapshot
    assert all(p.doses[CONTEXT_ID] is None for p in project_target(draft).points)
    mono = copy.deepcopy(draft)
    mono['profile']['single_drug_analysis']['combination_present'] = False
    assert all(p.doses[CONTEXT_ID] == 0 for p in project_target(mono).points)
    from phase1_prediction.multidrug.contracts import TargetRegimen
    assert _single_signature(TargetRegimen.model_validate(mono)) != _single_signature(TargetRegimen.model_validate(draft))
    with pytest.raises(ValueError, match='only one'):
        _validate_draft_shape(snapshot, single_drug_only=True)


@pytest.mark.parametrize('method', ['average', 'mixture', 'historical'])
def test_one_drug_ui_and_all_three_combination_priors_reach_each_method(tmp_path, monkeypatch, method):
    import phase1_web.multidrug_app as workspace
    calls = []
    def predict(library, target, **kwargs):
        calls.append((target, kwargs))
        return SimpleNamespace(to_dict=lambda: {'run_id': 'test-' + str(len(calls)),
            'target': target.model_dump(), 'methods': {}})
    monkeypatch.setattr(workspace, 'predict', predict)
    app = target_app(tmp_path, monkeypatch)
    assert len([w for w in app.text_input if w.label == 'Drug name or development code']) == 1
    assert not any(b.label in {'Add drug', 'Move up', 'Move down'} for b in app.button)
    assert not any(r.label == 'Combination toxicity prior' for r in app.radio)
    app.radio(key='plan_combination_use').set_value('Yes').run()
    assert app.radio(key='plan_combination_prior').value == 'medium'
    for level, scale in COMBINATION_PRIOR_SCALES.items():
        app.radio(key='plan_combination_prior').set_value(level).run()
        app.sidebar.radio[0].set_value('Prediction Results').run()
        assert not any(w.label == 'Focal drug' for w in app.multiselect)
        app.radio(key='prior_settings_single_drug_logistic_risk_preset').set_value('regularized').run()
        app.radio(key='workflow_formula_aggregation').set_value(method).run()
        next(b for b in app.button if b.label == 'Generate prediction').click().run()
        target, kwargs = calls[-1]
        assert len(target.drugs) == 1
        assert kwargs['prior'].combination_sd == scale
        assert kwargs['prior'].intercept_sd == 1.5
        assert kwargs['prior'].log_slope_sd == .5
        assert all(p.doses[CONTEXT_ID] is None for p in project_target(target).points)
        assert any('\\delta' in equation.value for equation in app.latex)
        app.sidebar.radio[0].set_value('Target Trial').run()
        assert app.radio(key='plan_combination_prior').value == level
    app.radio(key='plan_combination_use').set_value('No').run()
    assert not any(r.label == 'Combination toxicity prior' for r in app.radio)
    app.radio(key='plan_combination_use').set_value('Yes').run()
    assert app.radio(key='plan_combination_prior').value == 'high'
    app.radio(key='plan_combination_use').set_value('No').run()
    app.sidebar.radio[0].set_value('Prediction Results').run()
    next(b for b in app.button if b.label == 'Generate prediction').click().run()
    assert all(p.doses[CONTEXT_ID] == 0 for p in project_target(calls[-1][0]).points)
    assert not app.error and not app.exception


def test_legacy_multidrug_plan_is_backed_up_before_web_migration(tmp_path, monkeypatch):
    app = target_app(tmp_path, monkeypatch)
    _, target = fixture(2)
    original = target.model_dump_json()
    app.session_state['multidrug_target_text'] = original
    app.session_state['multidrug_reset_editor'] = True
    app.run()
    saved = json.loads(app.session_state['multidrug_target_text'])
    assert len(saved['drugs']) == 1
    assert saved['profile']['single_drug_analysis']['combination_present'] is True
    backups = list((tmp_path / 'state').glob('*.before_single_drug_*.json'))
    assert len(backups) == 1 and backups[0].read_text() == original
    assert not app.error and not app.exception
