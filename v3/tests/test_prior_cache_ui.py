"""Cache controls inspect on render and fit only the explicitly selected recipes."""
from types import SimpleNamespace

import pytest

from phase1_prediction.multidrug.contracts import PriorSpec
from phase1_prediction.multidrug import power_prior
from phase1_web import prior_cache as ui


class FakeUI:
    def __init__(self, *, clicks=(), choices=None):
        self.clicks = set(clicks)
        self.choices = choices or {}
        self.messages = []
        self.tables = []
        self.selections = []
        self.buttons = []
        self.session_state = {'multidrug_prediction': {'sentinel': 'keep current prediction'}}

    def multiselect(self, label, options, default, **kwargs):
        self.selections.append((label, options, default, kwargs))
        return self.choices.get(kwargs['key'], default)

    def button(self, label, *, key, disabled=False):
        self.buttons.append((label, key, disabled))
        return key in self.clicks and not disabled

    def empty(self):
        return self

    def dataframe(self, rows, **kwargs):
        self.tables.append(rows)

    def expander(self, label):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def caption(self, text): self.messages.append(('caption', text))
    def info(self, text): self.messages.append(('info', text))
    def error(self, text): self.messages.append(('error', text))
    def success(self, text): self.messages.append(('success', text))
    def warning(self, text): self.messages.append(('warning', text))
    def json(self, data): self.messages.append(('json', data))


def source_status(*, ready=1, missing=2, failed=0, skipped=0):
    return {'counts': {'ready': ready, 'missing': missing, 'cached_failure': failed,
                       'phase_excluded': skipped},
            'failed': [{'scope_id': 'bad', 'fit_status': 'diagnostics_failed'}] * failed,
            'errors': []}


def test_source_render_inspects_all_three_candidates_without_fitting(monkeypatch):
    calls = []
    monkeypatch.setattr(ui.presets, 'source_cache_status',
                        lambda scopes, **kw: calls.append((scopes, kw)) or source_status(skipped=1))
    monkeypatch.setattr(ui.presets, 'prepare_source_cache',
                        lambda *a, **kw: pytest.fail('Rendering must not start sampling'))
    st = FakeUI()
    ui.render_source_cache(st, ['scope'], key='library')
    assert len(calls) == 3
    assert {kw['backend'] for _, kw in calls} == {'logistic'}
    assert st.selections[0][2] == ['balanced', 'regularized', 'broad']
    assert len(st.tables[-1]) == 3
    assert all(row['Cached'] == 1 and row['Pending'] == 2 and row['Skipped'] == 1 for row in st.tables[-1])
    assert st.session_state['multidrug_prediction'] == {'sentinel': 'keep current prediction'}


def test_source_prepare_keeps_failed_fits_visible(monkeypatch):
    fit_calls = []
    monkeypatch.setattr(ui.presets, 'source_cache_status', lambda *a, **kw: source_status(failed=1))
    def prepare(scopes, **kw):
        fit_calls.append(kw)
        kw['progress'](1, 4, 'PMC123', kw['backend'])
        return source_status(ready=3, missing=0, failed=1)
    monkeypatch.setattr(ui.presets, 'prepare_source_cache', prepare)
    st = FakeUI(clicks=['target_source_prepare'])
    ui.render_source_cache(st, ['scope'], backends=('logistic',),
                           allow_backend_selection=False, key='target_source')
    assert len(fit_calls) == 3
    assert not any(label == 'Models to cache' for label, *_ in st.selections)
    assert all(row['Cached'] == 3 and row['Needs attention'] == 1 for row in st.tables[-1])
    assert any(kind == 'warning' and '9 usable cached fits' in text for kind, text in st.messages if isinstance(text, str))
    assert not any(kind == 'success' for kind, _ in st.messages)
    assert st.session_state['multidrug_prediction'] == {'sentinel': 'keep current prediction'}


def test_custom_prior_is_offered_only_when_numerically_distinct(monkeypatch):
    monkeypatch.setattr(ui.presets, 'source_cache_status', lambda *a, **kw: source_status(missing=0))
    ordinary = FakeUI()
    ui.render_source_cache(ordinary, [], backends=('logistic',), allow_backend_selection=False,
                           selected_prior=PriorSpec(prior_id='custom label'))
    assert ordinary.selections[0][1] == ['balanced', 'regularized', 'broad']
    custom = FakeUI()
    ui.render_source_cache(custom, [], backends=('logistic',), allow_backend_selection=False,
                           selected_prior=PriorSpec(intercept_mean=-2.))
    assert custom.selections[0][1] == ['balanced', 'regularized', 'broad', 'current']
    assert custom.buttons[0][2]


def target_arguments(**overrides):
    values = {'library': object(), 'target': object(),
              'preview': {'selected_series': [
                  {'series_key': 'keep', 'mapping': {'manual': 'preserved'}, 'similarity': .8, 'coverage': .5},
                  {'series_key': 'drop', 'mapping': {'automatic': True}, 'similarity': .9, 'coverage': .7}]},
              'included': ['keep'], 'relevance_decisions': {'PMC1': 'Uncertain'},
              'weights': {'disease': 1}, 'threshold': .2, 'top_articles': 5, 'backend': 'logistic',
              'power_settings': {'strength': .5, 'patient_budget': 15., 'heterogeneity_scale': .75},
              'review_ready': True,
              'scopes': [SimpleNamespace(scope_id='keep'), SimpleNamespace(scope_id='drop')]}
    values.update(overrides)
    return values


@pytest.mark.parametrize('override', [{'preview': None}, {'included': []}, {'review_ready': False}])
def test_target_cache_requires_selected_reviewed_preview(monkeypatch, override):
    monkeypatch.setattr(power_prior, 'prepare_power_prior', lambda *a: pytest.fail('Evidence not ready'))
    monkeypatch.setattr(ui.presets, 'prepared_target_cache_status', lambda *a, **kw: pytest.fail('Evidence not ready'))
    st = FakeUI(clicks=['target_cache_prepare'])
    ui.render_target_cache(st, **target_arguments(**override))
    assert not st.buttons
    assert any(kind == 'info' for kind, _ in st.messages)


def target_status(status='missing'):
    return {'cache_status': status, 'fit_status': 'fitted' if status == 'ready' else None,
            'fit_id': 'power_test', 'article_count': 1, 'discounted_patient_count': 1.5}


def test_target_render_reuses_mappings_and_borrowing_settings_without_fitting(monkeypatch):
    preparations = []
    status_calls = []
    def prepare(scored, target, spec, decisions):
        preparations.append((scored, target, spec, decisions))
        return {'sentinel': 'prepared once'}
    monkeypatch.setattr(power_prior, 'prepare_power_prior', prepare)
    monkeypatch.setattr(ui.presets, 'prepared_target_cache_status',
                        lambda prepared, **kw: status_calls.append((prepared, kw)) or target_status())
    monkeypatch.setattr(ui.presets, 'prepare_prepared_target_cache',
                        lambda *a, **kw: pytest.fail('Rendering must not start sampling'))
    monkeypatch.setattr(ui.presets, 'target_cache_status',
                        lambda *a, **kw: pytest.fail('Must not repeat similarity selection'))
    st = FakeUI()
    args = target_arguments()
    ui.render_target_cache(st, **args)
    assert len(preparations) == 1 and len(status_calls) == 3
    scored, target, spec, decisions = preparations[0]
    assert len(scored) == 1 and scored[0][0].scope_id == 'keep'
    assert scored[0][1] == {'manual': 'preserved'} and scored[0][2] == pytest.approx(.8)
    assert target is args['target'] and decisions == args['relevance_decisions']
    assert spec.model_dump() == args['power_settings']
    assert all(call[0] == {'sentinel': 'prepared once'} for call in status_calls)
    assert len(st.tables[-1]) == 3
    assert st.session_state['multidrug_prediction'] == {'sentinel': 'keep current prediction'}


def test_target_preparation_is_three_current_recipes_and_reports_sampling_failure(monkeypatch):
    fit_calls = []
    monkeypatch.setattr(power_prior, 'prepare_power_prior', lambda *a: {'fixed': 'current borrowing recipe'})
    monkeypatch.setattr(ui.presets, 'prepared_target_cache_status', lambda *a, **kw: target_status())
    def fit(prepared, **kw):
        fit_calls.append((prepared, kw))
        kw['progress']('sampling')
        return target_status('cached_failure' if len(fit_calls) == 2 else 'ready')
    monkeypatch.setattr(ui.presets, 'prepare_prepared_target_cache', fit)
    st = FakeUI(clicks=['target_cache_prepare'])
    ui.render_target_cache(st, **target_arguments())
    assert len(fit_calls) == 3
    assert all(prepared == {'fixed': 'current borrowing recipe'} for prepared, _ in fit_calls)
    assert any(kind == 'warning' and '2/3 usable target fits' in text for kind, text in st.messages if isinstance(text, str))
    assert st.tables[-1][1]['Status'] == 'Sampling needs attention'
    assert not any(kind == 'success' for kind, _ in st.messages)
    assert st.session_state['multidrug_prediction'] == {'sentinel': 'keep current prediction'}


def test_target_missing_scope_stops_before_cache_calls(monkeypatch):
    monkeypatch.setattr(power_prior, 'prepare_power_prior', lambda *a: pytest.fail('Invalid scope selection'))
    st = FakeUI()
    ui.render_target_cache(st, **target_arguments(scopes=[]))
    assert any(kind == 'error' and 'changed' in text for kind, text in st.messages)
