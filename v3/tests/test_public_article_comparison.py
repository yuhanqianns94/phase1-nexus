"""Public matrix comparisons are bounded before any pair computation."""

from contextlib import nullcontext
from types import SimpleNamespace

from phase1_web import article_comparison


class UI:
    def __init__(self, selected):
        self.selected = selected
        self.errors = []
        self.matrix_options = None

    def title(self, *_args): pass
    def caption(self, *_args): pass
    def write(self, *_args): pass
    def info(self, *_args): pass
    def json(self, *_args): pass
    def dataframe(self, *_args, **_kwargs): pass
    def download_button(self, *_args, **_kwargs): pass
    def error(self, value): self.errors.append(value)
    def radio(self, _label, options, **_kwargs): return options[0]
    def selectbox(self, _label, options, **_kwargs): return options[0]
    def number_input(self, _label, **kwargs): return kwargs["value"]
    def expander(self, *_args): return nullcontext()
    def columns(self, count): return [nullcontext() for _ in range(count)]
    def multiselect(self, _label, options, **kwargs):
        self.matrix_options = kwargs
        return self.selected
    column_config = SimpleNamespace(NumberColumn=lambda *_args, **_kwargs: None)


def setup(monkeypatch, selected, public=True):
    monkeypatch.setenv("PHASE1_DEPLOYMENT", "public" if public else "local")
    calls = []
    def compare(*args, **kwargs):
        calls.append(args)
        return {"similarity": 1.0, "groups": {}, "weights": {}, "pairing": {}}
    monkeypatch.setattr(article_comparison, "compare_profiles", compare)
    library = SimpleNamespace(matching_records=[
        {"pmcid": f"DOC{i}", "data": {"_comparison_profile": {"title": f"Trial {i}"}}}
        for i in range(25)
    ])
    ui = UI(selected)
    article_comparison._render(ui, library)
    return ui, calls


def test_public_matrix_blocks_oversized_selection_before_computing(monkeypatch):
    ui, calls = setup(monkeypatch, [f"DOC{i}" for i in range(21)])
    assert ui.matrix_options["max_selections"] == 20
    assert ui.errors and "20 articles" in ui.errors[0]
    assert not calls


def test_selection_cannot_refer_to_another_library(monkeypatch):
    ui, calls = setup(monkeypatch, ["ANOTHER_VISITORS_ARTICLE"])
    assert ui.errors and "unavailable" in ui.errors[0]
    assert not calls


def test_local_comparison_keeps_existing_unlimited_behavior(monkeypatch):
    ui, calls = setup(monkeypatch, [f"DOC{i}" for i in range(21)], public=False)
    assert "max_selections" not in ui.matrix_options
    assert not ui.errors
    assert len(calls) == 21 * 22 // 2 + 1
