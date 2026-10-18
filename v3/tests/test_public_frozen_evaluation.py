"""Published source fits remain usable when visitors change BOIN coordinates."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from phase1_web import jobs


def _shared_prediction(tmp_path, monkeypatch):
    from test_multidrug_transport import fixture
    from phase1_prediction.multidrug import service
    from phase1_prediction.multidrug.contracts import PriorSpec, SamplerSpec
    from phase1_prediction.multidrug.store import PosteriorStore, posterior_cache_context, source_fit_id

    scope, arrays, manifest, target = fixture()
    shared, workspace = tmp_path / "published/source_posteriors", tmp_path / "visitor"
    private = workspace / "source_posteriors"
    fit_id = source_fit_id(scope, "logistic", PriorSpec(), SamplerSpec())
    PosteriorStore(shared).save(fit_id, scope, arrays, manifest)
    monkeypatch.setattr(service, "load_model_library", lambda *args, **kwargs: ([scope], {"notices": []}))
    library = SimpleNamespace(source_roots=[], evidence_policy="project_default", review_policy="extracted")
    with posterior_cache_context([shared]):
        prediction = service.predict(library, target, store_root=private, draw_count=10).to_dict()
    assert next(iter(prediction["methods"].values()))["status"] == "fitted"
    assert not private.exists()
    candidates = target.model_copy(deep=True)
    candidates.points[0].doses["A"] = 30.0
    return prediction, candidates, shared, workspace


def test_real_frozen_evaluation_reads_published_fit_without_private_copy(tmp_path, monkeypatch):
    from phase1_prediction.multidrug import service

    prediction, candidates, shared, workspace = _shared_prediction(tmp_path, monkeypatch)
    before = {str(path.relative_to(shared)): path.read_bytes() for path in shared.rglob("*") if path.is_file()}
    # The original direct UI path cannot find a fit that exists only in the
    # published fallback. This makes the regression exercise an actual miss.
    with pytest.raises(FileNotFoundError):
        service.evaluate_frozen_prediction(prediction, candidates)
    result = jobs._dispatch("evaluate", {"prediction": prediction,
        "target": candidates.model_dump(mode="json"), "shared_posterior_root": str(shared)}, workspace)
    assert result["target"]["points"][0]["doses"]["A"] == 30.0
    assert result["run_id"] != prediction["run_id"]
    assert next(iter(result["methods"].values()))["status"] == "fitted"
    assert not (workspace / "source_posteriors").exists()
    assert before == {str(path.relative_to(shared)): path.read_bytes() for path in shared.rglob("*") if path.is_file()}
    # Context restoration prevents accidentally reusing a prior visitor's root.
    with pytest.raises(FileNotFoundError):
        service.evaluate_frozen_prediction(prediction, candidates)


def test_frozen_evaluation_works_in_real_public_worker(tmp_path, monkeypatch):
    prediction, candidates, shared, workspace = _shared_prediction(tmp_path, monkeypatch)
    monkeypatch.setenv("PHASE1_DEPLOYMENT", "public")
    result = jobs.run_task("evaluate", {"prediction": prediction,
        "target": candidates.model_dump(mode="json"), "shared_posterior_root": str(shared)}, workspace, timeout=30)
    assert result["target"]["points"][0]["doses"]["A"] == 30.0
    assert not (workspace / "source_posteriors").exists()
    assert not (workspace / ".job-active").exists()


def test_ui_helper_passes_current_private_workspace_and_exact_shared_release(tmp_path, monkeypatch):
    import streamlit as st
    from test_multidrug_transport import fixture

    target = fixture()[3]
    monkeypatch.setenv("PHASE1_DEPLOYMENT", "public")
    monkeypatch.setenv("PHASE1_DATA_ROOT", str(tmp_path))
    state = {"public_session_id": "a" * 64,
             "public_shared_posterior_root": str(tmp_path / "shared/releases/exact/source_posteriors")}
    monkeypatch.setattr(st, "session_state", state)
    captured = []
    monkeypatch.setattr(jobs, "run_task", lambda *args, **kwargs: captured.append((args, kwargs)) or {"ok": True})
    assert jobs.evaluate_for_ui({"run_id": "prior"}, target) == {"ok": True}
    args, _ = captured[0]
    assert args[0] == "evaluate"
    assert args[1]["shared_posterior_root"] == state["public_shared_posterior_root"]
    assert args[1]["target"] == target.model_dump(mode="json")
    assert Path(args[2]) == tmp_path / "sessions" / ("a" * 64)


def test_ui_helper_preserves_local_evaluation_path(monkeypatch):
    from phase1_prediction.multidrug import service

    monkeypatch.setenv("PHASE1_DEPLOYMENT", "local")
    captured = []
    monkeypatch.setattr(service, "evaluate_frozen_prediction", lambda *args: captured.append(args) or {"ok": True})
    monkeypatch.setattr(jobs, "run_task", lambda *args, **kwargs: pytest.fail("Local evaluation must stay local"))
    prediction, target = {"run_id": "existing"}, object()
    assert jobs.evaluate_for_ui(prediction, target) == {"ok": True}
    assert captured == [(prediction, target)]
