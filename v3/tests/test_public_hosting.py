import json
import os
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phase1_web import hosted


def test_email_is_not_a_storage_identity(tmp_path):
    a, b = {"public_email": "same@example.com"}, {"public_email": "same@example.com"}
    first = hosted.session_workspace(a, tmp_path)
    second = hosted.session_workspace(b, tmp_path)
    assert first != second
    assert hosted.session_workspace(a, tmp_path) == first
    assert "same" not in str(first)
    assert not any("same@example.com" in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


def test_cleanup_preserves_active_job_and_recent_session(tmp_path):
    expired = hosted.session_workspace({}, tmp_path)
    active = hosted.session_workspace({}, tmp_path)
    recent = hosted.session_workspace({}, tmp_path)
    then = time.time() - 25 * 3600
    for path in [expired, active]:
        os.utime(path / ".active", (then, then))
    (active / ".job-active").touch()
    assert hosted.cleanup_sessions(tmp_path) == 1
    assert not expired.exists() and active.exists() and recent.exists()


def test_public_mode_rejects_owner_key_without_changing_local_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "owner-test-token")
    monkeypatch.setenv("PHASE1_DEPLOYMENT", "local")
    hosted.validate_public_environment()
    monkeypatch.setenv("PHASE1_DEPLOYMENT", "public")
    with pytest.raises(RuntimeError, match="site-owner"):
        hosted.validate_public_environment()
    assert os.environ["OPENAI_API_KEY"] == "owner-test-token"


def _public_app(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE1_DEPLOYMENT", "public")
    monkeypatch.setenv("PHASE1_DATA_ROOT", str(tmp_path))
    for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENALEX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return AppTest.from_file(str(Path(__file__).parents[1] / "phase1_web/app.py"), default_timeout=30)


def _enter(app, email="reader@example.com"):
    app.run()
    app.text_input[0].set_value(email)
    app.button[0].click().run()
    assert not app.exception
    return app


def test_public_gate_and_private_drafts_for_same_email(tmp_path, monkeypatch):
    a = _public_app(tmp_path, monkeypatch).run()
    assert a.title[0].value == "Welcome to Phasal"
    a.text_input[0].set_value("invalid")
    a.button[0].click().run()
    assert a.error and not a.exception
    a.text_input[0].set_value("reader@example.com")
    a.button[0].click().run()
    assert not a.exception
    b = _enter(_public_app(tmp_path, monkeypatch))
    wa = tmp_path / "sessions" / a.session_state["public_session_id"]
    wb = tmp_path / "sessions" / b.session_state["public_session_id"]
    assert wa != wb
    assert a.radio(key="public_library_source").options == ["Shared research library", "My uploaded literature"]
    assert "Settings" not in a.radio(key="workflow_page").options
    assert "Extraction jobs" not in a.radio(key="workflow_page").options
    next(widget for widget in a.text_input if widget.label == "Drug name or development code").set_value("Session A drug").run()
    assert not a.exception
    b.run()
    assert not b.exception
    assert next(widget for widget in b.text_input if widget.label == "Drug name or development code").value != "Session A drug"
    assert "Session A drug" in (wa / "shared_state/multidrug_target_draft.json").read_text()
    assert "Session A drug" not in (wb / "shared_state/multidrug_target_draft.json").read_text()


def test_personal_library_shows_upload_and_requires_own_key(tmp_path, monkeypatch):
    app = _enter(_public_app(tmp_path, monkeypatch))
    app.radio(key="public_library_source").set_value("personal").run()
    assert not app.exception
    app.radio(key="workflow_page").set_value("Upload literature").run()
    assert not app.exception
    assert any("API key" in widget.label for widget in app.text_input)
    assert not any("owner" in widget.value for widget in app.text_input if isinstance(widget.value, str))


def test_public_similarity_rejects_server_api_even_outside_ui(monkeypatch):
    from phase1_prediction.similarity_engine import similarity_context
    monkeypatch.setenv("PHASE1_DEPLOYMENT", "public")
    with pytest.raises(ValueError, match="local similarity"):
        with similarity_context("openai"):
            pytest.fail("Public embedding API must not run")


def test_published_cache_is_read_only_fallback(tmp_path, monkeypatch):
    from test_multidrug_transport import fixture
    from phase1_prediction.multidrug.contracts import PriorSpec, SamplerSpec
    from phase1_prediction.multidrug.store import PosteriorStore, posterior_cache_context, source_fit_id
    scope, arrays, manifest, _ = fixture()
    prior, sampler = PriorSpec(), SamplerSpec()
    fid = source_fit_id(scope, "logistic", prior, sampler)
    shared, private = tmp_path / "shared", tmp_path / "private"
    PosteriorStore(shared).save(fid, scope, arrays, manifest)
    before = {str(p.relative_to(shared)): p.read_bytes() for p in shared.rglob("*") if p.is_file()}
    def fail_fit(*args, **kwargs):
        pytest.fail("A compatible published fit must not be refitted")
    monkeypatch.setattr("phase1_prediction.multidrug.fitting.fit_source", fail_fit)
    with posterior_cache_context([shared]):
        loaded, _, metadata = PosteriorStore(private).fit(scope, "logistic", prior=prior, sampler=sampler)
    assert loaded == scope and metadata["source_fit_id"] == fid
    assert not private.exists()
    assert before == {str(p.relative_to(shared)): p.read_bytes() for p in shared.rglob("*") if p.is_file()}
    with pytest.raises(FileNotFoundError):
        PosteriorStore(private).load(fid)


def test_public_worker_real_process_and_no_persisted_payload(tmp_path):
    from phase1_web.jobs import run_task
    with pytest.raises(RuntimeError, match="Unknown calculation"):
        run_task("not-a-job", {"api_key": "secret-only-in-memory"}, tmp_path, timeout=20)
    assert not (tmp_path / ".job-active").exists()
    assert all(b"secret-only-in-memory" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())


def test_public_library_switch_never_uses_local_or_other_session_roots(tmp_path, monkeypatch):
    from phase1_project.shared_release import pack_shared_library, install_shared_library
    from test_shared_release import article
    article(tmp_path / "owner", "OWNER_PRIVATE")
    article(tmp_path / "export", "PUBLIC_ARTICLE")
    monkeypatch.setenv("PHASE1_V3_EXTRACTION_ROOTS", str(tmp_path / "owner"))
    monkeypatch.setattr("phase1_project.catalog.default_extraction_roots", lambda: [tmp_path / "owner"])
    # With no release, the public catalogue must stay empty even on the owner's machine.
    app = _enter(_public_app(tmp_path / "data", monkeypatch))
    app.radio(key="workflow_page").set_value("Library").run()
    assert not app.exception
    assert app.metric[0].value == "0"
    pack_shared_library([tmp_path / "export"], tmp_path / "release.tar.gz")
    install_shared_library(tmp_path / "release.tar.gz", tmp_path / "data/shared")
    app.run()
    app.radio(key="workflow_page").set_value("Library").run()
    assert not app.exception and app.metric[0].value == "1"
    assert app.selectbox[2].options == ["PUBLIC_ARTICLE"]
    workspace = tmp_path / "data/sessions" / app.session_state["public_session_id"]
    article(workspace / "library", "PERSONAL_ARTICLE")
    app.radio(key="public_library_source").set_value("personal").run()
    app.radio(key="workflow_page").set_value("Library").run()
    assert not app.exception
    assert app.selectbox[2].options == ["PERSONAL_ARTICLE"]
    assert app.session_state["v3_settings"]["extraction_roots"] == [str(workspace / "library")]
    app.radio(key="public_library_source").set_value("shared").run()
    assert str(workspace / "library") not in app.session_state["v3_settings"]["extraction_roots"]


def test_public_simulation_work_limit_precedes_calculation(tmp_path):
    from phase1_web.jobs import _dispatch
    with pytest.raises(ValueError, match="20,000 trials"):
        _dispatch("simulate", {"options": {"posterior_curve_count": 1000, "trials_per_curve": 1000}}, tmp_path)


def test_shared_bootstrap_verifies_download_before_activation(tmp_path, monkeypatch):
    import io
    from phase1_project.shared_release import pack_shared_library, resolve_shared_release
    from phase1_web import server
    from test_shared_release import article
    article(tmp_path / "source")
    report = pack_shared_library([tmp_path / "source"], tmp_path / "bundle.tar.gz")
    data = (tmp_path / "bundle.tar.gz").read_bytes()
    monkeypatch.setenv("PHASE1_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("PHASE1_SHARED_BUNDLE_URL", "https://example.com/library.tar.gz")
    monkeypatch.setenv("PHASE1_SHARED_BUNDLE_SHA256", "0" * 64)
    class Response(io.BytesIO):
        def geturl(self):
            return "https://example.com/library.tar.gz"
    monkeypatch.setattr(server, "urlopen", lambda *args, **kwargs: Response(data))
    with pytest.raises(ValueError, match="checksum mismatch"):
        server.bootstrap_shared_library()
    assert resolve_shared_release(tmp_path / "data/shared") is None
    monkeypatch.setenv("PHASE1_SHARED_BUNDLE_SHA256", report["bundle_sha256"])
    server.bootstrap_shared_library()
    release = resolve_shared_release(tmp_path / "data/shared")
    assert release and (release / "library/DOC/curve_rows.json").is_file()
    monkeypatch.setattr(server, "urlopen", lambda *args, **kwargs: pytest.fail("Unchanged release must reuse its installation"))
    server.bootstrap_shared_library()
