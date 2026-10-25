"""Publication boundaries: private files and credentials must stay local."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/publish_website.py"
spec = importlib.util.spec_from_file_location("publish_website_under_test", MODULE_PATH)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "private" / "v3"
    write(source, "phase1_web/app.py", 'print("public application")\n')
    write(source, "pyproject.toml", '[project]\nname = "test"\n')
    return source


def test_export_excludes_private_data_documents_and_populated_env(source):
    secret = "sk-proj-" + "q" * 50
    for name in (
        ".env", ".env.example", ".streamlit/secrets.toml",
        "output/library/article.json", "docs/PRIVATE.md", "README.md",
        "tests/real_payload.json", "scripts/logs.txt", ".venv/credentials.json",
        "phase1_web/__pycache__/private.py", "phase1_web/.env.txt",
    ):
        write(source, name, secret)
    exported = publisher.source_export(source)
    assert set(exported) == {".env.example", "v3/.env.example", "v3/pyproject.toml", "v3/phase1_web/app.py"}
    assert all(secret.encode() not in value for value in exported.values())
    assert b"OPENAI_API_KEY=\n" in exported[".env.example"]


def test_rejects_symlink_into_private_data(source, tmp_path):
    private = write(tmp_path, "credentials.txt", "private")
    (source / "phase1_web/linked.py").symlink_to(private)
    with pytest.raises(publisher.PublicationError, match="Symlinks"):
        publisher.source_export(source)


def test_secret_scan_redacts_provider_key_and_known_environment_secret(monkeypatch):
    value = "sk-proj-" + "r" * 50
    with pytest.raises(publisher.PublicationError) as error:
        publisher.builtin_secret_scan({"app.py": value.encode()})
    assert value not in str(error.value)
    assert "app.py" in str(error.value)
    monkeypatch.setenv("CUSTOM_API_KEY", "private-test-value-123456789")
    with pytest.raises(publisher.PublicationError):
        publisher.builtin_secret_scan({"config.py": b"private-test-value-123456789"})


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    checkout = tmp_path / "public"
    checkout.mkdir()
    git(checkout, "init", "-b", "main")
    git(checkout, "config", "user.email", "test@example.invalid")
    git(checkout, "config", "user.name", "Publication Test")
    git(checkout, "remote", "add", "origin", "https://github.com/yuhanqianns94/phase1-nexus.git")
    write(checkout, "README.md", "Public features only\n")
    write(checkout, "v3/README.md", "Public v3 features only\n")
    write(checkout, "v3/phase1_web/app.py", "old\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Initial public source")
    return checkout


def test_default_check_does_not_write_or_contact_remote(source, checkout, monkeypatch):
    monkeypatch.setattr(publisher, "scanner_path", lambda *_: None)
    before = (checkout / "v3/phase1_web/app.py").read_bytes()
    assert publisher.main(["--source", str(source), "--checkout", str(checkout)]) == 0
    assert (checkout / "v3/phase1_web/app.py").read_bytes() == before
    assert not (checkout / ".env.example").exists()
    assert git(checkout, "status", "--porcelain").stdout == b""


def test_stage_preserves_feature_readme_and_refuses_overwrite_of_manual_edit(source, checkout, monkeypatch):
    monkeypatch.setattr(publisher, "scanner_path", lambda *_: None)
    args = ["--source", str(source), "--checkout", str(checkout), "--stage"]
    write(checkout, "v3/phase1_web/app.py", "a separate public edit\n")
    assert publisher.main(args) == 1
    assert (checkout / "v3/phase1_web/app.py").read_text() == "a separate public edit\n"
    git(checkout, "checkout", "--", "v3/phase1_web/app.py")
    assert publisher.main(args) == 0
    assert (checkout / "README.md").read_text() == "Public features only\n"
    assert (checkout / "v3/README.md").read_text() == "Public v3 features only\n"
    assert (checkout / "v3/phase1_web/app.py").read_bytes() == (source / "phase1_web/app.py").read_bytes()
    # Staging for review does not create a Git commit or stage Git's index.
    assert git(checkout, "rev-list", "--count", "HEAD").stdout.strip() == b"1"
    assert git(checkout, "diff", "--cached", "--name-only").stdout == b""


def test_push_requires_full_scanner_before_changing_checkout(source, checkout, monkeypatch):
    monkeypatch.setattr(publisher, "scanner_path", lambda *_: None)
    assert publisher.main(["--source", str(source), "--checkout", str(checkout), "--push"]) == 1
    assert git(checkout, "status", "--porcelain").stdout == b""


def test_credential_bearing_remote_is_rejected_without_echoing_secret(source, checkout):
    token = "private-credential-must-not-appear"
    git(checkout, "remote", "set-url", "origin", f"https://{token}@github.com/yuhanqianns94/phase1-nexus.git")
    with pytest.raises(publisher.PublicationError) as error:
        publisher.validate_checkout(checkout, source, "yuhanqianns94/phase1-nexus")
    assert token not in str(error.value)


def test_new_remote_commits_stop_push_before_export(source, checkout, monkeypatch):
    original_git = publisher.git
    original_run = subprocess.run
    monkeypatch.setattr(publisher, "scanner_path", lambda *_: "/test/scanner")
    monkeypatch.setattr(publisher, "gitleaks_scan", lambda *_args, **_kwargs: None)

    def fake_git(root, *args, **kwargs):
        if args[0] == "fetch":
            return ""
        return original_git(root, *args, **kwargs)

    def fake_run(command, **kwargs):
        if "merge-base" in command:
            return subprocess.CompletedProcess(command, 1, b"", b"")
        return original_run(command, **kwargs)

    monkeypatch.setattr(publisher, "git", fake_git)
    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    assert publisher.main(["--source", str(source), "--checkout", str(checkout), "--push"]) == 1
    assert original_git(checkout, "status", "--porcelain") == ""
