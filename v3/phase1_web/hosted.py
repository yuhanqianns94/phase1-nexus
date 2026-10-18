"""Public-site session boundaries. Local desktop defaults remain unchanged."""
from __future__ import annotations

import os
import re
import secrets
import shutil
import time
from pathlib import Path


def is_public() -> bool:
    return os.environ.get("PHASE1_DEPLOYMENT", "local").lower() == "public"


def data_root() -> Path:
    return Path(os.environ.get("PHASE1_DATA_ROOT", "/var/data/phase1")).expanduser()


def validate_public_environment() -> None:
    if is_public() and any(os.environ.get(name) for name in (
        "OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENALEX_API_KEY",
    )):
        raise RuntimeError("Public mode must not have a site-owner API key in its environment. Remove server API credentials before starting.")


def valid_email(value: str) -> bool:
    return len(value) <= 254 and bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value))


def session_workspace(state, root: Path | None = None) -> Path:
    """An email address is never used as an authorization or filesystem key."""
    token = state.get("public_session_id")
    if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{64}", token):
        token = secrets.token_hex(32)
        state["public_session_id"] = token
    workspace = (root or data_root()) / "sessions" / token
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    (workspace / ".active").touch()
    return workspace


def cleanup_sessions(root: Path | None = None, *, max_age_hours: int = 24) -> int:
    """Remove expired anonymous workspaces; active jobs hold a separate lease."""
    removed = 0
    cutoff = time.time() - max_age_hours * 3600
    for path in ((root or data_root()) / "sessions").glob("*"):
        if not re.fullmatch(r"[a-f0-9]{64}", path.name) or path.is_symlink():
            continue
        marker = path / ".active"
        try:
            lease = path / ".job-active"
            # Tasks have at most 30 minutes waiting plus 30 minutes running.
            # Old leases left by a crashed process must not retain files forever.
            active_job = lease.exists() and lease.stat().st_mtime > time.time() - 2 * 3600
            if marker.stat().st_mtime < cutoff and not active_job:
                shutil.rmtree(path)
                removed += 1
        except FileNotFoundError:
            pass
    return removed


class SessionNative:
    def __init__(self, native, workspace, release, source):
        self._native = native
        self.WORKSPACE = workspace
        self.STATE_ROOT = workspace / ("shared_state" if source == "shared" else "personal_state")
        self.STATE_ROOT.mkdir(exist_ok=True)
        self.SETTINGS_PATH = self.STATE_ROOT / "workflow_settings.json"
        self.POSTERIOR_ROOT = workspace / "source_posteriors"
        self.SHARED_POSTERIOR_ROOT = release / "source_posteriors" if release and source == "shared" else None
        self.PUBLIC = True
        self.LIBRARY_SOURCE = source
        self._roots = ([release / "library"] if release else []) if source == "shared" else [workspace / "library"]

    def __getattr__(self, name):
        return getattr(self._native, name)

    def _initial_settings(self):
        return {"extraction_roots": [str(p) for p in self._roots],
                "evidence_policy": "project_default", "review_policy": "extracted"}


def public_session(native, st):
    if not is_public():
        return native
    validate_public_environment()
    if not st.session_state.get("public_email"):
        st.title("Welcome to Phasal")
        st.write("Explore Phase I evidence and dose–toxicity risks.")
        with st.form("public_welcome"):
            email = st.text_input("Email address", placeholder="you@example.com", max_chars=254).strip()
            st.caption("Your email stays in this browser session. It is not a verified account and does not give access to another session's files.")
            submitted = st.form_submit_button("Continue", type="primary")
        if submitted:
            if valid_email(email):
                st.session_state["public_email"] = email
                st.rerun()
            st.error("Enter a valid email address.")
        st.stop()
    workspace = session_workspace(st.session_state)
    from phase1_project.shared_release import resolve_shared_release
    release = resolve_shared_release(data_root() / "shared")
    source = st.sidebar.radio("Evidence library", ["shared", "personal"],
        format_func=lambda key: "Shared research library" if key == "shared" else "My uploaded literature",
        key="public_library_source")
    identity = (source, str(release) if source == "shared" else str(workspace))
    if st.session_state.get("public_library_identity") != identity:
        # Preserve only this browser's identity/upload controls across library changes.
        for key in list(st.session_state):
            if not str(key).startswith(("public_", "upload_")):
                del st.session_state[key]
        st.session_state["public_library_identity"] = identity
    st.session_state["public_shared_posterior_root"] = str(release / "source_posteriors") if release and source == "shared" else None
    st.sidebar.caption("Private files are isolated to this browser session. Download results before leaving. Inactive files are removed after 24 hours.")
    if source == "shared" and release is None:
        st.info("The shared research library is being prepared. It will appear here after the owner publishes completed extractions. You can also use your own literature.")
    if st.sidebar.button("Clear my session and files", key="public_clear"):
        if (workspace / ".job-active").exists():
            st.sidebar.warning("Wait for your current task to finish before clearing files.")
        else:
            shutil.rmtree(workspace)
            st.session_state.clear()
            st.rerun()
    return SessionNative(native, workspace, release, source)
