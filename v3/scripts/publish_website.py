#!/usr/bin/env python3
"""Export reviewed application source to the independent public website checkout.

Default: inspect the proposed export, without modifying files or contacting GitHub.
--stage: write the export for review, without making a commit.
--push: export, scan, commit, and push main; Render deploys after CI passes.

Literature, caches, personal documents and credentials are never source exports.
Publish processed shared-library bundles separately with the release tooling.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping

PACKAGE_DIRS = (
    "phase1_acquisition", "phase1_extraction", "phase1_prediction",
    "phase1_project", "phase1_simulation", "phase1_web",
)
CODE_DIRS = (*PACKAGE_DIRS, "scripts", "tests", "tools")
CODE_SUFFIXES = {".py", ".json", ".txt"}
SOURCE_FILES = (
    "pyproject.toml", "requirements.txt", "LICENSE", ".gitignore",
    ".streamlit/config.toml", "docs/DEPLOYMENT.md",
    "examples/PMC3309850_CPE_with_EIASD.target.json",
    "examples/PMC3309850_structured_test.target.json",
    "examples/power_prior_settings.json", "examples/primary_first_pilot.json",
)
# These files belong to the public checkout and are deliberately not copied from
# the private development tree. In particular both READMEs stay feature-only.
PRESERVED_FILES = (
    "README.md", "LICENSE", ".gitignore", ".gitleaks.toml", "requirements.txt",
    "Dockerfile", ".dockerignore", "render.yaml", "requirements-deploy.txt",
    ".github/workflows/ci.yml", ".devcontainer/devcontainer.json", "v3/README.md",
)
EMPTY_ENV = (
    "# Empty template. Never commit real credentials.\n"
    "# Local development only. The public website never uses owner credentials.\n"
    "# Export variables in your shell; this file is not automatically loaded.\n"
    "OPENAI_API_KEY=\nOPENALEX_API_KEY=\n"
).encode()
SKIP_PARTS = {
    "__pycache__", ".pytest_cache", ".venv", "venv", "output", "outputs",
    "build", "dist", ".git", ".cache", "node_modules",
}
SECRET_PATTERNS = (
    re.compile(rb"\bsk-(?:proj-|svcacct-)[A-Za-z0-9_-]{32,}"),
    re.compile(rb"\b(?:ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9]{30,}"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
)


class PublicationError(RuntimeError):
    pass


def git(checkout: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args], capture_output=True, check=False,
    )
    if result.returncode:
        # Do not echo arbitrary Git diagnostics: a misconfigured credential URL
        # or hook could contain sensitive values.
        raise PublicationError(f"Git command {args[0]!r} failed (exit {result.returncode}).")
    return result.stdout if binary else result.stdout.decode().strip()


def read_public_file(path: Path) -> bytes:
    if path.is_symlink():
        raise PublicationError(f"Symlinks cannot be published: {path.name}")
    if path.stat().st_size > 2_000_000:
        raise PublicationError(f"Source file is unexpectedly large: {path.name}")
    data = path.read_bytes()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicationError(f"Only UTF-8 application source is exportable: {path.name}") from exc
    return data


def source_export(source: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {".env.example": EMPTY_ENV, "v3/.env.example": EMPTY_ENV}
    for name in SOURCE_FILES:
        path = source / name
        if path.is_file() or path.is_symlink():
            result[f"v3/{name}"] = read_public_file(path)
    for directory in CODE_DIRS:
        base = source / directory
        if base.is_symlink():
            raise PublicationError(f"Source directory cannot be a symlink: {directory}")
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(source)
            if set(relative.parts) & SKIP_PARTS:
                continue
            if path.is_symlink():
                raise PublicationError(f"Symlinks cannot be published: {relative}")
            if not path.is_file():
                continue
            if path.suffix not in CODE_SUFFIXES or path.name.startswith("."):
                continue
            if path.name.lower() in {"secrets.json", "credentials.json", "secrets.txt"}:
                raise PublicationError(f"Credential-like filename in source: {relative}")
            # Tests/tools/scripts may contain programs, but not arbitrary data.
            if directory not in PACKAGE_DIRS and path.suffix != ".py":
                continue
            result[f"v3/{relative.as_posix()}"] = read_public_file(path)
    if "v3/phase1_web/app.py" not in result or "v3/pyproject.toml" not in result:
        raise PublicationError("--source must be the original v3 directory.")
    return result


def builtin_secret_scan(files: Mapping[str, bytes]) -> None:
    known_secrets = [
        value.encode() for key, value in os.environ.items()
        if any(word in key.upper() for word in ("API_KEY", "TOKEN", "SECRET"))
        and len(value) >= 16 and value.lower() not in {"your_api_key_here", "replace_me"}
    ]
    failures = []
    for name, data in files.items():
        if any(pattern.search(data) for pattern in SECRET_PATTERNS) or any(
            value in data for value in known_secrets
        ):
            failures.append(name)
    if failures:
        raise PublicationError("Potential secret detected; values redacted. Files: " + ", ".join(failures))


def scanner_path(source: Path, explicit: str | None) -> str | None:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    installed = shutil.which("gitleaks")
    if installed:
        return installed
    previous = source.parent / "output/github-publication-20261010/scanner/gitleaks"
    return str(previous) if previous.is_file() else None


def gitleaks_scan(files: Mapping[str, bytes], scanner: str | None, required: bool) -> None:
    if not scanner:
        if required:
            raise PublicationError("--push requires Gitleaks. Install it or pass --gitleaks /path/to/gitleaks.")
        print("Gitleaks unavailable; only built-in secret checks ran. Push will require Gitleaks.")
        return
    with tempfile.TemporaryDirectory(prefix="phase1-publication-scan-") as directory:
        root = Path(directory)
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        args = [scanner, "dir", str(root), "--redact", "--no-banner", "--exit-code", "1"]
        config = root / ".gitleaks.toml"
        if config.exists():
            args.extend(["--config", str(config)])
        result = subprocess.run(args, capture_output=True, check=False)
        if result.returncode:
            raise PublicationError("Gitleaks rejected the prospective public files. No values were printed or pushed.")
    print("Gitleaks: prospective public files passed.")


def validate_checkout(checkout: Path, source: Path, repository: str) -> set[str]:
    if checkout == source or checkout == source.parent or source.is_relative_to(checkout):
        raise PublicationError("The destination must be an independent public checkout.")
    if git(checkout, "rev-parse", "--show-toplevel") != str(checkout):
        raise PublicationError("--checkout must point to the public Git repository root.")
    if git(checkout, "branch", "--show-current") != "main":
        raise PublicationError("The public checkout must be on main.")
    remote = str(git(checkout, "remote", "get-url", "origin"))
    allowed = {f"https://github.com/{repository}.git", f"https://github.com/{repository}", f"git@github.com:{repository}.git"}
    if remote not in allowed:
        raise PublicationError("Public checkout origin does not match --repository; credential-bearing URLs are not accepted.")
    return set(str(git(checkout, "ls-files", "-z")).split("\0")) - {""}


def prospective_export(source: Path, checkout: Path, tracked: set[str]) -> tuple[dict[str, bytes], set[str]]:
    files = source_export(source)
    for name in PRESERVED_FILES:
        path = checkout / name
        if path.is_file() or path.is_symlink():
            files[name] = read_public_file(path)
    unexpected = {name for name in tracked if not name.startswith("v3/") and name not in files}
    if unexpected:
        raise PublicationError("Tracked files outside the publication allowlist: " + ", ".join(sorted(unexpected)))
    # Source files removed locally are also removed from the next public commit.
    deleted = {name for name in tracked if name.startswith("v3/") and name not in files}
    return files, deleted


def changed_files(checkout: Path, files: Mapping[str, bytes]) -> list[str]:
    return [name for name, data in files.items() if not (checkout / name).is_file() or (checkout / name).read_bytes() != data]


def verify_no_overwrite(checkout: Path, files: Mapping[str, bytes], deleted: set[str], tracked: set[str]) -> None:
    for name in set(changed_files(checkout, files)) | deleted:
        path = checkout / name
        if name not in tracked or not path.exists():
            continue
        original = git(checkout, "show", f"HEAD:{name}", binary=True)
        if path.read_bytes() != original:
            raise PublicationError(f"Refusing to overwrite a separately edited public file: {name}. Review or commit it first.")


def export_files(checkout: Path, files: Mapping[str, bytes], deleted: set[str]) -> None:
    for name, data in files.items():
        path = checkout / name
        if path.is_file() and path.read_bytes() == data:
            continue
        if any(parent.is_symlink() for parent in path.parents if parent != checkout.parent):
            raise PublicationError(f"Destination contains a symlink: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        temporary.replace(path)
    for name in deleted:
        path = checkout / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def push(checkout: Path, files: Mapping[str, bytes], deleted: set[str], message: str, scanner: str) -> None:
    staged = set(str(git(checkout, "diff", "--cached", "--name-only", "-z")).split("\0")) - {""}
    allowed = set(files) | deleted
    if staged - allowed:
        raise PublicationError("The public checkout has unrelated staged files; unstage them before publishing.")
    git(checkout, "add", "-A", "--", *sorted(allowed))
    if git(checkout, "diff", "--cached", "--name-only"):
        print(git(checkout, "diff", "--cached", "--stat"))
        git(checkout, "commit", "-m", message)
    result = subprocess.run(
        [scanner, "git", str(checkout), "--redact", "--no-banner", "--log-opts=origin/main..HEAD", "--config", str(checkout / ".gitleaks.toml")],
        capture_output=True, check=False,
    )
    if result.returncode:
        raise PublicationError("Gitleaks rejected the outgoing commit history; the local commit has NOT been pushed.")
    git(checkout, "push", "origin", "HEAD:main")
    print("Pushed main. Render will deploy automatically after the GitHub checks pass.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--repository", default="yuhanqianns94/phase1-toxicity-v3")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Preview and scan only (the default).")
    modes.add_argument("--stage", action="store_true", help="Write source export to the public working tree; do not commit or push.")
    modes.add_argument("--push", action="store_true", help="Export, scan, commit and push main using existing Git credentials.")
    parser.add_argument("--message", default="Update public website from local v3")
    parser.add_argument("--gitleaks", help="Path to the Gitleaks executable.")
    args = parser.parse_args(argv)
    try:
        source = args.source.expanduser().resolve()
        checkout = (args.checkout or source.parent / "output/github-publication-20261010/phase1-toxicity-v3").expanduser().resolve()
        tracked = validate_checkout(checkout, source, args.repository)
        files, deleted = prospective_export(source, checkout, tracked)
        builtin_secret_scan(files)
        scanner = scanner_path(source, args.gitleaks)
        gitleaks_scan(files, scanner, required=args.push)
        changes = changed_files(checkout, files)
        digest = hashlib.sha256(b"".join(name.encode() + b"\0" + files[name] for name in sorted(files))).hexdigest()
        print(f"Export: {len(files)} allowed files; {len(changes)} changes; {len(deleted)} removals.")
        print(f"Export SHA256: {digest}")
        for name in sorted(changes):
            print(f"  update {name}")
        for name in sorted(deleted):
            print(f"  remove {name}")
        if not (args.stage or args.push):
            print("Check only: no files changed. Use --stage to prepare files, or --push to publish reviewed code.")
            return 0
        verify_no_overwrite(checkout, files, deleted, tracked)
        if args.push:
            # Fetch before changing local files. Never reset, merge, rebase or force
            # push a public checkout that someone edited through Codespaces.
            git(checkout, "fetch", "origin", "main")
            result = subprocess.run(["git", "-C", str(checkout), "merge-base", "--is-ancestor", "origin/main", "HEAD"], capture_output=True)
            if result.returncode:
                raise PublicationError("Remote main has newer or divergent commits. Reconcile the public checkout before publishing.")
        export_files(checkout, files, deleted)
        if args.push:
            push(checkout, files, deleted, args.message, scanner or "")
        else:
            print("Export prepared. Review git diff in the public checkout. No commit or push was made.")
        return 0
    except (PublicationError, OSError) as exc:
        print(f"Publication stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
