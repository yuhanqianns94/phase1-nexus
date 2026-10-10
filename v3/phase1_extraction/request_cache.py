"""Bind raw service responses to requests without changing extraction artifacts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .artifacts import read_json, sha256_file, utc_now, write_json


def cache_response(raw_path: Path, request_sha256: str, response: dict[str, Any]) -> None:
    # The receipt is committed last. A crash between writes leaves either the
    # old valid pair or a checksum mismatch, never an old response under a new key.
    write_json(raw_path, response)
    write_json(raw_path.with_suffix(".receipt.json"), {
        "format": "phase1_response_receipt_v1",
        "request_sha256": request_sha256,
        "response_sha256": sha256_file(raw_path),
        "saved_at_utc": utc_now(),
    })


def load_cached_response(raw_path: Path, request_sha256: str) -> dict[str, Any] | None:
    try:
        receipt_path = raw_path.with_suffix(".receipt.json")
        if not receipt_path.exists():
            return _migrate_completed_response(raw_path, request_sha256)
        receipt = read_json(receipt_path)
        if not isinstance(receipt, dict) or receipt.get("request_sha256") != request_sha256:
            return None
        if receipt.get("response_sha256") != sha256_file(raw_path):
            return None
        response = read_json(raw_path)
        return response if isinstance(response, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _migrate_completed_response(raw_path: Path, request_sha256: str) -> dict[str, Any] | None:
    """Reuse legacy runs only when their completed summary binds the service ID.

    Legacy request files alone are unsafe: they may have been overwritten by a
    failed new request. The completion summary was written after successful
    parsing, and its response ID must match the actual raw response as well.
    """

    summary = read_json(raw_path.parent.parent / "request_summary.json")
    response = read_json(raw_path)
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        return None
    if (not isinstance(response, dict) or not isinstance(response.get("id"), str)
            or not response["id"] or response.get("status") != "completed"):
        return None
    records = summary.get("requests", [summary])
    if not isinstance(records, list):
        return None
    if any(isinstance(record, dict)
           and record.get("request_sha256") == request_sha256
           and ("requests" not in summary or record.get("packet_id") == raw_path.stem)
           and record.get("response_id") == response["id"] for record in records):
        cache_response(raw_path, request_sha256, response)
        return response
    return None


@contextmanager
def artifact_lock(lock_path: Path):
    """Serialize local state mutations; OS locks are released after a crash."""

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            # Windows locks a byte range, including on a newly created file.
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl

            acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f"Artifacts are in use by another operation: {lock_path}") from error
        try:
            yield
        finally:
            release()
