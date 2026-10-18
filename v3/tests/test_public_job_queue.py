"""Bounded queue and workspace leases, without processes, models or paid APIs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest

from phase1_web import jobs


class FakeConnection:
    def __init__(self, state):
        self.state = state
        self.closed = False

    def poll(self, timeout):
        return "result" in self.state

    def recv(self):
        return self.state["result"]

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, controller, state, args):
        self.controller, self.state, self.args = controller, state, args
        self.pid = None

    def start(self):
        self.pid = 123
        _, name, payload, workspace = self.args
        controller = self.controller
        with controller.guard:
            controller.active += 1
            controller.maximum_active = max(controller.maximum_active, controller.active)
            controller.started.append(payload["session"])
        try:
            if payload.get("hold"):
                controller.first_started.set()
                if not controller.release_first.wait(timeout=3):
                    raise RuntimeError("Test worker was not released within its bounded wait")
            self.state["result"] = True, {
                "session": payload["session"], "value": payload["value"],
                "workspace": workspace, "task": name,
            }
        finally:
            with controller.guard:
                controller.active -= 1

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass

    def terminate(self):
        raise AssertionError("Completed fake process should not be terminated")

    def kill(self):
        raise AssertionError("Completed fake process should not be killed")


class FakeContext:
    def __init__(self):
        self.guard = threading.Lock()
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.started = []
        self.active = 0
        self.maximum_active = 0
        self.local = threading.local()

    def Pipe(self, duplex):
        assert duplex is False
        self.local.state = {}
        return FakeConnection(self.local.state), FakeConnection(self.local.state)

    def Process(self, *, target, args, daemon):
        assert target is jobs._worker and daemon is True
        return FakeProcess(self, self.local.state, args)


@pytest.fixture
def queue(monkeypatch):
    controller = FakeContext()
    monkeypatch.setattr(jobs, "_QUEUE", threading.BoundedSemaphore(5))
    monkeypatch.setattr(jobs, "_COMPUTE", threading.Lock())

    def context(method):
        assert method == "spawn"
        return controller

    monkeypatch.setattr(jobs.multiprocessing, "get_context", context)
    yield controller
    # Every event wait and run_task timeout is bounded even if an assertion fails.
    controller.release_first.set()


def payload(session, *, hold=False):
    return {"session": session, "value": f"private-result-{session}", "hold": hold}


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Expected queue state did not appear within the bounded wait")
        time.sleep(.005)


def assert_capacity_released():
    acquired = 0
    try:
        for _ in range(5):
            assert jobs._QUEUE.acquire(blocking=False), "An admission slot leaked"
            acquired += 1
        assert not jobs._QUEUE.acquire(blocking=False)
    finally:
        for _ in range(acquired):
            jobs._QUEUE.release()
    assert jobs._COMPUTE.acquire(blocking=False), "The compute lock leaked"
    jobs._COMPUTE.release()


def test_queued_job_holds_lease_and_duplicate_session_is_refused(queue, tmp_path):
    running, queued = tmp_path / "running", tmp_path / "queued"
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(jobs.run_task, "predict", payload("first", hold=True), running, timeout=2)
        try:
            assert queue.first_started.wait(timeout=2)
            second = executor.submit(jobs.run_task, "simulate", payload("second"), queued, timeout=2)
            wait_until(lambda: (queued / ".job-active").exists())
            assert (running / ".job-active").exists()
            assert queue.started == ["first"]
            for workspace in (running, queued):
                with pytest.raises(RuntimeError, match="already has a queued or running task"):
                    jobs.run_task("predict", payload("duplicate"), workspace, timeout=.1)
                assert (workspace / ".job-active").exists()
        finally:
            queue.release_first.set()
        assert first.result(timeout=2)["session"] == "first"
        assert second.result(timeout=2)["session"] == "second"
    assert queue.started == ["first", "second"]
    for workspace in (running, queued):
        assert not (workspace / ".job-active").exists()
        assert (workspace / ".active").is_file()
    assert_capacity_released()


def test_five_sessions_are_admitted_then_run_sequentially_with_isolated_results(queue, tmp_path):
    workspaces = [tmp_path / f"session-{index}" for index in range(5)]
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(jobs.run_task, "predict", payload("0", hold=True), workspaces[0], timeout=2)]
        try:
            assert queue.first_started.wait(timeout=2)
            futures.extend(executor.submit(jobs.run_task, "predict", payload(str(index)), workspace, timeout=2)
                           for index, workspace in enumerate(workspaces[1:], start=1))
            wait_until(lambda: all((workspace / ".job-active").exists() for workspace in workspaces))
            assert queue.started == ["0"]
            with pytest.raises(RuntimeError, match="All five calculation slots"):
                jobs.run_task("predict", payload("sixth"), tmp_path / "sixth", timeout=.1)
            assert not (tmp_path / "sixth").exists()
        finally:
            queue.release_first.set()
        results = [future.result(timeout=2) for future in futures]
    assert queue.maximum_active == 1
    assert set(queue.started) == {str(index) for index in range(5)}
    for index, (workspace, result) in enumerate(zip(workspaces, results)):
        assert result == {"session": str(index), "value": f"private-result-{index}",
                          "workspace": str(workspace), "task": "predict"}
        assert not (workspace / ".job-active").exists()
        assert (workspace / ".active").is_file()
    assert_capacity_released()


def test_workspace_creation_failure_does_not_leak_capacity(queue, tmp_path, monkeypatch):
    workspace = tmp_path / "unwritable"
    original = Path.mkdir

    def mkdir(path, *args, **kwargs):
        if path == workspace:
            raise PermissionError("Injected workspace creation failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(PermissionError, match="workspace creation"):
        jobs.run_task("predict", payload("failed"), workspace, timeout=.1)
    assert queue.started == []
    assert_capacity_released()
    assert jobs.run_task("predict", payload("next"), tmp_path / "healthy", timeout=.1)["session"] == "next"
    assert_capacity_released()


@pytest.mark.parametrize("failure", ["lease_removal", "heartbeat"])
def test_workspace_cleanup_failure_releases_compute_lock_and_admission(queue, tmp_path, monkeypatch, failure):
    workspace = tmp_path / "cleanup-failure"
    operation = "unlink" if failure == "lease_removal" else "touch"
    target = workspace / (".job-active" if failure == "lease_removal" else ".active")
    original = getattr(Path, operation)

    def fail(path, *args, **kwargs):
        if path == target:
            raise PermissionError("Injected cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, fail)
    with pytest.raises(PermissionError, match="cleanup failure"):
        jobs.run_task("predict", payload("failed"), workspace, timeout=.1)
    assert queue.started == ["failed"]
    assert_capacity_released()
    result = jobs.run_task("predict", payload("next"), tmp_path / "healthy", timeout=.1)
    assert result["session"] == "next"
    assert_capacity_released()


def test_queue_wait_timeout_releases_queued_lease_and_admission(queue, tmp_path):
    workspace = tmp_path / "timed-out"
    assert jobs._COMPUTE.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="queue timed out"):
            jobs.run_task("predict", payload("timeout"), workspace, timeout=.01)
        assert not (workspace / ".job-active").exists()
        assert queue.started == []
    finally:
        jobs._COMPUTE.release()
    assert_capacity_released()
