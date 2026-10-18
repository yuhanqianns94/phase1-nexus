"""Bounded, isolated public computations; no API credentials are persisted."""
from __future__ import annotations

import multiprocessing
import os
import threading
import time
from pathlib import Path


_QUEUE = threading.BoundedSemaphore(5)
_COMPUTE = threading.Lock()


def evaluate_for_ui(prediction, target):
    from .hosted import is_public, session_workspace
    if not is_public():
        from phase1_prediction.multidrug.service import evaluate_frozen_prediction
        return evaluate_frozen_prediction(prediction, target)
    import streamlit as st
    return run_task("evaluate", {
        "prediction": prediction,
        "target": target.model_dump(mode="json") if hasattr(target, "model_dump") else target,
        "shared_posterior_root": st.session_state.get("public_shared_posterior_root"),
    }, session_workspace(st.session_state))


def _dispatch(name, payload, workspace):
    if name == "extract":
        from phase1_web.user_library import extract_uploaded_documents
        return extract_uploaded_documents(workspace=workspace, **payload)
    if name == "evaluate":
        from phase1_prediction.multidrug.service import evaluate_frozen_prediction
        from phase1_prediction.multidrug.store import posterior_cache_context
        from phase1_prediction.multidrug.contracts import TargetRegimen
        fallback = payload.get("shared_posterior_root")
        with posterior_cache_context([Path(fallback)] if fallback else []):
            return evaluate_frozen_prediction(payload["prediction"], TargetRegimen.model_validate(payload["target"]))
    if name in {"predict", "preview"}:
        from phase1_prediction.evidence import load_evidence_library
        from phase1_prediction.multidrug import service
        from phase1_prediction.multidrug.store import posterior_cache_context
        from phase1_prediction.multidrug.contracts import PriorSpec
        library = load_evidence_library([Path(p) for p in payload["roots"]],
            evidence_policy="project_default", review_policy="extracted")
        options = dict(payload.get("options", {}))
        if options.get("similarity_engine", "local") != "local":
            raise ValueError("The public website uses local similarity comparison.")
        if name == "preview":
            return service.preview_planned(library, payload["target"], **options)
        if options.get("prior") is not None:
            options["prior"] = PriorSpec.model_validate(options["prior"])
        options["store_root"] = Path(workspace) / "source_posteriors"
        fallback = payload.get("shared_posterior_root")
        with posterior_cache_context([Path(fallback)] if fallback else []):
            return service.predict(library, payload["target"], **options).to_dict()
    if name in {"simulate", "search"}:
        from phase1_simulation.service import BoinSettings, search_boin_parameters
        from phase1_simulation.multidrug import simulate_regimen_path
        options = dict(payload["options"])
        curves = options.get("posterior_curve_count", 100)
        repetitions = options.get("trials_per_curve", 100)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (curves, repetitions)):
            raise ValueError("Simulation counts must be positive integers.")
        trials = curves * repetitions
        candidates = 1
        if name == "search":
            for key in ("target_dlt", "cohort_size", "max_sample_size", "elimination_cutoff", "stop_rule_m"):
                candidates *= len(options[key])
        if trials > 20000 or trials * candidates > 50000:
            raise ValueError("The public website supports up to 20,000 trials per design and 50,000 trials per comparison. Reduce scenarios, repetitions, or candidate designs.")
        setting_key = "settings" if name == "simulate" else "base_settings"
        options[setting_key] = BoinSettings(**options[setting_key])
        fn = simulate_regimen_path if name == "simulate" else search_boin_parameters
        return fn(payload["prediction"], **options)
    raise ValueError("Unknown calculation task.")


def _worker(connection, name, payload, workspace):
    # A fresh worker releases JAX/NumPy allocations after every expensive job.
    for key in ("OPENAI_API_KEY", "OPENALEX_API_KEY", "OPENAI_ADMIN_KEY"):
        os.environ.pop(key, None)
    try:
        connection.send((True, _dispatch(name, payload, workspace)))
    except Exception as error:
        # Do not return provider response bodies, credentials, or tracebacks.
        if name == "extract":
            message = "Extraction did not complete. Check your API key, model access, budget, and document format. Successful cached requests can be resumed."
        else:
            message = str(error) if isinstance(error, (ValueError, RuntimeError)) else "The calculation failed. Try a smaller request or contact the site owner."
        connection.send((False, message))
    finally:
        connection.close()


def run_task(name, payload, workspace, *, timeout=1800):
    """At most five admitted jobs and one running scientific worker per server."""
    workspace = Path(workspace)
    if not _QUEUE.acquire(blocking=False):
        raise RuntimeError("All five calculation slots are occupied. Please try again after a task finishes.")
    lease = workspace / ".job-active"
    owns_lease = False
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            with lease.open("x"):
                pass
            owns_lease = True
        except FileExistsError as error:
            raise RuntimeError("This session already has a queued or running task. Wait for it to finish.") from error
        if not _COMPUTE.acquire(timeout=timeout):
            raise RuntimeError("The calculation queue timed out. Please try again.")
        try:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_worker, args=(child, name, payload, str(workspace)), daemon=True)
            try:
                process.start()
                child.close()
                deadline = time.monotonic() + timeout
                while not parent.poll(0.25):
                    if not process.is_alive():
                        raise RuntimeError("The worker stopped before returning a result. Try a smaller calculation.")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("This task exceeded the server time limit. Try a smaller calculation.")
                try:
                    success, result = parent.recv()
                except EOFError as error:
                    raise RuntimeError("The worker stopped before returning a result. Try a smaller calculation.") from error
                if not success:
                    raise RuntimeError(result)
                return result
            finally:
                parent.close()
                child.close()
                if process.pid is not None:
                    process.join(timeout=2)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
        finally:
            _COMPUTE.release()
    finally:
        try:
            if owns_lease:
                lease.unlink(missing_ok=True)
                if workspace.exists():
                    (workspace / ".active").touch()
        finally:
            _QUEUE.release()
