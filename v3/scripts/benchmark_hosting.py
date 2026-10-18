#!/usr/bin/env python3
"""Bounded, credential-free memory benchmark; uses only temporary workspaces.

Runs five public AppTest sessions, three uncached source fits with the unchanged
SamplerSpec defaults, and 10,000 BOIN trials in separate processes. This is a local
capacity estimate, not a Linux production load test or a future-library guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def rss_bytes():
    try:
        value = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(value.strip()) * 1024
    except (OSError, subprocess.SubprocessError):
        # Sandboxed macOS may deny process-list inspection. Peak RSS remains
        # available through getrusage and is the capacity metric used below.
        return None


def fixture(k):
    from phase1_prediction.multidrug.contracts import DrugSlot, ModelScope, Observation, Exposure, TargetRegimen, TargetDrug, RegimenPoint
    slots = [DrugSlot(model_drug_id=f"D{i}", name=f"Synthetic drug {i}",
        research_priority=i + 1, research_role="primary" if i == 0 else "partner",
        mode="numeric", reference_dose=100, unit="mg",
        profile={"name": f"Synthetic drug {i}", "mechanisms": ["synthetic kinase mechanism"], "model_drug_id": f"D{i}"}) for i in range(k)]
    doses = [25., 50., 75., 100., 150., 200.]
    observations = [Observation(observation_id=f"O{j}", cohort_id=f"C{j}", n=6, y=y,
        exposures=[Exposure(drug_id=f"D{i}", presence="present", dose=doses[(j + i) % len(doses)], dose_status="reported") for i in range(k)])
        for j, y in enumerate([0, 0, 1, 1, 2, 3])]
    scope = ModelScope(document_id="SYNTHETIC_BENCHMARK", scope_id=f"synthetic_{k}_drugs",
        label="Synthetic performance fixture", drugs=slots, observations=observations)
    target = TargetRegimen(name="Synthetic benchmark target", drugs=[TargetDrug(
        drug_id=s.model_drug_id, name=s.name, research_priority=s.research_priority,
        research_role=s.research_role, mode=s.mode, reference_dose=s.reference_dose,
        unit=s.unit, profile=s.profile) for s in slots], points=[RegimenPoint(
        level_id=f"L{j}", doses={s.model_drug_id: dose for s in slots}) for j, dose in enumerate(doses)],
        path_level_ids=[f"L{j}" for j in range(len(doses))])
    return scope, target


def fit_benchmark(k):
    from phase1_prediction.multidrug.contracts import SamplerSpec, PriorSpec
    from phase1_prediction.multidrug.fitting import fit_source
    scope, _ = fixture(k)
    sampler = SamplerSpec()
    started = time.monotonic()
    arrays, diagnostics = fit_source(scope, "logistic", PriorSpec(), sampler)
    return {"kind": "uncached_source_fit", "synthetic": True, "drugs": k,
        "cohorts": len(scope.observations), "backend": "logistic",
        "sampler": sampler.model_dump(), "elapsed_seconds": time.monotonic() - started,
        "diagnostics": diagnostics, "posterior_bytes": sum(a.nbytes for a in arrays.values()),
        "peak_rss_bytes": peak_bytes(), "final_rss_bytes": rss_bytes()}


def boin_benchmark():
    import numpy as np
    from phase1_prediction.multidrug import TARGET_PREDICTION_VERSION
    from phase1_simulation.multidrug import simulate_regimen_path
    from phase1_simulation.service import BoinSettings
    _, target = fixture(3)
    rng = np.random.default_rng(20261018)
    risks = np.clip(np.linspace(.05, .6, len(target.points))[None, :] + rng.normal(0, .025, (100, 1)), .001, .999)
    prediction = {"format": "multidrug_target_prediction_v1", "status": "fitted",
        "prediction_engine_version": TARGET_PREDICTION_VERSION, "target": target.model_dump(),
        "backend": "logistic", "synthesis": "benchmark", "posterior_draws": risks.tolist()}
    started = time.monotonic()
    result = simulate_regimen_path(prediction, settings=BoinSettings(), posterior_curve_count=100, trials_per_curve=100)
    return {"kind": "boin_simulation", "synthetic_truth": True, "drugs": 3,
        "path_levels": len(target.points), "posterior_curves": 100, "trials_per_curve": 100,
        "total_trials": result["total_trials"], "elapsed_seconds": time.monotonic() - started,
        "json_result_bytes": len(json.dumps(result).encode()),
        "peak_rss_bytes": peak_bytes(), "final_rss_bytes": rss_bytes()}


def sessions_benchmark():
    from phase1_project.catalog import default_extraction_roots
    from phase1_prediction.evidence import discover_article_paths, load_evidence_library
    from phase1_project.shared_release import pack_shared_library, install_shared_library
    from phase1_prediction.multidrug.observations import load_model_library
    from streamlit.testing.v1 import AppTest
    selected, _ = discover_article_paths(default_extraction_roots())
    completed = [p for p in selected.values() if json.loads(p.reported_extraction.read_text()).get("extraction_complete") is True]
    if not completed:
        raise ValueError("No completed selected local artifacts available; no incomplete article was promoted.")
    with tempfile.TemporaryDirectory(prefix="phase1-hosting-benchmark-") as directory:
        root = Path(directory)
        extracted = root / "completed"
        for item in completed:
            dest = extracted / item.document_id
            dest.mkdir(parents=True)
            for name in ("reported_extraction.json", "curve_rows.json"):
                shutil.copyfile(item.root / name, dest / name)
        packed = pack_shared_library([extracted], root / "shared.tar.gz")
        installed = install_shared_library(root / "shared.tar.gz", root / "data/shared")
        library_path = Path(installed["release_path"]) / "library"
        library = load_evidence_library([library_path])
        scopes, scope_info = load_model_library([library_path])
        os.environ["PHASE1_DEPLOYMENT"] = "public"
        os.environ["PHASE1_DATA_ROOT"] = str(root / "data")
        samples = []
        sessions = []
        started = time.monotonic()
        for index in range(5):
            app = AppTest.from_file(str(V3 / "phase1_web/app.py"), default_timeout=90)
            app.session_state["public_email"] = f"benchmark{index}@example.invalid"
            app.run()
            if len(app.exception):
                raise RuntimeError("Public AppTest failed: " + "; ".join(str(e.message) for e in app.exception))
            sessions.append(app)
            samples.append({"sessions": len(sessions), "current_rss_bytes": rss_bytes(), "peak_rss_bytes": peak_bytes()})
        identities = [app.session_state["public_session_id"] for app in sessions]
        return {"kind": "five_public_apptest_sessions", "synthetic": False,
            "selected_local_document_count": len(selected), "selected_completed_document_count": len(completed),
            "shared_library_document_count": library.document_count, "scope_counts": scope_info,
            "shared_processed_file_bytes": sum(p.stat().st_size for p in library_path.rglob("*") if p.is_file()),
            "bundle_bytes": (root / "shared.tar.gz").stat().st_size,
            "distinct_session_count": len(set(identities)), "samples": samples,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_bytes(), "final_rss_bytes": rss_bytes(),
            "notes": ["Five AppTest sessions retained in one process; no network, browser, TLS or Linux server overhead.",
                "Only already-completed artifacts were copied into a temporary validated bundle. No incomplete article was changed or published.",
                "The current selected library is partly incomplete; this is not a measurement of the future complete corpus."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=["sessions", "fit1", "fit2", "fit3", "boin"])
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.child:
        started = time.monotonic()
        try:
            result = sessions_benchmark() if args.child == "sessions" else boin_benchmark() if args.child == "boin" else fit_benchmark(int(args.child[-1]))
            result["completed"] = True
        except Exception as error:
            result = {"completed": False, "error_type": type(error).__name__, "error": str(error),
                "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": peak_bytes()}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["completed"] else 1
    if args.output is None:
        parser.error("--output is required")
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI_") and key != "OPENALEX_API_KEY"}
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"})
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
        "memory_metric": "resource.getrusage(RUSAGE_SELF).ru_maxrss; bytes on macOS, KiB converted to bytes on Linux",
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpyro", "numpy", "scipy", "streamlit")},
        "timeout_per_case_seconds": args.timeout, "cases": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase1-benchmark-results-") as directory:
        for case in ("sessions", "fit1", "fit2", "fit3", "boin"):
            path = Path(directory) / f"{case}.json"
            started = time.monotonic()
            command = [sys.executable, str(Path(__file__).resolve()), "--child", case, "--output", str(path)]
            try:
                result = subprocess.run(command, env=env, capture_output=True, timeout=args.timeout)
                value = json.loads(path.read_text()) if path.exists() else {"completed": False, "process_exit_code": result.returncode, "error": "Child produced no report; captured logs withheld."}
            except subprocess.TimeoutExpired:
                value = {"completed": False, "timed_out": True, "elapsed_seconds": time.monotonic() - started}
            report["cases"][case] = value
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"case": case, "completed": value.get("completed"), "elapsed_seconds": value.get("elapsed_seconds"), "peak_rss_bytes": value.get("peak_rss_bytes")}), flush=True)
    session_peak = report["cases"]["sessions"].get("peak_rss_bytes")
    worker_peak = max((report["cases"][c].get("peak_rss_bytes", 0) for c in ("fit1", "fit2", "fit3", "boin")), default=0)
    if session_peak and worker_peak:
        combined = session_peak + worker_peak
        report["capacity_estimate"] = {"sum_of_separate_session_and_largest_worker_peaks_bytes": combined,
            "recommended_initial_ram_gib": 4, "recommended_cpus": 2,
            "is_simultaneous_linux_peak_measurement": False,
            "notes": ["2 GiB may fit these small measured cases, but offers limited headroom for result retention, larger owner evidence, scientific compilation and the operating system.",
                "Start with 4 GiB and one scientific worker for 4-5 connected users; validate on the chosen Linux host before reducing the allocation.",
                "This excludes the storage size of optional visitor-uploaded corpora and is not a guarantee for the future complete owner library."]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
