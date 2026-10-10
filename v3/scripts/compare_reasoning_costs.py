"""Compare low/medium on one frozen article, preserving the extraction contract.

No requests without --execute. Successful raw responses are reused; an ambiguous
network outcome is never automatically retried. Usage costs are local estimates,
not invoices. Both variants share all request fields except reasoning.effort.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

from openai import OpenAI

from phase1_extraction.artifacts import ArticlePaths, read_json, sha256_file, sha256_json, utc_now, write_json
from phase1_extraction.llm import BudgetGuard, parse_model_response, response_payload
from phase1_extraction.openai_batch import freeze_single_packet_request, persist_transport_result
from phase1_extraction.request_cache import artifact_lock, cache_response, load_cached_response


PRICING = {
    "model": "gpt-6.1-sol", "service_tier": "default", "as_of": "2026-10-07",
    "input_usd_per_million": 2.0, "cached_input_usd_per_million": 0.1,
    "cache_write_usd_per_million": 2.5, "output_usd_per_million": 10.0,
    "long_context_threshold": 272000,
    "source": "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
}


def usage_cost(usage: dict) -> dict:
    total_in, total_out = usage["input_tokens"], usage["output_tokens"]
    details = usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = details.get("cache_write_tokens")
    long = total_in > PRICING["long_context_threshold"]
    input_rate = PRICING["input_usd_per_million"] * (2 if long else 1)
    cached_rate = PRICING["cached_input_usd_per_million"] * (2 if long else 1)
    write_rate = PRICING["cache_write_usd_per_million"] * (2 if long else 1)
    output_rate = PRICING["output_usd_per_million"] * (1.5 if long else 1)
    output_cost = total_out * output_rate / 1e6
    ordinary = total_in - cached - int(written or 0)
    low_cost = (ordinary * input_rate + cached * cached_rate + int(written or 0) * write_rate) / 1e6 + output_cost
    high_cost = low_cost if written is not None else ((total_in - cached) * write_rate + cached * cached_rate) / 1e6 + output_cost
    return {
        "input_tokens": total_in, "cached_input_tokens": cached,
        "cache_write_tokens": written, "output_tokens": total_out,
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        "usage_based_usd": low_cost if written is not None else None,
        "usage_based_usd_range": [low_cost, high_cost],
        "output_usd": output_cost,
        "normalized_no_cache_usd": (total_in * input_rate + total_out * output_rate) / 1e6,
        "long_context": long,
    }


def prepare(source: Path, workspace: Path):
    document = read_json(source / "document_ir.json")
    document_id = document["document_id"]
    for material in read_json(source / "material_manifest.json")["materials"]:
        if sha256_file(Path(material["path"])) != material["sha256"]:
            raise RuntimeError(f"Source material changed: {material['path']}")
    paths = ArticlePaths(workspace / "inputs", document_id)
    paths.ensure()
    for name in ("document_ir.json", "material_manifest.json", "manifest.json"):
        if not (paths.root / name).exists():
            shutil.copy2(source / name, paths.root / name)
    low = freeze_single_packet_request(paths, model="gpt-6.1-sol", reasoning_effort="low")
    low.body["service_tier"] = "default"
    low_digest = sha256_json(low.body)
    low = replace(low, request_sha256=low_digest, custom_id=f"effort-{document_id}-low-{low_digest[:16]}")
    medium_body = copy.deepcopy(low.body)
    medium_body["reasoning"]["effort"] = "medium"
    medium_digest = sha256_json(medium_body)
    medium = replace(low, body=medium_body, request_sha256=medium_digest,
                     custom_id=f"effort-{document_id}-medium-{medium_digest[:16]}")
    shared = lambda body: {key: value for key, value in body.items() if key != "reasoning"}
    assert shared(low.body) == shared(medium.body)
    plan = {
        "document_id": document_id, "title": document.get("title"), "model": low.body["model"],
        "source_article_root": str(source.resolve()), "shared_request_sha256": sha256_json(shared(low.body)),
        "source_document_sha256": low.source_document_sha256,
        "packet_count": 1, "page_count": len(document["pages"]), "table_count": len(document["tables"]),
        "image_count": sum(item["type"] == "input_image" for item in low.body["input"][0]["content"]),
        "max_output_tokens": low.body["max_output_tokens"], "pricing": PRICING,
        "efforts": ["low", "medium"], "sample_count_per_effort": 1,
        "comparison_note": "Actual usage cost and equal-input-rate normalization are both reported; one sample is not a quality benchmark.",
    }
    old_plan_path = workspace / "experiment_plan.json"
    if old_plan_path.exists() and read_json(old_plan_path) != plan:
        raise RuntimeError("Experiment inputs changed; choose a fresh workspace")
    write_json(old_plan_path, plan)
    for effort, request in (("low", low), ("medium", medium)):
        write_json(workspace / effort / document_id / "raw_requests" / f"{request.packet_id}.json", {
            "request_sha256": request.request_sha256, "body": request.body,
        })
    return plan, [low, medium]


def execute(request, workspace: Path, client: OpenAI, guard: BudgetGuard):
    effort = request.body["reasoning"]["effort"]
    paths = ArticlePaths(workspace / effort, request.document_id)
    attempt_path = paths.root / "attempt.json"
    raw_path = paths.raw_responses / f"{request.packet_id}.json"
    result_path = paths.root / "cost_result.json"
    raw = load_cached_response(raw_path, request.request_sha256)
    previous = read_json(attempt_path) if attempt_path.exists() else {}
    if raw is None and previous:
        raise RuntimeError(f"{effort}: previous API attempt has no reusable response; inspect before retrying")
    elapsed = previous.get("elapsed_seconds")
    if raw is None:
        guard.admit_request(request.body)
        started = time.monotonic()
        write_json(attempt_path, {"status": "requesting", "request_sha256": request.request_sha256,
                                  "started_at_utc": utc_now()})
        print(json.dumps({"event": "request_started", "effort": effort}), flush=True)
        try:
            response = client.responses.create(**request.body)
        except Exception as error:
            write_json(attempt_path, {"status": "outcome_unknown", "error_type": type(error).__name__,
                                      "message": str(error), "request_sha256": request.request_sha256})
            raise
        elapsed = time.monotonic() - started
        raw = response_payload(response)
        cache_response(raw_path, request.request_sha256, raw)
        guard.record(raw.get("usage"))
        write_json(attempt_path, {"status": "response_saved", "response_id": raw.get("id"),
                                  "elapsed_seconds": elapsed, "request_sha256": request.request_sha256})
    result = {"effort": effort, "model_returned": raw.get("model"), "service_tier": raw.get("service_tier"),
              "response_id": raw.get("id"), "status": raw.get("status"), "elapsed_seconds": elapsed,
              "usage": raw.get("usage"), "cost": usage_cost(raw["usage"]), "postprocessing": "pending"}
    write_json(result_path, result)
    try:
        parsed = parse_model_response(raw, document_id=request.document_id)
        persist_transport_result(output_root=workspace / effort, request=request, parsed=parsed,
                                 raw_response=raw, transport="synchronous")
        result["postprocessing"] = "completed"
        result["schema_version"] = parsed.schema_version
        result["collection_counts"] = {key: len(value) for key, value in parsed.model_dump().items() if isinstance(value, list)}
    except Exception as error:
        result["postprocessing"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(result_path, result)
    print(json.dumps({"event": "request_finished", "effort": effort, "cost": result["cost"],
                      "elapsed_seconds": elapsed, "postprocessing": result["postprocessing"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_article_root", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-usd", type=float, default=2.0)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    with artifact_lock(workspace / ".experiment.lock"):
        plan, requests = prepare(args.input_article_root.resolve(), workspace)
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        guard = BudgetGuard(max_usd=args.max_usd, input_usd_per_million=2, output_usd_per_million=10,
                            cached_input_multiplier=0.05)
        results = []
        with OpenAI(max_retries=0, timeout=1200.0) as client:
            for request in requests:
                result = execute(request, workspace, client, guard)
                results.append(result)
                write_json(workspace / "cost_comparison.json", {**plan, "results": results})
        low, medium = [item["cost"] for item in results]
        comparison = {**plan, "results": results,
            "normalized_medium_minus_low_usd": medium["normalized_no_cache_usd"] - low["normalized_no_cache_usd"],
            "normalized_medium_increase_percent": 100 * (medium["normalized_no_cache_usd"] / low["normalized_no_cache_usd"] - 1),
            "actual_medium_minus_low_usd": None if any(item["usage_based_usd"] is None for item in (low, medium)) else medium["usage_based_usd"] - low["usage_based_usd"],
            "actual_medium_increase_percent": None if any(item["usage_based_usd"] is None for item in (low, medium)) or not low["usage_based_usd"] else 100 * (medium["usage_based_usd"] / low["usage_based_usd"] - 1),
            "total_usage_based_usd": None if any(item["usage_based_usd"] is None for item in (low, medium)) else low["usage_based_usd"] + medium["usage_based_usd"],
            "input_and_cache_usage_identical": all(low[field] == medium[field] for field in ("input_tokens", "cached_input_tokens", "cache_write_tokens")),
        }
        write_json(workspace / "cost_comparison.json", comparison)
        print(json.dumps({"comparison": str(workspace / "cost_comparison.json"),
                          "normalized_medium_increase_percent": comparison["normalized_medium_increase_percent"]}), flush=True)


if __name__ == "__main__":
    main()
