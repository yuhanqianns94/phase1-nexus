"""Frozen, restart-safe OpenAI Batch transport for binary screening only."""
from __future__ import annotations

import json
import math
from pathlib import Path

from openai import OpenAI

from phase1_extraction.artifacts import read_json, sha256_file, sha256_json, utc_now, write_json
from phase1_extraction.llm import BudgetGuard
from phase1_extraction.request_cache import artifact_lock
from .screening import POLICY_VERSION, SUPPORTED_POLICIES, MODEL, EFFORT, make_request, validate_decision, verify_materials, validate_screening_gate, extraction_fulltext_available, reference_binding_aliases, is_retained, is_excluded

PRICING = {"input_usd_per_million": .05, "output_usd_per_million": .25,
           "cache_write_multiplier": 1.25, "cached_input_multiplier": .1}
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def request_reserve(body: dict) -> float:
    upper_tokens = len(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    long = upper_tokens > 272_000
    return (upper_tokens * PRICING["input_usd_per_million"] * 1.25 * (2 if long else 1)
            + body["max_output_tokens"] * PRICING["output_usd_per_million"] * (1.5 if long else 1)) / 1_000_000


def freeze_documents(documents: list[dict], workspace: Path, *, ledger_path: Path | None = None) -> dict:
    from .acquire import _atomic_write_text
    workspace = Path(workspace)
    with artifact_lock(workspace / ".screening.lock"):
        if (workspace / "screening_plan.json").exists():
            raise ValueError("Screening workspace already frozen; use submit/collect to resume, or a new directory")
        if ledger_path is not None:
            from .ledger import filter_completed
            original_count = len(documents)
            # Check current files even when the scientific decision can be reused.
            for document in documents:
                verify_materials(document)
            documents, reused = filter_completed(documents, Path(ledger_path))
            report = {"requested_count": original_count, "request_count": len(documents),
                      "reused_count": len(reused), "reused": reused,
                      "ledger_path": str(Path(ledger_path).resolve())}
            write_json(workspace / "screening_reuse_report.json", report)
            if reused and not documents:
                return {"status": "all_reused", "request_count": 0, "reserved_usd": 0,
                        "reused_count": len(reused), "reuse_report": str((workspace / "screening_reuse_report.json").resolve())}
        requests, index, reserve = [], [], 0.0
        versions = {d["policy_version"] for d in documents}
        if len(versions) != 1 or not versions.issubset(SUPPORTED_POLICIES):
            raise ValueError("Batch requires one supported screening policy")
        seen = set()
        for document in documents:
            if document["document_id"] in seen:
                raise ValueError("Duplicate screening article")
            seen.add(document["document_id"])
            verify_materials(document)
            body = make_request(document)
            request_hash = sha256_json(body)
            custom_id = "screen-" + request_hash[:40]
            requests.append({"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body})
            index.append({"custom_id": custom_id, "document": document, "request_sha256": request_hash})
            # UTF-8 bytes are a deliberately high text token bound, including schema/prompt.
            # Reserve long-context rates whenever that bound crosses the price threshold.
            reserve += request_reserve(body)
        if not index:
            raise ValueError("No screening documents")
        path = workspace / "screening_requests.jsonl"
        encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests)
        if len(index) > 50_000 or len(encoded.encode()) > 190_000_000:
            raise ValueError("Screening Batch exceeds file limit; prepare smaller independent batches")
        _atomic_write_text(path, encoded)
        plan = {"format": "phase1_screening_batch_v1", "policy_version": next(iter(versions)),
                "model": MODEL, "effort": EFFORT, "created_at": utc_now(), "input_path": str(path.resolve()),
                "input_sha256": sha256_file(path), "request_count": len(index), "requests": index,
                "pricing": PRICING, "reserved_usd": reserve}
        if ledger_path is not None:
            plan["ledger_path"] = str(Path(ledger_path).resolve())
        write_json(workspace / "screening_plan.json", plan)
        return plan


def validate_plan(workspace: Path) -> dict:
    plan = read_json(workspace / "screening_plan.json")
    if plan["policy_version"] not in SUPPORTED_POLICIES:
        raise ValueError("Screening policy changed")
    path = Path(plan["input_path"])
    if sha256_file(path) != plan["input_sha256"]:
        raise ValueError("Frozen screening input changed")
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {r["custom_id"]: r for r in plan["requests"]}
    if len(expected) != len(lines) or len({r["custom_id"] for r in lines}) != len(lines):
        raise ValueError("Duplicate or missing Batch requests")
    for line in lines:
        item = expected[line["custom_id"]]
        if item["document"]["policy_version"] != plan["policy_version"]:
            raise ValueError("Mixed screening policies")
        verify_materials(item["document"])
        if (line["method"] != "POST" or line["url"] != "/v1/responses"
                or sha256_json(line["body"]) != item["request_sha256"]
                or sha256_json(make_request(item["document"])) != item["request_sha256"]):
            raise ValueError("Frozen body no longer matches screened evidence and policy")
    if (plan["pricing"] != PRICING or plan["request_count"] != len(lines)
            or not math.isclose(plan["reserved_usd"], sum(request_reserve(line["body"]) for line in lines))):
        raise ValueError("Frozen screening budget/count changed")
    return plan


def _ledger_path(plan: dict, supplied: Path | None) -> Path | None:
    saved = plan.get("ledger_path")
    if saved and supplied is not None and Path(saved).resolve() != Path(supplied).resolve():
        raise ValueError("Screening ledger differs from the frozen plan")
    return Path(saved or supplied).resolve() if saved or supplied is not None else None


def submit(workspace: Path, *, max_usd: float = 5, client=None, ledger_path: Path | None = None) -> dict:
    workspace = Path(workspace)
    with artifact_lock(workspace / ".screening.lock"):
        plan = validate_plan(workspace)
        active_ledger = _ledger_path(plan, ledger_path)
        gate_path = workspace / "submission_gate.json"
        if gate_path.exists() and read_json(gate_path).get("allow_submission") is not True:
            raise ValueError("Screening pilot gate blocks this submission")
        if not math.isfinite(max_usd) or max_usd <= 0 or plan["reserved_usd"] > max_usd:
            raise ValueError(f"Screening reserve ${plan['reserved_usd']:.4f} exceeds valid budget")
        state_path = workspace / "screening_state.json"
        state = read_json(state_path) if state_path.exists() else {"status": "prepared", "plan_sha256": sha256_file(workspace / "screening_plan.json")}
        if state["plan_sha256"] != sha256_file(workspace / "screening_plan.json"):
            raise ValueError("Screening plan changed after submission")
        if state.get("batch_id"):
            return state
        if state["status"] in {"creating", "submission_unknown"}:
            raise RuntimeError("Previous paid submission outcome unknown; inspect remote account, do not resubmit")
        if active_ledger is not None:
            from .ledger import claim_submission
            claim_submission(plan, workspace, active_ledger)
        client = client or OpenAI(max_retries=0)
        creation_client = client.with_options(max_retries=0)
        if not state.get("input_file_id"):
            with Path(plan["input_path"]).open("rb") as stream:
                uploaded = client.files.create(file=stream, purpose="batch")
            state.update(input_file_id=uploaded.id, status="uploaded", max_usd=max_usd)
            write_json(state_path, state)
        state.update(status="creating", submitted_at=utc_now())
        write_json(state_path, state)
        try:
            batch = creation_client.batches.create(input_file_id=state["input_file_id"], endpoint="/v1/responses",
                                                   completion_window="24h", metadata={"pipeline": plan["policy_version"]})
        except Exception as error:
            state.update(status="submission_unknown", error=type(error).__name__)
            write_json(state_path, state)
            raise
        state.update(batch_id=batch.id, status=batch.status)
        write_json(state_path, state)
        return state


def collect(workspace: Path, *, client=None, ledger_path: Path | None = None) -> dict:
    workspace = Path(workspace)
    with artifact_lock(workspace / ".screening.lock"):
        plan = validate_plan(workspace)
        active_ledger = _ledger_path(plan, ledger_path)
        state = read_json(workspace / "screening_state.json")
        if state["plan_sha256"] != sha256_file(workspace / "screening_plan.json"):
            raise ValueError("Screening plan changed after submission")
        if not state.get("batch_id"):
            raise ValueError("No submitted screening Batch")
        client = client or OpenAI()
        batch = client.batches.retrieve(state["batch_id"])
        state.update(status=batch.status, last_checked_at=utc_now(),
                     request_counts=batch.request_counts.model_dump() if batch.request_counts else None)
        write_json(workspace / "screening_state.json", state)
        if batch.status not in TERMINAL:
            return {"status": batch.status, "batch_id": batch.id, "complete": False}
        outputs = {}
        for attr in ("output_file_id", "error_file_id"):
            file_id = getattr(batch, attr, None)
            if not file_id:
                continue
            path = workspace / (attr + ".jsonl")
            if not path.exists() or state.get(attr) != file_id:
                payload = client.files.content(file_id).content
                from .acquire import _atomic_write_text
                _atomic_write_text(path, payload.decode("utf-8"))
            state[attr] = file_id
            for line in path.read_text().splitlines():
                item = json.loads(line)
                cid = item.get("custom_id")
                if cid in outputs:
                    raise ValueError("Duplicate custom_id in screening Batch results")
                outputs[cid] = item
        write_json(workspace / "screening_state.json", state)
        if set(outputs) - {r["custom_id"] for r in plan["requests"]}:
            raise ValueError("Unknown article in screening Batch output")
        outcomes, total_input, total_output, total_cost = [], 0, 0, 0.0
        usage_missing = 0
        for item in plan["requests"]:
            doc = item["document"]
            outcome = {"document_id": doc["document_id"]}
            response = outputs.get(item["custom_id"], {})
            body = response.get("response", {}).get("body")
            raw_path = workspace / "raw_responses" / (item["custom_id"] + ".json")
            # Persist before local validation: collection can be replayed without another API call.
            if body:
                write_json(raw_path, body)
                usage = body.get("usage")
                if usage and "input_tokens" in usage and "output_tokens" in usage:
                    inp, out = usage["input_tokens"], usage["output_tokens"]
                    total_input += inp
                    total_output += out
                    long = inp > 272_000
                    guard = BudgetGuard(max_usd=1e9, input_usd_per_million=.05 * (2 if long else 1),
                                        output_usd_per_million=.25 * (1.5 if long else 1), cached_input_multiplier=.1)
                    guard.record(usage)
                    total_cost += guard.spent_usd
                else:
                    usage_missing += 1
            try:
                if response.get("error") or response.get("response", {}).get("status_code") != 200 or not body:
                    raise ValueError("API failure or missing response; no screening decision was made")
                result = validate_decision(body, doc)
                record_path = workspace / "decisions" / (doc["document_id"] + ".json")
                record = {"policy_version": plan["policy_version"], "request_sha256": item["request_sha256"],
                          "raw_response_path": str(raw_path.resolve()), "raw_response_sha256": sha256_json(body),
                          "document": doc, "result": result.model_dump(), "usage": body.get("usage")}
                aliases = reference_binding_aliases(body, doc)
                if aliases:
                    record["reference_binding_aliases"] = aliases
                if record_path.exists():
                    prior = read_json(record_path)
                    if (prior.get("request_sha256") == record["request_sha256"]
                            and prior.get("raw_response_sha256") == record["raw_response_sha256"]):
                        for key in ("extraction_materials", "pdf_provenance"):
                            if key in prior:
                                record[key] = prior[key]
                write_json(record_path, record)
                outcome.update(status="completed", decision=result.decision, reason=result.reason,
                               record_path=str(record_path.resolve()), fulltext_available=doc["fulltext_available"])
            except Exception as error:
                # Technical failures are not falsely labelled as scientific exclusions.
                outcome.update(status="failed", error=str(error))
            outcomes.append(outcome)
        summary = {"status": batch.status, "complete": True, "batch_id": batch.id,
                   "article_count": len(outcomes), "retained": sum(is_retained(o.get("decision")) for o in outcomes),
                   "excluded": sum(is_excluded(o.get("decision")) for o in outcomes),
                   "technical_failures": sum(o["status"] == "failed" for o in outcomes),
                   "input_tokens": total_input, "output_tokens": total_output, "estimated_usd": total_cost,
                   "usage_missing_responses": usage_missing, "outcomes": outcomes}
        write_json(workspace / "screening_summary.json", summary)
        if active_ledger is not None:
            from .ledger import record_collection
            record_collection(plan, workspace, summary, active_ledger)
        return summary


def material_preflight(document_id: str, sources: list[Path], output_root: Path) -> dict:
    """Run the existing extraction parser/packetizer locally, with no model calls."""
    from phase1_extraction.pipeline import ingest, parse_sources, prepare_packets
    paths = ingest(document_id=document_id, source_paths=sources, output_root=output_root)
    parse_sources(paths)
    _, packets = prepare_packets(paths)
    manifest = read_json(paths.extraction_packets)
    missing_visuals = [item for packet in packets for item in packet.get("image_inventory", [])
                       if item.get("required", True)
                       and item.get("status") not in {"included", "covered_by_context_page"}]
    return {"ready": bool(packets) and bool(manifest.get("material_coverage_complete")) and not missing_visuals,
            "packet_count": len(packets), "not_covered_source_ids": manifest.get("not_covered_source_ids", []),
            "missing_visuals": missing_visuals, "packets_path": str(paths.extraction_packets.resolve())}


def export_retained(workspace: Path, output: Path) -> dict:
    from .acquire import _atomic_write_text
    workspace = Path(workspace)
    with artifact_lock(workspace / ".screening.lock"):
        validate_plan(workspace)
        summary = read_json(workspace / "screening_summary.json")
        if not summary["complete"] or summary["technical_failures"]:
            raise ValueError("Finish and resolve technical screening failures before exporting")
        rows, missing, incomplete = [], [], []
        for outcome in summary["outcomes"]:
            if not is_retained(outcome.get("decision")):
                continue
            path = Path(outcome["record_path"])
            record = read_json(path)
            if not extraction_fulltext_available(record):
                missing.append(outcome["document_id"])
                continue
            sources = [Path(m["path"]) for m in record.get("extraction_materials", record["document"]["materials"])]
            validate_screening_gate(path, sources)
            try:
                coverage = material_preflight(outcome["document_id"], sources, workspace / "material_preflight")
            except Exception as error:
                coverage = {"ready": False, "error": str(error)}
            if not coverage["ready"]:
                incomplete.append({"document_id": outcome["document_id"], **coverage})
                continue
            rows.append({"document_id": outcome["document_id"], "sources": [str(p) for p in sources],
                         "screening_record": str(path.resolve())})
        _atomic_write_text(Path(output), "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        report = {"exported": len(rows), "retained_without_fulltext": missing, "manifest": str(Path(output).resolve()),
                  "retained_with_incomplete_materials": incomplete,
                  "note": "Local V3 material/visual coverage preflight passed for exported articles; this is not a semantic extraction audit."}
        write_json(Path(output).with_suffix(".summary.json"), report)
        return report


def hydrate_pdfs(workspace: Path, output_root: Path) -> dict:
    """Add identity-verified official PDFs after retention, without another model call."""
    from .pdf_assets import download_pmc_pdf_for_article
    workspace = Path(workspace)
    with artifact_lock(workspace / ".screening.lock"):
        validate_plan(workspace)
        summary = read_json(workspace / "screening_summary.json")
        results = []
        for outcome in summary["outcomes"]:
            if not is_retained(outcome.get("decision")):
                continue
            path = Path(outcome["record_path"])
            record = read_json(path)
            document = record["document"]
            verify_materials(document)
            if not is_retained(validate_decision(read_json(Path(record["raw_response_path"])), document).decision):
                raise ValueError("Cannot hydrate excluded article")
            xml = next((Path(m["path"]) for m in document["materials"] if Path(m["path"]).suffix == ".xml"), None)
            if xml is None:
                results.append({"document_id": outcome["document_id"], "status": "no_source_xml"})
                continue
            manifest = download_pmc_pdf_for_article(xml, output_root)
            pmcid = manifest["pmcid"]
            if manifest["status"] == "ready":
                pdf = (Path(output_root) / pmcid / manifest["pdf_path"]).resolve()
                if sha256_file(pdf) != manifest["pdf_sha256"]:
                    raise ValueError("Hydrated PDF hash mismatch")
                record["extraction_materials"] = list(document["materials"])
                if str(pdf) not in {m["path"] for m in document["materials"]}:
                    record["extraction_materials"].append({"path": str(pdf), "sha256": sha256_file(pdf)})
                record["pdf_provenance"] = manifest
                write_json(path, record)
            results.append({"document_id": outcome["document_id"], "status": manifest["status"]})
        report = {"pdf_ready": sum(r["status"] == "ready" for r in results), "articles": results}
        write_json(workspace / "pdf_hydration_report.json", report)
        return report
