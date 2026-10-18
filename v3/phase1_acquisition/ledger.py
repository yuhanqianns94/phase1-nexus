"""Auditable screening history and cross-workspace protection against duplicate charges.

Only validated scientific decisions are reusable. A submitted request that failed or
whose outcome is unknown remains a payment claim, never an exclusion decision.
"""
from __future__ import annotations

import re
from pathlib import Path

from phase1_extraction.artifacts import read_json, sha256_file, sha256_json, utc_now, write_json
from phase1_extraction.request_cache import artifact_lock
from .screening import make_request, validate_decision, verify_materials

FORMAT = "phase1_screening_ledger_v1"


def document_identifiers(document: dict) -> set[str]:
    metadata = document.get("article_metadata", document)
    result = set()
    for key in ("pmid", "pmcid", "doi"):
        value = str(metadata.get(key) or "").strip()
        if not value:
            continue
        if key == "doi":
            value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I).casefold()
        elif key == "pmcid":
            value = value.upper()
            if value.isdigit():
                value = "PMC" + value
        else:
            value = re.sub(r"^PMID\s*:?\s*", "", value, flags=re.I)
        if value:
            result.add(key + ":" + value)
    return result


def evidence_sha256(document: dict) -> str:
    """Ignore storage/source labels, but keep the actual ordered text and metadata.

    Bibliographic identifiers are matched separately, so adding a PMCID to the
    same DOI does not turn unchanged evidence into a new paid screening request.
    Text is deliberately not fuzzy-matched or normalized.
    """
    metadata = {key: value for key, value in document.get("article_metadata", {}).items()
                if key not in {"pmid", "pmcid", "doi"}}
    return sha256_json({"article_metadata": metadata,
                        "fulltext_available": document.get("fulltext_available", False),
                        "blocks": [block["text"] for block in document["blocks"]]})


def _load(path: Path) -> dict:
    if not path.exists():
        return {"format": FORMAT, "entries": [], "created_at": utc_now()}
    ledger = read_json(path)
    if ledger.get("format") != FORMAT or not isinstance(ledger.get("entries"), list):
        raise ValueError("Unsupported screening ledger format")
    return ledger


def _lock(path: Path):
    return artifact_lock(path.with_name(path.name + ".lock"))


def _save(path: Path, ledger: dict) -> None:
    ledger["updated_at"] = utc_now()
    write_json(path, ledger)


def _index(entries: list[dict]) -> dict[str, list[dict]]:
    result = {}
    for entry in entries:
        for identity in entry["identifiers"]:
            result.setdefault(identity, []).append(entry)
    return result


def _candidates(document: dict, index: dict[str, list[dict]]) -> list[dict]:
    result, seen = [], set()
    for identity in sorted(document_identifiers(document)):
        for entry in index.get(identity, []):
            if id(entry) not in seen:
                seen.add(id(entry))
                result.append(entry)
    return result


def _matches(document: dict, entry: dict) -> bool:
    incoming, saved = document_identifiers(document), set(entry["identifiers"])
    if not incoming.intersection(saved):
        return False
    # A shared DOI must not conceal conflicting PMID/PMCID metadata.
    for kind in ("pmid:", "pmcid:", "doi:"):
        left = {value for value in incoming if value.startswith(kind)}
        right = {value for value in saved if value.startswith(kind)}
        if left and right and left != right:
            raise ValueError("Conflicting article identifiers in screening ledger")
    return (entry["policy_version"] == document["policy_version"]
            and entry["evidence_sha256"] == evidence_sha256(document))


def verify_record(record_path: Path, *, expected_item: dict | None = None) -> dict:
    """Verify raw response, policy/request, evidence files, and parsed decision."""
    record = read_json(record_path)
    document = record["document"]
    if (record.get("policy_version") != document["policy_version"]
            or record.get("request_sha256") != sha256_json(make_request(document))):
        raise ValueError("Screening history policy/request integrity failure")
    if expected_item is not None and (document != expected_item["document"]
                                     or record["request_sha256"] != expected_item["request_sha256"]):
        raise ValueError("Screening history does not match frozen plan")
    verify_materials(document)
    raw = read_json(Path(record["raw_response_path"]))
    if sha256_json(raw) != record["raw_response_sha256"]:
        raise ValueError("Screening history raw response changed")
    if validate_decision(raw, document).model_dump() != record["result"]:
        raise ValueError("Screening history decision changed")
    return record


def _verified_entry(entry: dict) -> dict:
    path = Path(entry["record_path"])
    # Collection may legitimately add extraction materials later. The immutable
    # screening components below are checked instead of hashing the whole file.
    record = verify_record(path)
    if (record["request_sha256"] != entry["request_sha256"]
            or record["raw_response_sha256"] != entry["raw_response_sha256"]
            or record["result"]["decision"] != entry["decision"]
            or not _matches(record["document"], entry)):
        raise ValueError("Screening ledger entry no longer matches its evidence")
    return record


def find_completed(document: dict, ledger_path: Path) -> dict | None:
    path = Path(ledger_path)
    with _lock(path):
        for entry in _candidates(document, _index(_load(path)["entries"])):
            if _matches(document, entry) and entry["status"] == "completed":
                _verified_entry(entry)
                return dict(entry)
    return None


def filter_completed(documents: list[dict], ledger_path: Path) -> tuple[list[dict], list[dict]]:
    """Return unpaid documents and explicit pointers to reusable old decisions."""
    path = Path(ledger_path)
    pending, skipped = [], []
    with _lock(path):
        index = _index(_load(path)["entries"])
        for document in documents:
            matched = next((e for e in _candidates(document, index) if _matches(document, e) and e["status"] == "completed"), None)
            if matched is None:
                pending.append(document)
            else:
                _verified_entry(matched)
                skipped.append({"document_id": document["document_id"], "decision": matched["decision"],
                                "policy_version": matched["policy_version"], "evidence_sha256": matched["evidence_sha256"],
                                "record_path": matched["record_path"], "reason": "identical_policy_and_evidence"})
    return pending, skipped


def seen_identifiers(ledger_path: Path) -> set[str]:
    """Discovery excludes previously completed or already claimed articles."""
    path = Path(ledger_path)
    with _lock(path):
        return {identity for entry in _load(path)["entries"] for identity in entry["identifiers"]}


def _entry(document: dict, workspace: Path, request_sha256: str) -> dict:
    return {"identifiers": sorted(document_identifiers(document)), "policy_version": document["policy_version"],
            "evidence_sha256": evidence_sha256(document), "document_id": document["document_id"],
            "workspace": str(Path(workspace).resolve()), "request_sha256": request_sha256,
            "created_at": utc_now()}


def claim_submission(plan: dict, workspace: Path, ledger_path: Path) -> None:
    """Atomically claim the entire batch before any upload/create operation.

    Claims survive errors and crashes. Only the same workspace/request can resume;
    even technical failures do not silently authorize another paid attempt.
    """
    path, workspace = Path(ledger_path), Path(workspace).resolve()
    with _lock(path):
        ledger = _load(path)
        index = _index(ledger["entries"])
        additions = []
        for item in plan["requests"]:
            document = item["document"]
            if not document_identifiers(document):
                continue  # No bibliographic identity: never pretend this is reusable.
            matches = [e for e in _candidates(document, index) if _matches(document, e)]
            for entry in matches:
                same_owner = (entry["workspace"] == str(workspace)
                              and entry["request_sha256"] == item["request_sha256"])
                if entry["status"] == "completed":
                    _verified_entry(entry)
                    raise RuntimeError(f"Article {document['document_id']} already screened; prepare a new filtered workspace")
                if not same_owner:
                    raise RuntimeError(f"Article {document['document_id']} already has a paid-submission claim in {entry['workspace']}; do not resubmit")
            if not matches:
                entry = {**_entry(document, workspace, item["request_sha256"]), "status": "claimed"}
                additions.append(entry)
                for identity in entry["identifiers"]:
                    index.setdefault(identity, []).append(entry)
        ledger["entries"].extend(additions)
        _save(path, ledger)


def record_collection(plan: dict, workspace: Path, summary: dict, ledger_path: Path) -> None:
    path, workspace = Path(ledger_path), Path(workspace).resolve()
    indexed = {o["document_id"]: o for o in summary["outcomes"]}
    verified = {}
    for item in plan["requests"]:
        outcome = indexed[item["document"]["document_id"]]
        if outcome["status"] == "completed":
            verified[item["custom_id"]] = verify_record(Path(outcome["record_path"]), expected_item=item)
    with _lock(path):
        ledger = _load(path)
        identity_index = _index(ledger["entries"])
        for item in plan["requests"]:
            document = item["document"]
            if not document_identifiers(document):
                continue
            outcome = indexed[document["document_id"]]
            entry = next((e for e in _candidates(document, identity_index) if _matches(document, e)
                          and e["workspace"] == str(workspace)
                          and e["request_sha256"] == item["request_sha256"]), None)
            if entry is None:
                entry = _entry(document, workspace, item["request_sha256"])
                ledger["entries"].append(entry)
                for identity in entry["identifiers"]:
                    identity_index.setdefault(identity, []).append(entry)
            entry.update(batch_id=summary["batch_id"], updated_at=utc_now())
            if item["custom_id"] in verified:
                record = verified[item["custom_id"]]
                entry.update(status="completed", decision=record["result"]["decision"],
                             record_path=str(Path(outcome["record_path"]).resolve()),
                             raw_response_sha256=record["raw_response_sha256"])
                entry.pop("error", None)
            else:
                entry.update(status="technical_failure", error=outcome.get("error", "Missing response"))
                for key in ("decision", "record_path", "raw_response_sha256"):
                    entry.pop(key, None)
        _save(path, ledger)


def import_workspace(workspace: Path, ledger_path: Path) -> dict:
    """Import completed historical records without accepting technical failures.

    Invalid records are reported explicitly and do not become reusable. Technical
    failures are imported only as payment claims, never as scientific exclusions.
    Existing paid plans/state remain untouched; only a report and ledger are written.
    """
    from .screening_batch import validate_plan
    workspace, path = Path(workspace).resolve(), Path(ledger_path)
    try:
        plan = validate_plan(workspace)
        summary = read_json(workspace / "screening_summary.json")
        state = read_json(workspace / "screening_state.json")
        terminal = {"completed", "failed", "expired", "cancelled"}
        if (summary.get("complete") is not True or summary.get("batch_id") != state.get("batch_id")
                or summary.get("status") not in terminal or state.get("status") not in terminal
                or state.get("plan_sha256") != sha256_file(workspace / "screening_plan.json")):
            raise ValueError("Historical screening campaign is incomplete or changed")
        indexed = {o["document_id"]: o for o in summary["outcomes"]}
        if len(indexed) != len(summary["outcomes"]) or set(indexed) != {i["document"]["document_id"] for i in plan["requests"]}:
            raise ValueError("Historical summary does not match frozen plan")
    except Exception as error:
        write_json(workspace / "screening_ledger_import_report.json", {
            "workspace": str(workspace), "ledger_path": str(path.resolve()), "imported": 0,
            "verified": 0, "fatal_error": str(error), "created_at": utc_now()})
        raise
    accepted, rejected, technical, failed_claims = [], [], [], []
    for item in plan["requests"]:
        document = item["document"]
        outcome = indexed[document["document_id"]]
        if outcome.get("status") != "completed":
            technical.append(document["document_id"])
            if document_identifiers(document):
                failed_claims.append({**_entry(document, workspace, item["request_sha256"]),
                                      "status": "technical_failure", "batch_id": summary["batch_id"],
                                      "error": outcome.get("error", "Missing response")})
            continue
        try:
            record_path = Path(outcome["record_path"])
            record = verify_record(record_path, expected_item=item)
            if outcome.get("decision") != record["result"]["decision"]:
                raise ValueError("Summary decision differs from verified response")
            if not document_identifiers(document):
                raise ValueError("No bibliographic identifiers for reliable reuse")
            accepted.append({**_entry(document, workspace, item["request_sha256"]), "status": "completed",
                             "decision": record["result"]["decision"], "record_path": str(record_path.resolve()),
                             "raw_response_sha256": record["raw_response_sha256"], "batch_id": summary["batch_id"]})
        except Exception as error:
            rejected.append({"document_id": document["document_id"], "error": str(error)})
    imported = 0
    with _lock(path):
        ledger = _load(path)
        owner_index = {(e["workspace"], e["request_sha256"]): e for e in ledger["entries"]}
        for entry in accepted + failed_claims:
            owner_key = (entry["workspace"], entry["request_sha256"])
            existing = owner_index.get(owner_key)
            if existing is None:
                ledger["entries"].append(entry)
                owner_index[owner_key] = entry
                imported += entry["status"] == "completed"
            else:
                existing.update(entry)
                if entry["status"] == "completed":
                    existing.pop("error", None)
                else:
                    for key in ("decision", "record_path", "raw_response_sha256"):
                        existing.pop(key, None)
        _save(path, ledger)
    report = {"workspace": str(workspace), "ledger_path": str(path.resolve()), "imported": imported,
              "verified": len(accepted), "technical_failures_not_imported": technical,
              "technical_failure_claims_recorded": len(failed_claims),
              "rejected": rejected, "created_at": utc_now()}
    write_json(workspace / "screening_ledger_import_report.json", report)
    return report
