"""Fixed PubMed cohort orchestration, independent of old discovery campaigns."""
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from phase1_extraction.artifacts import read_json, write_json, sha256_file, utc_now
from phase1_extraction.request_cache import artifact_lock
from .discovery import ARTICLE_COUNT, PUBMED_QUERY, PUBMED_SORT, discover
from .preparation import select_fulltext, validate_selection, prepare_rescreen
from .screening import MODEL, EFFORT, POLICY_VERSION, make_request, is_retained, is_excluded
from .screening_batch import freeze_documents, validate_plan, submit, collect

MAX_BATCH_REQUESTS = 1000
MAX_BATCH_BYTES = 100_000_000


def _configuration(workspace: Path, through: str | None, source_workspace: Path | None = None) -> dict:
    path = workspace / "configuration.json"
    previous = read_json(path) if path.exists() else None
    cutoff = through or (previous["through"] if previous else
                         datetime.now(ZoneInfo("America/New_York")).strftime("%Y/%m/%d"))
    cutoff = datetime.strptime(cutoff.replace("-", "/"), "%Y/%m/%d").strftime("%Y/%m/%d")
    config = {"format": "pubmed_fulltext_campaign_v2", "source": "pubmed", "count": ARTICLE_COUNT,
              "query": PUBMED_QUERY, "sort": PUBMED_SORT, "through": cutoff,
              "model": MODEL, "effort": EFFORT, "policy_version": POLICY_VERSION,
              "require_fulltext": True, "selection_rule": "first_verified_fulltext"}
    if source_workspace is not None:
        config.update(selection_rule="reuse_verified_fulltext_cohort",
                      source_workspace=str(source_workspace.resolve()))
    if previous and previous != config:
        raise ValueError("Campaign configuration changed; existing paid inputs cannot be replaced")
    if not previous:
        write_json(path, config)
    return config


def _document_groups(documents):
    group, size = [], 0
    for doc in documents:
        # Includes space for the Batch envelope, custom ID, and JSONL newline.
        item_size = len(json.dumps(make_request(doc), ensure_ascii=False).encode()) + 512
        if item_size > MAX_BATCH_BYTES:
            raise ValueError("One screening request exceeds Batch file size")
        if group and (len(group) == MAX_BATCH_REQUESTS or size + item_size > MAX_BATCH_BYTES):
            yield group
            group, size = [], 0
        group.append(doc)
        size += item_size
    if group:
        yield group


def _freeze(workspace: Path, documents: list[dict]) -> dict:
    config = read_json(workspace / "configuration.json")
    if (len(documents) != config["count"]
            or any(not d["fulltext_available"] or not d["materials"] for d in documents)):
        raise ValueError("The full-text target must be complete before freezing any paid requests")
    batches = []
    for number, group in enumerate(_document_groups(documents), 1):
        folder = workspace / "batches" / f"{number:03d}"
        plan = validate_plan(folder) if (folder / "screening_plan.json").exists() else freeze_documents(group, folder)
        if [r["document"] for r in plan["requests"]] != group:
            raise ValueError("Existing Batch does not match the selected cohort")
        batches.append({"workspace": str(folder.resolve()), "count": len(group),
                        "reserved_usd": plan["reserved_usd"],
                        "plan_sha256": sha256_file(folder / "screening_plan.json")})
    plan = {"configuration_sha256": sha256_file(workspace / "configuration.json"),
            "candidates_sha256": sha256_file(workspace / "candidates.json"),
            "selection_receipt_sha256": sha256_file(workspace / "selection_receipt.json"),
            "selected_count": len(documents),
            "document_ids": [d["document_id"] for d in documents], "count": len(documents),
            "reserved_usd": sum(b["reserved_usd"] for b in batches), "batches": batches}
    path = workspace / "campaign_plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Frozen campaign plan changed")
    write_json(path, plan)
    return plan


def validate_campaign(workspace: Path) -> dict:
    plan = read_json(workspace / "campaign_plan.json")
    if (sha256_file(workspace / "configuration.json") != plan["configuration_sha256"]
            or sha256_file(workspace / "candidates.json") != plan["candidates_sha256"]):
        raise ValueError("Campaign source selection changed")
    config = read_json(workspace / "configuration.json")
    if config.get("selection_rule") in {"first_verified_fulltext", "reuse_verified_fulltext_cohort"}:
        if sha256_file(workspace / "selection_receipt.json") != plan.get("selection_receipt_sha256"):
            raise ValueError("Full-text selection receipt changed")
        receipt = validate_selection(workspace)
        if receipt["selected_count"] != config["count"] or plan["count"] != config["count"]:
            raise ValueError("Selected full-text count differs from target")
    candidates = read_json(workspace / "candidates.json")["candidates"]
    selected_ids = [c["document_id"] for c in candidates]
    exclusions = _fulltext_exclusions(workspace, plan)
    excluded_ids = [o["document_id"] for o in exclusions]
    excluded_set = set(excluded_ids)
    if (len(selected_ids) != config["count"] or plan.get("selected_count", config["count"]) != config["count"]
            or len(set(selected_ids)) != len(selected_ids)
            or len(excluded_ids) != len(set(excluded_ids)) or not set(excluded_ids) <= set(selected_ids)):
        raise ValueError("Invalid selected/excluded article set")
    ids, reserve = [], 0
    for batch in plan["batches"]:
        folder = Path(batch["workspace"])
        if folder.parent.resolve() != (workspace / "batches").resolve():
            raise ValueError("Batch workspace outside this campaign")
        frozen = validate_plan(folder)
        if sha256_file(folder / "screening_plan.json") != batch["plan_sha256"]:
            raise ValueError("Frozen Batch plan changed")
        if batch["count"] != frozen["request_count"] or batch["reserved_usd"] != frozen["reserved_usd"]:
            raise ValueError("Batch count or budget changed")
        if config.get("require_fulltext") and any(not r["document"]["fulltext_available"]
                or not r["document"]["materials"] for r in frozen["requests"]):
            raise ValueError("Only verified full-text documents may enter model screening")
        ids.extend(r["document"]["document_id"] for r in frozen["requests"])
        reserve += frozen["reserved_usd"]
    expected_ids = [identity for identity in selected_ids if identity not in excluded_set]
    if (len(set(ids)) != len(ids) or plan["count"] != len(ids)
            or ids != plan["document_ids"] or ids != expected_ids
            or not math.isclose(reserve, plan["reserved_usd"])):
        raise ValueError("Each selected article must be screened once or explicitly excluded for missing full text")
    return plan


def _fulltext_exclusions(workspace: Path, plan: dict) -> list[dict]:
    """Old campaigns remain readable under their original frozen availability rule."""
    config = read_json(workspace / "configuration.json")
    if not config.get("require_fulltext") or config.get("selection_rule") in {
            "first_verified_fulltext", "reuse_verified_fulltext_cohort"}:
        return []
    path = workspace / "fulltext_exclusions.json"
    if sha256_file(path) != plan.get("fulltext_exclusions_sha256"):
        raise ValueError("Full-text exclusion list changed")
    payload = read_json(path)
    if payload["count"] != len(payload["outcomes"]) or any(
            not is_excluded(o.get("decision")) or o.get("fulltext_available") is not False
            or o.get("stage") != "fulltext_availability" for o in payload["outcomes"]):
        raise ValueError("Invalid full-text exclusions")
    return payload["outcomes"]


def status(workspace: Path) -> dict:
    """Local state only: no polling and no network access."""
    path = Path(workspace) / "campaign_state.json"
    return read_json(path) if path.exists() else {"status": "not_started"}


def run(workspace: Path, *, through: str | None = None, max_usd: float = 25, client=None,
        source_workspace: Path | None = None) -> dict:
    workspace = Path(workspace).resolve()
    if not math.isfinite(max_usd) or max_usd <= 0:
        raise ValueError("A positive finite budget is required")
    if source_workspace is not None:
        source_workspace = Path(source_workspace).resolve()
        if source_workspace == workspace:
            raise ValueError("Rescreening requires a separate workspace")
        source_config = read_json(source_workspace / "configuration.json")
        if source_config["count"] != ARTICLE_COUNT or not source_config.get("require_fulltext"):
            raise ValueError("Rescreening source must contain the complete verified full-text target")
        if through is not None and through.replace("-", "/") != source_config["through"]:
            raise ValueError("Rescreening cannot change the original search cutoff")
        through = source_config["through"]
    with artifact_lock(workspace / ".campaign.lock"):
        config = _configuration(workspace, through, source_workspace)
        state = status(workspace)
        try:
            if not (workspace / "campaign_plan.json").exists():
                if source_workspace is not None:
                    state.update(status="preparing", target=config["count"],
                                 source_workspace=str(source_workspace), updated_at=utc_now())
                    write_json(workspace / "campaign_state.json", state)
                    source_plan = validate_campaign(source_workspace)
                    docs = prepare_rescreen(source_workspace, workspace, source_plan)
                else:
                    state.update(status="discovering", updated_at=utc_now())
                    write_json(workspace / "campaign_state.json", state)
                    found = discover(workspace, through=config["through"])
                    state.update(status="preparing", target=config["count"],
                                 ranked_candidates=found["candidate_count"], updated_at=utc_now())
                    write_json(workspace / "campaign_state.json", state)
                    docs = select_fulltext(found["candidates"], workspace, target=config["count"])
                _freeze(workspace, docs)
            plan = validate_campaign(workspace)
            if plan["reserved_usd"] > max_usd:
                raise ValueError(f"Total reserve ${plan['reserved_usd']:.4f} exceeds ${max_usd:.2f}")
            if state.get("status") in {"completed", "completed_with_errors"}:
                return state
            # Check the whole budget before submitting the first shard. Existing IDs
            # are reused; an ambiguous create blocks subsequent submissions.
            for batch in plan["batches"]:
                submit(Path(batch["workspace"]), max_usd=batch["reserved_usd"], client=client)
                receipts = [(b, read_json(Path(b["workspace"]) / "screening_state.json"))
                            for b in plan["batches"] if (Path(b["workspace"]) / "screening_state.json").exists()]
                state.update(status="submitting", selected=ARTICLE_COUNT,
                             batch_ids=[s["batch_id"] for _, s in receipts if s.get("batch_id")],
                             submitted=sum(b["count"] for b, s in receipts if s.get("batch_id")),
                             updated_at=utc_now())
                write_json(workspace / "campaign_state.json", state)
            submitted = [read_json(Path(b["workspace"]) / "screening_state.json") for b in plan["batches"]]
            state.update(status="submitted" if submitted else "completed", selected=ARTICLE_COUNT,
                         submitted=plan["count"], excluded_without_fulltext=ARTICLE_COUNT - plan["count"],
                         model=MODEL, effort=EFFORT, policy_version=POLICY_VERSION,
                         output_language="English", reserved_usd=plan["reserved_usd"],
                         batch_ids=[s["batch_id"] for s in submitted], monitoring_enabled=False,
                         extraction_submitted=0, updated_at=utc_now())
            receipt = read_json(workspace / "selection_receipt.json")
            state.update(examined_candidates=receipt["examined_count"],
                         skipped_without_fulltext=receipt["skipped_without_fulltext"])
            state.pop("error", None)
        except Exception as error:
            state.update(status="stopped", error=f"{type(error).__name__}: {error}", updated_at=utc_now())
            write_json(workspace / "campaign_state.json", state)
            raise
        write_json(workspace / "campaign_state.json", state)
        return state


def collect_results(workspace: Path, *, client=None) -> dict:
    """One read of each submitted Batch; never creates or retries inference."""
    workspace = Path(workspace).resolve()
    with artifact_lock(workspace / ".campaign.lock"):
        plan = validate_campaign(workspace)
        exclusions = _fulltext_exclusions(workspace, plan)
        outcomes, batches = list(exclusions), []
        for batch in plan["batches"]:
            folder = Path(batch["workspace"])
            result = collect(folder, client=client) if (folder / "screening_state.json").exists() else {
                "status": "not_submitted", "complete": False}
            outcomes.extend(result.get("outcomes", []))
            batches.append({k: v for k, v in result.items() if k != "outcomes"})
        complete = all(b["complete"] for b in batches)
        failed = sum(o["status"] == "failed" for o in outcomes)
        selected_count = read_json(workspace / "configuration.json")["count"]
        summary = {"complete": complete, "article_count": selected_count,
                   "status": ("completed_with_errors" if failed else "completed") if complete else "in_progress",
                   "retained": sum(is_retained(o.get("decision")) for o in outcomes),
                   "excluded": sum(is_excluded(o.get("decision")) for o in outcomes),
                   "excluded_without_fulltext": len(exclusions), "model_screening_count": plan["count"],
                   "technical_failures": failed, "pending": selected_count - len(outcomes),
                   "estimated_usd": sum(b.get("estimated_usd", 0) for b in batches),
                   "input_tokens": sum(b.get("input_tokens", 0) for b in batches),
                   "output_tokens": sum(b.get("output_tokens", 0) for b in batches),
                   "batches": batches, "updated_at": utc_now()}
        write_json(workspace / "screening_summary.json", {**summary, "outcomes": outcomes})
        write_json(workspace / "retained_candidates.json", {
            "count": summary["retained"], "articles": [o for o in outcomes if is_retained(o.get("decision"))],
            "ready_for_extraction": False})
        state = status(workspace)
        state.update(**summary)
        write_json(workspace / "campaign_state.json", state)
        return summary
