"""Prepare every selected article; a retrieval error never shrinks the cohort."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from phase1_extraction.artifacts import read_json, write_json, sha256_json, sha256_file, utc_now
from .discovery import fetch_fulltext
from .screening import POLICY_VERSION, build_document, make_request, verify_materials


def prepare_documents(candidates: list[dict], workspace: Path, *, workers: int = 4) -> list[dict]:
    """Resume per-article checkpoints, using PMC only to enrich selected PubMed hits."""
    workspace = Path(workspace)
    ids = [c["document_id"] for c in candidates]
    if not ids or len(set(ids)) != len(ids) or not 1 <= workers <= 8:
        raise ValueError("Distinct candidates and 1–8 workers required")

    def prepare_one(candidate):
        checkpoint = workspace / "documents" / (candidate["document_id"] + ".json")
        fingerprint = sha256_json([candidate, POLICY_VERSION])
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if saved["input_sha256"] != fingerprint:
                raise ValueError("Preparation input changed")
            doc = saved["document"]
            if sha256_json(doc) != saved["document_sha256"]:
                raise ValueError("Prepared document changed")
            verify_materials(doc)
        else:
            source = fetch_fulltext(candidate, workspace / "materials")
            doc = build_document(candidate, [source] if source else [])
            make_request(doc)
            write_json(checkpoint, {"input_sha256": fingerprint, "document": doc,
                                   "document_sha256": sha256_json(doc)})
        make_request(doc)
        return doc

    documents, failures = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(prepare_one, c): c["document_id"] for c in candidates}
        for future in as_completed(futures):
            identity = futures[future]
            try:
                documents[identity] = future.result()
            except Exception as error:
                failures.append({"document_id": identity, "error": f"{type(error).__name__}: {error}"})
            write_json(workspace / "preparation_progress.json", {
                "selected": len(ids), "prepared": len(documents), "failures": failures,
                "updated_at": utc_now()})
    if failures:
        raise ValueError(f"{len(failures)} article(s) failed preparation; no partial cohort submitted")
    return [documents[identity] for identity in ids]


def select_fulltext(candidates: list[dict], workspace: Path, *, target: int) -> list[dict]:
    """Walk PubMed order until exactly target verified full texts are available.

    Missing full text does not consume a place. A technical failure stops the
    run, rather than silently skipping an article that might belong in the cohort.
    Checkpoints let interrupted free retrieval resume without paying for inference.
    """
    workspace = Path(workspace)
    if target < 1:
        raise ValueError("Positive full-text target required")
    selected, documents, skipped, examined = [], [], [], 0
    while len(selected) < target and examined < len(candidates):
        # At the boundary this avoids fetching past the final selected article.
        page = candidates[examined:examined + min(100, target - len(selected))]
        prepared = prepare_documents(page, workspace)
        for candidate, document in zip(page, prepared, strict=True):
            if document["fulltext_available"] and document["materials"]:
                selected.append(candidate)
                documents.append(document)
            else:
                skipped.append(candidate["document_id"])
        examined += len(page)
        write_json(workspace / "selection_progress.json", {
            "target": target, "selected": len(selected), "examined": examined,
            "skipped_without_fulltext": len(skipped), "updated_at": utc_now()})
    if len(selected) != target:
        raise ValueError(f"Ranked PubMed pool exhausted: found {len(selected)} verified full texts "
                         f"of required {target} after {examined} candidates; nothing submitted")
    write_json(workspace / "candidates.json", {
        "format": "pubmed_verified_fulltext_cohort_v1", "candidate_count": target,
        "examined_count": examined, "candidates": selected})
    write_json(workspace / "selection_receipt.json", {
        "ranked_candidates_sha256": sha256_file(workspace / "ranked_candidates.json"),
        "candidates_sha256": sha256_file(workspace / "candidates.json"),
        "selected_count": target, "examined_count": examined,
        "skipped_without_fulltext": len(skipped), "skipped_document_ids": skipped})
    return documents


def validate_selection(workspace: Path) -> dict:
    """Bind selection to the inspected ranking and every availability checkpoint."""
    receipt = read_json(workspace / "selection_receipt.json")
    if receipt.get("selection_rule") == "reuse_verified_fulltext_cohort":
        return _validate_rescreen_selection(workspace, receipt)
    if (sha256_file(workspace / "ranked_candidates.json") != receipt["ranked_candidates_sha256"]
            or sha256_file(workspace / "candidates.json") != receipt["candidates_sha256"]):
        raise ValueError("Full-text selection changed")
    ranked = read_json(workspace / "ranked_candidates.json")["candidates"]
    candidates = read_json(workspace / "candidates.json")["candidates"]
    policy = read_json(workspace / "configuration.json")["policy_version"]
    examined = receipt["examined_count"]
    if not 0 < examined <= len(ranked):
        raise ValueError("Invalid inspected ranking length")
    selected, skipped = [], []
    for candidate in ranked[:examined]:
        saved = read_json(workspace / "documents" / (candidate["document_id"] + ".json"))
        doc = saved["document"]
        if (doc["policy_version"] != policy
                or saved["input_sha256"] != sha256_json([candidate, policy])
                or saved["document_sha256"] != sha256_json(doc)):
            raise ValueError("Availability checkpoint changed")
        verify_materials(doc)
        if doc["fulltext_available"] and doc["materials"]:
            selected.append(candidate)
        else:
            skipped.append(candidate["document_id"])
    if (selected != candidates or len(selected) != receipt["selected_count"]
            or skipped != receipt["skipped_document_ids"]
            or len(skipped) != receipt["skipped_without_fulltext"]
            or not selected or selected[-1] != ranked[examined - 1]):
        raise ValueError("Selection is not the first requested verified full texts")
    return receipt


def prepare_rescreen(source: Path, workspace: Path, source_plan: dict) -> list[dict]:
    """Build fresh policy inputs from the exact original cohort and verified text.

    This does not search, download, reuse model decisions, or copy paid states.
    Historical material paths remain immutable and are checked before submission.
    """
    from .screening_batch import validate_plan
    selected = read_json(source / "candidates.json")
    candidates = selected["candidates"]
    ids = [c["document_id"] for c in candidates]
    if ids != source_plan["document_ids"]:
        raise ValueError("Rescreening requires a complete full-text source cohort")
    records = {}
    for batch in source_plan["batches"]:
        plan = validate_plan(Path(batch["workspace"]))
        records.update((r["document"]["document_id"], r["document"]) for r in plan["requests"])
    documents = []
    for candidate in candidates:
        original = records[candidate["document_id"]]
        verify_materials(original)
        doc = build_document(candidate, [Path(m["path"]) for m in original["materials"]])
        if not doc["fulltext_available"] or not doc["materials"]:
            raise ValueError("Rescreened article no longer has verified full text")
        for field in ("document_id", "article_metadata", "blocks", "materials"):
            if doc[field] != original[field]:
                raise ValueError("Rescreening changed original article evidence")
        make_request(doc)
        write_json(workspace / "documents" / (doc["document_id"] + ".json"), {
            "input_sha256": sha256_json([candidate, POLICY_VERSION]), "document": doc,
            "document_sha256": sha256_json(doc)})
        documents.append(doc)
    write_json(workspace / "candidates.json", selected)
    source_receipt = read_json(source / "selection_receipt.json")
    write_json(workspace / "selection_receipt.json", {
        "selection_rule": "reuse_verified_fulltext_cohort", "source_workspace": str(source.resolve()),
        "source_plan_sha256": sha256_file(source / "campaign_plan.json"),
        "source_candidates_sha256": sha256_file(source / "candidates.json"),
        "candidates_sha256": sha256_file(workspace / "candidates.json"),
        "selected_count": len(documents), "examined_count": source_receipt["examined_count"],
        "skipped_without_fulltext": source_receipt["skipped_without_fulltext"],
        "previous_decisions_reused": False})
    return documents


def _validate_rescreen_selection(workspace: Path, receipt: dict) -> dict:
    source = Path(receipt["source_workspace"])
    if (source.resolve() == workspace.resolve()
            or sha256_file(source / "campaign_plan.json") != receipt["source_plan_sha256"]
            or sha256_file(source / "candidates.json") != receipt["source_candidates_sha256"]
            or sha256_file(workspace / "candidates.json") != receipt["candidates_sha256"]
            or receipt["candidates_sha256"] != receipt["source_candidates_sha256"]):
        raise ValueError("Rescreening source cohort changed")
    candidates = read_json(workspace / "candidates.json")["candidates"]
    config = read_json(workspace / "configuration.json")
    source_plan = read_json(source / "campaign_plan.json")
    if ([c["document_id"] for c in candidates] != source_plan["document_ids"]
            or len(candidates) != receipt["selected_count"]):
        raise ValueError("Rescreening cohort membership changed")
    for candidate in candidates:
        saved = read_json(workspace / "documents" / (candidate["document_id"] + ".json"))
        doc = saved["document"]
        if (saved["input_sha256"] != sha256_json([candidate, config["policy_version"]])
                or saved["document_sha256"] != sha256_json(doc)
                or doc["policy_version"] != config["policy_version"]
                or not doc["fulltext_available"] or not doc["materials"]):
            raise ValueError("Rescreening checkpoint changed")
        verify_materials(doc)
    return receipt
