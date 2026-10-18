from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from phase1_acquisition.ledger import (claim_submission, document_identifiers, evidence_sha256,
                                     find_completed, import_workspace, seen_identifiers)
from phase1_acquisition.screening_batch import collect, freeze_documents, submit, validate_plan
from phase1_extraction.artifacts import read_json, write_json
from test_literature_screening import FakeClient, document, response


def complete(tmp_path, documents, *, ledger=None, bad_first=False):
    workspace = tmp_path / "first"
    plan = freeze_documents(documents, workspace, ledger_path=ledger)
    rows = [{"custom_id": r["custom_id"], "response": {"status_code": 200, "body": response(r["document"])}, "error": None}
            for r in plan["requests"]]
    if bad_first:
        rows[0]["response"]["body"]["status"] = "incomplete"
    client = FakeClient(rows)
    submit(workspace, client=client)
    summary = collect(workspace, client=client)
    return workspace, summary


def test_normalized_identifiers_and_content_key_ignore_storage_labels(tmp_path):
    doc = document(tmp_path)
    doc["article_metadata"].update(doi="https://doi.org/10.1234/TEST", pmid="PMID: 42", pmcid="123")
    assert document_identifiers(doc) == {"doi:10.1234/test", "pmid:42", "pmcid:PMC123"}
    renamed = deepcopy(doc)
    renamed["document_id"] = "different-local-name"
    renamed["materials"][0]["path"] = "/different/path.xml"
    renamed["blocks"][0]["source_id"] = "another-label"
    renamed["article_metadata"]["doi"] = "doi:10.1234/test"
    assert evidence_sha256(renamed) == evidence_sha256(doc)
    renamed["blocks"][0]["text"] += " New observed toxicity."
    assert evidence_sha256(renamed) != evidence_sha256(doc)


def test_reuses_verified_decision_without_changing_original_request(tmp_path):
    ledger = tmp_path / "ledger.json"
    doc = document(tmp_path)
    first, _ = complete(tmp_path, [doc], ledger=ledger)
    frozen = read_json(first / "screening_plan.json")
    relocated = deepcopy(doc)
    relocated["document_id"] = "new_alias"
    relocated["blocks"][0]["source_id"] = "different_source_label"
    new_file = tmp_path / "relocated.xml"
    new_file.write_bytes(Path(doc["materials"][0]["path"]).read_bytes())
    relocated["materials"][0]["path"] = str(new_file)
    second = tmp_path / "second"
    result = freeze_documents([relocated], second, ledger_path=ledger)
    assert result["status"] == "all_reused"
    assert result["request_count"] == 0 and result["reserved_usd"] == 0
    assert not (second / "screening_plan.json").exists()
    report = read_json(second / "screening_reuse_report.json")
    assert report["reused"][0]["document_id"] == "new_alias"
    assert report["reused"][0]["decision"] == "\u4fdd\u7559"
    assert read_json(first / "screening_plan.json") == frozen
    assert seen_identifiers(ledger) == {"pmcid:PMC123"}


def test_only_same_policy_and_evidence_are_reused(tmp_path):
    ledger = tmp_path / "ledger.json"
    doc = document(tmp_path)
    complete(tmp_path, [doc], ledger=ledger)
    changed = deepcopy(doc)
    changed["blocks"][0]["text"] += " Follow-up results."
    assert find_completed(changed, ledger) is None
    changed = deepcopy(doc)
    changed["policy_version"] = "phase1_binary_20261011_v2"
    assert find_completed(changed, ledger) is None
    another = document(tmp_path, "PMC456")
    plan = freeze_documents([doc, another], tmp_path / "second", ledger_path=ledger)
    assert plan["request_count"] == 1
    assert plan["requests"][0]["document"]["document_id"] == "PMC456"
    assert validate_plan(tmp_path / "second") == plan


def test_new_dlt_policy_does_not_reuse_old_scientific_decisions(tmp_path):
    from phase1_acquisition.screening import POLICY_VERSION
    ledger = tmp_path / "ledger.json"
    doc = document(tmp_path)
    complete(tmp_path, [doc], ledger=ledger)
    new_policy_document = deepcopy(doc)
    new_policy_document["policy_version"] = POLICY_VERSION
    assert evidence_sha256(new_policy_document) == evidence_sha256(doc)
    assert find_completed(new_policy_document, ledger) is None
    plan = freeze_documents([new_policy_document], tmp_path / "new_policy", ledger_path=ledger)
    assert plan["request_count"] == 1 and plan["policy_version"] == POLICY_VERSION
    # A prior broad-policy payment claim must not prevent an authorized stricter-policy run.
    claim_submission(plan, tmp_path / "new_policy", ledger)
    assert {entry["policy_version"] for entry in read_json(ledger)["entries"]} == {doc["policy_version"], POLICY_VERSION}


def test_pending_or_ambiguous_payment_blocks_other_workspace(tmp_path):
    ledger = tmp_path / "ledger.json"
    doc = document(tmp_path)
    for name in ("one", "two"):
        freeze_documents([doc], tmp_path / name, ledger_path=ledger)
    first_client = FakeClient(unknown=True)
    with pytest.raises(TimeoutError):
        submit(tmp_path / "one", client=first_client)
    assert first_client.creates == 1
    second_client = FakeClient()
    with pytest.raises(RuntimeError, match="paid-submission claim"):
        submit(tmp_path / "two", client=second_client)
    assert second_client.creates == 0
    with pytest.raises(RuntimeError, match="outcome unknown"):
        submit(tmp_path / "one", client=first_client)
    assert first_client.creates == 1


def test_failed_response_is_not_a_scientific_exclusion_or_automatic_retry(tmp_path):
    ledger = tmp_path / "ledger.json"
    doc = document(tmp_path)
    _, summary = complete(tmp_path, [doc], ledger=ledger, bad_first=True)
    assert summary["technical_failures"] == 1
    assert find_completed(doc, ledger) is None
    entry = read_json(ledger)["entries"][0]
    assert entry["status"] == "technical_failure" and "decision" not in entry
    freeze_documents([doc], tmp_path / "second", ledger_path=ledger)
    client = FakeClient()
    with pytest.raises(RuntimeError, match="paid-submission claim"):
        submit(tmp_path / "second", client=client)
    assert client.creates == 0


def test_import_legacy_history_is_verified_idempotent_and_skips_failures(tmp_path):
    one, two = document(tmp_path), document(tmp_path, "PMC456")
    workspace, _ = complete(tmp_path, [one, two], bad_first=True)
    frozen = (workspace / "screening_plan.json").read_bytes()
    ledger = tmp_path / "ledger.json"
    report = import_workspace(workspace, ledger)
    assert report["imported"] == 1 and report["verified"] == 1
    assert report["technical_failures_not_imported"] == ["PMC123"]
    assert report["technical_failure_claims_recorded"] == 1
    assert report["rejected"] == []
    assert import_workspace(workspace, ledger)["imported"] == 0
    assert find_completed(one, ledger) is None
    assert find_completed(two, ledger)["decision"] == "\u4fdd\u7559"
    assert (workspace / "screening_plan.json").read_bytes() == frozen
    freeze_documents([one], tmp_path / "failed_retry", ledger_path=ledger)
    retry_client = FakeClient()
    with pytest.raises(RuntimeError, match="paid-submission claim"):
        submit(tmp_path / "failed_retry", client=retry_client)
    assert retry_client.creates == 0


def test_tampered_raw_response_is_not_reused_and_import_reports_error(tmp_path):
    doc = document(tmp_path)
    workspace, summary = complete(tmp_path, [doc])
    ledger = tmp_path / "ledger.json"
    import_workspace(workspace, ledger)
    record = read_json(Path(summary["outcomes"][0]["record_path"]))
    raw_path = Path(record["raw_response_path"])
    raw = read_json(raw_path)
    raw["usage"]["input_tokens"] += 1
    write_json(raw_path, raw)
    with pytest.raises(ValueError, match="raw response changed"):
        freeze_documents([doc], tmp_path / "second", ledger_path=ledger)
    report = import_workspace(workspace, tmp_path / "other_ledger.json")
    assert report["imported"] == 0 and report["rejected"][0]["document_id"] == "PMC123"


def test_claim_all_or_nothing_when_one_article_is_already_pending(tmp_path):
    ledger = tmp_path / "ledger.json"
    doc = document(tmp_path)
    occupied = freeze_documents([doc], tmp_path / "occupied", ledger_path=ledger)
    claim_submission(occupied, tmp_path / "occupied", ledger)
    other = document(tmp_path, "PMC456")
    plan = freeze_documents([other, doc], tmp_path / "second", ledger_path=ledger)
    with pytest.raises(RuntimeError, match="paid-submission claim"):
        claim_submission(plan, tmp_path / "second", ledger)
    assert seen_identifiers(ledger) == {"pmcid:PMC123"}


def test_ledger_path_cannot_silently_change_for_new_plan(tmp_path):
    workspace = tmp_path / "batch"
    freeze_documents([document(tmp_path)], workspace, ledger_path=tmp_path / "ledger.json")
    client = FakeClient()
    with pytest.raises(ValueError, match="ledger differs"):
        submit(workspace, client=client, ledger_path=tmp_path / "other.json")
    assert client.creates == 0


def test_shared_doi_cannot_hide_conflicting_pmid(tmp_path):
    ledger = tmp_path / "ledger.json"
    doc = document(tmp_path)
    doc["article_metadata"].update(doi="10.1/example", pmid="123")
    complete(tmp_path, [doc], ledger=ledger)
    conflict = deepcopy(doc)
    conflict["article_metadata"]["pmid"] = "456"
    with pytest.raises(ValueError, match="Conflicting article identifiers"):
        find_completed(conflict, ledger)


def test_scientific_exclusions_are_reusable(tmp_path):
    ledger, workspace = tmp_path / "ledger.json", tmp_path / "batch"
    doc = document(tmp_path)
    plan = freeze_documents([doc], workspace, ledger_path=ledger)
    client = FakeClient([{"custom_id": plan["requests"][0]["custom_id"],
                          "response": {"status_code": 200, "body": response(doc, keep=False)}, "error": None}])
    submit(workspace, client=client)
    assert collect(workspace, client=client)["excluded"] == 1
    assert find_completed(doc, ledger)["decision"] == "\u4e0d\u4fdd\u7559"
    assert freeze_documents([doc], tmp_path / "repeat", ledger_path=ledger)["status"] == "all_reused"
