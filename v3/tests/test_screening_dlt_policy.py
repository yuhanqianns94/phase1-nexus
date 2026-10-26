"""Policy/transport boundaries; these fixtures do not measure LLM clinical accuracy."""
from copy import deepcopy
import json

import pytest

from phase1_acquisition.screening import (
    CRITERIA, CURRENT_PROMPT, DLTDecision, DLT_POLICIES, DLT_POLICY_VERSION, DLT_PROMPT, DRUG_POLICY_VERSION, DRUG_PROMPT,
    EFFORT, LEGACY_CRITERIA, LEGACY_POLICY_VERSION, MODEL, POLICY_VERSION,
    REFERENCE_POLICY_VERSION, SUPPORTIVE_CARE_POLICY_VERSION,
    build_document, is_excluded, is_retained, make_request, validate_decision,
)
from phase1_acquisition.screening_batch import collect, export_retained, freeze_documents, submit
from phase1_extraction.artifacts import read_json, sha256_json
from test_literature_screening import FakeClient, change_result, document


def current_response(doc, *, keep=True):
    block = next(b for b in doc["blocks"] if "Twelve" in b["text"])
    checks = [{"criterion": name,
               "satisfied": keep or name != "actual_dose_dlt_count_and_population",
               "reason": "The cited article evidence establishes this criterion." if keep
                         else "The dose cohort lacks a recoverable DLT count.",
               "evidence": [{"source_id": block["source_id"]}]} for name in sorted(CRITERIA)]
    value = {"document_id": doc["document_id"], "decision": "retain" if keep else "exclude",
             "reason": "One actual dose cohort has 0 DLTs among 12 treated patients." if keep
                       else "No dose cohort has a recoverable DLT count.", "checks": checks}
    return {"status": "completed", "model": MODEL, "reasoning": {"effort": EFFORT},
            "usage": {"input_tokens": 1000, "output_tokens": 500},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}


@pytest.fixture(params=[DLT_POLICY_VERSION, DRUG_POLICY_VERSION, POLICY_VERSION], ids=["frozen_v4", "frozen_v5", "current_v6"])
def current_document(tmp_path, request):
    doc = document(tmp_path)
    doc["policy_version"] = request.param
    return doc


@pytest.mark.parametrize("policy,digest", [
    (LEGACY_POLICY_VERSION, "9259e0d443bb39945695b71192ac6fd6a4cb7a9a97a1bb1928dcdc20c7b3a009"),
    (REFERENCE_POLICY_VERSION, "beca0cc114b5bce31de26e7da9faacbcbc44851e41838ebb446b2c34204bfbe9"),
    (SUPPORTIVE_CARE_POLICY_VERSION, "5b1053b592c8db2c973bb12cfa20ae972a2520e17bad35b050befb3bb1fec331"),
    (DLT_POLICY_VERSION, "879b2e70067af331a88e33ce6d973773947c110eb634215edb67a57b65796f08"),
    (DRUG_POLICY_VERSION, "337456c39bf58e630f6fab97c9d729a0f1d44f5b2081ea2af54807ccefa3d774"),
])
def test_all_legacy_request_bodies_keep_their_pre_migration_hash(policy, digest):
    doc = {"document_id": "PMID1", "policy_version": policy,
           "article_metadata": {"title": "Phase I cancer study"}, "fulltext_available": False,
           "materials": [], "blocks": [{"source_id": "metadata:title:s0:0", "text": "Phase I cancer study"}]}
    assert sha256_json(make_request(doc)) == digest


def test_dlt_policies_have_english_binary_decisions_and_preserve_wire_schema(current_document):
    request = make_request(current_document)
    expected_prompt = {DLT_POLICY_VERSION: DLT_PROMPT, DRUG_POLICY_VERSION: DRUG_PROMPT, POLICY_VERSION: CURRENT_PROMPT}[current_document["policy_version"]]
    assert request["instructions"] == expected_prompt
    schema = request["text"]["format"]["schema"]
    assert schema["properties"]["decision"]["enum"] == ["retain", "exclude"]
    assert set(schema["$defs"]["DLTReferenceCheck"]["properties"]["criterion"]["enum"]) == CRITERIA
    assert "quote" not in schema["$defs"]["SourceReference"]["properties"]
    assert CRITERIA - LEGACY_CRITERIA == {"actual_dose_dlt_count_and_population"}
    assert build_document({"document_id": "PMID1", "title": "Example"}, [])["policy_version"] == POLICY_VERSION


def test_drug_scope_migration_changes_only_versioned_article_and_prompt(current_document):
    prior_document = deepcopy(current_document)
    prior_document["policy_version"] = DLT_POLICY_VERSION
    latest_document = deepcopy(current_document)
    latest_document["policy_version"] = POLICY_VERSION
    prior_request = make_request(prior_document)
    latest_request = make_request(latest_document)

    assert DLT_POLICIES == {DLT_POLICY_VERSION, DRUG_POLICY_VERSION, POLICY_VERSION}
    assert DLT_POLICY_VERSION != POLICY_VERSION
    assert prior_request["instructions"] != latest_request["instructions"]
    for key in prior_request.keys() - {"instructions", "input"}:
        assert prior_request[key] == latest_request[key]
    assert prior_document["blocks"] == latest_document["blocks"]
    assert prior_document["materials"] == latest_document["materials"]


@pytest.mark.parametrize("keep", [True, False])
def test_current_decisions_bind_verbatim_evidence_without_mutating_response(current_document, keep):
    raw = current_response(current_document, keep=keep)
    before = deepcopy(raw)
    result = validate_decision(raw, current_document)
    assert isinstance(result, DLTDecision)
    assert result.decision == ("retain" if keep else "exclude")
    assert is_retained(result.decision) is keep and is_excluded(result.decision) is (not keep)
    sources = {b["source_id"]: b["text"] for b in current_document["blocks"]}
    assert all(e.quote == sources[e.source_id] for c in result.checks for e in c.evidence)
    assert raw == before


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(decision="exclude"), "contradicts"),
    (lambda r: r.update(decision="review"), "literal"),
    (lambda r: r.update(decision="\u4fdd\u7559"), "literal"),
    (lambda r: r.update(document_id="OTHER"), "identity"),
    (lambda r: r["checks"][0].update(evidence=[]), "positive criterion"),
    (lambda r: r["checks"][0]["evidence"][0].update(source_id="invented"), "Unknown evidence"),
    (lambda r: r["checks"].pop(), "four distinct"),
    (lambda r: r["checks"].__setitem__(0, deepcopy(r["checks"][1])), "four distinct"),
    (lambda r: r["checks"][0].update(criterion="actual_dose_and_toxicity"), "literal"),
    (lambda r: r.update(reason="\u6709\u539f\u6587\u4f9d\u636e"), "English"),
    (lambda r: r["checks"][0].update(reason="\u4ec5\u8ba1\u5212"), "English"),
])
def test_current_invalid_output_is_a_validation_failure(current_document, mutation, match):
    with pytest.raises(ValueError, match=match):
        validate_decision(change_result(current_response(current_document), mutation), current_document)


def test_original_source_language_is_preserved(current_document):
    current_document["blocks"].append({"source_id": "body:original", "text": "\u65e0\u5242\u91cf\u9650\u5236\u6027\u6bd2\u6027"})
    raw = current_response(current_document)
    change_result(raw, lambda r: r["checks"][0]["evidence"].append({"source_id": "body:original"}))
    result = validate_decision(raw, current_document)
    assert result.checks[0].evidence[-1].quote == current_document["blocks"][-1]["text"]


def test_current_english_roundtrip_and_technical_failure_counts(tmp_path, current_document, monkeypatch):
    workspace = tmp_path / "batch"
    second = deepcopy(current_document)
    second["document_id"] = "other_article"
    plan = freeze_documents([current_document, second], workspace)
    bad = current_response(second)
    change_result(bad, lambda r: r.update(reason="\u4fdd\u7559"))
    rows = [{"custom_id": item["custom_id"], "response": {"status_code": 200,
             "body": current_response(current_document) if index == 0 else bad}, "error": None}
            for index, item in enumerate(plan["requests"])]
    client = FakeClient(rows)
    submit(workspace, client=client)
    summary = collect(workspace, client=client)
    assert (summary["retained"], summary["excluded"], summary["technical_failures"]) == (1, 0, 1)
    assert "decision" not in summary["outcomes"][1]
    assert "English" in summary["outcomes"][1]["error"]
    rows[1]["response"]["body"] = current_response(second, keep=False)
    (workspace / "output_file_id.jsonl").unlink()
    summary = collect(workspace, client=client)
    assert (summary["retained"], summary["excluded"], summary["technical_failures"]) == (1, 1, 0)
    from phase1_acquisition import screening_batch
    monkeypatch.setattr(screening_batch, "material_preflight", lambda *a: {"ready": True})
    assert export_retained(workspace, tmp_path / "eligible.jsonl")["exported"] == 1
    record = read_json(workspace / "decisions" / (current_document["document_id"] + ".json"))
    assert record["result"]["decision"] == "retain"
