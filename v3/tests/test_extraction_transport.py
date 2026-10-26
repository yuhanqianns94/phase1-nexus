"""Offline regressions for request identity, paid-response recovery and budgets."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase1_extraction import llm
from phase1_extraction.artifacts import read_json, sha256_json, write_json
from phase1_extraction.models import DocumentIR, PageIR, ReportedExtraction
from phase1_extraction.request_cache import artifact_lock


def packet(index=1, count=1):
    return {"document_id": "DOC", "packet_id": f"packet_{index:04d}",
            "packet_index": index, "packet_count": count,
            "is_final_packet": index == count, "source_units": []}


def response(title="result", complete=True):
    parsed = ReportedExtraction(document_id="DOC", title=title, extraction_complete=complete)
    return {"id": "resp_test", "status": "completed", "usage": {
        "input_tokens": 1000, "output_tokens": 100,
        "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300}},
        "output": [{"content": [{"type": "output_text", "text": parsed.model_dump_json()}]}]}


class Client:
    def __init__(self, callback=None):
        self.calls = []
        self.callback = callback or (lambda body: response(title=body["model"]))
        self.responses = self

    def create(self, **body):
        self.calls.append(body)
        return self.callback(body)


def run(tmp_path, client, **kwargs):
    return llm.extract_packets(
        packets=kwargs.pop("packets", [packet()]),
        document=kwargs.pop("document", DocumentIR(document_id="DOC", materials=[])),
        raw_response_dir=tmp_path / "raw_responses", client=client, **kwargs)


def budget(limit=10):
    return llm.BudgetGuard(max_usd=limit, input_usd_per_million=4, output_usd_per_million=20,
                           cached_input_multiplier=0.1)


def test_cached_response_is_free_and_rebuilds_missing_or_tampered_parsed(tmp_path):
    client = Client()
    original, _ = run(tmp_path, client)
    parsed_path = tmp_path / "parsed_packets/packet_0001.json"
    parsed_path.unlink()
    guard = budget(0.000001)
    restored, summaries = run(tmp_path, client, budget_guard=guard)
    assert restored == original and len(client.calls) == 1
    assert guard.spent_usd == 0
    assert summaries[0]["resumed_without_request"]
    assert summaries[0]["usage"]["input_tokens"] == 1000
    write_json(parsed_path, ReportedExtraction(document_id="WRONG", title="corrupted"))
    assert run(tmp_path, client)[0] == original
    assert len(client.calls) == 1


def test_failed_new_configuration_never_relabels_old_cached_result(tmp_path):
    client = Client()
    run(tmp_path, client, model="old")
    with pytest.raises(RuntimeError, match="reserve"):
        run(tmp_path, client, model="new", budget_guard=budget(0.000001))
    new, summary = run(tmp_path, client, model="new")
    assert new.title == "new" and len(client.calls) == 2
    assert not summary[0]["resumed_without_request"]


def test_api_failure_does_not_bind_old_raw_to_new_request(tmp_path):
    client = Client()
    run(tmp_path, client, model="old")
    def fail(body):
        raise RuntimeError("network failed")
    client.callback = fail
    with pytest.raises(RuntimeError, match="network failed"):
        run(tmp_path, client, model="new")
    client.callback = lambda body: response(body["model"])
    assert run(tmp_path, client, model="new")[0].title == "new"
    assert len(client.calls) == 3


def test_parse_failure_is_recovered_locally_without_second_payment(tmp_path, monkeypatch):
    client = Client()
    parse = llm.parse_model_response
    def fail(*args, **kwargs):
        raise RuntimeError("local parser failed")
    monkeypatch.setattr(llm, "parse_model_response", fail)
    guard = budget()
    with pytest.raises(RuntimeError, match="local parser failed"):
        run(tmp_path, client, budget_guard=guard)
    assert guard.spent_usd > 0
    monkeypatch.setattr(llm, "parse_model_response", parse)
    result, summary = run(tmp_path, client, budget_guard=budget(0.000001))
    assert result.document_id == "DOC" and len(client.calls) == 1
    assert summary[0]["resumed_without_request"]
    assert not (tmp_path / "raw_responses/packet_0001.failure.json").exists()


def test_incomplete_paid_response_is_not_repeated_automatically(tmp_path):
    client = Client(lambda body: {**response(), "status": "incomplete",
                                  "incomplete_details": {"reason": "max_output_tokens"}})
    for _ in range(2):
        with pytest.raises(RuntimeError, match="incomplete"):
            run(tmp_path, client)
    assert len(client.calls) == 1
    client.callback = lambda body: response()
    assert run(tmp_path, client, max_output_tokens=50000)[0].extraction_complete
    assert len(client.calls) == 2


def test_wrong_document_response_never_enters_merge(tmp_path):
    wrong = response()
    wrong["output"][0]["content"][0]["text"] = ReportedExtraction(document_id="OTHER").model_dump_json()
    client = Client(lambda body: wrong)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="different document_id"):
            run(tmp_path, client)
    assert len(client.calls) == 1


def test_explicit_no_resume_and_corrupt_receipt_trigger_fresh_requests(tmp_path):
    client = Client()
    run(tmp_path, client)
    run(tmp_path, client, resume=False)
    assert len(client.calls) == 2
    raw = tmp_path / "raw_responses/packet_0001.json"
    write_json(raw, response("tampered"))
    result, _ = run(tmp_path, client)
    assert result.title != "tampered" and len(client.calls) == 3


def test_legacy_unbound_parsed_cache_is_not_trusted(tmp_path):
    client = Client()
    run(tmp_path, client)
    (tmp_path / "raw_responses/packet_0001.receipt.json").unlink()
    run(tmp_path, client)
    assert len(client.calls) == 2


@pytest.mark.parametrize("final_complete", [True, False])
def test_multi_packet_completion_follows_successful_final_packet(tmp_path, final_complete):
    replies = iter([response(complete=False), response(complete=final_complete)])
    client = Client(lambda body: next(replies))
    result, _ = run(tmp_path, client, packets=[packet(1, 2), packet(2, 2)])
    assert result.extraction_complete is final_complete
    cached, _ = run(tmp_path, client, packets=[packet(1, 2), packet(2, 2)])
    assert cached == result and len(client.calls) == 2


def test_missing_required_visual_fails_before_paid_request(tmp_path):
    page = PageIR(page_id="p1", material_id="mat", page_number=1, width=100, height=100, native_text_status="readable", image_path=str(tmp_path / "missing.png"))
    doc = DocumentIR(document_id="DOC", materials=[], pages=[page])
    source = packet()
    source["source_units"] = [{"unit_kind": "page_visual", "page": {"source_id": "p1"}}]
    client = Client()
    with pytest.raises(ValueError, match="missing visual"):
        run(tmp_path, client, document=doc, packets=[source])
    assert client.calls == []


def test_competing_extraction_is_rejected_before_any_request(tmp_path):
    client = Client()
    with artifact_lock(tmp_path / ".extraction.lock"):
        with pytest.raises(RuntimeError, match="in use"):
            run(tmp_path, client)
    assert client.calls == []
    run(tmp_path, client)
    assert len(client.calls) == 1


def test_budget_counts_full_request_and_cache_pricing():
    guard = budget(0.01)
    body = llm.build_request_body(packet=packet(), prior=None, image_paths=[], model="test",
                                  reasoning_effort="low", max_output_tokens=1)
    guard.admit(packet_chars=100, image_count=0, max_output_tokens=1)
    with pytest.raises(RuntimeError, match="reserve"):
        guard.admit_request(body)
    usage = response()["usage"]
    guard.record(usage)
    assert guard.spent_usd == pytest.approx(((500 + 200 * 0.1 + 300 * 1.25) * 4 + 100 * 20) / 1e6)
    sdk_guard = budget()
    sdk_guard.record(SimpleNamespace(input_tokens=1000, output_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=200, cache_write_tokens=300)))
    assert sdk_guard.spent_usd == guard.spent_usd


@pytest.mark.parametrize("parameter", ["max_usd", "input_usd_per_million", "output_usd_per_million",
                                        "cache_write_multiplier", "cached_input_multiplier"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_budget_values_rejected(parameter, value):
    kwargs = {"max_usd": 10, "input_usd_per_million": 4, "output_usd_per_million": 20}
    kwargs[parameter] = value
    with pytest.raises(ValueError):
        llm.BudgetGuard(**kwargs)


def test_missing_usage_conservatively_exhausts_budget():
    guard = budget()
    guard.record(None)
    assert guard.spent_usd == guard.max_usd


def test_request_contract_is_unchanged():
    body = llm.build_request_body(packet={"document_id": "CONTRACT", "packet_id": "packet_0001",
        "source_units": [], "is_final_packet": True}, prior=None, image_paths=[],
        model="gpt-5.6-sol", reasoning_effort="low", max_output_tokens=48000)
    # Refresh for primary/co-primary research roles and removal of the R3 imputation rule.
    # The complete prompt, strict schema, settings and serialization stay pinned.
    assert sha256_json(body) == "9e8255bf50bc1b69ffbee259b998a4784e2ebb89548b18adc967ba83c5b03f4f"


def test_resume_restores_missing_and_stale_request_provenance(tmp_path):
    client = Client()
    run(tmp_path, client, model="old")
    request_path = tmp_path / "raw_requests/packet_0001.json"
    request_path.unlink()
    _, summaries = run(tmp_path, client, model="old")
    assert read_json(request_path)["request_sha256"] == summaries[0]["request_sha256"]
    with pytest.raises(RuntimeError, match="reserve"):
        run(tmp_path, client, model="new", budget_guard=budget(0.000001))
    _, summaries = run(tmp_path, client, model="old")
    assert read_json(request_path)["body"]["model"] == "old"
    assert len(client.calls) == 1


@pytest.mark.parametrize("matches", [True, False])
def test_legacy_complete_response_migrates_only_with_matching_service_id(tmp_path, matches):
    client = Client()
    _, summaries = run(tmp_path, client)
    (tmp_path / "raw_responses/packet_0001.receipt.json").unlink()
    if not matches:
        summaries[0]["response_id"] = "some_other_response"
    write_json(tmp_path / "request_summary.json", {"status": "completed", "requests": summaries})
    _, recovered = run(tmp_path, client)
    assert recovered[0]["resumed_without_request"] is matches
    assert len(client.calls) == (1 if matches else 2)


def test_pipeline_conflict_does_not_touch_existing_manifest(tmp_path):
    from phase1_extraction.pipeline import run_pipeline
    paths_root = tmp_path / "DOC"
    manifest = {"document_id": "DOC", "status": "in_progress", "preserve": True}
    write_json(paths_root / "manifest.json", manifest)
    with artifact_lock(paths_root / ".pipeline.lock"):
        with pytest.raises(RuntimeError, match="in use"):
            run_pipeline(document_id="DOC", source_paths=[tmp_path / "missing.xml"], output_root=tmp_path)
    assert read_json(paths_root / "manifest.json") == manifest


def test_empty_usage_conservatively_exhausts_budget():
    guard = budget()
    guard.record({})
    assert guard.spent_usd == guard.max_usd


def test_cli_exposes_no_resume_for_all_sync_entrypoints():
    from phase1_extraction.cli import build_parser
    parser = build_parser()
    for args in (["run", "DOC", "source.xml"], ["extract", "DOC"], ["batch", "manifest.jsonl"]):
        assert parser.parse_args(args + ["--no-resume"]).no_resume
        assert not parser.parse_args(args).no_resume


def test_failed_atomic_write_keeps_previous_artifact_and_cleans_temporary(tmp_path, monkeypatch):
    from phase1_extraction import artifacts
    path = tmp_path / "artifact.json"
    write_json(path, {"old": True})
    def fail_replace(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        write_json(path, {"new": True})
    assert read_json(path) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_default_model_migration_preserves_extraction_contract():
    from phase1_extraction.cli import build_parser
    assert llm.DEFAULT_MODEL == "gpt-6.1-sol"
    assert llm.DEFAULT_REASONING_EFFORT == "low"
    assert llm.DEFAULT_CACHED_INPUT_MULTIPLIER == 0.05
    args = build_parser().parse_args(["extract", "DOC"])
    assert args.model == llm.DEFAULT_MODEL and args.reasoning_effort == "low"
    assert args.cached_input_multiplier == 0.05
    common = dict(packet=packet(), prior=None, image_paths=[], reasoning_effort="low", max_output_tokens=48000)
    old_body = llm.build_request_body(model="gpt-5.6-sol", **common)
    new_body = llm.build_request_body(model=llm.DEFAULT_MODEL, **common)
    assert {k: v for k, v in old_body.items() if k != "model"} == {
        k: v for k, v in new_body.items() if k != "model"}
