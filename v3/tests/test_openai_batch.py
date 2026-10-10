from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from openai import APIStatusError, OpenAI

from phase1_extraction.artifacts import sha256_file, sha256_json, write_json
from phase1_extraction.llm import build_request_body
from phase1_extraction.models import ReportedExtraction
from phase1_extraction.openai_batch import (
    FrozenBatchRequest,
    collect_batch_results,
    extract_batch_response,
    read_batch_output,
    run_frozen_sync,
    submit_batch_files,
    validate_batch_input_files,
    write_frozen_request_index,
    write_batch_input_files,
)
from phase1_extraction.models import DocumentIR, Material, PageIR, TableBundle, TableFragmentIR
from phase1_extraction.packets import packet_image_paths


def prepared_workspace(tmp_path: Path) -> tuple[FrozenBatchRequest, Path, list[Path]]:
    frozen = request("DOC")
    source_root = tmp_path / "source"
    source_root.mkdir()
    write_json(source_root / "document_ir.json", DocumentIR(document_id="DOC", materials=[]))
    frozen = replace(
        frozen, request_sha256=sha256_json(frozen.body), input_article_root=str(source_root),
        source_document_sha256=sha256_file(source_root / "document_ir.json"),
    )
    write_json(source_root / "raw_requests" / "packet_0001.json", {"body": frozen.body})
    index = write_frozen_request_index([frozen], tmp_path / "frozen_requests.json")
    return frozen, index, write_batch_input_files([frozen], tmp_path)


def request(document_id: str, padding: int = 0) -> FrozenBatchRequest:
    packet = {
        "document_id": document_id,
        "packet_id": "packet_0001",
        "source_units": [{"unit_kind": "text", "items": [{"text": "x" * padding}]}],
        "is_final_packet": True,
    }
    body = build_request_body(
        packet=packet,
        prior=None,
        image_paths=[],
        model="gpt-5.6-sol",
        reasoning_effort="low",
        max_output_tokens=48_000,
    )
    return FrozenBatchRequest(
        document_id=document_id,
        packet_id="packet_0001",
        custom_id=f"v3_1-{document_id}",
        request_sha256="hash",
        body=body,
        input_article_root="/tmp/input",
        material_coverage_complete=True,
    )


def test_batch_line_reuses_the_exact_sync_body_and_strict_schema(tmp_path: Path) -> None:
    frozen = request("DOC")
    paths = write_batch_input_files([frozen], tmp_path)
    line = json.loads(paths[0].read_text())
    assert line["body"] == frozen.body
    assert line["body"]["text"]["format"]["type"] == "json_schema"
    assert line["body"]["text"]["format"]["strict"] is True
    assert line["body"]["input"][0]["content"] == frozen.body["input"][0]["content"]


def test_batch_files_split_by_serialized_bytes_without_splitting_a_request(tmp_path: Path) -> None:
    first, second = request("A", 200), request("B", 200)
    one_line_size = len((json.dumps(first.jsonl_record(), ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    paths = write_batch_input_files([first, second], tmp_path, max_file_bytes=one_line_size + 20)
    assert len(paths) == 2
    assert [len(path.read_text().splitlines()) for path in paths] == [1, 1]


def test_batch_output_is_matched_by_custom_id_not_return_order(tmp_path: Path) -> None:
    first, second = request("A"), request("B")
    path = tmp_path / "responses.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"custom_id": item.custom_id, "response": {"status_code": 200, "body": {
                "output": [{"content": [{"type": "output_text", "text": ReportedExtraction(document_id=item.document_id).model_dump_json()}]}]
            }}})
            for item in (second, first)
        ),
        encoding="utf-8",
    )
    outputs = read_batch_output([path])
    assert list(outputs) == [second.custom_id, first.custom_id]
    parsed, _raw = extract_batch_response(outputs[first.custom_id], first)
    assert parsed.document_id == "A"


def test_packet_images_are_not_silently_truncated_at_sixteen(tmp_path: Path) -> None:
    tables = []
    pages = []
    units = []
    for index in range(20):
        image = tmp_path / f"table_{index}.png"
        image.write_bytes(b"png")
        page_id = f"page_{index}"
        table_id = f"table_{index}"
        pages.append(
            PageIR(
                page_id=page_id, material_id="material", page_number=index + 1,
                width=100, height=100, native_text_status="readable", image_path=str(image),
            )
        )
        tables.append(
            TableBundle(
                table_id=table_id, material_id="material", row_count=0, column_count=0,
                parse_status="visual_only",
                fragments=[
                    TableFragmentIR(
                        fragment_id=f"fragment_{index}", material_id="material", page_id=page_id,
                        image_path=str(image), source="visual_candidate",
                    )
                ],
            )
        )
        units.append({"unit_kind": "table", "table": {"source_id": table_id}})
    document = DocumentIR(
        document_id="DOC",
        materials=[
            Material(
                material_id="material", path=str(tmp_path), media_type="other",
                sha256="0" * 64, byte_count=0,
            )
        ],
        pages=pages,
        tables=tables,
    )
    assert len(packet_image_paths({"source_units": units}, document)) == 20


def test_incomplete_response_is_not_accepted_even_with_valid_json() -> None:
    frozen = request("DOC")
    item = {
        "custom_id": frozen.custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"content": [{"type": "output_text", "text": ReportedExtraction(document_id="DOC").model_dump_json()}]}],
            },
        },
    }
    with pytest.raises(RuntimeError, match="incomplete"):
        extract_batch_response(item, frozen)


def test_validation_rejects_changed_batch_body(tmp_path: Path) -> None:
    frozen = request("DOC")
    frozen = replace(frozen, request_sha256=sha256_json(frozen.body))
    frozen_path = tmp_path / "packet.json"
    frozen_path.write_text(json.dumps({"body": frozen.body}), encoding="utf-8")
    indexed = replace(frozen, input_article_root=str(tmp_path))
    # Match the path convention used by the request index.
    raw_dir = tmp_path / "raw_requests"
    raw_dir.mkdir()
    (raw_dir / "packet_0001.json").write_text(json.dumps({"body": frozen.body}), encoding="utf-8")
    index = write_frozen_request_index([indexed], tmp_path / "frozen_requests.json")
    files = write_batch_input_files([indexed], tmp_path)
    record = json.loads(files[0].read_text())
    record["body"]["max_output_tokens"] = 1
    files[0].write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after freezing"):
        validate_batch_input_files(files, index)


def test_submit_is_idempotent_after_local_state_records_batch_id(tmp_path: Path) -> None:
    frozen = request("DOC")
    frozen = replace(frozen, request_sha256=sha256_json(frozen.body), input_article_root=str(tmp_path))
    raw_dir = tmp_path / "raw_requests"
    raw_dir.mkdir()
    (raw_dir / "packet_0001.json").write_text(json.dumps({"body": frozen.body}), encoding="utf-8")
    index = write_frozen_request_index([frozen], tmp_path / "frozen_requests.json")
    files = write_batch_input_files([frozen], tmp_path)

    class Files:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(id="file_1")

    class Batches:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(id="batch_1", status="validating", output_file_id=None, error_file_id=None)

    client = SimpleNamespace(files=Files(), batches=Batches())
    submit_batch_files(files, tmp_path, client=client, frozen_index_path=index)
    submit_batch_files(files, tmp_path, client=client, frozen_index_path=index)
    assert client.files.calls == 1
    assert client.batches.calls == 1


def test_collection_persists_success_and_isolates_incomplete_response(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    frozen_requests = []
    for document_id in ("GOOD", "BAD"):
        source_root = tmp_path / "inputs" / document_id
        (source_root / "raw_requests").mkdir(parents=True)
        document_ir = source_root / "document_ir.json"
        write_json(document_ir, DocumentIR(document_id=document_id, materials=[]))
        frozen = request(document_id)
        frozen = replace(
            frozen,
            request_sha256=sha256_json(frozen.body),
            input_article_root=str(source_root),
            source_document_sha256=sha256_file(document_ir),
        )
        write_json(source_root / "raw_requests" / "packet_0001.json", {"body": frozen.body})
        frozen_requests.append(frozen)
    index = write_frozen_request_index(frozen_requests, workspace / "frozen_requests.json")
    input_files = write_batch_input_files(frozen_requests, workspace)
    success = ReportedExtraction(document_id="GOOD").model_dump_json()
    failure = ReportedExtraction(document_id="BAD").model_dump_json()
    output = "\n".join(
        [
            json.dumps({
                "custom_id": frozen_requests[0].custom_id,
                "response": {"status_code": 200, "body": {
                    "status": "completed", "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    "output": [{"content": [{"type": "output_text", "text": success}]}],
                }},
            }),
            json.dumps({
                "custom_id": frozen_requests[1].custom_id,
                "response": {"status_code": 200, "body": {
                    "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [{"content": [{"type": "output_text", "text": failure}]}],
                }},
            }),
        ]
    ).encode()
    write_json(
        workspace / "batch_state.json",
        {
            "submissions": [{
                "batch_id": "batch", "input_path": str(input_files[0]), "status": "completed",
                "output_file_id": "output", "error_file_id": None,
            }]
        },
    )

    class Files:
        def content(self, _file_id):
            return SimpleNamespace(content=output)

    class Batches:
        def retrieve(self, _batch_id):
            return SimpleNamespace(
                status="completed", output_file_id="output", error_file_id=None,
                request_counts=SimpleNamespace(model_dump=lambda **_kwargs: {"total": 2, "completed": 2, "failed": 0}),
                errors=None,
            )

    results = collect_batch_results(
        state_path=workspace / "batch_state.json",
        frozen_index_path=index,
        output_root=tmp_path / "results",
        client=SimpleNamespace(files=Files(), batches=Batches()),
    )
    assert [item.document_id for item in results] == ["GOOD"]
    summary = json.loads((workspace / "collection_summary.json").read_text())
    assert (summary["completed_count"], summary["failed_count"]) == (1, 1)
    assert len((workspace / "failure_only_retry.jsonl").read_text().splitlines()) == 1


def test_concurrent_submission_is_blocked_before_another_upload(tmp_path: Path) -> None:
    _frozen, index, files = prepared_workspace(tmp_path)

    class Files:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            with pytest.raises(RuntimeError, match="in use by another operation"):
                submit_batch_files(files, tmp_path, client=client, frozen_index_path=index)
            return SimpleNamespace(id="file_1")

    client = SimpleNamespace(
        files=Files(),
        batches=SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(id="batch_1", status="validating")),
    )
    submit_batch_files(files, tmp_path, client=client, frozen_index_path=index)
    assert client.files.calls == 1


def test_sdk_does_not_retry_ambiguous_batch_creation(tmp_path: Path) -> None:
    _frozen, index, files = prepared_workspace(tmp_path)
    calls: list[str] = []

    def handler(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming.url.path)
        if incoming.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file_1", "object": "file", "purpose": "batch"})
        return httpx.Response(500, json={"error": {"message": "Unknown creation outcome"}})

    with OpenAI(api_key="not-a-real-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        with pytest.raises(APIStatusError):
            submit_batch_files(files, tmp_path, client=client, frozen_index_path=index)
        with pytest.raises(RuntimeError, match="outcome is ambiguous"):
            submit_batch_files(files, tmp_path, client=client, frozen_index_path=index)
    assert calls == ["/v1/files", "/v1/batches"]
    assert json.loads((tmp_path / "batch_state.json").read_text())["submissions"][0]["status"] == "submission_unknown"


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_usd": float("nan")},
        {"max_usd": -1},
        {"input_usd_per_million": -1},
        {"output_usd_per_million": float("inf")},
        {"cache_write_multiplier": float("nan")},
        {"cached_input_multiplier": float("nan")},
    ],
)
def test_batch_rejects_invalid_budget_before_upload(tmp_path: Path, overrides: dict) -> None:
    _frozen, index, files = prepared_workspace(tmp_path)
    prices = {"max_usd": 10, "input_usd_per_million": 1, "output_usd_per_million": 1, **overrides}
    with pytest.raises(ValueError):
        submit_batch_files(files, tmp_path, client=SimpleNamespace(), frozen_index_path=index, **prices)
    assert not (tmp_path / "batch_state.json").exists()


def test_resuming_batch_enforces_a_new_budget_before_submission(tmp_path: Path) -> None:
    _frozen, index, files = prepared_workspace(tmp_path)
    write_json(tmp_path / "batch_state.json", {"submissions": [{
        "input_path": str(files[0]), "input_sha256": sha256_file(files[0]), "status": "planned",
    }]})
    with pytest.raises(RuntimeError, match="exceeds --max-usd"):
        submit_batch_files(
            files, tmp_path, client=SimpleNamespace(), frozen_index_path=index,
            max_usd=0.001, input_usd_per_million=1, output_usd_per_million=1,
        )


def test_local_collection_failure_is_recoverable_without_api_retry(tmp_path: Path, monkeypatch) -> None:
    import phase1_extraction.openai_batch as batch_module

    frozen, index, files = prepared_workspace(tmp_path)
    raw_body = {"status": "completed", "output": [{"content": [{
        "type": "output_text", "text": ReportedExtraction(document_id="DOC").model_dump_json(),
    }]}]}
    output = json.dumps({"custom_id": frozen.custom_id, "response": {"status_code": 200, "body": raw_body}}).encode()
    write_json(tmp_path / "batch_state.json", {"submissions": [{
        "batch_id": "batch_1", "input_path": str(files[0]), "status": "completed",
        "output_file_id": "output_1", "error_file_id": None,
    }]})
    # Leftover error files from another run must not poison this collection.
    (tmp_path / "errors_0002.jsonl").write_text("invalid stale JSON", encoding="utf-8")
    client = SimpleNamespace(
        files=SimpleNamespace(content=lambda _file_id: SimpleNamespace(content=output)),
        batches=SimpleNamespace(retrieve=lambda _batch_id: SimpleNamespace(
            status="completed", output_file_id="output_1", error_file_id=None,
        )),
    )
    original_persist = batch_module.persist_transport_result

    def fail_locally(**_kwargs):
        raise RuntimeError("Local audit temporarily unavailable")

    monkeypatch.setattr(batch_module, "persist_transport_result", fail_locally)
    arguments = dict(state_path=tmp_path / "batch_state.json", frozen_index_path=index,
                     output_root=tmp_path / "results", client=client)
    assert collect_batch_results(**arguments) == []
    summary = json.loads((tmp_path / "collection_summary.json").read_text())
    assert summary["outcomes"][0]["failure_stage"] == "postprocessing"
    assert summary["outcomes"][0]["retry_api_request"] is False
    assert (tmp_path / "failure_only_retry.jsonl").read_text() == ""
    assert json.loads((tmp_path / "results" / "DOC" / "raw_responses" / "packet_0001.json").read_text()) == raw_body
    monkeypatch.setattr(batch_module, "persist_transport_result", original_persist)
    assert len(collect_batch_results(**arguments)) == 1


@pytest.mark.parametrize("failure_stage", ["validation", "postprocessing"])
def test_frozen_sync_recovers_saved_response_without_repaying(tmp_path: Path, monkeypatch, failure_stage: str) -> None:
    import phase1_extraction.openai_batch as batch_module

    frozen, _index, _files = prepared_workspace(tmp_path)
    raw = {"status": "completed", "output": [{"content": [{
        "type": "output_text", "text": ReportedExtraction(document_id="DOC").model_dump_json(),
    }]}]}

    class Responses:
        calls = 0

        def create(self, **body):
            self.calls += 1
            assert body == frozen.body
            return raw

    client = SimpleNamespace(responses=Responses())
    method = "parse_model_response" if failure_stage == "validation" else "persist_transport_result"
    original = getattr(batch_module, method)

    def fail_once(*_args, **_kwargs):
        raise RuntimeError("Local processing failed")

    monkeypatch.setattr(batch_module, method, fail_once)
    with pytest.raises(RuntimeError, match="Local processing failed"):
        run_frozen_sync(frozen, output_root=tmp_path / "results", client=client)
    monkeypatch.setattr(batch_module, method, original)
    paths = run_frozen_sync(frozen, output_root=tmp_path / "results", client=client)
    assert paths.reported_extraction.is_file()
    assert client.responses.calls == 1
