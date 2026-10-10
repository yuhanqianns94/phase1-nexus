"""True OpenAI Batch transport for frozen, single-packet v3 requests.

The synchronous and Batch paths consume the exact same request ``body``.
Batch only changes transport and scheduling; it does not rebuild the prompt,
schema, images, or model parameters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from .artifacts import ArticlePaths, read_json, sha256_file, sha256_json, utc_now, write_json
from .llm import (
    DEFAULT_CACHED_INPUT_MULTIPLIER,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    build_request_body,
    parse_model_response,
    response_payload,
)
from .models import DocumentIR, ReportedExtraction
from .packets import packet_image_paths
from .pipeline import ingest, parse_sources, prepare_packets
from .quality import apply_audit_status, audit_reported_extraction
from .request_cache import artifact_lock, cache_response, load_cached_response
from .resolver import resolve_extraction


BATCH_ENDPOINT = "/v1/responses"
DEFAULT_MAX_BATCH_FILE_BYTES = 190 * 1024 * 1024
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _batch_state_lock(state_path: Path):
    return artifact_lock(Path(state_path).with_suffix(".lock"))


@dataclass(frozen=True)
class FrozenBatchRequest:
    document_id: str
    packet_id: str
    custom_id: str
    request_sha256: str
    body: dict[str, Any]
    input_article_root: str
    material_coverage_complete: bool
    source_document_sha256: str | None = None

    def jsonl_record(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": self.body,
        }


def freeze_single_packet_request(
    paths: ArticlePaths,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> FrozenBatchRequest:
    document, packets = prepare_packets(paths)
    if len(packets) != 1:
        raise ValueError(
            f"{paths.document_id} has {len(packets)} packets; v3.1 Batch accepts exactly one packet per article"
        )
    packet = packets[0]
    images = packet_image_paths(packet, document)
    missing = [
        item for item in packet.get("image_inventory", [])
        if item.get("required", True)
        and item.get("status") not in {"included", "covered_by_context_page"}
    ]
    if missing:
        raise ValueError(f"{paths.document_id} has missing visual request material")
    body = build_request_body(
        packet=packet,
        prior=None,
        image_paths=images,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    digest = sha256_json(body)
    custom_id = f"v3_1-{paths.document_id}-{packet['packet_id']}-{digest[:16]}"
    frozen = FrozenBatchRequest(
        document_id=paths.document_id,
        packet_id=packet["packet_id"],
        custom_id=custom_id,
        request_sha256=digest,
        body=body,
        input_article_root=str(paths.root.resolve()),
        material_coverage_complete=bool(read_json(paths.extraction_packets).get("material_coverage_complete")),
        source_document_sha256=sha256_file(paths.document_ir),
    )
    write_json(
        paths.raw_requests / f"{packet['packet_id']}.json",
        {
            "format": "phase1_frozen_responses_request_v3.1",
            "created_at_utc": utc_now(),
            "custom_id": custom_id,
            "request_sha256": digest,
            "image_inventory": packet.get("image_inventory", []),
            "body": body,
        },
    )
    return frozen


def write_batch_input_files(
    requests: Iterable[FrozenBatchRequest],
    workspace: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_BATCH_FILE_BYTES,
) -> list[Path]:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    current: list[bytes] = []
    current_size = 0
    for request in requests:
        line = (json.dumps(request.jsonl_record(), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(line) > max_file_bytes:
            raise ValueError(f"One frozen request exceeds the Batch input-file limit: {request.custom_id}")
        if current and current_size + len(line) > max_file_bytes:
            path = workspace / f"requests_{len(paths) + 1:04d}.jsonl"
            path.write_bytes(b"".join(current))
            paths.append(path)
            current, current_size = [], 0
        current.append(line)
        current_size += len(line)
    if current:
        path = workspace / f"requests_{len(paths) + 1:04d}.jsonl"
        path.write_bytes(b"".join(current))
        paths.append(path)
    if not paths:
        raise ValueError("No Batch requests were supplied")
    return paths


def write_frozen_request_index(requests: Iterable[FrozenBatchRequest], path: Path) -> Path:
    values = list(requests)
    write_json(
        path,
        {
            "format": "phase1_frozen_request_index_v3.1",
            "created_at_utc": utc_now(),
            "requests": [
                {
                    "document_id": item.document_id,
                    "packet_id": item.packet_id,
                    "custom_id": item.custom_id,
                    "request_sha256": item.request_sha256,
                    "input_article_root": item.input_article_root,
                    "material_coverage_complete": item.material_coverage_complete,
                    "source_document_sha256": item.source_document_sha256,
                    "frozen_request_path": str(
                        (Path(item.input_article_root) / "raw_requests" / f"{item.packet_id}.json").resolve()
                    ),
                }
                for item in values
            ],
        },
    )
    return Path(path)


def load_frozen_request_index(path: Path) -> list[FrozenBatchRequest]:
    payload = read_json(path)
    requests: list[FrozenBatchRequest] = []
    for item in payload.get("requests", []):
        frozen_record = read_json(Path(item["frozen_request_path"]))
        body = frozen_record["body"]
        if sha256_json(body) != item["request_sha256"]:
            raise ValueError(f"Frozen request changed for {item['custom_id']}")
        source_document_path = Path(item["input_article_root"]) / "document_ir.json"
        if item.get("source_document_sha256") and (
            not source_document_path.is_file()
            or sha256_file(source_document_path) != item["source_document_sha256"]
        ):
            raise ValueError(f"Frozen DocumentIR changed for {item['custom_id']}")
        requests.append(
            FrozenBatchRequest(
                document_id=item["document_id"],
                packet_id=item["packet_id"],
                custom_id=item["custom_id"],
                request_sha256=item["request_sha256"],
                body=body,
                input_article_root=item["input_article_root"],
                material_coverage_complete=bool(item["material_coverage_complete"]),
                source_document_sha256=item.get("source_document_sha256"),
            )
        )
    return requests


def validate_batch_input_files(
    input_files: Iterable[Path],
    frozen_index_path: Path,
) -> dict[str, Any]:
    """Prove that upload files exactly match the current frozen request set."""

    requests = load_frozen_request_index(frozen_index_path)
    expected = {item.custom_id: item for item in requests}
    if len(expected) != len(requests):
        raise ValueError("Frozen request index contains duplicate custom_id values")
    seen: dict[str, tuple[Path, int]] = {}
    file_records: list[dict[str, Any]] = []
    estimated_input_tokens = 0
    reserved_output_tokens = 0
    for raw_path in input_files:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        line_count = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            line_count += 1
            record = json.loads(line)
            custom_id = str(record.get("custom_id") or "")
            if custom_id in seen:
                raise ValueError(f"Duplicate custom_id {custom_id!r} in Batch inputs")
            if custom_id not in expected:
                raise ValueError(f"Stale or unknown custom_id {custom_id!r} in {path}")
            if record.get("method") != "POST" or record.get("url") != BATCH_ENDPOINT:
                raise ValueError(f"Invalid endpoint record for {custom_id}")
            body = record.get("body")
            if not isinstance(body, dict) or sha256_json(body) != expected[custom_id].request_sha256:
                raise ValueError(f"Batch body changed after freezing for {custom_id}")
            seen[custom_id] = (path, line_number)
            text_chars = len(str(body.get("instructions") or ""))
            text_chars += len(json.dumps((body.get("text") or {}).get("format") or {}, separators=(",", ":")))
            image_count = 0
            for message in body.get("input", []):
                for content in message.get("content", []):
                    if content.get("type") == "input_text":
                        text_chars += len(str(content.get("text") or ""))
                    elif content.get("type") == "input_image":
                        image_count += 1
            estimated_input_tokens += (text_chars + 3) // 4 + image_count * 5_000
            reserved_output_tokens += int(body.get("max_output_tokens") or 0)
        file_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
                "request_count": line_count,
            }
        )
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ValueError("Frozen requests are missing from Batch input files: " + ", ".join(missing))
    return {
        "request_count": len(seen),
        "file_records": file_records,
        "estimated_input_tokens": estimated_input_tokens,
        "reserved_output_tokens": reserved_output_tokens,
    }


def submit_batch_files(
    input_files: Iterable[Path],
    workspace: Path,
    *,
    client: OpenAI | None = None,
    frozen_index_path: Path | None = None,
    max_usd: float | None = None,
    input_usd_per_million: float | None = None,
    output_usd_per_million: float | None = None,
    cache_write_multiplier: float = 1.25,
    cached_input_multiplier: float = DEFAULT_CACHED_INPUT_MULTIPLIER,
) -> Path:
    with _batch_state_lock(Path(workspace) / "batch_state.json"):
        return _submit_batch_files(
            input_files, workspace, client=client, frozen_index_path=frozen_index_path,
            max_usd=max_usd, input_usd_per_million=input_usd_per_million,
            output_usd_per_million=output_usd_per_million,
            cache_write_multiplier=cache_write_multiplier,
            cached_input_multiplier=cached_input_multiplier,
        )


def _submit_batch_files(
    input_files: Iterable[Path],
    workspace: Path,
    *,
    client: OpenAI | None,
    frozen_index_path: Path | None,
    max_usd: float | None,
    input_usd_per_million: float | None,
    output_usd_per_million: float | None,
    cache_write_multiplier: float,
    cached_input_multiplier: float,
) -> Path:
    workspace = Path(workspace)
    input_files = [Path(path).resolve() for path in input_files]
    frozen_index_path = Path(frozen_index_path or workspace / "frozen_requests.json")
    validation = validate_batch_input_files(input_files, frozen_index_path)
    state_path = workspace / "batch_state.json"
    has_existing_state = state_path.is_file()
    has_budget = any(value is not None for value in (max_usd, input_usd_per_million, output_usd_per_million))
    if has_budget:
        if None in (max_usd, input_usd_per_million, output_usd_per_million):
            raise ValueError("Batch budget check requires max_usd and both token prices")
        if any(not math.isfinite(float(value)) for value in (max_usd, input_usd_per_million, output_usd_per_million)):
            raise ValueError("Batch budget and prices must be finite")
        if float(max_usd) <= 0 or min(float(input_usd_per_million), float(output_usd_per_million)) < 0:
            raise ValueError("Batch max_usd must be positive and token prices must be non-negative")
        if not math.isfinite(cache_write_multiplier) or cache_write_multiplier < 1:
            raise ValueError("cache_write_multiplier must be at least 1")
        if not math.isfinite(cached_input_multiplier) or not 0 <= cached_input_multiplier <= 1:
            raise ValueError("cached_input_multiplier must be between 0 and 1")
        # Assume every estimated input token is billed as a cache write. This
        # is deliberately stricter than the eventual mix of cached/write
        # tokens and reflects the documented cache-write premium.
        estimated_usd = (
            validation["estimated_input_tokens"] * float(input_usd_per_million) * cache_write_multiplier
            + validation["reserved_output_tokens"] * float(output_usd_per_million)
        ) / 1_000_000
        validation["conservative_reserved_usd"] = estimated_usd
        validation["max_usd"] = float(max_usd)
        validation["cache_write_multiplier"] = cache_write_multiplier
        validation["pricing"] = {
            "input_usd_per_million": float(input_usd_per_million),
            "output_usd_per_million": float(output_usd_per_million),
            "cache_write_multiplier": cache_write_multiplier,
            "cached_input_multiplier": cached_input_multiplier,
            "recorded_at_utc": utc_now(),
            "source": "caller-supplied current Batch prices",
        }
        if estimated_usd > float(max_usd):
            raise RuntimeError(
                f"Conservative Batch reserve ${estimated_usd:.4f} exceeds --max-usd ${float(max_usd):.4f}"
            )
    if has_existing_state:
        state = read_json(state_path)
        prior_files = {item["input_path"]: item for item in state.get("submissions", [])}
        if set(prior_files) != {str(path) for path in input_files}:
            raise RuntimeError("Existing Batch state belongs to a different set of input files")
        if has_budget:
            state["validation"] = validation
    else:
        state = {
            "format": "phase1_openai_batch_state_v3.1",
            "endpoint": BATCH_ENDPOINT,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "validation": validation,
            "submissions": [
                {
                    "input_path": record["path"],
                    "input_sha256": record["sha256"],
                    "input_byte_count": record["byte_count"],
                    "request_count": record["request_count"],
                    "status": "planned",
                }
                for record in validation["file_records"]
            ],
        }
        write_json(state_path, state)

    for item in state["submissions"]:
        path = Path(item["input_path"])
        if sha256_file(path) != item["input_sha256"]:
            raise RuntimeError(f"Batch input file changed after state creation: {path}")
    if has_existing_state and has_budget:
        state["updated_at_utc"] = utc_now()
        write_json(state_path, state)
    # SDK retries can repeat a successful remote create after a lost response.
    # Keep reads/uploads retryable, but leave ambiguous creation for inspection.
    client = client or OpenAI()
    creation_client = client.with_options(max_retries=0) if hasattr(client, "with_options") else client
    for item in state["submissions"]:
        path = Path(item["input_path"])
        if item.get("batch_id"):
            continue
        if item.get("status") in {"creating", "submission_unknown"}:
            raise RuntimeError(
                f"Prior Batch creation outcome is ambiguous for {path}; inspect the account before resubmitting"
            )
        if not item.get("input_file_id"):
            try:
                with path.open("rb") as stream:
                    uploaded = client.files.create(file=stream, purpose="batch")
            except Exception as error:
                item.update(status="upload_failed", error=f"{type(error).__name__}: {error}", last_attempt_at_utc=utc_now())
                state["updated_at_utc"] = utc_now()
                write_json(state_path, state)
                raise
            item.update(input_file_id=uploaded.id, status="uploaded", uploaded_at_utc=utc_now())
            state["updated_at_utc"] = utc_now()
            write_json(state_path, state)
        item.update(status="creating", create_attempt_started_at_utc=utc_now())
        state["updated_at_utc"] = utc_now()
        write_json(state_path, state)
        try:
            batch = creation_client.batches.create(
                input_file_id=item["input_file_id"],
                endpoint=BATCH_ENDPOINT,
                completion_window="24h",
                metadata={"pipeline": "phase1_extraction_v3.1"},
            )
        except Exception as error:
            item.update(status="submission_unknown", error=f"{type(error).__name__}: {error}")
            state["updated_at_utc"] = utc_now()
            write_json(state_path, state)
            raise
        item.update(
            batch_id=batch.id,
            status=batch.status,
            submitted_at_utc=utc_now(),
            output_file_id=getattr(batch, "output_file_id", None),
            error_file_id=getattr(batch, "error_file_id", None),
            error=None,
        )
        state["updated_at_utc"] = utc_now()
        write_json(state_path, state)
    return state_path


def refresh_batch_state(state_path: Path, *, client: OpenAI | None = None) -> dict[str, Any]:
    with _batch_state_lock(state_path):
        return _refresh_batch_state(state_path, client=client)


def _refresh_batch_state(state_path: Path, *, client: OpenAI | None = None) -> dict[str, Any]:
    client = client or OpenAI()
    state = read_json(state_path)
    for item in state["submissions"]:
        if not item.get("batch_id"):
            continue
        batch = client.batches.retrieve(item["batch_id"])
        item.update(
            {
                "status": batch.status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "request_counts": response_payload(getattr(batch, "request_counts", None)),
                "errors": response_payload(getattr(batch, "errors", None)),
                "last_checked_at_utc": utc_now(),
            }
        )
    state["updated_at_utc"] = utc_now()
    write_json(state_path, state)
    return state


def download_batch_results(state_path: Path, *, client: OpenAI | None = None) -> list[Path]:
    client = client or OpenAI()
    state = refresh_batch_state(state_path, client=client)
    workspace = Path(state_path).parent
    downloaded: list[Path] = []
    for index, item in enumerate(state["submissions"], start=1):
        if item.get("output_file_id"):
            content = client.files.content(item["output_file_id"])
            payload = content.content if hasattr(content, "content") else bytes(content)
            path = workspace / f"responses_{index:04d}.jsonl"
            path.write_bytes(payload)
            downloaded.append(path)
        if item.get("error_file_id"):
            error_content = client.files.content(item["error_file_id"])
            error_payload = error_content.content if hasattr(error_content, "content") else bytes(error_content)
            (workspace / f"errors_{index:04d}.jsonl").write_bytes(error_payload)
    return downloaded


def read_batch_output(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    by_custom_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = str(item.get("custom_id") or "")
            if not custom_id or custom_id in by_custom_id:
                raise ValueError(f"Invalid or duplicate custom_id in {path}:{line_number}")
            by_custom_id[custom_id] = item
    return by_custom_id


def read_batch_errors(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    by_custom_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = str(item.get("custom_id") or "")
            if not custom_id or custom_id in by_custom_id:
                raise ValueError(f"Invalid or duplicate custom_id in {path}:{line_number}")
            by_custom_id[custom_id] = item
    return by_custom_id


def extract_batch_response(item: dict[str, Any], request: FrozenBatchRequest) -> tuple[ReportedExtraction, dict[str, Any]]:
    response = item.get("response") or {}
    if int(response.get("status_code") or 0) != 200:
        raise RuntimeError(f"Batch request failed for {request.custom_id}: {item.get('error') or response}")
    body = response.get("body")
    if not isinstance(body, dict):
        raise RuntimeError(f"Batch response body is missing for {request.custom_id}")
    parsed = parse_model_response(body, document_id=request.document_id)
    return parsed, body


def persist_transport_result(
    *,
    output_root: Path,
    request: FrozenBatchRequest,
    parsed: ReportedExtraction,
    raw_response: dict[str, Any],
    transport: str,
    run_quality_audit: bool = True,
) -> ArticlePaths:
    paths = ArticlePaths(Path(output_root), request.document_id)
    paths.ensure()
    parsed.material_coverage_complete = request.material_coverage_complete
    if not parsed.material_coverage_complete:
        parsed.coverage_notes.append("Programmatic material coverage is incomplete; R2 zero imputation is disabled.")
    source_document_path = Path(request.input_article_root) / "document_ir.json"
    if request.source_document_sha256 and (
        not source_document_path.is_file()
        or sha256_file(source_document_path) != request.source_document_sha256
    ):
        raise RuntimeError(
            f"Frozen DocumentIR changed before postprocessing for {request.custom_id}"
        )
    source_document = (
        DocumentIR.model_validate(read_json(source_document_path))
        if source_document_path.is_file()
        else None
    )
    if source_document is not None:
        # Keep the exact evidence index beside Batch outputs so the review app
        # does not depend on a separately configured preprocessing directory.
        write_json(paths.document_ir, source_document)
    from .drug_ordering import apply_drug_ordering
    parsed = apply_drug_ordering(parsed)
    resolved = resolve_extraction(parsed)
    if run_quality_audit:
        audit = audit_reported_extraction(parsed, document=source_document, resolved=resolved)
        apply_audit_status(parsed, audit)
        apply_audit_status(resolved.reported_extraction, audit)
    else:
        audit = {"status": "not_run", "reason": "Quality audit skipped by explicit run configuration."}
    write_json(paths.extraction_audit, audit)
    write_json(paths.reported_extraction, parsed)
    write_json(paths.raw_responses / f"{request.packet_id}.json", raw_response)
    write_json(
        paths.request_summary,
        {
            "status": "completed",
            "transport": transport,
            "quality_audit_performed": run_quality_audit,
            "custom_id": request.custom_id,
            "request_sha256": request.request_sha256,
            "input_article_root": request.input_article_root,
            "source_document_sha256": request.source_document_sha256,
            "response_id": raw_response.get("id"),
            "usage": raw_response.get("usage"),
        },
    )
    write_json(paths.resolved_extraction, resolved)
    write_json(
        paths.curve_rows,
        {
            "policy_version": resolved.policy_version,
            "project_default": [row.model_dump(mode="json") for row in resolved.curve_rows if row.included_in_default],
            "reported_only": [row.model_dump(mode="json") for row in resolved.curve_rows if row.included_in_reported_only],
            "all_rows": [row.model_dump(mode="json") for row in resolved.curve_rows],
            "model_outcome_rows": [row.model_dump(mode="json") for row in resolved.curve_rows if row.outcome_usable],
        },
    )
    write_json(paths.rule_applications, [item.model_dump(mode="json") for item in resolved.rule_applications])
    write_json(paths.issues, [item.model_dump(mode="json") for item in resolved.issues])
    return paths


def run_frozen_sync(
    request: FrozenBatchRequest,
    *,
    output_root: Path,
    client: OpenAI | None = None,
) -> ArticlePaths:
    paths = ArticlePaths(Path(output_root), request.document_id)
    with artifact_lock(paths.root / ".extraction.lock"):
        return _run_frozen_sync(request, output_root=output_root, client=client)


def _run_frozen_sync(
    request: FrozenBatchRequest,
    *,
    output_root: Path,
    client: OpenAI | None,
) -> ArticlePaths:
    paths = ArticlePaths(Path(output_root), request.document_id)
    raw_path = paths.raw_responses / f"{request.packet_id}.json"
    raw = load_cached_response(raw_path, request.request_sha256)
    if raw is None:
        client = client or OpenAI()
        response = client.responses.create(**request.body)
        raw = response_payload(response)
        # Bind the response before local validation or postprocessing can fail.
        paths.ensure()
        cache_response(raw_path, request.request_sha256, raw)
    parsed = parse_model_response(raw, document_id=request.document_id)
    return persist_transport_result(
        output_root=output_root,
        request=request,
        parsed=parsed,
        raw_response=raw,
        transport="synchronous",
    )


def prepare_batch_manifest(
    *,
    manifest_path: Path,
    input_root: Path,
    workspace: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> tuple[list[FrozenBatchRequest], Path]:
    """Locally preprocess articles and freeze eligible one-packet requests."""

    from .batch import load_batch_manifest

    frozen: list[FrozenBatchRequest] = []
    outcomes: list[dict[str, Any]] = []
    synchronous_fallback: list[dict[str, Any]] = []
    for entry in load_batch_manifest(manifest_path):
        paths = ingest(
            document_id=entry["document_id"],
            source_paths=entry["sources"],
            output_root=input_root,
        )
        parse_sources(paths)
        _document, packets = prepare_packets(paths)
        if len(packets) != 1:
            synchronous_fallback.append(
                {
                    "document_id": entry["document_id"],
                    "sources": [str(path.resolve()) for path in entry["sources"]],
                }
            )
            outcomes.append(
                {
                    "document_id": entry["document_id"],
                    "packet_count": len(packets),
                    "route": "synchronous",
                    "reason": "Multi-packet article",
                }
            )
            continue
        request = freeze_single_packet_request(
            paths,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
        frozen.append(request)
        outcomes.append(
            {
                "document_id": entry["document_id"],
                "packet_count": 1,
                "route": "openai_batch",
                "custom_id": request.custom_id,
                "request_sha256": request.request_sha256,
                "page_count": len(_document.pages),
                "table_count": len(_document.tables),
                "structured_table_count": sum(bool(item.cells) for item in _document.tables),
                "visual_only_table_count": sum(item.parse_status == "visual_only" for item in _document.tables),
                "request_image_count": sum(
                    item.get("status") == "included"
                    for item in packets[0].get("image_inventory", [])
                ),
                "material_coverage_complete": request.material_coverage_complete,
                "not_covered_source_ids": read_json(paths.extraction_packets).get("not_covered_source_ids", []),
            }
        )
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    index_path = write_frozen_request_index(frozen, workspace / "frozen_requests.json")
    files = write_batch_input_files(frozen, workspace) if frozen else []
    validation = validate_batch_input_files(files, index_path) if files else {
        "request_count": 0,
        "file_records": [],
        "estimated_input_tokens": 0,
        "reserved_output_tokens": 0,
    }
    fallback_path = workspace / "synchronous_fallback.jsonl"
    fallback_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in synchronous_fallback),
        encoding="utf-8",
    )
    write_json(
        workspace / "routing_plan.json",
        {
            "format": "phase1_batch_routing_plan_v3.1",
            "created_at_utc": utc_now(),
            "manifest_path": str(Path(manifest_path).resolve()),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "outcomes": outcomes,
            "batch_input_files": [str(path.resolve()) for path in files],
            "batch_input_file_records": validation["file_records"],
            "batch_request_count": validation["request_count"],
            "estimated_input_tokens": validation["estimated_input_tokens"],
            "reserved_output_tokens": validation["reserved_output_tokens"],
            "synchronous_fallback_manifest": str(fallback_path.resolve()),
            "synchronous_fallback_count": len(synchronous_fallback),
        },
    )
    return frozen, index_path


def collect_batch_results(
    *,
    state_path: Path,
    frozen_index_path: Path,
    output_root: Path,
    client: OpenAI | None = None,
    run_quality_audit: bool = True,
) -> list[ArticlePaths]:
    client = client or OpenAI()
    requests = load_frozen_request_index(frozen_index_path)
    by_id = {item.custom_id: item for item in requests}
    output_files = download_batch_results(state_path, client=client)
    results = read_batch_output(output_files)
    state = read_json(state_path)
    workspace = Path(state_path).parent
    # A previous collection can leave files behind. Only consume error files
    # attached to this state's current submissions.
    errors = read_batch_errors(
        workspace / f"errors_{index:04d}.jsonl"
        for index, submission in enumerate(state.get("submissions", []), start=1)
        if submission.get("error_file_id")
    )
    status_by_custom_id: dict[str, str] = {}
    for submission in state.get("submissions", []):
        input_path = Path(submission["input_path"])
        if not input_path.is_file():
            continue
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                status_by_custom_id[str(json.loads(line).get("custom_id") or "")] = str(submission.get("status") or "unknown")
    paths: list[ArticlePaths] = []
    outcomes: list[dict[str, Any]] = []
    retry_records: list[dict[str, Any]] = []
    usage_totals = {
        "input_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    for custom_id, request in by_id.items():
        item = results.get(custom_id)
        if item is None:
            batch_status = status_by_custom_id.get(custom_id, "unknown")
            is_pending = batch_status not in TERMINAL_BATCH_STATUSES
            failure = errors.get(custom_id) or {
                "error": "Output is not available" if is_pending else "Terminal Batch output omitted this request"
            }
            outcomes.append(
                {
                    "document_id": request.document_id,
                    "custom_id": custom_id,
                    "status": "pending" if is_pending else "failed",
                    "batch_status": batch_status,
                    "error": failure,
                }
            )
            if not is_pending:
                retry_records.append(request.jsonl_record())
            continue
        raw_body = ((item.get("response") or {}).get("body"))
        if isinstance(raw_body, dict):
            article_paths = ArticlePaths(Path(output_root), request.document_id)
            article_paths.ensure()
            write_json(article_paths.raw_responses / f"{request.packet_id}.json", raw_body)
            usage = raw_body.get("usage") or {}
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                usage_totals[field] += int(usage.get(field) or 0)
            input_details = usage.get("input_tokens_details") or {}
            output_details = usage.get("output_tokens_details") or {}
            usage_totals["cache_write_tokens"] += int(input_details.get("cache_write_tokens") or 0)
            usage_totals["cached_tokens"] += int(input_details.get("cached_tokens") or 0)
            usage_totals["reasoning_tokens"] += int(output_details.get("reasoning_tokens") or 0)
        response_validated = False
        try:
            parsed, raw = extract_batch_response(item, request)
            response_validated = True
            result_path = persist_transport_result(
                output_root=output_root,
                request=request,
                parsed=parsed,
                raw_response=raw,
                transport="openai_batch",
                run_quality_audit=run_quality_audit,
            )
            paths.append(result_path)
            outcomes.append(
                {
                    "document_id": request.document_id,
                    "custom_id": custom_id,
                    "status": "completed",
                    "output_root": str(result_path.root.resolve()),
                }
            )
        except Exception as error:
            article_paths = ArticlePaths(Path(output_root), request.document_id)
            article_paths.ensure()
            failure_payload = {
                "status": "failed",
                "transport": "openai_batch",
                "custom_id": custom_id,
                "request_sha256": request.request_sha256,
                "error_type": type(error).__name__,
                "message": str(error),
                "failure_stage": "postprocessing" if response_validated else "response_validation",
                "retry_api_request": not response_validated,
                "recovery_action": "rerun_collection" if response_validated else "inspect_response_before_retry",
            }
            write_json(article_paths.request_summary, failure_payload)
            write_json(article_paths.root / "batch_failure.json", {"service_item": item, **failure_payload})
            outcomes.append({"document_id": request.document_id, **failure_payload})
            # A valid paid response survives local resolver/audit/persistence
            # failures. Recollecting can recover it without another API call.
            if not response_validated:
                retry_records.append(request.jsonl_record())
    retry_path = workspace / "failure_only_retry.jsonl"
    retry_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in retry_records),
        encoding="utf-8",
    )
    pricing = (state.get("validation") or {}).get("pricing")
    usage_cost_estimate: dict[str, Any] | None = None
    if pricing:
        uncached_tokens = max(
            0,
            usage_totals["input_tokens"]
            - usage_totals["cache_write_tokens"]
            - usage_totals["cached_tokens"],
        )
        input_rate = float(pricing["input_usd_per_million"])
        output_rate = float(pricing["output_usd_per_million"])
        estimated_cost = (
            uncached_tokens * input_rate
            + usage_totals["cache_write_tokens"] * input_rate * float(pricing["cache_write_multiplier"])
            + usage_totals["cached_tokens"] * input_rate * float(pricing["cached_input_multiplier"])
            + usage_totals["output_tokens"] * output_rate
        ) / 1_000_000
        usage_cost_estimate = {
            "usd": estimated_cost,
            "uncached_input_tokens": uncached_tokens,
            "pricing": pricing,
            "note": "Usage-based estimate; the platform billing ledger is authoritative.",
        }
    write_json(
        workspace / "collection_summary.json",
        {
            "format": "phase1_batch_collection_v3.1",
            "collected_at_utc": utc_now(),
            "outcomes": outcomes,
            "completed_count": sum(item["status"] == "completed" for item in outcomes),
            "failed_count": sum(item["status"] == "failed" for item in outcomes),
            "pending_count": sum(item["status"] == "pending" for item in outcomes),
            "failure_only_retry_path": str(retry_path.resolve()),
            "usage": usage_totals,
            "usage_cost_estimate": usage_cost_estimate,
        },
    )
    return paths
