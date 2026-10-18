"""Structured model transport and deterministic multi-packet merging."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from openai.lib._parsing._responses import type_to_text_format_param

from .artifacts import read_json, sha256_json, utc_now, write_json
from .models import ReportedExtraction, UnresolvedItem, SCHEMA_VERSION
from .packets import entity_index, packet_image_inventory, packet_image_paths
from .exposures import exposure_identity
from .request_cache import artifact_lock, cache_response, load_cached_response


DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 48_000
DEFAULT_CACHED_INPUT_MULTIPLIER = 0.05


class BudgetGuard:
    """Conservative request admission using caller-supplied current prices."""

    def __init__(
        self, *, max_usd: float, input_usd_per_million: float,
        output_usd_per_million: float, cache_write_multiplier: float = 1.25,
        cached_input_multiplier: float = DEFAULT_CACHED_INPUT_MULTIPLIER,
    ) -> None:
        values = (max_usd, input_usd_per_million, output_usd_per_million,
                  cache_write_multiplier, cached_input_multiplier)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Budget and prices must be finite")
        if max_usd <= 0 or input_usd_per_million < 0 or output_usd_per_million < 0:
            raise ValueError("Budget and prices must be non-negative, with max_usd positive")
        if cache_write_multiplier < 1 or not 0 <= cached_input_multiplier <= 1:
            raise ValueError("Cache write multiplier must be >=1 and cached input multiplier in [0, 1]")
        self.max_usd = max_usd
        self.input_rate = input_usd_per_million
        self.output_rate = output_usd_per_million
        self.cache_write_multiplier = cache_write_multiplier
        self.cached_input_multiplier = cached_input_multiplier
        self.spent_usd = 0.0

    def admit(self, *, packet_chars: int, image_count: int, max_output_tokens: int) -> None:
        if packet_chars < 0 or image_count < 0 or max_output_tokens <= 0:
            raise ValueError("Input sizes must be non-negative and max_output_tokens positive")
        estimated_input_tokens = (packet_chars + 3) // 4 + image_count * 5_000
        reserve = (
            estimated_input_tokens * self.input_rate * self.cache_write_multiplier
            + max_output_tokens * self.output_rate
        ) / 1_000_000
        if self.spent_usd + reserve > self.max_usd:
            raise RuntimeError(
                f"Conservative request reserve ${reserve:.4f} would exceed remaining budget "
                f"${self.max_usd - self.spent_usd:.4f}"
            )

    def admit_request(self, body: dict[str, Any]) -> None:
        # Count instructions, strict schema and prior entity IDs as well as
        # source text. Base64 image bytes are not text tokens.
        text_chars = len(body.get("instructions") or "")
        text_chars += len(json.dumps(body.get("text") or {}, ensure_ascii=False, separators=(",", ":")))
        image_count = 0
        for message in body.get("input", []):
            for content in message.get("content", []):
                if content.get("type") == "input_text":
                    text_chars += len(content.get("text") or "")
                elif content.get("type") == "input_image":
                    image_count += 1
        self.admit(packet_chars=text_chars, image_count=image_count,
                   max_output_tokens=body["max_output_tokens"])

    def record(self, usage: Any) -> None:
        def value(obj: Any, key: str) -> Any:
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        if usage is None or value(usage, "input_tokens") is None or value(usage, "output_tokens") is None:
            # Unknown usage must not silently grant another full budget.
            self.spent_usd = self.max_usd
            return
        input_tokens = max(0, int(value(usage, "input_tokens") or 0))
        output_tokens = max(0, int(value(usage, "output_tokens") or 0))
        details = value(usage, "input_tokens_details")
        cached = min(input_tokens, max(0, int(value(details, "cached_tokens") or 0)))
        written_value = value(details, "cache_write_tokens")
        # Older SDKs/providers can omit cache writes; reserve the premium for
        # all remaining input rather than claim an unobserved discount.
        written = (input_tokens - cached if written_value is None else
                   min(input_tokens - cached, max(0, int(written_value))))
        ordinary = input_tokens - cached - written
        self.spent_usd += (
            (ordinary + cached * self.cached_input_multiplier + written * self.cache_write_multiplier)
            * self.input_rate + output_tokens * self.output_rate
        ) / 1_000_000


def load_prompt() -> str:
    from .controlled_vocabulary import CATALOG
    prompt = Path(__file__).with_name("prompt_v3_en.txt").read_text(encoding="utf-8")
    choices = {name: list(spec['options']) for name, spec in CATALOG.items()}
    return prompt + "\nSHARED CLOSED CHOICE CATALOG (codes, not free text):\n" + json.dumps(choices, ensure_ascii=False, indent=2)


def _image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _content(packet: dict[str, Any], prior: ReportedExtraction | None, images: list[Path]) -> list[dict[str, Any]]:
    payload = dict(packet)
    payload["prior_entity_index"] = entity_index(prior)
    payload["packet_instruction"] = (
        "Extract all facts supported by this packet. Reuse prior IDs for recurring entities. "
        "Set extraction_complete=true only on the final packet and only if coverage is complete."
    )
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
    ]
    for path in images:
        content.append({"type": "input_text", "text": f"Source image: {path.name}"})
        content.append({"type": "input_image", "image_url": _image_data_url(path), "detail": "original"})
    return content


def build_request_body(
    *,
    packet: dict[str, Any],
    prior: ReportedExtraction | None,
    image_paths: list[Path],
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build the one canonical Responses request used by sync and Batch."""

    output_format = type_to_text_format_param(ReportedExtraction)
    # Legacy versions remain readable locally, while every new extraction has
    # one unambiguous schema version in both synchronous and Batch requests.
    output_format['schema']['properties']['schema_version']['enum'] = [SCHEMA_VERSION]
    body = {
        "model": model,
        "instructions": load_prompt(),
        "input": [{"role": "user", "content": _content(packet, prior, image_paths)}],
        "text": {"format": output_format},
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def response_payload(response: Any) -> Any:
    return response.model_dump(mode="json") if hasattr(response, "model_dump") else response


def response_output_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return str(direct)
    payload = response_payload(response)
    texts: list[str] = []
    for output in (payload or {}).get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(str(content["text"]))
    if not texts:
        raise RuntimeError("Model response did not contain structured output text")
    return "".join(texts)


def validate_response_completion(response: Any) -> None:
    """Reject service-level incomplete, failed, or refused responses.

    Structured JSON may still be present after an output-token truncation. It
    must not enter resolver rules as though the article had been read fully.
    """

    payload = response_payload(response) or {}
    if not isinstance(payload, dict):
        raise RuntimeError("Model response payload is not an object")
    status = payload.get("status")
    if status is not None and status != "completed":
        details = payload.get("incomplete_details") or payload.get("error") or {}
        raise RuntimeError(f"Model response status is {status!r}: {details}")
    for output in payload.get("output", []):
        if output.get("type") == "refusal":
            raise RuntimeError(f"Model response was refused: {output.get('refusal') or output}")
        for content in output.get("content", []):
            if content.get("type") == "refusal":
                raise RuntimeError(f"Model response was refused: {content.get('refusal') or content}")


def parse_model_response(response: Any, *, document_id: str) -> ReportedExtraction:
    validate_response_completion(response)
    parsed = ReportedExtraction.model_validate_json(response_output_text(response))
    if parsed.document_id != document_id:
        raise RuntimeError("Model returned a different document_id")
    from .field_normalization import normalize_fields
    return normalize_fields(parsed)[0]


def call_model(
    *,
    packet: dict[str, Any],
    prior: ReportedExtraction | None,
    image_paths: list[Path],
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    client: OpenAI | None = None,
) -> tuple[ReportedExtraction, Any]:
    if client is None and os.getenv("PHASE1_DEPLOYMENT", "").strip().lower() == "public":
        raise ValueError("Public extraction requires an explicit user API client")
    client = client if client is not None else OpenAI()
    body = build_request_body(
        packet=packet,
        prior=prior,
        image_paths=image_paths,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    response = client.responses.create(**body)
    parsed = parse_model_response(response, document_id=packet["document_id"])
    return parsed, response


_COLLECTION_ID_FIELDS = {
    "studies": "study_id",
    "study_parts": "part_id",
    "arms": "arm_id",
    "populations": "population_id",
    "interventions": "intervention_id",
    "regimens": "regimen_id",
    "designs": "design_id",
    "dlt_windows": "window_id",
    "dlt_definitions": "definition_id",
    "dose_series": "series_id",
    "cohorts": "cohort_id",
    "analysis_populations": "analysis_population_id",
    "population_counts": "count_id",
    "result_scopes": "result_scope_id",
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _semantic_key(field: str, item: dict[str, Any]) -> tuple[Any, ...] | None:
    if field == "studies":
        return (tuple(sorted(_norm(x) for x in item.get("registration_ids", []))), _norm(item.get("label")))
    if field == "study_parts":
        return (_norm(item.get("study_id")), _norm(item.get("label")), item.get("sequence_index"))
    if field == "arms":
        return (_norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("label")))
    if field == "populations":
        return (_norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("disease_raw") or item.get("disease_core")))
    if field == "interventions":
        return (_norm(item.get("name_raw")), _norm(item.get("product_or_formulation")))
    if field == "regimens":
        return (_norm(item.get("label")), item.get("composition"))
    if field == "designs":
        return (_norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("raw_name")))
    if field == "dlt_windows":
        return (_norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("raw_definition")))
    if field == "dlt_definitions":
        return (_norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("version_label")))
    if field == "dose_series":
        return (
            _norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("arm_id")),
            _norm(item.get("regimen_id")), _norm(item.get("varying_component_id")), _norm(item.get("label")),
        )
    if field == "cohorts":
        dose = item.get("dose") or {}
        return (
            _norm(item.get("series_id")), _norm(item.get("regimen_id")), item.get("cohort_role"),
            dose.get("value_kind"), dose.get("value"), dose.get("range_min"), dose.get("range_max"),
            tuple(dose.get("discrete_values") or []), _norm(dose.get("unit_normalized") or dose.get("unit_raw")),
            _norm(item.get("label")),
            exposure_identity(item.get("exposures", [])),
        )
    if field == "analysis_populations":
        return (_norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("author_name")))
    if field == "population_counts":
        return (
            _norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("arm_id")),
            _norm(item.get("cohort_id")), _norm(item.get("result_scope_id")), item.get("role"),
            _norm(item.get("analysis_population_id")), _norm(item.get("timepoint_raw")),
            _norm(item.get("window_id")), _norm(item.get("data_cutoff_raw")),
        )
    if field == "result_scopes":
        return (
            _norm(item.get("study_id")), _norm(item.get("part_id")), _norm(item.get("arm_id")),
            _norm(item.get("window_id")), tuple(sorted(_norm(x) for x in item.get("cohort_ids", []))),
        )
    return None


def _replace_ids(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_ids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_ids(item, mapping) for key, item in value.items()}
    return value


def _align_semantic_ids(base: ReportedExtraction, incoming: ReportedExtraction) -> ReportedExtraction:
    """Align independently generated IDs using clinical identity before merging."""

    data = incoming.model_dump(mode="json")
    mapping: dict[str, str] = {}
    for field, id_field in _COLLECTION_ID_FIELDS.items():
        if mapping:
            data = _replace_ids(data, mapping)
        existing_by_key: dict[tuple[Any, ...], str] = {}
        for existing in getattr(base, field):
            payload = existing.model_dump(mode="json")
            key = _semantic_key(field, payload)
            if key is not None and any(part not in ("", None, ()) and part != [] for part in key):
                existing_by_key[key] = str(payload[id_field])
        for item in data.get(field, []):
            key = _semantic_key(field, item)
            current_id = str(item.get(id_field) or "")
            if key in existing_by_key and current_id and current_id != existing_by_key[key]:
                mapping[current_id] = existing_by_key[key]
    if mapping:
        data = _replace_ids(data, mapping)
    return ReportedExtraction.model_validate(data)


def _merge_models(base: ReportedExtraction | None, incoming: ReportedExtraction) -> ReportedExtraction:
    if base is None:
        return incoming.model_copy(deep=True)
    merged = base.model_copy(deep=True)
    merged.exposure_contract = incoming.exposure_contract or base.exposure_contract
    incoming = _align_semantic_ids(merged, incoming)
    conflicts: list[UnresolvedItem] = []

    def merge_value(left: Any, right: Any, path: str) -> tuple[Any, list[str]]:
        if left == right:
            return left, []
        if left is None or left == "" or (isinstance(left, str) and left in {"unknown", "unclear"}):
            return right, []
        if right is None or right == "" or (isinstance(right, str) and right in {"unknown", "unclear"}):
            return left, []
        if isinstance(left, list) and isinstance(right, list):
            if not left:
                return right, []
            if not right:
                return left, []
            def item_id(item: Any) -> str | None:
                if not isinstance(item, dict):
                    return None
                return next(
                    (str(value) for key, value in item.items() if key.endswith("_id") and value),
                    None,
                )

            if all(item_id(item) for item in left + right):
                merged_list = list(left)
                positions = {item_id(item): index for index, item in enumerate(merged_list)}
                errors: list[str] = []
                for item in right:
                    key = item_id(item)
                    if key not in positions:
                        positions[key] = len(merged_list)
                        merged_list.append(item)
                        continue
                    position = positions[key]
                    merged_list[position], child_errors = merge_value(
                        merged_list[position], item, f"{path}[{key}]"
                    )
                    errors.extend(child_errors)
                return merged_list, errors
            merged_list = list(left)
            seen = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in merged_list}
            for item in right:
                marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if marker not in seen:
                    seen.add(marker)
                    merged_list.append(item)
            return merged_list, []
        if isinstance(left, dict) and isinstance(right, dict):
            merged_dict = dict(left)
            errors: list[str] = []
            for key, value in right.items():
                if key not in merged_dict:
                    merged_dict[key] = value
                    continue
                merged_dict[key], child_errors = merge_value(merged_dict[key], value, f"{path}.{key}")
                errors.extend(child_errors)
            if errors and merged_dict.get("reporting_status") in {
                "reported", "not_reported", "unclear", "conflicting"
            }:
                merged_dict["reporting_status"] = "conflicting"
            from .drug_ordering import ORDER_FIELDS
            clinical_errors = [p for p in errors if p.rsplit(".", 1)[-1] not in ORDER_FIELDS]
            if clinical_errors and "exposure_id" in merged_dict:
                merged_dict["dose_status"] = "conflicting"
            return merged_dict, errors
        return left, [path]
    fields = (
        "studies", "study_parts", "arms", "populations", "interventions", "regimens",
        "designs", "dlt_windows", "dlt_definitions", "dose_series", "cohorts",
        "analysis_populations", "population_counts", "result_scopes", "dlt_observations",
        "count_equations", "unresolved_items",
    )
    for field in fields:
        current = getattr(merged, field)
        by_id: dict[str, Any] = {}
        for item in current:
            data = item.model_dump(mode="json")
            key = next((str(value) for name, value in data.items() if name.endswith("_id") and value), "")
            if key:
                by_id[key] = item
        for item in getattr(incoming, field):
            data = item.model_dump(mode="json")
            key = next((str(value) for name, value in data.items() if name.endswith("_id") and value), "")
            if not key or key not in by_id:
                current.append(item)
                if key:
                    by_id[key] = item
            elif by_id[key].model_dump(mode="json") != data:
                existing = by_id[key]
                merged_data, conflicting_paths = merge_value(existing.model_dump(mode="json"), data, field)
                merged_item = type(existing).model_validate(merged_data)
                position = current.index(existing)
                current[position] = merged_item
                by_id[key] = merged_item
                if conflicting_paths:
                    digest = hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
                    conflicts.append(
                        UnresolvedItem(
                            item_id=f"merge_conflict_{digest}",
                            category="cross_packet_conflict",
                            description=(
                                f"Packet outputs disagree for {field} ID {key} at "
                                + ", ".join(conflicting_paths[:8])
                                + "; both complete records are preserved in this conflict: existing="
                                + json.dumps(existing.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                                + "; incoming="
                                + json.dumps(data, ensure_ascii=False, sort_keys=True)
                            ),
                            affected_ids=[key],
                        )
                    )
    merged.unresolved_items.extend(conflicts)
    coverage_by_category = {item.category: item for item in merged.extraction_coverage}
    for item in incoming.extraction_coverage:
        existing = coverage_by_category.get(item.category)
        if existing is None:
            merged.extraction_coverage.append(item)
            coverage_by_category[item.category] = item
        elif existing.status != item.status:
            existing.status = "conflict"
            existing.source_ids = list(dict.fromkeys([*existing.source_ids, *item.source_ids]))
            existing.note = " | ".join(filter(None, [existing.note, item.note, "Packet coverage statuses disagree"]))
    merged.coverage_notes.extend(note for note in incoming.coverage_notes if note not in merged.coverage_notes)
    merged.extraction_complete = merged.extraction_complete and incoming.extraction_complete
    if merged.material_coverage_complete is False or incoming.material_coverage_complete is False:
        merged.material_coverage_complete = False
    elif incoming.material_coverage_complete is not None:
        merged.material_coverage_complete = incoming.material_coverage_complete
    merged.title = merged.title or incoming.title
    return merged


def validate_evidence_references(extraction: ReportedExtraction, source_ids: set[str]) -> list[str]:
    warnings: list[str] = []

    def walk(value: Any, context: str) -> None:
        if hasattr(value, "model_dump"):
            data = value.model_dump(mode="json")
            record_id = next(
                (str(item) for name, item in data.items() if name.endswith("_id") and item),
                context,
            )
            evidence = getattr(value, "evidence", None)
            if isinstance(evidence, list):
                for link in evidence:
                    unknown = sorted(set(link.source_ids) - source_ids)
                    if unknown:
                        warnings.append(f"{record_id} cites unknown source IDs: {', '.join(unknown)}")
            for field_name in type(value).model_fields:
                if field_name == "evidence":
                    continue
                walk(getattr(value, field_name), record_id)
        elif isinstance(value, list):
            for item in value:
                walk(item, context)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, context)

    walk(extraction, extraction.document_id)
    return warnings


def _extract_packets(
    *,
    packets: list[dict[str, Any]],
    document: Any,
    raw_response_dir: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    client: OpenAI | None = None,
    caller: Callable[..., tuple[ReportedExtraction, Any]] = call_model,
    budget_guard: BudgetGuard | None = None,
    raw_request_dir: Path | None = None,
    parsed_packet_dir: Path | None = None,
    resume: bool = True,
    sanitize_errors: bool = False,
) -> tuple[ReportedExtraction, list[dict[str, Any]]]:
    public = os.getenv("PHASE1_DEPLOYMENT", "").strip().lower() == "public"
    sanitize_errors = sanitize_errors or public
    if client is None and public:
        raise ValueError("Public extraction requires an explicit user API client")
    raw_response_dir.mkdir(parents=True, exist_ok=True)
    raw_request_dir = raw_request_dir or raw_response_dir.parent / "raw_requests"
    parsed_packet_dir = parsed_packet_dir or raw_response_dir.parent / "parsed_packets"
    raw_request_dir.mkdir(parents=True, exist_ok=True)
    parsed_packet_dir.mkdir(parents=True, exist_ok=True)
    merged: ReportedExtraction | None = None
    summaries: list[dict[str, Any]] = []
    active_client = client
    for packet in packets:
        # Recheck actual files: a saved inventory can become stale after packetization.
        inventory = packet_image_inventory(packet, document)
        missing = [item for item in inventory if item.get("required", True)
                   and item.get("status") not in {"included", "covered_by_context_page"}]
        if missing:
            raise ValueError(f"{packet['packet_id']} has missing visual request material")
        images = packet_image_paths(packet, document)
        body = build_request_body(
            packet=packet,
            prior=merged,
            image_paths=images,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
        request_hash = sha256_json(body)
        request_path = raw_request_dir / f"{packet['packet_id']}.json"
        raw_path = raw_response_dir / f"{packet['packet_id']}.json"
        parsed_path = parsed_packet_dir / f"{packet['packet_id']}.json"
        failure_path = raw_response_dir / f"{packet['packet_id']}.failure.json"
        response = load_cached_response(raw_path, request_hash) if resume else None
        resumed = response is not None
        write_json(request_path, {
            "format": "phase1_frozen_responses_request_v3.1",
            "created_at_utc": utc_now(),
            "request_sha256": request_hash,
            "image_inventory": packet.get("image_inventory", []),
            "body": body,
        })
        usage: Any = None
        try:
            if resumed:
                # Replay the paid raw response through today's local validation;
                # no network call or new budget is required after a local crash.
                usage = response.get("usage")
                parsed = parse_model_response(response, document_id=packet["document_id"])
            else:
                if budget_guard is not None:
                    budget_guard.admit_request(body)
                if caller is call_model:
                    active_client = active_client if active_client is not None else OpenAI()
                    response = active_client.responses.create(**body)
                    payload = response_payload(response)
                    usage = payload.get("usage")
                    # Save before parsing either model output or usage. Even a
                    # local write failure must still count the paid response.
                    try:
                        cache_response(raw_path, request_hash, payload)
                    finally:
                        if budget_guard is not None:
                            budget_guard.record(usage)
                    parsed = parse_model_response(payload, document_id=packet["document_id"])
                else:
                    parsed, response = caller(
                        packet=packet,
                        prior=merged,
                        image_paths=images,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        max_output_tokens=max_output_tokens,
                        client=client,
                    )
                    payload = response_payload(response)
                    usage = payload.get("usage")
                    try:
                        cache_response(raw_path, request_hash, payload)
                    finally:
                        if budget_guard is not None:
                            budget_guard.record(usage)
                    if parsed.document_id != packet["document_id"]:
                        raise RuntimeError("Model returned a different document_id")
            write_json(parsed_path, parsed)
            failure_path.unlink(missing_ok=True)
        except Exception as error:
            write_json(failure_path, {
                "request_sha256": request_hash,
                "failed_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "message": "Extraction request failed; provider details are withheld." if sanitize_errors else str(error),
                "usage": usage.model_dump(mode="json") if hasattr(usage, "model_dump") else usage,
                "resumed_without_request": resumed,
            })
            raise
        merged = _merge_models(merged, parsed)
        payload = response_payload(response) or {}
        summaries.append({
            "packet_id": packet["packet_id"],
            "model": model,
            "image_count": len(images),
            "request_sha256": request_hash,
            "request_path": str(request_path.resolve()),
            "raw_response_path": str(raw_path.resolve()),
            "resumed_without_request": resumed,
            "response_id": payload.get("id"),
            "usage": usage.model_dump(mode="json") if hasattr(usage, "model_dump") else usage,
            "cumulative_estimated_usd": budget_guard.spent_usd if budget_guard is not None else None,
        })
    if merged is None:
        raise RuntimeError("No extraction packets were processed")
    # The prompt intentionally makes intermediate packets incomplete. Only the
    # final response can declare completion, after every packet has succeeded.
    merged.extraction_complete = bool(packets[-1].get("is_final_packet") and parsed.extraction_complete)
    source_ids = {block.block_id for block in document.blocks} | {table.table_id for table in document.tables}
    source_ids |= {cell.cell_id for table in document.tables for cell in table.cells}
    source_ids |= {fragment.fragment_id for table in document.tables for fragment in table.fragments}
    source_ids |= {page.page_id for page in document.pages}
    merged.coverage_notes.extend(validate_evidence_references(merged, source_ids))
    from .drug_ordering import apply_drug_ordering
    return apply_drug_ordering(merged), summaries


@wraps(_extract_packets)
def extract_packets(*, raw_response_dir: Path, **kwargs: Any) -> tuple[ReportedExtraction, list[dict[str, Any]]]:
    with artifact_lock(Path(raw_response_dir).parent / ".extraction.lock"):
        return _extract_packets(raw_response_dir=Path(raw_response_dir), **kwargs)
