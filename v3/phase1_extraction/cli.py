"""Command line interface for the independent v3 extraction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import ArticlePaths, DEFAULT_OUTPUT_ROOT, read_json
from .batch import run_batch
from .comparison import compare_pilot_roots
from .llm import BudgetGuard, DEFAULT_CACHED_INPUT_MULTIPLIER, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL, DEFAULT_REASONING_EFFORT
from .packets import DEFAULT_MAX_PACKET_CHARS
from .pipeline import extract_reported, ingest, parse_sources, resolve_reported, run_pipeline
from .openai_batch import (
    collect_batch_results,
    prepare_batch_manifest,
    refresh_batch_state,
    submit_batch_files,
)


def _paths(args: argparse.Namespace) -> ArticlePaths:
    return ArticlePaths(Path(args.output_root), args.document_id)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("document_id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def _add_extraction_options(parser: argparse.ArgumentParser, *, include_reported_input: bool = True) -> None:
    if include_reported_input:
        parser.add_argument("--reported-input", type=Path)
    parser.add_argument("--execute", action="store_true", help="Send paid model requests")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-packet-chars", type=int, default=DEFAULT_MAX_PACKET_CHARS)
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--cache-write-multiplier", type=float, default=1.25)
    parser.add_argument("--cached-input-multiplier", type=float, default=DEFAULT_CACHED_INPUT_MULTIPLIER)
    parser.add_argument("--no-resume", action="store_true", help="Send fresh requests instead of reusing paid responses")


def _budget(args: argparse.Namespace) -> BudgetGuard | None:
    if not args.execute:
        return None
    missing = [
        name
        for name, value in (
            ("--max-usd", args.max_usd),
            ("--input-usd-per-million", args.input_usd_per_million),
            ("--output-usd-per-million", args.output_usd_per_million),
        )
        if value is None
    ]
    if missing:
        raise SystemExit("Paid extraction requires explicit current pricing and budget: " + ", ".join(missing))
    return BudgetGuard(
        max_usd=args.max_usd,
        input_usd_per_million=args.input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,
        cache_write_multiplier=args.cache_write_multiplier,
        cached_input_multiplier=args.cached_input_multiplier,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase1-extract-v3")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest", help="Register source files")
    _add_common(ingest_parser)
    ingest_parser.add_argument("sources", nargs="+", type=Path)
    parse_parser = sub.add_parser("parse", help="Build DocumentIR and table/page assets")
    _add_common(parse_parser)
    extract_parser = sub.add_parser("extract", help="Build packets and optionally run model extraction")
    _add_common(extract_parser)
    _add_extraction_options(extract_parser)
    resolve_parser = sub.add_parser("resolve", help="Apply denominator/DLT project rules")
    _add_common(resolve_parser)
    run_parser = sub.add_parser("run", help="Run ingest, parse, extract, and resolve")
    _add_common(run_parser)
    run_parser.add_argument("sources", nargs="+", type=Path)
    _add_extraction_options(run_parser)
    status_parser = sub.add_parser("status", help="Print the run manifest")
    _add_common(status_parser)
    batch_parser = sub.add_parser("batch", help="Run a failure-isolated JSONL article manifest")
    batch_parser.add_argument("manifest_path", type=Path)
    batch_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_extraction_options(batch_parser, include_reported_input=False)
    api_prepare = sub.add_parser("api-batch-prepare", help="Freeze one-packet articles for the OpenAI Batch API")
    api_prepare.add_argument("manifest_path", type=Path)
    api_prepare.add_argument("--input-root", type=Path, required=True)
    api_prepare.add_argument("--workspace", type=Path, required=True)
    api_prepare.add_argument("--model", default=DEFAULT_MODEL)
    api_prepare.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    api_prepare.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    api_submit = sub.add_parser("api-batch-submit", help="Upload frozen JSONL and create true Batch jobs")
    api_submit.add_argument("workspace", type=Path)
    api_submit.add_argument("--execute", action="store_true", help="Acknowledge that this creates paid Batch jobs")
    api_submit.add_argument("--max-usd", type=float, required=True)
    api_submit.add_argument("--input-usd-per-million", type=float, required=True)
    api_submit.add_argument("--output-usd-per-million", type=float, required=True)
    api_submit.add_argument("--cache-write-multiplier", type=float, default=1.25)
    api_submit.add_argument("--cached-input-multiplier", type=float, default=DEFAULT_CACHED_INPUT_MULTIPLIER)
    api_status = sub.add_parser("api-batch-status", help="Refresh a true Batch job state file")
    api_status.add_argument("state_path", type=Path)
    api_collect = sub.add_parser("api-batch-collect", help="Download and resolve completed Batch outputs")
    api_collect.add_argument("state_path", type=Path)
    api_collect.add_argument("frozen_index_path", type=Path)
    api_collect.add_argument("--output-root", type=Path, required=True)
    compare_parser = sub.add_parser("compare-transports", help="Compare synchronous and Batch extraction outputs")
    compare_parser.add_argument("sync_root", type=Path)
    compare_parser.add_argument("batch_root", type=Path)
    compare_parser.add_argument("output_path", type=Path)
    sub.add_parser("schema", help="Print the model response JSON schema")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "schema":
        from .models import ReportedExtraction

        print(json.dumps(ReportedExtraction.model_json_schema(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "ingest":
        paths = ingest(document_id=args.document_id, source_paths=args.sources, output_root=args.output_root)
    elif args.command == "parse":
        paths = _paths(args)
        parse_sources(paths)
    elif args.command == "extract":
        paths = _paths(args)
        extract_reported(
            paths,
            reported_input=args.reported_input,
            execute_model=args.execute,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            max_packet_chars=args.max_packet_chars,
            budget_guard=_budget(args),
            resume=not args.no_resume,
        )
    elif args.command == "resolve":
        paths = _paths(args)
        resolve_reported(paths)
    elif args.command == "run":
        paths = run_pipeline(
            document_id=args.document_id,
            source_paths=args.sources,
            output_root=args.output_root,
            reported_input=args.reported_input,
            execute_model=args.execute,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            max_packet_chars=args.max_packet_chars,
            budget_guard=_budget(args),
            resume=not args.no_resume,
        )
    elif args.command == "status":
        paths = _paths(args)
        print(json.dumps(read_json(paths.manifest), ensure_ascii=False, indent=2))
        return 0
    elif args.command == "batch":
        state_path = run_batch(
            manifest_path=args.manifest_path,
            output_root=args.output_root,
            execute_model=args.execute,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            max_packet_chars=args.max_packet_chars,
            budget_guard=_budget(args),
            resume=not args.no_resume,
        )
        print(state_path.resolve())
        return 0
    elif args.command == "api-batch-prepare":
        _requests, index_path = prepare_batch_manifest(
            manifest_path=args.manifest_path,
            input_root=args.input_root,
            workspace=args.workspace,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        )
        print(index_path.resolve())
        return 0
    elif args.command == "api-batch-submit":
        if not args.execute:
            raise SystemExit("api-batch-submit requires --execute because it creates paid Batch jobs")
        routing_plan = read_json(args.workspace / "routing_plan.json")
        input_files = [Path(path) for path in routing_plan.get("batch_input_files", [])]
        state_path = submit_batch_files(
            input_files,
            args.workspace,
            frozen_index_path=args.workspace / "frozen_requests.json",
            max_usd=args.max_usd,
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
            cache_write_multiplier=args.cache_write_multiplier,
            cached_input_multiplier=args.cached_input_multiplier,
        )
        print(state_path.resolve())
        return 0
    elif args.command == "api-batch-status":
        print(json.dumps(refresh_batch_state(args.state_path), ensure_ascii=False, indent=2))
        return 0
    elif args.command == "api-batch-collect":
        paths = collect_batch_results(
            state_path=args.state_path,
            frozen_index_path=args.frozen_index_path,
            output_root=args.output_root,
        )
        print("\n".join(str(path.root.resolve()) for path in paths))
        return 0
    elif args.command == "compare-transports":
        payload = compare_pilot_roots(args.sync_root, args.batch_root, args.output_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    else:
        raise AssertionError(args.command)
    print(paths.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
