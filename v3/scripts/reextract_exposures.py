"""Re-extract the previously completed v3 corpus using preserved source IR.

Prepare is offline. Submit is explicit and bounded by the saved run prices.
Old outputs are not input to the model; only original document IR/images are.
"""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
from phase1_extraction.artifacts import ArticlePaths, read_json, write_json, utc_now
from phase1_extraction.llm import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT
from phase1_extraction.openai_batch import (
    freeze_single_packet_request, write_frozen_request_index, write_batch_input_files,
    validate_batch_input_files, submit_batch_files, refresh_batch_state, collect_batch_results,
    TERMINAL_BATCH_STATUSES,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'output/v3_batch_exposures_20260910'


def batch_pricing(model: str) -> dict[str, float]:
    # Keep frozen historical jobs on their original pricing basis.
    if model == 'gpt-6.1-sol':
        return {'input_usd_per_million': 1, 'output_usd_per_million': 5,
                'cache_write_multiplier': 1.25, 'cached_input_multiplier': 0.05}
    if model == 'gpt-5.6-sol':
        return {'input_usd_per_million': 2, 'output_usd_per_million': 10,
                'cache_write_multiplier': 1.25, 'cached_input_multiplier': 0.1}
    raise ValueError(f'No verified Batch pricing for {model}; supply current prices through the extraction CLI')


def prepare(run: Path, source_root: Path | None = None, skip_quality_audit: bool = False) -> None:
    workspace = run / 'batch_workspace'
    if (workspace / 'batch_state.json').exists():
        raise RuntimeError('Already submitted; use status/collect, never rebuild a submitted request.')
    old = {}
    for path in sorted((source_root or ROOT / 'output').rglob('reported_extraction.json')):
        if run in path.parents:
            continue
        value = read_json(path)
        if value.get('schema_version') in {'phase1_extraction_v3', 'phase1_extraction_v3.2'}:
            old.setdefault(value['document_id'], []).append(str(path.parent.resolve()))
    if not old:
        raise RuntimeError('No prior v3 results found')
    records, frozen = [], []
    for document_id, previous_roots in sorted(old.items()):
        candidates = [ROOT / 'output/v3_batch_10_20260907/inputs' / document_id,
                      ROOT / 'output/v3_batch_pilot_3_20260906/inputs' / document_id]
        candidates += [Path(p) for p in previous_roots]
        source = next((p for p in candidates if (p / 'document_ir.json').is_file()
                       and (p / 'material_manifest.json').is_file() and (p / 'manifest.json').is_file()), None)
        if source is None:
            raise RuntimeError(f'Missing original parsed sources for {document_id}')
        paths = ArticlePaths(run / 'inputs', document_id)
        paths.ensure()
        for name in ('document_ir.json', 'material_manifest.json', 'manifest.json'):
            shutil.copy2(source / name, paths.root / name)
        request = freeze_single_packet_request(paths)
        frozen.append(request)
        records.append({'document_id': document_id, 'previous_result_roots': previous_roots,
                        'source_input_root': str(source.resolve()), 'new_input_root': str(paths.root.resolve())})
    index = write_frozen_request_index(frozen, workspace / 'frozen_requests.json')
    files = write_batch_input_files(frozen, workspace)
    validation = validate_batch_input_files(files, index)
    prices = batch_pricing(DEFAULT_MODEL)
    write_json(workspace / 'routing_plan.json', {
        'created_at_utc': utc_now(), 'model': DEFAULT_MODEL, 'reasoning_effort': DEFAULT_REASONING_EFFORT,
        'pricing': prices,
        'batch_input_files': [str(p.resolve()) for p in files],
        'batch_input_file_records': validation['file_records'],
        'batch_request_count': len(frozen), 'synchronous_fallback_count': 0,
        **{k: validation[k] for k in ('estimated_input_tokens', 'reserved_output_tokens')},
    })
    write_json(run / 'rerun_manifest.json', {'created_at_utc': utc_now(), 'articles': records,
               'skip_quality_audit': skip_quality_audit,
               'exposure_contract': 'cohort_exposures_v1',
               'pricing_basis': f'{DEFAULT_MODEL} Batch prices verified 2026-10-07',
               'pricing': prices,
               'max_reserved_usd': 15})
    reserved = (validation['estimated_input_tokens'] * prices['input_usd_per_million'] * prices['cache_write_multiplier']
                + validation['reserved_output_tokens'] * prices['output_usd_per_million']) / 1e6
    print(json.dumps({'articles': len(frozen), 'conservative_reserve_usd_at_saved_rates': reserved, 'workspace': str(workspace)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'submit', 'status', 'collect', 'wait'])
    parser.add_argument('--run', type=Path, default=DEFAULT)
    parser.add_argument('--source-root', type=Path, help='Limit preparation to this existing extraction library')
    parser.add_argument('--skip-quality-audit', action='store_true', help='Persist results without the optional extraction quality audit')
    args = parser.parse_args()
    run = args.run.resolve()
    workspace = run / 'batch_workspace'
    if args.action == 'prepare':
        prepare(run, args.source_root, args.skip_quality_audit)
    elif args.action == 'submit':
        plan = read_json(workspace / 'routing_plan.json')
        prices = plan.get('pricing') or batch_pricing(plan['model'])
        state = submit_batch_files([Path(p) for p in plan['batch_input_files']], workspace,
            max_usd=15, **prices)
        print(state)
    elif args.action == 'status':
        state = refresh_batch_state(workspace / 'batch_state.json')
        print(json.dumps([{'status': x['status'], 'counts': x.get('request_counts'),
                           'errors': x.get('errors')} for x in state['submissions']]))
    else:
        if args.action == 'wait':
            last = None
            while True:
                state = refresh_batch_state(workspace / 'batch_state.json')
                statuses = [{'status': x['status'], 'counts': x.get('request_counts')} for x in state['submissions']]
                if statuses != last:
                    print(json.dumps(statuses), flush=True)
                    write_json(run / 'progress.json', {'updated_at_utc': utc_now(), 'submissions': statuses})
                    last = statuses
                if all(x['status'] in TERMINAL_BATCH_STATUSES for x in state['submissions']):
                    break
                time.sleep(30)
        paths = collect_batch_results(state_path=workspace / 'batch_state.json',
            frozen_index_path=workspace / 'frozen_requests.json', output_root=run / 'results',
            run_quality_audit=not (args.skip_quality_audit or read_json(run / 'rerun_manifest.json').get('skip_quality_audit', False)))
        print(json.dumps({'collected': [p.document_id for p in paths]}))


if __name__ == '__main__':
    main()
