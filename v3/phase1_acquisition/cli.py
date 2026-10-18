"""First 4,000 verified PubMed full texts → binary screening."""
import argparse
import json
from pathlib import Path

from phase1_project.runtime import WORKSPACE_ROOT
from .campaign import run, collect_results, status
from .discovery import ARTICLE_COUNT

DEFAULT_WORKSPACE = WORKSPACE_ROOT / f'v3/output/pubmed_fulltext_{ARTICLE_COUNT}_drug_dlt_v5'


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=f'Select and screen the first {ARTICLE_COUNT:,} PubMed oncology Phase I hits with verified full text')
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('run', help=f'Continue retrieval until {ARTICLE_COUNT:,} verified full texts are ready, then submit all')
    start.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    start.add_argument('--through', help='Inclusive publication cutoff YYYY/MM/DD; fixed on first run')
    start.add_argument('--max-usd', type=float, default=25, help='Total conservative request reserve, checked before any submission')
    rescreen = commands.add_parser('rescreen', help='Rescreen an identical verified full-text cohort with the current policy')
    rescreen.add_argument('--source-workspace', type=Path, required=True)
    rescreen.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    rescreen.add_argument('--max-usd', type=float, default=25)
    for name, help_text in [('collect', 'Collect existing Batch results once; never resubmit'),
                            ('status', 'Read local campaign state without network access')]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    args = parser.parse_args(argv)
    if args.command == 'run':
        result = run(args.workspace, through=args.through, max_usd=args.max_usd)
    elif args.command == 'rescreen':
        result = run(args.workspace, source_workspace=args.source_workspace, max_usd=args.max_usd)
    elif args.command == 'collect':
        result = collect_results(args.workspace)
    else:
        result = status(args.workspace)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
