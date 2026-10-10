"""Audit primary-first allocation; optionally fit the isolated numeric pilot.

Run from the repository root with the V3 interpreter. No external interaction
lookup is converted to a numeric prior, and no application settings are changed.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1_prediction.multidrug.contracts import TargetRegimen
from phase1_prediction.multidrug.interaction_pilot import PairPriorScenario, predict_primary_first_pilot
from phase1_prediction.multidrug.observations import load_model_library
from phase1_prediction.multidrug.primary_first import allocate_primary_first
from phase1_prediction.multidrug.service import fit_library
from phase1_project.catalog import default_extraction_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True, help='Saved TargetRegimen JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit', action='store_true', help='Reuse/fill Component Logistic source caches and evaluate the pilot')
    parser.add_argument('--top-articles-per-drug', type=int)
    parser.add_argument('--aggregation', choices=['average', 'mixture'], default='mixture')
    parser.add_argument('--interaction-mean', type=float, default=0.)
    parser.add_argument('--interaction-sd', type=float, default=1.)
    parser.add_argument('--draw-count', type=int, default=1000)
    args = parser.parse_args()
    target = TargetRegimen.model_validate(json.loads(args.target.read_text()))
    scopes, diagnostics = load_model_library(default_extraction_roots())
    allocation = allocate_primary_first(scopes, target,
        top_articles_per_drug=args.top_articles_per_drug, require_usable_coordinates=True)
    output = dict(allocation=allocation, library_diagnostics=diagnostics,
                  source_article_count=len({s.document_id for s in scopes}), source_scope_count=len(scopes))
    if args.fit:
        # Fit all eligible monotherapy candidates, so failed A sources do not
        # reserve a drug that a successful B source could otherwise support.
        mono = [s for s in scopes if len(s.drugs) == 1 and s.drugs[0].mode == 'numeric' and s.observations]
        entries, errors = fit_library(mono, backends=('logistic',),
            progress=lambda i, n, doc, backend: print(f'{i+1}/{n} {doc} {backend}', flush=True))
        scenario = PairPriorScenario(mean=args.interaction_mean, sd=args.interaction_sd,
            basis='weak_default' if (args.interaction_mean, args.interaction_sd) == (0., 1.) else 'sensitivity_scenario')
        output['prediction'] = predict_primary_first_pilot(entries, target,
            aggregation=args.aggregation, scenario=scenario, draw_count=args.draw_count,
            top_articles_per_drug=args.top_articles_per_drug)
        output['fit_errors'] = errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    print(f'Saved {args.output}')
    print('Allocation complete:', output.get('prediction', {}).get('allocation', allocation)['complete'])
    if args.fit:
        print('Prediction status:', output['prediction']['status'])
        print('External interaction data used: false (sensitivity analysis only)')


if __name__ == '__main__':
    main()
