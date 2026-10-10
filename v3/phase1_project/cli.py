from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from phase1_extraction.artifacts import read_json, write_json
from phase1_prediction.evidence import load_evidence_library
from phase1_prediction.models import TargetTrial
from phase1_prediction.multidrug.contracts import TargetRegimen, PriorSpec, SamplerSpec
from phase1_prediction.multidrug.service import DEFAULT_STORE, BACKENDS, SYNTHESIS_MODES
from phase1_prediction.service import predict_target_curve
from phase1_simulation.service import BoinSettings, run_boin_simulation

from .catalog import write_catalog


def _root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extraction-root", action="append", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Complete v3 evidence, prediction, and BOIN workflow")
    commands = root.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register-existing", help="Index existing local materials without crawling or extracting")
    register.add_argument("--output", type=Path, default=Path("output/library_index.json"))
    register.add_argument("--extraction-root", action="append", type=Path)
    register.add_argument("--xml-dir", action="append", type=Path)
    register.add_argument("--pdf-dir", action="append", type=Path)
    predict = commands.add_parser("predict", help="Fit target curves on demand from native v3 evidence")
    _root_args(predict)
    predict.add_argument("--target", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--evidence-policy", choices=["project_default", "reported_only"], default="project_default")
    predict.add_argument("--review-policy", choices=["extracted"], default="extracted")
    predict.add_argument("--backend", choices=BACKENDS, default="logistic")
    predict.add_argument("--synthesis", choices=SYNTHESIS_MODES, default=SYNTHESIS_MODES[0])
    predict.add_argument("--prior-json", type=Path)
    predict.add_argument("--sampler-json", type=Path)
    predict.add_argument("--power-prior-json", type=Path, help="Method 3 fixed discount, patient budget and heterogeneity prior settings")
    predict.add_argument("--store", type=Path, default=DEFAULT_STORE)
    predict.add_argument("--threshold", type=float, default=0.)
    predict.add_argument("--top-articles", type=int)
    predict.add_argument("--draw-count", type=int, default=1000)
    predict.add_argument("--seed", type=int, default=20260919)
    simulate = commands.add_parser("simulate", help="Run BOIN from a complete target posterior")
    simulate.add_argument("--prediction", type=Path, required=True)
    simulate.add_argument("--method", required=True)
    simulate.add_argument("--settings", type=Path, required=True)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--posterior-curves", type=int, default=100)
    simulate.add_argument("--trials-per-curve", type=int, default=100)
    simulate.add_argument("--seed", type=int, default=9)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "register-existing":
        payload = write_catalog(args.output, extraction_roots=args.extraction_root, xml_directories=args.xml_dir, pdf_directories=args.pdf_dir)
        print(json.dumps({"output": str(args.output), "article_count": payload["article_count"], "crawler_was_run": False}, indent=2))
    elif args.command == "predict":
        raw = read_json(args.target)
        target = TargetRegimen.model_validate(raw) if raw.get('version') == 'target_regimen_v1' else TargetTrial(**raw)
        library = load_evidence_library(args.extraction_root, evidence_policy=args.evidence_policy, review_policy=args.review_policy)
        result = predict_target_curve(library, target, backend=args.backend, synthesis=args.synthesis,
            prior=PriorSpec.model_validate(read_json(args.prior_json)) if args.prior_json else None,
            sampler=SamplerSpec.model_validate(read_json(args.sampler_json)) if args.sampler_json else None,
            power_prior=read_json(args.power_prior_json) if args.power_prior_json else None,
            store_root=args.store, threshold=args.threshold, top_articles=args.top_articles,
            draw_count=args.draw_count, seed=args.seed)
        write_json(args.output, result.to_dict())
        print(json.dumps({"output": str(args.output), "selected_series": len(result.selected_series), "methods": {key: value["status"] for key, value in result.methods.items()}}, indent=2))
    else:
        prediction = read_json(args.prediction)
        settings = BoinSettings(**read_json(args.settings))
        result = run_boin_simulation(prediction, method_key=args.method, settings=settings,
                                     posterior_curve_count=args.posterior_curves, trials_per_curve=args.trials_per_curve, seed=args.seed)
        write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "total_trials": result["total_trials"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
