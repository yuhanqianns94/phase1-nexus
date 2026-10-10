#!/usr/bin/env python3
"""Audit Bayesian per-article dose--DLT posterior curves.

The standalone CLI is an auditable descriptive view of the historical library.
Its canonical-unit arm-line builder is also used by the target curve estimator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .bayesian_curve_cache import cached_segment_fit
from .bayesian_toxicity import (
    DEFAULT_POSTERIOR_DRAW_COUNT,
    DEFAULT_PRIOR_CATALOG,
    BayesianPrior,
    annotate_figure_with_prior_panel,
    fit_bayesian_binomial_logistic,
    load_prior_catalog,
    posterior_probability_draws,
    resolve_prior,
)
from .dlt_evidence import (
    build_candidate_rows as build_canonical_candidate_rows,
    regimen_signature as canonical_regimen_signature,
)
from .dose_unit_conversion import conversion_metadata, describe_dose_unit
from .toxicity_curve_common import (
    ToxicityQuery,
    article_segments,
    is_not_reached,
    segment_is_all_zero,
    segment_mad_resolution,
    split_article_weight,
)
from .toxicity_similarity import (
    DEFAULT_EXTRACTION_DIR,
    canonical_dose_value,
    load_extractions,
)


DEFAULT_OUTPUT_PATH = Path("output/mtd_similarity/dlt_bayesian_regression_summary.json")
DEFAULT_PLOT_DIR = Path("output/mtd_similarity/dlt_toxicity_plots")
MISSING_VALUES = {"", "not_reported", "unclear", "not_applicable", "unknown", None}
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_SINGLE_OBSERVATION_RELIABILITY_WEIGHT = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit post-trial, per-arm Bayesian Binomial-logistic dose--DLT "
            "curves from the latest compact MTD extraction library."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--pmcid",
        help="Fit one specific article, for example PMC7328801.",
    )
    selection.add_argument(
        "--random",
        action="store_true",
        help=(
            "Select one article from the complete library. If it has no extracted "
            "DLT rows, no formula or plot is emitted."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used together with --random.",
    )
    parser.add_argument(
        "--extraction-dir",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR,
        help=f"Directory with compact extractions (default: {DEFAULT_EXTRACTION_DIR}).",
    )
    parser.add_argument(
        "--pattern",
        default="*_compact.json",
        help="Extraction filename glob (default: *_compact.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Batch summary JSON path. This is written only when neither --pmcid "
            "nor --random is selected."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help=f"Directory for single-article PNG plots (default: {DEFAULT_PLOT_DIR}).",
    )
    parser.add_argument(
        "--bayesian-prior",
        required=True,
        help="Named prior used for the standalone Bayesian audit.",
    )
    parser.add_argument(
        "--bayesian-prior-catalog",
        type=Path,
        default=DEFAULT_PRIOR_CATALOG,
    )
    parser.add_argument(
        "--posterior-draws",
        type=int,
        default=DEFAULT_POSTERIOR_DRAW_COUNT,
    )
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in MISSING_VALUES)


def field_value(value: Any) -> Any:
    """Unwrap a compact evidence field without unwrapping Dose_Value itself."""

    if isinstance(value, dict) and "value" in value and "evidence" in value:
        value = value["value"]
    return None if is_missing(value) else value


def normalized_key(value: Any) -> str:
    value = field_value(value)
    if is_missing(value):
        return ""
    return str(value).strip().casefold()


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def scalar_dose(dose_level: dict[str, Any]) -> float | None:
    dose_value = field_value(dose_level.get("Dose_Value"))
    if not isinstance(dose_value, dict) or dose_value.get("value_type") != "single":
        return None
    value = numeric(dose_value.get("value"))
    return value if value is not None and value > 0 else None


def dose_join_key(
    dose_level: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        normalized_key(dose_level.get("Dose_Level")),
        normalized_key(dose_level.get("arm_or_cohort")),
        normalized_key(dose_level.get("Drug_Name")),
    )


def dlt_join_key(dlt_row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalized_key(dlt_row.get("dose_level")),
        normalized_key(dlt_row.get("arm_or_cohort")),
        normalized_key(dlt_row.get("drug_name")),
    )


def match_dose_level(
    dlt_row: dict[str, Any], dose_levels: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    """Join a DLT row to exactly one dose level without guessing ambiguous fields."""

    level, arm, drug = dlt_join_key(dlt_row)
    if not level:
        return None, "missing_dose_level_label"

    candidates = [
        row
        for row in dose_levels
        if normalized_key(row.get("Dose_Level")) == level
    ]
    for expected, field in ((drug, "Drug_Name"), (arm, "arm_or_cohort")):
        if expected:
            narrowed = [
                row for row in candidates if normalized_key(row.get(field)) == expected
            ]
            if not narrowed:
                return None, f"dose_level_{field}_mismatch"
            candidates = narrowed

    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, "dose_level_not_found"
    return None, "ambiguous_dose_level_join"


def denominator_and_source(dlt_row: dict[str, Any]) -> tuple[float | None, str | None]:
    evaluable = numeric(dlt_row.get("n_patients_evaluable_for_dlt"))
    if evaluable is not None:
        return evaluable, "evaluable_for_dlt"
    treated = numeric(dlt_row.get("n_patients_treated"))
    if treated is not None:
        return treated, "treated"
    return None, None


def regimen_signature(frequency: Any, schedule: Any) -> str:
    """Preserve dose-regimen context when forming per-arm Bayesian curves."""

    return canonical_regimen_signature(frequency, schedule)


def build_candidate_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    canonical_rows, reasons = build_canonical_candidate_rows(data)
    return (
        [
            {
                "arm_or_cohort": row["arm_or_cohort"],
                "drug_name": row["drug_name"],
                "frequency": row["frequency"],
                "schedule": row["schedule"],
                "regimen_signature": row["regimen_signature"],
                "dose_level": row["dose_level"],
                "dose": row["canonical_dose"],
                "dose_unit": row["canonical_unit"],
                "source_dose": row["dose"],
                "source_dose_unit": row["dose_unit"],
                "unit_conversion": row["unit_conversion"],
                "n_patients_with_dlt": row["n_patients_with_dlt"],
                "n_patients": row["n_patients"],
                "denominator_source": row["denominator_source"],
                "evidence": row.get("evidence"),
                "dlt_value_origin": row.get("dlt_value_origin", "extracted"),
                "inference_note": row.get("inference_note"),
            }
            for row in canonical_rows
        ],
        reasons,
    )


def group_candidate_rows(
    candidates: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], set[str]]:
    """Group by arm, drug, and regimen, rejecting any mixed-unit group."""

    by_arm_drug_regimen: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            normalized_key(row["arm_or_cohort"]) or "unspecified_arm",
            normalized_key(row["drug_name"]) or "unspecified_drug",
            str(row["regimen_signature"]),
        )
        by_arm_drug_regimen[key].append(row)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    reasons: set[str] = set()
    for key, rows in by_arm_drug_regimen.items():
        units = {row["dose_unit"] for row in rows}
        if len(units) != 1:
            reasons.add("mixed_dose_units_within_arm")
            continue
        groups[key] = rows
    return groups, reasons


def aggregate_duplicate_doses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine same-dose cohorts only inside an already homogeneous arm/drug/unit."""

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dose"]].append(row)

    combined: list[dict[str, Any]] = []
    for dose, dose_rows in grouped.items():
        sources = sorted({row["denominator_source"] for row in dose_rows})
        combined.append(
            {
                "arm_or_cohort": dose_rows[0]["arm_or_cohort"],
                "drug_name": dose_rows[0]["drug_name"],
                "frequency": dose_rows[0]["frequency"],
                "schedule": dose_rows[0]["schedule"],
                "regimen_signature": dose_rows[0]["regimen_signature"],
                "dose_level": ", ".join(
                    str(row["dose_level"]) for row in dose_rows if row["dose_level"] is not None
                )
                or None,
                "dose": dose,
                "dose_unit": dose_rows[0]["dose_unit"],
                "n_patients_with_dlt": sum(row["n_patients_with_dlt"] for row in dose_rows),
                "n_patients": sum(row["n_patients"] for row in dose_rows),
                "denominator_source": "+".join(sources),
                "evidence": "; ".join(
                    str(row["evidence"])
                    for row in dose_rows
                    if row.get("evidence")
                )
                or None,
                "dlt_value_origins": sorted(
                    {str(row["dlt_value_origin"]) for row in dose_rows}
                ),
                "inference_notes": sorted(
                    {
                        str(row["inference_note"])
                        for row in dose_rows
                        if row.get("inference_note")
                    }
                ),
            }
        )
    return sorted(combined, key=lambda row: row["dose"])


def observed_probability(row: dict[str, Any]) -> float:
    value = row.get("observed_dlt_probability")
    if isinstance(value, (int, float)):
        return float(value)
    return row["n_patients_with_dlt"] / row["n_patients"]


def fit_segment(
    rows: list[dict[str, Any]],
    *,
    prior: BayesianPrior,
    posterior_draw_count: int = DEFAULT_POSTERIOR_DRAW_COUNT,
    random_seed: int = 0,
) -> dict[str, Any]:
    rows = aggregate_duplicate_doses(rows)
    identity = {
        "arm_or_cohort": rows[0]["arm_or_cohort"],
        "drug_name": rows[0]["drug_name"],
        "dose_unit": rows[0]["dose_unit"],
    }
    fit = fit_bayesian_binomial_logistic(
        rows,
        prior,
        draw_count=posterior_draw_count,
        random_seed=random_seed,
    )
    if fit["status"] != "fitted":
        return {
            "status": "not_fit_eligible",
            **identity,
            "skip_reasons": fit.get("skip_reasons", ["bayesian_fit_failed"]),
            "fit_diagnostics": fit.get("fit_diagnostics", {}),
            "dose_levels": rows,
        }
    dose_summaries = {
        float(point["dose"]): point
        for point in fit["fitted_dlt_probability"]
    }
    return {
        "status": "fitted",
        **identity,
        "model": fit["model"],
        "predictor": "dose",
        "assumptions": [
            "Observed DLT counts use a Binomial likelihood.",
            "The dose slope is constrained non-negative through softplus(eta).",
        ],
        "formula": (
            "logit(P(DLT | dose)) = alpha + softplus(eta) × "
            f"ln(dose / {fit['dose_standardization']['reference_dose']:.6g} "
            f"{identity['dose_unit']})"
        ),
        "probability_formula": "P(DLT | dose) = expit(logit(P(DLT | dose)))",
        "bayesian_fit": fit,
        "dose_levels": [
            {
                **row,
                "observed_dlt_probability": observed_probability(row),
                "fitted_dlt_probability": dose_summaries[float(row["dose"])][
                    "posterior_mean"
                ],
                "posterior_credible_interval_95": dose_summaries[
                    float(row["dose"])
                ]["posterior_credible_interval_95"],
            }
            for row in rows
        ],
    }


def analyze_article(
    record: dict[str, Any],
    *,
    prior: BayesianPrior,
    posterior_draw_count: int = DEFAULT_POSTERIOR_DRAW_COUNT,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Return every eligible per-arm fit, otherwise a transparent skip status."""

    pmcid = record["pmcid"]
    data = record["data"]
    dlt_rows = data.get("dlt_count_at_each_dose_level")
    if not isinstance(dlt_rows, list) or not dlt_rows:
        return {
            "pmcid": pmcid,
            "status": "no_dlt_extraction",
            "fits": [],
            "skip_reasons": ["no_dlt_count_at_each_dose_level"],
        }

    candidates, reasons = build_candidate_rows(data)
    groups, group_reasons = group_candidate_rows(candidates)
    reasons.update(group_reasons)
    fits = []
    for index, rows in enumerate(groups.values()):
        fits.append(
            fit_segment(
                rows,
                prior=prior,
                posterior_draw_count=posterior_draw_count,
                random_seed=random_seed + index,
            )
        )
    fitted = [fit for fit in fits if fit["status"] == "fitted"]
    if fitted:
        return {
            "pmcid": pmcid,
            "status": "fitted",
            "fits": fits,
            "skip_reasons": sorted(reasons),
        }

    for fit in fits:
        reasons.update(fit.get("skip_reasons", []))
    if not candidates:
        reasons.add("no_eligible_dlt_rows")
    return {
        "pmcid": pmcid,
        "status": "not_fit_eligible",
        "fits": fits,
        "skip_reasons": sorted(reasons),
    }


def build_similarity_historical_lines(
    scored_articles: list[dict[str, Any]],
    spec: ToxicityQuery,
    *,
    bayesian_cache: dict[str, Any],
    bayesian_prior: str,
    single_observation_reliability_weight: float = (
        DEFAULT_SINGLE_OBSERVATION_RELIABILITY_WEIGHT
    ),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build comparable historical posterior curves from the validated cache.

    A single observed dose is no longer forced through an assumed zero-DLT
    anchor. It remains a legitimate Binomial observation, regularized by the
    selected explicit prior. The fixed 0.25 reliability factor is retained as
    a transparent conservative mixture policy, not as fabricated evidence.
    """

    if not math.isfinite(single_observation_reliability_weight) or not (
        0 < single_observation_reliability_weight <= 1
    ):
        raise ValueError(
            "single_observation_reliability_weight must be in the interval (0, 1]."
        )
    lines: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for article in scored_articles:
        segments, reasons = article_segments(article, spec)
        for reason in reasons:
            skipped[reason] = skipped.get(reason, 0) + 1
        if not segments:
            continue
        article_similarity = max(0.0, float(article["similarity"]))
        if article_similarity <= 0:
            skipped["zero_similarity"] = skipped.get("zero_similarity", 0) + 1
            continue
        from ..profile_scoring import segment_weight

        for segment_index, segment in enumerate(segments, start=1):
            segment_similarity_weight = segment_weight(article, segment, segments)
            if segment_similarity_weight <= 0:
                continue
            fitted, cache_reason = cached_segment_fit(
                bayesian_cache,
                pmcid=article["pmcid"],
                segment=segment,
                prior_name=bayesian_prior,
            )
            if fitted is None:
                reason = cache_reason or "bayesian_cache_segment_unavailable"
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            resolved_mad, mad_resolution_reason = segment_mad_resolution(
                article, segment, spec
            )
            not_reached_all_zero = (
                is_not_reached(article)
                and segment_is_all_zero(segment)
                and resolved_mad is not None
            )
            mad = resolved_mad if not_reached_all_zero else None
            single_observation = len(segment) == 1
            lines.append(
                {
                    "pmcid": article["pmcid"],
                    "similarity": article["similarity"],
                    "article_similarity_weight": article_similarity,
                    "segment_similarity_weight": segment_similarity_weight,
                    "segment_index": segment_index,
                    "arm_or_cohort": segment[0]["arm_or_cohort"],
                    "drug_name": segment[0]["drug_name"],
                    "dose_unit": spec.target_unit,
                    "observed_dose_min": min(
                        row["canonical_dose"] for row in segment
                    ),
                    "observed_dose_max": max(
                        row["canonical_dose"] for row in segment
                    ),
                    "source_patient_count": sum(
                        row["n_patients"] for row in segment
                    ),
                    "source_points": segment,
                    "bayesian_fit": fitted,
                    "bayesian_prior": bayesian_prior,
                    "line_type": (
                        "single_observation_bayesian_binomial_logistic"
                        if single_observation
                        else "bayesian_binomial_logistic"
                    ),
                    "reliability_weight": (
                        single_observation_reliability_weight
                        if single_observation
                        else 1.0
                    ),
                    "reliability_policy": (
                        "fixed_single_observation_conservative_weight"
                        if single_observation
                        else "full_weight_for_two_or_more_observations"
                    ),
                    "mad_cap": mad,
                    "mad_cap_reason": (
                        "not_reached_all_zero_historical_arm"
                        if mad is not None
                        else None
                    ),
                    "mad_cap_unavailable_reason": (
                        mad_resolution_reason
                        if (
                            is_not_reached(article)
                            and segment_is_all_zero(segment)
                            and mad is None
                        )
                        else None
                    ),
                    "assumptions": (
                        [
                            "Laplace-approximated Bayesian Binomial-logistic "
                            "posterior under the selected prior."
                        ]
                        + (
                            [
                                "Only one observed dose is available; the curve is "
                                "prior-regularized and conservatively downweighted."
                            ]
                            if single_observation
                            else []
                        )
                    ),
                }
            )
    return lines, skipped


def safe_filename(value: str | None) -> str:
    return SAFE_FILENAME_RE.sub("_", value or "unspecified").strip("._") or "unspecified"


def plot_fit(pmcid: str, fit: dict[str, Any], plot_dir: Path) -> Path:
    """Write one observed-dose and fitted-probability curve PNG."""

    plot_dir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(plot_dir / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = fit["dose_levels"]
    doses = np.array([row["dose"] for row in rows], dtype=float)
    observed = np.array([row["observed_dlt_probability"] for row in rows], dtype=float)
    curve_doses = np.linspace(float(doses.min()), float(doses.max()), 200)
    bayesian_fit = fit["bayesian_fit"]
    curve_draws = np.vstack(
        [posterior_probability_draws(bayesian_fit, float(dose)) for dose in curve_doses]
    )
    curve_probabilities = np.mean(curve_draws, axis=1)
    curve_low = np.quantile(curve_draws, 0.025, axis=1)
    curve_high = np.quantile(curve_draws, 0.975, axis=1)

    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.plot(
        curve_doses, curve_probabilities, label="Bayesian posterior mean"
    )
    axis.fill_between(
        curve_doses,
        curve_low,
        curve_high,
        alpha=0.18,
        label="Bayesian posterior 95% credible interval",
    )
    extracted_rows = [
        row
        for row in rows
        if row.get("dlt_value_origins", ["extracted"]) == ["extracted"]
    ]
    inferred_rows = [row for row in rows if row not in extracted_rows]
    axis.scatter(
        [row["dose"] for row in extracted_rows],
        [row["observed_dlt_probability"] for row in extracted_rows],
        color="black",
        zorder=3,
        label="Extracted DLT rate",
    )
    if inferred_rows:
        axis.scatter(
            [row["dose"] for row in inferred_rows],
            [row["observed_dlt_probability"] for row in inferred_rows],
            marker="v",
            facecolors="none",
            edgecolors="dimgray",
            zorder=3,
            label="Model-inferred unreported zero",
        )
    for row, observed_probability in zip(rows, observed, strict=True):
        label = f"{int(row['n_patients_with_dlt'])}/{int(row['n_patients'])}"
        axis.annotate(
            label,
            (row["dose"], observed_probability),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel(f"Dose ({fit['dose_unit']})")
    axis.set_ylabel("DLT probability")
    arm = fit["arm_or_cohort"] or "unspecified arm"
    drug = fit["drug_name"] or "unspecified drug"
    prior_name = fit.get("prior_id") or (fit.get("prior") or {}).get("name")
    axis.set_title(f"{pmcid}: {drug}, {arm}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    annotate_figure_with_prior_panel(figure, prior_name)

    filename = (
        f"{safe_filename(pmcid)}__{safe_filename(drug)}__"
        f"{safe_filename(arm)}.png"
    )
    path = plot_dir / filename
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def build_summary(
    records: list[dict[str, Any]],
    *,
    prior: BayesianPrior,
    posterior_draw_count: int = DEFAULT_POSTERIOR_DRAW_COUNT,
) -> dict[str, Any]:
    articles = [
        analyze_article(
            record,
            prior=prior,
            posterior_draw_count=posterior_draw_count,
            random_seed=index,
        )
        for index, record in enumerate(records)
    ]
    counts: dict[str, int] = defaultdict(int)
    fit_count = 0
    for article in articles:
        counts[article["status"]] += 1
        fit_count += sum(fit["status"] == "fitted" for fit in article["fits"])
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "post_trial_per_article_bayesian_binomial_logistic",
        "model_formula": (
            "logit(P(DLT | dose)) = alpha + softplus(eta) × "
            "ln(dose / geometric_mean_observed_dose)"
        ),
        "prior_id": prior.name,
        "prior": prior.to_payload(),
        "posterior_draw_count": posterior_draw_count,
        "article_count": len(articles),
        "status_counts": dict(sorted(counts.items())),
        "fitted_arm_count": fit_count,
        "articles": articles,
    }


def print_article_result(article: dict[str, Any], plot_dir: Path) -> int:
    """Print formulas and save plots only for successfully fitted DLT curves."""

    if article["status"] == "no_dlt_extraction":
        return 0

    fitted = [fit for fit in article["fits"] if fit["status"] == "fitted"]
    if not fitted:
        print(
            f"{article['pmcid']}: DLT counts were extracted but no reliable Bayesian "
            f"fit was possible ({', '.join(article['skip_reasons'])})."
        )
        return 0

    for fit in fitted:
        plot_path = plot_fit(article["pmcid"], fit, plot_dir)
        print(f"{article['pmcid']} | {fit['drug_name']} | {fit['arm_or_cohort']}")
        print(fit["formula"])
        print(fit["probability_formula"])
        print(f"Example plot: {plot_path}")
    return 0


def main() -> int:
    args = parse_args()
    records = load_extractions(args.extraction_dir, args.pattern)
    try:
        catalog, _payload, _fingerprint = load_prior_catalog(
            args.bayesian_prior_catalog
        )
        prior = resolve_prior(args.bayesian_prior, catalog)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.pmcid:
        selected = [record for record in records if record["pmcid"] == args.pmcid]
        if not selected:
            print(f"PMCID not found in extraction library: {args.pmcid}", file=sys.stderr)
            return 2
        return print_article_result(
            analyze_article(
                selected[0],
                prior=prior,
                posterior_draw_count=args.posterior_draws,
            ),
            args.plot_dir,
        )

    if args.random:
        if not records:
            return 0
        selected = random.Random(args.seed).choice(records)
        return print_article_result(
            analyze_article(
                selected,
                prior=prior,
                posterior_draw_count=args.posterior_draws,
                random_seed=args.seed or 0,
            ),
            args.plot_dir,
        )

    summary = build_summary(
        records,
        prior=prior,
        posterior_draw_count=args.posterior_draws,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: {summary['fitted_arm_count']} fitted arm curves across "
        f"{summary['article_count']} articles."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
