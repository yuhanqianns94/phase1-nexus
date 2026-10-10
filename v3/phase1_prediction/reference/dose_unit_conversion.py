#!/usr/bin/env python3
"""Exact, dimension-preserving dose-unit conversion and extraction auditing.

This module deliberately supports only conversions that are mathematical
identities of the recorded unit.  It never substitutes a reference body weight,
body-surface area, concentration, molecular weight, potency, or dosing
schedule.  Consequently, for example:

* ``ug`` and ``mg`` can be converted to canonical ``mg``;
* ``ug/kg`` and ``mg/kg`` can be converted to canonical ``mg/kg``;
* ``mCi`` and ``MBq`` can be converted to canonical ``MBq``;
* ``mg``, ``mg/kg``, ``mg/m2``, ``mg/day``, ``mg/dose``, and ``AUC`` remain
  distinct.

The toxicity-curve pipeline uses the canonical value and canonical unit for
comparability.  Raw values and raw units remain in audit output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXTRACTION_DIR = Path("output/mtd_llm_extractions")
NON_EXTRACTION_SUFFIXES = (
    ".validation.json",
    ".run_metadata.json",
    ".raw_response.json",
    ".invalid.json",
    ".values.json",
    ".standardized.json",
)
UNSUPPORTED_UNITS = frozenset(
    {
        "",
        "other",
        "unknown",
        "unitless",
        "not_reported",
        "unclear",
        "not_applicable",
    }
)
SCIENTIFIC_SCALE_RE = re.compile(
    r"^(?:1\s*[x×]\s*)?10\^([+-]?\d+)\s*(.*)$", flags=re.IGNORECASE
)

# All factors convert the numerator to the indicated canonical numerator.  The
# suffix is preserved exactly after syntax/alias normalization, so mg/kg is
# never mixed with mg/m2 or mg/day.
MASS_TO_MG: tuple[tuple[str, float], ...] = (
    ("kg", 1e6),
    ("mg", 1.0),
    ("mcg", 1e-3),
    ("ug", 1e-3),
    ("ng", 1e-6),
    ("pg", 1e-9),
    ("g", 1e3),
)
ACTIVITY_TO_IU: tuple[tuple[str, float], ...] = (
    ("miu", 1e6),
    ("kiu", 1e3),
    ("iu", 1.0),
    ("u", 1.0),
)
RADIATION_TO_GY: tuple[tuple[str, float], ...] = (
    ("kgy", 1e3),
    ("gy", 1.0),
    ("cgy", 1e-2),
    ("mgy", 1e-3),
    ("rad", 1e-2),
)
RADIOACTIVITY_TO_MBQ: tuple[tuple[str, float], ...] = (
    ("tbq", 1e6),
    ("gbq", 1e3),
    ("mbq", 1.0),
    ("kbq", 1e-3),
    ("bq", 1e-6),
    # 1 Ci = 3.7e10 Bq, therefore 1 mCi = 37 MBq exactly.
    ("ci", 3.7e4),
    ("mci", 37.0),
    ("uci", 0.037),
    ("nci", 0.000037),
)
MOLAR_TO_MMOL: tuple[tuple[str, float], ...] = (
    ("mol", 1e3),
    ("mmol", 1.0),
    ("umol", 1e-3),
    ("nmol", 1e-6),
    ("pmol", 1e-9),
)
COUNT_UNITS = frozenset({"cells", "cfu", "pfu", "tcid50", "vp"})
PASSTHROUGH_UNITS = frozenset({"auc"})


@dataclass(frozen=True)
class DoseUnitInfo:
    """Normalized representation of one unit string.

    ``multiplier_to_canonical`` is always applied to the *numeric value*, not
    to the unit string.  An unrecognized nonempty unit is retained as a
    passthrough canonical unit so legacy data with an exact same unit can still
    be audited and compared; it is never considered convertible to a different
    spelling.
    """

    raw_unit: str | None
    normalized_unit: str | None
    canonical_unit: str | None
    multiplier_to_canonical: float
    category: str
    recognized: bool

    @property
    def conversion_applied(self) -> bool:
        return bool(
            self.canonical_unit
            and (
                not math.isclose(self.multiplier_to_canonical, 1.0)
                or self.normalized_unit != self.canonical_unit
            )
        )


def normalize_dose_unit_text(unit: str | None) -> str | None:
    """Normalize syntax and documented aliases without changing dimensions."""

    if not isinstance(unit, str) or not unit.strip():
        return None

    text = unicodedata.normalize("NFKC", unit).casefold().strip()
    text = (
        text.replace("μ", "u")
        .replace("µ", "u")
        .replace("−", "-")
        .replace("–", "-")
        .replace("⁄", "/")
        .replace("∕", "/")
        .replace("×", "x")
        .replace("㎡", "m2")
    )

    # Convert wording that the extraction schema explicitly defines as exact
    # aliases.  Longer phrases must be handled before shorter ones.
    replacements = (
        (r"\bmillion\s+international\s+units?\b", "miu"),
        (r"\bthousand\s+international\s+units?\b", "kiu"),
        (r"\binternational\s+units?\b", "iu"),
        (r"\bmegabecquerels?\b", "mbq"),
        (r"\bgigabecquerels?\b", "gbq"),
        (r"\bkilobecquerels?\b", "kbq"),
        (r"\bbecquerels?\b", "bq"),
        (r"\bmillicuries?\b", "mci"),
        (r"\bmicrocuries?\b", "uci"),
        (r"\bcuries?\b", "ci"),
        (r"\bcentigrays?\b", "cgy"),
        (r"\bmilligrays?\b", "mgy"),
        (r"\bgrays?\b", "gy"),
        (r"\bcolony[-\s]*forming\s+units?\b", "cfu"),
        (r"\bplaque[-\s]*forming\s+units?\b", "pfu"),
        (r"\btissue[-\s]*culture[-\s]*infectious[-\s]*dose[-\s]*50\b", "tcid50"),
        (r"\bviral\s+particles?\b", "vp"),
        (r"\bvirus\s+particles?\b", "vp"),
        (r"\bmicrograms?\b", "ug"),
        (r"\bmilligrams?\b", "mg"),
        (r"\bnanograms?\b", "ng"),
        (r"\bpicograms?\b", "pg"),
        (r"\bkilograms?\b", "kg"),
        (r"\bgrams?\b", "g"),
        (r"\bmicromoles?\b", "umol"),
        (r"\bmillimoles?\b", "mmol"),
        (r"\bnanomoles?\b", "nmol"),
        (r"\bpicomoles?\b", "pmol"),
        (r"\bmoles?\b", "mol"),
        (r"\bsquare\s+met(?:er|re)s?\b", "m2"),
        (r"\bsq\.?\s*m\.?\b", "m2"),
        (r"\bmet(?:er|re)s?\s+squared\b", "m2"),
        (r"\bday(s)?\b", "day"),
        (r"\bweek(s)?\b", "week"),
        (r"\bhour(s)?\b", "h"),
        (r"\bminute(s)?\b", "min"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    # Common unit spellings written using an exponent instead of the extraction
    # schema's m2 form.
    text = re.sub(r"\bm\s*\^\s*2\b", "m2", text)
    text = re.sub(r"\bm\s*-\s*2\b", "m2", text)
    text = re.sub(r"\b10\s*\^\s*([+-]?\d+)", r"10^\1", text)
    text = re.sub(r"\bper\b", "/", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _split_prefixed_unit(
    text: str, factors: tuple[tuple[str, float], ...], canonical_numerator: str
) -> tuple[str, float] | None:
    """Return a canonical unit and scale when a numerator plus suffix matches."""

    for symbol, factor in factors:
        if text == symbol:
            return canonical_numerator, factor
        prefix = f"{symbol}/"
        if text.startswith(prefix):
            return f"{canonical_numerator}/{text[len(prefix):]}", factor
    return None


def _unit_category(canonical_unit: str, family: str) -> str:
    suffix = canonical_unit.partition("/")[2]
    return family if not suffix else f"{family}_per_{suffix.replace('/', '_per_')}"


def describe_dose_unit(unit: str | None) -> DoseUnitInfo:
    """Parse one dose unit into a safe canonical representation.

    The method performs exact SI-prefix and named physical-unit conversions
    only.  It preserves every denominator term, including ``/kg``, ``/m2``,
    ``/day``, and ``/dose``.
    """

    normalized = normalize_dose_unit_text(unit)
    raw = unit if isinstance(unit, str) else None
    if normalized is None or normalized in UNSUPPORTED_UNITS:
        return DoseUnitInfo(
            raw_unit=raw,
            normalized_unit=normalized,
            canonical_unit=None,
            multiplier_to_canonical=1.0,
            category="unsupported",
            recognized=False,
        )

    source_normalized = normalized
    multiplier = 1.0
    scaled = SCIENTIFIC_SCALE_RE.match(normalized)
    if scaled:
        multiplier *= 10.0 ** int(scaled.group(1))
        normalized = scaled.group(2).strip()
        if not normalized:
            return DoseUnitInfo(
                raw_unit=raw,
                normalized_unit=None,
                canonical_unit=None,
                multiplier_to_canonical=1.0,
                category="unsupported",
                recognized=False,
            )
    parse_unit = normalized

    for factors, canonical_numerator, family in (
        (ACTIVITY_TO_IU, "iu", "international_activity"),
        (RADIATION_TO_GY, "gy", "absorbed_radiation"),
        (RADIOACTIVITY_TO_MBQ, "mbq", "radioactivity"),
        (MOLAR_TO_MMOL, "mmol", "amount_of_substance"),
        (MASS_TO_MG, "mg", "mass"),
    ):
        matched = _split_prefixed_unit(parse_unit, factors, canonical_numerator)
        if matched is not None:
            canonical, factor = matched
            return DoseUnitInfo(
                raw_unit=raw,
                normalized_unit=source_normalized,
                canonical_unit=canonical,
                multiplier_to_canonical=multiplier * factor,
                category=_unit_category(canonical, family),
                recognized=True,
            )

    for count_unit in COUNT_UNITS:
        if parse_unit == count_unit:
            canonical = count_unit
        elif parse_unit.startswith(f"{count_unit}/"):
            canonical = parse_unit
        else:
            continue
        return DoseUnitInfo(
            raw_unit=raw,
            normalized_unit=source_normalized,
            canonical_unit=canonical,
            multiplier_to_canonical=multiplier,
            category=_unit_category(canonical, "count"),
            recognized=True,
        )

    if parse_unit in PASSTHROUGH_UNITS:
        return DoseUnitInfo(
            raw_unit=raw,
            normalized_unit=source_normalized,
            canonical_unit=parse_unit,
            multiplier_to_canonical=multiplier,
            category="pharmacokinetic_target",
            recognized=True,
        )

    # Preserve a same-spelled legacy/unit-specific unit for backwards
    # compatibility, but do not allow it to numerically convert to another
    # spelling.  This does not claim that the unit is clinically interpretable.
    return DoseUnitInfo(
        raw_unit=raw,
        normalized_unit=source_normalized,
        canonical_unit=normalized,
        multiplier_to_canonical=multiplier,
        category="unrecognized_passthrough",
        recognized=False,
    )


def parse_dose_unit(unit: str | None) -> tuple[str | None, float]:
    """Return ``(canonical_unit, multiplier_to_canonical)`` for callers."""

    info = describe_dose_unit(unit)
    return info.canonical_unit, info.multiplier_to_canonical


def parse_scaled_unit(unit: str | None) -> tuple[str | None, float]:
    """Backward-compatible alias for historical callers."""

    return parse_dose_unit(unit)


def canonical_dose_unit(unit: str | None) -> str | None:
    """Return the unit family used as the toxicity-curve comparison key."""

    return describe_dose_unit(unit).canonical_unit


def canonical_dose_value(
    value: float | None, unit: str | None
) -> tuple[float | None, str | None]:
    """Convert an amount to its canonical-unit numeric value."""

    info = describe_dose_unit(unit)
    if value is None or isinstance(value, bool):
        return None, info.canonical_unit
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None, info.canonical_unit
    if not math.isfinite(numeric_value) or info.canonical_unit is None:
        return None, info.canonical_unit
    return numeric_value * info.multiplier_to_canonical, info.canonical_unit


def units_are_exactly_convertible(left: str | None, right: str | None) -> bool:
    """Whether two units can be converted without external clinical assumptions."""

    left_info = describe_dose_unit(left)
    right_info = describe_dose_unit(right)
    if (
        left_info.canonical_unit is None
        or right_info.canonical_unit is None
        or left_info.canonical_unit != right_info.canonical_unit
    ):
        return False
    # Unknown units are comparable only when normalization made their spelling
    # exactly identical; known units can use their declared numeric factors.
    return (
        (left_info.recognized and right_info.recognized)
        or left_info.normalized_unit == right_info.normalized_unit
    )


def units_match(left: str | None, right: str | None) -> bool:
    """Backward-compatible comparison name used by older code."""

    return units_are_exactly_convertible(left, right)


def convert_dose_value(
    value: float | None, from_unit: str | None, to_unit: str | None
) -> float | None:
    """Convert a numeric dose only when the two units are exactly compatible.

    ``None`` means that a conversion would require a non-numeric clinical
    assumption or that the input is invalid.
    """

    if not units_are_exactly_convertible(from_unit, to_unit):
        return None
    from_info = describe_dose_unit(from_unit)
    to_info = describe_dose_unit(to_unit)
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value * from_info.multiplier_to_canonical / to_info.multiplier_to_canonical


def conversion_metadata(unit: str | None) -> dict[str, Any]:
    """Return JSON-ready provenance for a source unit's canonicalization."""

    info = describe_dose_unit(unit)
    return {
        "source_unit": info.raw_unit,
        "normalized_source_unit": info.normalized_unit,
        "canonical_unit": info.canonical_unit,
        "multiplier_to_canonical": info.multiplier_to_canonical,
        "category": info.category,
        "recognized_exact_conversion": info.recognized,
        "conversion_applied": info.conversion_applied,
    }


def _unwrap_field(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value and (
        "evidence" in value or "status" in value
    ):
        return value.get("value")
    return value


def iter_extraction_units(data: dict[str, Any]) -> Iterable[tuple[str, str | None]]:
    """Yield unit-bearing fields from a compact extraction for audit."""

    for field_name in ("MTD_Unit", "MAD_Unit"):
        value = _unwrap_field(data.get(field_name))
        yield field_name, value if isinstance(value, str) else None

    for collection_name in ("dose_levels", "MTD_components", "MAD_components"):
        rows = data.get(collection_name) or []
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            value = _unwrap_field(row.get("Dose_Unit"))
            yield (
                f"{collection_name}[{index}].Dose_Unit",
                value if isinstance(value, str) else None,
            )

    # The current schema does not require this field on DLT rows, but audit it
    # when older/manual extractions happen to provide it.
    for index, row in enumerate(data.get("dlt_count_at_each_dose_level") or []):
        if not isinstance(row, dict) or "Dose_Unit" not in row:
            continue
        value = _unwrap_field(row.get("Dose_Unit"))
        yield (
            f"dlt_count_at_each_dose_level[{index}].Dose_Unit",
            value if isinstance(value, str) else None,
        )


def audit_extraction_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw units and exactly convertible canonical-unit groups."""

    raw_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    units_by_canonical: dict[str, Counter[str]] = defaultdict(Counter)
    info_by_raw: dict[str, DoseUnitInfo] = {}
    record_count = 0
    unit_occurrence_count = 0

    for record in records:
        record_count += 1
        data = record.get("data", record)
        if not isinstance(data, dict):
            continue
        for field_path, raw_unit in iter_extraction_units(data):
            if raw_unit is None:
                continue
            unit_occurrence_count += 1
            raw = raw_unit.strip()
            raw_counts[raw] += 1
            field_counts[field_path.split("[", 1)[0]] += 1
            info = describe_dose_unit(raw)
            info_by_raw.setdefault(raw, info)
            category_counts[info.category] += 1
            if info.canonical_unit is not None:
                canonical_counts[info.canonical_unit] += 1
                units_by_canonical[info.canonical_unit][raw] += 1

    def counted(counter: Counter[str], key_name: str = "unit") -> list[dict[str, Any]]:
        return [
            {key_name: key, "count": count}
            for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ]

    variant_groups = []
    for canonical_unit, variants in sorted(units_by_canonical.items()):
        variant_rows = []
        for raw_unit, count in sorted(variants.items(), key=lambda item: (-item[1], item[0])):
            info = info_by_raw[raw_unit]
            variant_rows.append(
                {
                    "raw_unit": raw_unit,
                    "count": count,
                    "normalized_unit": info.normalized_unit,
                    "multiplier_to_canonical": info.multiplier_to_canonical,
                    "category": info.category,
                    "recognized_exact_conversion": info.recognized,
                    "conversion_applied": info.conversion_applied,
                }
            )
        variant_groups.append(
            {
                "canonical_unit": canonical_unit,
                "occurrence_count": canonical_counts[canonical_unit],
                "raw_variants": variant_rows,
            }
        )

    unsupported = [
        {
            "raw_unit": raw_unit,
            "count": count,
            "normalized_unit": info_by_raw[raw_unit].normalized_unit,
            "category": info_by_raw[raw_unit].category,
        }
        for raw_unit, count in sorted(raw_counts.items(), key=lambda item: (-item[1], item[0]))
        if info_by_raw[raw_unit].canonical_unit is None
        or not info_by_raw[raw_unit].recognized
    ]

    return {
        "record_count": record_count,
        "unit_occurrence_count": unit_occurrence_count,
        "raw_unit_counts": counted(raw_counts, "raw_unit"),
        "canonical_unit_counts": counted(canonical_counts, "canonical_unit"),
        "category_counts": counted(category_counts, "category"),
        "field_collection_counts": counted(field_counts, "field_collection"),
        "canonical_variant_groups": variant_groups,
        "unsupported_or_unrecognized_units": unsupported,
    }


def audit_eligible_dlt_evidence(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count DLT rows that reach the main pipeline after canonicalization.

    The import is intentionally local: :mod:`dlt_evidence` itself imports this
    module for conversion metadata.
    """

    from .dlt_evidence import build_candidate_rows

    source_unit_counts: Counter[str] = Counter()
    canonical_unit_counts: Counter[str] = Counter()
    converted_source_unit_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    article_count = 0
    eligible_article_count = 0
    eligible_row_count = 0

    for record in records:
        article_count += 1
        data = record.get("data", record)
        if not isinstance(data, dict):
            continue
        rows, reasons = build_candidate_rows(data)
        skip_reason_counts.update(reasons)
        if rows:
            eligible_article_count += 1
        for row in rows:
            eligible_row_count += 1
            source_unit = str(row["dose_unit"])
            source_unit_counts[source_unit] += 1
            canonical_unit_counts[str(row["canonical_unit"])] += 1
            if row["unit_conversion"]["conversion_applied"]:
                converted_source_unit_counts[source_unit] += 1

    def counted(counter: Counter[str], key_name: str) -> list[dict[str, Any]]:
        return [
            {key_name: key, "count": count}
            for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ]

    return {
        "article_count": article_count,
        "articles_with_eligible_dlt_rows": eligible_article_count,
        "eligible_dlt_row_count": eligible_row_count,
        "source_unit_counts": counted(source_unit_counts, "source_unit"),
        "canonical_unit_counts": counted(canonical_unit_counts, "canonical_unit"),
        "rows_requiring_canonicalization": counted(
            converted_source_unit_counts, "source_unit"
        ),
        "skip_reason_counts": counted(skip_reason_counts, "reason"),
    }


def load_compact_records(
    extraction_dir: Path, pattern: str = "*_compact.json"
) -> list[dict[str, Any]]:
    """Load compact extraction JSON solely for the unit-audit CLI."""

    records: list[dict[str, Any]] = []
    for path in sorted(extraction_dir.glob(pattern)):
        if not path.is_file() or any(path.name.endswith(suffix) for suffix in NON_EXTRACTION_SUFFIXES):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and not data.get("skipped_extraction"):
            records.append({"pmcid": path.stem, "path": str(path), "data": data})
    return records


def format_audit_summary(audit: dict[str, Any]) -> str:
    """Render a concise terminal report while preserving full detail in JSON."""

    lines = [
        "Dose-unit conversion audit",
        f"Records: {audit['record_count']}; unit occurrences: {audit['unit_occurrence_count']}",
        "",
        "Canonical exact-conversion groups:",
    ]
    for group in audit["canonical_variant_groups"]:
        variants = ", ".join(
            f"{variant['raw_unit']} ({variant['count']})"
            for variant in group["raw_variants"]
        )
        lines.append(
            f"  {group['canonical_unit']}: {group['occurrence_count']} [{variants}]"
        )
    if audit["unsupported_or_unrecognized_units"]:
        lines.extend(("", "Not safely convertible / unrecognized:"))
        for item in audit["unsupported_or_unrecognized_units"]:
            lines.append(
                f"  {item['raw_unit']}: {item['count']} ({item['category']})"
            )
    dlt_audit = audit.get("eligible_dlt_evidence")
    if isinstance(dlt_audit, dict):
        lines.extend(
            (
                "",
                "Eligible dose-level DLT evidence after canonicalization:",
                (
                    f"  articles={dlt_audit['articles_with_eligible_dlt_rows']}; "
                    f"rows={dlt_audit['eligible_dlt_row_count']}"
                ),
                "  Source units requiring canonicalization: "
                + ", ".join(
                    f"{item['source_unit']} ({item['count']})"
                    for item in dlt_audit["rows_requiring_canonicalization"]
                ),
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit compact-extraction dose units and report exact, "
            "dimension-preserving canonical conversions."
        )
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
        help="Optional JSON audit output path.",
    )
    parser.add_argument(
        "--include-dlt-evidence",
        action="store_true",
        help=(
            "Also count dose-level DLT rows eligible for the main toxicity "
            "pipeline after canonicalization."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_compact_records(args.extraction_dir, args.pattern)
    if not records:
        raise SystemExit(f"No compact extraction records found in {args.extraction_dir}")
    audit = audit_extraction_records(records)
    if args.include_dlt_evidence:
        audit["eligible_dlt_evidence"] = audit_eligible_dlt_evidence(records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
    print(format_audit_summary(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
