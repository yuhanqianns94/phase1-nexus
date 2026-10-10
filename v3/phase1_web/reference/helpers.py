"""Pure helpers shared by the local toxicity Streamlit interface.

This module deliberately has no Streamlit dependency so it can be tested and
reused by local callers without starting a web server.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping




from phase1_project.runtime import WORKSPACE_ROOT, OUTPUT_ROOT
PROJECT_ROOT = WORKSPACE_ROOT
DEFAULT_EXTRACTION_LIBRARY = OUTPUT_ROOT / "mtd_llm_extractions"
DEFAULT_BAYESIAN_CACHE = (
    OUTPUT_ROOT / "mtd_similarity" / "bayesian_historical_curve_cache.json"
)
DEFAULT_V2_EXTRACTION_LIBRARY = (
    OUTPUT_ROOT / "phase1_extractions_v2"
)
DEFAULT_V2_BAYESIAN_CACHE = (
    PROJECT_ROOT
    / "output"
    / "mtd_similarity"
    / "bayesian_historical_curve_cache_v2.json"
)
DEFAULT_EMBEDDING_CACHE = (
    OUTPUT_ROOT / "mtd_similarity" / "paragraph_embedding_cache.json"
)
DEFAULT_STAGING_WORKSPACE = OUTPUT_ROOT / "external_staging"
DEFAULT_PRIOR_CATALOG = Path(__file__).resolve().parents[2] / "phase1_prediction" / "reference" / "bayesian_prior_candidates.json"
DEFAULT_MANUAL_SIMILARITY_PARAMETERS = (
    Path(__file__).resolve().parents[2] / "phase1_prediction" / "reference" / "manual_similarity_parameters.json"
)

from phase1_prediction.profile_similarity import LABELS as SIMILARITY_FIELD_LABELS, normalize_weights as normalize_profile_weights
SIMILARITY_FIELD_ORDER = list(SIMILARITY_FIELD_LABELS)


AGE_GROUP_OPTIONS = [
    "adult",
    "children",
    "adolescent",
    "adult_and_children",
    "not_specified",
    "unknown",
]
GENDER_SEX_OPTIONS = ["all", "male", "female", "not_specified", "unknown"]
ADMIN_ROUTE_OPTIONS = [
    "oral",
    "intravenous",
    "subcutaneous",
    "intramuscular",
    "intradermal",
    "intraperitoneal",
    "intra-arterial",
    "hepatic arterial infusion",
    "intrathecal",
    "intratumoral",
    "intralesional",
    "intravesical",
    "intrapleural",
    "topical",
    "transdermal",
    "inhalation",
    "intranasal",
    "rectal",
    "vaginal",
    "radiation",
    "local implant",
    "regional perfusion",
    "cell infusion",
    "other",
    "unknown",
]
DOSE_FORM_OPTIONS = [
    "tablet",
    "capsule",
    "oral solution",
    "oral suspension",
    "oral powder/granule",
    "injection",
    "solution for injection/infusion",
    "powder for injection/infusion",
    "concentrate for solution for infusion",
    "suspension for injection",
    "implant",
    "cell therapy product",
    "gene/vector therapy product",
    "oncolytic virus preparation",
    "cancer vaccine preparation",
    "radiopharmaceutical",
    "radiation",
    "regimen",
    "not applicable",
    "other",
    "unknown",
]
FREQUENCY_OPTIONS = [
    "QD",
    "BID",
    "TID",
    "QID",
    "weekly",
    "twice weekly",
    "three times weekly",
    "Q2W",
    "Q3W",
    "Q4W",
    "Q48H",
    "Q72H",
    "Q96H",
    "continuous",
    "PRN",
    "other",
    "unknown",
]


def default_paths() -> dict[str, str]:
    """Return editable local-only path defaults for the interface."""

    return {
        "extraction_library": str(DEFAULT_EXTRACTION_LIBRARY),
        "bayesian_cache": str(DEFAULT_BAYESIAN_CACHE),
        "v2_extraction_library": str(DEFAULT_V2_EXTRACTION_LIBRARY),
        "v2_bayesian_cache": str(DEFAULT_V2_BAYESIAN_CACHE),
        "embedding_cache": str(DEFAULT_EMBEDDING_CACHE),
        "staging_workspace": str(DEFAULT_STAGING_WORKSPACE),
    }


def count_builtin_extractions(extraction_dir: Path | None = None) -> int:
    """Count built-in compact extraction records in the default working library."""

    directory = Path(extraction_dir or DEFAULT_EXTRACTION_LIBRARY)
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.glob("*_compact.json") if path.is_file())




def load_default_similarity_parameters(
    path: Path = DEFAULT_MANUAL_SIMILARITY_PARAMETERS,
    *,
    use_manual: bool = True,
) -> tuple[dict[str, float], float]:
    """Load the project default similarity weights and threshold."""

    if not use_manual:
        from phase1_prediction.reference.toxicity_similarity import FIELD_WEIGHTS

        return normalize_profile_weights(FIELD_WEIGHTS), 0.0
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_weights = payload.get("field_weights")
    if not isinstance(raw_weights, dict):
        raise ValueError(f"Similarity parameter file lacks field_weights: {path}")
    weights = normalize_profile_weights(raw_weights)
    threshold = float(payload.get("similarity_threshold", 0.0))
    return weights, threshold


def validate_similarity_inputs(
    field_weights: Mapping[str, float],
    similarity_threshold: float,
) -> tuple[dict[str, float], float]:
    """Validate UI-entered similarity weights and threshold.

    Weights must already sum to 1. They are not renormalized here.
    """

    missing = [field for field in SIMILARITY_FIELD_ORDER if field not in field_weights]
    if missing:
        raise ValueError(
            "Missing similarity weights for: " + ", ".join(missing)
        )
    cleaned: dict[str, float] = {}
    for field in SIMILARITY_FIELD_ORDER:
        value = field_weights[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Similarity weight for {field} must be a number.")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(
                f"Similarity weight for {field} must be a finite non-negative number."
            )
        cleaned[field] = numeric
    total = sum(cleaned.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(
            f"Similarity weights must sum to exactly 1 (current sum = {total:.5f})."
        )
    if (
        isinstance(similarity_threshold, bool)
        or not isinstance(similarity_threshold, (int, float))
        or not math.isfinite(float(similarity_threshold))
        or not 0 <= float(similarity_threshold) <= 1
    ):
        raise ValueError("Similarity threshold must be a number from 0 to 1.")
    return cleaned, float(similarity_threshold)


def parse_similarity_weight(value: Any) -> Fraction:
    """Parse one non-negative decimal or fraction weight exactly."""

    if isinstance(value, bool):
        raise ValueError("must be a decimal or fraction such as 0.2 or 1/5")
    if isinstance(value, Fraction):
        parsed = value
    elif isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("must be finite")
        parsed = Fraction(str(value))
    elif isinstance(value, str):
        cleaned = re.sub(r"\s+", "", value)
        if not cleaned:
            raise ValueError("cannot be blank")
        try:
            parsed = Fraction(cleaned)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                "must be a decimal or fraction such as 0.2 or 1/5"
            ) from error
    else:
        raise ValueError("must be a decimal or fraction such as 0.2 or 1/5")
    if parsed < 0:
        raise ValueError("must be non-negative")
    return parsed


def parse_similarity_weight_inputs(
    field_weights: Mapping[str, Any],
) -> tuple[dict[str, float], Fraction]:
    """Parse every UI weight and return floats plus the exact rational total."""

    missing = [field for field in SIMILARITY_FIELD_ORDER if field not in field_weights]
    if missing:
        raise ValueError("Missing similarity weights for: " + ", ".join(missing))
    parsed: dict[str, Fraction] = {}
    for field in SIMILARITY_FIELD_ORDER:
        try:
            parsed[field] = parse_similarity_weight(field_weights[field])
        except ValueError as error:
            raise ValueError(
                f"{SIMILARITY_FIELD_LABELS[field]} weight {error}."
            ) from error
    return (
        {field: float(value) for field, value in parsed.items()},
        sum(parsed.values(), Fraction(0, 1)),
    )


def format_similarity_weight_input(value: float) -> str:
    """Format a default weight as a concise exact fraction when appropriate."""

    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError("Default similarity weights must be finite and non-negative.")
    fraction = Fraction(numeric).limit_denominator(1000)
    if abs(float(fraction) - numeric) <= 1e-10:
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{numeric:.12g}"


def pick_local_directory() -> str | None:
    """Open a native folder dialog without Tk (safe for Streamlit on macOS).

    Tk crashes Streamlit on macOS because NSWindow must run on the main thread.
    Prefer AppleScript on Darwin and zenity elsewhere.
    """

    import platform
    import shutil
    import subprocess

    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'POSIX path of (choose folder with prompt "Select a local document folder")',
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return None
            selected = result.stdout.strip()
            return selected.rstrip("/") if selected else None
        if system == "Linux" and shutil.which("zenity"):
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return None
            selected = result.stdout.strip()
            return selected or None
    except Exception:
        return None
    return None


def friendly_library_label(path: Path | None = None) -> str:
    """Return a human-readable label for the built-in working library."""

    target = Path(path or DEFAULT_EXTRACTION_LIBRARY).resolve()
    builtin = DEFAULT_EXTRACTION_LIBRARY.resolve()
    if target == builtin:
        return "Built-in Phase I literature library"
    return "Custom literature library"


def parse_positive_doses(text: str) -> list[float]:
    """Parse comma, whitespace, or newline-separated positive finite doses.

    An empty string is allowed and means the automatic historical dose grid
    should be used.
    """

    tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]
    if not tokens:
        return []
    doses: list[float] = []
    for token in tokens:
        try:
            dose = float(token)
        except ValueError as error:
            raise ValueError(f"Invalid dose: {token!r}") from error
        if not math.isfinite(dose) or dose <= 0:
            raise ValueError(f"Dose must be positive and finite: {token!r}")
        doses.append(dose)
    return sorted(set(doses))


def parse_probability_list(
    text: str,
    *,
    expected_count: int | None = None,
) -> list[float]:
    """Parse an ordered comma/space/newline-separated probability vector."""

    tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]
    if not tokens:
        return []
    values: list[float] = []
    for token in tokens:
        cleaned = token[:-1] if token.endswith("%") else token
        try:
            value = float(cleaned)
        except ValueError as error:
            raise ValueError(f"Invalid probability: {token!r}") from error
        if token.endswith("%"):
            value /= 100.0
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                f"Probability must be between 0 and 1 (or use %): {token!r}"
            )
        values.append(value)
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"Expected {expected_count} probabilities, received {len(values)}."
        )
    return values


def split_drug_names(drug: str) -> list[str]:
    """Split a displayed drug regimen into names for similarity scoring."""

    names = [part.strip() for part in re.split(r"[,;+]", drug) if part.strip()]
    return names or [drug.strip()]


def split_term_list(text: str) -> list[str]:
    """Split a free-text list field on commas, semicolons, or newlines."""

    return [part.strip() for part in re.split(r"[,;\n]+", text) if part.strip()]


def parse_dlt_duration_text(text: str) -> dict[str, Any] | None:
    """Parse an optional DLT duration such as ``>7 days`` or ``2 cycles``."""

    cleaned = text.strip()
    if not cleaned:
        return None
    match = re.fullmatch(
        r"(>=|<=|>|<|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*"
        r"(hours?|days?|weeks?|cycles?)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            "DLT duration must look like '>7 days', '48 hours', "
            "'2 weeks', or '1 cycle'."
        )
    duration_value = float(match.group(2))
    if duration_value <= 0:
        raise ValueError("DLT duration must be positive.")
    if duration_value.is_integer():
        duration_value = int(duration_value)
    unit = match.group(3).lower()
    if unit.endswith("s"):
        unit = unit[:-1]
    return {
        "operator": match.group(1) or "=",
        "duration_value": duration_value,
        "duration_unit": unit,
    }


def parse_dlt_definition_text(text: str) -> list[dict[str, Any]]:
    """Parse one DLT definition per line, with optional duration metadata."""

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        parts = [part.strip() for part in cleaned.split(";")]
        name = parts[0] if parts else cleaned
        grade_text = parts[1] if len(parts) > 1 else ""
        # Keep the legacy three-column form (name; grade; comments). The new
        # four-column form is name; grade; duration; comments.
        duration_text = parts[2] if len(parts) > 3 else ""
        comments = "; ".join(parts[3:]) if len(parts) > 3 else (
            parts[2] if len(parts) > 2 else ""
        )
        grade_match = re.search(
            r"(>=|<=|>|<|=)?\s*([0-5](?:\.[0-9]+)?)",
            grade_text,
        )
        if grade_match:
            operator = grade_match.group(1) or ">="
            grade_value = float(grade_match.group(2))
            if grade_value.is_integer():
                grade_value = int(grade_value)
            grade = {
                "operator": operator,
                "grade_value": grade_value,
                "grade_min": None,
                "grade_max": None,
            }
        else:
            grade = {
                "operator": ">=",
                "grade_value": 3,
                "grade_min": None,
                "grade_max": None,
            }
            if grade_text and not comments:
                comments = grade_text
        rows.append(
            {
                "DLT_Name": name,
                "DLT_definition_Grade": grade,
                "DLT_definition_Duration": (
                    parse_dlt_duration_text(duration_text) or "not_reported"
                ),
                "DLT_definition_comments": comments or "user supplied",
            }
        )
    return rows


def parse_administration_days(text: str) -> list[int]:
    """Parse cycle administration days such as ``1, 8`` or ``1 8 15``."""

    tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]
    if not tokens:
        return []
    days: list[int] = []
    for token in tokens:
        try:
            day = int(token)
        except ValueError as error:
            raise ValueError(
                f"Administration days must be positive integers: {token!r}"
            ) from error
        if day <= 0:
            raise ValueError(
                f"Administration days must be positive integers: {token!r}"
            )
        days.append(day)
    return sorted(set(days))


def build_schedule_dose_levels(
    *,
    frequency: str = "",
    cycle_length_days: float | None = None,
    administration_days_text: str = "",
) -> list[dict[str, Any]]:
    """Build minimal ``dose_levels`` rows used by schedule similarity."""

    frequency_text = frequency.strip()
    if frequency_text in {"", "(blank)"}:
        frequency_text = ""
    interval = None
    if cycle_length_days is not None and float(cycle_length_days) > 0:
        interval = float(cycle_length_days)
    admin_days = parse_administration_days(administration_days_text)
    if not frequency_text and interval is None and not admin_days:
        return []
    schedule: dict[str, Any] = {
        "cycle_length_days": interval,
        "administration_days": admin_days or None,
    }
    row: dict[str, Any] = {
        "Schedule": schedule,
        "is_actual_escalation_dose": True,
    }
    if frequency_text:
        row["Frequency"] = {"frequency_standard": frequency_text}
    return [row]


def build_dlt_eva_field(
    *,
    duration_days: float | None = None,
    cycle_number: float | None = None,
    text: str = "",
) -> dict[str, Any] | None:
    """Build a ``DLT_EVA`` value usable by ``dlt_eva`` similarity."""

    text_value = text.strip()
    days = (
        float(duration_days)
        if duration_days is not None and float(duration_days) > 0
        else None
    )
    cycles = (
        float(cycle_number)
        if cycle_number is not None and float(cycle_number) > 0
        else None
    )
    if days is None and cycles is None and not text_value:
        return None
    if days is not None:
        status = "direct_day_value"
    elif cycles is not None:
        status = "cycle_length_missing"
    else:
        status = "not_reported"
    return {
        "value": {
            "dlt_eval_text": text_value or None,
            "dlt_eval_duration_days": days,
            "dlt_eval_cycle_number": cycles,
            "normalization_status": status,
        },
        "evidence": "user supplied",
    }


def build_optional_query_fields(
    *,
    cancer_type: str = "",
    routes: list[str] | None = None,
    schema: str = "",
    inclusion: str = "",
    exclusion: str = "",
    dlt_definition: str = "",
    mechanism: str = "",
    gene_targets: str = "",
    age_group: str = "",
    age_min: float | None = None,
    age_max: float | None = None,
    age_unit: str = "years",
    gender: str = "",
    dose_forms: list[str] | None = None,
    schedule_frequency: str = "",
    schedule_cycle_days: float | None = None,
    schedule_administration_days: str = "",
    dlt_eva_days: float | None = None,
    dlt_eva_cycles: float | None = None,
    dlt_eva_text: str = "",
) -> dict[str, Any]:
    """Convert UI optional fields into extraction-compatible query values."""

    fields: dict[str, Any] = {}
    cancers = split_term_list(cancer_type)
    if cancers:
        fields["cancer_type"] = {
            "value": [
                {
                    "mention": cancer,
                    "core_term": cancer,
                    "modifiers": [],
                }
                for cancer in cancers
            ],
            "evidence": "user supplied",
        }
    if routes:
        fields["Admin_Route"] = list(routes)
    schema_text = schema.strip()
    if schema_text:
        fields["schema"] = schema_text
    inclusion_text = inclusion.strip()
    if inclusion_text:
        fields["Trial_Inclusion"] = inclusion_text
    exclusion_text = exclusion.strip()
    if exclusion_text:
        fields["Trial_Exclusion"] = exclusion_text
    dlt_rows = parse_dlt_definition_text(dlt_definition)
    if dlt_rows:
        fields["DLT_definition"] = dlt_rows
    mechanisms = split_term_list(mechanism)
    if mechanisms:
        fields["Mechanism_of_action"] = mechanisms
    targets = split_term_list(gene_targets)
    if targets:
        fields["Drug_target_gene_target"] = targets
    cleaned_age_group = age_group if age_group != "(blank)" else ""
    cleaned_age_unit = age_unit.strip().lower() or "years"
    if cleaned_age_unit not in {"years", "months", "days"}:
        raise ValueError("Age unit must be years, months, or days.")
    for label, value in (("Minimum age", age_min), ("Maximum age", age_max)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"{label} must be a non-negative finite number.")
    if age_min is not None and age_max is not None and float(age_max) < float(age_min):
        raise ValueError("Maximum age must be greater than or equal to minimum age.")
    if cleaned_age_group or age_min is not None or age_max is not None:
        fields["Age"] = {
            "value": {
                "age_min": float(age_min) if age_min is not None else None,
                "age_max": float(age_max) if age_max is not None else None,
                "age_unit": (
                    cleaned_age_unit
                    if age_min is not None or age_max is not None
                    else None
                ),
                "age_group": cleaned_age_group or None,
            },
            "evidence": "user supplied",
        }
    if gender and gender != "(blank)":
        fields["Gender_Sex"] = gender
    if dose_forms:
        fields["Dose_Form"] = list(dose_forms)
    dose_levels = build_schedule_dose_levels(
        frequency=schedule_frequency,
        cycle_length_days=schedule_cycle_days,
        administration_days_text=schedule_administration_days,
    )
    if dose_levels:
        fields["dose_levels"] = dose_levels
    dlt_eva = build_dlt_eva_field(
        duration_days=dlt_eva_days,
        cycle_number=dlt_eva_cycles,
        text=dlt_eva_text,
    )
    if dlt_eva is not None:
        fields["DLT_EVA"] = dlt_eva
    return fields


def build_target_query(
    *,
    drug: str,
    trial_type: str,
    dose_unit: str,
    planned_doses_text: str,
    optional_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pre-trial-only query accepted by ``prediction_service``."""

    drug = drug.strip()
    dose_unit = dose_unit.strip()
    normalized_trial_type = trial_type.strip().lower()
    if not drug:
        raise ValueError("Drug or regimen is required.")
    if normalized_trial_type not in {"single", "combination"}:
        raise ValueError("Trial type must be single or combination.")
    if not dose_unit:
        raise ValueError("Dose unit is required.")

    query: dict[str, Any] = {
        "toxicity_curve_drug": drug,
        "Dose_Unit": dose_unit,
        "Trial_Type": {
            "value": normalized_trial_type.capitalize(),
            "evidence": "user supplied",
        },
        "Drug_Name": split_drug_names(drug),
    }
    planned_doses = parse_positive_doses(planned_doses_text)
    if planned_doses:
        query["planned_doses"] = planned_doses
    for field_name, value in (optional_fields or {}).items():
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                query[field_name] = cleaned
        elif value not in (None, [], {}):
            query[field_name] = value
    dose_levels = query.get("dose_levels")
    if isinstance(dose_levels, list):
        enriched_levels: list[dict[str, Any]] = []
        for row in dose_levels:
            if not isinstance(row, dict):
                continue
            enriched = dict(row)
            enriched.setdefault("Drug_Name", query["Drug_Name"][0] if query["Drug_Name"] else drug)
            enriched.setdefault("Dose_Unit", dose_unit)
            enriched_levels.append(enriched)
        if enriched_levels:
            query["dose_levels"] = enriched_levels
    return query


def load_prior_catalog(path: Path = DEFAULT_PRIOR_CATALOG) -> dict[str, dict[str, Any]]:
    """Load Bayesian prior definitions keyed by name."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    priors = payload.get("priors")
    if not isinstance(priors, dict) or not priors:
        raise ValueError(f"Prior catalog has no priors: {path}")
    return {str(name): dict(prior) for name, prior in priors.items()}


def load_default_prior_name(path: Path = DEFAULT_PRIOR_CATALOG) -> str:
    """Return the catalog-declared default prior after validating its identity."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    priors = payload.get("priors")
    default_prior = payload.get("default_prior")
    if not isinstance(priors, dict) or not priors:
        raise ValueError(f"Prior catalog has no priors: {path}")
    if not isinstance(default_prior, str) or default_prior not in priors:
        raise ValueError(
            f"Prior catalog default_prior must name one configured prior: {path}"
        )
    return default_prior


def format_prior_option_label(name: str, prior: Mapping[str, Any]) -> str:
    """Build a selectbox label that includes the Normal alpha/eta parameters."""

    alpha = prior.get("alpha", {})
    eta = prior.get("eta", {})
    display_name = name.replace("_", " ")
    return (
        f"{display_name} | alpha~Normal({alpha.get('mean')}, {alpha.get('sd')}), "
        f"eta~Normal({eta.get('mean')}, {eta.get('sd')})"
    )


def load_prior_options(
    path: Path = DEFAULT_PRIOR_CATALOG,
) -> list[tuple[str, str, str]]:
    """Return ``(name, select_label, description)`` rows in stable order."""

    catalog = load_prior_catalog(path)
    rows: list[tuple[str, str, str]] = []
    for name in sorted(catalog):
        prior = catalog[name]
        description = str(prior.get("description") or "")
        rows.append((name, format_prior_option_label(name, prior), description))
    return rows


def prior_definition_rows(
    path: Path = DEFAULT_PRIOR_CATALOG,
) -> list[dict[str, str]]:
    """Return table rows describing every catalog prior for the UI."""

    catalog = load_prior_catalog(path)
    rows: list[dict[str, str]] = []
    for name in sorted(catalog):
        prior = catalog[name]
        alpha = prior.get("alpha", {})
        eta = prior.get("eta", {})
        rows.append(
            {
                "prior": name,
                "alpha": f"Normal(mean={alpha.get('mean')}, sd={alpha.get('sd')})",
                "eta": f"Normal(mean={eta.get('mean')}, sd={eta.get('sd')})",
                "description": str(prior.get("description") or ""),
            }
        )
    return rows


def load_prior_names(path: Path = DEFAULT_PRIOR_CATALOG) -> list[str]:
    """Return the locally configured Bayesian-prior names in stable order."""

    return [name for name, _label, _description in load_prior_options(path)]


def load_json_file(path: Path) -> dict[str, Any] | None:
    """Load a JSON object when it exists, otherwise return ``None``."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def dose_support_summary(curve_result: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize requested-dose support and extrapolation for UI diagnostics."""

    method = (
        curve_result.get("toxicity_curve", {})
        .get("article_line_weighted", {})
    )
    points = method.get("curve_points", [])
    supported = [point for point in points if point.get("dlt_probability") is not None]
    extrapolation = [
        float(point.get("extrapolation_weight_fraction", 0.0))
        for point in supported
    ]
    stats = method.get("aggregation_stats", {})
    return {
        "method_status": method.get("status", "unavailable"),
        "requested_dose_count": len(points),
        "supported_dose_count": len(supported),
        "unsupported_dose_count": len(points) - len(supported),
        "historical_article_count": stats.get("article_count"),
        "historical_line_count": stats.get("line_count"),
        "segment_weight_ess": stats.get("segment_weight_ess"),
        "mean_extrapolation_weight_fraction": (
            sum(extrapolation) / len(extrapolation) if extrapolation else None
        ),
        "max_extrapolation_weight_fraction": max(extrapolation, default=None),
        "skipped_reasons": stats.get("skipped_reasons", {}),
    }


def prediction_json(curve_result: Mapping[str, Any]) -> str:
    """Render a downloadable prediction result without adding UI state."""

    return json.dumps(curve_result, ensure_ascii=False, indent=2, default=str)
