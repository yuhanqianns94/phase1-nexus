"""Cohort-by-cohort 3+3 simulation matching GPT-wrapper web.

It simulates Binomial cohort outcomes and applies the reference state machine:

* 0/3: escalate;
* 1/3: remain at the dose and expand to six participants;
* at most 1/6: escalate;
* at least 2/3: move down and resolve the reference stop state there;
* exactly 2/6: immediately select the preceding dose;
* more than 2/6: move down and resolve the reference stop state there;
* if the lowest dose is overly toxic: stop without selecting a dose.

Dose indices in both the input and output are zero-based.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np


_HIGHEST_SAFE_POLICIES = frozenset({"select_highest", "expand_to_six"})


def _validate_probabilities(true_probabilities: Sequence[float]) -> np.ndarray:
    """Return a validated one-dimensional true-DLT probability vector."""

    if isinstance(true_probabilities, (str, bytes)):
        raise ValueError("true_probabilities must be a non-empty 1D sequence.")
    try:
        raw = np.asarray(true_probabilities)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "true_probabilities must be a non-empty 1D sequence."
        ) from error
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("true_probabilities must be a non-empty 1D sequence.")

    raw_values = raw.tolist()
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
        raise ValueError("Every true DLT probability must be a finite number.")
    probabilities = raw.astype(float, copy=False)
    if not np.isfinite(probabilities).all():
        raise ValueError("Every true DLT probability must be a finite number.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Every true DLT probability must be between 0 and 1.")
    return probabilities


def _validate_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer.")
    return int(value)


def _validate_target_dlt(target_dlt: Any) -> float:
    if isinstance(target_dlt, bool) or not isinstance(target_dlt, Real):
        raise ValueError("target_dlt must be a finite probability strictly between 0 and 1.")
    result = float(target_dlt)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("target_dlt must be a finite probability strictly between 0 and 1.")
    return result


def simulate_three_plus_three_trial(
    true_probabilities: Sequence[float],
    *,
    start_index: int,
    max_sample_size: int,
    target_dlt: float,
    rng: np.random.Generator,
    highest_safe_policy: str = "select_highest",
) -> dict[str, Any]:
    """Simulate the GPT-wrapper 3+3 state machine.

    Parameters
    ----------
    true_probabilities:
        True patient-level DLT probability at each ordered dose.
    start_index:
        Zero-based starting dose index.
    max_sample_size:
        Maximum number of participants.  It must be a positive multiple of
        three because partial cohorts are not part of the standard 3+3 design.
    target_dlt:
        Target toxicity used by downstream OC summaries.  Standard 3+3 does
        not use this value for dose decisions, but it is validated here so all
        design engines share the same safe public contract.
    rng:
        Random-number generator (or compatible object) exposing
        ``binomial(n, p)``.
    highest_safe_policy:
        ``"select_highest"`` selects the highest dose after 0/3.  The optional
        conservative ``"expand_to_six"`` policy treats a second cohort there
        and selects it only if the total is at most 1/6.

    Notes
    -----
    If the cohort budget is exhausted before a natural stopping rule, the
    reference implementation returns its current state, including an
    escalated-to dose that has not yet enrolled a cohort.
    """

    probabilities = _validate_probabilities(true_probabilities)
    start = _validate_integer(start_index, label="start_index")
    if not 0 <= start < probabilities.size:
        raise ValueError("start_index must refer to an available dose.")

    sample_limit = _validate_integer(max_sample_size, label="max_sample_size")
    if sample_limit < 3 or sample_limit % 3 != 0:
        raise ValueError("max_sample_size must be a positive multiple of 3 and at least 3.")

    _validate_target_dlt(target_dlt)
    if not hasattr(rng, "binomial") or not callable(rng.binomial):
        raise ValueError("rng must provide a callable binomial(n, p) method.")
    if highest_safe_policy not in _HIGHEST_SAFE_POLICIES:
        choices = ", ".join(sorted(_HIGHEST_SAFE_POLICIES))
        raise ValueError(f"highest_safe_policy must be one of: {choices}.")

    dose_count = int(probabilities.size)
    treated_by_dose = np.zeros(dose_count, dtype=int)
    dlt_by_dose = np.zeros(dose_count, dtype=int)
    action_counts = {"escalate": 0, "stay": 0, "deescalate": 0}

    current_index = start
    flag_up = 0
    selected_index: int | None = None
    stop_reason = ""
    cohort_count = 0
    max_cohort_count = sample_limit // 3

    def terminal_result() -> tuple[bool, int | None, str]:
        if current_index < 0:
            return True, None, "lowest_dose_over_toxic"
        if current_index >= dose_count:
            return True, dose_count - 1, "escalated_above_highest_dose"
        if flag_up == 1 and treated_by_dose[current_index] == 6:
            previous = current_index - 1
            return (
                True,
                previous if previous >= 0 else None,
                "three_plus_three_terminal_rule_after_upward_move",
            )
        if flag_up == 0 and treated_by_dose[current_index] == 6:
            return True, current_index, "three_plus_three_terminal_rule_after_deescalation"
        return False, None, ""

    while cohort_count < max_cohort_count:
        should_stop, terminal_selection, terminal_reason = terminal_result()
        if should_stop:
            selected_index = terminal_selection
            stop_reason = terminal_reason
            break
        cohort_dlt_raw = rng.binomial(3, float(probabilities[current_index]))
        if isinstance(cohort_dlt_raw, bool) or not isinstance(cohort_dlt_raw, Integral):
            raise ValueError("rng.binomial(3, p) must return an integer from 0 to 3.")
        cohort_dlt = int(cohort_dlt_raw)
        if not 0 <= cohort_dlt <= 3:
            raise ValueError("rng.binomial(3, p) must return an integer from 0 to 3.")

        treated_by_dose[current_index] += 3
        dlt_by_dose[current_index] += cohort_dlt
        cohort_count += 1
        treated_here = int(treated_by_dose[current_index])
        dlt_here = int(dlt_by_dose[current_index])

        if treated_here == 3:
            if dlt_here == 0:
                if (
                    current_index == dose_count - 1
                    and highest_safe_policy == "expand_to_six"
                ):
                    action_counts["stay"] += 1
                else:
                    action_counts["escalate"] += 1
                    current_index += 1
                    flag_up = 1
            elif dlt_here == 1:
                action_counts["stay"] += 1
            else:
                action_counts["deescalate"] += 1
                current_index -= 1
                flag_up = 0

        elif treated_here == 6:
            if dlt_here <= 1:
                action_counts["escalate"] += 1
                current_index += 1
                flag_up = 1
            elif dlt_here == 2:
                action_counts["deescalate"] += 1
                selected_index = current_index - 1 if current_index > 0 else None
                stop_reason = (
                    "three_plus_three_two_of_six"
                    if selected_index is not None
                    else "lowest_dose_over_toxic"
                )
                break
            else:
                action_counts["deescalate"] += 1
                current_index -= 1
                flag_up = 0
        else:
            raise RuntimeError("Internal 3+3 state exceeded six participants at one dose.")

        should_stop, terminal_selection, terminal_reason = terminal_result()
        if should_stop:
            selected_index = terminal_selection
            stop_reason = terminal_reason
            break

        if cohort_count >= max_cohort_count:
            selected_index = current_index
            stop_reason = "max_cohort_count_reached_return_current_dose"
            break

    if not stop_reason:
        selected_index = current_index if 0 <= current_index < dose_count else None
        stop_reason = "max_cohort_count_reached_return_current_dose"

    total_n = int(treated_by_dose.sum())
    return {
        "selected_index": selected_index,
        "treated_by_dose": treated_by_dose.tolist(),
        "dlt_by_dose": dlt_by_dose.tolist(),
        "total_n": total_n,
        "total_dlt": int(dlt_by_dose.sum()),
        "cohort_count": cohort_count,
        # Match the existing OC meaning: an early stop is a stop with no MTD,
        # rather than every natural rule that happens before the N cap.
        "early_stopped": selected_index is None,
        "stop_reason": stop_reason,
        "action_counts": action_counts,
    }
