"""Cohort-only BOIN regressions and optional parity with the local reference."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from phase1_simulation.boin import (
    _select_isotonic_mtd, _select_raw_mtd, simulate_boin_trial,
)
from phase1_simulation.service import BoinSettings, SIMULATION_MODEL_VERSION, run_boin_simulation


class _ScriptedRng(np.random.Generator):
    def __init__(self, outcomes):
        super().__init__(np.random.PCG64(1))
        self.outcomes = iter(outcomes)

    def binomial(self, n, p, size=None):
        value = next(self.outcomes)
        assert 0 <= value <= n
        return value


def _trial(outcomes, **changes):
    arguments = {
        "start_index": 0, "cohort_size": 3, "max_sample_size": 12,
        "target_dlt": .25, "stop_rule_enabled": False,
        "rng": _ScriptedRng(outcomes),
    }
    arguments.update(changes)
    return simulate_boin_trial([.05, .15, .35, .6], **arguments)


def test_settings_expose_only_ordinary_cohort_controls():
    assert {field.name for field in fields(BoinSettings)} == {
        "target_dlt", "cohort_size", "max_sample_size", "start_index",
        "elimination_cutoff", "stop_rule_enabled", "stop_rule_m", "use_isotonic",
        "mtd_definition", "acceptable_low", "acceptable_high", "phi1", "phi2",
        "check_lowest_dose", "lowest_dose_delta", "check_final_mtd_deesc_bound", "modify_1_3_deesc_to_stay",
    }


def test_complete_cohort_budget_is_a_strict_total_patient_cap():
    for cohort_size in (1, 2, 3, 4):
        outcome = _trial([0] * 6, cohort_size=cohort_size, max_sample_size=6 * cohort_size)
        assert outcome["total_n"] == 6 * cohort_size
        assert outcome["cohort_count"] == 6
        assert all(n % cohort_size == 0 for n in outcome["treated_by_dose"])


def test_safety_is_checked_after_the_first_single_patient_cohort():
    outcome = _trial([1], cohort_size=1, max_sample_size=6)
    assert outcome["total_n"] == 1
    assert outcome["early_stopped"]
    assert outcome["selected_index"] is None
    assert outcome["stop_reason"] == "lowest_dose_extra_safety_stop"


def test_deescalation_includes_equality_with_the_reference_boundary():
    outcome = _trial(
        [1, 0], start_index=1, max_sample_size=6,
        lambda_e=.15, lambda_d=1 / 3, elimination_cutoff=1.0,
        check_lowest_dose=False,
    )
    assert outcome["treated_by_dose"] == [3, 3, 0, 0]
    assert outcome["action_counts"]["deescalate"] == 1


def test_elimination_precedes_ordinary_boundary_and_blocks_reescalation():
    outcome = _trial([3, 0], start_index=1, max_sample_size=6)
    assert outcome["treated_by_dose"] == [3, 3, 0, 0]
    assert outcome["action_counts"] == {"deescalate_after_elimination": 1, "stay": 1}


def test_stop_at_m_requires_stay_and_does_not_count_as_safety_stop():
    outcome = _trial([0, 0, 0, 0], stop_rule_enabled=True, stop_rule_m=3)
    assert outcome["total_n"] == 12  # Reaching m while escalating did not stop.
    assert outcome["stop_reason"] == "single_dose_stay_stop_rule_m_reached"
    assert not outcome["early_stopped"]
    assert outcome["selected_index"] == 3


def test_bottom_dose_deescalation_request_stays_without_automatic_safety_stop():
    outcome = _trial([1], max_sample_size=3, lambda_e=.15, lambda_d=1 / 3,
                     elimination_cutoff=1.0, check_lowest_dose=False)
    assert outcome["action_counts"] == {"stay": 1}
    assert not outcome["early_stopped"]


def test_raw_selection_ties_choose_lowest_but_isotonic_below_target_ties_choose_highest():
    y, n, excluded = np.array([0, 0]), np.array([3, 3]), np.array([False, False])
    assert _select_raw_mtd(y, n, excluded, .25, None) == 0
    assert _select_isotonic_mtd(y, n, excluded, .25) == 1


def test_final_selection_can_be_empty_without_safety_stop():
    outcome = _trial([2], max_sample_size=3, elimination_cutoff=1.0, check_lowest_dose=False)
    assert not outcome["early_stopped"]
    assert outcome["selected_index"] is None


def test_simulation_reports_current_engine_version_and_strict_budget():
    prediction = {"run_id": "test", "methods": {"test": {
        "status": "fitted", "doses": [1, 2, 3],
        "posterior_draws": [[0.0, 0.0, 0.0]] * 4,
    }}}
    outcome = run_boin_simulation(prediction, method_key="test", settings=BoinSettings(max_sample_size=6),
                                  posterior_curve_count=4, trials_per_curve=2)
    assert outcome["simulation_model_version"] == SIMULATION_MODEL_VERSION
    assert outcome["mean_total_n"] == 6


@pytest.fixture
def reference(monkeypatch):
    """Import the user's specified reference read-only when installed locally."""
    path = Path(__file__).resolve().parents[3] / "GPT-wrapper web" / "BOIN_GPT.py"
    if not path.is_file():
        pytest.skip("The optional local GPT-wrapper BOIN reference is unavailable.")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("phase1_boin_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cohort_size", [1, 2, 3, 4])
@pytest.mark.parametrize("use_isotonic", [True, False])
@pytest.mark.parametrize("start_index", [0, 2])
@pytest.mark.parametrize("stop_enabled", [True, False])
def test_ordinary_cohort_trajectories_match_reference(reference, cohort_size, use_isotonic, start_index, stop_enabled):
    truth = np.array([.05, .15, .25, .45, .7])
    for seed in range(8):
        config = {
            "design": "BOIN", "target_tox": .25,
            "dose": {"n_dose": len(truth), "start_dose_idx": start_index + 1},
            "cohort": {"cohort_size": cohort_size, "ncohort": 8},
            "stop_rule": {"enabled": stop_enabled, "m": 8},
            "boin": {
                "accelerated_titration": {"enabled": False},  # Disable reference-only path.
                "overdose_control": {"cutoff": .95, "check_lowest_dose": True,
                                     "lowest_dose_delta": .03, "check_final_mtd_deesc_bound": True},
                "select_mtd": {"use_isotonic": use_isotonic},
            },
        }
        engine = reference.BOINTrialEngine(config)
        rng = np.random.default_rng(seed)
        while not engine.stop:
            engine.step(int(rng.binomial(cohort_size, truth[engine.dose_curr])))
        outcome = simulate_boin_trial(
            truth, start_index=start_index, cohort_size=cohort_size, max_sample_size=8 * cohort_size,
            target_dlt=.25, rng=np.random.default_rng(seed), stop_rule_enabled=stop_enabled,
            stop_rule_m=8, use_isotonic=use_isotonic,
        )
        assert outcome["treated_by_dose"] == engine.n.astype(int).tolist()
        assert outcome["dlt_by_dose"] == engine.y.astype(int).tolist()
        assert outcome["selected_index"] == engine.mtd_idx
        assert outcome["cohort_count"] == engine.cohort_idx
        assert outcome["total_n"] <= 8 * cohort_size


def test_isotonic_recommendations_match_reference_including_untried_and_tied_levels(reference):
    for y, n, eliminated, target in [
        ([0, 0, 0], [3, 3, 0], [False] * 3, .25),
        ([1, 2, 0], [3, 6, 0], [False] * 3, .25),
        ([0, 2, 1, 0], [0, 3, 6, 0], [False, False, False, True], .25),
        ([1, 1], [4, 4], [False, False], .25),
    ]:
        y, n, eliminated = np.asarray(y), np.asarray(n), np.asarray(eliminated)
        for upper in (None, .3, .4):
            expected = reference.choose_mtd_from_isotonic(y, n, target, ~eliminated, upper)
            assert _select_isotonic_mtd(y, n, eliminated, target, upper) == expected
