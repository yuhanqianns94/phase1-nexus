"""Cost arithmetic and pinned historical pricing for the model migration."""
import importlib.util
from pathlib import Path
import pytest


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_read_write_and_output_are_charged_once():
    cost = script("compare_reasoning_costs").usage_cost({
        "input_tokens": 100000, "output_tokens": 10000,
        "input_tokens_details": {"cached_tokens": 20000, "cache_write_tokens": 70000},
        "output_tokens_details": {"reasoning_tokens": 1000},
    })
    assert cost["usage_based_usd"] == pytest.approx(0.297)
    assert cost["normalized_no_cache_usd"] == pytest.approx(0.3)
    assert cost["output_usd"] == pytest.approx(0.1)
    assert cost["reasoning_tokens"] == 1000


def test_unknown_cache_writes_produce_a_range_not_a_false_exact_cost():
    cost = script("compare_reasoning_costs").usage_cost({
        "input_tokens": 100000, "output_tokens": 10000,
        "input_tokens_details": {"cached_tokens": 20000},
    })
    assert cost["usage_based_usd"] is None
    assert cost["usage_based_usd_range"] == pytest.approx([0.262, 0.302])


def test_long_context_pricing_applies_to_the_entire_request():
    cost = script("compare_reasoning_costs").usage_cost({
        "input_tokens": 300000, "output_tokens": 10000,
        "input_tokens_details": {"cached_tokens": 100000, "cache_write_tokens": 150000},
    })
    assert cost["long_context"]
    assert cost["usage_based_usd"] == pytest.approx(1.12)
    assert cost["normalized_no_cache_usd"] == pytest.approx(1.35)


def test_rerun_script_keeps_old_and_new_model_rates_separate():
    pricing = script("reextract_exposures").batch_pricing
    assert pricing("gpt-6.1-sol") == {"input_usd_per_million": 1, "output_usd_per_million": 5,
                                      "cache_write_multiplier": 1.25, "cached_input_multiplier": 0.05}
    assert pricing("gpt-5.6-sol")["cached_input_multiplier"] == 0.1
    with pytest.raises(ValueError):
        pricing("unpriced-model")
