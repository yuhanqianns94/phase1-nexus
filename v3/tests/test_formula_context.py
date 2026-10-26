"""The displayed target context must follow real exposure coordinates."""
from phase1_web.model_formulas import target_combination, model_formulas
from phase1_web.prior_settings import prior_formulas, prior_parameter_rows
from phase1_prediction.multidrug.contracts import PriorSpec
from test_single_drug_pipeline import example


def test_unused_companions_do_not_display_a_combination_term():
    _, target = example(target_combo=False)
    assert len(target.drugs) == 2
    assert target_combination(target) is False
    assert r'\delta' not in model_formulas(combination=target_combination(target))[1]
    # Explicitly used, dose-unknown companion is still combination treatment.
    target.drugs[1].mode = 'background'
    for point in target.points:
        point.doses[target.drugs[1].drug_id] = None
    assert target_combination(target) is True
    assert r'+\delta' in model_formulas(combination=True)[1]
    assert target_combination(None) is None
    assert r'+\delta C' in model_formulas(combination=None)[1]


def test_primary_priors_match_target_but_do_not_change_inference_settings():
    prior = PriorSpec()
    mono = prior_formulas(prior, 'single_drug_logistic', combination=False)
    combo = prior_formulas(prior, 'single_drug_logistic', combination=True)
    assert len(mono) == 2 and len(combo) == 3
    assert combo[:2] == mono
    assert r'\delta' in combo[-1]
    assert prior == PriorSpec()
    rows = prior_parameter_rows(prior, 'single_drug_logistic')
    assert 'Baseline' in rows[0]['Parameter'] and 'focal' in rows[1]['Parameter']
    assert not any('interaction' in row['Parameter'].lower() for row in rows)
