"""Opt-in research pilot: primary-first monotherapy allocation and pair priors.

No website score is converted into a clinical toxicity coefficient. This module
does not change the application's default model, matching or prediction route.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .contracts import Contract, PriorSpec, fingerprint


class PairPriorScenario(Contract):
    """An explicit sensitivity setting, not an estimated database effect."""

    mean: float = 0.
    sd: float = Field(default=1., gt=0)
    basis: Literal['weak_default', 'sensitivity_scenario'] = 'weak_default'
    note: str = 'Uncalibrated interaction uncertainty; no external evidence was fitted.'

    @model_validator(mode='after')
    def honest_origin(self):
        if self.basis == 'weak_default' and self.mean != 0:
            raise ValueError('A shifted interaction prior must be labelled sensitivity_scenario.')
        return self


def interaction_evidence_gate(kind: str) -> dict:
    """Classify evidence suitability; this deliberately returns no numeric eta."""
    reasons = {
        'clinical_dlt_counts': ('needs_clinical_harmonization',
            'Verify unique DLT patients, denominator, joint doses, window, population, schedule and study overlap before fitting.'),
        'pk_exposure_ratio': ('requires_exposure_toxicity_bridge',
            'An AUC or Cmax ratio is not a DLT interaction coefficient.'),
        'ddi_description': ('qualitative_evidence_only',
            'Interaction descriptions and severity categories have no calibrated eta scale.'),
        'faers_signal': ('qualitative_evidence_only',
            'Spontaneous-report signals lack the exposed-patient denominator for DLT incidence.'),
        'cell_efficacy_synergy': ('different_endpoint',
            'Cell growth/viability synergy is not human DLT synergy.'),
    }
    if kind not in reasons:
        raise ValueError('Unknown interaction evidence kind')
    status, reason = reasons[kind]
    return dict(kind=kind, status=status, reason=reason, automatic_numeric_prior=False)


def predict_primary_first_pilot(entries, target, *, aggregation='mixture',
        scenario=None, prior=None, draw_count=1000, seed=20261005,
        weights=None, relevance_decisions=None, top_articles_per_drug=None):
    """Reuse source posterior draws under an explicit research-only allocation.

Only numeric monotherapy sources and a two-numeric-drug target are supported.
The caller must use current, validated source fits (e.g. service.fit_library).
Source posteriors are not refitted or rewritten. A custom eta setting is always
labelled sensitivity analysis; clinical calibration is a separate future step.
"""
    from .component_assembly import assemble_posteriors
    from .primary_first import allocate_primary_first
    from .transport import _base_result

    if len(target.drugs) != 2 or any(d.mode != 'numeric' for d in target.drugs):
        raise ValueError('This pilot requires two numeric target drugs.')
    if aggregation not in {'average', 'mixture'}:
        raise ValueError('Unknown component curve aggregation')
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 1:
        raise ValueError('draw_count must be positive')
    scenario = scenario if isinstance(scenario, PairPriorScenario) else PairPriorScenario.model_validate(scenario or {})
    prior = prior or PriorSpec()
    usable, excluded = [], []
    seen = set()
    for scope, arrays, manifest in entries:
        if scope.scope_id in seen:
            raise ValueError('Use exactly one source fit per model scope in the pilot.')
        seen.add(scope.scope_id)
        reason = None
        if manifest.get('backend') != 'logistic' or manifest.get('status') != 'fitted':
            reason = 'needs_fitted_component_logistic'
        elif len(scope.drugs) != 1:
            reason = 'combination_outcomes_are_not_single_drug_outcomes'
        elif scope.drugs[0].mode != 'numeric':
            reason = 'pilot_requires_numeric_source_axis'
        elif not scope.observations:
            reason = 'no_observed_dlt_data'
        if reason:
            excluded.append(dict(scope_id=scope.scope_id, document_id=scope.document_id, reason=reason))
        else:
            usable.append((scope, arrays, manifest))
    allocation = allocate_primary_first([s for s, _, _ in usable], target,
        weights=weights, relevance_decisions=relevance_decisions,
        top_articles_per_drug=top_articles_per_drug, require_usable_coordinates=True)
    allocated = {}
    for row in allocation['allocations']:
        allocated.setdefault(row['scope_id'], {})[row['target_id']] = row['source_id']
    selected = [e for e in usable if e[0].scope_id in allocated]
    frozen_target = target.model_copy(deep=True)
    frozen_target.mapping_overrides = allocated
    if {row['target_id'] for row in allocation['allocations']} != {d.drug_id for d in target.drugs}:
        result = _base_result(target, 'logistic', 'primary_first_component_pilot', seed)
    else:
        result = assemble_posteriors(selected, frozen_target, aggregation=aggregation,
            draw_count=draw_count, seed=seed, prior=prior, weights=weights,
            relevance_decisions=relevance_decisions, pair_prior_scenario=scenario)
    result.update(format='primary_first_component_pilot_v1', experimental=True,
        target=target.model_dump(mode='json'), allocation=allocation,
        pair_prior_scenario=scenario.model_dump(mode='json'),
        external_interaction_data_used=False,
        assembly_prior_settings=prior.model_dump(mode='json'),
        source_prior_settings=[{'source_fit_id': m['source_fit_id'], 'prior': m.get('prior')}
            for _, _, m in selected],
        target_outcomes_used=False,
        interpretation='Borrowed component posterior predictive curves with an uncalibrated interaction prior; not a posterior updated by target-trial outcomes.')
    result['excluded'].extend(excluded)
    result['prediction_id'] = fingerprint(result)
    return result
