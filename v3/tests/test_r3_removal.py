"""R3 retirement applies to new extraction and historical evidence under every policy."""
from types import SimpleNamespace

import pytest

from phase1_acquisition.screening import CURRENT_PROMPT
from phase1_extraction.artifacts import read_json, write_json
from phase1_extraction.llm import load_prompt
from phase1_extraction.models import CurveRow, DLTObservation, PopulationCount, ValueProvenance
from phase1_extraction.resolver import resolve_extraction
from phase1_prediction.evidence import load_evidence_library
from phase1_prediction.multidrug.observations import build_article_scopes, outcome_reasons
from phase1_prediction.multidrug.store import PosteriorStore, source_fit_id
from test_cohort_exposures import extraction, write_article
from test_resolver import base_extraction


@pytest.mark.parametrize('design_type', ['classic_3plus3', 'modified_3plus3'])
@pytest.mark.parametrize('y', [None, 0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('expansion', [
    'unknown', 'completed_expansion', 'completed_to_six', 'added_three',
    'initial_three_only', 'stopped_before_expansion',
])
def test_design_and_dlt_never_supply_actual_denominator(design_type, y, expansion):
    e = base_extraction(doses=(100,), design=True)
    e.designs[0].normalized_type = design_type
    e.designs[0].initial_cohort_n = 2
    e.designs[0].added_cohort_n = 4
    e.designs[0].expanded_total_n = 6
    e.designs[0].expand_when_y_equals = [1]
    e.designs[0].stop_when_y_at_least = 2
    e.cohorts[0].expansion_status = expansion
    if y is not None:
        e.dlt_observations = [DLTObservation(observation_id='o', study_id='study',
            result_scope_id='scope', cohort_id='c1', count_kind='patients', reported_count=y)]
    before = e.model_dump()
    result = resolve_extraction(e)
    row = result.curve_rows[0]
    assert (row.y, row.n) == (y, None)
    assert row.denominator_basis == 'missing' and row.design_candidate_ns == []
    assert not row.included_in_default and not row.outcome_usable
    assert not {'R2', 'R3'}.intersection(row.rule_ids)
    assert all(a.rule_id != 'R3' for a in result.rule_applications)
    assert e.model_dump() == before


def test_legacy_r3_counts_and_their_arithmetic_descendants_are_unusable():
    e = base_extraction(doses=(100,), design=True, explicit_no_dlt=True)
    e.population_counts = [
        PopulationCount(count_id='old', study_id='study', role='treated', reported_value=6,
                        provenance=ValueProvenance(basis='r3_design_inferred')),
        PopulationCount(count_id='derived', study_id='study', cohort_id='c1', role='treated',
                        reported_value=5, derived_from_count_ids=['old'],
                        provenance=ValueProvenance(basis='arithmetic_derived')),
    ]
    row = resolve_extraction(e).curve_rows[0]
    assert (row.y, row.n) == (0, None)
    assert not row.outcome_usable


@pytest.mark.parametrize('marker', [
    {'rule_ids': ['R3']}, {'denominator_basis': 'design_inferred'},
    {'denominator_role': 'design_assumed_cohort'},
])
@pytest.mark.parametrize('policy', ['project_default', 'reported_only'])
def test_legacy_rows_are_rejected_in_both_libraries_without_rewriting_files(tmp_path, marker, policy):
    paths = write_article(tmp_path, extraction())
    payload = read_json(paths.curve_rows)
    raw = payload['all_rows'][0]
    rejected_id = raw['curve_row_id']
    raw.update(marker, included_in_default=True, included_in_reported_only=True, outcome_usable=True)
    # Mimic an older artifact whose policy projections still include the R3 row.
    payload['project_default'] = payload['all_rows']
    payload['reported_only'] = payload['all_rows']
    write_json(paths.curve_rows, payload)
    before = paths.curve_rows.read_bytes()
    parsed = CurveRow.model_validate(raw)
    assert not parsed.included_in_default and not parsed.included_in_reported_only
    assert not parsed.outcome_usable
    assert 'retired_r3_denominator' in outcome_reasons(parsed, policy)
    library = load_evidence_library([tmp_path], evidence_policy=policy)
    assert library.skipped['retired_r3_denominator'] == 1
    assert all(p.curve_row_id != rejected_id for s in library.series for p in s.points)
    scopes, _ = build_article_scopes(paths, evidence_policy=policy)
    assert all(o.observation_id != rejected_id for s in scopes for o in s.observations)
    assert any('retired_r3_denominator' in r['reasons'] for s in scopes for r in s.exclusions)
    assert paths.curve_rows.read_bytes() == before


def test_retired_counts_cannot_create_posterior_cache_or_likelihood(tmp_path):
    scope = SimpleNamespace(observations=[SimpleNamespace(provenance={'rule_ids': ['R3']})])
    with pytest.raises(ValueError, match='Retired R3'):
        source_fit_id(scope, 'logistic', None, None)
    with pytest.raises(ValueError, match='Retired R3'):
        PosteriorStore(tmp_path).save('legacy', scope, {}, {'backend': 'logistic'})
    from phase1_prediction.multidrug.fitting import observe_scope
    with pytest.raises(ValueError, match='Retired R3'):
        observe_scope(scope, None, None)
    assert not list(tmp_path.iterdir())


def test_preexisting_posterior_with_r3_is_not_loaded(tmp_path):
    import hashlib
    import numpy as np
    from phase1_prediction.multidrug import VERSION
    from phase1_prediction.multidrug.contracts import Exposure, Observation
    from test_multidrug_transport import fixture

    scope, *_ = fixture(1)
    scope.observations = [Observation(observation_id='old', cohort_id='old', n=6, y=1,
                                     exposures=[Exposure(drug_id=d.model_drug_id, presence='present') for d in scope.drugs],
                                     provenance={'denominator_basis': 'design_inferred'})]
    directory = tmp_path / 'old'
    directory.mkdir()
    np.savez_compressed(directory / 'draws.npz', a=np.zeros(2))
    (directory / 'scope.json').write_text(scope.model_dump_json())
    write_json(directory / 'manifest.json', {
        'source_fit_id': 'old', 'model_version': VERSION, 'backend': 'logistic',
        'draws_sha256': hashlib.sha256((directory / 'draws.npz').read_bytes()).hexdigest(),
    })
    with pytest.raises(ValueError, match='Retired R3'):
        PosteriorStore(tmp_path).load('old')


def test_current_prompts_explicitly_prohibit_design_denominators():
    assert 'R3 is retired: never infer a' in CURRENT_PROMPT
    assert 'Do not demand an explicitly printed denominator in that case' not in CURRENT_PROMPT
    prompt = load_prompt()
    assert 'Never infer an\nobserved denominator from a design label' in prompt
    assert 'R3 is left to Python' not in prompt
