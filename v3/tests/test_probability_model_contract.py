"""Unsupported model identities must never be interpreted as Logistic fits."""
import json

import pytest
from numpyro import handlers

from phase1_prediction.multidrug import BACKENDS, VERSION
from phase1_prediction.multidrug.contracts import PriorSpec
from phase1_prediction.multidrug.fitting import source_model
from phase1_prediction.multidrug.power_prior import (
    POWER_PRIOR_VERSION, PowerPriorSpec, PowerPriorStore, power_prior_model,
)
from phase1_prediction.multidrug.store import PosteriorStore
from phase1_prediction.multidrug.transport import (
    transport_joint_posteriors, average_joint_posteriors, assemble_components,
)
from test_multidrug_transport import fixture


@pytest.mark.parametrize('transport', [transport_joint_posteriors, average_joint_posteriors, assemble_components])
def test_transport_rejects_unsupported_probability_model_before_prediction(transport):
    _, _, _, target = fixture()
    with pytest.raises(ValueError, match='Unsupported probability model'):
        transport([], target, backend='unsupported')


def test_sampler_rejects_unsupported_probability_model_before_sampling():
    scope, *_ = fixture()
    for model, args in (
        (source_model, (scope, 'unsupported', PriorSpec())),
        (power_prior_model, ([scope], [.5], 'unsupported', PriorSpec(), PowerPriorSpec())),
    ):
        with pytest.raises(ValueError, match='Unsupported probability model'):
            handlers.trace(handlers.seed(model, rng_seed=1)).get_trace(*args)


@pytest.mark.parametrize('kind', ['source', 'target'])
def test_stored_unsupported_model_is_rejected_before_arrays_are_loaded(tmp_path, kind):
    if kind == 'source':
        identity = 'fit_invalid'
        manifest = {'source_fit_id': identity, 'model_version': VERSION, 'backend': 'unsupported'}
        store = PosteriorStore(tmp_path)
    else:
        identity = 'power_invalid'
        manifest = {'target_fit_id': identity, 'version': POWER_PRIOR_VERSION, 'backend': 'unsupported'}
        store = PowerPriorStore(tmp_path)
    directory = store.root / identity
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsupported probability model'):
        store.load(identity)


def test_cli_exposes_only_the_supported_model(monkeypatch, capsys):
    from phase1_prediction.multidrug.__main__ import main
    assert BACKENDS == ('logistic', 'logistic_product', 'single_drug_logistic')
    monkeypatch.setattr('sys.argv', ['models', 'source', '--report', 'report.json', '--backend', 'unsupported'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert '--backend {logistic,logistic_product,single_drug_logistic}' in capsys.readouterr().err
