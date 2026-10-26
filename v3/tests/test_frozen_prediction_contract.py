"""A saved curve freezes its evidence distribution, not only sampled recipes."""
import copy

import pytest

from phase1_prediction.multidrug import service
from phase1_prediction.multidrug.store import PosteriorStore
from phase1_prediction.multidrug.transport import transport_joint_posteriors
from test_multidrug_average import two_articles


def test_regrid_rejects_changed_unsampled_contributor(tmp_path, monkeypatch):
    entries, target = two_articles('logistic')
    original = target.model_copy(deep=True)
    store = PosteriorStore(tmp_path)
    entries = [store.save(m['source_fit_id'], s, a, m) for s, a, m in entries]
    monkeypatch.setattr(service, '_select', lambda *a, **kw:
        ([(s, {'score': {'similarity': 1.}}, 1.) for s, _, _ in entries], {'notices': []}))
    monkeypatch.setattr(service, 'fit_library', lambda *a, **kw: (entries, []))
    prediction = service.predict(None, original, store_root=tmp_path, draw_count=1, seed=2).to_dict()
    old = next(iter(prediction['methods'].values()))
    # Simulate a changed eligibility result without permitting dose-unknown
    # evidence to anchor a numeric target (which is now always prohibited).
    changed = transport_joint_posteriors(entries[:1], target, draw_count=1, seed=2)
    monkeypatch.setattr(service, 'transport_joint_posteriors',
        lambda rows, *args, **kwargs: transport_joint_posteriors(rows[:1], *args, **kwargs))
    assert len(old['contributors']) == 2 and len(changed['contributors']) == 1
    # A finite draw happened to miss the disappearing contributor. A recipe-only
    # check therefore accepts a different evidence distribution by accident.
    assert old['provenance'] == changed['provenance']
    snapshot = copy.deepcopy(prediction)
    with pytest.raises(ValueError, match='evidence distribution'):
        service.evaluate_frozen_prediction(prediction, target)
    assert prediction == snapshot
