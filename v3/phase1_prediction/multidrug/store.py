"""Atomic, checksummed, versioned joint posteriors. Source inputs only."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from . import VERSION, validate_backend
from .contracts import ModelScope, PriorSpec, SamplerSpec, fingerprint


_SOURCE_FALLBACKS = ContextVar("source_posterior_fallbacks", default=())


@contextmanager
def posterior_cache_context(roots):
    """Read verified published source fits without writing into shared releases."""
    token = _SOURCE_FALLBACKS.set(tuple(Path(p) for p in roots))
    try:
        yield
    finally:
        _SOURCE_FALLBACKS.reset(token)


def numerical_dependencies():
    import jax, jaxlib, numpyro, scipy
    return {'numpy':np.__version__,'scipy':scipy.__version__,'jax':jax.__version__,
            'jaxlib':jaxlib.__version__,'numpyro':numpyro.__version__}


def engine_hash():
    here = Path(__file__).parent
    sources={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
             for name in ("math.py", "fitting.py", "contracts.py")}
    return fingerprint({'sources':sources,'dependencies':numerical_dependencies()})


def source_fit_id(scope, backend, prior, sampler):
    from .prior_presets import canonical_prior
    prior = canonical_prior(prior, backend)
    source = scope.model_dump(mode="json")
    source["lineage"].pop("root", None)
    return "fit_" + fingerprint({"scope": source, "backend": backend, "prior": prior.model_dump(),
                                 "sampler": sampler.model_dump(), "version": VERSION, "engine": engine_hash()})[:32]


class PosteriorStore:
    def __init__(self, root, *, fallback_roots=None):
        self.root = Path(root)
        self.fallback_roots = tuple(p for p in (
            _SOURCE_FALLBACKS.get() if fallback_roots is None else map(Path, fallback_roots)
        ) if p.resolve() != self.root.resolve())

    def load(self, fit_id):
        directory = self.root / fit_id
        if not (directory / "manifest.json").is_file():
            for root in self.fallback_roots:
                if (root / fit_id / "manifest.json").is_file():
                    return PosteriorStore(root, fallback_roots=()).load(fit_id)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["source_fit_id"] != fit_id or manifest["model_version"] != VERSION:
            raise ValueError("Incompatible posterior identity/version")
        validate_backend(manifest.get('backend', 'logistic'))
        path = directory / "draws.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["draws_sha256"]:
            raise ValueError("Posterior checksum mismatch")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        scope = ModelScope.model_validate(json.loads((directory / "scope.json").read_text()))
        if manifest.get('scope_sha256') and hashlib.sha256((directory/'scope.json').read_bytes()).hexdigest()!=manifest['scope_sha256']:
            raise ValueError('Source scope checksum mismatch')
        if not manifest.get('scope_sha256') and manifest.get('engine_hash') and manifest.get('prior') and manifest.get('sampler'):
            source=scope.model_dump(mode='json');source['lineage'].pop('root',None)
            expected='fit_'+fingerprint({'scope':source,'backend':manifest['backend'],'prior':manifest['prior'],
                'sampler':manifest['sampler'],'version':VERSION,'engine':manifest['engine_hash']})[:32]
            if expected!=fit_id:raise ValueError('Source scope identity mismatch')
        return scope, arrays, manifest

    def save(self, fit_id, scope, arrays, manifest):
        validate_backend(manifest.get('backend', 'logistic'))
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / fit_id
        if final.exists():
            return self.load(fit_id)
        staging = Path(tempfile.mkdtemp(prefix=".writing_", dir=self.root))
        try:
            np.savez_compressed(staging / "draws.npz", **arrays)
            (staging / "scope.json").write_text(scope.model_dump_json(indent=2))
            manifest = {**manifest, "source_fit_id": fit_id, "model_version": VERSION,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "scope_sha256": hashlib.sha256((staging/'scope.json').read_bytes()).hexdigest(),
                        "draws_sha256": hashlib.sha256((staging / "draws.npz").read_bytes()).hexdigest()}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
            try:
                staging.rename(final)
            except FileExistsError:
                return self.load(fit_id)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return self.load(fit_id)

    def fit(self, scope, backend, *, prior=None, sampler=None):
        import fcntl
        from .prior_presets import canonical_prior
        prior,sampler=canonical_prior(prior, backend),sampler or SamplerSpec()
        fit_id=source_fit_id(scope,backend,prior,sampler)
        if not (self.root / fit_id / "manifest.json").is_file() and any(
            (root / fit_id / "manifest.json").is_file() for root in self.fallback_roots
        ):
            return self.load(fit_id)
        self.root.mkdir(parents=True,exist_ok=True)
        with (self.root/(fit_id+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._fit_locked(scope,backend,prior=prior,sampler=sampler)

    def _fit_locked(self, scope, backend, *, prior=None, sampler=None):
        from .fitting import fit_source
        from .prior_presets import canonical_prior
        prior, sampler = canonical_prior(prior, backend), sampler or SamplerSpec()
        fit_id = source_fit_id(scope, backend, prior, sampler)
        if (self.root / fit_id / "manifest.json").is_file():
            return self.load(fit_id)
        arrays, diagnostics = fit_source(scope, backend, prior, sampler)
        return self.save(fit_id, scope, arrays, {
            **diagnostics, "document_id": scope.document_id, "scope_id": scope.scope_id,
            "backend": backend, "prior": prior.model_dump(), "sampler": sampler.model_dump(),
            "engine_hash": engine_hash(), "dependencies": numerical_dependencies(),
            "observation_count": len(scope.observations), "patient_count": sum(o.n for o in scope.observations),
            "parameter_shapes": {k: list(v.shape) for k, v in arrays.items()},
            "interaction_ids": [list(x) for x in __import__("phase1_prediction.multidrug.math", fromlist=["interaction_sets"]).interaction_sets(len(scope.drugs))]})
