"""Publication safety and portable evidence/cache identity, without live APIs."""
from __future__ import annotations

import hashlib
import gzip
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pytest

from phase1_extraction.artifacts import ArticlePaths, write_json
from phase1_extraction.models import (
    CohortExposure, DLTObservation, Intervention, PopulationCount, Regimen, RegimenComponent,
)
from phase1_extraction.resolver import resolve_extraction
from phase1_prediction.evidence import load_evidence_library
from phase1_prediction.multidrug.contracts import PriorSpec, SamplerSpec
from phase1_prediction.multidrug.observations import load_model_library
from phase1_prediction.multidrug.store import PosteriorStore, engine_hash, numerical_dependencies, source_fit_id
from phase1_project.shared_release import (
    install_shared_library, pack_shared_library, resolve_shared_release,
)
from test_resolver import base_extraction


def article(root: Path, document_id="DOC", *, complete=True):
    extraction = base_extraction(doses=(50, 100))
    extraction.document_id = document_id
    extraction.extraction_complete = complete
    extraction.interventions = [Intervention(intervention_id="drug", name_raw="Example drug")]
    extraction.regimens = [Regimen(regimen_id="regimen", label="Example regimen",
        components=[RegimenComponent(component_id="drug", intervention_id="drug", role="escalated")])]
    for index, cohort in enumerate(extraction.cohorts):
        cohort.exposures = [CohortExposure(exposure_id=f"ex{index}", name_raw="Example drug",
            component_id="drug", intervention_id="drug", presence="present", role="escalated",
            dose=cohort.dose, dose_status="reported")]
        extraction.population_counts.append(PopulationCount(count_id=f"n{index}", study_id="study",
            cohort_id=cohort.cohort_id, role="paired_dlt_denominator", reported_value=3))
        extraction.dlt_observations.append(DLTObservation(observation_id=f"o{index}", study_id="study",
            result_scope_id="scope", cohort_id=cohort.cohort_id, count_kind="patients",
            explicit_patient_count=index, denominator_count_id=f"n{index}"))
    paths = ArticlePaths(root, document_id)
    rows = resolve_extraction(extraction).curve_rows
    write_json(paths.reported_extraction, extraction)
    write_json(paths.curve_rows, {
        "project_default": [row.model_dump(mode="json") for row in rows if row.included_in_default],
        "reported_only": [row.model_dump(mode="json") for row in rows if row.included_in_reported_only],
        "all_rows": [row.model_dump(mode="json") for row in rows],
    })
    return paths


def fit(library: Path, root: Path):
    scopes, _ = load_model_library([library])
    scope = scopes[0]
    prior, sampler = PriorSpec(), SamplerSpec()
    identity = source_fit_id(scope, "logistic", prior, sampler)
    arrays = {"a": np.zeros((2, 4, len(scope.drugs))),
              "log_b": np.zeros((2, 4, len(scope.drugs))), "eta": np.zeros((2, 4, 0))}
    PosteriorStore(root).save(identity, scope, arrays, {
        "backend": "logistic", "status": "fitted", "prior": prior.model_dump(),
        "sampler": sampler.model_dump(), "engine_hash": engine_hash(),
        "dependencies": numerical_dependencies(),
    })
    return identity, scope, arrays


def contents(bundle):
    with tarfile.open(bundle) as archive:
        return [(member, archive.extractfile(member).read()) for member in archive]


def archive_at(path, records):
    with tarfile.open(path, "w:gz") as archive:
        for member, data in records:
            archive.addfile(member, io.BytesIO(data) if member.isfile() else None)


def test_allowlist_excludes_credentials_raw_documents_reviews_and_user_state(tmp_path):
    root = tmp_path / "local"
    paths = article(root)
    private = "sk-" + "x" * 50
    for name in (".env", "request_summary.json", "raw_response.json", "human_review.json",
                 "friendly_human_review.json", "article.pdf", "target_trial_draft.json"):
        (paths.root / name).write_text(private)
    (root / "owner.env").write_text(private)
    article(root, "INCOMPLETE", complete=False)
    bundle = tmp_path / "shared.tar.gz"
    report = pack_shared_library([root], bundle)
    assert report["document_count"] == 1
    assert report["skipped"] == [{"document_id": "INCOMPLETE", "reason": "extraction_incomplete"}]
    files = contents(bundle)
    assert {member.name for member, _ in files} == {
        "release.json", "library/DOC/reported_extraction.json", "library/DOC/curve_rows.json"}
    assert all(private.encode() not in data for _, data in files)
    installed = install_shared_library(bundle, tmp_path / "server", expected_sha256=report["bundle_sha256"])
    assert load_evidence_library([installed["library_root"]]).fingerprint == load_evidence_library([paths.root]).fingerprint
    assert load_evidence_library([installed["library_root"]]).document_count == 1


def test_incomplete_only_library_fails(tmp_path):
    article(tmp_path / "inputs", complete=False)
    with pytest.raises(ValueError, match="No completed"):
        pack_shared_library([tmp_path / "inputs"], tmp_path / "bundle.tar.gz")


def test_owner_can_publish_partial_coverage_without_rewriting_scientific_artifacts(tmp_path):
    paths = article(tmp_path / "inputs", complete=False)
    original = {name: (paths.root / name).read_bytes()
                for name in ("reported_extraction.json", "curve_rows.json")}
    bundle = tmp_path / "bundle.tar.gz"
    report = pack_shared_library([paths.root], bundle, include_incomplete=True)
    assert report["document_count"] == 1 and report["skipped"] == []
    assert report["incomplete_documents"] == ["DOC"]
    installed = install_shared_library(bundle, tmp_path / "server", expected_sha256=report["bundle_sha256"])
    for name, data in original.items():
        assert (Path(installed["library_root"]) / "DOC" / name).read_bytes() == data
    manifest = json.loads((Path(installed["library_root"]).parent / "release.json").read_text())
    assert manifest["incomplete_documents"] == ["DOC"]
    assert load_evidence_library([installed["library_root"]]).fingerprint == load_evidence_library([paths.root]).fingerprint


def test_partial_coverage_cannot_be_installed_without_explicit_manifest_declaration(tmp_path):
    from phase1_project.shared_release import _canonical
    article(tmp_path / "inputs", complete=False)
    bundle = tmp_path / "bundle.tar.gz"
    pack_shared_library([tmp_path / "inputs"], bundle, include_incomplete=True)
    records = contents(bundle)
    member, data = records[0]
    manifest = json.loads(data)
    manifest.pop("incomplete_documents")
    manifest.pop("release_id")
    manifest["release_id"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    data = _canonical(manifest)
    member.size = len(data)
    records[0] = (member, data)
    archive_at(tmp_path / "undeclared.tar.gz", records)
    with pytest.raises(ValueError, match="completed extraction"):
        install_shared_library(tmp_path / "undeclared.tar.gz", tmp_path / "server")


def test_evidence_and_cache_identity_survive_root_scrubbing_and_relocation(tmp_path, monkeypatch):
    original, cache = tmp_path / "local", tmp_path / "cache"
    article(original)
    identity, source, arrays = fit(original, cache)
    original_files = {name: (cache / identity / name).read_bytes()
                      for name in ("manifest.json", "scope.json", "draws.npz")}
    bundle = tmp_path / "bundle.tar.gz"
    report = pack_shared_library([original], bundle, posterior_roots=[cache, cache])
    installed = install_shared_library(bundle, tmp_path / "server")
    assert report["source_posterior_count"] == 1
    assert load_evidence_library([installed["library_root"]]).fingerprint == load_evidence_library([original]).fingerprint
    relocated, _ = load_model_library([installed["library_root"]])
    assert source_fit_id(relocated[0], "logistic", PriorSpec(), SamplerSpec()) == identity
    for name, data in original_files.items():
        assert (cache / identity / name).read_bytes() == data
    installed_cache = Path(installed["posterior_root"]) / identity
    assert (installed_cache / "draws.npz").read_bytes() == original_files["draws.npz"]
    assert (installed_cache / "scope.json").read_bytes() != original_files["scope.json"]
    loaded, restored_arrays, manifest = PosteriorStore(installed["posterior_root"]).load(identity)
    expected_scope = source.model_copy(deep=True)
    expected_scope.lineage.pop("root")
    assert loaded == expected_scope and manifest["dependencies"] == numerical_dependencies()
    expected_manifest = json.loads(original_files["manifest.json"])
    expected_manifest["scope_sha256"] = hashlib.sha256((installed_cache / "scope.json").read_bytes()).hexdigest()
    assert manifest == expected_manifest
    assert all(str(original).encode() not in data for _, data in contents(bundle))
    np.testing.assert_array_equal(restored_arrays["a"], arrays["a"])
    from phase1_prediction.multidrug import fitting

    def must_not_refit(*args, **kwargs):
        raise AssertionError("Relocated matching posterior should be reused without sampling")

    monkeypatch.setattr(fitting, "fit_source", must_not_refit)
    writable_store = tmp_path / "session_posteriors"
    reused, reused_arrays, reused_manifest = PosteriorStore(
        writable_store, fallback_roots=[installed["posterior_root"]]).fit(
        relocated[0], "logistic", prior=PriorSpec(), sampler=SamplerSpec())
    assert reused == expected_scope and reused_manifest["source_fit_id"] == identity
    np.testing.assert_array_equal(reused_arrays["a"], arrays["a"])
    assert not writable_store.exists()
    assert not list(Path(installed["posterior_root"]).glob("*.lock"))


def test_install_and_pack_are_idempotent_and_old_release_remains(tmp_path):
    root, destination = tmp_path / "local", tmp_path / "server"
    article(root)
    bundle = tmp_path / "bundle.tar.gz"
    first = pack_shared_library([root], bundle)
    second = pack_shared_library([root], tmp_path / "again.tar.gz")
    assert first["release_id"] == second["release_id"]
    old = install_shared_library(bundle, destination)
    repeated = install_shared_library(bundle, destination)
    assert repeated["already_installed"] is True
    assert repeated["release_path"] == old["release_path"]
    article(root, "SECOND")
    pack_shared_library([root], bundle)
    new = install_shared_library(bundle, destination)
    assert resolve_shared_release(destination) == Path(new["release_path"])
    assert Path(old["release_path"]).is_dir()
    assert new["release_id"] != old["release_id"]


def test_single_drug_projected_cache_survives_publication(tmp_path, monkeypatch):
    from phase1_prediction.multidrug.single_drug import project_scope
    from phase1_prediction.multidrug.prior_presets import canonical_prior
    from phase1_prediction.multidrug import fitting

    original, cache = tmp_path / 'local', tmp_path / 'cache'
    article(original)
    scopes, _ = load_model_library([original])
    source = project_scope(scopes[0], scopes[0].drugs[0].model_drug_id)[0]
    backend = 'single_drug_logistic'
    prior, sampler = canonical_prior(None, backend), SamplerSpec()
    identity = source_fit_id(source, backend, prior, sampler)
    arrays = {'alpha': np.zeros((2, 4)), 'delta': np.full((2, 4), .5),
              'log_b': np.zeros((2, 4, 2)), 'eta': np.zeros((2, 4, 0))}
    PosteriorStore(cache).save(identity, source, arrays, {
        'backend': backend, 'status': 'fitted', 'prior': prior.model_dump(),
        'sampler': sampler.model_dump(), 'engine_hash': engine_hash(),
        'dependencies': numerical_dependencies(),
    })
    bundle = tmp_path / 'single.tar.gz'
    report = pack_shared_library([original], bundle, posterior_roots=[cache])
    assert report['source_posterior_count'] == 1
    installed = install_shared_library(bundle, tmp_path / 'server')
    relocated, _ = load_model_library([installed['library_root']])
    projection = project_scope(relocated[0], relocated[0].drugs[0].model_drug_id)[0]
    assert source_fit_id(projection, backend, prior, sampler) == identity
    def no_refit(*args, **kwargs):
        pytest.fail('A compatible published single-drug posterior must be reused')
    monkeypatch.setattr(fitting, 'fit_source', no_refit)
    _, restored, _ = PosteriorStore(tmp_path / 'session',
        fallback_roots=[installed['posterior_root']]).fit(projection, backend, prior=prior, sampler=sampler)
    np.testing.assert_array_equal(restored['delta'], arrays['delta'])
    assert not (tmp_path / 'session').exists()


@pytest.mark.parametrize("attack", ["traversal", "absolute", "symlink", "hardlink", "duplicate",
                                  "extra_secret", "missing", "checksum", "size"])
def test_malicious_archives_never_change_active_release(tmp_path, attack):
    article(tmp_path / "local")
    bundle, destination = tmp_path / "bundle.tar.gz", tmp_path / "server"
    pack_shared_library([tmp_path / "local"], bundle)
    installed = install_shared_library(bundle, destination)
    pointer = (destination / "current.json").read_bytes()
    records = contents(bundle)
    if attack == "missing":
        records.pop()
    elif attack == "checksum":
        member, data = records[-1]
        records[-1] = member, data.replace(b'"DOC"', b'"BAD"')
    elif attack == "size":
        member, data = records[-1]
        member.size += 1
        records[-1] = member, data + b" "
    elif attack == "duplicate":
        records.append(records[-1])
    else:
        names = {"traversal": "../../escaped", "absolute": str(tmp_path / "escaped"),
                 "symlink": "library/DOC/curve_rows.json", "hardlink": "library/DOC/curve_rows.json",
                 "extra_secret": "library/DOC/.env"}
        member = tarfile.TarInfo(names[attack])
        if attack in {"symlink", "hardlink"}:
            member.type = tarfile.SYMTYPE if attack == "symlink" else tarfile.LNKTYPE
            member.linkname = str(tmp_path / "escaped")
        else:
            member.size = 1
        records.append((member, b"x"))
    malicious = tmp_path / "malicious.tar.gz"
    archive_at(malicious, records)
    with pytest.raises(ValueError):
        install_shared_library(malicious, destination)
    assert (destination / "current.json").read_bytes() == pointer
    assert resolve_shared_release(destination) == Path(installed["release_path"])
    assert not (tmp_path / "escaped").exists()
    assert not list((destination / "releases").glob(".install-*"))


def test_limits_manifest_path_and_expected_digest_fail_closed(tmp_path):
    article(tmp_path / "local")
    bundle = tmp_path / "bundle.tar.gz"
    pack_shared_library([tmp_path / "local"], bundle)
    with pytest.raises(ValueError, match="SHA-256"):
        install_shared_library(bundle, tmp_path / "server", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="size limit"):
        install_shared_library(bundle, tmp_path / "server", max_total_bytes=1)
    records = contents(bundle)
    manifest = json.loads(records[0][1])
    manifest["files"][0]["path"] = "library/../../escape"
    manifest.pop("release_id")
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    manifest["release_id"] = hashlib.sha256(canonical(manifest)).hexdigest()
    data = canonical(manifest)
    records[0][0].size = len(data)
    records[0] = records[0][0], data
    archive_at(bundle, records)
    with pytest.raises(ValueError, match="Unsafe"):
        install_shared_library(bundle, tmp_path / "server")


def test_scope_checksum_corruption_is_rejected_without_rewriting_local_cache(tmp_path):
    article(tmp_path / "local")
    identity, *_ = fit(tmp_path / "local", tmp_path / "cache")
    scope = tmp_path / "cache" / identity / "scope.json"
    original = scope.read_bytes()
    # Whitespace does not change source identity but must invalidate the byte checksum.
    scope.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="checksum"):
        pack_shared_library([tmp_path / "local"], tmp_path / "bundle.tar.gz", posterior_roots=[tmp_path / "cache"])
    assert scope.read_bytes() == original + b" "
    assert not (tmp_path / "bundle.tar.gz").exists()


def test_source_symlinks_conflicts_and_embedded_credentials_are_rejected(tmp_path):
    paths = article(tmp_path / "local")
    path = paths.reported_extraction
    data = json.loads(path.read_bytes())
    data["title"] = "sk-" + "X" * 40
    write_json(path, data)
    with pytest.raises(ValueError, match="credential"):
        pack_shared_library([paths.root], tmp_path / "bundle.tar.gz")
    paths = article(tmp_path / "local")
    backup = paths.root / "backup.json"
    path.rename(backup)
    path.symlink_to(backup)
    with pytest.raises(ValueError, match="symbolic"):
        pack_shared_library([paths.root], tmp_path / "bundle.tar.gz")


def test_unrelated_caches_are_not_published(tmp_path):
    article(tmp_path / "public")
    article(tmp_path / "private", "PRIVATE")
    identity, *_ = fit(tmp_path / "private", tmp_path / "cache")
    report = pack_shared_library([tmp_path / "public"], tmp_path / "bundle.tar.gz", posterior_roots=[tmp_path / "cache"])
    assert report["source_posterior_count"] == 0
    assert report["skipped"][0]["source_fit_id"] == identity


def test_oversized_pax_metadata_is_rejected_before_expansion(tmp_path):
    member = tarfile.TarInfo("pax-metadata")
    member.type, member.size = tarfile.XHDTYPE, 2**40
    bundle = tmp_path / "pax.tar.gz"
    with gzip.open(bundle, "wb") as writer:
        writer.write(member.tobuf(format=tarfile.GNU_FORMAT))
    with pytest.raises(ValueError, match="regular USTAR"):
        install_shared_library(bundle, tmp_path / "server")


def test_false_large_array_shape_is_rejected_before_numpy_allocation(tmp_path):
    article(tmp_path / "local")
    identity, *_ = fit(tmp_path / "local", tmp_path / "cache")
    directory = tmp_path / "cache" / identity
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": "<f8", "fortran_order": False, "shape": (2**45,),
    })
    with zipfile.ZipFile(directory / "draws.npz", "w") as writer:
        writer.writestr("a.npy", header.getvalue())
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["draws_sha256"] = hashlib.sha256((directory / "draws.npz").read_bytes()).hexdigest()
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="array type, shape or size"):
        pack_shared_library([tmp_path / "local"], tmp_path / "bundle.tar.gz", posterior_roots=[tmp_path / "cache"])


@pytest.mark.parametrize("private_value", ["/Users/owner/private/paper.pdf", "/home/owner/paper.pdf",
                                         r"C:\Users\owner\paper.pdf", "file:///private/owner/paper.pdf"])
def test_local_paths_in_scientific_free_text_fail_without_rewriting_provenance(tmp_path, private_value):
    paths = article(tmp_path / "local")
    value = json.loads(paths.reported_extraction.read_bytes())
    value["coverage_notes"] = [f"Source location: {private_value}"]
    write_json(paths.reported_extraction, value)
    original_bytes = paths.reported_extraction.read_bytes()
    with pytest.raises(ValueError, match="Local filesystem path") as caught:
        pack_shared_library([paths.root], tmp_path / "bundle.tar.gz")
    assert private_value not in str(caught.value)
    assert paths.reported_extraction.read_bytes() == original_bytes


def test_environment_keys_are_rejected_but_public_evidence_locators_remain_unchanged(tmp_path):
    paths = article(tmp_path / "local")
    value = json.loads(paths.reported_extraction.read_bytes())
    value["cohorts"][0]["exposures"][0]["evidence"] = [{
        "source_ids": ["table1"], "source_text": "Example drug 50 mg",
        "locator": "https://example.org/papers/DOC#table1", "support_type": "table"}]
    write_json(paths.reported_extraction, value)
    original_bytes = paths.reported_extraction.read_bytes()
    bundle = tmp_path / "bundle.tar.gz"
    pack_shared_library([paths.root], bundle)
    exported = dict((member.name, data) for member, data in contents(bundle))
    assert exported["library/DOC/reported_extraction.json"] == original_bytes
    value["OPENAI_API_KEY"] = "owner-private-value"
    write_json(paths.reported_extraction, value)
    with pytest.raises(ValueError, match="Credential") as caught:
        pack_shared_library([paths.root], tmp_path / "invalid.tar.gz")
    assert "owner-private-value" not in str(caught.value)
