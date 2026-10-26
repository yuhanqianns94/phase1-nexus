"""Portable, allowlisted releases of completed evidence and source posterior caches.

Only resolved clinical JSON and validated, source-only fits are exported. Secrets,
raw model responses, original documents, reviews and personal trial plans are
never selected. A bundle is integrity checked before the active release changes.
"""
from __future__ import annotations

import hashlib
import gzip
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import zipfile


SCHEMA_VERSION = "phase1_shared_library_v1"
LIBRARY_FILES = ("reported_extraction.json", "curve_rows.json")
POSTERIOR_FILES = ("manifest.json", "scope.json", "draws.npz")
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_FILES = 50_000
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_FIT_ID = re.compile(r"fit_[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_KEYS = {
    "apikey", "openaiapikey", "authorization", "accesstoken", "refreshtoken",
    "privatekey", "clientsecret", "rawresponse", "rawresponses",
    "environmentvariables", "environmentvars", "envvars", "openaibaseurl",
}
_SECRET_VALUE = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
_LOCAL_PATH = re.compile(
    r"(?i)(?:file://|(?<![\w:/])/(?:Users|home|private|tmp|var|opt|mnt|media|Volumes|root)/"
    r"|(?<!\w)[A-Z]:[\\/]|(?<!\w)~[/\\])"
)
_SECRET_ASSIGNMENT = re.compile(r"\b[A-Z][A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|PRIVATE_KEY|CLIENT_SECRET)\s*=")
CACHE_POLICY = (
    "Source posterior caches are reused only when source data, prior, sampler, "
    "model code and numerical dependency versions match exactly. Similarity "
    "weights and target predictions depend on the target trial and are not "
    "universal cached parameters. Target priors and personal plans are excluded."
)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(data: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("Nonfinite JSON number")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)


def _check_public_json(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if re.sub(r"[^a-z]", "", key.lower()) in _SENSITIVE_KEYS:
                raise ValueError("Credential or raw-response field in a public artifact")
            _check_public_json(item)
    elif isinstance(value, list):
        for item in value:
            _check_public_json(item)
    elif isinstance(value, str) and (_SECRET_VALUE.search(value) or _SECRET_ASSIGNMENT.search(value)):
        # Do not repeat a matching value in logs or the publication report.
        raise ValueError("Possible credential in a public artifact")


def _check_no_local_paths(value, *, ignored_path=None, location=()) -> None:
    if location == ignored_path:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _check_no_local_paths(item, ignored_path=ignored_path, location=(*location, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_no_local_paths(item, ignored_path=ignored_path, location=(*location, index))
    elif isinstance(value, str) and _LOCAL_PATH.search(value):
        # Clinical free text and evidence locators are not silently rewritten:
        # their original bytes are part of scientific/source cache identity.
        raise ValueError("Local filesystem path in a public scientific artifact")


def _regular(path: Path, *, limit: int = MAX_FILE_BYTES) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Release inputs must be regular files, not symbolic links")
    if path.stat().st_size > limit:
        raise ValueError("Release file exceeds the size limit")


def _read_json(path: Path):
    _regular(path, limit=MAX_JSON_BYTES)
    value = _json_bytes(path.read_bytes())
    _check_public_json(value)
    return value


def _safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or
            str(path) != name or any(not _COMPONENT.fullmatch(p) for p in path.parts)):
        raise ValueError("Unsafe release path")
    if any(p in {".", ".."} for p in path.parts):
        raise ValueError("Unsafe release path")
    return name


def _allowlisted(name: str) -> bool:
    _safe_relative(name)
    parts = PurePosixPath(name).parts
    if len(parts) != 3:
        return False
    section, identity, filename = parts
    return ((section == "library" and filename in LIBRARY_FILES) or
            (section == "source_posteriors" and bool(_FIT_ID.fullmatch(identity)) and filename in POSTERIOR_FILES))


def _validate_article(directory: Path, *, require_complete: bool = True) -> bool:
    from phase1_extraction.models import CurveRow, ReportedExtraction

    extraction_payload = _read_json(directory / LIBRARY_FILES[0])
    _check_no_local_paths(extraction_payload)
    extraction = ReportedExtraction.model_validate(extraction_payload)
    if extraction.document_id != directory.name:
        raise ValueError("Document directory and extraction identity differ")
    rows = _read_json(directory / LIBRARY_FILES[1])
    _check_no_local_paths(rows)
    row_keys = {"project_default", "reported_only", "all_rows", "model_outcome_rows"}
    if (not isinstance(rows, dict) or set(rows) - row_keys - {"policy_version"}
            or not {"project_default", "reported_only"} <= set(rows)):
        raise ValueError("Unsupported curve-row artifact")
    for key in row_keys & rows.keys():
        if not isinstance(rows[key], list):
            raise ValueError("Curve rows must be lists")
        for raw in rows[key]:
            if CurveRow.model_validate(raw).document_id != extraction.document_id:
                raise ValueError("Curve-row document identity mismatch")
    if require_complete and not extraction.extraction_complete:
        raise ValueError("Shared releases require completed extraction artifacts")
    return extraction.extraction_complete


def _scope_identity(scope):
    value = scope.model_dump(mode="json")
    value["lineage"].pop("root", None)
    return value


def _library_scopes(library: Path) -> set[str]:
    from phase1_prediction.multidrug.contracts import fingerprint
    from phase1_prediction.multidrug.observations import load_model_library
    from phase1_prediction.multidrug.single_drug import project_scope

    identities = set()
    for policy in ("project_default", "reported_only"):
        scopes, _ = load_model_library([library], evidence_policy=policy)
        identities.update(fingerprint(_scope_identity(scope)) for scope in scopes)
        for scope in scopes:
            for drug in scope.drugs:
                if drug.mode == "numeric":
                    try:
                        projected = project_scope(scope, drug.model_drug_id)
                    except ValueError:
                        continue
                    identities.update(fingerprint(_scope_identity(item)) for item in projected)
    return identities


def _validate_posterior(directory: Path, allowed_scopes: set[str], *, allow_local_root=False) -> None:
    from phase1_prediction.multidrug import VERSION, validate_backend
    from phase1_prediction.multidrug.contracts import ModelScope, PriorSpec, SamplerSpec, fingerprint
    from phase1_prediction.multidrug.store import PosteriorStore
    from phase1_prediction.multidrug import store as store_module

    fit_id = directory.name
    if not _FIT_ID.fullmatch(fit_id):
        raise ValueError("Invalid source posterior identity")
    manifest = _read_json(directory / "manifest.json")
    scope_payload = _read_json(directory / "scope.json")
    _check_no_local_paths(manifest)
    _check_no_local_paths(scope_payload, ignored_path=("lineage", "root") if allow_local_root else None)
    if not allow_local_root and "root" in scope_payload.get("lineage", {}):
        raise ValueError("Published posterior scope must not contain a local root")
    scope = ModelScope.model_validate(scope_payload)
    _regular(directory / "draws.npz")
    validate_backend(manifest.get("backend"))
    if manifest.get("model_version") != VERSION or manifest.get("status") not in {"fitted", "prior_only"}:
        raise ValueError("Source posterior is not a completed compatible fit")
    if not _SHA256.fullmatch(str(manifest.get("engine_hash", ""))):
        raise ValueError("Missing source posterior engine fingerprint")
    dependencies = manifest.get("dependencies")
    if (not isinstance(dependencies, dict) or not dependencies or
            any(not isinstance(k, str) or not isinstance(v, str) for k, v in dependencies.items())):
        raise ValueError("Missing numerical dependency versions")
    engine_root = Path(store_module.__file__).parent
    source_hashes = {name: _sha(engine_root / name) for name in ("math.py", "fitting.py", "contracts.py", "single_drug.py")}
    if fingerprint({"sources": source_hashes, "dependencies": dependencies}) != manifest["engine_hash"]:
        raise ValueError("Source posterior engine metadata is incompatible with the current model code")
    PriorSpec.model_validate(manifest.get("prior"))
    SamplerSpec.model_validate(manifest.get("sampler"))
    source = _scope_identity(scope)
    expected = "fit_" + fingerprint({"scope": source, "backend": manifest["backend"],
        "prior": manifest["prior"], "sampler": manifest["sampler"], "version": VERSION,
        "engine": manifest["engine_hash"]})[:32]
    if expected != fit_id or manifest.get("source_fit_id") != fit_id:
        raise ValueError("Source posterior content does not match its identity")
    if fingerprint(source) not in allowed_scopes:
        raise ValueError("Source posterior does not match the published library")
    # Validate archive sizes before np.load so nested compression cannot bypass
    # the bundle limits. Pickle/object arrays are rejected by PosteriorStore.
    with zipfile.ZipFile(directory / "draws.npz") as archive:
        entries = archive.infolist()
        if (len(entries) > 256 or sum(e.file_size for e in entries) > MAX_FILE_BYTES
                or len({e.filename for e in entries}) != len(entries)
                or any(e.is_dir() or not re.fullmatch(r"[A-Za-z0-9_]+\.npy", e.filename) for e in entries)):
            raise ValueError("Invalid or oversized posterior array archive")
        from numpy.lib import format as npy_format

        for entry in entries:
            with archive.open(entry) as array_file:
                version = npy_format.read_magic(array_file)
                if version not in {(1, 0), (2, 0)}:
                    raise ValueError("Unsupported posterior array format")
                header_reader = (npy_format.read_array_header_1_0 if version == (1, 0)
                                 else npy_format.read_array_header_2_0)
                shape, _, dtype = header_reader(array_file, max_header_size=10_000)
                payload_bytes = math.prod(shape) * dtype.itemsize
                if (dtype.hasobject or dtype.kind not in "biufc" or payload_bytes > MAX_FILE_BYTES
                        or array_file.tell() + payload_bytes != entry.file_size):
                    raise ValueError("Invalid posterior array type, shape or size")
    # Check the original scope bytes before any staged privacy transformation.
    # A corrupt checksum must never be repaired by the publication process.
    _, arrays, _ = PosteriorStore(directory.parent).load(fit_id)
    import numpy as np

    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("Source posterior contains nonfinite values")
    shapes = manifest.get("parameter_shapes")
    if shapes is not None and shapes != {name: list(array.shape) for name, array in arrays.items()}:
        raise ValueError("Source posterior parameter shapes do not match its manifest")


def _remove_staged_local_root(directory: Path) -> None:
    """Remove only non-scientific root metadata after full original validation."""
    scope_path, manifest_path = directory / "scope.json", directory / "manifest.json"
    scope = _read_json(scope_path)
    if "root" not in scope.get("lineage", {}):
        return
    del scope["lineage"]["root"]
    _check_no_local_paths(scope)
    scope_bytes = _canonical(scope)
    manifest = _read_json(manifest_path)
    manifest["scope_sha256"] = hashlib.sha256(scope_bytes).hexdigest()
    scope_path.write_bytes(scope_bytes)
    manifest_path.write_bytes(_canonical(manifest))


def _snapshot(source: Path, destination: Path) -> None:
    _regular(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 1024 * 1024)
    _regular(destination)


def _manifest_for(directory: Path, paths: list[str], documents: list[str], *, incomplete_documents=()) -> dict:
    records = [{"path": name, "size": (directory / name).stat().st_size,
                "sha256": _sha(directory / name)} for name in sorted(paths)]
    if len(records) > MAX_FILES or sum(r["size"] for r in records) > MAX_TOTAL_BYTES:
        raise ValueError("Shared release exceeds the size limit")
    payload = {"schema_version": SCHEMA_VERSION, "documents": sorted(documents),
               "files": records, "cache_policy": CACHE_POLICY}
    if incomplete_documents:
        payload["incomplete_documents"] = sorted(incomplete_documents)
    return {**payload, "release_id": hashlib.sha256(_canonical(payload)).hexdigest()}


def pack_shared_library(extraction_roots, output, *, posterior_roots=(), include_incomplete=False) -> dict:
    """Build a bundle without modifying source data or uploading anything.

    Each extraction root is either one document directory or a directory whose
    immediate children are documents. Incomplete results are omitted and listed
    in the report unless the owner explicitly includes them. Their original
    completeness flags, missing values and row exclusions remain unchanged.
    Conflicting duplicate document IDs fail closed.
    """
    output = Path(output).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    skipped, documents, paths, original_cache_hashes = [], [], [], {}
    incomplete_documents = []
    with tempfile.TemporaryDirectory(prefix=".shared-pack-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for raw_root in extraction_roots:
            root = Path(raw_root).expanduser().absolute()
            if root.is_symlink() or not root.is_dir():
                raise ValueError("Extraction roots must be real directories")
            candidates = [root] if (root / LIBRARY_FILES[0]).exists() else sorted(root.iterdir())
            for article in candidates:
                if article.is_symlink():
                    raise ValueError("Symbolic document directories are not permitted")
                if not article.is_dir() or not (article / LIBRARY_FILES[0]).exists():
                    continue
                if not _COMPONENT.fullmatch(article.name):
                    raise ValueError("Unsafe document identity")
                if not (article / LIBRARY_FILES[1]).exists():
                    skipped.append({"document_id": article.name, "reason": "missing_curve_rows"})
                    continue
                target = staging / "library" / article.name
                if target.exists():
                    if all(_sha(target / f) == _sha(article / f) for f in LIBRARY_FILES):
                        continue
                    raise ValueError("Conflicting duplicate document identity")
                for filename in LIBRARY_FILES:
                    _snapshot(article / filename, target / filename)
                if not _validate_article(target, require_complete=False):
                    if not include_incomplete:
                        skipped.append({"document_id": article.name, "reason": "extraction_incomplete"})
                        shutil.rmtree(target)
                        continue
                    incomplete_documents.append(article.name)
                documents.append(article.name)
                paths.extend(f"library/{article.name}/{f}" for f in LIBRARY_FILES)
        if not documents:
            raise ValueError("No completed extraction artifacts are available to publish")
        allowed_scopes = _library_scopes(staging / "library") if posterior_roots else set()
        for raw_root in posterior_roots:
            root = Path(raw_root).expanduser().absolute()
            if root.is_symlink() or not root.is_dir():
                raise ValueError("Posterior roots must be real directories")
            for fit in sorted(root.iterdir()):
                if not _FIT_ID.fullmatch(fit.name):
                    continue
                if fit.is_symlink() or not fit.is_dir():
                    raise ValueError("Source posterior directories must not be symbolic links")
                target = staging / "source_posteriors" / fit.name
                if target.exists():
                    if original_cache_hashes.get(fit.name) == tuple(_sha(fit / f) for f in POSTERIOR_FILES):
                        continue
                    raise ValueError("Conflicting source posterior cache identity")
                for filename in POSTERIOR_FILES:
                    _snapshot(fit / filename, target / filename)
                try:
                    _validate_posterior(target, allowed_scopes, allow_local_root=True)
                except ValueError as error:
                    # Old model versions and unrelated library caches are not
                    # published. Integrity failures must stop publication.
                    if str(error) not in {"Source posterior is not a completed compatible fit",
                                          "Source posterior does not match the published library",
                                          "Source posterior engine metadata is incompatible with the current model code"}:
                        raise
                    skipped.append({"source_fit_id": fit.name, "reason": str(error)})
                    shutil.rmtree(target)
                    continue
                original_cache_hashes[fit.name] = tuple(_sha(target / f) for f in POSTERIOR_FILES)
                _remove_staged_local_root(target)
                paths.extend(f"source_posteriors/{fit.name}/{f}" for f in POSTERIOR_FILES)
        manifest = _manifest_for(staging, paths, documents, incomplete_documents=incomplete_documents)
        manifest_bytes = _canonical(manifest)
        bundle = staging / "bundle.tar.gz"
        with tarfile.open(bundle, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
            header = tarfile.TarInfo("release.json")
            header.size, header.mode = len(manifest_bytes), 0o644
            archive.addfile(header, io.BytesIO(manifest_bytes))
            for name in sorted(paths):
                header = tarfile.TarInfo(name)
                header.size, header.mode = (staging / name).stat().st_size, 0o644
                with (staging / name).open("rb") as reader:
                    archive.addfile(header, reader)
        os.replace(bundle, output)
    return {"bundle": str(output), "bundle_sha256": _sha(output),
            "release_id": manifest["release_id"], "document_count": len(documents),
            "source_posterior_count": sum(name.endswith("/draws.npz") for name in paths),
            "uncompressed_bytes": sum(item["size"] for item in manifest["files"]),
            "skipped": skipped, "incomplete_documents": sorted(incomplete_documents), "cache_policy": CACHE_POLICY}


def _validate_manifest(manifest: dict, *, max_total_bytes: int) -> dict[str, dict]:
    required = {"schema_version", "documents", "files", "cache_policy", "release_id"}
    if (not isinstance(manifest, dict) or not required <= set(manifest)
            or set(manifest) - required - {"incomplete_documents"}
            or manifest["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Unsupported shared release manifest")
    payload = {key: value for key, value in manifest.items() if key != "release_id"}
    if manifest["release_id"] != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("Shared release identity checksum mismatch")
    records = manifest["files"]
    if not isinstance(records, list) or not records or len(records) > MAX_FILES:
        raise ValueError("Invalid shared release file count")
    expected, total = {}, 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("Invalid shared release file record")
        name, size, digest = record["path"], record["size"], record["sha256"]
        if (not isinstance(name, str) or not _allowlisted(name) or name in expected
                or type(size) is not int or size < 0 or size > MAX_FILE_BYTES
                or not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
            raise ValueError("Invalid or disallowed shared release file")
        total += size
        expected[name] = record
    if total > max_total_bytes:
        raise ValueError("Shared release exceeds the total size limit")
    documents = sorted({name.split("/")[1] for name in expected if name.startswith("library/")})
    if not documents or manifest["documents"] != documents:
        raise ValueError("Shared release document index mismatch")
    incomplete = manifest.get("incomplete_documents", [])
    if (not isinstance(incomplete, list) or any(not isinstance(d, str) for d in incomplete)
            or incomplete != sorted(set(incomplete)) or not set(incomplete) <= set(documents)):
        raise ValueError("Invalid incomplete-document index")
    for document in documents:
        if any(f"library/{document}/{f}" not in expected for f in LIBRARY_FILES):
            raise ValueError("Incomplete shared document")
    fits = {name.split("/")[1] for name in expected if name.startswith("source_posteriors/")}
    if any(f"source_posteriors/{fit}/{f}" not in expected for fit in fits for f in POSTERIOR_FILES):
        raise ValueError("Incomplete shared posterior")
    return expected


def _validate_installed(directory: Path, manifest: dict, expected: dict) -> None:
    stored_manifest = directory / "release.json"
    _regular(stored_manifest, limit=MAX_MANIFEST_BYTES)
    if _json_bytes(stored_manifest.read_bytes()) != manifest:
        raise ValueError("Installed release manifest mismatch")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symbolic links are not permitted in shared releases")
        if path.is_file() and path != directory / "release.json":
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(expected):
        raise ValueError("Shared release file inventory mismatch")
    for name, record in expected.items():
        path = directory / name
        _regular(path)
        if path.stat().st_size != record["size"] or _sha(path) != record["sha256"]:
            raise ValueError("Shared release file checksum mismatch")
    for document in manifest["documents"]:
        explicitly_incomplete = document in manifest.get("incomplete_documents", [])
        complete = _validate_article(directory / "library" / document, require_complete=not explicitly_incomplete)
        if explicitly_incomplete and complete:
            raise ValueError("Incomplete-document index disagrees with extraction coverage")
    fits = {name.split("/")[1] for name in expected if name.startswith("source_posteriors/")}
    scopes = _library_scopes(directory / "library") if fits else set()
    for fit in fits:
        _validate_posterior(directory / "source_posteriors" / fit, scopes)


def _read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated shared release archive")
    return data


def _tar_header(stream):
    """Read only plain tar headers; never expand unbounded PAX/GNU metadata."""
    header = _read_exact(stream, tarfile.BLOCKSIZE)
    if header == b"\0" * tarfile.BLOCKSIZE:
        if _read_exact(stream, tarfile.BLOCKSIZE) != b"\0" * tarfile.BLOCKSIZE:
            raise ValueError("Invalid shared release archive terminator")
        # tarfile writes at most one trailing record of zero padding. Reject
        # concatenated archives and compressed streams with unbounded padding.
        trailer = stream.read(tarfile.RECORDSIZE + 1)
        if len(trailer) > tarfile.RECORDSIZE or trailer.strip(b"\0"):
            raise ValueError("Unexpected data after the shared release archive")
        return None
    try:
        member = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
    except tarfile.HeaderError as error:
        raise ValueError("Invalid shared release archive header") from error
    if not member.isfile() or member.sparse is not None:
        raise ValueError("Only regular USTAR files are permitted in shared releases")
    return member


def _tar_padding(stream, size: int) -> None:
    padding = (-size) % tarfile.BLOCKSIZE
    if padding and _read_exact(stream, padding) != b"\0" * padding:
        raise ValueError("Invalid shared release archive padding")


def install_shared_library(bundle, destination, *, expected_sha256: str | None = None,
                           max_total_bytes: int = MAX_TOTAL_BYTES) -> dict:
    """Validate and atomically activate a versioned bundle; keep prior releases."""
    import fcntl

    bundle, destination = Path(bundle).expanduser().absolute(), Path(destination).expanduser().absolute()
    _regular(bundle, limit=max_total_bytes)
    if expected_sha256 is not None and (not _SHA256.fullmatch(expected_sha256) or _sha(bundle) != expected_sha256):
        raise ValueError("Bundle SHA-256 mismatch")
    if destination.is_symlink():
        raise ValueError("Shared release destination must not be a symbolic link")
    destination.mkdir(parents=True, exist_ok=True)
    releases = destination / "releases"
    if releases.is_symlink():
        raise ValueError("Shared releases directory must not be a symbolic link")
    releases.mkdir(exist_ok=True)
    # Staging and the final directory share a filesystem for atomic renames.
    with tempfile.TemporaryDirectory(prefix=".install-", dir=releases) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        with gzip.open(bundle, "rb") as archive:
            first = _tar_header(archive)
            if first is None or first.name != "release.json" or not 0 < first.size <= MAX_MANIFEST_BYTES:
                raise ValueError("Bundle must begin with a bounded release manifest")
            manifest_bytes = _read_exact(archive, first.size)
            _tar_padding(archive, first.size)
            manifest = _json_bytes(manifest_bytes)
            expected = _validate_manifest(manifest, max_total_bytes=max_total_bytes)
            (staging / "release.json").write_bytes(manifest_bytes)
            seen = set()
            while (member := _tar_header(archive)) is not None:
                if member.name not in expected or member.name in seen:
                    raise ValueError("Unexpected, duplicate or nonregular archive member")
                record = expected[member.name]
                if member.size != record["size"]:
                    raise ValueError("Archive member size mismatch")
                target = staging / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as writer:
                    remaining = member.size
                    while remaining:
                        chunk = _read_exact(archive, min(remaining, 1024 * 1024))
                        writer.write(chunk)
                        remaining -= len(chunk)
                _tar_padding(archive, member.size)
                seen.add(member.name)
            if seen != set(expected):
                raise ValueError("Bundle is missing required files")
        _validate_installed(staging, manifest, expected)
        relative = f"releases/{manifest['release_id']}"
        final = destination / relative
        lock_path = destination / ".install.lock"
        if lock_path.is_symlink():
            raise ValueError("Invalid shared release lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reused = final.exists()
            if final.is_symlink():
                raise ValueError("Invalid existing shared release")
            if reused:
                _validate_installed(final, manifest, expected)
            else:
                staging.rename(final)
            current = {"schema_version": SCHEMA_VERSION, "release_id": manifest["release_id"], "path": relative}
            fd, raw_pointer = tempfile.mkstemp(prefix=".current-", dir=destination)
            pointer = Path(raw_pointer)
            try:
                with os.fdopen(fd, "wb") as writer:
                    writer.write(_canonical(current))
                    writer.flush()
                    os.fsync(writer.fileno())
                os.replace(pointer, destination / "current.json")
            finally:
                pointer.unlink(missing_ok=True)
    return {"release_id": manifest["release_id"], "release_path": str(final),
            "library_root": str(final / "library"), "posterior_root": str(final / "source_posteriors"),
            "document_count": len(manifest["documents"]), "already_installed": reused,
            "cache_policy": CACHE_POLICY}


def resolve_shared_release(destination) -> Path | None:
    """Resolve the small active pointer without rehashing large arrays per rerun."""
    root = Path(destination).expanduser().absolute()
    pointer = root / "current.json"
    if not pointer.exists():
        return None
    _regular(pointer, limit=4096)
    current = _json_bytes(pointer.read_bytes())
    if (not isinstance(current, dict) or current.get("schema_version") != SCHEMA_VERSION
            or not _SHA256.fullmatch(str(current.get("release_id", "")))
            or current.get("path") != f"releases/{current['release_id']}"):
        raise ValueError("Invalid active shared release pointer")
    release = root / current["path"]
    if root.is_symlink() or (root / "releases").is_symlink() or release.is_symlink() or not release.is_dir():
        raise ValueError("Active shared release is unavailable")
    return release
