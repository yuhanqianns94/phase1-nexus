"""Download reusable PDFs through official PMC Cloud and OpenAlex APIs.

PMC does not expose a PDF for every article in its automated-retrieval
datasets.  In particular, some author manuscripts and text-mining-only records
provide XML/TXT but no PDF.  When an OpenAlex API key is supplied, those cases
may fall back to OpenAlex's official cached-content API.  Interactive article
pages and publisher PDF URLs are never scraped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote, unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .visual_assets import (
    PMC_CLOUD_HTTPS_BASE,
    _cloud_article_metadata,
    _cloud_object_record,
    _download_limited,
    _pmcid_from_root,
    _write_bytes_atomic,
    _write_json_atomic,
)


PMC_PDF_MANIFEST_VERSION = "phase1_pmc_pdf_assets_v1"
DEFAULT_MAX_PDF_BYTES = 150 * 1024 * 1024
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 0.75
DEFAULT_WORKERS = 4
DEFAULT_USER_AGENT = "phase1-pmc-pdf-archiver/1.0 (official PMC Cloud dataset)"
OPENALEX_API_BASE = "https://api.openalex.org"
OPENALEX_CONTENT_BASE = "https://content.openalex.org"
OPENALEX_API_KEY_ENV = "OPENALEX_API_KEY"
OPENALEX_RIGHTS_NOTICE = (
    "The PDF retains its original copyright; OpenAlex grants no additional "
    "reuse or redistribution rights."
)


class _OpenAlexUnavailable(Exception):
    """A normal, non-error reason why OpenAlex content cannot be downloaded."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _OpenAlexIntegrityError(ValueError):
    """OpenAlex metadata/content failed an identity or integrity check."""


def _retrying_session(
    *,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> requests.Session:
    """Return a GET-only session that retries transient PMC/S3 failures."""

    if retry_attempts < 0:
        raise ValueError("retry_attempts must be non-negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be non-negative")
    retry = Retry(
        total=retry_attempts,
        connect=retry_attempts,
        read=retry_attempts,
        status=retry_attempts,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(408, 429, 500, 502, 503, 504),
        backoff_factor=retry_backoff_seconds,
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_identity(
    xml_path: Path, explicit_pmcid: str | None = None
) -> tuple[str, str, str | None]:
    source_bytes = Path(xml_path).read_bytes()
    root = ET.fromstring(source_bytes)
    stem = Path(xml_path).stem.upper()
    filename_pmcid = stem if re.fullmatch(r"PMC\d+", stem) else None
    embedded_pmcid = _pmcid_from_root(root)
    if filename_pmcid is not None and filename_pmcid != embedded_pmcid:
        raise ValueError(
            "XML filename PMCID does not match the JATS article-id: "
            f"{filename_pmcid} != {embedded_pmcid}"
        )
    if explicit_pmcid is not None:
        normalized_explicit = _pmcid_from_root(root, explicit_pmcid)
        if normalized_explicit != embedded_pmcid:
            raise ValueError(
                "Explicit PMCID does not match the JATS article-id: "
                f"{normalized_explicit} != {embedded_pmcid}"
            )
    pmcid = embedded_pmcid
    source_version: str | None = None
    for element in root.iter():
        if (
            element.tag.rsplit("}", 1)[-1] == "article-id"
            and str(element.attrib.get("pub-id-type") or "").casefold() == "pmcid-ver"
        ):
            value = "".join(element.itertext()).strip().upper()
            if re.fullmatch(rf"{re.escape(pmcid)}\.\d+", value):
                source_version = value
            break
    return pmcid, hashlib.sha256(source_bytes).hexdigest(), source_version


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize_doi(value: object) -> str | None:
    text = unquote(str(value or "")).strip()
    text = re.sub(
        r"^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)",
        "",
        text,
        flags=re.I,
    )
    text = text.strip().rstrip(". ").casefold()
    return text if re.fullmatch(r"10\.\d{4,9}/\S+", text) else None


def _normalize_pmid(value: object) -> str | None:
    match = re.search(r"(?:^|/)(\d+)(?:/)?$", str(value or "").strip())
    return match.group(1) if match else None


def _normalize_pmcid(value: object) -> str | None:
    match = re.search(r"PMC\d+", str(value or ""), flags=re.I)
    return match.group(0).upper() if match else None


def _source_article_identifiers(xml_path: Path) -> dict[str, str | None]:
    """Read identifiers used to prove an OpenAlex work matches the JATS source."""

    root = ET.fromstring(Path(xml_path).read_bytes())
    values: dict[str, str | None] = {"doi": None, "pmid": None}
    for element in root.iter():
        if _local_name(element.tag) != "article-id":
            continue
        kind = str(element.attrib.get("pub-id-type") or "").casefold()
        value = "".join(element.itertext()).strip()
        if kind == "doi" and values["doi"] is None:
            values["doi"] = _normalize_doi(value)
        elif kind == "pmid" and values["pmid"] is None:
            values["pmid"] = _normalize_pmid(value)
    return values


def _redact_secret(message: object, secret: str | None) -> str:
    text = str(message)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return re.sub(r"(?i)(api_key=)[^&\s]+", r"\1[REDACTED]", text)


def _metadata_for_source_version(
    session: requests.Session,
    pmcid: str,
    source_version: str | None,
    timeout: int,
) -> tuple[dict[str, Any], str, str]:
    """Resolve metadata for the exact XML article version when it is recorded."""

    if source_version is None:
        return _cloud_article_metadata(session, pmcid, timeout)
    metadata_url = (
        f"{PMC_CLOUD_HTTPS_BASE}/metadata/{quote(source_version, safe='')}.json"
    )
    response = session.get(metadata_url, timeout=timeout)
    if response.status_code == 404:
        raise FileNotFoundError(
            f"{source_version} is not present in the reusable PMC Cloud dataset"
        )
    response.raise_for_status()
    metadata = json.loads(response.content.decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("PMC Cloud metadata is not a JSON object")
    if str(metadata.get("pmcid") or "").upper() != pmcid:
        raise ValueError("PMC Cloud metadata PMCID does not match the source XML")
    expected_version = int(source_version.rsplit(".", 1)[1])
    reported_version = metadata.get("version")
    try:
        version_matches = int(reported_version) == expected_version
    except (TypeError, ValueError):
        version_matches = False
    if not version_matches:
        raise ValueError(
            "PMC Cloud metadata version does not match the source XML: "
            f"{reported_version!r} != {expected_version}"
        )
    return metadata, f"{source_version}/", metadata_url


def _openalex_work_metadata(
    session: requests.Session,
    *,
    doi: str,
    api_key: str,
    timeout: int,
) -> tuple[dict[str, Any], str, str]:
    """Resolve one work by DOI without putting the API key in a URL."""

    metadata_url = f"{OPENALEX_API_BASE}/works/doi:{quote(doi, safe='')}"
    response = session.get(
        metadata_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    if response.status_code == 404:
        raise FileNotFoundError(f"OpenAlex has no work for DOI {doi}")
    response.raise_for_status()
    raw = response.content
    metadata = json.loads(raw.decode("utf-8"))
    if not isinstance(metadata, dict):
        raise _OpenAlexIntegrityError("OpenAlex work metadata is not a JSON object")
    return metadata, metadata_url, hashlib.sha256(raw).hexdigest()


def _openalex_content_record(
    metadata: Mapping[str, Any],
    *,
    source_doi: str,
    source_pmcid: str,
    source_pmid: str | None,
) -> dict[str, Any]:
    """Validate work identity and return an official cached-PDF record.

    A DOI match is mandatory.  PMID and PMCID are checked whenever OpenAlex
    supplies them.  The function intentionally ignores ``locations[].pdf_url``:
    those are publisher/repository URLs, not OpenAlex archive objects.
    """

    metadata_doi = _normalize_doi(metadata.get("doi"))
    if metadata_doi != source_doi:
        raise _OpenAlexIntegrityError(
            "OpenAlex work DOI does not match source XML: "
            f"{metadata_doi!r} != {source_doi!r}"
        )
    work_type = metadata.get("type")
    if work_type != "article":
        raise _OpenAlexIntegrityError(
            f"OpenAlex work type is not an article: {work_type!r}"
        )

    ids = metadata.get("ids")
    if ids is not None and not isinstance(ids, Mapping):
        raise _OpenAlexIntegrityError("OpenAlex work ids field is malformed")
    ids = ids if isinstance(ids, Mapping) else {}
    ids_doi = _normalize_doi(ids.get("doi"))
    if ids.get("doi") is not None and ids_doi != source_doi:
        raise _OpenAlexIntegrityError("OpenAlex ids.doi does not match source XML")
    ids_pmcid = _normalize_pmcid(ids.get("pmcid"))
    if ids.get("pmcid") is not None and ids_pmcid != source_pmcid:
        raise _OpenAlexIntegrityError("OpenAlex ids.pmcid does not match source XML")
    ids_pmid = _normalize_pmid(ids.get("pmid"))
    if source_pmid and ids.get("pmid") is not None and ids_pmid != source_pmid:
        raise _OpenAlexIntegrityError("OpenAlex ids.pmid does not match source XML")

    open_access = metadata.get("open_access")
    if open_access is not None and not isinstance(open_access, Mapping):
        raise _OpenAlexIntegrityError("OpenAlex open_access field is malformed")
    open_access = open_access if isinstance(open_access, Mapping) else {}
    if open_access.get("is_oa") is not True:
        raise _OpenAlexUnavailable("manual_required_non_oa")

    has_content = metadata.get("has_content")
    if not isinstance(has_content, Mapping) or has_content.get("pdf") is not True:
        raise _OpenAlexUnavailable("cached_pdf_unavailable")
    content_urls = metadata.get("content_urls")
    if not isinstance(content_urls, Mapping):
        raise _OpenAlexIntegrityError(
            "OpenAlex reports has_content.pdf=true but supplies no content_urls"
        )

    work_id_value = str(metadata.get("id") or "")
    work_id_match = re.fullmatch(r"https://openalex\.org/(W\d+)", work_id_value)
    if not work_id_match:
        raise _OpenAlexIntegrityError(f"Invalid OpenAlex work id: {work_id_value!r}")
    work_id = work_id_match.group(1)
    content_url = content_urls.get("pdf")
    if not isinstance(content_url, str):
        raise _OpenAlexIntegrityError(
            "OpenAlex reports has_content.pdf=true but supplies no PDF URL"
        )
    expected_url = f"{OPENALEX_CONTENT_BASE}/works/{work_id}.pdf"
    if content_url != expected_url:
        raise _OpenAlexIntegrityError(
            "OpenAlex PDF URL is not the official content endpoint for this work"
        )

    best_oa_location = metadata.get("best_oa_location")
    if best_oa_location is not None and not isinstance(best_oa_location, Mapping):
        raise _OpenAlexIntegrityError("OpenAlex best_oa_location field is malformed")
    best_oa_location = (
        best_oa_location if isinstance(best_oa_location, Mapping) else {}
    )
    license_code = best_oa_location.get("license")
    if license_code is not None and not isinstance(license_code, str):
        raise _OpenAlexIntegrityError("OpenAlex best OA license field is malformed")
    return {
        "work_id": work_id,
        "doi": metadata_doi,
        "content_url": content_url,
        "license": license_code,
        "is_oa": open_access.get("is_oa"),
        "oa_status": open_access.get("oa_status"),
        "is_retracted": metadata.get("is_retracted"),
    }


def _download_openalex_pdf(
    session: requests.Session,
    content_url: str,
    *,
    api_key: str,
    timeout: int,
    max_bytes: int,
) -> bytes:
    """Download only from OpenAlex's authenticated content endpoint."""

    response = session.get(
        content_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()
    content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0]
    content_type = content_type.strip().casefold()
    if content_type and content_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise _OpenAlexIntegrityError(
            f"OpenAlex content endpoint returned {content_type!r}"
        )
    content_length_text = response.headers.get("Content-Length")
    content_length: int | None = None
    if content_length_text:
        try:
            content_length = int(content_length_text)
        except ValueError as error:
            raise _OpenAlexIntegrityError(
                "OpenAlex returned an invalid Content-Length"
            ) from error
        if content_length < 0 or content_length > max_bytes:
            raise _OpenAlexIntegrityError(
                "OpenAlex PDF exceeds byte limit: "
                f"{content_length} > {max_bytes}"
            )
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise _OpenAlexIntegrityError(
                f"Downloaded OpenAlex PDF exceeds byte limit: {total} > {max_bytes}"
            )
        chunks.append(chunk)
    if content_length is not None and total != content_length:
        raise _OpenAlexIntegrityError(
            "OpenAlex PDF Content-Length mismatch: "
            f"received {total}, expected {content_length}"
        )
    payload = b"".join(chunks)
    if not _pdf_payload_is_valid(payload):
        raise _OpenAlexIntegrityError(
            "OpenAlex content is not a structurally recognizable PDF"
        )
    try:
        import fitz

        document = fitz.open(stream=payload, filetype="pdf")
        try:
            if (
                not document.is_pdf
                or document.page_count < 1
                or document.is_repaired
                or document.needs_pass
            ):
                raise _OpenAlexIntegrityError(
                    "OpenAlex PDF is empty, repaired, encrypted, or not a valid PDF"
                )
            document.load_page(0)
            if document.page_count > 1:
                document.load_page(document.page_count - 1)
        finally:
            document.close()
    except _OpenAlexIntegrityError:
        raise
    except Exception as error:
        raise _OpenAlexIntegrityError(
            f"OpenAlex PDF failed PyMuPDF validation: {error}"
        ) from error
    return payload


def _pdf_payload_is_valid(payload: bytes) -> bool:
    return payload.startswith(b"%PDF-") and b"%%EOF" in payload[-8192:]


def _pdf_payload_opens_cleanly(payload: bytes) -> bool:
    """Return whether a cached payload is a readable, non-encrypted PDF.

    The lightweight header/trailer check catches obvious HTML/error payloads;
    opening the first and last pages catches truncated or otherwise unusable
    local files before a ready manifest is trusted.
    """

    if not _pdf_payload_is_valid(payload):
        return False
    try:
        import fitz

        document = fitz.open(stream=payload, filetype="pdf")
        try:
            if (
                not document.is_pdf
                or document.page_count < 1
                or document.is_repaired
                or document.needs_pass
            ):
                return False
            document.load_page(0)
            if document.page_count > 1:
                document.load_page(document.page_count - 1)
        finally:
            document.close()
    except Exception:
        return False
    return True


def _expected_manifest_pmcid(article_root: Path, pmcid: str | None) -> str | None:
    expected = str(pmcid or article_root.name).strip().upper()
    return expected if re.fullmatch(r"PMC\d+", expected) else None


def _valid_local_ready_pdf(
    manifest: Mapping[str, Any],
    article_root: Path,
    *,
    source_xml_sha256: str | None = None,
    pmcid: str | None = None,
    max_pdf_bytes: int | None = None,
) -> bool:
    """Verify a prior ready artifact without tying it to one XML serialization.

    ``source_xml_sha256`` remains accepted for call-site compatibility, but is
    deliberately not an identity gate: cleaned and raw JATS files for the same
    PMCID can have different bytes while referring to the same PDF artifact.
    """

    del source_xml_sha256

    if manifest.get("format") != PMC_PDF_MANIFEST_VERSION:
        return False
    if manifest.get("status") != "ready":
        return False
    expected_pmcid = _expected_manifest_pmcid(article_root, pmcid)
    if expected_pmcid is None or manifest.get("pmcid") != expected_pmcid:
        return False
    relative = manifest.get("pdf_path")
    expected_sha = manifest.get("pdf_sha256")
    if not isinstance(relative, str) or not isinstance(expected_sha, str):
        return False
    candidate = (article_root / relative).resolve()
    try:
        candidate.relative_to(article_root.resolve())
    except ValueError:
        return False
    if not candidate.is_file():
        return False
    if max_pdf_bytes is not None and candidate.stat().st_size > max_pdf_bytes:
        return False
    payload = candidate.read_bytes()
    return (
        _pdf_payload_opens_cleanly(payload)
        and hashlib.sha256(payload).hexdigest() == expected_sha
    )


def _valid_reusable_pdf(
    manifest: Mapping[str, Any],
    article_root: Path,
    *,
    source_xml_sha256: str | None = None,
    pmcid: str | None = None,
    max_pdf_bytes: int | None = None,
    cloud_object_key: str,
    cloud_expected_md5: str,
) -> bool:
    if manifest.get("cloud_object_key") != cloud_object_key:
        return False
    if str(manifest.get("cloud_expected_md5") or "") != cloud_expected_md5:
        return False
    return _valid_local_ready_pdf(
        manifest,
        article_root,
        source_xml_sha256=source_xml_sha256,
        pmcid=pmcid,
        max_pdf_bytes=max_pdf_bytes,
    )


def _source_xml_provenance_entry(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    source_path = manifest.get("source_xml_path")
    source_sha = manifest.get("source_xml_sha256")
    if not isinstance(source_path, str) or not isinstance(source_sha, str):
        return None
    return {
        "source_xml_path": source_path,
        "source_xml_sha256": source_sha,
        "source_pmcid_version": manifest.get("source_pmcid_version"),
    }


def _source_xml_was_observed(
    manifest: Mapping[str, Any],
    *,
    source_xml_path: str,
    source_xml_sha256: str,
) -> bool:
    """Return whether this exact XML path/hash pair is recorded in provenance."""

    expected = (source_xml_path, source_xml_sha256)
    candidates: list[Mapping[str, Any]] = [manifest]
    raw_history = manifest.get("source_xml_history")
    if isinstance(raw_history, list):
        candidates.extend(item for item in raw_history if isinstance(item, Mapping))
    return any(
        (candidate.get("source_xml_path"), candidate.get("source_xml_sha256"))
        == expected
        for candidate in candidates
    )


def _refresh_source_xml_provenance(
    reusable: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a reused manifest pointing at the current XML source.

    Every distinct XML path/hash previously observed is kept for audit, while
    the top-level source fields always describe the XML used in this run.
    """

    refreshed = dict(reusable)
    history: list[dict[str, Any]] = []
    raw_history = reusable.get("source_xml_history")
    if isinstance(raw_history, list):
        for item in raw_history:
            if isinstance(item, Mapping):
                entry = _source_xml_provenance_entry(item)
                if entry is not None and entry not in history:
                    history.append(entry)
    for source in (reusable, current):
        entry = _source_xml_provenance_entry(source)
        if entry is not None and entry not in history:
            history.append(entry)
    refreshed.update(
        {
            "source_xml_path": current.get("source_xml_path"),
            "source_xml_sha256": current.get("source_xml_sha256"),
            "source_pmcid_version": current.get("source_pmcid_version"),
            "source_xml_history": history,
        }
    )
    return refreshed


def _base_manifest(
    *,
    pmcid: str,
    xml_path: Path,
    source_xml_sha256: str,
    source_pmcid_version: str | None,
) -> dict[str, Any]:
    return {
        "format": PMC_PDF_MANIFEST_VERSION,
        "pmcid": pmcid,
        "source_xml_path": str(xml_path),
        "source_xml_sha256": source_xml_sha256,
        "source_pmcid_version": source_pmcid_version,
        "source_xml_history": [
            {
                "source_xml_path": str(xml_path),
                "source_xml_sha256": source_xml_sha256,
                "source_pmcid_version": source_pmcid_version,
            }
        ],
        "source_doi": None,
        "source_pmid": None,
        "status": "failed",
        "availability_reason": None,
        "pmc_availability_reason": None,
        "pmc_error": None,
        "cloud_article_prefix": None,
        "cloud_metadata_url": None,
        "cloud_object_key": None,
        "cloud_expected_md5": None,
        "download_method": None,
        "source_url": None,
        "pdf_path": None,
        "pdf_bytes": None,
        "pdf_sha256": None,
        "reused": False,
        "is_pmc_openaccess": None,
        "is_manuscript": None,
        "license_code": None,
        "openalex_fallback_status": None,
        "openalex_metadata_url": None,
        "openalex_metadata_sha256": None,
        "openalex_work_id": None,
        "openalex_content_url": None,
        "openalex_doi": None,
        "openalex_license": None,
        "openalex_is_oa": None,
        "openalex_oa_status": None,
        "openalex_is_retracted": None,
        "openalex_rights_notice": None,
        "openalex_identity_verification": None,
        "openalex_error": None,
        "error": None,
    }


def _valid_openalex_ready_pdf(
    manifest: Mapping[str, Any],
    article_root: Path,
    *,
    source_xml_sha256: str | None = None,
    source_doi: str | None,
    pmcid: str | None = None,
    source_pmcid_version: str | None = None,
    source_xml_path: str | None = None,
    cloud_article_prefix: str | None = None,
    max_pdf_bytes: int | None = None,
) -> bool:
    work_id = str(manifest.get("openalex_work_id") or "")
    expected_content_url = f"{OPENALEX_CONTENT_BASE}/works/{work_id}.pdf"
    recorded_version = manifest.get("source_pmcid_version")
    recorded_prefix = manifest.get("cloud_article_prefix")
    # Prove revision identity using the strongest current evidence available:
    # an explicit JATS version, the live PMC object prefix, or (during a
    # metadata outage) an XML path/hash pair already recorded for this PDF.
    if source_pmcid_version is not None:
        version_matches = recorded_version == source_pmcid_version
        if recorded_prefix is not None:
            version_matches = (
                version_matches
                and recorded_prefix == f"{source_pmcid_version}/"
            )
    elif cloud_article_prefix is not None:
        version_matches = recorded_prefix == cloud_article_prefix
    elif source_xml_path is not None and source_xml_sha256 is not None:
        version_matches = _source_xml_was_observed(
            manifest,
            source_xml_path=source_xml_path,
            source_xml_sha256=source_xml_sha256,
        )
    else:
        version_matches = False
    if (
        version_matches
        and cloud_article_prefix is not None
        and recorded_prefix != cloud_article_prefix
    ):
        version_matches = False
    return (
        source_doi is not None
        and version_matches
        and manifest.get("download_method") == "openalex_content_api"
        and manifest.get("source_doi") == source_doi
        and manifest.get("openalex_is_oa") is True
        and bool(re.fullmatch(r"W\d+", work_id))
        and manifest.get("openalex_content_url") == expected_content_url
        and manifest.get("source_url") == expected_content_url
        and _valid_local_ready_pdf(
            manifest,
            article_root,
            source_xml_sha256=source_xml_sha256,
            pmcid=pmcid,
            max_pdf_bytes=max_pdf_bytes,
        )
    )


def _valid_recorded_pmc_ready_pdf(
    manifest: Mapping[str, Any],
    article_root: Path,
    *,
    pmcid: str,
    source_pmcid_version: str | None,
    source_xml_path: str,
    source_xml_sha256: str,
    max_pdf_bytes: int | None = None,
) -> bool:
    """Validate a cached PMC PDF when fresh metadata is temporarily unavailable."""

    object_key = manifest.get("cloud_object_key")
    prefix = manifest.get("cloud_article_prefix")
    expected_md5 = str(manifest.get("cloud_expected_md5") or "")
    if manifest.get("download_method") != "pmc_cloud_aws":
        return False
    if not isinstance(object_key, str) or not isinstance(prefix, str):
        return False
    if not object_key.startswith(prefix) or not prefix.startswith(f"{pmcid}."):
        return False
    if source_pmcid_version is not None:
        if prefix != f"{source_pmcid_version}/":
            return False
    elif not _source_xml_was_observed(
        manifest,
        source_xml_path=source_xml_path,
        source_xml_sha256=source_xml_sha256,
    ):
        # With no JATS version marker and no live PMC metadata, a PMCID alone
        # cannot prove that a newly seen XML serialization belongs to the same
        # PMC article version as the cached PDF.
        return False
    if expected_md5 and not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
        return False
    expected_source_url = f"{PMC_CLOUD_HTTPS_BASE}/{quote(object_key, safe='/')}"
    if manifest.get("source_url") != expected_source_url:
        return False
    return _valid_local_ready_pdf(
        manifest,
        article_root,
        pmcid=pmcid,
        max_pdf_bytes=max_pdf_bytes,
    )


def _finish_pmc_unavailable_with_openalex(
    manifest: dict[str, Any],
    prior: Mapping[str, Any],
    article_root: Path,
    manifest_path: Path,
    *,
    session: requests.Session,
    api_key: str | None,
    source_xml_sha256: str,
    source_doi: str | None,
    source_pmid: str | None,
    pmcid: str,
    source_pmcid_version: str | None,
    timeout: int,
    max_pdf_bytes: int,
) -> dict[str, Any]:
    """Try OpenAlex only after PMC has authoritatively reported no PDF."""

    manifest["pmc_availability_reason"] = manifest.get("availability_reason")
    manifest["pmc_error"] = manifest.get("error")
    if _valid_openalex_ready_pdf(
        prior,
        article_root,
        source_xml_sha256=source_xml_sha256,
        source_doi=source_doi,
        pmcid=pmcid,
        source_pmcid_version=source_pmcid_version,
        source_xml_path=str(manifest.get("source_xml_path") or "") or None,
        cloud_article_prefix=(
            str(manifest.get("cloud_article_prefix") or "") or None
        ),
        max_pdf_bytes=max_pdf_bytes,
    ):
        reused = _refresh_source_xml_provenance(prior, manifest)
        reused.update(
            {
                "source_doi": source_doi,
                "source_pmid": source_pmid,
                "pmc_availability_reason": manifest["pmc_availability_reason"],
                "pmc_error": manifest["pmc_error"],
                "reused": True,
            }
        )
        reused.pop("refresh_warning", None)
        _write_json_atomic(manifest_path, reused)
        return reused
    if not api_key:
        manifest["openalex_fallback_status"] = "not_configured"
        _write_json_atomic(manifest_path, manifest)
        return manifest
    if not source_doi:
        manifest["openalex_fallback_status"] = "source_has_no_doi"
        _write_json_atomic(manifest_path, manifest)
        return manifest

    try:
        metadata, metadata_url, metadata_sha = _openalex_work_metadata(
            session,
            doi=source_doi,
            api_key=api_key,
            timeout=timeout,
        )
        manifest.update(
            {
                "openalex_metadata_url": metadata_url,
                "openalex_metadata_sha256": metadata_sha,
            }
        )
        record = _openalex_content_record(
            metadata,
            source_doi=source_doi,
            source_pmcid=pmcid,
            source_pmid=source_pmid,
        )
        manifest.update(
            {
                "openalex_work_id": record["work_id"],
                "openalex_content_url": record["content_url"],
                "openalex_doi": record["doi"],
                "openalex_license": record["license"],
                "openalex_is_oa": record["is_oa"],
                "openalex_oa_status": record["oa_status"],
                "openalex_is_retracted": record["is_retracted"],
                "openalex_rights_notice": OPENALEX_RIGHTS_NOTICE,
                "openalex_identity_verification": (
                    "source XML DOI equals OpenAlex DOI; supplied PMID/PMCID "
                    "identifiers agree; the cached content URL is bound to the "
                    "same OpenAlex work ID"
                ),
            }
        )
        payload = _download_openalex_pdf(
            session,
            record["content_url"],
            api_key=api_key,
            timeout=timeout,
            max_bytes=max_pdf_bytes,
        )
        pdf_path = article_root / f"{pmcid}.pdf"
        _write_bytes_atomic(pdf_path, payload)
        manifest.update(
            {
                "status": "ready",
                "availability_reason": None,
                "download_method": "openalex_content_api",
                "source_url": record["content_url"],
                "pdf_path": pdf_path.name,
                "pdf_bytes": len(payload),
                "pdf_sha256": hashlib.sha256(payload).hexdigest(),
                "openalex_fallback_status": "ready",
                "error": None,
            }
        )
    except _OpenAlexUnavailable as error:
        manifest.update(
            {
                "openalex_fallback_status": error.reason,
                "openalex_error": None,
                "error": manifest.get("pmc_error"),
            }
        )
    except FileNotFoundError as error:
        manifest.update(
            {
                "openalex_fallback_status": "work_not_found",
                "openalex_error": str(error),
                "error": manifest.get("pmc_error") or str(error),
            }
        )
    except _OpenAlexIntegrityError as error:
        openalex_error = _redact_secret(
            f"{type(error).__name__}: {error}", api_key
        )
        manifest.update(
            {
                "status": "unavailable",
                "openalex_fallback_status": "failed",
                "openalex_error": openalex_error,
                "error": manifest.get("pmc_error"),
            }
        )
    except Exception as error:
        # OpenAlex outages, throttling and exhausted budgets are fallback
        # failures, not evidence that the PMC article itself failed.
        openalex_error = _redact_secret(
            f"{type(error).__name__}: {error}", api_key
        )
        manifest.update(
            {
                "status": "unavailable",
                "openalex_fallback_status": "failed",
                "openalex_error": openalex_error,
                "error": manifest.get("pmc_error"),
            }
        )
    _write_json_atomic(manifest_path, manifest)
    return manifest


def download_pmc_pdf_for_article(
    xml_path: Path,
    output_root: Path,
    *,
    pmcid: str | None = None,
    session: requests.Session | None = None,
    timeout: int = 60,
    max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    openalex_api_key: str | None = None,
) -> dict[str, Any]:
    """Download one official reusable PDF, or record why none is distributed.

    OpenAlex is consulted only when PMC Cloud authoritatively has no reusable
    PDF and a non-empty API key was supplied.  The key is never persisted.
    """

    xml_path = Path(xml_path)
    resolved_pmcid, source_sha, source_version = _source_identity(xml_path, pmcid)
    source_ids = _source_article_identifiers(xml_path)
    source_doi = source_ids["doi"]
    source_pmid = source_ids["pmid"]
    openalex_api_key = str(openalex_api_key or "").strip() or None
    article_root = Path(output_root) / resolved_pmcid
    manifest_path = article_root / "manifest.json"
    prior = _read_json_object(manifest_path)
    manifest = _base_manifest(
        pmcid=resolved_pmcid,
        xml_path=xml_path,
        source_xml_sha256=source_sha,
        source_pmcid_version=source_version,
    )
    manifest.update({"source_doi": source_doi, "source_pmid": source_pmid})
    http = session or _retrying_session(
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    try:
        metadata, prefix, metadata_url = _metadata_for_source_version(
            http, resolved_pmcid, source_version, timeout
        )
        manifest.update(
            {
                "cloud_article_prefix": prefix,
                "cloud_metadata_url": metadata_url,
                "is_pmc_openaccess": metadata.get("is_pmc_openaccess"),
                "is_manuscript": metadata.get("is_manuscript"),
                "license_code": metadata.get("license_code"),
            }
        )
    except FileNotFoundError as error:
        manifest.update(
            {
                "status": "unavailable",
                "availability_reason": "not_in_reusable_pmc_cloud_dataset",
                "error": str(error),
            }
        )
        return _finish_pmc_unavailable_with_openalex(
            manifest,
            prior,
            article_root,
            manifest_path,
            session=http,
            api_key=openalex_api_key,
            source_xml_sha256=source_sha,
            source_doi=source_doi,
            source_pmid=source_pmid,
            pmcid=resolved_pmcid,
            source_pmcid_version=source_version,
            timeout=timeout,
            max_pdf_bytes=max_pdf_bytes,
        )
    except Exception as error:
        # A transient metadata outage must not demote an already verified PDF.
        reusable = _valid_recorded_pmc_ready_pdf(
            prior,
            article_root,
            pmcid=resolved_pmcid,
            source_pmcid_version=source_version,
            source_xml_path=str(xml_path),
            source_xml_sha256=source_sha,
            max_pdf_bytes=max_pdf_bytes,
        ) or _valid_openalex_ready_pdf(
            prior,
            article_root,
            source_xml_sha256=source_sha,
            source_doi=source_doi,
            pmcid=resolved_pmcid,
            source_pmcid_version=source_version,
            source_xml_path=str(xml_path),
            max_pdf_bytes=max_pdf_bytes,
        )
        if reusable:
            refreshed = _refresh_source_xml_provenance(prior, manifest)
            refreshed.update({"source_doi": source_doi, "source_pmid": source_pmid})
            refreshed["reused"] = True
            refreshed["refresh_warning"] = f"{type(error).__name__}: {error}"
            _write_json_atomic(manifest_path, refreshed)
            return refreshed
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json_atomic(manifest_path, manifest)
        return manifest

    pdf_url = metadata.get("pdf_url")
    if not isinstance(pdf_url, str) or not pdf_url.strip():
        manifest.update(
            {
                "status": "unavailable",
                "availability_reason": "pmc_cloud_metadata_has_no_pdf",
                "error": (
                    "PMC Cloud does not distribute a PDF for this article version; "
                    f"open_access={metadata.get('is_pmc_openaccess')!r}, "
                    f"manuscript={metadata.get('is_manuscript')!r}, "
                    f"license={metadata.get('license_code')!r}"
                ),
            }
        )
        return _finish_pmc_unavailable_with_openalex(
            manifest,
            prior,
            article_root,
            manifest_path,
            session=http,
            api_key=openalex_api_key,
            source_xml_sha256=source_sha,
            source_doi=source_doi,
            source_pmid=source_pmid,
            pmcid=resolved_pmcid,
            source_pmcid_version=source_version,
            timeout=timeout,
            max_pdf_bytes=max_pdf_bytes,
        )

    try:
        cloud = _cloud_object_record(pdf_url)
        if Path(cloud["basename"]).suffix.casefold() != ".pdf":
            raise ValueError("PMC Cloud pdf_url does not identify a PDF object")
        if not cloud["object_key"].startswith(prefix):
            raise ValueError(
                "PMC Cloud PDF object does not belong to the source XML version: "
                f"{cloud['object_key']!r} is outside {prefix!r}"
            )
        manifest.update(
            {
                "cloud_object_key": cloud["object_key"],
                "cloud_expected_md5": cloud["expected_md5"],
                "source_url": cloud["source_url"],
            }
        )
        if _valid_reusable_pdf(
            prior,
            article_root,
            source_xml_sha256=source_sha,
            pmcid=resolved_pmcid,
            max_pdf_bytes=max_pdf_bytes,
            cloud_object_key=cloud["object_key"],
            cloud_expected_md5=cloud["expected_md5"],
        ):
            refreshed = _refresh_source_xml_provenance(prior, manifest)
            refreshed.update(
                {
                    "cloud_article_prefix": prefix,
                    "cloud_metadata_url": metadata_url,
                    "is_pmc_openaccess": metadata.get("is_pmc_openaccess"),
                    "is_manuscript": metadata.get("is_manuscript"),
                    "license_code": metadata.get("license_code"),
                    "download_method": "pmc_cloud_aws",
                    "source_doi": source_doi,
                    "source_pmid": source_pmid,
                }
            )
            refreshed["reused"] = True
            refreshed.pop("refresh_warning", None)
            _write_json_atomic(manifest_path, refreshed)
            return refreshed

        payload = _download_limited(
            http,
            cloud["source_url"],
            timeout=timeout,
            max_bytes=max_pdf_bytes,
        )
        actual_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        if cloud["expected_md5"] and actual_md5 != cloud["expected_md5"]:
            raise ValueError(
                "PMC Cloud PDF MD5 mismatch: "
                f"{actual_md5} != {cloud['expected_md5']}"
            )
        if not _pdf_payload_opens_cleanly(payload):
            raise ValueError(
                "Downloaded object is not a readable, non-encrypted PDF"
            )
        pdf_path = article_root / f"{resolved_pmcid}.pdf"
        _write_bytes_atomic(pdf_path, payload)
        manifest.update(
            {
                "status": "ready",
                "availability_reason": None,
                "download_method": "pmc_cloud_aws",
                "pdf_path": pdf_path.name,
                "pdf_bytes": len(payload),
                "pdf_sha256": hashlib.sha256(payload).hexdigest(),
                "error": None,
            }
        )
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
    _write_json_atomic(manifest_path, manifest)
    return manifest


def download_pmc_pdfs_for_directory(
    xml_dir: Path,
    output_root: Path,
    *,
    pmcids: Iterable[str] | None = None,
    timeout: int = 60,
    max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    workers: int = 1,
    limit: int | None = None,
    openalex_api_key: str | None = None,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    allowed = (
        {str(item).strip().upper() for item in pmcids} if pmcids is not None else None
    )
    xml_paths = sorted(
        (
            path
            for path in Path(xml_dir).iterdir()
            if path.is_file() and re.fullmatch(r"PMC\d+\.xml", path.name, flags=re.I)
        ),
        key=lambda item: item.name.casefold(),
    )
    if allowed is not None:
        xml_paths = [path for path in xml_paths if path.stem.upper() in allowed]
    if limit is not None:
        xml_paths = xml_paths[:limit]
    outcomes: list[dict[str, Any]] = []
    session_local = threading.local()
    created_sessions: list[requests.Session] = []
    created_sessions_lock = threading.Lock()

    def worker_session() -> requests.Session:
        existing = getattr(session_local, "http", None)
        if isinstance(existing, requests.Session):
            return existing
        created = _retrying_session(
            retry_attempts=retry_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        session_local.http = created
        with created_sessions_lock:
            created_sessions.append(created)
        return created

    def one(xml_path: Path) -> dict[str, Any]:
        try:
            manifest = download_pmc_pdf_for_article(
                xml_path,
                output_root,
                session=worker_session(),
                timeout=timeout,
                max_pdf_bytes=max_pdf_bytes,
                retry_attempts=retry_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                openalex_api_key=openalex_api_key,
            )
            return {
                "pmcid": manifest["pmcid"],
                "status": manifest["status"],
                "download_method": manifest.get("download_method"),
                "openalex_fallback_status": manifest.get(
                    "openalex_fallback_status"
                ),
                "pdf_bytes": manifest.get("pdf_bytes"),
                "reused": bool(manifest.get("reused")),
                "reason": manifest.get("availability_reason") or manifest.get("error"),
            }
        except Exception as error:
            # One malformed XML/network result must not abort the remaining corpus.
            return {
                "pmcid": xml_path.stem.upper(),
                "status": "failed",
                "download_method": None,
                "openalex_fallback_status": None,
                "pdf_bytes": None,
                "reused": False,
                "reason": f"{type(error).__name__}: {error}",
            }

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ordered = sorted(rows, key=lambda item: str(item["pmcid"]))
        return {
            "format": PMC_PDF_MANIFEST_VERSION,
            "article_count": len(ordered),
            "ready_count": sum(item["status"] == "ready" for item in ordered),
            "unavailable_count": sum(
                item["status"] == "unavailable" for item in ordered
            ),
            "failed_count": sum(item["status"] == "failed" for item in ordered),
            "pmc_cloud_ready_count": sum(
                item.get("download_method") == "pmc_cloud_aws" for item in ordered
            ),
            "openalex_ready_count": sum(
                item.get("download_method") == "openalex_content_api"
                for item in ordered
            ),
            "fallback_failed_count": sum(
                item.get("openalex_fallback_status") == "failed"
                for item in ordered
            ),
            "reused_count": sum(bool(item["reused"]) for item in ordered),
            "available_pdf_bytes": sum(
                int(item.get("pdf_bytes") or 0) for item in ordered
            ),
            # Backward-compatible alias. It includes files reused this run.
            "downloaded_pdf_bytes": sum(
                int(item.get("pdf_bytes") or 0) for item in ordered
            ),
            "outcomes": ordered,
        }

    total = len(xml_paths)
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    if workers == 1:
        iterator = map(one, xml_paths)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="pmc-pdf",
        )
        iterator = executor.map(one, xml_paths)
    try:
        for completed, outcome in enumerate(iterator, start=1):
            outcomes.append(outcome)
            checkpoint = summarize(outcomes)
            _write_json_atomic(Path(output_root) / "summary.json", checkpoint)
            if progress is not None:
                progress(completed, total, outcome)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for created_session in created_sessions:
            created_session.close()
    summary = summarize(outcomes)
    _write_json_atomic(Path(output_root) / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download all officially distributed PMC PDFs for local XML files. "
            "The official OpenAlex cached-content fallback is used only with "
            f"--openalex-fallback and ${OPENALEX_API_KEY_ENV}."
        )
    )
    parser.add_argument("--xml-dir", type=Path, default=Path("output/pmc_xml_no_refs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/pmc_pdf_assets")
    )
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--max-pdf-bytes", type=int, default=DEFAULT_MAX_PDF_BYTES)
    parser.add_argument(
        "--pmcid",
        action="append",
        default=None,
        help="Download only this PMCID; repeat for multiple articles.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--openalex-fallback",
        action="store_true",
        help=(
            "Allow the OpenAlex cached-content fallback for PMC-unavailable "
            f"records. Requires ${OPENALEX_API_KEY_ENV}; this explicit switch "
            "prevents accidental paid API calls."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.xml_dir.is_dir():
        raise SystemExit(f"XML directory not found: {args.xml_dir}")
    if args.max_pdf_bytes <= 0:
        raise SystemExit("--max-pdf-bytes must be positive")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retry_attempts < 0 or args.retry_backoff_seconds < 0:
        raise SystemExit("retry settings must be non-negative")
    if args.openalex_fallback and not str(
        os.environ.get(OPENALEX_API_KEY_ENV) or ""
    ).strip():
        raise SystemExit(
            f"--openalex-fallback requires ${OPENALEX_API_KEY_ENV}"
        )

    progress_lock = threading.Lock()

    def report(completed: int, total: int, outcome: Mapping[str, Any]) -> None:
        if args.quiet:
            return
        with progress_lock:
            print(
                f"[{completed:>4}/{total}] {outcome['pmcid']} " f"{outcome['status']}",
                file=sys.stderr,
                flush=True,
            )

    summary = download_pmc_pdfs_for_directory(
        args.xml_dir,
        args.output_dir,
        pmcids=args.pmcid,
        timeout=args.request_timeout,
        max_pdf_bytes=args.max_pdf_bytes,
        retry_attempts=args.retry_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        workers=args.workers,
        limit=args.limit,
        openalex_api_key=(
            (str(os.environ.get(OPENALEX_API_KEY_ENV) or "").strip() or None)
            if args.openalex_fallback
            else None
        ),
        progress=report,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "outcomes"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
