"""Build an auditable PDF-discovery report without scraping article pages.

The report is intended for PMC records whose reusable PMC Cloud manifest is
``unavailable``.  It combines identifiers from the local JATS XML with
metadata returned by the official Europe PMC and OpenAlex APIs.  It does not
request a publisher page, PMC article page, or candidate PDF URL.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DISCOVERY_REPORT_VERSION = "phase1_pdf_discovery_report_v1"
CACHE_VERSION = "phase1_pdf_discovery_cache_v1"
EUROPE_PMC_SEARCH_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
)
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_USER_AGENT = (
    "phase1-pdf-discovery/1.0 "
    "(metadata-only; Europe PMC and OpenAlex official APIs)"
)
DEFAULT_CACHE_MAX_AGE_DAYS = 30.0
DEFAULT_REQUEST_INTERVAL_SECONDS = 0.12


CSV_FIELDS = (
    "pmcid",
    "pmid",
    "doi",
    "title",
    "journal",
    "current_status",
    "current_reason",
    "pmc_open_access",
    "pmc_license",
    "xml_license",
    "europe_pmc_open_access",
    "europe_pmc_license",
    "openalex_open_access",
    "openalex_oa_status",
    "openalex_license",
    "openalex_cached_pdf",
    "candidate_source",
    "access_class",
    "recommended_action",
    "pmc_article_url",
    "pubmed_url",
    "doi_url",
    "europe_pmc_url",
    "europe_pmc_pdf_url",
    "openalex_url",
    "openalex_oa_pdf_url",
    "openalex_cached_pdf_url",
    "discovery_errors",
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalized_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _first_descendant(root: ET.Element, local_name: str) -> ET.Element | None:
    return next(
        (element for element in root.iter() if _local_name(element.tag) == local_name),
        None,
    )


def _xlink_href(element: ET.Element | None) -> str:
    if element is None:
        return ""
    for key, value in element.attrib.items():
        if key == "href" or key.endswith("}href"):
            return str(value).strip()
    return ""


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"y", "yes", "true", "1"}:
        return True
    if normalized in {"n", "no", "false", "0"}:
        return False
    return None


def _bool_text(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _safe_http_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    return text if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _doi_url(doi: str) -> str:
    return f"https://doi.org/{quote(doi, safe='/():;._-')}" if doi else ""


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MetadataCache:
    """Small persistent cache with atomic writes and a configurable TTL."""

    def __init__(
        self,
        path: Path,
        *,
        max_age_days: float = DEFAULT_CACHE_MAX_AGE_DAYS,
        refresh: bool = False,
    ) -> None:
        if max_age_days < 0:
            raise ValueError("cache max age must be non-negative")
        self.path = Path(path)
        self.max_age = timedelta(days=max_age_days)
        self.refresh = refresh
        loaded = _read_json_object(self.path)
        self.payload: dict[str, Any] = (
            loaded
            if loaded.get("format") == CACHE_VERSION
            else {"format": CACHE_VERSION, "entries": {}}
        )
        if not isinstance(self.payload.get("entries"), dict):
            self.payload["entries"] = {}

    def get(
        self,
        provider: str,
        key: str,
        *,
        allow_expired: bool = False,
    ) -> tuple[bool, Any]:
        if self.refresh:
            return False, None
        cache_key = f"{provider}:{key.casefold()}"
        entry = self.payload["entries"].get(cache_key)
        if not isinstance(entry, dict) or "value" not in entry:
            return False, None
        try:
            fetched_at = datetime.fromisoformat(
                str(entry["fetched_at"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            return False, None
        if not allow_expired and _utc_now() - fetched_at > self.max_age:
            return False, None
        return True, entry["value"]

    def put(self, provider: str, key: str, value: Any) -> None:
        cache_key = f"{provider}:{key.casefold()}"
        self.payload["entries"][cache_key] = {
            "fetched_at": _utc_now().isoformat().replace("+00:00", "Z"),
            "value": value,
        }
        _atomic_write_json(self.path, self.payload)


class RequestRateLimiter:
    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds < 0:
            raise ValueError("request interval must be non-negative")
        self.interval_seconds = interval_seconds
        self._last_request_at: dict[str, float] = {}

    def wait(self, provider: str) -> None:
        if self.interval_seconds == 0:
            return
        now = time.monotonic()
        remaining = (
            self.interval_seconds - (now - self._last_request_at.get(provider, 0.0))
        )
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at[provider] = time.monotonic()


def _retrying_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(408, 429, 500, 502, 503, 504),
        backoff_factor=0.75,
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    session.mount("https://", adapter)
    return session


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: int,
) -> Any:
    response = session.get(
        url,
        params=dict(params),
        headers=dict(headers or {}),
        timeout=timeout,
    )
    if response.status_code >= 400:
        # Do not call raise_for_status(): its message can contain the prepared
        # URL, including an OpenAlex API key supplied as a query parameter.
        raise RuntimeError(f"metadata API returned HTTP {response.status_code}")
    payload = json.loads(response.content.decode("utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError("metadata API did not return a JSON object or list")
    return payload


def _parse_xml_metadata(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    discovered_article_meta = _first_descendant(root, "article-meta")
    article_meta = discovered_article_meta if discovered_article_meta is not None else root
    identifiers: dict[str, str] = {}
    for element in article_meta.iter():
        if _local_name(element.tag) != "article-id":
            continue
        id_type = str(element.attrib.get("pub-id-type") or "").casefold()
        value = _normalized_text(element)
        if id_type and value and id_type not in identifiers:
            identifiers[id_type] = value

    title = _normalized_text(_first_descendant(article_meta, "article-title"))
    journal = _normalized_text(_first_descendant(root, "journal-title"))
    license_element = _first_descendant(article_meta, "license")
    license_type = (
        str(license_element.attrib.get("license-type") or "").strip()
        if license_element is not None
        else ""
    )
    license_url = _safe_http_url(_xlink_href(license_element))
    license_text = _normalized_text(license_element)
    if len(license_text) > 300:
        license_text = license_text[:297].rstrip() + "..."
    xml_license = " | ".join(
        part for part in (license_type, license_url, license_text) if part
    )
    return {
        "pmcid": identifiers.get("pmcid") or identifiers.get("pmc") or xml_path.stem,
        "pmid": identifiers.get("pmid", ""),
        "doi": identifiers.get("doi", "").removeprefix("https://doi.org/"),
        "title": title,
        "journal": journal,
        "xml_license": xml_license,
    }


def _select_matching_result(
    results: Iterable[Any],
    *,
    pmcid: str,
    pmid: str,
    doi: str,
) -> dict[str, Any] | None:
    normalized_doi = doi.casefold()
    for candidate in results:
        if not isinstance(candidate, dict):
            continue
        candidate_pmcid = str(candidate.get("pmcid") or "").upper()
        candidate_pmid = str(candidate.get("pmid") or "")
        candidate_doi = str(candidate.get("doi") or "").casefold()
        if (
            (pmcid and candidate_pmcid == pmcid.upper())
            or (pmid and candidate_pmid == pmid)
            or (normalized_doi and candidate_doi == normalized_doi)
        ):
            return candidate
    return None


def discover_europe_pmc(
    *,
    session: requests.Session,
    cache: MetadataCache,
    limiter: RequestRateLimiter,
    pmcid: str,
    pmid: str,
    doi: str,
    timeout: int,
    offline: bool,
) -> dict[str, Any]:
    cache_key = pmid or pmcid or doi
    hit, cached = cache.get("europe_pmc", cache_key, allow_expired=offline)
    if hit:
        return cached if isinstance(cached, dict) else {}
    if offline:
        return {}
    query = f"EXT_ID:{pmid} AND SRC:MED" if pmid else f"PMC_ID:{pmcid}"
    limiter.wait("europe_pmc")
    payload = _request_json(
        session,
        EUROPE_PMC_SEARCH_URL,
        params={
            "query": query,
            "resultType": "core",
            "format": "json",
            "pageSize": 5,
        },
        timeout=timeout,
    )
    result_list = payload.get("resultList", {}) if isinstance(payload, dict) else {}
    results = result_list.get("result", []) if isinstance(result_list, dict) else []
    match = _select_matching_result(
        results if isinstance(results, list) else [],
        pmcid=pmcid,
        pmid=pmid,
        doi=doi,
    )
    normalized = _normalize_europe_pmc(match)
    cache.put("europe_pmc", cache_key, normalized)
    return normalized


def _normalize_europe_pmc(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    url_list = result.get("fullTextUrlList")
    raw_urls = url_list.get("fullTextUrl", []) if isinstance(url_list, dict) else []
    pdf_candidates: list[dict[str, str]] = []
    if isinstance(raw_urls, list):
        for record in raw_urls:
            if not isinstance(record, dict):
                continue
            url = _safe_http_url(record.get("url"))
            style = str(record.get("documentStyle") or "").casefold()
            if not url or (style != "pdf" and "pdf" not in url.casefold()):
                continue
            pdf_candidates.append(
                {
                    "url": url,
                    "availability": str(record.get("availability") or ""),
                    "site": str(record.get("site") or ""),
                }
            )
    free_pdf = next(
        (
            item
            for item in pdf_candidates
            if item["availability"].casefold() == "free"
        ),
        pdf_candidates[0] if pdf_candidates else None,
    )
    return {
        "found": True,
        "pmcid": str(result.get("pmcid") or ""),
        "pmid": str(result.get("pmid") or ""),
        "doi": str(result.get("doi") or ""),
        "is_open_access": _parse_bool(result.get("isOpenAccess")),
        "license": str(result.get("license") or ""),
        "has_pdf": _parse_bool(result.get("hasPDF")),
        "pdf_url": free_pdf["url"] if free_pdf else "",
        "pdf_availability": free_pdf["availability"] if free_pdf else "",
        "pdf_site": free_pdf["site"] if free_pdf else "",
    }


def discover_openalex(
    *,
    session: requests.Session,
    cache: MetadataCache,
    limiter: RequestRateLimiter,
    doi: str,
    timeout: int,
    offline: bool,
    api_key: str | None,
) -> dict[str, Any]:
    if not doi:
        return {}
    hit, cached = cache.get("openalex", doi, allow_expired=offline)
    if hit:
        return cached if isinstance(cached, dict) else {}
    if offline:
        return {}
    limiter.wait("openalex")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    identifier = f"doi:{quote(doi, safe='')}"
    payload = _request_json(
        session,
        f"{OPENALEX_WORKS_URL}/{identifier}",
        params={},
        headers=headers,
        timeout=timeout,
    )
    normalized = _normalize_openalex(payload if isinstance(payload, dict) else None)
    cache.put("openalex", doi, normalized)
    return normalized


def _normalize_openalex(work: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(work, Mapping):
        return {}
    openalex_id = str(work.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
    open_access = work.get("open_access")
    oa = open_access if isinstance(open_access, dict) else {}
    best_location = work.get("best_oa_location")
    best = best_location if isinstance(best_location, dict) else {}
    has_content = work.get("has_content")
    content = has_content if isinstance(has_content, dict) else {}
    cached_pdf = _parse_bool(content.get("pdf")) is True
    return {
        "found": True,
        "id": openalex_id,
        "is_oa": _parse_bool(oa.get("is_oa")),
        "oa_status": str(oa.get("oa_status") or ""),
        "license": str(best.get("license") or ""),
        "best_pdf_url": _safe_http_url(best.get("pdf_url")),
        "best_landing_url": _safe_http_url(best.get("landing_page_url")),
        "cached_pdf": cached_pdf,
        "cached_pdf_url": (
            f"https://content.openalex.org/works/{openalex_id}.pdf"
            if cached_pdf and re.fullmatch(r"W\d+", openalex_id)
            else ""
        ),
    }


def _candidate_decision(
    *,
    current_status: str,
    europe: Mapping[str, Any],
    openalex: Mapping[str, Any],
) -> tuple[str, str, str]:
    if current_status == "ready":
        return (
            "pmc_cloud_ready",
            "already_ready",
            "No action: the verified PMC Cloud PDF already exists.",
        )
    epmc_pdf = _safe_http_url(europe.get("pdf_url"))
    epmc_oa = europe.get("is_open_access") is True
    openalex_pdf = _safe_http_url(openalex.get("best_pdf_url"))
    openalex_oa = openalex.get("is_oa") is True
    cached_pdf = openalex.get("cached_pdf") is True
    if cached_pdf and openalex_oa:
        return (
            "openalex_cached_pdf",
            "oa",
            "Automatic OpenAlex Content API candidate. An API key is required; "
            "retain the source license and do not assume redistribution rights.",
        )
    if epmc_pdf and epmc_oa:
        return (
            "europe_pmc_oa_pdf",
            "oa",
            "Open-access candidate. Retrieve only through an official OA endpoint "
            "and preserve the recorded license.",
        )
    if openalex_pdf and openalex_oa:
        return (
            "openalex_oa_location",
            "oa",
            "Open-access location candidate. Verify the license and article identity "
            "before saving the PDF.",
        )
    if cached_pdf:
        return (
            "openalex_cached_pdf",
            "manual_required",
            "OpenAlex has a cached PDF, but this record is not marked open access. "
            "Review the source terms and copyright before requesting it through the API.",
        )
    if epmc_pdf:
        return (
            "europe_pmc_free_link_manual",
            "manual_required",
            "Free-to-read link, but not marked open access. Open manually or use a "
            "publisher/institutional TDM entitlement; do not bulk-download via Europe PMC.",
        )
    if openalex_pdf:
        return (
            "openalex_location_manual",
            "subscription_required",
            "Candidate publisher/repository link. Access manually and follow the "
            "publisher license or institutional entitlement.",
        )
    return (
        "no_public_pdf_route_found",
        "no_route",
        "No public PDF route was found. Try institutional access, interlibrary loan, "
        "a publisher TDM agreement, or an author manuscript request.",
    )


def _base_links(pmcid: str, pmid: str, doi: str) -> dict[str, str]:
    numeric_pmcid = pmcid.removeprefix("PMC")
    return {
        "pmc_article_url": (
            f"https://pmc.ncbi.nlm.nih.gov/articles/{quote(pmcid, safe='')}/"
            if pmcid
            else ""
        ),
        "pubmed_url": (
            f"https://pubmed.ncbi.nlm.nih.gov/{quote(pmid, safe='')}/" if pmid else ""
        ),
        "doi_url": _doi_url(doi),
        "europe_pmc_url": (
            f"https://europepmc.org/article/PMC/{quote(numeric_pmcid, safe='')}"
            if numeric_pmcid
            else ""
        ),
    }


def build_discovery_row(
    *,
    xml_metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    europe: Mapping[str, Any],
    openalex: Mapping[str, Any],
    errors: Iterable[str] = (),
) -> dict[str, str]:
    pmcid = str(xml_metadata.get("pmcid") or manifest.get("pmcid") or "").upper()
    pmid = str(xml_metadata.get("pmid") or "")
    doi = str(xml_metadata.get("doi") or "")
    current_status = str(manifest.get("status") or "missing_manifest")
    source, access_class, action = _candidate_decision(
        current_status=current_status,
        europe=europe,
        openalex=openalex,
    )
    openalex_id = str(openalex.get("id") or "")
    row: dict[str, str] = {
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "title": str(xml_metadata.get("title") or ""),
        "journal": str(xml_metadata.get("journal") or ""),
        "current_status": current_status,
        "current_reason": str(
            manifest.get("availability_reason") or manifest.get("error") or ""
        ),
        "pmc_open_access": _bool_text(_parse_bool(manifest.get("is_pmc_openaccess"))),
        "pmc_license": str(manifest.get("license_code") or ""),
        "xml_license": str(xml_metadata.get("xml_license") or ""),
        "europe_pmc_open_access": _bool_text(
            europe.get("is_open_access")
            if isinstance(europe.get("is_open_access"), bool)
            else None
        ),
        "europe_pmc_license": str(europe.get("license") or ""),
        "openalex_open_access": _bool_text(
            openalex.get("is_oa")
            if isinstance(openalex.get("is_oa"), bool)
            else None
        ),
        "openalex_oa_status": str(openalex.get("oa_status") or ""),
        "openalex_license": str(openalex.get("license") or ""),
        "openalex_cached_pdf": _bool_text(
            openalex.get("cached_pdf")
            if isinstance(openalex.get("cached_pdf"), bool)
            else None
        ),
        "candidate_source": source,
        "access_class": access_class,
        "recommended_action": action,
        **_base_links(pmcid, pmid, doi),
        "europe_pmc_pdf_url": _safe_http_url(europe.get("pdf_url")),
        "openalex_url": (
            f"https://openalex.org/{quote(openalex_id, safe='')}" if openalex_id else ""
        ),
        "openalex_oa_pdf_url": _safe_http_url(openalex.get("best_pdf_url")),
        "openalex_cached_pdf_url": _safe_http_url(openalex.get("cached_pdf_url")),
        "discovery_errors": " | ".join(str(error) for error in errors if error),
    }
    return {field: row.get(field, "") for field in CSV_FIELDS}


def _write_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(
        {
            field: _csv_safe_cell(str(row.get(field, "")))
            for field in CSV_FIELDS
        }
        for row in rows
    )
    _atomic_write_text(path, buffer.getvalue())


def _csv_safe_cell(value: str) -> str:
    """Neutralize spreadsheet formulas without altering the HTML report."""

    stripped = value.lstrip()
    if stripped.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def _link_button(url: str, label: str) -> str:
    safe_url = _safe_http_url(url)
    if not safe_url:
        return ""
    return (
        f'<a class="source-link" href="{html.escape(safe_url, quote=True)}" '
        f'target="_blank" rel="noopener noreferrer">{html.escape(label)}</a>'
    )


def _render_html(rows: list[Mapping[str, str]], generated_at: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("candidate_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    automatic = counts.get("openalex_cached_pdf", 0) + counts.get(
        "europe_pmc_oa_pdf", 0
    ) + counts.get("openalex_oa_location", 0)
    manual = counts.get("europe_pmc_free_link_manual", 0) + counts.get(
        "openalex_location_manual", 0
    )
    unresolved = counts.get("no_public_pdf_route_found", 0)

    cards = "".join(
        f'<div class="card"><strong>{value}</strong><span>{html.escape(label)}</span></div>'
        for value, label in (
            (len(rows), "records reviewed"),
            (automatic, "official API / OA candidates"),
            (manual, "manual or entitled access"),
            (unresolved, "no public route found"),
        )
    )
    table_rows: list[str] = []
    for row in rows:
        source = str(row.get("candidate_source") or "")
        access_class = str(row.get("access_class") or "")
        links = "".join(
            item
            for item in (
                _link_button(str(row.get("pmc_article_url") or ""), "PMC"),
                _link_button(str(row.get("pubmed_url") or ""), "PubMed"),
                _link_button(str(row.get("doi_url") or ""), "DOI"),
                _link_button(str(row.get("europe_pmc_url") or ""), "Europe PMC"),
                _link_button(str(row.get("europe_pmc_pdf_url") or ""), "EPMC PDF candidate"),
                _link_button(str(row.get("openalex_url") or ""), "OpenAlex"),
                _link_button(str(row.get("openalex_oa_pdf_url") or ""), "OA PDF candidate"),
                _link_button(
                    str(row.get("openalex_cached_pdf_url") or ""),
                    "OpenAlex cached PDF",
                ),
            )
        ) or '<span class="muted">No link found</span>'
        license_bits = [
            f"PMC: {row.get('pmc_license')}" if row.get("pmc_license") else "",
            f"Europe PMC: {row.get('europe_pmc_license')}"
            if row.get("europe_pmc_license")
            else "",
            f"OpenAlex: {row.get('openalex_license')}"
            if row.get("openalex_license")
            else "",
        ]
        license_text = " · ".join(bit for bit in license_bits if bit) or "Not reported"
        search_text = " ".join(str(value) for value in row.values()).casefold()
        table_rows.append(
            "<tr "
            f'data-source="{html.escape(source, quote=True)}" '
            f'data-search="{html.escape(search_text, quote=True)}">'
            f"<td><strong>{html.escape(str(row.get('pmcid') or ''))}</strong>"
            f"<div class=\"muted\">PMID {html.escape(str(row.get('pmid') or '—'))}</div>"
            f"<div class=\"muted\">{html.escape(str(row.get('doi') or ''))}</div></td>"
            f"<td><strong>{html.escape(str(row.get('title') or 'Untitled'))}</strong>"
            f"<div class=\"muted\">{html.escape(str(row.get('journal') or ''))}</div></td>"
            f"<td><span class=\"badge\">{html.escape(access_class.replace('_', ' '))}</span>"
            f"<div class=\"muted\">{html.escape(source.replace('_', ' '))}</div>"
            f"<div class=\"action\">{html.escape(str(row.get('recommended_action') or ''))}</div>"
            f"<div class=\"muted\">Current: {html.escape(str(row.get('current_status') or ''))}</div></td>"
            f"<td>{html.escape(license_text)}"
            f"<div class=\"muted\">OA — PMC {html.escape(str(row.get('pmc_open_access') or 'unknown'))}; "
            f"EPMC {html.escape(str(row.get('europe_pmc_open_access') or 'unknown'))}; "
            f"OpenAlex {html.escape(str(row.get('openalex_open_access') or 'unknown'))}</div></td>"
            f"<td><div class=\"links\">{links}</div>"
            f"<div class=\"error\">{html.escape(str(row.get('discovery_errors') or ''))}</div></td>"
            "</tr>"
        )

    option_values = sorted(counts)
    filter_options = '<option value="">All candidate types</option>' + "".join(
        f'<option value="{html.escape(value, quote=True)}">'
        f"{html.escape(value.replace('_', ' '))} ({counts[value]})</option>"
        for value in option_values
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDF discovery and manual-access report</title>
<style>
:root {{ --ink:#17323b; --muted:#64777d; --teal:#087f79; --pale:#edf7f6; --line:#d8e2e4; --warn:#fff4df; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#f5f8f8; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ width:min(1500px,96vw); margin:34px auto 60px; }}
h1 {{ margin:0; font:700 clamp(28px,4vw,48px)/1.1 Georgia,serif; }}
.subtitle {{ color:var(--muted); margin:10px 0 20px; }}
.notice {{ background:var(--warn); border-left:5px solid #d69320; padding:14px 18px; border-radius:8px; margin:18px 0; }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:20px 0; }}
.card {{ background:white; border:1px solid var(--line); border-radius:14px; padding:18px; box-shadow:0 8px 24px #17323b0d; }}
.card strong {{ display:block; color:var(--teal); font-size:30px; }}
.card span {{ color:var(--muted); }}
.toolbar {{ display:flex; gap:12px; margin:20px 0 12px; }}
input,select {{ width:100%; padding:12px 14px; border:1px solid #aebfc3; border-radius:9px; background:white; color:var(--ink); font:inherit; }}
select {{ max-width:360px; }}
.table-wrap {{ overflow:auto; max-height:75vh; background:white; border:1px solid var(--line); border-radius:14px; }}
table {{ border-collapse:separate; border-spacing:0; width:100%; min-width:1280px; }}
th {{ position:sticky; top:0; z-index:1; background:var(--pale); text-align:left; padding:13px; border-bottom:2px solid var(--line); }}
td {{ padding:14px 13px; vertical-align:top; border-bottom:1px solid var(--line); }}
tr:hover td {{ background:#fbfdfd; }}
th:nth-child(1) {{ width:190px; }} th:nth-child(2) {{ width:34%; }} th:nth-child(3) {{ width:27%; }}
.badge {{ display:inline-block; padding:4px 9px; border-radius:999px; color:#06615d; background:#def2f0; font-size:12px; font-weight:700; }}
.muted {{ color:var(--muted); font-size:13px; margin-top:4px; }}
.action {{ margin-top:8px; }}
.links {{ display:flex; flex-wrap:wrap; gap:6px; }}
.source-link {{ color:white; background:var(--teal); text-decoration:none; padding:5px 9px; border-radius:7px; font-size:12px; }}
.source-link:hover {{ background:#056a65; }}
.error {{ color:#9a3c2d; font-size:12px; margin-top:7px; }}
.empty {{ display:none; text-align:center; background:white; border:1px solid var(--line); border-radius:12px; padding:30px; }}
@media(max-width:850px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} .toolbar {{ flex-direction:column; }} select {{ max-width:none; }} }}
</style>
</head>
<body><main>
<h1>PDF discovery &amp; manual-access report</h1>
<p class="subtitle">Generated {html.escape(generated_at)} from local JATS XML, PDF manifests, and cached/official metadata APIs.</p>
<div class="notice"><strong>Access note:</strong> “Free to read” is not the same as open access. This report only discovers links; it never scrapes article pages or downloads candidate PDFs. Use automated retrieval only where the source API and the article license permit it.</div>
<div class="cards">{cards}</div>
<div class="toolbar"><input id="search" type="search" placeholder="Search PMCID, PMID, DOI, title, journal, license…"><select id="source">{filter_options}</select></div>
<div class="table-wrap"><table><thead><tr><th>Identifiers</th><th>Article</th><th>Candidate &amp; next step</th><th>OA / license evidence</th><th>Links</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p class="empty" id="empty">No records match the current filters.</p>
</main>
<script>
const search=document.getElementById('search'), source=document.getElementById('source'), rows=[...document.querySelectorAll('tbody tr')], empty=document.getElementById('empty');
function filterRows() {{ const q=search.value.trim().toLowerCase(), s=source.value; let visible=0; rows.forEach(row=>{{const show=(!q||row.dataset.search.includes(q))&&(!s||row.dataset.source===s); row.hidden=!show; if(show) visible++;}}); empty.style.display=visible?'none':'block'; }}
search.addEventListener('input',filterRows); source.addEventListener('change',filterRows);
</script></body></html>"""


def build_pdf_discovery_report(
    *,
    xml_dir: Path,
    pdf_assets_dir: Path,
    output_dir: Path,
    session: requests.Session | None = None,
    include_ready: bool = False,
    timeout: int = 30,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    cache_max_age_days: float = DEFAULT_CACHE_MAX_AGE_DAYS,
    refresh_cache: bool = False,
    offline: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Discover PDF routes and atomically write CSV, HTML, and summary files."""

    xml_dir = Path(xml_dir)
    pdf_assets_dir = Path(pdf_assets_dir)
    output_dir = Path(output_dir)
    if not xml_dir.is_dir():
        raise ValueError(f"XML directory not found: {xml_dir}")
    if not pdf_assets_dir.is_dir():
        raise ValueError(f"PDF assets directory not found: {pdf_assets_dir}")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if offline and refresh_cache:
        raise ValueError("offline and refresh_cache cannot be used together")
    openalex_api_key = os.environ.get("OPENALEX_API_KEY")

    cache = MetadataCache(
        output_dir / "metadata_cache.json",
        max_age_days=cache_max_age_days,
        refresh=refresh_cache,
    )
    limiter = RequestRateLimiter(request_interval_seconds)
    http = session or _retrying_session()
    close_session = session is None
    rows: list[dict[str, str]] = []
    xml_paths = sorted(xml_dir.glob("PMC*.xml"), key=lambda path: path.name)
    try:
        for xml_path in xml_paths:
            xml_metadata = _parse_xml_metadata(xml_path)
            pmcid = str(xml_metadata["pmcid"]).upper()
            manifest = _read_json_object(pdf_assets_dir / pmcid / "manifest.json")
            status = str(manifest.get("status") or "missing_manifest")
            if not include_ready and status == "ready":
                continue
            if limit is not None and len(rows) >= limit:
                break
            errors: list[str] = []
            europe: dict[str, Any] = {}
            openalex: dict[str, Any] = {}
            try:
                europe = discover_europe_pmc(
                    session=http,
                    cache=cache,
                    limiter=limiter,
                    pmcid=pmcid,
                    pmid=str(xml_metadata["pmid"]),
                    doi=str(xml_metadata["doi"]),
                    timeout=timeout,
                    offline=offline,
                )
            except Exception as error:
                errors.append(
                    _redacted_error("Europe PMC", error, secrets=())
                )
            try:
                openalex = discover_openalex(
                    session=http,
                    cache=cache,
                    limiter=limiter,
                    doi=str(xml_metadata["doi"]),
                    timeout=timeout,
                    offline=offline,
                    api_key=openalex_api_key,
                )
            except Exception as error:
                errors.append(
                    _redacted_error(
                        "OpenAlex",
                        error,
                        secrets=(openalex_api_key or "",),
                    )
                )
            rows.append(
                build_discovery_row(
                    xml_metadata=xml_metadata,
                    manifest=manifest,
                    europe=europe,
                    openalex=openalex,
                    errors=errors,
                )
            )
    finally:
        if close_session:
            http.close()

    rows.sort(key=lambda row: row["pmcid"])
    generated_at = _utc_now().isoformat().replace("+00:00", "Z")
    csv_path = output_dir / "pdf_discovery.csv"
    html_path = output_dir / "pdf_discovery.html"
    summary_path = output_dir / "summary.json"
    _write_csv(csv_path, rows)
    _atomic_write_text(html_path, _render_html(rows, generated_at))
    source_counts: dict[str, int] = {}
    access_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["candidate_source"]] = (
            source_counts.get(row["candidate_source"], 0) + 1
        )
        access_counts[row["access_class"]] = (
            access_counts.get(row["access_class"], 0) + 1
        )
    summary = {
        "format": DISCOVERY_REPORT_VERSION,
        "generated_at": generated_at,
        "metadata_only": True,
        "offline": offline,
        "article_count": len(rows),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "access_class_counts": dict(sorted(access_counts.items())),
        "articles_with_discovery_errors": sum(
            bool(row["discovery_errors"]) for row in rows
        ),
        "csv_path": str(csv_path),
        "html_path": str(html_path),
        "cache_path": str(cache.path),
    }
    _atomic_write_json(summary_path, summary)
    return summary


def _redacted_error(
    provider: str,
    error: Exception,
    *,
    secrets: Iterable[str],
) -> str:
    message = f"{provider}: {type(error).__name__}: {error}"
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a metadata-only CSV/HTML report for PDFs absent from the "
            "reusable PMC Cloud dataset."
        )
    )
    parser.add_argument("--xml-dir", type=Path, default=Path("output/pmc_xml_no_refs"))
    parser.add_argument(
        "--pdf-assets-dir", type=Path, default=Path("output/pmc_pdf_assets")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/pdf_discovery_report")
    )
    parser.add_argument("--include-ready", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-max-age-days", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL_SECONDS,
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.offline and args.refresh_cache:
        parser.error("--offline and --refresh-cache cannot be used together")
    summary = build_pdf_discovery_report(
        xml_dir=args.xml_dir,
        pdf_assets_dir=args.pdf_assets_dir,
        output_dir=args.output_dir,
        include_ready=args.include_ready,
        timeout=args.request_timeout,
        request_interval_seconds=args.request_interval_seconds,
        cache_max_age_days=args.cache_max_age_days,
        refresh_cache=args.refresh_cache,
        offline=args.offline,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
