"""Bounded recovery from explicitly free publisher/repository links.

This is a fallback after official PMC Cloud retrieval. It never guesses publisher
PDF paths, uses authentication, or converts a landing page into article evidence.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

from phase1_extraction.artifacts import read_json, sha256_file, sha256_json, utc_now, write_json
from .screening import article_identity, normalized

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MAX_BYTES = 30_000_000
MAX_REQUESTS = 8
TIMEOUT = (10, 35)


def _doi(value: str) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(value).strip(), flags=re.I).casefold().rstrip(".,;")


def _identity(record: dict) -> dict:
    return {"doi": _doi(record.get("doi") or ""), "pmid": str(record.get("pmid") or "").strip(),
            "pmcid": str(record.get("pmcid") or "").strip().upper()}


def _matching_ids(expected: dict, actual: dict) -> list[str]:
    """Reject any explicit conflict, even when another identifier matches."""
    expected, actual = _identity(expected), _identity(actual)
    for key in expected:
        if expected[key] and actual[key] and expected[key] != actual[key]:
            raise ValueError(f"Full-text {key} identity mismatch")
    return [key + ":" + expected[key] for key in expected if expected[key] and actual[key] == expected[key]]


def _title_key(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def _safe_url(value: str) -> str:
    parsed = urlsplit(str(value))
    host = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise ValueError("Only public HTTP(S) full-text links are allowed")
    if parsed.port not in {None, 80, 443} or host == "localhost" or host.endswith((".localhost", ".local", ".internal")) or "." not in host:
        raise ValueError("Non-public full-text link")
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not literal_ip.is_global:
        raise ValueError("Non-public full-text link")
    # Old metadata often contains HTTP DOI/publisher links. Never downgrade HTTPS.
    netloc = host if parsed.port in {None, 80, 443} else parsed.netloc
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))


class _Fetcher:
    def __init__(self, session, max_requests: int):
        self.session = session
        self.remaining = max_requests

    def get(self, url: str, *, max_bytes: int = MAX_BYTES) -> tuple[bytes, str, str]:
        url = _safe_url(url)
        for _ in range(4):
            if self.remaining <= 0:
                raise ValueError("Full-text request budget exhausted")
            self.remaining -= 1
            response = self.session.get(url, timeout=TIMEOUT, stream=True, allow_redirects=False)
            try:
                if response.status_code in {301, 302, 303, 307, 308}:
                    destination = response.headers.get("Location")
                    if not destination:
                        raise ValueError("Full-text redirect has no location")
                    url = _safe_url(urljoin(url, destination))
                    continue
                response.raise_for_status()
                size = response.headers.get("Content-Length", "")
                if size.isdigit() and int(size) > max_bytes:
                    raise ValueError("Full-text response exceeds byte limit")
                payload = bytearray()
                for chunk in response.iter_content(128 * 1024):
                    payload.extend(chunk)
                    if len(payload) > max_bytes:
                        raise ValueError("Full-text response exceeds byte limit")
                mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                return bytes(payload), mime, url
            finally:
                response.close()
        raise ValueError("Full-text redirect limit exceeded")


def _links(record: dict) -> list[dict]:
    values = list(record.get("full_text_urls") or [])
    listing = record.get("fullTextUrlList") or {}
    if isinstance(listing, dict):
        values.extend(listing.get("fullTextUrl") or [])
    result = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        code = str(item.get("availabilityCode") or "").strip().casefold()
        availability = str(item.get("availability") or "").strip().casefold()
        if code not in {"f", "free", "oa"} and availability not in {"free", "open access", "free full text"}:
            continue
        try:
            url = _safe_url(item.get("url") or "")
        except ValueError:
            continue
        result[url] = {**item, "url": url}
    priority = {"pdf": 0, "xml": 1, "html": 2, "doi": 3}
    return sorted(result.values(), key=lambda item: priority.get(str(item.get("documentStyle", "")).casefold(), 4))


def _europe_pmc(candidate: dict, directory: Path, fetcher: _Fetcher) -> dict:
    pmid = str(candidate.get("pmid") or "")
    if not pmid.isdigit():
        return {}
    cache = directory / "open_metadata_cache" / f"PMID{pmid}.json"
    if cache.exists():
        record = read_json(cache)["record"]
        if not _matching_ids(candidate, record):
            raise ValueError("Cached full-text metadata identity missing")
        return record
    url = EPMC_SEARCH + f"?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json&resultType=core&pageSize=1"
    payload, mime, final_url = fetcher.get(url, max_bytes=2_000_000)
    if mime not in {"application/json", "text/json"}:
        raise ValueError("Full-text metadata endpoint did not return JSON")
    results = json.loads(payload).get("resultList", {}).get("result", [])
    if len(results) != 1 or str(results[0].get("id")) != pmid or results[0].get("source") != "MED":
        return {}
    record = {**results[0], "pmid": pmid}
    if not _matching_ids(candidate, record):
        raise ValueError("Full-text metadata identity missing")
    write_json(cache, {"metadata_url": final_url, "fetched_at": utc_now(), "record": record})
    return record


class _LandingPage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.alternates: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            key = (attrs.get("name") or attrs.get("property") or "").casefold()
            if key and attrs.get("content"):
                self.meta.setdefault(key, []).append(attrs["content"])
        if tag.lower() == "link" and "alternate" in attrs.get("rel", "").lower().split() and attrs.get("type", "").lower() in {"application/pdf", "application/xml", "text/xml", "application/jats+xml"}:
            self.alternates.append(attrs.get("href", ""))

    def identifiers(self) -> dict:
        names = {"doi": ("citation_doi", "dc.identifier.doi", "prism.doi"),
                 "pmid": ("citation_pmid", "citation_pmid_id"), "pmcid": ("citation_pmcid",)}
        result = {}
        for key, aliases in names.items():
            values = [value for alias in aliases for value in self.meta.get(alias, [])]
            norm = {_identity({key: value})[key] for value in values}
            if len(norm) > 1:
                raise ValueError("Conflicting article identifiers on landing page")
            if values:
                result[key] = values[0]
        return result


def _landing_links(payload: bytes, url: str, candidate: dict) -> tuple[list[str], dict]:
    page = _LandingPage()
    page.feed(payload.decode("utf-8", errors="replace"))
    matches = _matching_ids(candidate, page.identifiers())
    title = _title_key(candidate.get("title") or "")
    titles = page.meta.get("citation_title", []) + page.meta.get("dc.title", [])
    title_match = bool(len(title) >= 30 and any(_title_key(t) == title for t in titles))
    # The authoritative metadata link may identify the article even when a landing
    # page omits citation tags; each downloaded material still needs its own IDs.
    if titles and title and not title_match:
        raise ValueError("Full-text landing-page title mismatch")
    urls = page.meta.get("citation_pdf_url", []) + page.meta.get("citation_xml_url", []) + page.alternates
    links = []
    for target in urls:
        try:
            target = _safe_url(urljoin(url, target))
        except ValueError:
            continue
        if target not in links:
            links.append(target)
    return links[:3], {"identifiers_matched": matches, "title_matched": title_match, "source_url": url}


_NOTICE_HEADING = r"(?:(?:author|publisher)\s+)?(?:correction|corrigendum|erratum|retraction(?:\s+notice)?)"


def _pdf_primary_identity(first_page: str, candidate: dict) -> dict:
    """Separate an article's own header from a cited original article's identity.

    This is intentionally a narrow identity check, not a publication-type filter:
    a correction can still be retrieved when it is the requested candidate.
    """
    title = _title_key(candidate.get("title") or "")
    offsets = []
    key = []
    for offset, char in enumerate(first_page):
        chars = _title_key(char)
        key.extend(chars)
        offsets.extend([offset] * len(chars))
    position = "".join(key).find(title) if title else -1
    if position >= 0 and not re.match(r"^\s*" + _NOTICE_HEADING + r"\b", candidate.get("title") or "", flags=re.I):
        prefix = first_page[max(0, offsets[position] - 300):offsets[position]]
        if re.search(r"(?:^|\n)\s*" + _NOTICE_HEADING + r"(?:\s+(?:to|of|for))?\s*[:\-–—]?\s*$", prefix, flags=re.I):
            raise ValueError("Downloaded PDF is a correction/retraction of the requested article")

    primary = {}
    patterns = {
        "doi": r"(?im)^\s*(?:doi|digital object identifier)\s*[:=]\s*(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{1,9}/[^\s<>\"]+)",
        "pmid": r"(?im)^\s*PMID\s*:\s*(\d+)\b",
        "pmcid": r"(?im)^\s*PMCID\s*:\s*(PMC\d+)\b",
    }
    for identity, pattern in patterns.items():
        for match in re.finditer(pattern, first_page):
            prefix = first_page[max(0, match.start() - 100):match.start()]
            if re.search(r"(?:original|corrected|related|referenced)\s+(?:article|publication|paper)\s*:?\s*$", prefix, flags=re.I):
                continue
            primary[identity] = match.group(1)
            break
    _matching_ids(candidate, primary)
    return _identity(primary)


def _validate_material(payload: bytes, mime: str, candidate: dict, landing: dict | None) -> tuple[str, dict] | None:
    if payload.startswith(b"%PDF-"):
        if mime not in {"application/pdf", "application/octet-stream", "binary/octet-stream"}:
            raise ValueError("PDF signature conflicts with response MIME")
        import fitz
        with fitz.open(stream=payload, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("Encrypted full-text PDF")
            first_page = pdf[0].get_text() if len(pdf) else ""
            first_pages = "\n".join(pdf[i].get_text() for i in range(min(3, len(pdf))))[:100_000]
            text = first_pages
            if len(normalized(text)) <= 2000:
                text += "\n".join(pdf[i].get_text() for i in range(3, min(20, len(pdf))))
            if len(normalized(text)) <= 2000:
                raise ValueError("PDF lacks sufficient readable full text")
        title = _title_key(candidate.get("title") or "")
        if len(title) < 30 or title not in _title_key(first_pages):
            raise ValueError("Downloaded PDF article title mismatch or unavailable")
        primary_identity = _pdf_primary_identity(first_page, candidate)
        expected = _identity(candidate)
        matches = []
        identifier_source = "pdf_first_pages"
        # Permit line wrapping inside a DOI, but never a prefix of another DOI.
        doi_pattern = r"\s*".join(re.escape(c) for c in expected["doi"])
        if expected["doi"] and re.search(doi_pattern + r"(?![A-Za-z0-9/_-])", first_pages, flags=re.I):
            matches.append("doi:" + expected["doi"])
        for key in ("pmid", "pmcid"):
            if expected[key] and re.search(r"(?<![A-Za-z0-9])" + re.escape(expected[key]) + r"(?![A-Za-z0-9])", first_pages, flags=re.I):
                # Unlabelled numerical PMIDs could be clinical counts; require label.
                if key == "pmcid" or re.search(r"PMID\s*:?\s*" + re.escape(expected[key]) + r"\b", first_pages, flags=re.I):
                    matches.append(key + ":" + expected[key])
        if not matches and landing and landing.get("title_matched"):
            matches = list(landing.get("identifiers_matched") or [])
            identifier_source = "landing_page_citation_metadata"
        if not matches:
            raise ValueError("Downloaded PDF has no verified article identifier")
        return ".pdf", {"identifiers_matched": matches, "title_matched": True, "method": "pdf_title_and_identifier", "identifier_source": identifier_source,
                        "primary_identity": primary_identity}
    if mime in {"application/xml", "text/xml", "application/jats+xml", "application/octet-stream"}:
        root = ET.fromstring(payload)
        for element in root.iter():
            element.tag = element.tag.rsplit("}", 1)[-1]
        article = root if root.tag == "article" else root.find(".//article")
        if article is None:
            raise ValueError("Downloaded XML is not a JATS article")
        matches = _matching_ids(candidate, article_identity(article))
        if not matches:
            raise ValueError("Downloaded XML has no verified article identifier")
        body = article.find("body")
        if body is None or len(normalized(" ".join(body.itertext()))) < 200:
            raise ValueError("Downloaded XML has no article body")
        return ".xml", {"identifiers_matched": matches, "method": "jats_article_metadata"}
    if mime in {"text/html", "application/xhtml+xml"}:
        return None
    raise ValueError("Unsupported full-text MIME or missing PDF signature")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_open_fulltext(candidate: dict, output_dir: Path, *, session=None, max_requests: int = MAX_REQUESTS) -> Path | None:
    """Return a verified PDF/JATS path, or None with an auditable failure receipt.

    Only links explicitly marked free by Europe PMC are followed. At most eight
    HTTP requests (including redirects/metadata) are made by default. Existing
    verified material is reused without a network request. No model calls occur.
    """
    document_id = str(candidate.get("document_id") or candidate.get("pmcid") or "PMID" + str(candidate.get("pmid") or ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", document_id):
        raise ValueError("Unsafe full-text document_id")
    if not isinstance(max_requests, int) or isinstance(max_requests, bool) or not 1 <= max_requests <= 20:
        raise ValueError("Full-text request limit must be between 1 and 20")
    directory = Path(output_dir)
    identity_hash = sha256_json(_identity(candidate))
    receipt = directory / f"{document_id}.open.provenance.json"
    if receipt.exists():
        prior = read_json(receipt)
        path = directory / str(prior.get("filename") or "")
        prior_identity = _identity(prior.get("article_identity") or {})
        if (path.parent.resolve() != directory.resolve() or not path.is_file()
                or prior.get("identity_sha256") != sha256_json(prior_identity)
                or prior.get("sha256") != sha256_file(path)):
            raise ValueError("Existing open-fulltext receipt/material integrity mismatch")
        if not _matching_ids(candidate, prior_identity):
            raise ValueError("Cached full-text article identity does not overlap requested article")
        # Metadata enrichment is not changed article content. Revalidate the exact
        # cached bytes, including every newly supplied identifier, without rewriting
        # immutable materials/receipts or making a network request.
        expected = _identity(candidate)
        check_candidate = {**candidate, **{key: expected[key] or prior_identity[key] for key in expected}}
        result = _validate_material(path.read_bytes(), prior.get("mime", ""), check_candidate, prior.get("landing_page"))
        if result is None:
            raise ValueError("Cached full-text material is not a verified article")
        new_ids = {key + ":" + value for key, value in expected.items() if value and not prior_identity[key]}
        if not new_ids.issubset(set(result[1]["identifiers_matched"])):
            raise ValueError("New article identifiers could not be verified against cached full text")
        return path
    http = session or requests.Session()
    if session is None:
        http.headers["User-Agent"] = "phase1-literature-acquisition/1.0 (public free full-text recovery)"
    fetcher = _Fetcher(http, max_requests)
    attempts = []
    try:
        links = _links(candidate)
        if not links:
            try:
                links = _links(_europe_pmc(candidate, directory, fetcher))
            except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                attempts.append({"stage": "metadata", "error": str(error)[:300]})
        queue = [(item["url"], item, None) for item in links]
        visited = set()
        while queue and fetcher.remaining > 0:
            url, source, landing = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                payload, mime, final_url = fetcher.get(url)
                result = _validate_material(payload, mime, candidate, landing)
                if result is None:
                    # Follow a landing page once, using only publisher-supplied
                    # citation/alternate full-text links rather than guessed URLs.
                    if landing is None:
                        additional, proof = _landing_links(payload, final_url, candidate)
                        queue[0:0] = [(link, source, proof) for link in additional if link not in visited]
                    attempts.append({"url": url, "final_url": final_url, "status": "landing_page"})
                    continue
                suffix, proof = result
                path = directory / f"{document_id}.open{suffix}"
                _atomic_bytes(path, payload)
                write_json(receipt, {"format": "open_fulltext_v1", "document_id": document_id,
                    "identity_sha256": identity_hash, "article_identity": _identity(candidate),
                    "filename": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                    "source_url": final_url, "metadata_link": source, "landing_page": landing,
                    "retrieved_at": utc_now(), "identity_verification": proof,
                    "mime": mime, "bytes": len(payload), "http_requests": max_requests - fetcher.remaining})
                return path
            except (requests.RequestException, ValueError, ET.ParseError, RuntimeError) as error:
                attempts.append({"url": url, "error": str(error)[:300]})
        return None
    finally:
        write_json(directory / f"{document_id}.open_attempts.json", {"document_id": document_id,
            "attempted_at": utc_now(), "http_requests": max_requests - fetcher.remaining,
            "request_limit": max_requests, "attempts": attempts, "recovered": receipt.exists()})
        if session is None:
            http.close()
