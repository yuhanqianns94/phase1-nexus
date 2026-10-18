"""Frozen PubMed ranking for selection by verified full-text availability."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import tempfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from phase1_extraction.artifacts import read_json, sha256_file, sha256_json, utc_now, write_json
from .screening import article_element, article_identity, normalized

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PHASE_TERMS = '("Clinical Trial, Phase I"[pt] OR "phase I"[tiab] OR "phase 1"[tiab] OR "phase Ia"[tiab] OR "phase Ib"[tiab] OR "phase 1a"[tiab] OR "phase 1b"[tiab] OR "first-in-human"[tiab] OR "dose escalation"[tiab] OR "dose finding"[tiab])'
ONCOLOGY_TERMS = '("Neoplasms"[mh] OR cancer*[tiab] OR tumor*[tiab] OR tumour*[tiab] OR neoplasm*[tiab] OR malignan*[tiab] OR leukemia*[tiab] OR leukaemia*[tiab] OR lymphoma*[tiab] OR myeloma*[tiab] OR sarcoma*[tiab] OR carcinoma*[tiab] OR melanoma*[tiab] OR glioma*[tiab] OR glioblastoma*[tiab] OR mesothelioma*[tiab] OR neuroblastoma*[tiab] OR myelodysplastic*[tiab] OR myelofibrosis*[tiab])'
PUBMED_QUERY = PHASE_TERMS + " AND " + ONCOLOGY_TERMS
ARTICLE_COUNT = 4000
PUBMED_SORT = "pub_date"
# PubMed ESearch exposes at most 9,999 ranked results. This is a retrieval
# boundary, not the article target; exhaustion must stop without paid submission.
SEARCH_LIMIT = 9999


def http_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "phase1-literature-acquisition/1.0"
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
    return session


def text(element) -> str:
    return normalized(" ".join(element.itertext())) if element is not None else ""


class PublicMetadata:
    def __init__(self, cache_dir: Path, session=None):
        self.cache_dir = Path(cache_dir)
        self.session = session or http_session()

    def get(self, url: str, params: dict, *, xml=False):
        # API keys are never serialized to the cache or diagnostic messages.
        clean = {k: v for k, v in params.items() if k != "api_key"}
        path = self.cache_dir / (sha256_json([url, clean]) + ".json")
        if path.exists():
            return read_json(path)["payload"]
        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.text if xml else response.json()
            if xml:
                parsed = ET.fromstring(payload)
                if parsed.find(".//ERROR") is not None:
                    raise ValueError("NCBI returned an error document")
            elif isinstance(payload, dict) and payload.get("error"):
                raise ValueError("Metadata endpoint returned an error")
            write_json(path, {"url": url, "params": clean, "fetched_at": utc_now(), "payload": payload})
            return payload
        except requests.RequestException as error:
            raise RuntimeError(f"Public metadata request failed: {type(error).__name__}") from None
        finally:
            time.sleep(0.35)

    def ncbi_params(self):
        params = {"tool": "phase1_literature_acquisition"}
        if os.environ.get("NCBI_API_KEY"):
            params["api_key"] = os.environ["NCBI_API_KEY"]
        if os.environ.get("NCBI_EMAIL"):
            params["email"] = os.environ["NCBI_EMAIL"]
        return params

    def pubmed_records(self, pmids: list[str], origin: dict) -> list[dict]:
        records = []
        for start in range(0, len(pmids), 100):
            xml = self.get(EUTILS + "efetch.fcgi", {**self.ncbi_params(), "db": "pubmed", "id": ",".join(pmids[start:start + 100]), "retmode": "xml"}, xml=True)
            for article in ET.fromstring(xml):
                book = article.tag == "PubmedBookArticle"
                ids = {i.get("IdType"): text(i) for i in article.findall(
                    "./BookDocument/ArticleIdList/ArticleId" if book else "./PubmedData/ArticleIdList/ArticleId")}
                body = article.find("BookDocument" if book else "./MedlineCitation/Article")
                if body is None:
                    continue
                records.append({"pmid": ids.get("pubmed") or text(article.find(
                                    "./BookDocument/PMID" if book else "./MedlineCitation/PMID")),
                                "pmcid": ids.get("pmc", "").upper(), "doi": ids.get("doi", ""),
                                "title": text(body.find("ArticleTitle")) or text(body.find("./Book/BookTitle")),
                                "abstract": text(body.find("Abstract")),
                                "publication_types": [text(x) for x in body.findall("./PublicationTypeList/PublicationType")] or (["Book"] if book else []),
                                "origins": [origin]})
        return records

def discover(workspace: Path, *, through: str, client=None) -> dict:
    """Freeze the ranked search pool; full-text selection happens after retrieval."""
    from phase1_extraction.request_cache import artifact_lock
    workspace = Path(workspace)
    cutoff = date.fromisoformat(through.replace("/", "-")).strftime("%Y/%m/%d")
    query = PUBMED_QUERY + f' AND ("1800/01/01"[dp] : "{cutoff}"[dp])'
    configuration = {"source": "pubmed", "query": query, "sort": PUBMED_SORT,
                     "through": cutoff, "limit": SEARCH_LIMIT}
    with artifact_lock(workspace / ".discovery.lock"):
        search_path = workspace / "pubmed_search.json"
        client = client or PublicMetadata(workspace / "metadata_cache")
        if search_path.exists():
            search = read_json(search_path)
            if search["configuration"] != configuration:
                raise ValueError("Frozen PubMed search configuration changed")
            if sha256_json(search["pmids"]) != search["pmids_sha256"]:
                raise ValueError("Frozen PMID order changed")
        else:
            response = client.get(EUTILS + "esearch.fcgi", {**client.ncbi_params(),
                "db": "pubmed", "term": query, "retmode": "json", "retmax": SEARCH_LIMIT,
                "retstart": 0, "sort": PUBMED_SORT})
            result = response.get("esearchresult", {})
            pmids = result.get("idlist", [])
            if (result.get("ERROR") or not str(result.get("count", "")).isdigit()
                    or len(pmids) != min(int(result["count"]), SEARCH_LIMIT)
                    or len(set(pmids)) != len(pmids) or any(not str(p).isdigit() for p in pmids)):
                raise ValueError("PubMed did not return the requested distinct PMIDs; nothing submitted")
            search = {"configuration": configuration, "created_at": utc_now(),
                      "total_hits": int(result["count"]), "pmids": pmids,
                      "pmids_sha256": sha256_json(pmids),
                      "query_translation": result.get("querytranslation")}
            write_json(search_path, search)
        pmids = search["pmids"]
        if len(pmids) != min(search["total_hits"], SEARCH_LIMIT) or len(set(pmids)) != len(pmids):
            raise ValueError("Invalid frozen PubMed ranking")
        result_path = workspace / "ranked_candidates.json"
        receipt_path = workspace / "discovery_receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (sha256_file(result_path) != receipt["candidates_sha256"]
                    or sha256_file(search_path) != receipt["search_sha256"]):
                raise ValueError("Frozen discovery artifacts changed")
            return read_json(result_path)
        records = []
        for offset in range(0, len(pmids), 100):
            ids = pmids[offset:offset + 100]
            fetched = client.pubmed_records(ids, {"source": "pubmed", "query": query,
                                                  "sort": PUBMED_SORT})
            by_id = {r["pmid"]: r for r in fetched}
            if len(fetched) != len(ids) or set(by_id) != set(ids):
                raise ValueError("PubMed metadata did not return all requested PMIDs; selection unchanged")
            for pmid in ids:
                record = by_id[pmid]
                # PMID is the unique unit returned by this single-source search.
                records.append({**record, "document_id": "PMID" + pmid, "rank": len(records) + 1})
            write_json(workspace / "discovery_progress.json", {
                "fetched": len(records), "ranked_pool": len(pmids), "updated_at": utc_now()})
        result = {"format": "pubmed_ranked_cohort_v1", **configuration,
                  "total_hits": search["total_hits"], "candidate_count": len(records),
                  "pmids_sha256": search["pmids_sha256"], "candidates": records}
        write_json(result_path, result)
        write_json(receipt_path, {"candidates_sha256": sha256_file(result_path),
                                 "search_sha256": sha256_file(search_path)})
        return result


def _immutable_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish new material without replacing an existing inode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"Refusing to overwrite existing PMC XML material: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verified_xml_cache(path: Path, pmcid: str) -> Path | None:
    receipt = path.with_suffix(".provenance.json")
    if not receipt.exists():
        return None
    proof = read_json(receipt)
    if not path.is_file() or proof.get("sha256") != sha256_file(path):
        raise ValueError(f"PMC XML cache integrity mismatch for {pmcid}: {path}")
    if article_identity(article_element(path)).get("pmcid") != pmcid:
        raise ValueError(f"Cached PMC XML article identity mismatch for {pmcid}: {path}")
    return path


def fetch_fulltext(candidate: dict, output_dir: Path, *, session=None) -> Path | None:
    """Official reusable PMC Cloud XML; existing source bytes are immutable.

    An unreceipted legacy XML (including a title-only stub) remains in place. New
    full text is stored by content hash under versions/<PMCID>/ and reused there.
    A previously verified but changed/missing material is an integrity failure,
    never a request to silently replace source files bound to historical results.
    """
    from .visual_assets import _cloud_article_metadata, _cloud_object_record, _download_limited
    pmcid = candidate.get("pmcid", "")
    if not re.fullmatch(r"PMC\d+", pmcid):
        return None
    path = Path(output_dir) / (pmcid + ".xml")
    cached = _verified_xml_cache(path, pmcid)
    if cached:
        return cached
    versions = Path(output_dir) / "versions" / pmcid
    for receipt in sorted(versions.glob("*.provenance.json")):
        name = receipt.name.removesuffix(".provenance.json")
        cached = _verified_xml_cache(versions / (name + ".xml"), pmcid)
        if cached:
            return cached
    http = session or http_session()
    try:
        metadata, prefix, metadata_url = _cloud_article_metadata(http, pmcid, 60)
    except FileNotFoundError:
        return None
    url = metadata.get("xml_url")
    if not url:
        return None
    record = _cloud_object_record(url)
    if not record["object_key"].startswith(prefix):
        raise ValueError("Full-text object does not match article version")
    payload = _download_limited(http, record["source_url"], timeout=60, max_bytes=20_000_000)
    if record["expected_md5"] and hashlib.md5(payload).hexdigest() != record["expected_md5"]:
        raise ValueError("PMC XML checksum mismatch")
    root = ET.fromstring(payload)
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    article = root if root.tag == "article" else root.find(".//article")
    if article is None or article_identity(article).get("pmcid") != pmcid:
        raise ValueError("Downloaded XML article identity mismatch")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        path = versions / (digest + ".xml")
    _immutable_bytes(path, payload)
    receipt = path.with_suffix(".provenance.json")
    proof = {"source_url": record["source_url"], "metadata_url": metadata_url,
             "version": prefix, "sha256": digest, "license_code": metadata.get("license_code")}
    _immutable_bytes(receipt, (json.dumps(proof, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return path
