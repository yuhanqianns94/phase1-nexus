import hashlib
import json
from pathlib import Path

import fitz
import pytest
import requests

from phase1_acquisition.fulltext import fetch_open_fulltext, _links, _safe_url, _validate_material


TITLE = "Safety and tolerability of conditioning chemotherapy followed by CD19-targeted CAR T cells for relapsed/refractory CLL."
DOI = "10.1172/jci.insight.122627"
PDF_URL = "https://insight.jci.org/articles/view/122627/files/pdf"
HTML_URL = "https://insight.jci.org/articles/view/122627"


def candidate(**kwargs):
    return {"document_id": "PMC6538317", "pmcid": "PMC6538317", "pmid": "30938714", "doi": DOI, "title": TITLE, **kwargs}


def free_link(url=PDF_URL, style="pdf", **kwargs):
    return {"url": url, "availabilityCode": "F", "availability": "Free", "documentStyle": style, "site": "DOI", **kwargs}


def pdf_bytes(title=TITLE, doi=DOI, short=False):
    doc = fitz.open()
    page = doc.new_page()
    lines = [title, "doi: " + doi, "Phase I trial actual administered doses and observed toxicities."]
    if not short:
        lines += [f"Patient outcome {n}. Treatment dose was administered and clinical results were observed." for n in range(34)]
    page.insert_textbox(fitz.Rect(30, 30, 560, 800), "\n".join(lines), fontsize=8)
    result = doc.tobytes()
    doc.close()
    return result


def xml_bytes(pmcid="PMC6538317", doi=DOI, pmid="30938714", body=True):
    content = "Observed clinical treatment doses and toxicities in human participants. " * 20 if body else ""
    return f'<article><front><article-meta><article-id pub-id-type="pmc">{pmcid}</article-id><article-id pub-id-type="pmid">{pmid}</article-id><article-id pub-id-type="doi">{doi}</article-id></article-meta></front><body><p>{content}</p></body></article>'.encode()


class Response:
    def __init__(self, payload, mime="application/pdf", status=200, headers=None):
        self.payload = payload
        self.status_code = status
        self.headers = {"Content-Type": mime, **(headers or {})}
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("public retrieval failed")

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert self.responses, "Unexpected network request"
        return self.responses.pop(0)


def test_direct_free_pdf_and_integrity_checked_reuse(tmp_path):
    payload = pdf_bytes()
    http = Session([Response(payload)])
    item = candidate(full_text_urls=[free_link()])
    path = fetch_open_fulltext(item, tmp_path, session=http)
    assert path == tmp_path / "PMC6538317.open.pdf"
    assert path.read_bytes() == payload
    receipt = json.loads((tmp_path / "PMC6538317.open.provenance.json").read_text())
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["source_url"] == PDF_URL
    assert receipt["identity_verification"]["identifiers_matched"] == ["doi:" + DOI]
    assert fetch_open_fulltext(item, tmp_path, session=http) == path
    assert len(http.calls) == 1
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        fetch_open_fulltext(item, tmp_path, session=http)


def test_europepmc_core_identity_checked_and_cached(tmp_path):
    metadata = {"resultList": {"result": [{"id": "30938714", "source": "MED", "doi": DOI, "pmcid": "PMC6538317", "fullTextUrlList": {"fullTextUrl": [free_link()]}}]}}
    http = Session([Response(json.dumps(metadata).encode(), "application/json"), Response(pdf_bytes())])
    path = fetch_open_fulltext(candidate(), tmp_path, session=http)
    assert path and path.suffix == ".pdf"
    assert len(http.calls) == 2
    assert "EXT_ID:30938714" in http.calls[0][0]
    assert (tmp_path / "open_metadata_cache" / "PMID30938714.json").exists()


def test_non_med_metadata_never_creates_pmid_link(tmp_path):
    metadata = {"resultList": {"result": [{"id": "30938714", "source": "PPR", "fullTextUrlList": {"fullTextUrl": [free_link()]}}]}}
    http = Session([Response(json.dumps(metadata).encode(), "application/json")])
    assert fetch_open_fulltext(candidate(), tmp_path, session=http) is None
    assert len(http.calls) == 1


def test_metadata_identifier_conflict_blocks_links(tmp_path):
    metadata = {"resultList": {"result": [{"id": "30938714", "source": "MED", "doi": "10.1000/wrong", "fullTextUrlList": {"fullTextUrl": [free_link()]}}]}}
    http = Session([Response(json.dumps(metadata).encode(), "application/json")])
    assert fetch_open_fulltext(candidate(), tmp_path, session=http) is None
    report = json.loads((tmp_path / "PMC6538317.open_attempts.json").read_text())
    assert "identity mismatch" in report["attempts"][0]["error"]


def test_free_landing_page_uses_declared_pdf_url(tmp_path):
    html = f'<html><meta name="citation_title" content="{TITLE}"><meta name="citation_doi" content="{DOI}"><meta name="citation_pdf_url" content="/articles/view/122627/files/pdf"></html>'
    http = Session([Response(html.encode(), "text/html"), Response(pdf_bytes())])
    path = fetch_open_fulltext(candidate(full_text_urls=[free_link(HTML_URL, "html")]), tmp_path, session=http)
    assert path
    assert http.calls[1][0] == PDF_URL
    receipt = json.loads((tmp_path / "PMC6538317.open.provenance.json").read_text())
    assert receipt["landing_page"]["identifiers_matched"] == ["doi:" + DOI]


def test_pdf_without_identifier_can_use_matching_landing_metadata(tmp_path):
    html = f'<meta name="citation_title" content="{TITLE}"><meta name="citation_doi" content="{DOI}"><meta name="citation_pdf_url" content="{PDF_URL}">'
    http = Session([Response(html.encode(), "text/html"), Response(pdf_bytes(doi=""))])
    assert fetch_open_fulltext(candidate(full_text_urls=[free_link(HTML_URL, "html")]), tmp_path, session=http)


def test_wrong_landing_title_or_identifier_does_not_follow_pdf(tmp_path):
    html = f'<meta name="citation_title" content="Unrelated article"><meta name="citation_pdf_url" content="{PDF_URL}">'
    http = Session([Response(html.encode(), "text/html")])
    assert fetch_open_fulltext(candidate(full_text_urls=[free_link(HTML_URL, "html")]), tmp_path, session=http) is None
    assert len(http.calls) == 1


@pytest.mark.parametrize("payload,mime", [(pdf_bytes(title="An unrelated article with the same DOI in its reference list"), "application/pdf"), (pdf_bytes(doi="10.1000/wrong"), "application/pdf"), (pdf_bytes(doi=DOI + "123"), "application/pdf"), (pdf_bytes(short=True), "application/pdf"), (pdf_bytes(), "text/html"), (b"not a real PDF", "application/pdf")])
def test_wrong_or_unreadable_material_is_not_evidence(tmp_path, payload, mime):
    http = Session([Response(payload, mime)])
    assert fetch_open_fulltext(candidate(full_text_urls=[free_link()]), tmp_path, session=http) is None
    assert not list(tmp_path.glob("*.pdf"))


def test_verified_full_jats_allowed(tmp_path):
    http = Session([Response(xml_bytes(), "application/xml")])
    path = fetch_open_fulltext(candidate(fullTextUrlList={"fullTextUrl": [free_link(style="xml")]}), tmp_path, session=http)
    assert path and path.suffix == ".xml"


@pytest.mark.parametrize("payload", [xml_bytes(pmcid="PMC9999999"), xml_bytes(doi="10.1000/wrong"), xml_bytes(body=False)])
def test_wrong_or_abstract_only_xml_rejected(tmp_path, payload):
    http = Session([Response(payload, "application/xml")])
    assert fetch_open_fulltext(candidate(full_text_urls=[free_link(style="xml")]), tmp_path, session=http) is None


def test_nonfree_links_not_used_and_no_publisher_pdf_guessing(tmp_path):
    assert not _links(candidate(full_text_urls=[free_link(availabilityCode="S", availability="Subscription")]))
    assert not _links(candidate(full_text_urls=[{"url": PDF_URL}]))
    html = b'<html><a href="/guessed.pdf">Read full text</a></html>'
    http = Session([Response(html, "text/html")])
    assert fetch_open_fulltext(candidate(full_text_urls=[free_link(HTML_URL, "html")]), tmp_path, session=http) is None
    assert len(http.calls) == 1


def test_redirect_request_budget_and_private_destination(tmp_path):
    response = Response(b"", status=302, headers={"Location": "http://127.0.0.1/internal"})
    http = Session([response])
    assert fetch_open_fulltext(candidate(full_text_urls=[free_link()]), tmp_path, session=http) is None
    assert response.closed
    assert len(http.calls) == 1
    with pytest.raises(ValueError):
        _safe_url("https://localhost/private")


def test_redirects_and_downloads_share_request_limit(tmp_path):
    http = Session([Response(b"", status=302, headers={"Location": PDF_URL}), Response(pdf_bytes())])
    item = candidate(full_text_urls=[free_link(HTML_URL, "html")])
    assert fetch_open_fulltext(item, tmp_path, session=http, max_requests=1) is None
    assert len(http.calls) == 1


def test_size_bound_rejects_before_content(tmp_path):
    http = Session([Response(pdf_bytes(), headers={"Content-Length": "30000001"})])
    assert fetch_open_fulltext(candidate(full_text_urls=[free_link()]), tmp_path, session=http) is None
    report = json.loads((tmp_path / "PMC6538317.open_attempts.json").read_text())
    assert "byte limit" in report["attempts"][0]["error"]


def test_cached_xml_revalidates_enriched_identifiers_without_rewriting(tmp_path):
    original = candidate(doi="", full_text_urls=[free_link(style="xml")])
    http = Session([Response(xml_bytes(), "application/xml")])
    path = fetch_open_fulltext(original, tmp_path, session=http)
    receipt = tmp_path / "PMC6538317.open.provenance.json"
    original_bytes, original_receipt = path.read_bytes(), receipt.read_bytes()
    assert fetch_open_fulltext({**original, "doi": DOI}, tmp_path, session=http) == path
    assert len(http.calls) == 1
    assert path.read_bytes() == original_bytes
    assert receipt.read_bytes() == original_receipt


def test_cached_xml_rejects_conflicting_added_identifier(tmp_path):
    original = candidate(doi="", full_text_urls=[free_link(style="xml")])
    http = Session([Response(xml_bytes(), "application/xml")])
    path = fetch_open_fulltext(original, tmp_path, session=http)
    receipt = tmp_path / "PMC6538317.open.provenance.json"
    original_bytes, original_receipt = path.read_bytes(), receipt.read_bytes()
    with pytest.raises(ValueError, match="doi identity mismatch"):
        fetch_open_fulltext({**original, "doi": "10.1000/wrong"}, tmp_path, session=http)
    assert len(http.calls) == 1
    assert path.read_bytes() == original_bytes
    assert receipt.read_bytes() == original_receipt


def test_cached_pdf_rejects_unverified_added_identifier(tmp_path):
    original = candidate(pmid="", full_text_urls=[free_link()])
    http = Session([Response(pdf_bytes())])
    path = fetch_open_fulltext(original, tmp_path, session=http)
    with pytest.raises(ValueError, match="could not be verified"):
        fetch_open_fulltext({**original, "pmid": "999999"}, tmp_path, session=http)
    assert len(http.calls) == 1
    assert path.is_file()


def notice_pdf(title_prefix="Correction: ", own_doi="10.1/correction", original_doi="10.1/original"):
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    content = title_prefix + TITLE + "\nDOI: " + own_doi + "\nOriginal article DOI: " + original_doi + "\n"
    content += "This notice corrects author affiliations; the original article contains the clinical trial results. " * 35
    page.insert_textbox(fitz.Rect(25, 25, 675, 875), content, fontsize=10)
    payload = doc.tobytes()
    doc.close()
    return payload


@pytest.mark.parametrize("prefix", ["Correction: ", "Corrigendum: ", "Erratum: ", "Retraction: ", "Publisher Correction to:\n", "Author Correction:\n", "Retraction notice: "])
def test_original_title_and_cited_doi_do_not_identify_a_correction(prefix):
    with pytest.raises(ValueError, match="correction/retraction"):
        _validate_material(notice_pdf(prefix), "application/pdf", candidate(doi="10.1/original"), None)


@pytest.mark.parametrize("prefix", ["Correction: ", "Corrigendum: ", "Erratum: ", "Retraction: "])
def test_notice_candidate_can_match_its_own_title_and_doi(prefix):
    suffix, proof = _validate_material(notice_pdf(prefix), "application/pdf", candidate(title=prefix + TITLE, doi="10.1/correction"), None)
    assert suffix == ".pdf"
    assert proof["primary_identity"]["doi"] == "10.1/correction"
    assert proof["identifiers_matched"] == ["doi:10.1/correction"]


def test_explicit_primary_doi_conflict_rejected_even_if_cited_doi_matches():
    with pytest.raises(ValueError, match="doi identity mismatch"):
        _validate_material(notice_pdf(title_prefix=""), "application/pdf", candidate(doi="10.1/original"), None)


def test_short_report_not_excluded_by_publication_label():
    payload = notice_pdf(title_prefix="Letter: ", own_doi=DOI, original_doi="10.1/other")
    suffix, proof = _validate_material(payload, "application/pdf", candidate(), None)
    assert suffix == ".pdf"
    assert proof["identifiers_matched"] == ["doi:" + DOI]


def test_previously_cached_correction_is_revalidated_and_rejected(tmp_path):
    payload = notice_pdf()
    path = tmp_path / "PMC6538317.open.pdf"
    path.write_bytes(payload)
    item = candidate(doi="10.1/original")
    identity = {key: item[key] for key in ("doi", "pmid", "pmcid")}
    from phase1_extraction.artifacts import sha256_json
    receipt = tmp_path / "PMC6538317.open.provenance.json"
    receipt.write_text(json.dumps({"filename": path.name, "mime": "application/pdf", "identity_sha256": sha256_json(identity), "article_identity": identity, "sha256": hashlib.sha256(payload).hexdigest()}))
    prior_receipt = receipt.read_bytes()
    http = Session([])
    with pytest.raises(ValueError, match="correction/retraction"):
        fetch_open_fulltext(item, tmp_path, session=http)
    assert not http.calls
    assert path.read_bytes() == payload
    assert receipt.read_bytes() == prior_receipt
