"""Offline public upload, isolation, explicit credentials and cache regressions."""

import json
from pathlib import Path

import fitz
import pytest

from phase1_extraction import llm, pipeline
from phase1_extraction.models import DocumentIR, ReportedExtraction
from phase1_web import user_library as library


ARTICLE = b"<article><front><article-title>Example trial</article-title></front><body><sec><p>Twenty participants received study treatment.</p></sec></body></article>"
KEY = "sk-test-visitor-private-key"


class Client:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.responses = self
        self.calls = []
        self.closed = False

    def create(self, **body):
        self.calls.append(body)
        data = json.loads(body["input"][0]["content"][0]["text"])
        result = ReportedExtraction(document_id=data["document_id"], title="A trial", extraction_complete=True)
        return {"id": "test-response", "status": "completed",
                "usage": {"input_tokens": 100, "output_tokens": 30},
                "output": [{"content": [{"type": "output_text", "text": result.model_dump_json()}]}]}

    def close(self):
        self.closed = True


def kwargs(tmp_path):
    return dict(workspace=tmp_path,
                document_paths=library.stage_uploaded_documents(tmp_path, [("paper.xml", ARTICLE)]),
                api_key=KEY, model="user-chosen-model", reasoning_effort="",
                max_output_tokens=4096, max_usd=10,
                input_usd_per_million=2, output_usd_per_million=4)


def test_uploaded_filename_never_controls_paths(tmp_path):
    paths = library.stage_uploaded_documents(tmp_path, [("../../escape.xml", ARTICLE)])
    assert paths[0].startswith("uploads/") and len(Path(paths[0]).stem) == 64
    assert (tmp_path / paths[0]).read_bytes() == ARTICLE
    assert not (tmp_path.parent / "escape.xml").exists()


@pytest.mark.parametrize("relative", ["../../secrets.xml", "/tmp/secret.pdf", "uploads/../secret.pdf", "uploads/file.xml"])
def test_worker_rejects_path_escape(tmp_path, relative):
    options = kwargs(tmp_path)
    options["document_paths"] = [relative]
    with pytest.raises(library.UserLibraryError, match="unavailable"):
        library.extract_uploaded_documents(**options)


def test_symlink_cannot_access_another_session(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    paths = library.stage_uploaded_documents(first, [("paper.xml", ARTICLE)])
    second.mkdir()
    (second / "uploads").symlink_to(first / "uploads", target_is_directory=True)
    with pytest.raises(library.UserLibraryError, match="unavailable"):
        library._load_staged_documents(second, paths)


def test_modified_staged_document_is_rejected(tmp_path):
    options = kwargs(tmp_path)
    (tmp_path / options["document_paths"][0]).write_bytes(ARTICLE + b" ")
    with pytest.raises(library.UserLibraryError, match="changed"):
        library.extract_uploaded_documents(**options)


@pytest.mark.parametrize("data", [
    b'<!DOCTYPE article [<!ENTITY x SYSTEM "file:///etc/passwd">]><article>&x;</article>',
    b'<!DOCTYPE article SYSTEM "https://example.com/data.dtd"><article/>',
    b'<article xmlns:xi="http://www.w3.org/2001/XInclude"><xi:include href="file:///etc/passwd"/></article>',
    b'<?xml-stylesheet href="https://example.com/a"?><article/>',
    '<!DOCTYPE article [<!ENTITY x "boom">]><article>&x;</article>'.encode("utf-16"),
])
def test_xml_external_resources_and_entities_rejected_before_storage(tmp_path, data):
    with pytest.raises(library.UserLibraryError, match="standalone JATS"):
        library.stage_uploaded_documents(tmp_path, [("unsafe.xml", data)])
    assert not (tmp_path / "uploads").exists()


def test_xml_depth_bound(tmp_path):
    data = b"<article>" + b"<sec>" * 65 + b"x" + b"</sec>" * 65 + b"</article>"
    with pytest.raises(library.UserLibraryError, match="complexity"):
        library.stage_uploaded_documents(tmp_path, [("deep.xml", data)])


def test_xml_table_span_is_bounded_before_parsing(tmp_path):
    data = b'<article><body><table-wrap><table><tr><td rowspan="9999999999">x</td></tr></table></table-wrap></body></article>'
    with pytest.raises(library.UserLibraryError, match="complexity"):
        library.stage_uploaded_documents(tmp_path, [("span.xml", data)])


def test_xml_total_expanded_table_grid_is_bounded(tmp_path):
    data = b'<article><body><table-wrap><table><tr>' + b'<td rowspan="100" colspan="100">x</td>' * 3 + b'</tr></table></table-wrap></body></article>'
    with pytest.raises(library.UserLibraryError, match="complexity"):
        library.stage_uploaded_documents(tmp_path, [("span.xml", data)])


@pytest.mark.parametrize("resource", [
    '<graphic href="../../owner.env"/>',
    '<graphic href="/etc/passwd"/>',
    '<graphic xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="file:///etc/passwd"/>',
    '<graphic href="https://example.com/image.png"/>',
    '<inline-graphic href="data:image/png;base64,AAAA"/>',
    '<media src="../source.pdf"/>',
    '<media data="/private/secrets"/>',
    '<table-wrap xml:base="file:///private/"/>',
    '<ext-link href="file:///etc/passwd">Not a web citation</ext-link>',
])
def test_xml_resource_references_are_rejected(tmp_path, resource):
    data = f'<article><body><table-wrap>{resource}</table-wrap></body></article>'.encode()
    with pytest.raises(library.UserLibraryError, match="cannot reference"):
        library.stage_uploaded_documents(tmp_path, [("unsafe.xml", data)])
    assert not (tmp_path / "uploads").exists()


def test_worker_revalidation_blocks_existing_host_file_reference(tmp_path, monkeypatch):
    import hashlib
    secret = tmp_path / "owner-secret.png"
    secret.write_bytes(b"private-owner-credential")
    workspace = tmp_path / "visitor"
    options = kwargs(workspace)
    data = f'<article><body><table-wrap><graphic href="{secret}"/></table-wrap></body></article>'.encode()
    name = hashlib.sha256(data).hexdigest() + ".xml"
    (workspace / "uploads" / name).write_bytes(data)
    options["document_paths"] = ["uploads/" + name]
    monkeypatch.setattr(library, "OpenAI", lambda **_: pytest.fail("must reject before creating API client"))
    with pytest.raises(library.UserLibraryError, match="cannot reference"):
        library.extract_uploaded_documents(**options)
    assert not (workspace / "library").exists()


def test_bibliographic_web_links_remain_usable_without_fetching(tmp_path, monkeypatch):
    client = Client()
    monkeypatch.setattr(library, "OpenAI", lambda **_: client)
    options = kwargs(tmp_path)
    data = b'<article xmlns:xlink="http://www.w3.org/1999/xlink"><body><p>See <ext-link xlink:href="https://doi.org/10.1234/example">the published study</ext-link>.</p></body></article>'
    options["document_paths"] = library.stage_uploaded_documents(tmp_path, [("citations.xml", data)])
    result = library.extract_uploaded_documents(**options)
    assert result["completed"] and not result["failed"]


def test_no_partial_writes_when_one_upload_invalid(tmp_path):
    with pytest.raises(library.UserLibraryError, match="Only PDF"):
        library.stage_uploaded_documents(tmp_path, [("good.xml", ARTICLE), ("bad.zip", b"bad")])
    assert not (tmp_path / "uploads").exists()


def test_upload_bounds(tmp_path, monkeypatch):
    with pytest.raises(library.UserLibraryError, match="1 to 10"):
        library.stage_uploaded_documents(tmp_path, [("a.xml", ARTICLE)] * 11)
    monkeypatch.setattr(library, "MAX_FILE_BYTES", 10)
    with pytest.raises(library.UserLibraryError, match="20 MB"):
        library.stage_uploaded_documents(tmp_path, [("a.pdf", b"%PDF-" * 3)])


def test_pdf_validation_and_page_limit(tmp_path):
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((80, 80), "Clinical trial")
        valid = document.tobytes()
    assert library.stage_uploaded_documents(tmp_path, [("article.pdf", valid)])
    with fitz.open() as document:
        for _ in range(81):
            document.new_page()
        too_many = document.tobytes()
    with pytest.raises(library.UserLibraryError, match="80 pages"):
        library.stage_uploaded_documents(tmp_path, [("long.pdf", too_many)])


def test_oversized_pdf_render_rejected(tmp_path):
    with fitz.open() as document:
        document.new_page(width=5000, height=5000)
        data = document.tobytes()
    with pytest.raises(library.UserLibraryError, match="rendering size"):
        library.stage_uploaded_documents(tmp_path, [("large.pdf", data)])


@pytest.mark.parametrize("key", ["", "  ", "short", "a key with spaces"])
def test_missing_or_invalid_key_never_falls_back_to_owner(tmp_path, monkeypatch, key):
    options = kwargs(tmp_path)
    options["api_key"] = key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner-do-not-use")
    monkeypatch.setattr(library, "OpenAI", lambda **_: pytest.fail("must not create client"))
    with pytest.raises(library.UserLibraryError, match="own OpenAI API key"):
        library.extract_uploaded_documents(**options)


def test_explicit_user_client_and_model_flow_through_pipeline_and_reuse_cache(tmp_path, monkeypatch):
    clients = []

    def client_factory(**options):
        client = Client(**options)
        clients.append(client)
        return client

    monkeypatch.setenv("PHASE1_DEPLOYMENT", "public")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner-never-use")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://do-not-send-user-key.invalid/")
    monkeypatch.setattr(library, "OpenAI", client_factory)
    options = kwargs(tmp_path)
    first = library.extract_uploaded_documents(**options)
    second = library.extract_uploaded_documents(**options)
    assert first["completed"] and not first["failed"]
    assert second["completed"] == first["completed"]
    assert len(clients[0].calls) == 1 and clients[1].calls == []
    assert clients[0].options["api_key"] == KEY
    assert clients[0].options["base_url"] == "https://api.openai.com/v1"
    assert clients[0].calls[0]["model"] == "user-chosen-model"
    assert "reasoning" not in clients[0].calls[0]
    assert all(client.closed for client in clients)
    assert first["estimated_usd"] > 0 and second["estimated_usd"] == 0
    assert list((tmp_path / "library").glob("*/resolved_extraction.json"))
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert KEY.encode() not in path.read_bytes()
            assert b"sk-owner-never-use" not in path.read_bytes()


def test_private_request_cache_never_crosses_session(tmp_path, monkeypatch):
    clients = []

    def factory(**options):
        clients.append(Client(**options))
        return clients[-1]

    monkeypatch.setattr(library, "OpenAI", factory)
    first = library.extract_uploaded_documents(**kwargs(tmp_path / "first"))
    second = library.extract_uploaded_documents(**kwargs(tmp_path / "second"))
    assert first["completed"] and second["completed"]
    assert [len(client.calls) for client in clients] == [1, 1]


def test_grouped_article_recognizes_supplied_supplement(tmp_path, monkeypatch):
    client = Client()
    monkeypatch.setattr(library, "OpenAI", lambda **_: client)
    options = kwargs(tmp_path)
    main = ARTICLE.replace(b"Twenty participants", b"See supplementary table 1. Twenty participants")
    supplement = ARTICLE.replace(b"Example trial", b"Supplementary table 1")
    options["document_paths"] = library.stage_uploaded_documents(tmp_path, [("main.xml", main), ("supp.xml", supplement)])
    result = library.extract_uploaded_documents(**options, group_as_article=True)
    assert len(result["completed"]) == 1
    payload = json.loads(client.calls[0]["input"][0]["content"][0]["text"])
    assert [material["role"] for material in payload["materials"]] == ["main", "supplement"]
    assert payload["material_reference_inventory"][0]["availability"] == "supplied"


def test_api_error_with_secret_is_not_written_or_returned(tmp_path, monkeypatch):
    class FailingClient(Client):
        def create(self, **_body):
            raise RuntimeError("provider failure: " + KEY)

    monkeypatch.setattr(library, "OpenAI", FailingClient)
    result = library.extract_uploaded_documents(**kwargs(tmp_path))
    assert result["failed"] and KEY not in str(result)
    for path in tmp_path.rglob("*.json"):
        assert KEY not in path.read_text()


def test_positive_prices_are_required_before_client_creation(tmp_path, monkeypatch):
    options = kwargs(tmp_path)
    options["input_usd_per_million"] = 0
    monkeypatch.setattr(library, "OpenAI", lambda **_: pytest.fail("must not create client"))
    with pytest.raises(library.UserLibraryError, match="current input"):
        library.extract_uploaded_documents(**options)


def test_rendered_asset_limit_stops_before_any_paid_request(tmp_path, monkeypatch):
    options = kwargs(tmp_path)
    with fitz.open() as document:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 500, 300), "A clinical trial reported patient counts and toxicity. " * 8)
        data = document.tobytes()
    options["document_paths"] = library.stage_uploaded_documents(tmp_path, [("article.pdf", data)])
    client = Client()
    monkeypatch.setattr(library, "OpenAI", lambda **_: client)
    monkeypatch.setattr(library, "MAX_PARSED_ASSET_BYTES", 1)
    result = library.extract_uploaded_documents(**options)
    assert not result["completed"] and client.calls == []
    assert "too large" in result["failed"][0]["message"]


def test_public_pipeline_without_user_client_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE1_DEPLOYMENT", "public")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner-never-use")
    with pytest.raises(ValueError, match="explicit user API client"):
        pipeline._extract_reported(None, execute_model=True)
    with pytest.raises(ValueError, match="explicit user API client"):
        llm.extract_packets(packets=[], document=DocumentIR(document_id="D", materials=[]),
                            raw_response_dir=tmp_path / "raw")


def test_local_model_and_reasoning_defaults_unchanged():
    import inspect
    for function in (pipeline._run_pipeline, pipeline._extract_reported, llm._extract_packets):
        parameters = inspect.signature(function).parameters
        assert parameters["model"].default == llm.DEFAULT_MODEL
        assert parameters["reasoning_effort"].default == llm.DEFAULT_REASONING_EFFORT
    assert llm.DEFAULT_MODEL == "gpt-6.1-sol"
    assert llm.DEFAULT_REASONING_EFFORT == "low"
