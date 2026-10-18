from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase1_acquisition.screening import (LEGACY_CRITERIA as CRITERIA, Decision, build_document, make_request, validate_decision,
                                         MODEL, EFFORT, PROMPT, validate_screening_gate)
from phase1_acquisition.screening_batch import freeze_documents, submit, collect, export_retained
from phase1_acquisition.discovery import PublicMetadata
from phase1_acquisition.acquire import prepare_output_dir
from phase1_extraction.artifacts import read_json, write_json
from phase1_extraction.batch import load_batch_manifest


def document(tmp_path, name="PMC123", fulltext=True):
    xml = tmp_path / (name + ".xml")
    title = "Phase Ib trial of drug X in cancer patients"
    result = "Twelve cancer patients received 10 mg of drug X. No dose limiting toxicity occurred."
    xml.write_text(f'<article><front><article-meta><article-id pub-id-type="pmc">{name}</article-id>'
                   f'<title-group><article-title>{title}</article-title></title-group>'
                   f'<abstract><p>{result}</p></abstract></article-meta></front>'
                   + (f'<body><sec><title>Results</title><p>{result}</p></sec></body>' if fulltext else "")
                   + '<back><ref-list><ref>OTHER study with Phase I results.</ref></ref-list></back></article>')
    doc = build_document({"document_id": name, "pmcid": name}, [xml])
    # Keep the original transport/replay regression fixtures on the immutable v1 schema.
    doc["policy_version"] = "phase1_binary_20261010_v1"
    return doc


def response(doc, keep=True):
    block = next(b for b in doc["blocks"] if "Twelve" in b["text"])
    checks = [{"criterion": c, "satisfied": keep or c != "actual_patient_results", "reason": "Supported by article evidence." if keep else "Planned results only.",
               "evidence": [{"source_id": block["source_id"], "quote": "Twelve cancer patients received 10 mg of drug X."}]} for c in sorted(CRITERIA)]
    value = {"document_id": doc["document_id"], "decision": "\u4fdd\u7559" if keep else "\u4e0d\u4fdd\u7559", "reason": "Actual dose and outcomes are reported." if keep else "Observed outcomes are not established.", "checks": checks}
    return {"status": "completed", "model": MODEL, "reasoning": {"effort": EFFORT},
            "usage": {"input_tokens": 1000, "output_tokens": 500, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}


def change_result(raw, fn):
    value = json.loads(raw["output"][0]["content"][0]["text"])
    fn(value)
    raw["output"][0]["content"][0]["text"] = json.dumps(value)
    return raw


class FakeClient:
    def __init__(self, responses=None, unknown=False):
        self.creates = 0
        self.responses = responses or []
        self.unknown = unknown
        self.files = SimpleNamespace(create=lambda **kw: SimpleNamespace(id="file1"), content=self.content)
        self.batches = SimpleNamespace(create=self.create, retrieve=lambda _: SimpleNamespace(
            id="batch1", status="completed", request_counts=SimpleNamespace(model_dump=lambda: {"completed": len(self.responses)}),
            output_file_id="output1", error_file_id=None))

    def with_options(self, **kwargs):
        assert kwargs == {"max_retries": 0}
        return self

    def create(self, **kwargs):
        self.creates += 1
        if self.unknown:
            raise TimeoutError("Ambiguous response")
        return SimpleNamespace(id="batch1", status="validating")

    def content(self, _):
        return SimpleNamespace(content=("\n".join(json.dumps(r) for r in self.responses)).encode())


def submitted(tmp_path, documents):
    workspace = tmp_path / "batch"
    plan = freeze_documents(documents, workspace)
    rows = [{"custom_id": r["custom_id"], "response": {"status_code": 200, "body": response(r["document"])}, "error": None} for r in plan["requests"]]
    client = FakeClient(rows)
    submit(workspace, client=client)
    return workspace, client


def test_binary_schema_and_verifiable_evidence(tmp_path):
    doc = document(tmp_path)
    assert validate_decision(response(doc), doc).decision == "\u4fdd\u7559"
    assert validate_decision(response(doc, False), doc).decision == "\u4e0d\u4fdd\u7559"
    assert set(Decision.model_json_schema()["properties"]["decision"]["enum"]) == {"\u4fdd\u7559", "\u4e0d\u4fdd\u7559"}
    assert "OTHER study" not in json.dumps(doc)
    assert make_request(doc)["text"]["format"]["strict"] is True
    assert "zero DLT" in PROMPT and "missing DLT window" in PROMPT


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(decision="\u4e0d\u4fdd\u7559"),
    lambda r: r.update(decision="\u5f85\u5b9a"),
    lambda r: r.update(document_id="OTHER"),
    lambda r: r["checks"].pop(),
    lambda r: r["checks"][0].update(evidence=[]),
    lambda r: r["checks"][0]["evidence"][0].update(quote="Invented evidence from another paper"),
    lambda r: r["checks"][0]["evidence"][0].update(source_id="references"),
    lambda r: r["checks"][0].update(satisfied="true"),
])
def test_invalid_decision_never_passes(tmp_path, mutation):
    doc = document(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        validate_decision(change_result(response(doc), mutation), doc)


@pytest.mark.parametrize("field,value", [("status", "incomplete"), ("model", "another-model"), ("reasoning", {"effort": "medium"})])
def test_wrong_transport_never_passes(tmp_path, field, value):
    doc = document(tmp_path)
    raw = response(doc); raw[field] = value
    with pytest.raises(ValueError):
        validate_decision(raw, doc)


def test_source_identity_and_silent_truncation_guard(tmp_path):
    doc = document(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        build_document({"document_id": "PMC456", "pmcid": "PMC456"}, [Path(doc["materials"][0]["path"])])
    doc["blocks"][0]["text"] = "x" * 700001
    with pytest.raises(ValueError, match="no text was truncated"):
        make_request(doc)


def test_resume_submits_once_and_never_rebuilds(tmp_path):
    doc = document(tmp_path)
    workspace, client = submitted(tmp_path, [doc])
    submit(workspace, client=client)
    assert client.creates == 1
    with pytest.raises(ValueError, match="already frozen"):
        freeze_documents([doc], workspace)


def test_unknown_submission_cannot_retry(tmp_path):
    workspace = tmp_path / "batch"
    freeze_documents([document(tmp_path)], workspace)
    client = FakeClient(unknown=True)
    with pytest.raises(TimeoutError):
        submit(workspace, client=client)
    with pytest.raises(RuntimeError, match="unknown"):
        submit(workspace, client=client)
    assert client.creates == 1


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -1, 0, 0.000001])
def test_budget_blocks_before_upload(tmp_path, budget):
    workspace = tmp_path / "batch"
    freeze_documents([document(tmp_path)], workspace)
    client = FakeClient()
    with pytest.raises(ValueError):
        submit(workspace, client=client, max_usd=budget)
    assert client.creates == 0


def test_collect_export_and_source_mutation_gate(tmp_path):
    doc = document(tmp_path)
    workspace, client = submitted(tmp_path, [doc])
    summary = collect(workspace, client=client)
    assert summary["retained"] == 1 and summary["technical_failures"] == 0
    assert summary["estimated_usd"] == pytest.approx(.000175)
    manifest = tmp_path / "eligible.jsonl"
    assert export_retained(workspace, manifest)["exported"] == 1
    assert load_batch_manifest(manifest)[0]["document_id"] == "PMC123"
    with Path(doc["materials"][0]["path"]).open("a") as stream:
        stream.write("\n<!--changed-->")
    with pytest.raises(ValueError, match="changed"):
        load_batch_manifest(manifest)


def test_failed_response_is_not_scientific_exclusion(tmp_path):
    doc = document(tmp_path)
    workspace, client = submitted(tmp_path, [doc])
    client.responses[0]["response"]["body"]["status"] = "incomplete"
    summary = collect(workspace, client=client)
    assert summary["retained"] == summary["excluded"] == 0
    assert summary["technical_failures"] == 1
    assert (workspace / "raw_responses").is_dir()
    with pytest.raises(ValueError):
        export_retained(workspace, tmp_path / "eligible.jsonl")


def test_abstract_only_retained_not_exported_as_fulltext(tmp_path):
    workspace, client = submitted(tmp_path, [document(tmp_path, fulltext=False)])
    summary = collect(workspace, client=client)
    assert summary["retained"] == 1
    report = export_retained(workspace, tmp_path / "eligible.jsonl")
    assert report["exported"] == 0 and report["retained_without_fulltext"] == ["PMC123"]


def test_excluded_response_is_not_exported(tmp_path):
    doc = document(tmp_path)
    workspace, client = submitted(tmp_path, [doc])
    client.responses[0]["response"]["body"] = response(doc, False)
    assert collect(workspace, client=client)["excluded"] == 1
    assert export_retained(workspace, tmp_path / "eligible.jsonl")["exported"] == 0


def test_existing_corpus_survives_legacy_initialization(tmp_path):
    directory = tmp_path / "pmc_xml"; directory.mkdir()
    document(directory)
    (tmp_path / "summary.json").write_text('{"original":true}')
    prepare_output_dir(tmp_path)
    assert (directory / "PMC123.xml").is_file()
    assert read_json(tmp_path / "summary.json") == {"original": True}


def test_frozen_reserve_and_body_cannot_be_changed(tmp_path):
    workspace = tmp_path / "batch"
    freeze_documents([document(tmp_path)], workspace)
    plan = read_json(workspace / "screening_plan.json")
    plan["reserved_usd"] = 0
    write_json(workspace / "screening_plan.json", plan)
    with pytest.raises(ValueError, match="budget"):
        submit(workspace, client=FakeClient())


def test_changed_response_cannot_enter_extraction(tmp_path):
    workspace, client = submitted(tmp_path, [document(tmp_path)])
    result = collect(workspace, client=client)
    record_path = Path(result["outcomes"][0]["record_path"])
    record = read_json(record_path)
    raw = read_json(Path(record["raw_response_path"]))
    raw["id"] = "changed"
    write_json(Path(record["raw_response_path"]), raw)
    with pytest.raises(ValueError, match="response changed"):
        validate_screening_gate(record_path, [Path(m["path"]) for m in record["document"]["materials"]])


def test_pubmed_metadata_contains_complete_abstract_and_types(tmp_path, monkeypatch):
    xml = '<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>' \
          '<ArticleTitle>A phase <i>Ib</i> report</ArticleTitle><Abstract><AbstractText Label="METHODS">Actual treatment.</AbstractText>' \
          '<AbstractText Label="RESULTS">No DLT.</AbstractText></Abstract>' \
          '<PublicationTypeList><PublicationType>Letter</PublicationType></PublicationTypeList></Article></MedlineCitation>' \
          '<PubmedData><ArticleIdList><ArticleId IdType="pmc">PMC42</ArticleId></ArticleIdList></PubmedData>' \
          '</PubmedArticle></PubmedArticleSet>'
    client = PublicMetadata(tmp_path)
    monkeypatch.setattr(client, "get", lambda *a, **k: xml)
    record = client.pubmed_records(["42"], {"source": "pubmed"})[0]
    assert record["title"] == "A phase Ib report"
    assert record["abstract"] == "Actual treatment. No DLT."
    assert record["publication_types"] == ["Letter"] and record["pmcid"] == "PMC42"


def test_downloaded_xml_hash_and_identity_verified(tmp_path, monkeypatch):
    from phase1_acquisition import visual_assets
    from phase1_acquisition.discovery import fetch_fulltext
    import hashlib
    payload = b'<article><front><article-meta><article-id pub-id-type="pmc">42</article-id></article-meta></front><body><p>Results</p></body></article>'
    md5 = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(visual_assets, "_cloud_article_metadata", lambda *a: ({"xml_url": f"s3://pmc-oa-opendata/PMC42.1/PMC42.1.xml?md5={md5}"}, "PMC42.1/", "metadata-url"))
    monkeypatch.setattr(visual_assets, "_download_limited", lambda *a, **k: payload)
    path = fetch_fulltext({"pmcid": "PMC42"}, tmp_path)
    assert path.is_file() and path.with_suffix(".provenance.json").is_file()
    monkeypatch.setattr(visual_assets, "_cloud_article_metadata", lambda *a: (_ for _ in []).throw(AssertionError("cache should be used")))
    assert fetch_fulltext({"pmcid": "PMC42"}, tmp_path) == path


def test_broad_query_includes_ib_and_no_completed_results_gate():
    from phase1_acquisition.discovery import PUBMED_QUERY
    assert '"phase Ib"' in PUBMED_QUERY and '"first-in-human"' in PUBMED_QUERY
    assert "hasResults" not in PUBMED_QUERY


def test_catalog_discovers_v3_and_v2_downloads(tmp_path, monkeypatch):
    from phase1_project import catalog
    for folder in ["v3/output/pmc_xml", "v2/output/pmc_xml_no_refs", "v3/output/pmc_pdf_assets"]:
        (tmp_path / folder).mkdir(parents=True)
    monkeypatch.setattr(catalog, "WORKSPACE_ROOT", tmp_path)
    assert tmp_path / "v3/output/pmc_xml" in catalog.default_xml_directories()
    assert tmp_path / "v2/output/pmc_xml_no_refs" in catalog.default_xml_directories()
    assert tmp_path / "v3/output/pmc_pdf_assets" in catalog.default_pdf_directories()


def test_pdf_hydration_preserves_decision_and_survives_recollect(tmp_path, monkeypatch):
    from phase1_acquisition.screening_batch import hydrate_pdfs
    from phase1_acquisition import pdf_assets
    from phase1_extraction.artifacts import sha256_file
    doc = document(tmp_path, fulltext=False)
    workspace, client = submitted(tmp_path, [doc])
    collect(workspace, client=client)
    pdf_root = tmp_path / "pdfs"
    pdf = pdf_root / "PMC123" / "PMC123.pdf"
    pdf.parent.mkdir(parents=True)
    import fitz
    with fitz.open() as pdf_document:
        page = pdf_document.new_page()
        page.insert_text((72, 72), "Twelve cancer patients received 10 mg of drug X. No dose limiting toxicity occurred.")
        pdf_document.save(pdf)
    monkeypatch.setattr(pdf_assets, "download_pmc_pdf_for_article", lambda *a, **k: {
        "pmcid": "PMC123", "status": "ready", "pdf_path": "PMC123.pdf", "pdf_sha256": sha256_file(pdf),
        "source_xml_sha256": doc["materials"][0]["sha256"]})
    assert hydrate_pdfs(workspace, pdf_root)["pdf_ready"] == 1
    collect(workspace, client=client)
    manifest = tmp_path / "eligible.jsonl"
    assert export_retained(workspace, manifest)["exported"] == 1
    assert len(load_batch_manifest(manifest)[0]["sources"]) == 2
    pdf.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_batch_manifest(manifest)


def test_failed_pilot_gate_blocks_paid_submission(tmp_path):
    workspace = tmp_path / "batch"
    freeze_documents([document(tmp_path)], workspace)
    write_json(workspace / "submission_gate.json", {"allow_submission": False})
    client = FakeClient()
    with pytest.raises(ValueError, match="pilot gate"):
        submit(workspace, client=client)
    assert client.creates == 0


def test_incomplete_materials_keep_decision_but_block_export(tmp_path, monkeypatch):
    from phase1_acquisition import screening_batch
    workspace, client = submitted(tmp_path, [document(tmp_path)])
    collect(workspace, client=client)
    monkeypatch.setattr(screening_batch, "material_preflight", lambda *args: {
        "ready": False, "not_covered_source_ids": ["missing-table"]})
    result = export_retained(workspace, tmp_path / "eligible.jsonl")
    assert result["exported"] == 0
    assert result["retained_with_incomplete_materials"][0]["document_id"] == "PMC123"
    assert read_json(workspace / "decisions/PMC123.json")["result"]["decision"] == "\u4fdd\u7559"


def reference_fixture(tmp_path):
    from phase1_acquisition.screening import SUPPORTIVE_CARE_POLICY_VERSION
    doc = document(tmp_path)
    doc["policy_version"] = SUPPORTIVE_CARE_POLICY_VERSION
    raw = response(doc)
    change_result(raw, lambda value: [e.pop("quote") for c in value["checks"] for e in c["evidence"]])
    return doc, raw


def test_reference_schema_attaches_original_evidence(tmp_path):
    doc, raw = reference_fixture(tmp_path)
    result = validate_decision(raw, doc)
    assert result.decision == "\u4fdd\u7559"
    assert all(e.quote == next(b["text"] for b in doc["blocks"] if b["source_id"] == e.source_id)
               for c in result.checks for e in c.evidence)
    schema = make_request(doc)["text"]["format"]["schema"]
    assert "quote" not in schema["$defs"]["SourceReference"]["properties"]


def test_reference_schema_rejects_invented_source(tmp_path):
    doc, raw = reference_fixture(tmp_path)
    change_result(raw, lambda value: value["checks"][0]["evidence"][0].update(source_id="made-up"))
    with pytest.raises(ValueError, match="Unknown evidence"):
        validate_decision(raw, doc)


def test_reference_allows_short_existing_fragment_without_changing_raw(tmp_path):
    doc, raw = reference_fixture(tmp_path)
    doc["blocks"].append({"source_id": "metadata:abstract:s5:0", "text": "10 mg)."})
    change_result(raw, lambda value: next(c for c in value["checks"] if c["criterion"] == "actual_dose_and_toxicity")["evidence"].append({"source_id": "metadata:abstract:s5:0"}))
    original = json.dumps(raw, sort_keys=True)
    result = validate_decision(raw, doc)
    assert any(e.quote == "10 mg)." for c in result.checks for e in c.evidence)
    assert json.dumps(raw, sort_keys=True) == original


def test_title_field_alias_is_audited_and_requires_exact_existing_target(tmp_path):
    from phase1_acquisition.screening import reference_binding_aliases
    doc, raw = reference_fixture(tmp_path)
    doc["blocks"].append({"source_id": "metadata:title:s0:0", "text": "A phase Ib study in cancer patients."})
    change_result(raw, lambda value: next(c for c in value["checks"] if c["criterion"] == "own_phase1_results")["evidence"].append({"source_id": "metadata:article_title:s0:0"}))
    original = json.dumps(raw, sort_keys=True)
    result = validate_decision(raw, doc)
    assert any(e.source_id == "metadata:title:s0:0" for c in result.checks for e in c.evidence)
    aliases = reference_binding_aliases(raw, doc)
    assert aliases[0]["returned_source_id"] == "metadata:article_title:s0:0"
    assert aliases[0]["resolved_source_id"] == "metadata:title:s0:0"
    assert json.dumps(raw, sort_keys=True) == original
    doc["blocks"] = [b for b in doc["blocks"] if b["source_id"] != "metadata:title:s0:0"]
    with pytest.raises(ValueError, match="Unknown evidence"):
        validate_decision(raw, doc)


def test_reference_batch_roundtrip(tmp_path):
    doc, raw = reference_fixture(tmp_path)
    workspace, client = submitted(tmp_path, [doc])
    client.responses[0]["response"]["body"] = raw
    assert collect(workspace, client=client)["retained"] == 1
    manifest = tmp_path / "ready.jsonl"
    assert export_retained(workspace, manifest)["exported"] == 1
    assert load_batch_manifest(manifest)[0]["document_id"] == doc["document_id"]


def test_evidence_fragments_preserve_every_nonwhitespace_character():
    from phase1_acquisition.screening import evidence_fragments
    import re
    original = "Results. Twelve patients received drug. " + "x" * 3500
    fragments = evidence_fragments([{"source_id": "body", "text": original}])
    assert len({b["source_id"] for b in fragments}) == len(fragments)
    assert re.sub(r"\s", "", "".join(b["text"] for b in fragments)) == re.sub(r"\s", "", original)


def test_offline_metadata_cache_does_not_reveal_api_key(tmp_path, monkeypatch):
    class Session:
        def get(self, *a, **k):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"records": []})
    monkeypatch.setattr("phase1_acquisition.discovery.time.sleep", lambda _: None)
    client = PublicMetadata(tmp_path, session=Session())
    client.get("https://example.org/metadata", {"query": "cancer", "api_key": "PRIVATE_SENTINEL"})
    cache = next(tmp_path.glob("*.json")).read_text()
    assert "PRIVATE_SENTINEL" not in cache


def test_duplicate_service_results_are_rejected(tmp_path):
    workspace, client = submitted(tmp_path, [document(tmp_path)])
    client.responses.append(client.responses[0])
    with pytest.raises(ValueError, match="Duplicate"):
        collect(workspace, client=client)
