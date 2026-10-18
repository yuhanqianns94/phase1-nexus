"""Offline regression checks for bounded, resumable literature discovery."""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from phase1_acquisition import discovery
from phase1_acquisition.discovery import PublicMetadata
from phase1_extraction.artifacts import read_json


def cloud_xml(pmcid="PMC42", body="Observed dose-escalation results."):
    return (f'<article><front><article-meta><article-id pub-id-type="pmc">{pmcid[3:]}</article-id>'
            f'</article-meta></front><body><p>{body}</p></body></article>').encode()


def mock_cloud(monkeypatch, payload):
    import hashlib
    from phase1_acquisition import visual_assets
    digest = hashlib.md5(payload).hexdigest()
    url = f"s3://pmc-oa-opendata/PMC42.1/PMC42.1.xml?md5={digest}"
    monkeypatch.setattr(visual_assets, "_cloud_article_metadata", lambda *a: ({"xml_url": url}, "PMC42.1/", "metadata-url"))
    monkeypatch.setattr(visual_assets, "_download_limited", lambda *a, **k: payload)


def no_cloud(monkeypatch):
    from phase1_acquisition import visual_assets
    monkeypatch.setattr(visual_assets, "_cloud_article_metadata",
                        lambda *a: (_ for _ in ()).throw(AssertionError("Must not download or repair cached material")))


def test_legacy_stub_is_preserved_and_versioned_fulltext_is_reused(tmp_path, monkeypatch):
    import hashlib
    original = tmp_path / "PMC42.xml"
    stub = b'<article><front><article-meta><article-id pub-id-type="pmc">42</article-id></article-meta></front></article>'
    original.write_bytes(stub)
    payload = cloud_xml()
    mock_cloud(monkeypatch, payload)
    recovered = discovery.fetch_fulltext({"pmcid": "PMC42"}, tmp_path)
    assert recovered == tmp_path / "versions" / "PMC42" / (hashlib.sha256(payload).hexdigest() + ".xml")
    assert recovered.read_bytes() == payload
    assert original.read_bytes() == stub
    assert not original.with_suffix(".provenance.json").exists()
    receipt = recovered.with_suffix(".provenance.json")
    original_proof = receipt.read_bytes()
    no_cloud(monkeypatch)
    assert discovery.fetch_fulltext({"pmcid": "PMC42"}, tmp_path) == recovered
    assert original.read_bytes() == stub and receipt.read_bytes() == original_proof


@pytest.mark.parametrize("missing", [False, True])
def test_changed_or_missing_verified_xml_stops_without_download(tmp_path, monkeypatch, missing):
    payload = cloud_xml()
    mock_cloud(monkeypatch, payload)
    path = discovery.fetch_fulltext({"pmcid": "PMC42"}, tmp_path)
    assert path == tmp_path / "PMC42.xml"
    proof = path.with_suffix(".provenance.json").read_bytes()
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"changed historical source")
    no_cloud(monkeypatch)
    with pytest.raises(ValueError, match="cache integrity mismatch"):
        discovery.fetch_fulltext({"pmcid": "PMC42"}, tmp_path)
    assert path.with_suffix(".provenance.json").read_bytes() == proof
    assert not path.exists() if missing else path.read_bytes() == b"changed historical source"


def test_changed_verified_version_is_not_silently_repaired(tmp_path, monkeypatch):
    original = tmp_path / "PMC42.xml"
    original.write_bytes(b"legacy source")
    mock_cloud(monkeypatch, cloud_xml())
    recovered = discovery.fetch_fulltext({"pmcid": "PMC42"}, tmp_path)
    recovered.write_bytes(b"changed versioned source")
    no_cloud(monkeypatch)
    with pytest.raises(ValueError, match="cache integrity mismatch"):
        discovery.fetch_fulltext({"pmcid": "PMC42"}, tmp_path)
    assert original.read_bytes() == b"legacy source"
    assert recovered.read_bytes() == b"changed versioned source"


def test_immutable_publication_never_replaces_a_racing_writer(tmp_path, monkeypatch):
    target = tmp_path / "source.xml"
    real_link = discovery.os.link

    def other_writer_wins(source, destination):
        target.write_bytes(b"other process historical material")
        return real_link(source, destination)

    monkeypatch.setattr(discovery.os, "link", other_writer_wins)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        discovery._immutable_bytes(target, b"new material")
    assert target.read_bytes() == b"other process historical material"
    assert list(tmp_path.glob("*.tmp")) == []


def test_immutable_publication_can_reuse_identical_bytes(tmp_path):
    target = tmp_path / "source.xml"
    target.write_bytes(b"identical source")
    old_inode = target.stat().st_ino
    discovery._immutable_bytes(target, b"identical source")
    assert target.read_bytes() == b"identical source" and target.stat().st_ino == old_inode
