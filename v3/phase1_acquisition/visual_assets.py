"""Download and index JATS-linked PMC visual assets.

The ordinary PMC EFetch response contains JATS XML, captions, and links to
figures/tables, but not the referenced image bytes.  This module resolves those
links into a durable per-article asset directory through the official public
PMC Cloud dataset on AWS.  The legacy individual OA packages and article
``/bin/`` scraping path were retired/prohibited for automated retrieval in
August 2026 and are deliberately not used here.

Only JATS ``<graphic>`` elements owned by ``<table-wrap>`` or ``<fig>`` are
collected.  Inline decorative images are intentionally ignored.  Original
bytes are retained and a model-compatible PNG/JPEG/WEBP/GIF copy is produced
with a content hash and dimensions so later multimodal extraction is fully
auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mimetypes
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, quote, urlparse

import requests
from PIL import Image, ImageOps


VISUAL_ASSET_MANIFEST_VERSION = "phase1_pmc_visual_assets_v2"
VISUAL_ASSET_POLICY_VERSION = "phase1_visual_asset_discovery_v1"
PMC_CLOUD_HTTPS_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
SUPPORTED_MODEL_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
DEFAULT_MAX_ASSET_BYTES = 40 * 1024 * 1024
DEFAULT_MAX_IMAGE_DIMENSION = 6000
MAX_SOURCE_IMAGE_PIXELS = 80_000_000


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean_space(value: str) -> str:
    return " ".join(value.split())


def _text(element: ET.Element | None) -> str:
    return _clean_space("".join(element.itertext())) if element is not None else ""


def _direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _pmcid_from_root(root: ET.Element, explicit: str | None = None) -> str:
    if explicit:
        candidate = explicit.strip().upper()
        if candidate.isdigit():
            candidate = f"PMC{candidate}"
        if candidate.startswith("PMC") and candidate[3:].isdigit():
            return candidate
        raise ValueError(f"Invalid PMCID: {explicit!r}")
    for element in root.iter():
        if (
            _local_name(element.tag) == "article-id"
            and str(element.attrib.get("pub-id-type") or "").casefold() == "pmc"
        ):
            return _pmcid_from_root(root, _text(element))
    raise ValueError("JATS XML does not contain a PMC article-id")


def _asset_id(pmcid: str, owner_id: str, href: str) -> str:
    digest = hashlib.sha256(f"{pmcid}\0{owner_id}\0{href}".encode("utf-8")).hexdigest()
    return f"vis_{digest[:20]}"


def _section_path(owner: ET.Element, parent_by_child: Mapping[ET.Element, ET.Element]) -> list[str]:
    values: list[str] = []
    current = parent_by_child.get(owner)
    while current is not None:
        if _local_name(current.tag) == "sec":
            title = _text(_direct_child(current, "title"))
            if title:
                values.append(title)
        current = parent_by_child.get(current)
    return list(reversed(values))


def discover_visual_asset_references(
    xml_path: Path,
    *,
    pmcid: str | None = None,
) -> dict[str, Any]:
    """Return deterministic figure/table image references from one JATS file."""

    xml_path = Path(xml_path)
    source_bytes = xml_path.read_bytes()
    root = ET.fromstring(source_bytes)
    filename_pmcid = xml_path.stem if re.fullmatch(r"PMC\d+", xml_path.stem.upper()) else None
    resolved_pmcid = _pmcid_from_root(root, pmcid or filename_pmcid)
    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    references: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for graphic in root.iter():
        if _local_name(graphic.tag) != "graphic":
            continue
        owner = parent_by_child.get(graphic)
        while owner is not None and _local_name(owner.tag) not in {"table-wrap", "fig"}:
            owner = parent_by_child.get(owner)
        if owner is None:
            continue
        href = str(graphic.attrib.get(XLINK_HREF) or graphic.attrib.get("href") or "").strip()
        if not href:
            continue
        owner_tag = _local_name(owner.tag)
        owner_id = str(owner.attrib.get("id") or f"{owner_tag}-{len(references) + 1}")
        key = (owner_id, href)
        if key in seen:
            continue
        seen.add(key)
        references.append(
            {
                "visual_asset_id": _asset_id(resolved_pmcid, owner_id, href),
                "resource_kind": "table_image" if owner_tag == "table-wrap" else "figure_image",
                "jats_owner_id": owner_id,
                "jats_href": href,
                "label": _text(_direct_child(owner, "label")) or None,
                "caption": _text(_direct_child(owner, "caption")) or None,
                "section_path": _section_path(owner, parent_by_child),
                "ordinal": len(references) + 1,
            }
        )
    return {
        "format": VISUAL_ASSET_POLICY_VERSION,
        "pmcid": resolved_pmcid,
        "source_xml_path": str(xml_path),
        "source_xml_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "reference_count": len(references),
        "references": references,
    }


def _cloud_version_number(prefix: str, pmcid: str) -> int:
    match = re.fullmatch(rf"{re.escape(pmcid)}\.(\d+)/", prefix)
    return int(match.group(1)) if match else -1


def _cloud_article_metadata(
    session: requests.Session,
    pmcid: str,
    timeout: int,
) -> tuple[dict[str, Any], str, str]:
    """Resolve the newest reusable PMC article version in the public AWS dataset."""

    listing = session.get(
        PMC_CLOUD_HTTPS_BASE + "/",
        params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"},
        timeout=timeout,
    )
    listing.raise_for_status()
    root = ET.fromstring(listing.content)
    prefixes = sorted(
        {
            _text(element)
            for element in root.iter()
            if _local_name(element.tag) == "Prefix"
            and _cloud_version_number(_text(element), pmcid) >= 0
        },
        key=lambda value: _cloud_version_number(value, pmcid),
        reverse=True,
    )
    if not prefixes:
        raise FileNotFoundError(
            f"{pmcid} is not present in the reusable PMC Cloud article dataset"
        )
    errors: list[str] = []
    for prefix in prefixes:
        version_id = prefix.rstrip("/")
        metadata_url = (
            f"{PMC_CLOUD_HTTPS_BASE}/metadata/{quote(version_id, safe='')}.json"
        )
        try:
            response = session.get(metadata_url, timeout=timeout)
            response.raise_for_status()
            metadata = json.loads(response.content.decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("PMC Cloud metadata is not a JSON object")
            if str(metadata.get("pmcid") or "").upper() != pmcid:
                raise ValueError("PMC Cloud metadata PMCID does not match the request")
            return metadata, prefix, metadata_url
        except Exception as error:
            errors.append(f"{version_id}: {type(error).__name__}: {error}")
    raise RuntimeError("; ".join(errors))


def _cloud_object_record(value: str) -> dict[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or parsed.netloc != "pmc-oa-opendata":
        raise ValueError(f"Unsupported PMC Cloud object URL: {value!r}")
    key = parsed.path.lstrip("/")
    if not key or "/" not in key:
        raise ValueError(f"Invalid PMC Cloud object key: {value!r}")
    expected_md5 = str((parse_qs(parsed.query).get("md5") or [""])[0]).casefold()
    if expected_md5 and not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
        raise ValueError(f"Invalid PMC Cloud object MD5: {value!r}")
    return {
        "basename": Path(key).name,
        "source_url": f"{PMC_CLOUD_HTTPS_BASE}/{quote(key, safe='/')}",
        "expected_md5": expected_md5,
        "object_key": key,
    }


def _cloud_media_by_wanted_basename(
    metadata: Mapping[str, Any],
    wanted_basenames: set[str],
) -> dict[str, dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for value in metadata.get("media_urls") or []:
        if isinstance(value, str):
            candidates.append(_cloud_object_record(value))
    matches: dict[str, dict[str, str]] = {}
    for wanted in wanted_basenames:
        exact = [item for item in candidates if item["basename"] == wanted]
        compatible = exact or [
            item for item in candidates if Path(item["basename"]).stem == Path(wanted).stem
        ]
        if len(compatible) == 1:
            matches[wanted] = compatible[0]
    return matches


def _download_limited(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    max_bytes: int,
) -> bytes:
    response = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
    response.raise_for_status()
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > max_bytes:
        raise ValueError(f"Remote resource exceeds byte limit: {content_length} > {max_bytes}")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Downloaded resource exceeds byte limit: {total} > {max_bytes}")
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_filename(asset_id: str, href: str) -> str:
    suffix = Path(urlparse(href).path).suffix.casefold()
    if not re_safe_suffix(suffix):
        suffix = ".bin"
    return f"{asset_id}{suffix}"


def re_safe_suffix(value: str) -> bool:
    return bool(value) and len(value) <= 10 and value[1:].isalnum()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prepare_model_image(
    original_path: Path,
    model_dir: Path,
    *,
    max_dimension: int,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    rendered_from_svg = False
    try:
        opened_context = Image.open(original_path)
    except (OSError, ValueError):
        # PMC Cloud media objects sometimes provide SVG graphics. Responses accepts
        # raster images, so render the first SVG page locally while preserving
        # the untouched SVG original for audit.
        if original_path.suffix.casefold() != ".svg":
            raise
        import fitz

        document = fitz.open(stream=original_path.read_bytes(), filetype="svg")
        if len(document) != 1:
            document.close()
            raise ValueError("SVG visual asset did not render as one page")
        pixmap = document[0].get_pixmap(alpha=True, dpi=180)
        rendered = pixmap.tobytes("png")
        document.close()
        opened_context = Image.open(io.BytesIO(rendered))
        rendered_from_svg = True
    with opened_context as opened:
        source_width, source_height = opened.size
        if source_width * source_height > MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError(
                "Image exceeds the source pixel safety limit: "
                f"{source_width}x{source_height}"
            )
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("Image has invalid dimensions")
        resized = False
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            target = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(target, Image.Resampling.LANCZOS)
            width, height = image.size
            resized = True
        source_format = str(opened.format or "").upper()
        animated = bool(getattr(opened, "is_animated", False))
        if (
            source_format in SUPPORTED_MODEL_FORMATS
            and not resized
            and not animated
            and not rendered_from_svg
        ):
            suffix = ".jpg" if source_format == "JPEG" else f".{source_format.casefold()}"
            model_path = model_dir / f"{original_path.stem}{suffix}"
            _write_bytes_atomic(model_path, original_path.read_bytes())
            mime_type = Image.MIME.get(source_format) or mimetypes.guess_type(model_path.name)[0]
        else:
            model_path = model_dir / f"{original_path.stem}.png"
            if image.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
                image = image.convert("RGB")
            temporary = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
            image.save(temporary, format="PNG", optimize=True)
            os.replace(temporary, model_path)
            mime_type = "image/png"
    model_bytes = model_path.read_bytes()
    return {
        "model_path": model_path,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "model_bytes": len(model_bytes),
        "mime_type": mime_type,
        "width": width,
        "height": height,
        "resized_for_model": resized,
    }


def download_visual_assets_for_article(
    xml_path: Path,
    output_root: Path,
    *,
    pmcid: str | None = None,
    session: requests.Session | None = None,
    timeout: int = 60,
    max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
) -> dict[str, Any]:
    """Download all JATS-linked table/figure images for one article."""

    discovery = discover_visual_asset_references(xml_path, pmcid=pmcid)
    resolved_pmcid = discovery["pmcid"]
    article_root = Path(output_root) / resolved_pmcid
    original_dir = article_root / "original"
    model_dir = article_root / "model"
    http = session or requests.Session()
    references = [dict(item) for item in discovery["references"]]
    basenames = {
        Path(urlparse(str(item["jats_href"])).path).name
        for item in references
        if Path(urlparse(str(item["jats_href"])).path).name
    }
    cloud_metadata: dict[str, Any] | None = None
    cloud_article_prefix: str | None = None
    cloud_metadata_url: str | None = None
    cloud_metadata_error: str | None = None
    cloud_media: dict[str, dict[str, str]] = {}
    if basenames:
        try:
            cloud_metadata, cloud_article_prefix, cloud_metadata_url = (
                _cloud_article_metadata(http, resolved_pmcid, timeout)
            )
            cloud_media = _cloud_media_by_wanted_basename(cloud_metadata, basenames)
        except Exception as error:  # unavailable/non-reusable article is recorded
            cloud_metadata_error = f"{type(error).__name__}: {error}"

    assets: list[dict[str, Any]] = []
    for reference in references:
        asset_id = str(reference["visual_asset_id"])
        href = str(reference["jats_href"])
        basename = Path(urlparse(href).path).name
        media = cloud_media.get(basename)
        source_url = media["source_url"] if media else None
        method = "pmc_cloud_aws"
        payload: bytes | None = None
        error_text: str | None = None
        if media is None:
            error_text = cloud_metadata_error or (
                "PMC Cloud metadata has no unique media object matching " + basename
            )
        else:
            try:
                payload = _download_limited(
                    http,
                    str(source_url),
                    timeout=timeout,
                    max_bytes=max_asset_bytes,
                )
                expected_md5 = media.get("expected_md5") or ""
                actual_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
                if expected_md5 and actual_md5 != expected_md5:
                    raise ValueError(
                        f"PMC Cloud media MD5 mismatch: {actual_md5} != {expected_md5}"
                    )
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
        record = {
            **reference,
            "status": "failed",
            "download_method": method,
            "source_url": source_url,
            "cloud_object_key": media.get("object_key") if media else None,
            "cloud_expected_md5": media.get("expected_md5") if media else None,
            "original_path": None,
            "model_path": None,
            "original_sha256": None,
            "model_sha256": None,
            "mime_type": None,
            "width": None,
            "height": None,
            "resized_for_model": False,
            "error": error_text,
        }
        if payload is not None:
            original_path = original_dir / _safe_filename(asset_id, href)
            try:
                _write_bytes_atomic(original_path, payload)
                prepared = _prepare_model_image(
                    original_path,
                    model_dir,
                    max_dimension=max_image_dimension,
                )
                record.update(
                    {
                        "status": "ready",
                        "original_path": str(original_path.relative_to(article_root)),
                        "model_path": str(prepared["model_path"].relative_to(article_root)),
                        "original_sha256": hashlib.sha256(payload).hexdigest(),
                        "original_bytes": len(payload),
                        "model_sha256": prepared["model_sha256"],
                        "model_bytes": prepared["model_bytes"],
                        "mime_type": prepared["mime_type"],
                        "width": prepared["width"],
                        "height": prepared["height"],
                        "resized_for_model": prepared["resized_for_model"],
                        "error": None,
                    }
                )
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
        assets.append(record)

    manifest = {
        "format": VISUAL_ASSET_MANIFEST_VERSION,
        "pmcid": resolved_pmcid,
        "source_xml_sha256": discovery["source_xml_sha256"],
        "source_xml_path": str(Path(xml_path)),
        "cloud_article_prefix": cloud_article_prefix,
        "cloud_metadata_url": cloud_metadata_url,
        "cloud_metadata_error": cloud_metadata_error,
        "reference_count": len(references),
        "ready_count": sum(item["status"] == "ready" for item in assets),
        "failed_count": sum(item["status"] != "ready" for item in assets),
        "assets": assets,
    }
    _write_json_atomic(article_root / "manifest.json", manifest)
    return manifest


def download_visual_assets_for_directory(
    xml_dir: Path,
    output_root: Path,
    *,
    pmcids: Iterable[str] | None = None,
    timeout: int = 60,
    max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
) -> dict[str, Any]:
    session = requests.Session()
    outcomes: list[dict[str, Any]] = []
    allowed = (
        {str(item).strip().upper() for item in pmcids}
        if pmcids is not None
        else None
    )
    xml_paths = sorted(Path(xml_dir).glob("PMC*.xml"), key=lambda item: item.name)
    if allowed is not None:
        xml_paths = [path for path in xml_paths if path.stem.upper() in allowed]
    for xml_path in xml_paths:
        try:
            manifest = download_visual_assets_for_article(
                xml_path,
                output_root,
                session=session,
                timeout=timeout,
                max_asset_bytes=max_asset_bytes,
                max_image_dimension=max_image_dimension,
            )
            outcomes.append(
                {
                    "pmcid": manifest["pmcid"],
                    "status": "completed",
                    "ready_count": manifest["ready_count"],
                    "failed_count": manifest["failed_count"],
                }
            )
        except Exception as error:
            outcomes.append(
                {
                    "pmcid": xml_path.stem,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    summary = {
        "format": VISUAL_ASSET_MANIFEST_VERSION,
        "article_count": len(outcomes),
        "completed_count": sum(item["status"] == "completed" for item in outcomes),
        "failed_count": sum(item["status"] == "failed" for item in outcomes),
        "ready_asset_count": sum(int(item.get("ready_count") or 0) for item in outcomes),
        "failed_asset_count": sum(int(item.get("failed_count") or 0) for item in outcomes),
        "articles_with_asset_failures": sum(
            int(item.get("failed_count") or 0) > 0 for item in outcomes
        ),
        "outcomes": outcomes,
    }
    _write_json_atomic(Path(output_root) / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download JATS-linked PMC table and figure images."
    )
    parser.add_argument("--xml-dir", type=Path, default=Path("output/pmc_xml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pmc_visual_assets"),
    )
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--max-asset-bytes", type=int, default=DEFAULT_MAX_ASSET_BYTES)
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=DEFAULT_MAX_IMAGE_DIMENSION,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.xml_dir.is_dir():
        raise SystemExit(f"XML directory not found: {args.xml_dir}")
    summary = download_visual_assets_for_directory(
        args.xml_dir,
        args.output_dir,
        timeout=args.request_timeout,
        max_asset_bytes=args.max_asset_bytes,
        max_image_dimension=args.max_image_dimension,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
