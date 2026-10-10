#!/usr/bin/env python3
"""
Literature acquisition pipeline:
1) Query ClinicalTrials.gov studies:
   - Phase 1 (default website checkbox semantics, includes Phase 1/2)
   - Completed
   - Has Results
   - Condition/Disease = cancer (default)
2) Keep studies with linked PubMed PMIDs
3) Keep PMIDs that are free PMC articles (has PMCID)
4) Keep only free PMC articles whose title contains "Phase I" or "Phase 1"
5) Download those PMC XML full-text files and save summary outputs
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

from .pdf_assets import (
    DEFAULT_MAX_PDF_BYTES,
    DEFAULT_WORKERS as DEFAULT_PDF_WORKERS,
    OPENALEX_API_KEY_ENV,
    download_pmc_pdfs_for_directory,
)
from .visual_assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_IMAGE_DIMENSION,
    download_visual_assets_for_directory,
)

CTGOV_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI recommends <= 3 req/s without API key
NCBI_SLEEP_SECONDS = 0.34


@dataclass
class LinkedStudy:
    nct_id: str
    phases: List[str] = field(default_factory=list)
    overall_status: str = ""
    has_results: bool = False
    linked_pmids: List[str] = field(default_factory=list)

    def to_row(self) -> Dict[str, str]:
        return {
            "nct_id": self.nct_id,
            "clinicaltrials_url": f"https://clinicaltrials.gov/study/{self.nct_id}",
            "phases": "; ".join(self.phases),
            "overall_status": self.overall_status,
            "has_results": str(self.has_results),
            "linked_pmids": "; ".join(self.linked_pmids),
            "n_linked_pmids": str(len(self.linked_pmids)),
        }


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download only free PMC articles whose PubMed title contains "
            "'Phase I' or 'Phase 1' from CTGov-filtered studies."
        )
    )
    p.add_argument(
        "--phase-mode",
        choices=["phase1_only", "phase1_or_phase1_2"],
        default="phase1_or_phase1_2",
        help=(
            "Phase filter mode. Default keeps website checkbox semantics "
            "(records containing PHASE1)."
        ),
    )
    p.add_argument(
        "--condition",
        default="cancer",
        help="Condition/Disease filter mapped to query.cond (default: cancer).",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        help="Single output directory for all run artifacts.",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="ClinicalTrials.gov studies page size.",
    )
    p.add_argument(
        "--max-studies",
        type=int,
        default=0,
        help="Optional cap for fetched CTGov studies (0 means no cap).",
    )
    p.add_argument(
        "--ncbi-email",
        default="yuhan.qian@yale.edu",
        help="Email sent to NCBI E-utilities.",
    )
    p.add_argument(
        "--ncbi-api-key",
        default="",
        help="Optional NCBI API key.",
    )
    p.add_argument(
        "--request-timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PMC XML download and keep stats/CSV only.",
    )
    p.add_argument(
        "--skip-visual-assets",
        action="store_true",
        help="Download JATS XML but skip linked table/figure image assets.",
    )
    p.add_argument(
        "--visual-assets-dir",
        default=None,
        help=(
            "Directory for per-PMC visual assets. Defaults to "
            "<output-dir>/pmc_visual_assets."
        ),
    )
    p.add_argument(
        "--max-visual-asset-bytes",
        type=int,
        default=DEFAULT_MAX_ASSET_BYTES,
        help="Maximum bytes accepted for one linked image.",
    )
    p.add_argument(
        "--max-visual-image-dimension",
        type=int,
        default=DEFAULT_MAX_IMAGE_DIMENSION,
        help="Maximum width/height of the model-compatible image copy.",
    )
    p.add_argument(
        "--download-pdfs",
        action="store_true",
        help=(
            "Deprecated compatibility flag. PDFs are downloaded by default "
            "from PMC Cloud. Use --openalex-fallback explicitly for the "
            "optional official OpenAlex cached-content fallback."
        ),
    )
    p.add_argument(
        "--skip-pdfs",
        action="store_true",
        help="Download JATS XML but skip PDF acquisition.",
    )
    p.add_argument(
        "--pdf-assets-dir",
        default=None,
        help="PDF output directory. Defaults to <output-dir>/pmc_pdf_assets.",
    )
    p.add_argument(
        "--pdf-workers",
        type=int,
        default=DEFAULT_PDF_WORKERS,
        help=(
            "Concurrent PDF acquisition workers "
            f"(default: {DEFAULT_PDF_WORKERS})."
        ),
    )
    p.add_argument(
        "--max-pdf-bytes",
        type=int,
        default=DEFAULT_MAX_PDF_BYTES,
        help="Maximum bytes accepted for one PMC PDF.",
    )
    p.add_argument(
        "--openalex-fallback",
        action="store_true",
        help=(
            "Explicitly allow the official OpenAlex cached-content fallback "
            f"for PMC-unavailable PDFs. Requires ${OPENALEX_API_KEY_ENV}; "
            "omitting this flag prevents accidental paid API calls."
        ),
    )
    return p.parse_args()


def get_nested(data: Dict, path: List[str], default=None):
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def build_phase_clause(phase_mode: str) -> str:
    if phase_mode == "phase1_only":
        return (
            "AREA[protocolSection.designModule.phases]PHASE1 "
            "NOT AREA[protocolSection.designModule.phases]PHASE2"
        )
    return "AREA[protocolSection.designModule.phases]PHASE1"


def phase_matches_mode(phases: List[str], phase_mode: str) -> bool:
    phase_set = set(phases or [])
    if phase_mode == "phase1_only":
        return phase_set == {"PHASE1"}
    return "PHASE1" in phase_set


def fetch_candidate_studies(
    page_size: int,
    phase_mode: str,
    condition: str,
    timeout: int,
    max_studies: int = 0,
) -> List[Dict]:
    expr = (
        f"{build_phase_clause(phase_mode)} "
        "AND AREA[protocolSection.statusModule.overallStatus]COMPLETED "
        "AND AREA[hasResults]true"
    )

    studies: List[Dict] = []
    next_token: Optional[str] = None
    total_count: Optional[int] = None

    while True:
        params = {
            "query.term": expr,
            "pageSize": page_size,
            "countTotal": "true",
            "format": "json",
        }
        if condition.strip():
            params["query.cond"] = condition.strip()
        if next_token:
            params["pageToken"] = next_token

        resp = requests.get(CTGOV_STUDIES_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        if total_count is None:
            total_count = payload.get("totalCount")
            print(f"[CTGov] totalCount(query) = {total_count}")

        batch = payload.get("studies", [])
        if not batch:
            break

        studies.extend(batch)
        print(f"[CTGov] fetched {len(studies)} studies ...")

        if max_studies > 0 and len(studies) >= max_studies:
            studies = studies[:max_studies]
            break

        next_token = payload.get("nextPageToken")
        if not next_token:
            break

    return studies


def extract_linked_studies(studies: List[Dict], phase_mode: str) -> List[LinkedStudy]:
    out: List[LinkedStudy] = []
    for s in studies:
        nct_id = get_nested(s, ["protocolSection", "identificationModule", "nctId"], "") or ""
        phases = get_nested(s, ["protocolSection", "designModule", "phases"], []) or []
        overall_status = get_nested(s, ["protocolSection", "statusModule", "overallStatus"], "") or ""
        has_results = bool(s.get("hasResults", False))

        if overall_status != "COMPLETED":
            continue
        if not has_results:
            continue
        if not phase_matches_mode(phases, phase_mode):
            continue

        refs = get_nested(s, ["protocolSection", "referencesModule", "references"], []) or []
        pmids: List[str] = []
        for ref in refs:
            pmid_raw = str(ref.get("pmid") or "").strip()
            if pmid_raw.isdigit():
                pmids.append(pmid_raw)

        pmids = sorted(set(pmids), key=int)
        out.append(
            LinkedStudy(
                nct_id=nct_id,
                phases=list(phases),
                overall_status=overall_status,
                has_results=has_results,
                linked_pmids=pmids,
            )
        )
    return out


def ncbi_base_params(email: str, api_key: str) -> Dict[str, str]:
    params = {"tool": "ctgov_pmc_phase1_title_only", "email": email}
    if api_key:
        params["api_key"] = api_key
    return params


def extract_normalized_pmcid(articleids: List[Dict]) -> str:
    for obj in articleids or []:
        raw = str(obj.get("value") or "")
        m = re.search(r"(PMC\d+)", raw, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def title_has_phase1_marker(title: str) -> bool:
    return bool(re.search(r"\bphase\s*(?:i|1)\b", str(title or ""), re.IGNORECASE))


def collect_pubmed_metadata(
    pmids: List[str],
    email: str,
    api_key: str,
    timeout: int,
    batch_size: int = 200,
) -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
    valid_pmids: Set[str] = set()
    pmid_to_pmcid: Dict[str, str] = {}
    pmid_to_title: Dict[str, str] = {}
    base = ncbi_base_params(email, api_key)

    for chunk in chunked(pmids, batch_size):
        params = {
            **base,
            "db": "pubmed",
            "id": ",".join(chunk),
            "retmode": "json",
        }
        resp = requests.get(NCBI_ESUMMARY_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json().get("result", {})

        uids = [str(u) for u in payload.get("uids", []) if str(u).isdigit()]
        for uid in uids:
            valid_pmids.add(uid)
            rec = payload.get(uid, {})
            title = str(rec.get("title") or "").strip()
            if title:
                pmid_to_title[uid] = title
            pmcid = extract_normalized_pmcid(rec.get("articleids", []))
            if pmcid:
                pmid_to_pmcid[uid] = pmcid

        time.sleep(NCBI_SLEEP_SECONDS)

    return valid_pmids, pmid_to_pmcid, pmid_to_title


def download_pmc_xml_articles(
    pmid_to_pmcid: Dict[str, str],
    target_dir: Path,
    email: str,
    api_key: str,
    timeout: int,
) -> Tuple[int, List[str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed: List[str] = []
    base = ncbi_base_params(email, api_key)
    unique_pmcids = sorted(set(pmid_to_pmcid.values()))

    for idx, pmcid in enumerate(unique_pmcids, start=1):
        out_path = target_dir / f"{pmcid}.xml"
        if out_path.exists():
            downloaded += 1
            continue

        params = {**base, "db": "pmc", "id": pmcid, "retmode": "xml"}
        try:
            resp = requests.get(NCBI_EFETCH_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            if "<error>" in text.lower() or "<article" not in text.lower():
                failed.append(pmcid)
            else:
                out_path.write_text(text, encoding="utf-8")
                downloaded += 1
        except Exception:
            failed.append(pmcid)

        if idx % 100 == 0 or idx == len(unique_pmcids):
            print(f"[PMC] downloaded {downloaded}/{len(unique_pmcids)} XML files ...")
        time.sleep(NCBI_SLEEP_SECONDS)

    return downloaded, failed


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = [
        "summary.json",
        "studies_target_phase_status_results.csv",
        "studies_with_linked_pubmed.csv",
        "pmids_linked_valid.csv",
        "pmids_with_pmcid.csv",
        "pmids_phase1_title_with_pmcid.csv",
        "studies_with_phase1_title_free_pmc_pub.csv",
        "pmc_download_failures.csv",
    ]
    for name in generated_files:
        p = out_dir / name
        if p.exists():
            p.unlink()
    pmc_dir = out_dir / "pmc_xml"
    if pmc_dir.exists():
        shutil.rmtree(pmc_dir)


def main() -> None:
    args = parse_args()
    if args.pdf_workers < 1:
        raise SystemExit("--pdf-workers must be at least 1")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be positive")
    if args.max_pdf_bytes <= 0:
        raise SystemExit("--max-pdf-bytes must be positive")
    if args.download_pdfs and args.skip_pdfs:
        raise SystemExit("--download-pdfs and --skip-pdfs cannot be combined")
    if args.openalex_fallback and not str(
        os.environ.get(OPENALEX_API_KEY_ENV) or ""
    ).strip():
        raise SystemExit(
            f"--openalex-fallback requires ${OPENALEX_API_KEY_ENV}"
        )
    out_dir = Path(args.output_dir)
    prepare_output_dir(out_dir)

    raw_studies = fetch_candidate_studies(
        page_size=args.page_size,
        phase_mode=args.phase_mode,
        condition=args.condition,
        timeout=args.request_timeout,
        max_studies=args.max_studies,
    )

    parsed_studies = extract_linked_studies(raw_studies, args.phase_mode)
    studies_with_pubmed = [s for s in parsed_studies if s.linked_pmids]
    all_linked_pmids = sorted({p for s in studies_with_pubmed for p in s.linked_pmids}, key=int)

    print(f"[Filter] condition = {args.condition.strip() or '(none)'}")
    print(f"[Filter] phase_mode = {args.phase_mode}")
    print(f"[Filter] total parsed target studies: {len(parsed_studies)}")
    print(f"[Filter] studies with linked PubMed PMIDs: {len(studies_with_pubmed)}")
    print(f"[Filter] unique linked PMIDs: {len(all_linked_pmids)}")

    valid_pmids, pmid_to_pmcid, pmid_to_title = collect_pubmed_metadata(
        all_linked_pmids,
        args.ncbi_email,
        args.ncbi_api_key,
        timeout=args.request_timeout,
    )
    valid_pmids_sorted = sorted(valid_pmids, key=int)

    pmc_pmids = sorted([p for p in valid_pmids_sorted if p in pmid_to_pmcid], key=int)
    phase1_title_pmc_pmids = sorted(
        [p for p in pmc_pmids if title_has_phase1_marker(pmid_to_title.get(p, ""))],
        key=int,
    )

    phase1_title_pmid_to_pmcid = {p: pmid_to_pmcid[p] for p in phase1_title_pmc_pmids}
    unique_pmcids_filtered = sorted(set(phase1_title_pmid_to_pmcid.values()))
    studies_with_phase1_title_pmc = [
        s for s in studies_with_pubmed if any(p in phase1_title_pmid_to_pmcid for p in s.linked_pmids)
    ]

    print(f"[PubMed] valid linked PMIDs: {len(valid_pmids_sorted)}")
    print(f"[PubMed] PMIDs with PMCID (free PMC): {len(pmc_pmids)}")
    print(f"[PubMed] free PMC articles with Phase I/1 in title: {len(phase1_title_pmc_pmids)}")
    print(f"[PubMed] unique filtered PMCIDs: {len(unique_pmcids_filtered)}")

    downloaded_pmc_xml = 0
    failed_pmcids: List[str] = []
    visual_asset_summary: Dict[str, object] = {
        "article_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "ready_asset_count": 0,
    }
    pdf_asset_summary: Dict[str, object] = {
        "article_count": 0,
        "ready_count": 0,
        "unavailable_count": 0,
        "failed_count": 0,
    }
    if not args.skip_download and unique_pmcids_filtered:
        pmc_dir = out_dir / "pmc_xml"
        downloaded_pmc_xml, failed_pmcids = download_pmc_xml_articles(
            phase1_title_pmid_to_pmcid,
            target_dir=pmc_dir,
            email=args.ncbi_email,
            api_key=args.ncbi_api_key,
            timeout=args.request_timeout,
        )
        print(f"[PMC] downloaded XML files: {downloaded_pmc_xml}")
        if failed_pmcids:
            print(f"[PMC] failed downloads: {len(set(failed_pmcids))}")
        if not args.skip_visual_assets:
            visual_assets_dir = (
                Path(args.visual_assets_dir)
                if args.visual_assets_dir
                else out_dir / "pmc_visual_assets"
            )
            visual_asset_summary = download_visual_assets_for_directory(
                pmc_dir,
                visual_assets_dir,
                pmcids=unique_pmcids_filtered,
                timeout=args.request_timeout,
                max_asset_bytes=args.max_visual_asset_bytes,
                max_image_dimension=args.max_visual_image_dimension,
            )
            print(
                "[PMC visuals] ready assets: "
                f"{visual_asset_summary.get('ready_asset_count', 0)}"
            )
            print(f"[PMC visuals] directory: {visual_assets_dir}")
        else:
            print("[PMC visuals] skipped (--skip-visual-assets).")
        if not args.skip_pdfs:
            pdf_assets_dir = (
                Path(args.pdf_assets_dir)
                if args.pdf_assets_dir
                else out_dir / "pmc_pdf_assets"
            )
            pdf_asset_summary = download_pmc_pdfs_for_directory(
                pmc_dir,
                pdf_assets_dir,
                pmcids=unique_pmcids_filtered,
                timeout=args.request_timeout,
                max_pdf_bytes=args.max_pdf_bytes,
                workers=args.pdf_workers,
                openalex_api_key=(
                    (
                        str(os.environ.get(OPENALEX_API_KEY_ENV) or "").strip()
                        or None
                    )
                    if args.openalex_fallback
                    else None
                ),
            )
            print(
                "[PMC PDFs] ready/unavailable/failed: "
                f"{pdf_asset_summary.get('ready_count', 0)}/"
                f"{pdf_asset_summary.get('unavailable_count', 0)}/"
                f"{pdf_asset_summary.get('failed_count', 0)}"
            )
            print(f"[PMC PDFs] directory: {pdf_assets_dir}")
        else:
            print("[PMC PDFs] skipped (--skip-pdfs).")
    elif args.skip_download:
        print("[PMC] download skipped (--skip-download).")
    else:
        print("[PMC] no filtered PMCID records to download.")

    write_csv(out_dir / "studies_target_phase_status_results.csv", [s.to_row() for s in parsed_studies])
    write_csv(out_dir / "studies_with_linked_pubmed.csv", [s.to_row() for s in studies_with_pubmed])
    write_csv(out_dir / "pmids_linked_valid.csv", [{"pmid": p} for p in valid_pmids_sorted])
    write_csv(
        out_dir / "pmids_with_pmcid.csv",
        [
            {
                "pmid": p,
                "pmcid": pmid_to_pmcid[p],
                "title": pmid_to_title.get(p, ""),
                "title_has_phase1_or_phasei": str(title_has_phase1_marker(pmid_to_title.get(p, ""))),
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{p}/",
                "pmc_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid_to_pmcid[p]}/",
            }
            for p in pmc_pmids
        ],
    )
    write_csv(
        out_dir / "pmids_phase1_title_with_pmcid.csv",
        [
            {
                "pmid": p,
                "pmcid": phase1_title_pmid_to_pmcid[p],
                "title": pmid_to_title.get(p, ""),
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{p}/",
                "pmc_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{phase1_title_pmid_to_pmcid[p]}/",
            }
            for p in phase1_title_pmc_pmids
        ],
    )
    write_csv(
        out_dir / "studies_with_phase1_title_free_pmc_pub.csv",
        [s.to_row() for s in studies_with_phase1_title_pmc],
    )
    write_csv(out_dir / "pmc_download_failures.csv", [{"pmcid": x} for x in sorted(set(failed_pmcids))])

    summary = {
        "run_dir": str(out_dir),
        "criteria": [
            "Completed",
            "Has Results",
            f"Condition/Disease = {args.condition.strip()}" if args.condition.strip() else "Condition/Disease = (none)",
            "Phase filter by mode",
            "Has linked publication with PMID in ClinicalTrials.gov references",
            "Publication available in PubMed",
            "Free PMC article available in PubMed (has PMCID)",
            "Title contains 'Phase I' or 'Phase 1'",
        ],
        "counts": {
            "n_studies_phase_filter_completed_with_results": len(parsed_studies),
            "n_studies_with_linked_pubmed": len(studies_with_pubmed),
            "n_unique_linked_pmids": len(all_linked_pmids),
            "n_pubmed_available_pmids": len(valid_pmids_sorted),
            "n_free_pmc_articles_total": len(pmc_pmids),
            "n_free_pmc_articles_title_has_phase1_or_phasei": len(phase1_title_pmc_pmids),
            "n_unique_filtered_pmcids": len(unique_pmcids_filtered),
            "n_studies_with_at_least_one_phase1_title_free_pmc_pub": len(studies_with_phase1_title_pmc),
            "n_downloaded_pmc_xml": downloaded_pmc_xml,
            "n_failed_pmc_downloads": len(set(failed_pmcids)),
            "n_ready_pmc_visual_assets": int(
                visual_asset_summary.get("ready_asset_count", 0) or 0
            ),
            "n_failed_visual_asset_articles": int(
                visual_asset_summary.get("failed_count", 0) or 0
            ),
            "n_ready_pmc_pdfs": int(pdf_asset_summary.get("ready_count", 0) or 0),
            "n_unavailable_pmc_pdfs": int(
                pdf_asset_summary.get("unavailable_count", 0) or 0
            ),
            "n_failed_pmc_pdf_downloads": int(
                pdf_asset_summary.get("failed_count", 0) or 0
            ),
        },
        "generated_at": dt.datetime.now().isoformat(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for k, v in summary["counts"].items():
        print(f"{k}: {v}")
    print(f"Output written to: {out_dir}")


if __name__ == "__main__":
    main()
