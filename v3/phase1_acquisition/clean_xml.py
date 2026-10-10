#!/usr/bin/env python3
"""
Create reference-stripped copies of PMC XML files for downstream extraction.

Default input:
  output/pmc_xml/*.xml

Default output:
  output/pmc_xml_no_refs/*.xml

This removes JATS reference-list elements such as <ref-list> so downstream GPT
extraction does not spend tokens on bibliography/reference sections.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove reference sections from all PMC XML files in a folder."
    )
    p.add_argument(
        "--input-dir",
        default="output/pmc_xml",
        help="Directory containing original PMC XML files.",
    )
    p.add_argument(
        "--output-dir",
        default="output/pmc_xml_no_refs",
        help="Directory where reference-stripped XML files will be written.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output-dir.",
    )
    return p.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def register_namespaces(xml_path: Path) -> None:
    try:
        for _, item in ET.iterparse(xml_path, events=("start-ns",)):
            prefix, uri = item
            ET.register_namespace(prefix or "", uri)
    except ET.ParseError:
        # The main parse will report the actual error with file context.
        return


def find_xml_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.glob("*.xml") if p.is_file())


def removable_reference_elements(root: ET.Element) -> Iterable[ET.Element]:
    for elem in root.iter():
        tag = local_name(elem.tag).lower()
        if tag == "ref-list":
            yield elem


def strip_references(xml_path: Path, output_path: Path) -> int:
    register_namespaces(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    removed = 0
    for elem in list(removable_reference_elements(root)):
        parent = parent_by_child.get(elem)
        if parent is None:
            continue
        parent.remove(elem)
        removed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return removed


def write_report(rows: List[Dict[str, str]], output_dir: Path) -> None:
    report_csv = output_dir / "reference_strip_report.csv"
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["xml_file", "output_file", "removed_ref_lists", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_files": len(rows),
        "written": sum(1 for r in rows if r["status"] == "written"),
        "skipped_existing": sum(1 for r in rows if r["status"] == "skipped_existing"),
        "failed": sum(1 for r in rows if r["status"] == "failed"),
        "total_removed_ref_lists": sum(int(r["removed_ref_lists"] or 0) for r in rows),
        "report_csv": str(report_csv),
    }
    (output_dir / "reference_strip_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_all(input_dir: Path, output_dir: Path, overwrite: bool) -> Tuple[int, int]:
    xml_files = find_xml_files(input_dir)
    rows: List[Dict[str, str]] = []

    for xml_path in xml_files:
        output_path = output_dir / xml_path.name
        row = {
            "xml_file": str(xml_path),
            "output_file": str(output_path),
            "removed_ref_lists": "0",
            "status": "",
            "error": "",
        }

        if output_path.exists() and not overwrite:
            row["status"] = "skipped_existing"
            rows.append(row)
            continue

        try:
            removed = strip_references(xml_path, output_path)
            row["removed_ref_lists"] = str(removed)
            row["status"] = "written"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_report(rows, output_dir)
    return len(xml_files), sum(1 for r in rows if r["status"] == "failed")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    n_files, n_failed = process_all(input_dir, output_dir, args.overwrite)
    print(f"Processed XML files: {n_files}")
    print(f"Output directory: {output_dir}")
    print(f"Failed files: {n_failed}")
    if n_failed:
        print(f"See report: {output_dir / 'reference_strip_report.csv'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
