#!/usr/bin/env python3
"""Package completed local evidence, or atomically install a downloaded release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile
import zipfile

# Support running the checked-out script without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1_project.shared_release import install_shared_library, pack_shared_library


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack", help="Package completed clinical artifacts and matching source fits")
    pack.add_argument("--extraction-root", action="append", required=True, type=Path,
                      help="Repeat for additional extraction roots; only immediate document directories are read")
    pack.add_argument("--posterior-root", action="append", default=[], type=Path)
    pack.add_argument("--include-incomplete", action="store_true",
                      help="Include processed articles with incomplete coverage, retaining all completeness flags and exclusions")
    pack.add_argument("--output", required=True, type=Path)
    install = subparsers.add_parser("install", help="Validate and activate a versioned bundle")
    install.add_argument("--bundle", required=True, type=Path)
    install.add_argument("--destination", required=True, type=Path)
    install.add_argument("--sha256", help="Expected bundle digest from the publication report")
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            report = pack_shared_library(args.extraction_root, args.output, posterior_roots=args.posterior_root,
                                         include_incomplete=args.include_incomplete)
        else:
            report = install_shared_library(args.bundle, args.destination, expected_sha256=args.sha256)
    except (OSError, ValueError, EOFError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Shared release failed: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
