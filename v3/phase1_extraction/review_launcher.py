"""Console launcher for the v3 Streamlit review page."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("review_app.py")
    passthrough = list(sys.argv[1:] if argv is None else argv)
    sys.argv = ["streamlit", "run", str(app_path), "--", *passthrough]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
