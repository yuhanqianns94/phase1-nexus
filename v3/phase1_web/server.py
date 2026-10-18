"""Cloud launcher with verified shared-data bootstrap and bounded resources."""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


def bootstrap_shared_library():
    from .hosted import data_root
    from phase1_project.shared_release import install_shared_library
    url = os.environ.get("PHASE1_SHARED_BUNDLE_URL", "").strip()
    checksum = os.environ.get("PHASE1_SHARED_BUNDLE_SHA256", "").strip().lower()
    if not url and not checksum:
        return
    if urlparse(url).scheme != "https" or not re.fullmatch(r"[a-f0-9]{64}", checksum):
        raise ValueError("Shared library download requires an HTTPS URL and a SHA-256 checksum.")
    destination = data_root() / "shared"
    receipt = destination / "download.sha256"
    if receipt.is_file() and receipt.read_text().strip() == checksum and (destination / "current.json").is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shared-download-", dir=destination) as temporary:
        bundle = Path(temporary) / "library.tar.gz"
        digest = hashlib.sha256()
        total = 0
        with urlopen(url, timeout=60) as response, bundle.open("wb") as output:
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("Shared library redirects must remain HTTPS.")
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 4 * 1024**3:
                    raise ValueError("The shared library bundle exceeds the 4 GiB download limit.")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError("Shared library download checksum mismatch.")
        install_shared_library(bundle, destination, expected_sha256=checksum)
        receipt.write_text(checksum + "\n")


def _cleanup_loop():
    from .hosted import cleanup_sessions
    while True:
        try:
            cleanup_sessions()
        except OSError:
            pass
        threading.Event().wait(3600)


def main():
    from .hosted import is_public, validate_public_environment
    validate_public_environment()
    for key, value in {
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }.items():
        os.environ.setdefault(key, value)
    if is_public():
        bootstrap_shared_library()
        threading.Thread(target=_cleanup_loop, daemon=True).start()
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")),
                "--server.address", os.environ.get("PHASE1_BIND_ADDRESS", "0.0.0.0"), "--server.port", os.environ.get("PORT", "8501"),
                "--server.headless", "true", "--server.maxUploadSize", "20",
                "--browser.gatherUsageStats", "false", "--client.showErrorDetails", "none",
                "--client.toolbarMode", "viewer"]
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
