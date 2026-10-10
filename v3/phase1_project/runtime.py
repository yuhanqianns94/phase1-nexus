from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"


def activate_project_root() -> Path:
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    sys.path[:] = [item for item in sys.path if item != root]
    sys.path.insert(0, root)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return PROJECT_ROOT

