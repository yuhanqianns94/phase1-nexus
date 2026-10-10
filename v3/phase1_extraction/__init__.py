"""Phase I literature extraction v3.

The package deliberately separates source reading, model extraction, policy
resolution, and curve projection.  Importing it never performs network I/O.
"""

from .models import (
    DocumentIR,
    ReportedExtraction,
    ResolvedExtraction,
)
from .resolver import resolve_extraction

__all__ = [
    "DocumentIR",
    "ReportedExtraction",
    "ResolvedExtraction",
    "resolve_extraction",
]

__version__ = "3.0.0"

