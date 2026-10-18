"""Compatibility entry point and atomic text persistence; acquisition lives in cli."""
from pathlib import Path
import os
import tempfile


def prepare_output_dir(out_dir: Path) -> None:
    """Legacy entry point: never delete the reusable corpus on a rerun."""
    out_dir.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    from .cli import main as run
    run()


if __name__ == "__main__":
    main()
