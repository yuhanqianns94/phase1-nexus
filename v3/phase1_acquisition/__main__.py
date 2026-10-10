from __future__ import annotations

from phase1_project.runtime import activate_project_root


def main() -> None:
    activate_project_root()
    from .acquire import main as acquire_main

    acquire_main()


if __name__ == "__main__":
    main()
