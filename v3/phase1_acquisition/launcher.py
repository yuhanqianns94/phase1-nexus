from __future__ import annotations

from phase1_project.runtime import activate_project_root


def acquire() -> None:
    from .cli import main
    main()


def clean_xml() -> None:
    activate_project_root()
    from .clean_xml import main
    main()


def download_pdfs() -> None:
    activate_project_root()
    from .pdf_assets import main
    main()


def download_visuals() -> None:
    activate_project_root()
    from .visual_assets import main
    main()
