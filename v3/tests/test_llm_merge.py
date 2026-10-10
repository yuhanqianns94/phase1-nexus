from __future__ import annotations

from phase1_extraction.llm import _merge_models
from phase1_extraction.models import (
    DosingPhase,
    Regimen,
    RegimenComponent,
    ReportedExtraction,
)


def test_cross_packet_merge_enriches_nested_entities_without_duplicates() -> None:
    base = ReportedExtraction(
        document_id="DOC",
        regimens=[
            Regimen(
                regimen_id="r1",
                label="A",
                components=[RegimenComponent(component_id="c1", intervention_id="i1")],
            )
        ],
    )
    incoming = ReportedExtraction(
        document_id="DOC",
        regimens=[
            Regimen(
                regimen_id="r1",
                label="A",
                components=[
                    RegimenComponent(
                        component_id="c1",
                        intervention_id="i1",
                        dosing_phases=[DosingPhase(dosing_phase_id="p1", phase_label="cycle 1")],
                    )
                ],
            )
        ],
    )
    merged = _merge_models(base, incoming)
    assert len(merged.regimens) == 1
    assert len(merged.regimens[0].components) == 1
    assert merged.regimens[0].components[0].dosing_phases[0].dosing_phase_id == "p1"

