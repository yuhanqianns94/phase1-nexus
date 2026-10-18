"""Real extraction comparisons may mix known and unreported numeric values."""
from phase1_extraction.comparison import _reported_numeric_signature
from phase1_extraction.models import ReportedExtraction


def test_nullable_numeric_fields_compare_independently_of_record_order():
    extraction = ReportedExtraction.model_validate({
        "document_id": "comparison",
        "populations": [
            {"population_id": "lead_in", "study_id": "s", "age_min": 18},
            {"population_id": "expansion", "study_id": "s", "age_min": None},
        ],
        "dlt_windows": [
            {"window_id": "dlt", "study_id": "s", "raw_definition": "21 days", "normalized_days": 21},
            {"window_id": "safety", "study_id": "s", "raw_definition": "Throughout treatment"},
        ],
    })
    reordered = extraction.model_copy(deep=True)
    reordered.populations.reverse()
    reordered.dlt_windows.reverse()
    left = _reported_numeric_signature(extraction)
    right = _reported_numeric_signature(reordered)
    assert left == right
    assert set(left["dlt_window_days"]) == {21, None}
    assert {age[0] for age in left["population_age"]} == {18, None}
