"""Unit spelling for source-valued doses, without relabelling their magnitude."""
from .controlled_vocabulary import canonical_value


def unit_spelling(value):
    code = canonical_value('dose_unit', value)
    text = code if code not in (None, 'other') else value
    return str(text).replace('micrograms', 'ug') if text else None


def dose_unit_for_value(dose):
    """The numerical dose remains expressed in its reported unit.

    A new closed unit code must describe that same magnitude. Legacy normalized
    mass prefixes never relabel an unchanged source number (1000 ug is not
    1000 mg). Conversions are applied explicitly by downstream consumers.
    """
    if dose is None:
        return None
    data = dose if isinstance(dose, dict) else dose.model_dump()
    code = data.get('unit_code')
    raw = data.get('unit_raw')
    raw_code = canonical_value('dose_unit', raw)
    if code and code != 'other':
        if raw_code not in (None, 'other', code):
            raise ValueError('Structured dose unit conflicts with the unit of the source numeric value')
        return unit_spelling(code)
    return unit_spelling(raw or data.get('unit_normalized'))
