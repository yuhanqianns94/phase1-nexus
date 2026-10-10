"""Shared closed choices for article extraction and prospective trial entry.

Codes are exact, auditable categories, not a medical ontology. Missing/unclear
information is None, never a fabricated default. ``other`` retains the source
wording; it does not authorize a new category. Biological names stay open text.
"""
from __future__ import annotations

from typing import Literal, get_args
import re
import unicodedata

SexEligibility = Literal['female', 'male', 'no_restriction']
StudyPhase = Literal['phase_0', 'phase_i', 'phase_ia', 'phase_ib', 'phase_i_ii', 'phase_ii', 'phase_ii_iii', 'phase_iii', 'phase_iv', 'other']
ClinicalMetric = Literal['age', 'performance_status', 'ecog', 'karnofsky', 'anc', 'platelets', 'hemoglobin', 'bilirubin', 'ast', 'alt', 'creatinine', 'creatinine_clearance', 'egfr', 'lvef', 'qtc', 'prior_therapy', 'neutropenia', 'thrombocytopenia', 'anemia', 'febrile_neutropenia', 'hematologic_toxicity', 'nonhematologic_toxicity', 'dlt', 'other']
Comparator = Literal['<', '<=', '=', '!=', '>=', '>', 'between']
AgeLowerComparator = Literal['>', '>=']
AgeUpperComparator = Literal['<', '<=']
ConditionCategory = Literal['performance_status', 'organ_function', 'hematologic', 'prior_treatment', 'disease', 'demographic', 'comorbidity', 'concomitant_medication', 'toxicity', 'other']
ConditionUnit = Literal['grade', 'score', '%', 'x_uln', 'mg/dL', 'mmol/L', 'umol/L', 'g/dL', 'g/L', '/uL', '10^9/L', 'mL/min', 'mL/min/1.73m2', 'ms', 'days', 'weeks', 'months', 'years', 'other']
AgeUnit = Literal['years', 'months', 'days', 'weeks', 'other']
Route = Literal['oral', 'intravenous', 'subcutaneous', 'intramuscular', 'intrathecal', 'intraperitoneal', 'intratumoral', 'intravesical', 'topical', 'inhaled', 'rectal', 'transdermal', 'intranasal', 'other']
DosageForm = Literal['tablet', 'capsule', 'solution', 'suspension', 'emulsion', 'powder', 'injection', 'infusion', 'liposomal', 'nanoparticle', 'other']
TreatmentStage = Literal['treatment', 'lead_in', 'loading', 'induction', 'consolidation', 'maintenance', 'conditioning', 'other']
FrequencyPattern = Literal['intermittent', 'continuous', 'single_dose', 'as_needed', 'other']
DoseUnit = Literal['mg', 'g', 'micrograms', 'mg/day', 'mg/kg', 'mg/m2', 'mg/kg/day', 'mg/m2/day', 'micrograms/kg', 'micrograms/m2', 'micrograms/kg/day', 'mg/hour', 'micrograms/kg/min', 'AUC', 'IU', 'IU/kg', 'IU/m2', 'units', 'units/kg', 'mL', 'mL/kg', 'Gy', 'MBq', 'cells', 'cells/kg', 'cells/m2', 'pfu', 'viral_particles', 'other']
DoseDenominator = Literal['absolute', 'body_weight', 'body_surface_area', 'auc', 'other']
GradingSystem = Literal['ctcae', 'nci_ctc', 'who', 'rtog_eortc', 'other']
GradingVersion = Literal['1.0', '2.0', '3.0', '4.0', '4.03', '5.0', '6.0', 'other']
WindowAnchor = Literal['first_treatment_dose', 'first_specified_drug_dose', 'start_of_stage', 'end_of_first_cycle', 'end_of_specified_cycle', 'last_treatment_dose', 'end_of_treatment', 'other']
DiseaseModifier = Literal['advanced', 'locally_advanced', 'metastatic', 'recurrent', 'relapsed', 'refractory', 'unresectable', 'progressive', 'newly_diagnosed', 'treatment_naive', 'previously_treated', 'other']
AdjustmentAction = Literal['interrupt', 'resume', 'reduce', 'discontinue', 'withhold', 'other']
ResearchRole = Literal['primary', 'co_primary', 'partner', 'background', 'unclear']
TreatmentRole = Literal['escalated', 'fixed_backbone', 'combination_partner', 'conditioning', 'supportive', 'concomitant', 'radiation', 'maintenance', 'other', 'unclear']
StudyPartRole = Literal['escalation', 'deescalation', 'expansion', 'confirmation', 'lead_in', 'maintenance', 'mixed', 'other', 'unclear']
DoseBasis = Literal['per_administration', 'daily_total', 'infusion_rate', 'other', 'unclear']
IntervalUnit = Literal['hours', 'days', 'weeks']
CycleUnit = Literal['days', 'weeks']
InfusionUnit = Literal['minutes', 'hours']


def _entry(label, literal, *, display=None, aliases=None):
    values = get_args(literal)
    return {'label': label, 'options': tuple(values),
            'labels': {value: (display or {}).get(value, value.replace('_', ' ').capitalize()) for value in values},
            'aliases': dict(aliases or {})}


CATALOG = {
    'sex_eligibility': _entry('Sex eligibility', SexEligibility,
        display={'female':'Female', 'male':'Male', 'no_restriction':'No restriction'},
        aliases={'women':'female','females':'female','female only':'female','women only':'female','men':'male','males':'male','male only':'male','men only':'male','both':'no_restriction','both sexes':'no_restriction','male and female':'no_restriction','males and females':'no_restriction','all':'no_restriction','all sexes':'no_restriction','any sex':'no_restriction','unrestricted':'no_restriction','no sex restriction':'no_restriction'}),
    'study_phase': _entry('Study phase', StudyPhase, display={
        'phase_0':'Phase 0','phase_i':'Phase I','phase_ia':'Phase Ia','phase_ib':'Phase Ib',
        'phase_i_ii':'Phase I/II','phase_ii':'Phase II','phase_ii_iii':'Phase II/III','phase_iii':'Phase III','phase_iv':'Phase IV'},
        aliases={'phase 1':'phase_i','i':'phase_i','1':'phase_i','phase 1a':'phase_ia','ia':'phase_ia','phase 1b':'phase_ib','ib':'phase_ib','phase 1/2':'phase_i_ii','i/ii':'phase_i_ii','phase 2':'phase_ii','ii':'phase_ii','phase 2/3':'phase_ii_iii','phase 3':'phase_iii','phase 4':'phase_iv'}),
    'clinical_metric': _entry('Clinical measure or event', ClinicalMetric, display={
        'ecog':'ECOG performance status','karnofsky':'Karnofsky performance status','anc':'Absolute neutrophil count','ast':'AST','alt':'ALT','egfr':'eGFR','lvef':'LVEF','qtc':'QTc','dlt':'DLT','nonhematologic_toxicity':'Nonhematologic toxicity'},
        aliases={'ecog performance status':'ecog','absolute neutrophil count':'anc','neutrophil count':'anc','platelet count':'platelets','hb':'hemoglobin','hgb':'hemoglobin','creatinine clearance':'creatinine_clearance','crcl':'creatinine_clearance','kps':'karnofsky','karnofsky performance status':'karnofsky','left ventricular ejection fraction':'lvef','corrected qt interval':'qtc','non-hematologic toxicity':'nonhematologic_toxicity','haemoglobin':'hemoglobin','dose limiting toxicity':'dlt','dose-limiting toxicity':'dlt'}),
    'comparison': _entry('Comparison', Comparator, display={x:x for x in get_args(Comparator)},
        aliases={'≤':'<=','≥':'>=','==':'=','≠':'!=','at least':'>=','at most':'<=','less than':'<','greater than':'>','equal to':'=','range':'between'}),
    'age_min_comparison': _entry('Minimum age comparison', AgeLowerComparator, display={x:x for x in get_args(AgeLowerComparator)}, aliases={'≥':'>=','at least':'>=','greater than':'>'}),
    'age_max_comparison': _entry('Maximum age comparison', AgeUpperComparator, display={x:x for x in get_args(AgeUpperComparator)}, aliases={'≤':'<=','at most':'<=','less than':'<'}),
    'condition_category': _entry('Condition category', ConditionCategory),
    'condition_unit': _entry('Condition threshold unit', ConditionUnit, display={x:x for x in get_args(ConditionUnit)},
        aliases={'×uln':'x_uln','x uln':'x_uln','uln':'x_uln','times uln':'x_uln','percent':'%','μmol/l':'umol/L','µmol/l':'umol/L','/µl':'/uL','/μl':'/uL','cells/ul':'/uL','cells/mm3':'/uL','/mm3':'/uL','×10^9/l':'10^9/L','10⁹/l':'10^9/L','ml/min/1.73m²':'mL/min/1.73m2'}),
    'age_unit': _entry('Age unit', AgeUnit, aliases={'year':'years','yr':'years','yrs':'years','month':'months','mo':'months','day':'days','week':'weeks'}),
    'route': _entry('Administration route', Route, aliases={'po':'oral','p.o.':'oral','orally':'oral','iv':'intravenous','i.v.':'intravenous','iv infusion':'intravenous','intravenous infusion':'intravenous','intravenous injection':'intravenous','sc':'subcutaneous','s.c.':'subcutaneous','sq':'subcutaneous','im':'intramuscular','i.m.':'intramuscular','it':'intrathecal','ip':'intraperitoneal','inhalation':'inhaled'}),
    'dosage_form': _entry('Dosage form', DosageForm, aliases={'tablets':'tablet','capsules':'capsule','oral solution':'solution','solution for injection':'solution','solution for infusion':'solution','lyophilized powder':'powder','lyophilised powder':'powder','liposome':'liposomal','nanoparticles':'nanoparticle'}),
    'treatment_stage': _entry('Treatment stage', TreatmentStage, aliases={'lead-in':'lead_in','lead in':'lead_in','loading dose':'loading'}),
    'frequency_pattern': _entry('Frequency pattern', FrequencyPattern, aliases={'continuous infusion':'continuous','once only':'single_dose','single dose':'single_dose','prn':'as_needed'}),
    'dose_unit': _entry('Dose unit', DoseUnit, display={x:x.replace('m2','m²') for x in get_args(DoseUnit)},
        aliases={'mcg':'micrograms','ug':'micrograms','μg':'micrograms','µg':'micrograms','mg/m²':'mg/m2','mg m-2':'mg/m2','mg/m^2':'mg/m2','mg/m²/day':'mg/m2/day','mcg/kg':'micrograms/kg','μg/kg':'micrograms/kg','µg/kg':'micrograms/kg','ug/kg':'micrograms/kg','mcg/m2':'micrograms/m2','micrograms/m²':'micrograms/m2','mcg/kg/min':'micrograms/kg/min','mg/h':'mg/hour','mg/hr':'mg/hour','iu/m²':'IU/m2','cells/m²':'cells/m2','vp':'viral_particles'}),
    'dose_denominator': _entry('Dose denominator', DoseDenominator, aliases={'none':'absolute','kg':'body_weight','per kg':'body_weight','body weight':'body_weight','bsa':'body_surface_area','m2':'body_surface_area','m²':'body_surface_area','body surface area':'body_surface_area','area under the curve':'auc'}),
    'grading_system': _entry('Toxicity grading system', GradingSystem, display={'ctcae':'CTCAE','nci_ctc':'NCI CTC','who':'WHO','rtog_eortc':'RTOG/EORTC'}, aliases={'common terminology criteria for adverse events':'ctcae','nci-ctc':'nci_ctc','nci ctc':'nci_ctc','ctc':'nci_ctc','rtog':'rtog_eortc','rtog/eortc':'rtog_eortc'}),
    'grading_version': _entry('Grading version', GradingVersion, display={x:x for x in get_args(GradingVersion)}, aliases={str(x):f'{x}.0' for x in range(1,7)}),
    'window_anchor': _entry('DLT window anchor', WindowAnchor, display={
        'first_treatment_dose':'First treatment dose','first_specified_drug_dose':'First dose of a specified drug','start_of_stage':'Start of a specified stage','end_of_first_cycle':'End of first cycle','end_of_specified_cycle':'End of a specified cycle','last_treatment_dose':'Last treatment dose','end_of_treatment':'End of treatment'},
        aliases={'start of therapy':'first_treatment_dose','start of treatment':'first_treatment_dose','treatment start':'first_treatment_dose','first dose':'first_treatment_dose','end of cycle 1':'end_of_first_cycle','end of first cycle':'end_of_first_cycle'}),
    'disease_modifier': _entry('Disease characteristic', DiseaseModifier, aliases={'locally advanced':'locally_advanced','newly diagnosed':'newly_diagnosed','treatment-naive':'treatment_naive','treatment naïve':'treatment_naive','previously treated':'previously_treated'}),
    'adjustment_action': _entry('Dose-adjustment action', AdjustmentAction),
    'research_role': _entry('Research role', ResearchRole, display={'primary':'Primary study drug','co_primary':'Co-primary study drug','partner':'Combination partner','background':'Concomitant medication','unclear':'Undecided'}),
    'treatment_role': _entry('Treatment role', TreatmentRole),
    'study_part_role': _entry('Study part role', StudyPartRole),
    'dose_basis': _entry('Dose accounting basis', DoseBasis),
    'interval_unit': _entry('Frequency interval unit', IntervalUnit),
    'cycle_unit': _entry('Cycle length unit', CycleUnit),
    'infusion_unit': _entry('Infusion duration unit', InfusionUnit),
}


def choices(name: str) -> tuple[str, ...]:
    return CATALOG[name]['options']


def labels(name: str) -> dict[str, str]:
    return dict(CATALOG[name]['labels'])


def _key(value) -> str:
    # Case/whitespace/Unicode typography only; never infer meaning from a
    # substring, a number in arbitrary prose, or an absent statement.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value)).casefold()).strip()


def canonical_value(name: str, raw) -> str | None:
    if raw is None or not isinstance(raw, str) or not raw.strip():
        return None
    entry = CATALOG[name]
    lookup = {_key(value): value for value in entry['options']}
    lookup.update({_key(label): value for value, label in entry['labels'].items()})
    lookup.update({_key(alias): value for alias, value in entry['aliases'].items()})
    return lookup.get(_key(raw))
