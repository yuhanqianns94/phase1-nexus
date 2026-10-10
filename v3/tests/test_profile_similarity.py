import copy
import math
from pathlib import Path
import pytest
from phase1_prediction.profile_similarity import compare_profiles,condition,conditions,LABELS,days,normalize_weights
from phase1_prediction.profiles import extraction_profile,background_signature,query_profile
from phase1_prediction.profile_scoring import segment_weight
from phase1_web.target_profile import parse_rows,apply_extras
from phase1_extraction.models import CohortExposure,DoseExpression


def drug(name='A',presence='present',route='oral',kind='drug'):
    return {'name':name,'kind':kind,'presence':presence,'role':'escalated' if name=='A' else 'concomitant','phases':[{'phase':'treatment','route':route,'cycle':'28 days'}]}
def profile(*drugs):return {'configurations':[{'drugs':list(drugs)}]}
def group(a,b,k):return compare_profiles(a,b,weights={k:1})


def test_symmetric_and_order_duplicate_invariant():
    a=profile(drug(),drug('B'));b=profile(drug('B',route='intravenous'),drug())
    r=compare_profiles(a,b);reverse=compare_profiles(b,a)
    assert r['similarity']==reverse['similarity']
    a['configurations'][0]['drugs']*=2;a['configurations']*=2
    assert compare_profiles(a,b)['similarity']==r['similarity']


def test_missing_is_not_absent_or_a_perfect_self_match():
    assert compare_profiles({}, {})['similarity'] is None
    a=profile(drug('B','unknown'));b=profile(drug('B','present'));c=profile(drug('B','absent'))
    # Actual source use is inspected retrospectively; prospective drug matching
    # receives intended use separately in the dose coordinates.
    assert group(a,b,'drugs')['similarity']==group(b,c,'drugs')['similarity']
    full=lambda left,right:compare_profiles(left,right,weights={'drugs':1},mode='full')
    assert full(a,b)['similarity']==full(b,b)['similarity']
    assert full(b,c)['similarity']<full(b,b)['similarity']


def test_drug_binding_is_not_flat_route_bag():
    a=profile(drug('A',route='oral'),drug('B',route='intravenous'))
    b=profile(drug('A',route='intravenous'),drug('B',route='oral'))
    assert group(a,b,'administration')['similarity']<group(a,a,'administration')['similarity']
    assert compare_profiles(a,b)['pairing'][0]['drugs']==[('A','A'),('B','B')]


def test_class_not_invented_member():
    a=profile(drug('EIASD',kind='drug_class'));b=profile(drug('phenytoin'))
    r=compare_profiles(a,b)
    assert r['groups']['administration']['score'] is None
    assert r['groups']['drugs']['score'] is None  # No evidenced correspondence.


@pytest.mark.parametrize('a,b',[('ANC >= 1500 /uL','ANC < 1500 /uL'),('grade 3 neutropenia and fever','grade 3 neutropenia or fever'),('(A and B) or C','A and (B or C)')])
def test_predicate_logic_cannot_be_overridden_by_text_overlap(a,b):
    assert conditions([condition(a)],[condition(b)])['score']==0


def test_grades_compare_as_numeric_predicates():
    a=condition('grade 3 toxicity');b=condition('grade 4 toxicity')
    assert a['threshold']==3 and a['unit']=='grade'
    assert conditions([a],[b])['score']<1


def test_pretrial_excludes_observed_baseline_and_toxicity():
    a=profile(drug());a.update(population={'baseline':['prior therapy']},dlt={'observed':['renal failure']})
    b=copy.deepcopy(a);b['population']['baseline']=['different patients'];b['dlt']['observed']=['fever']
    assert compare_profiles(a,b)['similarity']==compare_profiles(a,a)['similarity']
    assert compare_profiles(a,b,mode='full')['similarity']<compare_profiles(a,a,mode='full')['similarity']


def test_windows_no_cycle_default_and_endpoint_types_differ():
    assert days('2 cycles') is None and days('1 month') is None
    a={'windows':[{'kind':'formal_dlt','days':28}]};b={'windows':[{'kind':'safety_monitoring','days':28}]}
    assert group(a,b,'window')['similarity']==0
    q=query_profile({'DLT_EVA':{'value':{'dlt_eval_cycle_number':2}}})
    assert q['windows'][0]['days'] is None


def test_fixed_partner_doses_separate_background_primary_doses_do_not():
    def exposures(a,b):
        return [CohortExposure(exposure_id=n,name_raw=n,component_id=n,role='escalated' if n=='A' else 'fixed_backbone',presence='present',dose_status='reported',dose=DoseExpression(raw_expression=str(v),value_kind='scalar',value=v,unit_raw='mg')) for n,v in [('A',a),('B',b)]]
    assert background_signature(exposures(1,5),'A')==background_signature(exposures(2,5),'A')
    assert background_signature(exposures(1,5),'A')!=background_signature(exposures(1,10),'A')


def test_configuration_count_cannot_multiply_publication_weight():
    segments=[[{'arm_or_cohort':'a'}],[{'arm_or_cohort':'b'}]]
    a={'similarity':.8,'scope_scores':{'a':{'similarity':.8},'b':{'similarity':.4}}}
    weights=[segment_weight(a,s,segments) for s in segments]
    assert sum(weights)==pytest.approx(.8);assert weights[0]/weights[1]==pytest.approx(2)


def test_design_has_no_weight_or_profile_path(tmp_path):
    from test_prediction_pipeline import _article
    from phase1_extraction.artifacts import read_json
    from phase1_extraction.models import ReportedExtraction,TrialDesign
    p=_article(tmp_path,'PMC1',rows=[(.1,3,0)]);e=ReportedExtraction.model_validate(read_json(p.reported_extraction))
    before=extraction_profile(e)
    e.designs.append(TrialDesign(design_id='d',study_id='study',raw_name='3+3',initial_cohort_n=3))
    e.regimens[0].label='CRM dose escalation design'
    assert extraction_profile(e)==before
    with pytest.raises(ValueError):normalize_weights({'design':1})
    assert set(normalize_weights())==set(LABELS)


def test_target_scenarios_presence_and_no_cartesian_pairs():
    q={'Drug_Name':['A'],'toxicity_curve_drug':'A','Dose_Unit':'mg','planned_doses':[1,2]}
    rows=[{'scenario':'s','drug':'A','role':'escalated','presence':'present','doses':'1,2','unit':'mg'},
          {'scenario':'s','drug':'EIASD','kind':'drug_class','role':'concomitant','presence':'present'}]
    r=parse_rows(rows,q,'s');assert r['Trial_Type']['value']=='single'
    assert r['_target_exposures'][1]['doses']==[]
    rows[1].update(drug='B',role='combination_partner',doses='5,10',unit='mg')
    paired=parse_rows(rows,q,'s')
    assert paired['_target_exposures'][1]['doses']==[5.,10.]
    rows[1]['doses']='5';assert parse_rows(rows,q,'s')['Trial_Type']['value']=='combination'
    rows[1]['presence']='absent'
    with pytest.raises(ValueError,match='absent'):parse_rows(rows,q,'s')


def test_target_window_uses_explicit_cycle_length_only():
    q={'Drug_Name':['A'],'toxicity_curve_drug':'A','DLT_EVA':{'value':{'dlt_eval_cycle_number':2}}}
    r=apply_extras(copy.deepcopy(q),[],'Main',{'native_window_cycle_days':28})
    assert r['_comparison_profile']['windows'][0]['days']==56
    r=apply_extras(copy.deepcopy(q),[],'Main',{})
    assert r['_comparison_profile']['windows'][0]['days'] is None


def test_single_point_scope_is_not_discarded():
    from phase1_prediction.reference.dlt_evidence import curve_arm_or_cohort
    from collections import Counter
    level={'arm_or_cohort':'v3scope_x','Dose_Level':'x','_v3_configuration_scope':True}
    assert curve_arm_or_cohort(level,Counter({'v3scope_x':1}))=='v3scope_x'


def test_compatible_numeric_units_and_grading_version():
    from phase1_prediction.profile_similarity import canonical_threshold
    assert canonical_threshold(100,'mg/dL')==canonical_threshold(1,'g/L')
    a={'dlt':{'grading':'NCI CTC','grading_version':'3.0'}}
    b={'dlt':{'grading':'NCI CTC','grading_version':'4.0'}}
    assert group(a,b,'dlt')['similarity']<group(a,a,'dlt')['similarity']


def test_corresponding_drugs_allow_route_comparison_across_roles():
    a=drug('X');b=drug('Y');a['role']=b['role']='escalated'
    a['mechanisms']=b['mechanisms']=['same mechanism']
    b['role']='fixed_backbone'
    assert group(profile(a),profile(b),'administration')['similarity']==1
    b['phases'][0]['route']='intravenous'
    assert group(profile(a),profile(b),'administration')['similarity']<1


def test_condition_preserves_grade_and_duration_together():
    a=condition('grade >=3 neutropenia >7 days');b=condition('grade >=4 neutropenia >7 days')
    assert conditions([a],[b])['score']<conditions([a],[a])['score']


def test_native_target_draft_tracks_drug_changes():
    import json
    from phase1_web.reference.target_trial_draft import target_trial_values_fingerprint,sanitize_target_trial_values
    a={'native_target_rows_json':json.dumps([{'drug':'A','presence':'present'}])}
    b={'native_target_rows_json':json.dumps([{'drug':'A','presence':'absent'}])}
    assert target_trial_values_fingerprint(a)!=target_trial_values_fingerprint(b)
    assert sanitize_target_trial_values(a)['native_target_rows_json']==a['native_target_rows_json']


def test_field_migration_keeps_biological_description_and_design():
    from phase1_extraction.models import ReportedExtraction,Intervention,TrialDesign
    from phase1_extraction.field_normalization import normalize_fields
    e=ReportedExtraction(document_id='x',interventions=[Intervention(intervention_id='i',name_raw='A',product_or_formulation='human IgG4 monoclonal antibody')],
                         designs=[TrialDesign(design_id='d',study_id='s',raw_name='3+3',initial_cohort_n=3)])
    n,changes=normalize_fields(e)
    assert n.interventions[0].product_or_formulation is None
    assert n.interventions[0].drug_class_raw=='human IgG4 monoclonal antibody'
    assert n.designs==e.designs and changes
    assert normalize_fields(n)[1]==[]


def test_prediction_can_select_one_configuration_without_losing_article_budget(tmp_path):
    from dataclasses import replace
    from test_prediction_pipeline import _article
    from phase1_prediction.evidence import load_evidence_library
    from phase1_prediction.models import TargetTrial
    from phase1_prediction.legacy_service import preview_target_evidence
    _article(tmp_path,'PMC1',rows=[(.1,3,0),(.3,6,2)])
    lib=load_evidence_library([tmp_path]);first=lib.series[0]
    second=replace(first,series_key=first.series_key+'_other',window_id='other')
    lib=replace(lib,series=(first,second))
    target=TargetTrial('T','IPH2101','multiple myeloma',(.1,.3),'mg/kg')
    preview=preview_target_evidence(lib,target,threshold=0,include_series_keys=[first.series_key])
    assert [s['series_key'] for s in preview['selected_series']]==[first.series_key]


def test_human_review_does_not_leak_observed_outcomes_into_pretrial(tmp_path):
    from test_prediction_pipeline import _article
    from phase1_extraction.artifacts import read_json
    from phase1_extraction.models import ReportedExtraction
    p=_article(tmp_path,'PMC1',rows=[(.1,3,0)]);e=ReportedExtraction.model_validate(read_json(p.reported_extraction))
    x=extraction_profile(e,overrides={'dlt_definition':'**DLT criteria**\n\ngrade 3 toxicity\n\n**Observed DLTs (cohort 1)**\n\nrenal failure'})
    y=extraction_profile(e,overrides={'dlt_definition':'**DLT criteria**\n\ngrade 3 toxicity\n\n**Observed DLTs (cohort 1)**\n\nfever'})
    assert compare_profiles(x,y)['similarity']==compare_profiles(x,x)['similarity']
    assert compare_profiles(x,y,mode='full')['similarity']<compare_profiles(x,x,mode='full')['similarity']
    unbound=extraction_profile(e,overrides={'dosing':'A: oral; B: intravenous'})
    assert all(not d['phases'] and d['schedule'] is None for c in unbound['configurations'] for d in c['drugs'])
