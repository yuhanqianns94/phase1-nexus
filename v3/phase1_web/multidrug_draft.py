"""Read-only migration of old widget drafts into the full-regimen contract."""
import json
from phase1_prediction.models import TargetTrial
from phase1_prediction.multidrug.service import native_target
from phase1_web.reference.helpers import build_target_query, build_optional_query_fields
from phase1_web.target_profile import apply_extras


def migrate_legacy_draft(payload):
    values=payload.get('values')
    if not isinstance(values,dict):raise ValueError('Legacy draft is missing the values field')
    mapping={'cancer_type':'opt_cancer_type','routes':'opt_routes','schema':'opt_schema','inclusion':'opt_inclusion',
        'exclusion':'opt_exclusion','dlt_definition':'opt_dlt_definition','mechanism':'opt_mechanism',
        'gene_targets':'opt_gene_targets','age_group':'opt_age_group','age_min':'opt_age_min','age_max':'opt_age_max',
        'age_unit':'opt_age_unit','gender':'opt_gender','dose_forms':'opt_dose_forms','schedule_frequency':'opt_schedule_frequency',
        'schedule_cycle_days':'opt_schedule_cycle_days','schedule_administration_days':'opt_schedule_admin_days',
        'dlt_eva_days':'opt_dlt_eva_days','dlt_eva_cycles':'opt_dlt_eva_cycles','dlt_eva_text':'opt_dlt_eva_text'}
    optional=build_optional_query_fields(**{k:values[v] for k,v in mapping.items() if v in values})
    query=build_target_query(drug=values.get('target_drug',''),trial_type=values.get('trial_type_control','single'),
        dose_unit=values.get('target_dose_unit',''),planned_doses_text=values.get('planned_doses_text',''),optional_fields=optional)
    rows=json.loads(values.get('native_target_rows_json') or '[]')
    scenario=values.get('native_target_scenario') or 'Main'
    query=apply_extras(query,rows,scenario,values)
    legacy=TargetTrial(name='Migrated target draft',drug=values['target_drug'],disease=values.get('opt_cancer_type',''),
        planned_doses=tuple(query.get('planned_doses',[])),dose_unit=values['target_dose_unit'],query_fields=query,
        exposure_rows=tuple(rows),selected_scenario=scenario)
    return native_target(legacy)
