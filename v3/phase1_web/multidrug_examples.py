"""Copy extracted clinical facts into a target draft without importing outcomes."""
import copy

from phase1_prediction.multidrug.contracts import TargetDrug, TargetRegimen, RegimenPoint, fingerprint


def target_from_scope(scope):
    drugs=[TargetDrug(drug_id=d.model_drug_id,name=d.name,research_priority=d.research_priority,
        research_role=d.research_role,mode=d.mode,reference_dose=d.reference_dose,unit=d.unit,
        profile=copy.deepcopy(d.profile)) for d in scope.drugs]
    points=[]
    seen=set()
    for obs in scope.observations:
        if any(e.presence not in {'present','absent'} or e.candidates or e.dose_subgroups for e in obs.exposures):
            raise ValueError('This scope contains uncertain use, mixed populations or complex doses. Create explicit target scenarios manually.')
        doses={e.drug_id:0. if e.presence=='absent' else e.dose for e in obs.exposures}
        identity=tuple(doses[d.drug_id] for d in drugs)
        if identity in seen:continue
        seen.add(identity)
        points.append(RegimenPoint(level_id=f'L{len(points)+1}',doses=doses))
    if not points:
        raise ValueError('This scope has no explicit regimen that can be copied.')
    profile=copy.deepcopy(scope.profile)
    profile.get('dlt',{}).pop('observed',None)
    profile.get('population',{}).pop('baseline',None)
    profile.pop('review_notes',None)
    for config in profile.get('configurations',[]):
        config['status']='planned'
        config.pop('drugs',None)
    source_fingerprint=fingerprint({'profile':profile,'drugs':[d.model_dump(mode='json') for d in drugs],
                                    'points':[p.model_dump(mode='json') for p in points]})
    profile['example_source']={'document_id':scope.document_id,'scope_id':scope.scope_id,
        'label':scope.label,'source_fingerprint':source_fingerprint,
        'note':'Extracted clinical fields and regimen doses; DLT/N not imported. Reference dose is a computational scale. Target DLT 0.25 is illustrative. Select a BOIN path separately.'}
    return TargetRegimen(name=f'{scope.document_id} · {scope.label} · Test example',drugs=drugs,
                         points=points,profile=profile,path_level_ids=[],target_dlt=.25)
