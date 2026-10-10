"""Reproduce illustrative target/BOIN checks from the locally fitted 13-paper corpus."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json

from phase1_prediction.multidrug.contracts import TargetRegimen,TargetDrug,RegimenPoint,PriorSpec,SamplerSpec
from phase1_prediction.multidrug.observations import load_model_library
from phase1_prediction.multidrug.store import PosteriorStore,source_fit_id
from phase1_prediction.multidrug.transport import transport_joint_posteriors,assemble_components
from phase1_simulation.multidrug import simulate_regimen_path
from phase1_simulation.service import BoinSettings


def target_from_slots(slots,name,profile):
    drugs=[]
    for i,sd in enumerate(slots):
        p=copy.deepcopy(sd.profile);p['name']='Illustrative '+chr(68+i);p['aliases']=[]
        drugs.append(TargetDrug(drug_id=chr(68+i),name=p['name'],research_priority=i+1,
            research_role='primary' if i==0 else 'partner',mode=sd.mode,unit=sd.unit,
            reference_dose=100. if sd.mode=='numeric' else None,profile=p))
    points=[RegimenPoint(level_id=f'L{i+1}',doses={d.drug_id:((value if j==0 else 100.) if d.mode=='numeric' else None)
        for j,d in enumerate(drugs)}) for i,value in enumerate([50.,100.,200.])]
    return TargetRegimen(name=name,drugs=drugs,points=points,path_level_ids=[p.level_id for p in points],profile=profile)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--store',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    scopes,_=load_model_library([args.root]);store=PosteriorStore(args.store);report=[]
    def entry(s,b):return store.load(source_fit_id(s,b,PriorSpec(),SamplerSpec()))
    for name,doc,k in [('single_from_combo','PMC3784336',1),('two_drug','PMC3784336',2),
                       ('three_drug','PMC3784336',3),('drug_group','PMC3309850',2)]:
        scope=next(s for s in scopes if s.document_id==doc and s.observations and len(s.drugs)>=k)
        target=target_from_slots(scope.drugs[:k],name,scope.profile)
        for backend in ('logistic',):
            result=transport_joint_posteriors([entry(scope,backend)],target,backend=backend,draw_count=40)
            assert result['status']=='fitted',result['excluded']
            simulation=simulate_regimen_path(result,settings=BoinSettings(),posterior_curve_count=10,trials_per_curve=10)
            path=out/f'{name}_{backend}.json';path.write_text(json.dumps({'prediction':result,'simulation':simulation},indent=2))
            report.append({'scenario':name,'backend':backend,'status':result['status'],'trials':simulation['total_trials'],'file':str(path)})
    singles=[next(s for s in scopes if s.document_id==doc and s.observations and len(s.drugs)==1)
             for doc in ('PMC3507143','PMC3556251')]
    target=target_from_slots([s.drugs[0] for s in singles],'cross_article_assembly',singles[0].profile)
    for backend in ('logistic',):
        result=assemble_components([entry(s,backend) for s in singles],target,backend=backend,draw_count=40)
        assert result['status']=='fitted'
        path=out/f'assembly_{backend}.json';path.write_text(json.dumps(result,indent=2))
        report.append({'scenario':'cross_article_assembly','backend':backend,'status':result['status'],'file':str(path)})
    (out/'end_to_end_examples.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
