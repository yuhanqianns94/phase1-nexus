"""Offline, resumable migration. Never submits paid extraction requests."""
import argparse
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from .contracts import PriorSpec, SamplerSpec, fingerprint
from .observations import load_model_library
from .service import fit_library, DEFAULT_STORE, BACKENDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+')
    p.add_argument('--store', default=str(DEFAULT_STORE))
    p.add_argument('--report', required=True)
    p.add_argument('--backend', choices=BACKENDS, default='logistic')
    p.add_argument('--evidence-policy', default='project_default', choices=('project_default','reported_only'))
    p.add_argument('--review-policy', default='extracted', choices=('extracted',))
    p.add_argument('--input-only', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Write audit sidecars only; do not fit')
    p.add_argument('--document', action='append', help='Optional document IDs for resumable per-article work')
    p.add_argument('--activate', action='store_true', help='Atomically activate a complete library model report')
    p.add_argument('--sampler-json', help='Explicit saved sampler config; no hidden reduced diagnostics')
    args=p.parse_args()
    scopes, diagnostics=load_model_library(args.roots,evidence_policy=args.evidence_policy,review_policy=args.review_policy)
    if args.document:
        scopes=[s for s in scopes if s.document_id in set(args.document)]
    report_path=Path(args.report); report_path.parent.mkdir(parents=True,exist_ok=True)
    inputs_path=report_path.with_suffix('.scopes.json')
    inputs=[s.model_dump(mode='json') for s in scopes]
    inputs_path.write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
    sampler=SamplerSpec.model_validate_json(Path(args.sampler_json).read_text()) if args.sampler_json else SamplerSpec()
    # Ordering annotations are separate sidecars. Original source and reviews
    # retain byte identities, so reviewed/gold records cannot become stale.
    from phase1_prediction.evidence import discover_article_paths
    from phase1_extraction.models import ReportedExtraction
    from phase1_extraction.drug_ordering import apply_drug_ordering, clinical_payload
    articles,_=discover_article_paths(Path(r) for r in args.roots)
    audit=[]
    for paths in articles.values():
        if args.document and paths.document_id not in args.document:continue
        original=ReportedExtraction.model_validate_json(paths.reported_extraction.read_text())
        ordered=apply_drug_ordering(original)
        unchanged=clinical_payload(original)==clinical_payload(ordered)
        if not unchanged:raise ValueError('Ordering migration changed clinical facts')
        destination=report_path.parent/'ordering'/paths.document_id
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'ordered_extraction.json').write_text(ordered.model_dump_json(indent=2))
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.root.rglob('*.json')
               if 'review' in p.name or p==paths.reported_extraction or p==paths.curve_rows}
        audit.append({'document_id':paths.document_id,'clinical_facts_unchanged':unchanged,'original_checksums':files})
    report={'format':'multidrug_library_migration_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'input_fingerprint':fingerprint(inputs),'diagnostics':diagnostics,'sampler':sampler.model_dump(),
            'prior':PriorSpec().model_dump(),'fits':[],'errors':[], 'complete':False,'ordering_audit':audit}
    def save():
        tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,report_path)
    save()
    if not (args.input_only or args.dry_run):
        for backend in (args.backend,):
            for scope in scopes:
                print(f'{scope.document_id} {scope.scope_id} {backend}',flush=True)
                entries,errors=fit_library([scope],backends=(backend,),store_root=args.store,sampler=sampler)
                report['fits'].extend(m for _,_,m in entries);report['errors'].extend(errors);save()
                print(entries[0][2]['status'] if entries else errors,flush=True)
    for article in audit:
        for file,digest in article['original_checksums'].items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:
                raise ValueError('Original extraction/review changed during migration: '+file)
    report['complete']=True;save()
    if args.activate:
        if args.document or args.input_only or args.dry_run or report['errors'] or any(m['status']=='numerical_failure' for m in report['fits']):
            raise ValueError('Activation requires a complete library run without fit errors')
        marker=Path(args.store).parent/'active_source_models.json'
        previous=json.loads(marker.read_text()) if marker.exists() else None
        activation_id=fingerprint([report['input_fingerprint'],sorted(m['source_fit_id'] for m in report['fits'])])
        snapshots=report_path.parent/'snapshots';snapshots.mkdir(parents=True,exist_ok=True)
        snapshot=snapshots/(activation_id+'.json')
        if not snapshot.exists():
            temp=snapshot.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,ensure_ascii=False));os.replace(temp,snapshot)
        if not previous or previous.get('activation_id')!=activation_id:
            payload={'report':str(snapshot.resolve()),'store':str(Path(args.store).resolve()),'roots':args.roots,
                     'input_fingerprint':report['input_fingerprint'],'activation_id':activation_id,
                     'previous':{k:v for k,v in previous.items() if k!='previous'} if previous else None}
            temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2));os.replace(temp,marker)
    print(json.dumps({'report':str(report_path),'documents':diagnostics['document_count'],'fits':len(report['fits']),'errors':len(report['errors'])}),flush=True)


if __name__=='__main__': main()
