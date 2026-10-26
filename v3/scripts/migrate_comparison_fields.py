"""Offline, idempotent field migration. Preserve source counts and historical review/provenance records; retired R3 is not usable evidence."""
from pathlib import Path
from phase1_extraction.artifacts import ArticlePaths,active_review_root,read_json,write_json,sha256_file,utc_now
from phase1_extraction.models import ReportedExtraction,DocumentIR
from phase1_extraction.field_normalization import normalize_fields
from phase1_extraction.resolver import resolve_extraction
from phase1_extraction.quality import audit_reported_extraction
from phase1_extraction.friendly_review import load_or_create_friendly_review,save_friendly_review


def migrate(root):
    report={'created_at':utc_now(),'external_calls':0,'articles':[]}
    for file in sorted(root.glob('*/reported_extraction.json')):
        paths=ArticlePaths(root,file.parent.name);old=ReportedExtraction.model_validate(read_json(file))
        new,changes=normalize_fields(old)
        old_curves=read_json(paths.curve_rows)
        before_counts=[(r['curve_row_id'],r.get('n'),r.get('y'),r.get('denominator_basis'),r.get('rule_ids')) for r in old_curves['all_rows']]
        # Read compatible review before changing source-derived presentations.
        before_review=load_or_create_friendly_review(paths)
        row_report={'document_id':old.document_id,'changes':changes,'design_preserved':True,'counts_preserved':True}
        assert old.designs==new.designs
        if changes:
            for path in (paths.reported_extraction,paths.curve_rows,paths.friendly_human_review):
                backup=path.with_name(path.stem+'.before_field_migration.json')
                if path.is_file() and not backup.exists():backup.write_bytes(path.read_bytes())
            write_json(file,new)
            resolved=resolve_extraction(new)
            after_counts=[(r.curve_row_id,r.n,r.y,r.denominator_basis,r.rule_ids) for r in resolved.curve_rows]
            assert sorted(before_counts)==sorted(after_counts),(old.document_id,'counts changed')
            write_json(paths.resolved_extraction,resolved)
            write_json(paths.curve_rows,{'policy_version':resolved.policy_version,'all_rows':[r.model_dump(mode='json') for r in resolved.curve_rows],
                'project_default':[r.model_dump(mode='json') for r in resolved.curve_rows if r.included_in_default],
                'reported_only':[r.model_dump(mode='json') for r in resolved.curve_rows if r.included_in_reported_only]})
            write_json(paths.rule_applications,[r.model_dump(mode='json') for r in resolved.rule_applications])
            write_json(paths.issues,[r.model_dump(mode='json') for r in resolved.issues])
        review=load_or_create_friendly_review(paths)
        save_friendly_review(paths,review,record_undo=False)
        old_decisions={u.review_unit_id:u.decision for u in before_review.units if u.decision!='pending'}
        row_report['retained_review_decisions']=sum(old_decisions.get(u.review_unit_id)==u.decision for u in review.units if u.review_unit_id in old_decisions)
        row_report['review_units_requiring_recheck']=[u.review_unit_id for u in review.units if u.review_unit_id in old_decisions and u.decision=='pending']
        report['articles'].append(row_report)
    write_json(root.parent/'comparison_field_migration.json',report)
    return report


if __name__=='__main__':
    report=migrate(active_review_root());print({'articles':len(report['articles']),'changed_articles':sum(bool(r['changes']) for r in report['articles']),'external_calls':0})
