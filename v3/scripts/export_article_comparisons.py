"""Export local article-pair matrices and readable seven-group diagnostics."""
import csv
from pathlib import Path
from phase1_extraction.artifacts import active_review_root,write_json
from phase1_prediction.evidence import load_evidence_library
from phase1_prediction.profile_similarity import compare_profiles,VERSION


def export(root):
    library=load_evidence_library([root],review_policy='reviewed_if_available')
    records={r['pmcid']:r['data']['_comparison_profile'] for r in library.matching_records}
    ids=sorted(records);out=root.parent/'article_comparison';out.mkdir(exist_ok=True)
    report={'version':VERSION,'article_count':len(ids),'library_fingerprint':library.fingerprint,'external_calls':0,'modes':{}}
    for mode in ('pretrial','full'):
        pairs={};matrix={a:{} for a in ids}
        for i,a in enumerate(ids):
            for b in ids[i:]:
                r=compare_profiles(records[a],records[b],mode=mode)
                reverse=compare_profiles(records[b],records[a],mode=mode)
                assert r['similarity']==reverse['similarity']
                matrix[a][b]=matrix[b][a]=r['similarity']
                pairs[a+'__'+b]=r
        for name,values in [('similarity',matrix)]:
            with (out/f'{mode}_{name}.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['Article',*ids]);w.writerows([[a,*[values[a][b] for b in ids]] for a in ids])
        write_json(out/f'{mode}_details.json',pairs)
        report['modes'][mode]={'distinct_pairs':len(ids)*(len(ids)-1)//2,'symmetric':True}
    write_json(out/'manifest.json',report)
    return report


if __name__=='__main__':print(export(active_review_root()))
