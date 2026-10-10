"""Offline structural audit and readable configuration report for the rerun."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from phase1_extraction.artifacts import ArticlePaths, read_json, write_json, utc_now
from phase1_extraction.models import ReportedExtraction, CurveRow
from phase1_extraction.exposures import exposure_text, exposure_schedule
from phase1_extraction.friendly_review import load_or_create_friendly_review, save_friendly_review


def audit(run: Path) -> dict:
    manifest = read_json(run / 'rerun_manifest.json')
    records = []
    lines = ['# 逐队列药物配置复核', '',
             '这是结构检查与人工复核清单；不代表所有临床事实已经人工确认。',
             'true=明确使用；false=明确未使用；unknown=原文未确定；mixed=明确混合人群。',
             '同一行的全部药物共享一组 DLT/N；未分层结果不拆成人工亚组。', '']
    for article in manifest['articles']:
        document_id = article['document_id']
        paths = ArticlePaths(run / 'results', document_id)
        extraction = ReportedExtraction.model_validate(read_json(paths.reported_extraction))
        rows = [CurveRow.model_validate(x) for x in read_json(paths.curve_rows)['all_rows']]
        actual = [c for c in extraction.cohorts if c.actual_status == 'actual']
        findings = read_json(paths.extraction_audit)['findings']
        errors = []
        if extraction.exposure_contract != 'cohort_exposures_v1':
            errors.append('Missing exposure contract')
        for c in actual:
            if not c.exposures:
                errors.append(f'{c.cohort_id}: no exposures')
        cohorts = {c.cohort_id: c for c in extraction.cohorts}
        for r in rows:
            if r.exposures != cohorts[r.cohort_id].exposures:
                errors.append(f'{r.curve_row_id}: exposure linkage mismatch')
            if r.n is not None and r.y is not None and not 0 <= r.y <= r.n:
                errors.append(f'{r.curve_row_id}: invalid DLT/N')
        record = {'document_id': document_id, 'actual_cohorts': len(actual), 'curve_rows': len(rows),
                  'included_rows': sum(r.included_in_default for r in rows),
                  'exposure_coverage': dict(Counter(c.exposure_coverage for c in actual)),
                  'exposure_status': dict(Counter(e.presence for c in actual for e in c.exposures)),
                  'audit_finding_codes': dict(Counter(f['code'] for f in findings)),
                  'structural_errors': errors}
        records.append(record)
        # New outputs start unreviewed; do not transfer old confirmations.
        review = load_or_create_friendly_review(paths)
        save_friendly_review(paths, review, record_undo=False)
        corrections = {u.predicted_point.cohort_id: u for u in review.units
                       if u.predicted_point and u.decision == 'incorrect' and u.gold_point}
        lines += [f'## {document_id}', '', f'实际队列：{len(actual)}；结果行：{len(rows)}；可用曲线行：{record["included_rows"]}。', '']
        def safe(value):
            return str(value).replace('|', '\\|').replace('\n', ' ')
        lines += ['| 队列 | 药物配置与日程 | DLT/N | 纳入曲线 |', '|---|---|---|---|']
        for r in rows:
            config = '; '.join(exposure_text(e) + (f' [{exposure_schedule(e)}]' if exposure_schedule(e) else '') for e in r.exposures)
            corrected = corrections.get(r.cohort_id)
            point = corrected.gold_point if corrected else r
            count = f'{point.y if point.y is not None else "?"}/{point.n if point.n is not None else "?"}'
            if corrected:
                count += f'（复核修正；原提取 {r.y}/{r.n}）'
            lines.append(f'| {safe(r.cohort_id)} | {safe(config)} | {count} | {r.included_in_default} |')
        lines += ['', '自动检查提示：' + (', '.join(f'{k} × {v}' for k, v in record['audit_finding_codes'].items()) or '无'), '']
        for unit in corrections.values():
            lines += [f'复核修正 {unit.predicted_point.cohort_id}：{unit.note}', '']
        if errors:
            lines += ['结构错误：' + '; '.join(errors), '']
    result = {'created_at_utc': utc_now(), 'article_count': len(records), 'articles': records,
              'structural_errors': sum(len(r['structural_errors']) for r in records),
              'note': 'Structural audit only; source interpretation remains subject to human review.'}
    write_json(run / 'exposure_qa.json', result)
    (run / 'EXPOSURE_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parents[2] / 'output/v3_batch_exposures_20260910')
    args = parser.parse_args()
    result = audit(args.run.resolve())
    print({k: v for k, v in result.items() if k != 'articles'})
    raise SystemExit(bool(result['structural_errors']))
