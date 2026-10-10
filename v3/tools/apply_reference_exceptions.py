"""One-time mechanical application of the explicitly approved v3 exceptions."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def cut(text, start, end, replacement=""):
    a, b = text.index(start), text.index(end, text.index(start))
    return text[:a] + replacement + text[b:]


p = ROOT / "phase1_prediction/reference/oc_simulation.py"
s = p.read_text()
s = cut(s, '    boin_accelerated_titration: bool', '    boin_check_lowest_dose:')
s = s.replace('                ("boin_accelerated_titration", self.boin_accelerated_titration),\n', '')
s = cut(s, '            if self.boin_accelerated_titration_cap_index', '            lowest_delta =')
s = cut(s, '            accelerated_titration=config.', '            check_lowest_dose=')
s = s.replace('        from .design_engines.boin import simulate_boin_trial', '        from phase1_simulation.boin import simulate_boin_trial')
s = s.replace('            elimination_min_n=config.boin_elimination_min_n,\n', '')
# Preserve the previously approved first-cohort safety checks. No inert knob.
s = s.replace('        target = _finite_float(self.target_dlt, label="target_dlt")',
              '        if self.boin_elimination_min_n != 1:\n            raise ValueError("v3 safety checks start from the first cohort; elimination_min_n must be 1.")\n        target = _finite_float(self.target_dlt, label="target_dlt")')
p.write_text(s)
# There must be no dormant second BOIN engine that could re-enable titration.
(ROOT / "phase1_prediction/reference/design_engines/boin.py").write_text(
    '"""The approved cohort-only BOIN implementation is shared by every v3 entry point."""\n'
    'from phase1_simulation.boin import *  # noqa: F401,F403\n')

p = ROOT / "phase1_prediction/reference/boin_parameter_search.py"
s = p.read_text()
s = s.replace('    accelerated_titration: bool\n', '')
s = s.replace('            boin_accelerated_titration=bool(self.accelerated_titration),\n', '')
s = s.replace('            "accelerated_titration": bool(self.accelerated_titration),\n', '')
s = s.replace('    accelerated_titration_options: Sequence[bool],\n', '')
s = s.replace('            accelerated_titration_options,\n', '')
s = cut(s, '                for accelerated in accelerated_titration_options:', '                    candidate = BOINParameterCandidate(')
a, b = s.index('                    candidate = BOINParameterCandidate('), s.index('    if len(candidates) > MAX_BOIN_PARAMETER_CANDIDATES:')
s = s[:a] + ''.join(line[4:] if line.startswith('    ') else line for line in s[a:b].splitlines(keepends=True)) + s[b:]
s = s.replace('                    accelerated_titration=accelerated,\n', '')
s = s.replace('    score_weights: BOINScoreWeights = BOINScoreWeights(),',
'''    score_weights: BOINScoreWeights = BOINScoreWeights(),
    ranking_priority: Sequence[str] = ("correct_decision", "overtoxic_selection", "safety_stop", "mean_total_sample_size"),''')
s = s.replace('    weights = score_weights.validated()', '''    weights = score_weights.validated()
    ranking_priority = tuple(ranking_priority)
    if len(ranking_priority) != 4 or set(ranking_priority) != {"correct_decision", "overtoxic_selection", "safety_stop", "mean_total_sample_size"}:
        raise ValueError("Ranking priority must contain each of the four metrics exactly once.")''')
s = cut(s, '    rows.sort(', '    for rank, row in enumerate(rows, start=1):', '''    rows.sort(key=lambda row: tuple(
        (-1 if metric == "correct_decision" else 1) * float(row["macro_metrics"][metric])
        for metric in ranking_priority) + (str(row["candidate_id"]),))
''')
s = s.replace('        "format": "phase1_boin_parameter_search_v2",', '        "format": "phase1_boin_parameter_search_v3_priority",\n        "ranking_priority": list(ranking_priority),')
s = s.replace('Highest utility under the recorded weights; not a universally optimal ', 'First under the recorded lexicographic priorities; not a universally optimal ')
p.write_text(s)

p = ROOT / "phase1_web/reference/components.py"
s = p.read_text()
s = s.replace('_DEFAULT_BOIN_ELIMINATION_MIN_N = 3', '_DEFAULT_BOIN_ELIMINATION_MIN_N = 1')
s = cut(s, '    accelerated_text = (', '\n\ndef _boin_ranking_rows', '    return f"{calibration_text} · {elimination_text} · {stop_text}"\n')
s = cut(s, '                "Accelerated titration": (', '                acceptable_label:')
s = s.replace('                "Utility score": candidate.get("score"),\n', '')
s = s.replace('    grid_cols = st.columns(3)\n', '    grid_cols = st.columns(2)\n')
s = cut(s, '    accelerated_labels = grid_cols[2].multiselect(', '    grid_error:')
s = cut(s, '        fixed_cols = st.columns(5)', '        check_lowest =', '        fixed_cols = st.columns(2)\n        elimination_min_n = 1\n')
s = s.replace('check_lowest = fixed_cols[2]', 'check_lowest = fixed_cols[0]').replace('lowest_delta = fixed_cols[3]', 'lowest_delta = fixed_cols[1]')
s = cut(s, '        grade2_probability =', '        fixed_flags =')
s = cut(s, '    st.markdown("### 4. Definition of', '    candidate_count =', '''    st.markdown("### 4. Definition of ‘best’")
    st.caption("Choose metric importance, highest first. Later metrics break exact ties only. The winner is conditional on this priority and the simulated scenarios.")
    labels = {"correct_decision": "Correct selection ↑", "overtoxic_selection": "Overtoxic selection ↓", "safety_stop": "Safety early stopping ↓", "mean_total_sample_size": "Mean N ↓"}
    remaining = list(labels)
    ranking_priority = []
    for index, column in enumerate(st.columns(4)):
        key = f"boin_search_priority_{index}"
        if st.session_state.get(key) not in remaining:
            st.session_state.pop(key, None)
        metric = column.selectbox(f"Importance {index + 1}", remaining, format_func=labels.get, key=key)
        ranking_priority.append(metric)
        remaining.remove(metric)

''')
s = s.replace('        * len(accelerated_labels)\n', '')
s = s.replace('        "accelerated": accelerated_labels,\n', '').replace('            accelerated_cap_level,\n', '').replace('            grade2_probability,\n', '')
s = cut(s, '        "weights": [', '    current_signature =', '        "ranking_priority": ranking_priority,\n    }\n')
s = s.replace('                boin_accelerated_titration_cap_index=(int(accelerated_cap_level) - 1),\n', '')
s = s.replace('                boin_accelerated_grade2_probability=float(grade2_probability),\n', '')
s = cut(s, '                accelerated_titration_options=tuple(', '                base_config=')
s = cut(s, '            weights = BOINScoreWeights(', '            with st.status(')
s = s.replace('                    score_weights=weights,', '                    ranking_priority=ranking_priority,')
s = s.replace('Best under the selected score weights', 'Best under the selected metric priorities')
s = s.replace('Results are ordered by the score weights selected above.', 'Results are ordered by the metric priorities selected above.')
# Remove the obsolete utility number from the headline, while leaving OC metrics.
s = s.replace('"Utility score"', '"Diagnostic utility (not ranking)"')
p.write_text(s)
for path in (ROOT / "phase1_prediction/reference").rglob('*.py'):
    ast.parse(path.read_text())
ast.parse(p.read_text())
print('Applied cohort-only BOIN, first-cohort safety, three methods, and configurable ranking priority.')
