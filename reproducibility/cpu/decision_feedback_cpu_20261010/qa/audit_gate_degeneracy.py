"""Post hoc read-only trajectory audit. Never imports/runs the simulator.

Reconstructs permitted probe candidates from saved decision records, budgets,
cooldowns, and completion-record deadlines. Writes the descriptive audit JSON;
it changes no frozen experiment, selection, inference, or simulation source.
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def device_type(device):
    return device.split('__')[0]


def audit():
    counts = collections.Counter()
    widths, alternative_models, candidate_counts, overlaps, service_overlap_gaps = [], [], [], [], []
    for path in sorted((ROOT / '02_heldout_closed_loop/raw/traces').glob('*__local_window__random__*.jsonl.gz')):
        counts['files'] += 1
        used, count, cumulative_reference, dispatches = 0.0, 0, 0.0, 0
        cooldown = {}
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle]
        deadlines = {r['task_id']: r['deadline'] for r in records if r['kind'] == 'completion'}
        for record in records:
            if record['kind'] != 'decision':
                continue
            candidates = json.loads(record['candidates_json'])
            best = candidates[0]
            context = tuple(json.loads(record['context']))
            now = record['now']
            deadline = deadlines[record['task_id']]
            viable = []
            for candidate in candidates[1:]:
                if device_type(candidate['device']) == device_type(best['device']):
                    continue
                if candidate['pred_s'] > 1.20 * best['pred_s']:
                    continue
                if record['qos'] == 1 and now + candidate['pred_s'] > deadline:
                    continue
                if now - cooldown.get((context, device_type(candidate['device'])), -1e9) < 1.0:
                    continue
                reservation = max(.001, .02 * best['pred_s'], candidate['pred_s'] - best['pred_s'])
                if used + reservation > .02 * (cumulative_reference + best['pred_s']) + 1e-12:
                    continue
                if count + 1 > .05 * (dispatches + 1) + 1e-12:
                    continue
                viable.append((candidate, reservation))
            if record['probed']:
                counts['actual_probes'] += 1
                assert record['qos'] < 2 and viable
                chosen = [v for v in viable if v[0]['device'] == record['selected_device']]
                assert len(chosen) == 1
                useful = [v for v in viable if v[0]['overlap'] > 0 or v[0]['deadline_crossing'] > 0]
                counts['viable_candidates_at_probes'] += len(viable)
                counts['useful_candidates_at_probes'] += len(useful)
                counts['equal_gate_sets'] += int(len(viable) == len(useful))
                counts['gate_set_changed'] += int(len(viable) != len(useful))
                candidate_counts.append(len(viable))
                alternative_models.append(len({device_type(v[0]['device']) for v in viable}))
                for candidate, _ in viable:
                    overlap = candidate['overlap'] > 0
                    crossing = candidate['deadline_crossing'] > 0
                    counts['overlap_positive'] += int(overlap)
                    counts['deadline_crossing_positive'] += int(crossing)
                    counts['overlap_only'] += int(overlap and not crossing)
                    counts['both'] += int(overlap and crossing)
                    counts['deadline_only'] += int(not overlap and crossing)
                    counts['best_cold_nlt8'] += int(best['updates'] < 8)
                    counts['other_cold_nlt8'] += int(candidate['updates'] < 8)
                    width = best['width_log'] + candidate['width_log']
                    widths.append(width)
                    overlaps.append(candidate['overlap'])
                    service_overlap_gaps.append(width - math.log(candidate['pred_s'] / best['pred_s']))
                used += chosen[0][1]
                count += 1
                cooldown[(context, device_type(record['selected_device']))] = now
            cumulative_reference += best['pred_s']
            dispatches += 1
    with (ROOT / '02_heldout_closed_loop/raw/per_run_results.csv').open(encoding='utf-8-sig') as handle:
        raw = list(csv.DictReader(handle))
    paired, gate_counts = {}, collections.defaultdict(lambda: [0.0, 0.0, 0])
    for row in raw:
        key = row['mode'], row['policy']
        gate_counts[key][0] += float(row['probe_eligible_count'])
        gate_counts[key][1] += float(row['critical_eligible_count'])
        gate_counts[key][2] += 1
        if row['mode'] == 'local_window' and row['policy'] in ('random', 'gated_random'):
            unit = tuple(row[k] for k in ('image_seed', 'rho', 'task_seed', 'load'))
            paired.setdefault(unit, {})[row['policy']] = row
    ignored = {'policy', 'arm_id', 'run_id', 'trace_sha256'}
    differing_fields = collections.Counter()
    for arms in paired.values():
        assert set(arms) == {'random', 'gated_random'}
        for field in set(arms['random']) - ignored:
            differing_fields[field] += int(arms['random'][field] != arms['gated_random'][field])
    differing_fields = {k: v for k, v in differing_fields.items() if v}
    denominator = counts['viable_candidates_at_probes']
    return dict(
        posthoc_descriptive_audit=True, simulator_executed=False,
        audit_population='288 LocalWindow/random runs; permitted candidates at actual probes',
        counts=dict(counts),
        mean_viable_workers=statistics.mean(candidate_counts),
        mean_distinct_alternative_models=statistics.mean(alternative_models),
        sum_log_width_min_median_max=[min(widths), statistics.median(widths), max(widths)],
        log_1p20=math.log(1.20),
        service_interval_overlap_fraction=sum(v >= 0 for v in service_overlap_gaps) / len(service_overlap_gaps),
        overlap_min_median_max=[min(overlaps), statistics.median(overlaps), max(overlaps)],
        best_cold_fraction=counts['best_cold_nlt8'] / denominator,
        other_cold_fraction=counts['other_cold_nlt8'] / denominator,
        random_vs_gated_random=dict(paired_runs=len(paired),
                                    excluded_identity_fields=sorted(ignored),
                                    different_fields_excluding_identity=differing_fields),
        gate_opportunity_summary=[dict(mode=k[0], policy=k[1], runs=v[2],
                                       eligible=v[0], critical_eligible=v[1],
                                       fraction=v[1] / v[0])
                                  for k, v in sorted(gate_counts.items()) if v[0] > 0],
        limitations=[
            'Descriptive diagnosis of this frozen implementation and simulation design only.',
            'Cold-window widths and the 1.20 prefilter are a plausible explanation, not a randomized causal decomposition.',
            'The source excludes same-model-as-best replicas from viable probes; replicas do not directly establish the opening condition.',
            'The audit does not refute a different discriminative gate, value-of-information model, or budget rule.'])


if __name__ == '__main__':
    result = audit()
    target = ROOT / 'qa/门控退化诊断.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
