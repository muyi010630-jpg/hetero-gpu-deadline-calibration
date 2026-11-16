"""Seed-level paired inference; events and scenarios are not pseudo-replicates."""
from __future__ import annotations
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

METRICS = ('prediction_mape', 'sla_violation_rate', 'strict_qos_violation_rate',
           'qos_bound_coverage', 'false_safe_rate', 'p95_completion', 'p99_completion',
           'mean_completion', 'mean_wait', 'optimal_device_rate', 'device_regret',
           'jain_fairness', 'window_jain_fairness_mean')
PP = {'prediction_mape', 'sla_violation_rate', 'strict_qos_violation_rate',
      'qos_bound_coverage', 'false_safe_rate', 'optimal_device_rate'}
PRIMARY = ('prediction_mape', 'sla_violation_rate')
MODES = ('full', 'pool_ema', 'local_matched', 'gated_pred')


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temp, path)


def inference(deltas, rng_seed=71009):
    a = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(rng_seed)
    boot = a[rng.integers(0, len(a), size=(5000, len(a)))].mean(axis=1)
    lo, hi = np.quantile(boot, (.025, .975))
    observed = abs(float(a.mean()))
    if np.all(a == 0):
        p = 1.0
    else:
        count = 0; total = 100000
        # Monte Carlo sign-flip paired randomization test, including observed draw.
        for _ in range(total // 10000):
            signs = rng.integers(0, 2, size=(10000, len(a)), dtype=np.int8) * 2 - 1
            perm = np.abs((signs * a).mean(axis=1))
            count += int(np.count_nonzero(perm >= observed - 1e-15))
        p = (count + 1) / (total + 1)
    return dict(n_seeds=len(a), mean_delta=float(a.mean()), ci95_low=float(lo), ci95_high=float(hi),
                p_paired_signflip=p, below_zero_seeds=int(np.sum(a < 0)),
                above_zero_seeds=int(np.sum(a > 0)), equal_zero_seeds=int(np.sum(a == 0)))


def holm(rows):
    sorted_rows = sorted(rows, key=lambda r: r['p_paired_signflip'])
    previous = 0.0
    for i, row in enumerate(sorted_rows):
        adjusted = min(1.0, max(previous, (len(sorted_rows) - i) * row['p_paired_signflip']))
        row['p_holm_primary6'] = adjusted
        row['significant_holm_05'] = adjusted < .05
        previous = adjusted


def analyze(stage_dir, rows, stage):
    analysis = stage_dir / 'analysis'; analysis.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['variant'], row['pool_size'], row['load'], row['deadline_setting'], row['strict_setting'], row['mode'])].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        rec = dict(zip(('variant', 'pool_size', 'load', 'deadline_setting', 'strict_setting', 'mode'), key))
        rec['n_seeds'] = len(group)
        for m in METRICS:
            vals = np.asarray([float(r[m]) for r in group]); rec[m] = float(vals.mean())
            rec[m + '_sd_seed'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summaries.append(rec)
    write_csv(analysis / 'summary_by_scenario_mode.csv', summaries)
    # All modes have the same seed/scenario cells: paired within cell, then equal
    # scenario weighting within seed. Inference resamples/sign-flips the 40 seeds.
    idx = {(r['variant'], r['pool_size'], r['load'], r['deadline_setting'], r['strict_setting'], r['seed'], r['mode']): r for r in rows}
    pairs = []
    for key, full in idx.items():
        if key[-1] != 'full': continue
        for mode in MODES[1:]:
            other = idx[key[:-1] + (mode,)]
            rec = dict(zip(('variant', 'pool_size', 'load', 'deadline_setting', 'strict_setting', 'seed'), key[:-1]))
            rec['mode'] = mode; rec['comparison'] = mode + '_minus_full'
            for m in METRICS: rec[m] = float(other[m]) - float(full[m])
            pairs.append(rec)
    write_csv(analysis / 'paired_cell_deltas_vs_full.csv', pairs)
    seeds = defaultdict(list)
    for rec in pairs: seeds[(rec['variant'], rec['seed'], rec['mode'])].append(rec)
    seed_rows = []
    for key, group in sorted(seeds.items()):
        rec = dict(zip(('variant', 'seed', 'mode'), key)); rec['scenarios_per_seed'] = len(group)
        for m in METRICS: rec[m] = float(np.mean([r[m] for r in group]))
        seed_rows.append(rec)
    write_csv(analysis / 'paired_seed_deltas_vs_full.csv', seed_rows)
    infer_rows = []
    variants = sorted({r['variant'] for r in rows})
    for variant in variants:
        variant_primary = []
        for mode in MODES[1:]:
            seed_group = [r for r in seed_rows if r['variant'] == variant and r['mode'] == mode]
            for i, m in enumerate(METRICS):
                scale = 100.0 if m in PP else 1.0
                rec = dict(variant=variant, mode=mode, comparison=mode + '_minus_full', metric=m,
                           unit='percentage_points' if m in PP else ('seconds' if 'completion' in m or 'wait' in m else 'unitless'))
                rec.update(inference([r[m] * scale for r in seed_group], 71009 + i))
                rec['primary_test'] = m in PRIMARY
                infer_rows.append(rec)
                if rec['primary_test']: variant_primary.append(rec)
        holm(variant_primary)
    write_csv(analysis / 'paired_seed_inference.csv', infer_rows)
    aggregate = []
    for variant in variants:
        for mode in MODES:
            group = [r for r in rows if r['variant'] == variant and r['mode'] == mode]
            aggregate.append(dict(variant=variant, mode=mode, runs=len(group), **{
                m: float(np.mean([float(r[m]) for r in group])) for m in METRICS}))
    write_csv(analysis / 'summary_by_variant_mode.csv', aggregate)
    (analysis / 'statistics_method.json').write_text(json.dumps(dict(
        experimental_unit='seed (40 full-run seeds)', scenario_weighting='equal within each seed',
        bootstrap_draws=5000, permutation_draws=100000, test='two-sided paired sign-flip Monte Carlo with plus-one correction',
        multiple_comparisons='Holm over three point-replacement contrasts × MAPE/SLA = six primary tests per variant',
        other_metrics='exploratory unadjusted p-values; intervals marginal, not simultaneous',
        rate_reporting='deltas converted to percentage points for designated rate metrics',
        dependence='reuse of fixed mismatch profiles; seeds randomize task streams, not independent migration environments'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    return aggregate, infer_rows
