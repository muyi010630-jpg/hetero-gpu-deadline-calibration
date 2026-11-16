# -*- coding: utf-8 -*-
"""N2 调度器基线实验 —— 并行重跑（修正 Scenario 参数口径）。

修复内容
--------
上一版 run_n2_cpu.py 传给 Scenario(...) 的后三个实参是 (0.0, 2.0, 2.5)，
而产出论文主表的 C2 口径是 (0.006, 1.60, 2.00)：
    runtime_anomaly_rate    : 0.0    -> 0.006
    runtime_anomaly_slowdown: 2.0    -> 1.60
    burst_multiplier        : 2.5    -> 2.00
burst_multiplier 直接进到达过程，导致同 seed 生成的任务流完全不同。

本脚本保持其余一切不变（同一 hetero_qos_sim_n2 模拟器、同一 N2Calibrator、
同一 seeds/pools/loads/deadlines/strict），只修正这三个实参，并改为多进程并行。

验收标准：full 模式在前 20 个 seed（601-620）上应与 C2 主表
per_seed_results.csv 的 full_qos 逐 cell 精确一致（|Δ| < 1e-9）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_n2_cpu as base  # noqa: F401  触发 sim.OnlineCalibrator = N2Calibrator 补丁
import hetero_qos_sim_n2 as sim  # noqa: E402

from run_n2_cpu import (  # noqa: E402
    N2Calibrator, MODES, SEEDS_FULL, SEEDS_SMOKE, POOL_FULL, POOL_SMOKE,
    LOADS, DEADLINES, STRICT_PROBS, DEADLINE_FACTORS, DEADLINE_JITTER, HORIZON,
    OUT_ROOT, PROFILE_ROOT, write_csv, mean, sd,
)

# ---- C2 主表口径（唯一被修改的部分）----
C2_ANOMALY_RATE = 0.006
C2_ANOMALY_SLOWDOWN = 1.60
C2_BURST = 2.00

FULL_QOS = dict(ema_rho=0.30, kappa_qos=0.0, hier_tau=1.5,
                risk_beta=0.80, cold_uncertainty=0.30)

LABELS = {'full': 'Full-QoS', 'gated': 'Gated-Full-QoS',
          'window': 'Window-QoS', 'kalman': 'Kalman-QoS'}

METRIC_COLS = ['prediction_mape', 'sla_violation_rate', 'strict_qos_violation_rate',
               'qos_bound_coverage', 'false_safe_rate', 'p95_completion',
               'p99_completion', 'jain_fairness', 'window_jain_fairness_mean',
               'device_regret']
KEY_COLS = ['pool_size', 'load', 'deadline_setting', 'strict_setting']


def run_unit(task):
    """一个任务单元 = 一个 (pool, load, deadline, strict, seed) 下的全部 4 个模式。"""
    (pool_size, load_name, load, deadline_name, deadline_scale,
     strict_name, strict_prob, seed) = task
    profile = str(PROFILE_ROOT / f'cluster_{pool_size:02d}_profile.csv')
    rows = []
    for mode in MODES:
        N2Calibrator.mode = mode
        sc = sim.Scenario(
            random.Random(seed), HORIZON, pool_size * load, True,
            profile, 1.0, 1.0, strict_prob, deadline_scale,
            C2_ANOMALY_RATE, C2_ANOMALY_SLOWDOWN, C2_BURST,
            'median', 1.0, DEADLINE_JITTER[0], DEADLINE_JITTER[1], 0.0, 0.0,
            DEADLINE_FACTORS,
        )
        p = FULL_QOS
        runner = sim.EventSimulator(
            sc, 'full_qos', seed, 0.0, 24, 4,
            p['ema_rho'], 0.20, p['kappa_qos'],
            1.0, 0.50, 0.22, 1.20, p['hier_tau'],
            0.65, 0.080, 0.015, 8, p['risk_beta'], p['cold_uncertainty'],
        )
        result = runner.run()
        result.update({
            'mode': mode, 'seed': seed, 'pool_size': pool_size, 'load': load_name,
            'load_per_device': load, 'deadline_setting': deadline_name,
            'deadline_scale': deadline_scale, 'strict_setting': strict_name,
            'strict_qos_prob': strict_prob, 'profile_csv': profile,
            'strategy_label': LABELS[mode],
        })
        result['strategy'] = mode
        rows.append(result)
    return rows


def build_tasks(run_mode):
    seeds = SEEDS_SMOKE if run_mode == 'smoke' else SEEDS_FULL
    pools = POOL_SMOKE if run_mode == 'smoke' else POOL_FULL
    loads = (('mid', 1.25),) if run_mode == 'smoke' else LOADS
    dls = (('base', 0.80),) if run_mode == 'smoke' else DEADLINES
    sts = (('strict25', 0.25),) if run_mode == 'smoke' else STRICT_PROBS
    tasks = []
    for pool in pools:
        for ln, lv in loads:
            for dn, dv in dls:
                for sn, sv in sts:
                    for seed in seeds:
                        tasks.append((pool, ln, lv, dn, dv, sn, sv, seed))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()

    tasks = build_tasks(args.mode)
    out_dir = OUT_ROOT / ('raw/smoke' if args.mode == 'smoke' else 'raw/full')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[n2-par] mode={args.mode} tasks={len(tasks)} workers={args.workers}', flush=True)
    t0 = time.time()
    rows = []
    done = 0
    with Pool(processes=args.workers) as pool:
        for unit_rows in pool.imap_unordered(run_unit, tasks, chunksize=1):
            rows.extend(unit_rows)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f'[n2-par] {done}/{len(tasks)}  elapsed={el/60:.1f}min  eta={eta/60:.1f}min',
                      flush=True)
    print(f'[n2-par] simulation done in {(time.time()-t0)/60:.1f} min, rows={len(rows)}', flush=True)

    # 稳定排序，保证可复现
    mode_order = {m: i for i, m in enumerate(MODES)}
    rows.sort(key=lambda r: (int(r['pool_size']), str(r['load']), str(r['deadline_setting']),
                             str(r['strict_setting']), int(r['seed']), mode_order[r['mode']]))
    write_csv(out_dir / 'per_seed_results.csv', rows)

    # ---- 与原脚本一致的聚合输出 ----
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(k, '')) for k in KEY_COLS) + (row['mode'],)].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        record = {k: v for k, v in zip(KEY_COLS, key[:len(KEY_COLS)])}
        record['mode'] = key[-1]
        record['seeds'] = len(group)
        for metric in METRIC_COLS:
            record[metric + '_mean'] = mean(r.get(metric) for r in group)
            record[metric + '_std'] = sd(r.get(metric) for r in group)
        summaries.append(record)
    write_csv(OUT_ROOT / 'analysis/summary_by_scenario.csv', summaries)

    paired = []
    index = {(tuple(str(r.get(k, '')) for k in KEY_COLS), str(r.get('seed', '')), r['mode']): r
             for r in rows}
    scenarios = sorted({tuple(str(r.get(k, '')) for k in KEY_COLS) for r in rows})
    for scenario in scenarios:
        seeds_here = sorted({str(r.get('seed', '')) for r in rows
                             if tuple(str(r.get(k, '')) for k in KEY_COLS) == scenario})
        for seed in seeds_here:
            full = index.get((scenario, seed, 'full'))
            if not full:
                continue
            for mode in ('window', 'kalman', 'gated'):
                other = index.get((scenario, seed, mode))
                if not other:
                    continue
                for metric in METRIC_COLS:
                    paired.append({**{k: v for k, v in zip(KEY_COLS, scenario)},
                                   'seed': seed, 'mode': mode, 'baseline': 'full',
                                   'metric': metric,
                                   'delta_mode_minus_full': float(other[metric]) - float(full[metric])})
    write_csv(OUT_ROOT / 'analysis/paired_deltas_vs_full.csv', paired)

    summary_mode = []
    for mode in MODES:
        group = [r for r in rows if r['mode'] == mode]
        summary_mode.append({'mode': mode, 'runs': len(group),
                             **{metric + '_mean': mean(r.get(metric) for r in group)
                                for metric in METRIC_COLS}})
    write_csv(OUT_ROOT / 'analysis/summary_by_mode.csv', summary_mode)

    config = {
        'run_mode': args.mode,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'version': 'n2_cpu_v5_c2_aligned_40seed',
        'result_policy': 'overwrite',
        'modes': list(MODES),
        'seeds': SEEDS_SMOKE if args.mode == 'smoke' else SEEDS_FULL,
        'seed_count': len(SEEDS_SMOKE if args.mode == 'smoke' else SEEDS_FULL),
        'pool_sizes': list(POOL_SMOKE if args.mode == 'smoke' else POOL_FULL),
        'loads': [list(x) for x in ((('mid', 1.25),) if args.mode == 'smoke' else LOADS)],
        'deadlines': [list(x) for x in ((('base', 0.80),) if args.mode == 'smoke' else DEADLINES)],
        'strict_probs': [list(x) for x in ((('strict25', 0.25),) if args.mode == 'smoke' else STRICT_PROBS)],
        'horizon': HORIZON,
        'deadline_anchor': 'median',
        'deadline_factors': list(DEADLINE_FACTORS),
        'deadline_jitter': list(DEADLINE_JITTER),
        'risk_beta': 0.80,
        'scenario_params_c2_aligned': {
            'runtime_anomaly_rate': C2_ANOMALY_RATE,
            'runtime_anomaly_slowdown': C2_ANOMALY_SLOWDOWN,
            'burst_multiplier': C2_BURST,
        },
        'scenario_params_previous_wrong': {
            'runtime_anomaly_rate': 0.0,
            'runtime_anomaly_slowdown': 2.0,
            'burst_multiplier': 2.5,
        },
        'gate_rule': 'borrow hierarchical state only while the target context/device cell has zero updates; otherwise use own EMA state',
        'note': 'All modes use EventSimulator strategy full_qos; only OnlineCalibrator predictor state changes. Scenario params aligned to C2 main table.',
        'acceptance': 'full mode over seeds 601-620 must match C2 per_seed_results.csv full_qos cell-by-cell (|delta|<1e-9).',
    }
    (OUT_ROOT / 'report').mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f'report/config_{args.mode}.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({'run_mode': args.mode, 'rows': len(rows),
                      'summary_rows': len(summaries), 'paired_rows': len(paired),
                      'elapsed_min': round((time.time() - t0) / 60, 1)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
