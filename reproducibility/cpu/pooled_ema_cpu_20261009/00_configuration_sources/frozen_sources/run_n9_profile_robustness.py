# -*- coding: utf-8 -*-
"""N9 机制一鲁棒性检验 —— 在四种画像偏差结构下重跑 fixed / hier_qos / full_qos。

问题
----
机制一（分层借力）的收益是否依赖「画像偏差是族级常数」这一设定？
本实验把**总偏差幅度固定**，只改变**偏差的相关结构**，观察 fixed→hier→full 链条的变化。

画像变体（由 build_profile_variants.py 生成）
  P0 完全相关（现状）: 组内 sd = 0
  P1 完全独立        : 逐上下文独立
  P2 部分相关        : 一半共享 + 一半独立
  P3 完全真实        : initial = 单次实测, true = 中位数（幅度小，仅作参考）

核心读数
  d(hier - fixed) 与 d(full - fixed) 在各变体下的 SLA / MAPE 差
  —— P0 下应有明显增益；P1 下若归零，则证明机制一依赖族级共享偏差。
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

ROOT = Path(r'$WORKSPACE')
_HERE = Path(__file__).resolve().parent          # .../N9_profile_robustness/scripts
OUT = _HERE.parent                                # .../N9_profile_robustness
PROFILE_BASE = OUT
N2_SCRIPTS = _HERE.parent.parent / 'N2_scheduler_baselines' / 'scripts'
sys.path.insert(0, str(N2_SCRIPTS))
import hetero_qos_sim_n2 as sim  # noqa: E402

VARIANTS = ('P0', 'P1', 'P2', 'P3')
STRATEGIES = ('fixed', 'hier_qos', 'full_qos')
SEEDS = list(range(601, 641))
POOLS = (2, 4, 8, 16, 32)
LOADS = (('low', 0.80), ('mid', 1.25), ('high', 1.70))
DEADLINE_SCALE = 0.80
STRICT_PROB = 0.25

C2_ANOMALY_RATE = 0.006
C2_ANOMALY_SLOWDOWN = 1.60
C2_BURST = 2.00
DEADLINE_FACTORS = (1.60, 1.30, 1.10)
DEADLINE_JITTER = (0.92, 1.08)
HORIZON = 12.0

# 与 C2 主表 STRATEGY_PARAMS 完全一致（缺省即模拟器默认值）
PARAMS = {
    'fixed': dict(ema_rho=0.30, kappa_qos=0.0, hier_tau=4.0,
                  risk_beta=1.35, cold_uncertainty=0.30),
    'hier_qos': dict(ema_rho=0.30, kappa_qos=1.30, hier_tau=1.5,
                     risk_beta=1.35, cold_uncertainty=0.30),
    'full_qos': dict(ema_rho=0.30, kappa_qos=0.0, hier_tau=1.5,
                     risk_beta=0.80, cold_uncertainty=0.30),
}

METRICS = ['sla_violation_rate', 'strict_qos_violation_rate', 'prediction_mape',
           'qos_bound_coverage', 'false_safe_rate', 'mean_completion',
           'p95_completion', 'p99_completion', 'window_jain_fairness_mean',
           'min_user_service_share', 'device_regret']


def run_unit(task):
    variant, pool, load_name, load, seed = task
    profile = str(PROFILE_BASE / f'profiles_{variant}' / f'cluster_{pool:02d}_profile.csv')
    rows = []
    for strategy in STRATEGIES:
        p = PARAMS[strategy]
        sc = sim.Scenario(random.Random(seed), HORIZON, pool * load, True, profile,
                          1.0, 1.0, STRICT_PROB, DEADLINE_SCALE,
                          C2_ANOMALY_RATE, C2_ANOMALY_SLOWDOWN, C2_BURST,
                          'median', 1.0, DEADLINE_JITTER[0], DEADLINE_JITTER[1],
                          0.0, 0.0, DEADLINE_FACTORS)
        runner = sim.EventSimulator(sc, strategy, seed, 0.0, 24, 4,
                                    p['ema_rho'], 0.20, p['kappa_qos'], 1.0, 0.5, 0.22,
                                    1.20, p['hier_tau'], 0.65, 0.080, 0.015, 8,
                                    p['risk_beta'], p['cold_uncertainty'])
        result = runner.run()
        result.update({'variant': variant, 'strategy': strategy, 'pool_size': pool,
                       'load': load_name, 'load_per_device': load, 'seed': seed,
                       'profile_csv': profile})
        rows.append(result)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def avg(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.fmean(xs) if xs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()

    tasks = [(v, pool, ln, lv, seed) for v in VARIANTS for pool in POOLS
             for ln, lv in LOADS for seed in SEEDS]
    print(f'[N9] variants={VARIANTS} tasks={len(tasks)} workers={args.workers}', flush=True)
    t0 = time.time()
    rows = []
    done = 0
    with Pool(processes=args.workers) as pool:
        for unit in pool.imap_unordered(run_unit, tasks, chunksize=1):
            rows.extend(unit)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f'[N9] {done}/{len(tasks)} elapsed={el/60:.1f}min '
                      f'eta={el/done*(len(tasks)-done)/60:.1f}min', flush=True)
    print(f'[N9] sim done in {(time.time()-t0)/60:.1f} min rows={len(rows)}', flush=True)

    vo = {v: i for i, v in enumerate(VARIANTS)}
    so = {s: i for i, s in enumerate(STRATEGIES)}
    lo = {'low': 0, 'mid': 1, 'high': 2}
    rows.sort(key=lambda r: (vo[r['variant']], int(r['pool_size']), lo[r['load']],
                             int(r['seed']), so[r['strategy']]))
    write_csv(OUT / 'raw/per_seed_results.csv', rows)

    summary = []
    for v in VARIANTS:
        for strategy in STRATEGIES:
            g = [r for r in rows if r['variant'] == v and r['strategy'] == strategy]
            summary.append({'variant': v, 'strategy': strategy, 'runs': len(g),
                            **{m + '_mean': avg(r[m] for r in g) for m in METRICS}})
    write_csv(OUT / 'analysis/summary_by_variant_strategy.csv', summary)

    # 逐 seed 配对增益：hier-full / fixed 链
    paired = []
    idx = {(r['variant'], int(r['pool_size']), r['load'], int(r['seed']), r['strategy']): r
           for r in rows}
    for v in VARIANTS:
        for pool in POOLS:
            for ln, _ in LOADS:
                for seed in SEEDS:
                    base = idx.get((v, pool, ln, seed, 'fixed'))
                    hier = idx.get((v, pool, ln, seed, 'hier_qos'))
                    full = idx.get((v, pool, ln, seed, 'full_qos'))
                    if not (base and hier and full):
                        continue
                    rec = {'variant': v, 'pool_size': pool, 'load': ln, 'seed': seed}
                    for m in METRICS:
                        rec[f'hier_minus_fixed_{m}'] = float(hier[m]) - float(base[m])
                        rec[f'full_minus_fixed_{m}'] = float(full[m]) - float(base[m])
                        rec[f'full_minus_hier_{m}'] = float(full[m]) - float(hier[m])
                    paired.append(rec)
    write_csv(OUT / 'analysis/paired_gains_by_variant.csv', paired)

    # 核心读数：各变体下机制一增益
    core = []
    for v in VARIANTS:
        g = [r for r in paired if r['variant'] == v]
        rec = {'variant': v, 'pairs': len(g)}
        for m in ('sla_violation_rate', 'prediction_mape', 'qos_bound_coverage',
                  'p95_completion', 'min_user_service_share'):
            hv = [r[f'hier_minus_fixed_{m}'] for r in g]
            fv = [r[f'full_minus_fixed_{m}'] for r in g]
            rec[f'hier_minus_fixed_{m}_mean'] = avg(hv)
            rec[f'full_minus_fixed_{m}_mean'] = avg(fv)
            rec[f'hier_better_share_{m}_pct'] = (
                sum(x < 0 for x in hv) / len(hv) * 100.0 if m in
                ('sla_violation_rate', 'prediction_mape', 'p95_completion') else
                sum(x > 0 for x in hv) / len(hv) * 100.0)
        core.append(rec)
    write_csv(OUT / 'analysis/core_mechanism1_gain.csv', core)

    write_csv(OUT / 'analysis/summary_by_mode.csv', summary)

    meta = {
        'experiment': 'N9 profile-bias structure robustness (mechanism-1 dependence test)',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'version': 'n9_cpu_v1',
        'variants': list(VARIANTS),
        'variant_definition': {
            'P0': 'log(true/initial) = -log(bias[dtype][family]); within-group sd = 0 (current setup)',
            'P1': 'log(true/initial) ~ N(0, sd0), independent per context',
            'P2': '0.7071*(-log bias) + N(0, 0.7071*sd0) — half shared, half independent',
            'P3': 'initial = one real measurement of that context, true = median of all (fully real)',
            'sd0': 'cross-context sd of -log(bias) in P0 = 0.19699',
        },
        'strategies': list(STRATEGIES),
        'seeds': SEEDS, 'seed_count': len(SEEDS),
        'pools': list(POOLS), 'loads': [list(x) for x in LOADS],
        'deadline_scale': DEADLINE_SCALE, 'strict_prob': STRICT_PROB,
        'scenario_params': {'runtime_anomaly_rate': C2_ANOMALY_RATE,
                            'runtime_anomaly_slowdown': C2_ANOMALY_SLOWDOWN,
                            'burst_multiplier': C2_BURST},
        'note': 'Total bias magnitude is held fixed across P0/P1/P2; only the correlation '
                'structure changes. P3 is fully empirical and much smaller in magnitude.',
    }
    (OUT / 'report').mkdir(parents=True, exist_ok=True)
    (OUT / 'report/metadata.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({'rows': len(rows), 'paired_rows': len(paired),
                      'elapsed_min': round((time.time() - t0) / 60, 1)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
