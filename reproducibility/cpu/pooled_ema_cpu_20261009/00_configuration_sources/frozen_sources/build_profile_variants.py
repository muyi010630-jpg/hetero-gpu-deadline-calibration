# -*- coding: utf-8 -*-
"""N9 画像变体生成器 —— 检验机制一（分层借力）是否依赖「族级常数偏差」这一设定。

问题
----
现有主表画像的 `initial_s = true_s * DEFAULT_BIASES[device][family]`，
残差 `log(true/initial)` 在（设备型号, 任务族）内是**常数**（组内 sd ≤ 1e-08）。
而分层借力 = "同族内跨上下文借力"，族内残差完全相同 ⇒ 必然一步命中。

设计（把「总偏差幅度」固定，只改「偏差的相关结构」）
--------------------------------------------------
  P0  完全相关：log比 = -log(bias[device][family])        （9 个常数，= 现状）
  P1  完全独立：log比 ~ N(0, sd0)，逐上下文独立
  P2  部分相关：0.7071×(-log bias) + N(0, 0.7071×sd0)      （一半方差共享、一半独立）
  P3  完全真实：initial = 该上下文某一次真实测量, true = 全部测量中位数
                （无任何合成；幅度由真实测量噪声决定，仅作参考上界）

P1/P2 的幅度都标定到与 P0 相同的 total sd，因此**唯一变量是相关结构**。
所有随机数用固定 seed，保证可复现。
"""

from __future__ import annotations

import csv
import collections
import math
import random
import statistics
from pathlib import Path

ROOT = Path(r'$WORKSPACE')
RAW_BASE = ROOT / 'EMA/性能画像实验'
REF_DIR = ROOT / '修补实验2_deadline重设_20260907/experiment_C2_replay_deadline_fixed/profiles_expanded'
OUT_BASE = Path(__file__).resolve().parent.parent   # .../N9_profile_robustness
SEED = 20260921

DEVS = [('01_RTX5070', 'RTX5070', 11.94), ('02_T4', 'T4', 15.00), ('03_A100', 'A100', 40.00)]

DEFAULT_BIASES = {
    'RTX5070': {'compute': 1.35, 'memory': 0.72, 'inference': 1.22},
    'T4': {'compute': 0.82, 'memory': 1.22, 'inference': 0.90},
}
FALLBACK_BIAS = 1.10

POOL_MIX = {
    2: {'RTX5070': 1, 'T4': 1},
    4: {'RTX5070': 1, 'T4': 2, 'A100': 1},
    8: {'RTX5070': 2, 'T4': 3, 'A100': 3},
    16: {'RTX5070': 5, 'T4': 6, 'A100': 5},
    32: {'RTX5070': 10, 'T4': 12, 'A100': 10},
}
FAMMAP = {'gemm': 'compute', 'saxpy': 'memory', 'cifar10_resnet18': 'inference'}


def family_to_context(row):
    fam = FAMMAP.get(row['family'], row['family'])
    if fam == 'compute':
        return fam, int(row['size']), int(row['batch']), ('fp16' if row['dtype'] == 'float16' else 'fp32')
    if fam == 'memory':
        return fam, int(row['size']), int(row['batch']), 'fp32'
    return fam, int(row['size']), int(row['batch']), 'fp16'


def load_measurements():
    out = collections.defaultdict(list)
    for devdir, devname, _ in DEVS:
        p = RAW_BASE / devdir / 'benchmark_raw.csv'
        for r in csv.DictReader(p.open(encoding='utf-8-sig')):
            if r.get('success', 'True') != 'True' or r.get('mode') != 'gpu':
                continue
            out[(devname, family_to_context(r))].append((int(r['sample_index']), float(r['gpu_ms']) / 1000.0))
    return out


def bias_of(device, family):
    return DEFAULT_BIASES.get(device, {}).get(family, FALLBACK_BIAS)


def build_rows(meas, variant):
    """返回 [{device, family, size, batch, dtype, initial_s, true_s, logratio}]"""
    keys = sorted(meas.keys(), key=lambda k: (k[0], k[1]))
    # 先算 P0 的 log比（= -log bias），用于标定 sd0
    p0_log = {k: -math.log(bias_of(k[0], k[1][0])) for k in keys}
    sd0 = statistics.pstdev(list(p0_log.values()))

    rng = random.Random(SEED)
    rows = []
    for k in keys:
        device, ctx = k
        fam, size, batch, dtype = ctx
        samples = sorted(meas[k])
        vals = [v for _, v in samples]
        true_s = statistics.median(vals)

        if variant == 'P0':
            logratio = p0_log[k]
        elif variant == 'P1':
            logratio = rng.gauss(0.0, sd0)
        elif variant == 'P2':
            logratio = 0.7071067811865476 * p0_log[k] + rng.gauss(0.0, 0.7071067811865476 * sd0)
        elif variant == 'P3':
            steady = [v for i, v in samples if i >= 1] or vals
            single = steady[rng.randrange(len(steady))]
            logratio = math.log(true_s / single)
        else:
            raise ValueError(variant)

        initial_s = true_s * math.exp(-logratio)
        rows.append({'device': device, 'family': fam, 'size': size, 'batch': batch,
                     'dtype': dtype, 'initial_s': round(initial_s, 12),
                     'true_s': round(true_s, 12), 'logratio': logratio})
    return rows


def expand(rows, pool_size):
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r['device']].append(r)
    mem = {d[1]: d[2] for d in DEVS}
    out = []
    for device_type, count in POOL_MIX[pool_size].items():
        for idx in range(1, count + 1):
            vname = f'{device_type}__{idx:02d}'
            for r in by_type[device_type]:
                out.append({'device': vname, 'vendor': 'NVIDIA', 'memory_gb': mem[device_type],
                            'family': r['family'], 'size': r['size'], 'batch': r['batch'],
                            'dtype': r['dtype'], 'initial_s': r['initial_s'], 'true_s': r['true_s']})
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['device', 'vendor', 'memory_gb', 'family',
                                          'size', 'batch', 'dtype', 'initial_s', 'true_s'])
        w.writeheader()
        w.writerows(rows)


def main():
    meas = load_measurements()
    print(f'真实测量上下文数: {len(meas)}')
    print()

    # 校验：自造 P0 应与现有 profiles_expanded 一致
    p0_rows = build_rows(meas, 'P0')
    ref = list(csv.DictReader((REF_DIR / 'cluster_08_profile.csv').open(encoding='utf-8-sig')))
    ref_map = {(r['device'].split('__')[0], r['family'], int(r['size']), int(r['batch']), r['dtype']): r
               for r in ref}
    maxd_init = maxd_true = 0.0
    for r in p0_rows:
        k = (r['device'], r['family'], r['size'], r['batch'], r['dtype'])
        if k in ref_map:
            maxd_init = max(maxd_init, abs(float(ref_map[k]['initial_s']) - r['initial_s']))
            maxd_true = max(maxd_true, abs(float(ref_map[k]['true_s']) - r['true_s']))
    print(f'[校验] 自造 P0 vs 现有 profiles_expanded: max|Δ initial_s|={maxd_init:.3e}  '
          f'max|Δ true_s|={maxd_true:.3e}  -> {"一致" if maxd_true < 1e-9 else "不一致(需查)"}')
    print()

    print(f'{"variant":<8}{"组内 sd(log比)":>17}{"跨上下文 sd":>14}{"|log比|中位":>13}{"|log比|P95":>12}')
    print('-' * 66)
    for variant in ('P0', 'P1', 'P2', 'P3'):
        rows = build_rows(meas, variant)
        if variant == 'P0':
            # 直接复制论文主表画像，保证 bit-identical
            # （自造版本与主表差 5e-10，会在仿真临界点改变 tie-break，造成 0.3% 的 cell 分叉）
            import shutil
            for pool_size in POOL_MIX:
                dst = OUT_BASE / f'profiles_{variant}/cluster_{pool_size:02d}_profile.csv'
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REF_DIR / f'cluster_{pool_size:02d}_profile.csv', dst)
        else:
            for pool_size in POOL_MIX:
                write_csv(OUT_BASE / f'profiles_{variant}/cluster_{pool_size:02d}_profile.csv',
                          expand(rows, pool_size))
        grp = collections.defaultdict(list)
        for r in rows:
            grp[(r['device'], r['family'])].append(r['logratio'])
        within = [statistics.pstdev(v) for v in grp.values() if len(v) > 1]
        lr = [r['logratio'] for r in rows]
        srt = sorted(abs(x) for x in lr)
        p95 = srt[int(0.95 * (len(srt) - 1))]
        print(f'{variant:<8}{statistics.fmean(within) if within else 0:>17.3e}'
              f'{statistics.pstdev(lr):>14.5f}'
              f'{statistics.median(abs(x) for x in lr)*100:>12.2f}%'
              f'{p95*100:>11.2f}%')

    print()
    print('说明：P0/P1/P2 的跨上下文 sd 已对齐（同一总幅度，只有相关结构不同）；')
    print('      P3 的幅度由真实测量噪声决定，不参与标定。')


if __name__ == '__main__':
    main()
