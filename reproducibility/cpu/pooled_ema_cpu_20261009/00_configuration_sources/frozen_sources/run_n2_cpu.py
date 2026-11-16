from __future__ import annotations
import argparse, csv, json, math, statistics, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import hetero_qos_sim_n2 as sim

ROOT = Path('$WORKSPACE')
# 输出根 = 本脚本所在 scripts/ 的上一级（即 N2_scheduler_baselines/），与目录位置解耦
OUT_ROOT = SCRIPT_DIR.parent
PROFILE_ROOT = ROOT / '修补实验2_deadline重设_20260907/experiment_C2_replay_deadline_fixed/profiles_expanded'

MODES = ('full', 'gated', 'window', 'kalman')
SEEDS_FULL = list(range(601, 641))
SEEDS_SMOKE = list(range(601, 606))
POOL_FULL = (2, 4, 8, 16, 32)
POOL_SMOKE = (8,)
LOADS = (('low', 0.80), ('mid', 1.25), ('high', 1.70))
DEADLINES = (('loose', 1.00), ('base', 0.80), ('tight', 0.64))
STRICT_PROBS = (('strict10', 0.10), ('strict25', 0.25), ('strict40', 0.40))
DEADLINE_FACTORS = (1.60, 1.30, 1.10)
DEADLINE_JITTER = (0.92, 1.08)
HORIZON = 12.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class N2Calibrator(sim.OnlineCalibrator):
    mode = 'full'

    def __init__(self, base, strategy, *args, **kwargs):
        super().__init__(base, 'full_qos', *args, **kwargs)
        self.n2_mode = type(self).mode
        self.kalman_p = {}

    def _custom_state(self, context, device):
        key = (context, device)
        cell = self._cell(key)
        n = len(cell.samples)
        if self.n2_mode == 'window':
            if n < self.min_samples:
                return 0.0, 0.25 / math.sqrt(n + 1.0)
            sample = cell.samples[-self.window:]
            z = statistics.mean(sample)
            sigma = statistics.pstdev(sample) if len(sample) > 1 else 0.25 / math.sqrt(n + 1.0)
            return clamp(z, self.z_min, self.z_max), max(sigma, 0.25 / math.sqrt(n + 1.0))
        p = self.kalman_p.get(key, 0.0625)
        z = cell.zhat
        sigma = 0.25 if n == 0 else math.sqrt(min(0.50, max(1e-9, p + 0.004)))
        return clamp(z, self.z_min, self.z_max), sigma

    def predict(self, context, device, qos, guarded=True):
        if self.n2_mode == 'full':
            return super().predict(context, device, qos, guarded=guarded)
        key = (context, device)
        base = self.base[key]
        if self.n2_mode == 'gated':
            # Borrow the hierarchical state only for a genuinely cold cell.
            # Once a cell has its own observation, use its own EMA state and
            # uncertainty so that a misleading neighbor cannot keep shrinking
            # the estimate after cold start.
            cell = self._cell(key)
            if len(cell.samples) == 0:
                return super().predict(context, device, qos, guarded=guarded)
            else:
                z = clamp(cell.zhat, self.z_min, self.z_max)
                sigma = max(cell.sigma, 0.25 / math.sqrt(len(cell.samples) + 1.0))
                guard = self.risk_beta * (1.0 + 0.35 * qos) * sigma if guarded and qos >= 1 else 0.0
                return base * math.exp(clamp(z + guard, self.z_min, self.z_max)), sigma
        z, sigma = self._custom_state(context, device)
        guard = 0.0
        if guarded and qos >= 1:
            guard = self.risk_beta * (1.0 + 0.35 * qos) * sigma
        value = clamp(z + guard, self.z_min, self.z_max) if guarded else z
        return base * math.exp(value), sigma

    def observe(self, context, device, observed, predicted, now):
        if self.n2_mode in ('full', 'gated'):
            return super().observe(context, device, observed, predicted, now)
        if predicted > 0:
            err = abs(predicted - observed)
            self.prediction_errors.append(err)
            self.prediction_abs_pct.append(err / observed if observed else 0.0)
        key = (context, device)
        base = self.base[key]
        z = clamp(math.log(max(observed, 1e-9) / max(base, 1e-9)), self.z_min, self.z_max)
        cell = self._cell(key)
        cell.samples.append(z)
        if len(cell.samples) > self.window:
            cell.samples.pop(0)
        if self.n2_mode == 'window':
            if len(cell.samples) >= self.min_samples:
                sample = cell.samples[-self.window:]
                cell.zhat = clamp(statistics.mean(sample), self.z_min, self.z_max)
                cell.sigma = min(0.70, statistics.pstdev(sample) if len(sample) > 1 else 0.0)
                cell.updates += 1
            cell.last_update = now
            return
        q = 0.004
        r = {'compute': 0.020, 'memory': 0.018, 'inference': 0.035}[context[0]]
        p = self.kalman_p.get(key, 0.0625)
        p_pred = max(1e-9, p + q)
        k = p_pred / (p_pred + r)
        cell.zhat = clamp(cell.zhat + k * (z - cell.zhat), self.z_min, self.z_max)
        p_new = max(1e-9, (1.0 - k) * p_pred)
        self.kalman_p[key] = p_new
        cell.sigma = math.sqrt(p_new)
        cell.updates += 1
        cell.last_update = now


sim.OnlineCalibrator = N2Calibrator


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.fmean(xs) if xs else float('nan')


def sd(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def profile_for(pool_size):
    path = PROFILE_ROOT / f'cluster_{pool_size:02d}_profile.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run_one(mode, seed, pool_size, load_name, load, deadline_name, deadline_scale, strict_name, strict_prob):
    N2Calibrator.mode = mode
    profile = str(profile_for(pool_size))
    scenario = sim.Scenario(
        __import__('random').Random(seed), HORIZON, pool_size * load, True,
        profile, 1.0, 1.0, strict_prob, deadline_scale, 0.0, 2.0, 2.5,
        'median', 1.0, DEADLINE_JITTER[0], DEADLINE_JITTER[1], 0.0, 0.0,
        DEADLINE_FACTORS,
    )
    runner = sim.EventSimulator(
        scenario, 'full_qos', seed, 0.0, 24, 4, 0.30, 0.20, 0.0,
        1.0, 0.5, 0.22, 1.20, 1.5, 0.65, 0.080, 0.015, 8,
        0.80, 0.30,
    )
    result = runner.run()
    result.update({
        'mode': mode, 'pool_size': pool_size, 'load': load_name,
        'load_per_device': load, 'deadline_setting': deadline_name,
        'deadline_scale': deadline_scale, 'strict_setting': strict_name,
        'strict_qos_prob': strict_prob, 'profile_csv': profile,
        'strategy_label': {'full': 'Full-QoS', 'gated': 'Gated-Full-QoS', 'window': 'Window-QoS', 'kalman': 'Kalman-QoS'}[mode],
    })
    result['strategy'] = mode
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    args = parser.parse_args()
    seeds = SEEDS_SMOKE if args.mode == 'smoke' else SEEDS_FULL
    pools = POOL_SMOKE if args.mode == 'smoke' else POOL_FULL
    out = OUT_ROOT / ('raw/smoke' if args.mode == 'smoke' else 'raw/full')
    rows = []
    total = len(MODES) * len(seeds) * len(pools) * len(LOADS) * len(DEADLINES) * len(STRICT_PROBS)
    done = 0
    for pool_size in pools:
        for load_name, load in LOADS if args.mode == 'full' else (('mid', 1.25),):
            for deadline_name, deadline_scale in DEADLINES if args.mode == 'full' else (('base', 0.80),):
                for strict_name, strict_prob in STRICT_PROBS if args.mode == 'full' else (('strict25', 0.25),):
                    for seed in seeds:
                        for mode in MODES:
                            rows.append(run_one(mode, seed, pool_size, load_name, load, deadline_name, deadline_scale, strict_name, strict_prob))
                            done += 1
                    print(f'completed {done}/{total}: pool={pool_size} load={load_name} deadline={deadline_name} strict={strict_name}', flush=True)
    write_csv(out / 'per_seed_results.csv', rows)

    key_cols = ['pool_size', 'load', 'deadline_setting', 'strict_setting']
    metric_cols = ['prediction_mape', 'sla_violation_rate', 'strict_qos_violation_rate', 'qos_bound_coverage', 'false_safe_rate', 'p95_completion', 'p99_completion', 'jain_fairness', 'window_jain_fairness_mean', 'device_regret']
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(k, '')) for k in key_cols) + (row['mode'],)].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        record = {k: v for k, v in zip(key_cols, key[:len(key_cols)])}
        record['mode'] = key[-1]; record['seeds'] = len(group)
        for metric in metric_cols:
            record[metric + '_mean'] = mean(r.get(metric) for r in group)
            record[metric + '_std'] = sd(r.get(metric) for r in group)
        summaries.append(record)
    write_csv(OUT_ROOT / 'analysis/summary_by_scenario.csv', summaries)

    paired = []
    index = {(tuple(str(r.get(k, '')) for k in key_cols), str(r.get('seed', '')), r['mode']): r for r in rows}
    scenarios = sorted({tuple(str(r.get(k, '')) for k in key_cols) for r in rows})
    for scenario in scenarios:
        seeds_here = sorted({str(r.get('seed', '')) for r in rows if tuple(str(r.get(k, '')) for k in key_cols) == scenario})
        for seed in seeds_here:
            full = index.get((scenario, seed, 'full'))
            if not full:
                continue
            for mode in ('window', 'kalman'):
                other = index.get((scenario, seed, mode))
                if not other:
                    continue
                for metric in metric_cols:
                    paired.append({**{k: v for k, v in zip(key_cols, scenario)}, 'seed': seed, 'mode': mode, 'baseline': 'full', 'metric': metric, 'delta_mode_minus_full': float(other[metric]) - float(full[metric])})
    write_csv(OUT_ROOT / 'analysis/paired_deltas_vs_full.csv', paired)

    summary_mode = []
    for mode in MODES:
        group = [r for r in rows if r['mode'] == mode]
        summary_mode.append({'mode': mode, 'runs': len(group), **{metric + '_mean': mean(r.get(metric) for r in group) for metric in metric_cols}})
    write_csv(OUT_ROOT / 'analysis/summary_by_mode.csv', summary_mode)
    config = {'run_mode': args.mode, 'generated_at': datetime.now().isoformat(timespec='seconds'), 'version': 'n2_cpu_v4_gated_reproducible', 'result_policy': 'overwrite', 'modes': MODES, 'seeds': seeds, 'seed_count': len(seeds), 'pool_sizes': pools, 'loads': LOADS, 'deadlines': DEADLINES, 'strict_probs': STRICT_PROBS, 'horizon': HORIZON, 'deadline_anchor': 'median', 'deadline_factors': DEADLINE_FACTORS, 'risk_beta': 0.80, 'gate_rule': 'borrow hierarchical state only while the target context/device cell has zero updates; otherwise use own EMA state', 'note': 'All modes use EventSimulator strategy full_qos; only OnlineCalibrator predictor state changes for gated/window/kalman.'}
    (OUT_ROOT / 'report').mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f'report/config_{args.mode}.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'run_mode': args.mode, 'rows': len(rows), 'summary_rows': len(summaries), 'paired_rows': len(paired)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
