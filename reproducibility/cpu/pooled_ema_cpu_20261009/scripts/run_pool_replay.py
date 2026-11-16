from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

from pool_adapter import ROOT, MODES, make_runner, task_fingerprint
from replay_statistics import METRICS, analyze, write_csv

CONFIG = ROOT / '00_configuration_sources'
STAGES = {'replay': '02_scheduling_replay', 'structure': '03_mismatch_structure'}
LOADS = (('low', .80), ('mid', 1.25), ('high', 1.70))
DEADLINES = (('loose', 1.00), ('base', .80), ('tight', .64))
STRICT = (('strict10', .10), ('strict25', .25), ('strict40', .40))
GOLDEN = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def load_golden(stage):
    if stage in GOLDEN: return GOLDEN[stage]
    idx = {}
    with (CONFIG / 'golden' / ('n2_per_seed_results.csv' if stage == 'replay' else 'n9_per_seed_results.csv')).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['strategy'] not in ('full', 'full_qos'): continue
            variant = row.get('variant', 'main')
            key = (variant, int(row['pool_size']), row['load'], row.get('deadline_setting', 'base'), row.get('strict_setting', 'strict25'), int(row['seed']))
            idx[key] = row
    GOLDEN[stage] = idx
    return idx


def build_tasks(stage, run_mode):
    seeds = range(601, 641) if run_mode == 'full' else (601, 602)
    tasks = []
    if stage == 'replay':
        cells = [(p, ln, lv, dn, dv, sn, sv) for p in (2,4,8,16,32) for ln,lv in LOADS for dn,dv in DEADLINES for sn,sv in STRICT]
        if run_mode == 'smoke':
            cells = [(2,'high',1.70,'tight',.64,'strict40',.40),
                     (8,'mid',1.25,'base',.80,'strict25',.25),
                     (32,'low',.80,'loose',1.00,'strict10',.10)]
        for cell in cells:
            for seed in seeds: tasks.append((stage,run_mode,'main',*cell,seed))
    else:
        pools = (2,4,8,16,32) if run_mode == 'full' else (8,)
        loads = LOADS if run_mode == 'full' else (('mid',1.25),)
        for variant in ('P0','P1','P2','P3'):
            for p in pools:
                for ln,lv in loads:
                    for seed in seeds: tasks.append((stage,run_mode,variant,p,ln,lv,'base',.80,'strict25',.25,seed))
    return tasks


def unit_id(task):
    _, _, v, p, ln, _, dn, _, sn, _, seed = task
    return f'{v}_pool{p:02d}_{ln}_{dn}_{sn}_seed{seed}'


def stage_path(stage, run_mode):
    return ROOT / STAGES[stage] / run_mode


def run_unit(task):
    stage, run_mode, variant, pool, ln, lv, dn, dv, sn, sv, seed = task
    output = stage_path(stage, run_mode)
    profile_dir = 'profiles_replay' if stage == 'replay' else f'profiles_{variant}'
    profile = CONFIG / profile_dir / f'cluster_{pool:02d}_profile.csv'
    key = variant,pool,ln,dn,sn,seed
    golden = load_golden(stage)[key]
    rows = []; plan_hashes = []; delta_max = 0.0; golden_metrics = []
    representative = seed == 601 and (run_mode == 'smoke' or (pool == 8 and ln == 'mid' and dn == 'base' and sn == 'strict25'))
    t0 = time.perf_counter()
    for mode in MODES:
        runner = make_runner(profile, mode, seed, pool, lv, dv, sv)
        plan_hash = task_fingerprint(runner.tasks)
        plan_hashes.append(plan_hash)
        potential = [dict(task_id=t.task_id, user=t.user, family=t.context[0], size=t.context[1], batch=t.context[2], dtype=t.context[3],
                          arrival=t.arrival, qos=t.qos, deadline=t.deadline, potential_service_times=json.dumps(t.true_times, sort_keys=True)) for t in runner.tasks] if representative else []
        result = runner.run()
        if mode == 'full':
            for metric in (*METRICS, 'arrived', 'completed', 'prediction_mae', 'throughput', 'probe_count', 'drift_alarm_count'):
                error = abs(float(result[metric]) - float(golden[metric]))
                delta_max = max(delta_max, error); golden_metrics.append(metric)
                if error > 1e-9:
                    raise RuntimeError(f'Full golden failed {unit_id(task)} {metric}: error={error}')
        result.update(mode=mode, strategy=mode, variant=variant, pool_size=pool, load=ln, load_per_device=lv,
                      deadline_setting=dn, deadline_scale=dv, strict_setting=sn, strict_qos_prob=sv, seed=seed,
                      profile_csv=str(profile), profile_sha256=sha(profile), exogenous_plan_sha256=plan_hash,
                      point_group_count=len(runner.cal.group_z), risk_local_cell_count=len(runner.cal.cells))
        rows.append(result)
        if representative:
            trace = runner.trace
            for tr in trace:
                tr.update(mode=mode, variant=variant, pool_size=pool, load=ln, deadline_setting=dn, strict_setting=sn,
                          point_log_residual_at_dispatch=math.log(max(float(tr['predicted_s']),1e-12)/max(runner.scenario.initial_base[((tr['family'],tr['size'],tr['batch'],tr['dtype']),tr['device'])],1e-12)))
            write_csv(output / 'raw/representative_traces' / (unit_id(task) + f'_{mode}.csv'), trace)
            if mode == 'full': write_csv(output / 'raw/representative_plans' / (unit_id(task)+'.csv'), potential)
    if len(set(plan_hashes)) != 1: raise RuntimeError('Strategy-dependent exogenous stream ' + unit_id(task))
    payload = dict(unit_id=unit_id(task), stage=stage, complete=True, modes=list(MODES), rows=rows,
                   exogenous_plan_sha256=plan_hashes[0], all_modes_same_plan=True,
                   full_golden_max_abs_delta=delta_max, full_golden_metrics=golden_metrics,
                   elapsed_seconds=time.perf_counter()-t0)
    path = output / 'raw/shards' / (unit_id(task)+'.json')
    atomic_json(path,payload)
    return dict(unit_id=unit_id(task), full_golden_max_abs_delta=delta_max, elapsed_seconds=payload['elapsed_seconds'])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=STAGES,required=True)
    ap.add_argument('--mode',choices=('smoke','full'),default='smoke'); ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--analyze-only',action='store_true'); args=ap.parse_args()
    if not 1 <= args.workers <= 6: raise ValueError('Maximum six workers')
    tasks=build_tasks(args.stage,args.mode); output=stage_path(args.stage,args.mode);output.mkdir(parents=True,exist_ok=True)
    metadata=dict(stage=args.stage,run_mode=args.mode,modes=list(MODES),total_units=len(tasks),total_runs=len(tasks)*len(MODES),workers=args.workers,
                  seed_range=[601,640] if args.mode=='full' else [601,602], frozen_simulator_sha256=sha(CONFIG/'frozen_sources/hetero_qos_sim_n2.py'),
                  adapter_sha256=sha(Path(__file__).parent/'pool_adapter.py'), runner_sha256=sha(__file__), statistics_sha256=sha(Path(__file__).parent/'replay_statistics.py'),
                  full_params=dict(horizon=12.0,cal_window=24,min_samples=4,ema_rho=.30,kappa_qos=0.0,hier_tau=1.5,risk_beta=.80,cold_uncertainty=.30,
                                   anomaly_rate=.006,anomaly_slowdown=1.60,burst=2.00,alpha_by_family=dict(compute=.35,memory=.45,inference=.22)),
                  grouping='measured_device_type(device) × family; no precision split',
                  common_risk='frozen Full _effective_state including sigma, self_weight and cold bonus; local states retained for this common wrapper',
                  intrinsic_differences='pool updates once per completed group event; group mixes precision; Full has precision-preferred neighbors and cross-type fallback',
                  feedback='only selected completed device execution is observed; histories are endogenous and differ between strategies',
                  stream='all potential device service times generated before dispatch, same input seed; hashes checked every unit')
    config_path=output/'config.json'
    if config_path.exists():
        old=json.loads(config_path.read_text(encoding='utf-8'))
        for k in ('adapter_sha256','frozen_simulator_sha256','runner_sha256','statistics_sha256'):
            if old[k]!=metadata[k]: raise RuntimeError('Resume code fingerprint changed: '+k)
    else: atomic_json(config_path,metadata)
    todo=[]
    for task in tasks:
        path=output/'raw/shards'/(unit_id(task)+'.json')
        if path.exists():
            saved=json.loads(path.read_text(encoding='utf-8'))
            if saved['complete'] and saved['modes']==list(MODES): continue
        todo.append(task)
    print(json.dumps(dict(stage=args.stage,mode=args.mode,total_units=len(tasks),resume_skipped=len(tasks)-len(todo),todo=len(todo),workers=args.workers)),flush=True)
    started=time.perf_counter(); completed=len(tasks)-len(todo)
    if not args.analyze_only and todo:
        with Pool(processes=args.workers) as workers:
            for status in workers.imap_unordered(run_unit,todo,chunksize=1):
                completed+=1
                with (output/'run_log.jsonl').open('a',encoding='utf-8') as log: log.write(json.dumps(status,ensure_ascii=False)+'\n')
                if completed%100==0 or completed==len(tasks) or args.mode=='smoke':
                    elapsed=time.perf_counter()-started
                    print(json.dumps(dict(stage=args.stage,completed=completed,total=len(tasks),elapsed_seconds=round(elapsed,2),golden_max_delta=status['full_golden_max_abs_delta'])),flush=True)
    rows=[]; deltas=[]
    for task in tasks:
        path=output/'raw/shards'/(unit_id(task)+'.json')
        if not path.exists(): raise RuntimeError('Incomplete stage: '+unit_id(task))
        payload=json.loads(path.read_text(encoding='utf-8'));rows.extend(payload['rows']);deltas.append(payload['full_golden_max_abs_delta'])
    write_csv(output/'raw/per_run_results.csv',rows)
    aggregate,infer_rows=analyze(output,rows,args.stage)
    check=dict(complete=True,units=len(tasks),runs=len(rows),same_plan_all_units=True,full_golden_max_abs_delta=max(deltas),
               all_full_golden_pass=True,elapsed_current_invocation_seconds=time.perf_counter()-started,
               completed_at_unix=time.time())
    atomic_json(output/'completion.json',check)
    print(json.dumps(check,ensure_ascii=False),flush=True)
    print(json.dumps(aggregate,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
