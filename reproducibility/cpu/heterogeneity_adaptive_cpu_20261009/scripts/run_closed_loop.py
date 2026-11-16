"""Heldout matched scheduler comparisons with passive common prediction panels."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import calibrator_core as core
from profile_flows import build_image, load_base_contexts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '04_closed_loop_scheduling'
MODES = core.PRIMARY_MODES
TIMES = (.5, 1., 2., 4., 6., 8., 10., 12.)
METRICS = ('sla_violation_rate','strict_qos_violation_rate','prediction_mape',
           'panel_noisy_mape','panel_median_mape','qos_bound_coverage',
           'false_safe_rate','p95_completion','p99_completion','mean_wait')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def write_csv(path, rows):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temp, path)


def profile_path(image, rho):
    path = OUT/'profiles'/f'image{image}_rho{rho:.2f}_pool08.csv'
    if path.exists(): return path
    unique = build_image(image, rho)
    counts = {'RTX5070':2, 'T4':3, 'A100':3}
    rows = []
    for row in unique:
        for i in range(1, counts[row['model']] + 1):
            rows.append(dict(device=f"{row['model']}__{i:02d}",vendor=row['vendor'],
                             memory_gb=row['memory_gb'], family=row['family'],
                             size=row['size'],batch=row['batch'],dtype=row['dtype'],
                             initial_s=row['initial_s'],true_s=row['true_s']))
    write_csv(path, rows)
    return path


def make_scenario(path, seed, load):
    return core.sim.Scenario(random.Random(seed),12.,8*load,False,str(path),1.,1.,.25,.80,
                             0.,1.60,2.,'median',1.,.92,1.08,0.,0.,(1.60,1.30,1.10))


class CurrentTimeSimulator(core.sim.EventSimulator):
    def _best_predicted_time(self, task, devices):
        self.cal.set_now(self.env.now)
        compatible = self._compatible(task, devices)
        return min((self.cal.predict(task.context,d.spec.name,task.qos,guarded=True)[0]
                    for d in compatible), default=float('inf'))

    def _mean_prediction(self, task, device):
        self.cal.set_now(self.env.now)
        return self.cal.predict(task.context,device.spec.name,task.qos,guarded=False)

    def _guarded_prediction(self, task, device):
        self.cal.set_now(self.env.now)
        return self.cal.predict(task.context,device.spec.name,task.qos,guarded=True)


class PanelSimulator(CurrentTimeSimulator):
    def _panel_process(self):
        previous = 0.
        for j, now in enumerate(TIMES):
            yield self.env.timeout(now-previous); previous = now
            # Passive evaluation of a copied state cannot affect dispatch/history.
            evaluator = copy.deepcopy(self.cal); evaluator.set_now(now)
            for i, (context, device) in enumerate(self.panel_keys):
                self.panel_predictions[j,i] = evaluator.predict(context,device,0,guarded=False)[0]
                detail = evaluator.diagnostics(context,device)
                self.panel_updates[j,i] = detail.get('local_updates_count',0)
                weight = detail.get('effective_local_weight')
                self.panel_weights[j,i] = float(weight) if weight is not None else np.nan

    def run(self):
        self.env.process(self._panel_process())
        super().run()
        # Auxiliary panel timers must not extend the physical accounting horizon.
        self.now = max((t.finish or 0. for t in self.completed), default=0.)
        result = self.summary()
        result['panel_noisy_mape'] = float(np.mean(np.abs(self.panel_predictions-self.panel_noisy_truth)/self.panel_noisy_truth))
        result['panel_median_mape'] = float(np.mean(np.abs(self.panel_predictions-self.panel_median_truth)/self.panel_median_truth))
        return result


def make_runner(path, mode, params, seed, load, panels=True):
    def factory(base, strategy, *args, **kwargs):
        if strategy != 'full_qos': raise ValueError('Common scheduler required')
        return core.create_calibrator(base,mode,params)
    core.sim.OnlineCalibrator = factory
    scenario = make_scenario(path,seed,load)
    cls = PanelSimulator if panels else CurrentTimeSimulator
    runner = cls(scenario,'full_qos',seed,0.,24,4,.30,.20,0.,1.,.50,.22,1.20,
                 1.5,.65,.080,.015,8,.80,.30)
    if panels:
        unique = [(r['context'],r['worker']) for r in load_base_contexts()]
        runner.panel_keys = unique
        median = np.asarray([scenario.base[k] for k in unique],dtype=float)
        noise = np.random.default_rng(seed+905001).normal(0.,.08,size=(len(TIMES),len(unique)))
        runner.panel_median_truth = np.tile(median,(len(TIMES),1))
        runner.panel_noisy_truth = runner.panel_median_truth*np.exp(noise)
        runner.panel_predictions = np.empty_like(noise)
        runner.panel_updates = np.empty_like(noise,dtype=np.int32)
        runner.panel_weights = np.empty_like(noise)
    return runner


def fingerprint(tasks):
    plan = [(t.task_id,t.user,t.context,t.arrival,t.qos,t.deadline,sorted(t.true_times.items())) for t in tasks]
    return hashlib.sha256(json.dumps(plan,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def task_rows(tasks):
    return [dict(task_id=t.task_id,user=t.user,context=json.dumps(t.context),arrival=t.arrival,
                 qos=t.qos,deadline=t.deadline,potential_times=json.dumps(t.true_times,sort_keys=True)) for t in tasks]


def run_unit(task):
    image,rho,seed,load,params,run_mode = task
    folder = OUT/run_mode; uid=f'image{image}_rho{rho:.2f}_seed{seed}_load{load:.2f}'
    path = profile_path(image,rho); rows=[]; hashes=[]; predictions=[]; updates=[]; weights=[]
    for mode in MODES:
        runner=make_runner(path,mode,params,seed,load)
        hashes.append(fingerprint(runner.tasks))
        result=runner.run()
        if result['arrived'] != result['completed']: raise RuntimeError('Incomplete task drain')
        result.update(image_seed=image,rho=rho,task_seed=seed,load=load,mode=mode,
                      exogenous_plan_sha256=hashes[-1],profile_sha256=sha(path))
        rows.append(result); predictions.append(runner.panel_predictions)
        updates.append(runner.panel_updates); weights.append(runner.panel_weights)
        if image==13001 and rho in (0.,.5,1.) and seed==23001:
            write_csv(folder/'raw/representative_traces'/f'{uid}_{mode}.csv',runner.trace)
        if mode==MODES[0]:
            panel_truth=runner.panel_noisy_truth.copy(); panel_median=runner.panel_median_truth.copy()
            write_csv(folder/'raw/task_plans'/f'seed{seed}_load{load:.2f}.csv',task_rows(runner.tasks))
    if len(set(hashes))!=1: raise RuntimeError('Different exogenous plans')
    npz=folder/'raw/panels'/f'{uid}.npz'; npz.parent.mkdir(parents=True,exist_ok=True)
    temp=npz.with_name(npz.name+f'.tmp.{os.getpid()}')
    with temp.open('wb') as f:
        np.savez_compressed(f,modes=np.asarray(MODES),predictions=np.asarray(predictions),
                            noisy_truth=panel_truth,median_truth=panel_median,
                            updates=np.asarray(updates),weights=np.asarray(weights),times=np.asarray(TIMES))
    os.replace(temp,npz)
    payload=dict(complete=True,unit_id=uid,rows=rows,same_plan=True,panel_sha256=sha(npz))
    atomic_json(folder/'raw/shards'/f'{uid}.json',payload)
    return payload


def infer(values):
    a=np.asarray(values,dtype=float); rng=np.random.default_rng(20261010)
    boots=a[rng.integers(0,len(a),(5000,len(a)))].mean(axis=1)
    count=0
    for _ in range(10):
        signs=rng.integers(0,2,(10000,len(a)),dtype=np.int8)*2-1
        count+=int(np.count_nonzero(np.abs((signs*a).mean(axis=1))>=abs(a.mean())-1e-15))
    lo,hi=np.quantile(boots,(.025,.975))
    return dict(n_images=len(a),mean_delta=float(a.mean()),ci95_low=float(lo),ci95_high=float(hi),
                p_raw=(count+1)/100001)


def analyze(folder,rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['image_seed'],r['rho'],r['load'],r['mode'])].append(r)
    blocks=[]
    for (image,rho,load,mode),rs in sorted(groups.items()):
        blocks.append(dict(image_seed=image,rho=rho,load=load,mode=mode,n_streams=len(rs),
                           **{m:float(np.mean([r[m] for r in rs])) for m in METRICS}))
    write_csv(folder/'analysis/image_block_metrics.csv',blocks)
    index={(r['image_seed'],r['rho'],r['load'],r['mode']):r for r in blocks}
    summaries=[]; comparisons=[]
    for rho in sorted({r['rho'] for r in rows}):
        for load in sorted({r['load'] for r in rows}):
            images=sorted({r['image_seed'] for r in rows if r['rho']==rho and r['load']==load})
            for mode in MODES:
                summaries.append(dict(rho=rho,load=load,mode=mode,n_images=len(images),
                                      **{m:float(np.mean([index[(i,rho,load,mode)][m] for i in images])) for m in METRICS}))
            for baseline in ('pool_ema','local_window','full_original'):
                for metric in ('sla_violation_rate','panel_median_mape','panel_noisy_mape'):
                    delta=[100*(index[(i,rho,load,'adaptive_pool_local')][metric]-index[(i,rho,load,baseline)][metric]) for i in images]
                    comparisons.append(dict(rho=rho,load=load,baseline=baseline,metric=metric,
                                            primary=(metric=='sla_violation_rate'),unit='percentage_points',**infer(delta)))
    primaries=sorted([r for r in comparisons if r['primary']],key=lambda r:r['p_raw'])
    prev=0.
    for j,r in enumerate(primaries):
        prev=min(1.,max(prev,(len(primaries)-j)*r['p_raw']));r['p_holm_primary']=prev
    write_csv(folder/'analysis/summary.csv',summaries)
    write_csv(folder/'analysis/paired_image_inference.csv',comparisons)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('smoke','full'),required=True)
    ap.add_argument('--freeze',required=True);ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args();freeze=Path(args.freeze);frozen=json.loads(freeze.read_text(encoding='utf-8'))
    if frozen.get('status')!='FROZEN' or not frozen.get('qa_approved'): raise RuntimeError('Formal freeze/QA required')
    for rel,expected in frozen['code_sha256'].items():
        if sha(ROOT/rel)!=expected: raise RuntimeError('Frozen code changed: '+rel)
    if not 1<=args.workers<=3: raise ValueError('At most three workers')
    params=frozen['selected_params'];folder=OUT/args.mode
    images=range(13001,13021) if args.mode=='full' else (13001,)
    rhos=(0.,.25,.5,.75,1.) if args.mode=='full' else (0.,1.)
    seeds=range(23001,23006) if args.mode=='full' else (23001,)
    tasks=[(i,r,s,l,params,args.mode) for i in images for r in rhos for s in seeds for l in (1.25,1.70)]
    meta=dict(mode=args.mode,total_units=len(tasks),total_runs=len(tasks)*len(MODES),
              freeze_sha256=sha(freeze),runner_sha256=sha(__file__),selected_params=params)
    cfg=folder/'config.json'
    if cfg.exists() and json.loads(cfg.read_text(encoding='utf-8'))!=meta: raise RuntimeError('Resume configuration changed')
    atomic_json(cfg,meta)
    missing=[];payloads=[]
    for task in tasks:
        i,r,s,l,_,_=task;uid=f'image{i}_rho{r:.2f}_seed{s}_load{l:.2f}'
        shard=folder/'raw/shards'/f'{uid}.json'
        if shard.exists():
            saved=json.loads(shard.read_text(encoding='utf-8'));payloads.append(saved)
        else:missing.append(task)
    started=time.perf_counter();done=len(payloads)
    # Prepare shared profiles once before workers read them on Windows.
    for image in images:
        for rho in rhos:profile_path(image,rho)
    print(json.dumps(dict(stage='closed_loop',total=len(tasks),resume=done,workers=args.workers)),flush=True)
    with Pool(args.workers) as pool:
        for payload in pool.imap_unordered(run_unit,missing):
            payloads.append(payload);done+=1
            if done%50==0 or done==len(tasks):print(json.dumps(dict(completed=done,total=len(tasks),elapsed_s=round(time.perf_counter()-started,2))),flush=True)
    rows=[row for p in payloads for row in p['rows']]
    write_csv(folder/'raw/per_run_results.csv',rows);analyze(folder,rows)
    atomic_json(folder/'completion.json',dict(complete=True,units=len(tasks),runs=len(rows),all_modes_same_plan=True,
                                            elapsed_seconds=time.perf_counter()-started,completed_at_unix=time.time()))
    print('CLOSED LOOP COMPLETE',flush=True)


if __name__=='__main__':main()
