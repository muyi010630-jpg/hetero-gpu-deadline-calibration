"""Exploratory CPU screen: decision losses and finite-budget acquired feedback.

Online decisions and legal updates never receive unselected true durations.
Truth is confined to physics, read-only diagnostics and named privileged arms.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frozen_support import calibrator_core as core
from frozen_support.profile_flows import build_image, load_base_contexts
from decision_calibrator import DecisionCalibrator

SIM = core.sim
PANEL_TIMES = (2., 6., 12., 24.)
HORIZON = 24.
PROBE_FRACTION = .05
BUDGET_FRACTION = .02
MAX_PREDICTED_SLOWDOWN = 1.20
RHO_VALUES = (0., .5, 1.)
LOAD_VALUES = (.75, 1.70)
BASE_SOURCE = ROOT.parent/'pooled_ema_cpu_20261009/00_configuration_sources/profiles_P0/cluster_08_profile.csv'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp,path)

def write_csv(path, rows):
    rows=list(rows); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temp,path)

def profile_path(image,rho):
    return ROOT/'00_design_sources/profiles'/f'image{image}_rho{rho:.2f}.csv'

def prepare_profiles(images):
    moments=[]
    for image in images:
        for rho in RHO_VALUES:
            unique=build_image(image,rho); rows=[]
            for row in unique:
                for j in range(1,{'RTX5070':2,'T4':3,'A100':3}[row['model']]+1):
                    rows.append(dict(device=f"{row['model']}__{j:02d}",vendor=row['vendor'],memory_gb=row['memory_gb'],
                                     family=row['family'],size=row['size'],batch=row['batch'],dtype=row['dtype'],
                                     initial_s=row['initial_s'],true_s=row['true_s']))
            write_csv(profile_path(image,rho),rows)
            z=np.array([r['z_log'] for r in unique]); gids=[(r['model'],r['family']) for r in unique]
            groups={g:np.mean([z[j] for j,h in enumerate(gids) if h==g]) for g in set(gids)}
            between=np.mean([groups[g]**2 for g in gids])
            moments.append(dict(image_seed=image,rho=rho,mean_log_mismatch=float(z.mean()),sd_log_mismatch=float(z.std()),
                                realized_between_fraction=float(between/.04),profile_sha256=sha(profile_path(image,rho))))
    return moments

def scenario_for(path, seed, load):
    return SIM.Scenario(random.Random(seed),HORIZON,8*load,False,str(path),1.,1.,.25,.80,
                        0.,1.60,2.,'median',1.,.92,1.08,0.,0.,(1.60,1.30,1.10))

def plan_hash(tasks):
    plan=[(t.task_id,t.user,t.context,t.arrival,t.qos,t.deadline,sorted(t.true_times.items())) for t in tasks]
    return hashlib.sha256(json.dumps(plan,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def score_value(service, now, task, load_fraction):
    budget=max(task.deadline-task.arrival,1e-6)
    return max(0.,now+service-task.arrival)/budget+.65*max(0.,now+service-task.deadline)/budget+.08*load_fraction

class FeasibilitySimulator(SIM.EventSimulator):
    def __init__(self, scenario, seed, arm):
        # Constructor uses an ordinary calibrator; replace without modifying module globals.
        super().__init__(scenario,'full_qos',seed,0.,24,4,.30,.20,0.,1.,.50,.22,1.20,
                         1.5,.65,.080,.015,8,.80,.30)
        self.arm=dict(arm); self.policy=arm['policy']; self.mode=arm['mode']
        self.perfect=self.mode=='perfect_information'
        self.cal=DecisionCalibrator(scenario.initial_base,'local_window' if self.perfect else self.mode,
                                    {'eta':arm.get('eta',.05),'ranking_weight':arm.get('ranking_weight',.1)})
        self.decision_records=[]; self.pending={}; self.panel_rows=[]
        self.dispatch_count=0; self.probes_used=0; self.predicted_base_service_sum=0.
        self.probe_budget_used=0.; self.probe_positive_cost=0.; self.probe_signed_cost=0.
        self.probe_reserved_cost={}; self.probe_eligible_count=0; self.critical_eligible_count=0
        self.probe_cooldown={}; self.probe_events=[]
        self.snapshot_rng=random.Random(seed+907321)

    def _mean_prediction(self,task,device):
        self.cal.set_now(self.env.now)
        if self.perfect: return task.true_times[device.spec.name],0.
        return self.cal.predict(task.context,device.spec.name,task.qos,guarded=False)

    def _guarded_prediction(self,task,device):
        self.cal.set_now(self.env.now)
        if self.perfect: return task.true_times[device.spec.name],0.
        return self.cal.predict(task.context,device.spec.name,task.qos,guarded=True)

    def _best_predicted_time(self,task,devices):
        compatible=self._compatible(task,devices)
        return min((self._guarded_prediction(task,d)[0] for d in compatible),default=float('inf'))

    def _snapshot(self,task,idle):
        candidates=[]
        for d in self._compatible(task,idle):
            pred,sigma=self._mean_prediction(task,d)
            cell=self.cal._cell((task.context,d.spec.name))
            key=(task.context,d.spec.name); values=self.cal.raw_local.get(key,())
            mean=(self.cal.base[key]*math.exp(statistics.mean(values))) if values else None
            age=self.env.now-cell.last_update if values else None
            # Empirical range proxy, not a probability-valid confidence interval.
            width=min(.40,max(.08,sigma,.25/math.sqrt(cell.updates+1)))
            load=d.served_work/max(self.total_work,1.)
            score=score_value(pred,self.env.now,task,load)
            lo=score_value(pred*math.exp(-width),self.env.now,task,load)
            hi=score_value(pred*math.exp(width),self.env.now,task,load)
            remaining=max(0.,task.deadline-self.env.now)
            deadline_crossing=float(pred*math.exp(-width)<=remaining<=pred*math.exp(width))
            candidates.append(dict(device=d.spec.name,pred_s=pred,load_fraction=load,updates=cell.updates,
                                   sigma=sigma,width_log=width,score=score,score_low=lo,score_high=hi,
                                   history_mean_s=mean,history_age_s=age,deadline_crossing=deadline_crossing))
        candidates.sort(key=lambda r:(r['score'],r['device']))
        # Same-model replicas have independent simulated noise but shared measured median.
        # Evaluate useful alternatives at distinct model granularity, not clone count.
        best=candidates[0]; alternatives=[r for r in candidates[1:]
                                          if SIM.measured_device_type(r['device'])!=SIM.measured_device_type(best['device'])]
        for r in candidates:
            if r['device']==best['device']: r['overlap']=0.; continue
            denom=max(best['score_high']-best['score_low']+r['score_high']-r['score_low'],1e-6)
            r['overlap']=max(0.,min(1.,(best['score_high']-r['score_low'])/denom))
        near=max((r['overlap'] for r in alternatives),default=0.)
        criticality=min(1.,max(best['deadline_crossing'],2.*near))
        return dict(now=float(self.env.now),arrival=task.arrival,deadline=task.deadline,qos=task.qos,
                    candidates=candidates,criticality=criticality,default_device=best['device'])

    def _probe_choice(self,task,idle,snapshot):
        if self.policy=='no_probe' or task.qos>=2: return None
        best=snapshot['candidates'][0]
        viable=[]
        for r in snapshot['candidates'][1:]:
            if SIM.measured_device_type(r['device'])==SIM.measured_device_type(best['device']): continue
            if r['pred_s']>MAX_PREDICTED_SLOWDOWN*best['pred_s']: continue
            if task.qos==1 and self.env.now+r['pred_s']>task.deadline: continue
            key=(task.context,SIM.measured_device_type(r['device']))
            if self.env.now-self.probe_cooldown.get(key,-1e9)<1.: continue
            reservation=max(.001,.02*best['pred_s'],r['pred_s']-best['pred_s'])
            total_budget=BUDGET_FRACTION*(self.predicted_base_service_sum+best['pred_s'])
            if self.probe_budget_used+reservation>total_budget+1e-12: continue
            # Reserve a predicted incremental cost and cap exploration count online.
            if self.probes_used+1>PROBE_FRACTION*(self.dispatch_count+1)+1e-12: continue
            useful=float(r['overlap']>0. or r['deadline_crossing']>0.)
            information=1/math.sqrt(r['updates']+1)
            # No future outcomes, rho, task truth or test labels enter this score.
            priority=(2.*r['overlap']+r['deadline_crossing'])*information/max(reservation,.001)
            viable.append((r,reservation,useful,priority))
        if not viable: return None
        self.probe_eligible_count+=1
        critical=[v for v in viable if v[2]>0.]
        if critical:self.critical_eligible_count+=1
        if self.policy=='random': chosen=self.snapshot_rng.choice(viable)
        elif self.policy=='uncertainty': chosen=min(viable,key=lambda v:(v[0]['updates'],-v[0]['sigma'],v[0]['device']))
        elif self.policy=='gated_random':
            if not critical:return None
            chosen=self.snapshot_rng.choice(critical)
        elif self.policy=='critical':
            if not critical:return None
            chosen=max(critical,key=lambda v:(v[3],-v[0]['updates'],v[0]['device']))
        elif self.policy=='legacy':
            # Legacy source priority within the selected task, same common budget.
            chosen=min(viable,key=lambda v:(v[0]['updates'],-v[0]['sigma'],v[0]['device']))
        else:raise ValueError('Unknown policy '+self.policy)
        row,reservation,_,priority=chosen
        return self.devices[row['device']],row['pred_s'],reservation,priority

    def dispatch(self):
        self.now=self.env.now
        while self.queue:
            idle=[d for d in self.devices.values() if d.running is None and d.spec.capacity>0]
            feasible=[t for t in self.queue if self._compatible(t,idle)]
            if not idle or not feasible:return
            task=min(feasible,key=lambda t:(self._task_priority(t,idle),t.task_id))
            snapshot=self._snapshot(task,idle)
            default=snapshot['candidates'][0]; dev=self.devices[default['device']]; pred=default['pred_s']
            choice=None
            if not self._urgent_strict_waiting(feasible,idle):choice=self._probe_choice(task,idle,snapshot)
            if choice:
                dev,pred,reservation,priority=choice
                task.probed=True; self.probes_used+=1; self.probe_budget_used+=reservation
                self.probe_reserved_cost[task.task_id]=reservation
                self.probe_cooldown[(task.context,SIM.measured_device_type(dev.spec.name))]=self.env.now
                self.probe_events.append(dict(task_id=task.task_id,now=self.env.now,selected=dev.spec.name,
                                              reference=default['device'],reservation_s=reservation,
                                              priority=priority,criticality=snapshot['criticality']))
            self.dispatch_count+=1; self.predicted_base_service_sum+=default['pred_s']
            snapshot['selected_device']=dev.spec.name
            self.cal.register_decision(task.task_id,task.context,dev.spec.name,snapshot)
            self.pending[task.task_id]=snapshot
            self.queue.remove(task); task.start=self.env.now; task.device=dev.spec.name; task.predicted=pred
            bound,sigma=self._guarded_prediction(task,dev)
            task.prediction_bound=bound; task.prediction_sigma=sigma
            # Below this line, truth is used solely for physics/evaluation.
            candidates=snapshot['candidates']
            true_scores={r['device']:score_value(task.true_times[r['device']],self.env.now,task,r['load_fraction']) for r in candidates}
            optimum=min(true_scores,key=lambda k:(true_scores[k],k))
            regret=max(0.,true_scores[dev.spec.name]-true_scores[optimum])
            model_wrong=(SIM.measured_device_type(dev.spec.name)!=SIM.measured_device_type(optimum))
            # Preserve the source simulator's legacy all-compatible-device
            # ratio, separately from the new valid idle-candidate score regret.
            legacy_compatible=self._compatible(task,list(self.devices.values()))
            true_service_best=min(legacy_compatible,key=lambda d:task.true_times[d.spec.name]).spec.name
            task.optimal_device=true_service_best
            task.selection_regret=task.true_times[dev.spec.name]/task.true_times[true_service_best]
            self.selection_regrets.append(task.selection_regret); self.selection_count+=1
            self.optimal_selections+=int(dev.spec.name==true_service_best)
            delta=task.true_times[dev.spec.name]-task.true_times[default['device']]
            if task.probed:
                self.probe_positive_cost+=max(0.,delta); self.probe_signed_cost+=delta
            hist_available=sum(r['updates']>0 for r in candidates)
            self.decision_records.append(dict(task_id=task.task_id,now=self.env.now,qos=task.qos,
                context=json.dumps(task.context),selected_device=dev.spec.name,default_device=default['device'],
                oracle_score_device=optimum,candidate_count=len(candidates),observed_candidate_count=hist_available,
                criticality=snapshot['criticality'],decision_score_regret=regret,wrong_choice=int(model_wrong),
                positive_probe_service_cost_s=max(0.,delta) if task.probed else 0.,
                signed_probe_service_delta_s=delta if task.probed else 0.,probed=int(task.probed),
                predicted_s=pred,actual_s=task.true_times[dev.spec.name],
                candidates_json=json.dumps(candidates,ensure_ascii=False,separators=(',',':'))))
            duration=task.true_times[dev.spec.name];task.finish=self.env.now+duration
            dev.running=task;dev.busy_until=task.finish
            self.env.process(self._complete_after(dev,task,duration))

    def _complete_after(self,device,task,duration):
        yield self.env.timeout(duration); self.now=self.env.now
        snapshot=self.pending.pop(task.task_id)
        vector=None
        if self.mode in ('rank_fullinfo','uniform_fullinfo'):
            vector={r['device']:task.true_times[r['device']] for r in snapshot['candidates']}
        self.cal.complete_feedback(task.task_id,duration,self.env.now,full_vector=vector)
        if task.qos>=2 and task.prediction_bound is not None:
            self.bound_total+=1;self.bound_hits+=int(duration<=task.prediction_bound)
        safe=task.start+task.prediction_bound<=task.deadline
        false=safe and task.finish>task.deadline
        self.predicted_safe_count+=int(safe);self.false_safe_count+=int(false)
        cell=self.cal._cell((task.context,device.spec.name))
        diag=self.cal.diagnostics(task.context,device.spec.name)
        self.trace.append(dict(task_id=task.task_id,arrival=task.arrival,start=task.start,finish=task.finish,
                               context=json.dumps(task.context),device=device.spec.name,qos=task.qos,deadline=task.deadline,
                               predicted_s=task.predicted,observed_s=duration,probed=int(task.probed),
                               sla_violation=int(task.finish>task.deadline),calibration_updates=cell.updates,
                               adapter_z=diag.get('adapter_z',0.),pair_updates=diag.get('pair_updates',0)))
        work=1.+.3*task.qos;device.served_work+=work
        self.user_work[task.user]=self.user_work.get(task.user,0.)+work;self.total_work+=work
        self.completed.append(task);device.running=None;device.busy_until=self.env.now
        self.dispatch()

    def _panels(self):
        previous=0.
        for at in PANEL_TIMES:
            yield self.env.timeout(at-previous);previous=at
            evaluator=copy.deepcopy(self.cal);evaluator.set_now(at)
            for row in load_base_contexts():
                key=(row['context'],row['worker'])
                truth=self.scenario.base[key]
                pred=truth if self.perfect else evaluator.predict(*key,0,guarded=False)[0]
                self.panel_rows.append(dict(at=at,context_id=row['context_id'],model=row['model'],
                                            predicted_s=pred,median_truth_s=truth,
                                            ape=abs(pred-truth)/truth,updates=evaluator._cell(key).updates))

    def run(self):
        self.env.process(self._panels());super().run()
        self.now=max((t.finish for t in self.completed),default=0.)
        out=self.summary();records=self.decision_records
        out.update(decision_score_regret_mean=statistics.mean(r['decision_score_regret'] for r in records),
                   decision_wrong_choice_rate=statistics.mean(r['wrong_choice'] for r in records),
                   probe_fraction=self.probes_used/max(1,self.dispatch_count),
                   probe_positive_service_cost_s=self.probe_positive_cost,
                   probe_signed_service_delta_s=self.probe_signed_cost,
                   probe_predicted_budget_s=BUDGET_FRACTION*self.predicted_base_service_sum,
                   probe_budget_used_s=self.probe_budget_used,
                   probe_eligible_count=self.probe_eligible_count,critical_eligible_count=self.critical_eligible_count,
                   supervised_updates=getattr(self.cal,'supervised_updates',0),pair_updates=getattr(self.cal,'pair_updates',0),
                   privileged_label_count=getattr(self.cal,'privileged_label_count',0),
                   missing_feedback_fraction=statistics.mean(self.cal._cell(k).updates==0 for k in self.cal.base),
                   panel_mape=statistics.mean(r['ape'] for r in self.panel_rows),
                   panel_final_mape=statistics.mean(r['ape'] for r in self.panel_rows if r['at']==PANEL_TIMES[-1]),
                   total_actual_service_s=sum(t.true_times[t.device] for t in self.completed))
        return out

def arm_id(arm):
    return f"{arm['mode']}__{arm['policy']}__eta{arm.get('eta',.05):g}__rw{arm.get('ranking_weight',.1):g}"

def validation_arms():
    arms=[dict(mode=m,policy='no_probe',eta=.05,ranking_weight=.1) for m in
          ('local_window','pool_ema','full_original','perfect_information')]
    for eta in (.01,.05,.15):
        for mode in ('uniform_adapter','decision_weighted','uniform_fullinfo'):
            arms.append(dict(mode=mode,policy='no_probe',eta=eta,ranking_weight=.1))
        for rw in (.1,1.):
            arms.extend(dict(mode=m,policy='no_probe',eta=eta,ranking_weight=rw) for m in ('rank_history','rank_fullinfo'))
    for policy in ('random','uncertainty','gated_random','critical'):
        arms.append(dict(mode='local_window',policy=policy,eta=.05,ranking_weight=.1))
    return arms

def code_hashes():
    paths=['scripts/run_feasibility.py','scripts/decision_calibrator.py','scripts/analyze_results.py',
           'frozen_support/calibrator_core.py','frozen_support/profile_flows.py',
           '00_protocol_sources/frozen_sources/hetero_qos_sim_n2.py','00_design_sources/实验协议.md']
    return {p:sha(ROOT/p) for p in paths}

def run_unit(args):
    stage,image,rho,seed,load,arms,source_fingerprint=args
    folder=ROOT/('01_development_validation' if stage=='validation' else '02_heldout_closed_loop')
    uid=f'image{image}_rho{rho:.2f}_task{seed}_load{load:.2f}'
    path=profile_path(image,rho);rows=[];hashes=[]
    for arm in arms:
        scenario=scenario_for(path,seed,load);runner=FeasibilitySimulator(scenario,seed,arm)
        plan=plan_hash(runner.tasks);hashes.append(plan);result=runner.run()
        if result['arrived']!=result['completed']:raise RuntimeError('Incomplete drain')
        if result['probe_budget_used_s']>result['probe_predicted_budget_s']+1e-10:raise RuntimeError('Online predicted budget exceeded')
        if result['probe_fraction']>PROBE_FRACTION+1e-12:raise RuntimeError('Count budget exceeded')
        if arm['mode'] not in ('rank_fullinfo','uniform_fullinfo') and result['privileged_label_count']:
            raise RuntimeError('Privileged label leak')
        result.update(stage=stage,image_seed=image,rho=rho,task_seed=seed,load=load,**arm,
                      arm_id=arm_id(arm),run_id=uid+'__'+arm_id(arm),plan_sha256=plan,
                      profile_sha256=sha(path),code_fingerprint=source_fingerprint)
        rows.append(result)
        # Keep every per-task trace/decision snapshot for posthoc audits; compressed JSONL.
        import gzip
        raw=folder/'raw/traces'/f"{uid}__{arm_id(arm)}.jsonl.gz";raw.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(raw,'wt',encoding='utf-8') as handle:
            for kind,items in (('decision',runner.decision_records),('completion',runner.trace),('panel',runner.panel_rows)):
                for item in items:handle.write(json.dumps(dict(kind=kind,**item),ensure_ascii=False,separators=(',',':'))+'\n')
        result['trace_sha256']=sha(raw)
        if arm==arms[0]:
            write_csv(folder/'raw/task_plans'/f'seed{seed}_load{load:.2f}.csv',
                      [dict(task_id=t.task_id,user=t.user,context=json.dumps(t.context),arrival=t.arrival,qos=t.qos,
                            deadline=t.deadline,potential_times=json.dumps(t.true_times,sort_keys=True)) for t in runner.tasks])
    if len(set(hashes))!=1:raise RuntimeError('Unequal task plans across arms')
    payload=dict(complete=True,unit_id=uid,stage=stage,same_plan=True,code_fingerprint=source_fingerprint,rows=rows)
    atomic_json(folder/'raw/shards'/f'{uid}.json',payload)
    return payload

def select_params(rows):
    # Validation only: average seeds within image/condition, then all cells equally.
    by=defaultdict(list)
    for r in rows:by[(r['arm_id'],r['image_seed'],r['rho'],r['load'])].append(r['sla_violation_rate'])
    scores=defaultdict(list)
    for (aid,_,_,_),values in by.items():scores[aid].append(float(np.mean(values)))
    table=[]
    arm_lookup={r['arm_id']:{k:r[k] for k in ('mode','policy','eta','ranking_weight')} for r in rows}
    for aid,values in scores.items():table.append(dict(arm_id=aid,**arm_lookup[aid],validation_sla=float(np.mean(values))))
    chosen={}
    for mode in ('uniform_adapter','decision_weighted','rank_history','uniform_fullinfo','rank_fullinfo'):
        subset=[r for r in table if r['mode']==mode and r['policy']=='no_probe']
        win=min(subset,key=lambda r:(r['validation_sla'],r['eta'],r['ranking_weight']))
        chosen[mode]={k:win[k] for k in ('mode','policy','eta','ranking_weight')}
    legal_winner=min((r for r in table if r['mode'] in ('decision_weighted','rank_history') and r['policy']=='no_probe'),
                     key=lambda r:(r['validation_sla'],r['mode'],r['eta'],r['ranking_weight']))
    baseline_winner=min((r for r in table if r['mode']=='local_window'),
                        key=lambda r:(r['validation_sla'],r['policy']))
    atomic_json(ROOT/'01_development_validation/参数选择.json',dict(objective='Equal image/rho/load mean E2E SLA; tie smaller eta, weight, then mode/policy.',
                 ranking=sorted(table,key=lambda r:r['validation_sla']),selected=chosen,
                 selected_legal={k:legal_winner[k] for k in ('mode','policy','eta','ranking_weight')},
                 strongest_probe_baseline={k:baseline_winner[k] for k in ('mode','policy','eta','ranking_weight')}))
    fixed=[dict(mode=m,policy='no_probe',eta=.05,ranking_weight=.1) for m in
           ('local_window','pool_ema','full_original','perfect_information')]+list(chosen.values())
    fixed.extend(dict(mode='local_window',policy=p,eta=.05,ranking_weight=.1) for p in
                 ('random','uncertainty','gated_random','critical'))
    selected_legal={k:legal_winner[k] for k in ('mode','policy','eta','ranking_weight')}
    fixed.extend(dict(selected_legal,policy=p) for p in ('uncertainty','critical','gated_random'))
    fixed.append(dict(chosen['uniform_adapter'],policy='critical'))
    primaries=[dict(name='decision_update_vs_window',family='feasibility_sla',
                   arm_a=selected_legal,arm_b=dict(mode='local_window',policy='no_probe',eta=.05,ranking_weight=.1)),
               dict(name='critical_feedback_vs_uncertainty',family='feasibility_sla',
                   arm_a=dict(mode='local_window',policy='critical',eta=.05,ranking_weight=.1),
                   arm_b=dict(mode='local_window',policy='uncertainty',eta=.05,ranking_weight=.1)),
               dict(name='combined_vs_validation_probe_baseline',family='feasibility_sla',
                   arm_a=dict(selected_legal,policy='critical'),
                   arm_b={k:baseline_winner[k] for k in ('mode','policy','eta','ranking_weight')}),
               dict(name='decision_objective_vs_uniform_adapter',family='feasibility_sla',
                   arm_a=selected_legal,arm_b=chosen['uniform_adapter']),
               dict(name='full_information_ranking_vs_uniform',family='feasibility_sla',
                   arm_a=chosen['rank_fullinfo'],arm_b=chosen['uniform_fullinfo']),
               dict(name='combined_vs_uniform_same_probe',family='feasibility_sla',
                   arm_a=dict(selected_legal,policy='critical'),arm_b=dict(chosen['uniform_adapter'],policy='critical'))]
    return dict(status='FROZEN',exploratory=True,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                code_sha256=code_hashes(),input_sha256={str(BASE_SOURCE):sha(BASE_SOURCE)},
                selected=chosen,selected_legal=selected_legal,test_arms=fixed,
                validation_images=list(range(31001,31005)),test_images=list(range(32001,32017)),
                validation_tasks=list(range(41001,41003)),test_tasks=list(range(42001,42004)),
                analysis=dict(rate_columns_are='fraction',bootstrap_repetitions=5000,seed=20261010,primary_comparisons=primaries,
                    design={'test':dict(image_seeds=list(range(32001,32017)),task_seeds=list(range(42001,42004)),
                                       rho_levels=list(RHO_VALUES),load_levels=list(LOAD_VALUES))}),
                screening_note='Prospective exploratory screen, not confirmatory proof. Old test data used only to choose research direction.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('smoke','validation','test'),required=True)
    ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    if not 1<=args.workers<=3:raise ValueError('Use 1-3 workers')
    stage='validation' if args.stage=='smoke' else args.stage
    if stage=='validation':
        images=(31001,) if args.stage=='smoke' else range(31001,31005)
        seeds=(41001,) if args.stage=='smoke' else range(41001,41003)
        arms=validation_arms(); folder=ROOT/'01_development_validation'
        if args.stage=='smoke':arms=[dict(mode='local_window',policy='no_probe',eta=.05,ranking_weight=.1)]
    else:
        frozen=json.loads((ROOT/'00_design_sources/final_freeze.json').read_text(encoding='utf-8'))
        if frozen['code_sha256']!=code_hashes():raise RuntimeError('Frozen executable changed')
        if sha(BASE_SOURCE)!=frozen['input_sha256'][str(BASE_SOURCE)]:raise RuntimeError('Source profile changed')
        images=frozen['test_images'];seeds=frozen['test_tasks'];arms=frozen['test_arms'];folder=ROOT/'02_heldout_closed_loop'
    moments=prepare_profiles(images);write_csv(folder/'profile_moments.csv',moments)
    fingerprint=hashlib.sha256(json.dumps(code_hashes(),sort_keys=True).encode()).hexdigest()
    loads=(.75,) if args.stage=='smoke' else LOAD_VALUES
    rhos=(0.,) if args.stage=='smoke' else RHO_VALUES
    jobs=[(stage,i,r,s,l,arms,fingerprint) for i in images for r in rhos for s in seeds for l in loads]
    atomic_json(folder/'config.json',dict(stage=args.stage,units=len(jobs),runs=len(jobs)*len(arms),arms=arms,
                source_fingerprint=fingerprint,horizon_s=HORIZON,probe_dispatch_fraction=PROBE_FRACTION,
                predicted_extra_service_budget_fraction=BUDGET_FRACTION,workers=args.workers))
    started=time.perf_counter();payloads=[]
    with Pool(args.workers) as pool:
        for p in pool.imap_unordered(run_unit,jobs):
            payloads.append(p)
            if len(payloads)%8==0 or len(payloads)==len(jobs):
                print(json.dumps(dict(stage=args.stage,completed_units=len(payloads),total_units=len(jobs),
                      runs=len(payloads)*len(arms),elapsed_s=round(time.perf_counter()-started,2))),flush=True)
    rows=[r for p in payloads for r in p['rows']];write_csv(folder/'raw/per_run_results.csv',rows)
    atomic_json(folder/'completion.json',dict(complete=True,units=len(jobs),runs=len(rows),
                all_plans_matched=True,elapsed_s=time.perf_counter()-started,completed_at_utc=datetime.now(timezone.utc).isoformat()))
    if args.stage=='validation':
        freeze=select_params(rows);atomic_json(ROOT/'00_design_sources/final_freeze.json',freeze)
        print(json.dumps(dict(selected=freeze['selected'],selected_legal=freeze['selected_legal']),ensure_ascii=False),flush=True)
    print('FEASIBILITY '+args.stage.upper()+' COMPLETE',flush=True)

if __name__=='__main__':main()
