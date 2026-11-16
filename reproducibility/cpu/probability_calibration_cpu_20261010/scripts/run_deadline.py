"""CPU screen of observable deadline-probability calibration; no active probes.

All online forecasts/updates use selected feedback only. The named mean oracle
and evaluation routines alone read profile truth; no legal arm sees task truth.
"""
from __future__ import annotations
import argparse, copy, csv, gzip, hashlib, json, math, os, random, statistics, sys, time
from collections import defaultdict, deque
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from frozen_support import calibrator_core as core
from frozen_support.profile_flows import build_image,load_base_contexts
from deadline_calibrator import DeadlineCalibrator
SIM=core.sim
NORMAL=statistics.NormalDist()
RHO_VALUES=(0.,.5,1.); LOAD_VALUES=(.75,1.70)
DEV_HORIZONS=(24.,120.); TEST_HORIZONS=(24.,120.,600.)
DEV_IMAGES=tuple(range(51001,51005)); TEST_IMAGES=tuple(range(52001,52017))
DEV_TASKS=(61001,61002); TEST_TASKS=(62001,62002)
BASE_SOURCE=ROOT.parent/'pooled_ema_cpu_20261009/00_configuration_sources/profiles_P0/cluster_08_profile.csv'
EPS=1e-6; BOUNDARY_LOG_PERTURB=.05; LEVELS=(.50,.80,.90)
TASK_CACHE={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r));tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w',encoding='utf-8-sig',newline='') as out:
        w=csv.DictWriter(out,fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
def profile_path(image,rho):return ROOT/'00_design_sources/profiles'/f'image{image}_rho{rho:.2f}.csv'
def prepare_profiles(images):
    moments=[]
    for image in images:
        for rho in RHO_VALUES:
            unique=build_image(image,rho);rows=[]
            for row in unique:
                for j in range(1,{'RTX5070':2,'T4':3,'A100':3}[row['model']]+1):
                    rows.append(dict(device=f"{row['model']}__{j:02d}",vendor=row['vendor'],memory_gb=row['memory_gb'],family=row['family'],size=row['size'],batch=row['batch'],dtype=row['dtype'],initial_s=row['initial_s'],true_s=row['true_s']))
            write_csv(profile_path(image,rho),rows)
            z=np.array([r['z_log'] for r in unique]);groups=[(r['model'],r['family']) for r in unique]
            gm={g:np.mean([z[i] for i,h in enumerate(groups) if h==g]) for g in set(groups)}
            moments.append(dict(image_seed=image,rho=rho,mean_log_mismatch=float(z.mean()),sd_log_mismatch=float(z.std()),realized_between_fraction=float(np.mean([gm[g]**2 for g in groups])/.04),profile_sha256=sha(profile_path(image,rho))))
    return moments
def bare_scenario(path,seed,load):
    return SIM.Scenario(random.Random(seed),24.,8*load,False,str(path),1.,1.,.25,.80,0.,1.60,2.,'median',1.,.92,1.08,0.,0.,(1.60,1.30,1.10))
def scenario_for(path,seed,load,horizon):
    """Periodic 24-s streams with exactly common exogenous 24/120/600 prefixes."""
    scenario=bare_scenario(path,seed,load)
    cache_key=(seed,load)
    if cache_key not in TASK_CACHE:
        tasks=[]
        for block in range(25):
            block_s=bare_scenario(path,seed+1000003*block,load)
            for task in block_s.generate_tasks():
                task.task_id=len(tasks);task.arrival+=24*block;task.deadline+=24*block;tasks.append(task)
        TASK_CACHE[cache_key]=tasks
    scenario.horizon=horizon
    scenario.generate_tasks=lambda:copy.deepcopy([t for t in TASK_CACHE[cache_key] if t.arrival<horizon])
    return scenario
def plan_hash(tasks):
    values=[(t.task_id,t.user,t.context,t.arrival,t.qos,t.deadline,sorted(t.true_times.items())) for t in tasks]
    return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def score(point,p,now,arrival,deadline,load):
    return max(0.,now+point-arrival)/max(deadline-arrival,1e-6)+.65*(1-p)+.08*load
def old_score(point,now,arrival,deadline,load):
    b=max(deadline-arrival,1e-6)
    return max(0.,now+point-arrival)/b+.65*max(0.,now+point-deadline)/b+.08*load
def probability_loss(p,y):
    cp=min(1-EPS,max(EPS,p));return (p-y)**2,-y*math.log(cp)-(1-y)*math.log1p(-cp)

class DeadlineSimulator(SIM.EventSimulator):
    def __init__(self,scenario,seed,arm,collect_common=False):
        super().__init__(scenario,'full_qos',seed,0.,24,4,.30,.20,0.,1.,.50,.22,1.20,1.5,.65,.080,.015,8,.80,.30)
        self.arm=dict(arm);self.mode=arm['mode'];self.horizon=scenario.horizon
        cmode=self.mode if self.mode in ('uniform','boundary') else 'uncalibrated'
        self.probcal=DeadlineCalibrator(scenario.initial_base,cmode,dict(eta=arm['eta'] if cmode!='uncalibrated' else .02,boundary_lambda=arm.get('weight',0.) if cmode=='boundary' else 0.))
        self.cal=self.probcal.core
        self.mean_oracle=self.mode=='mean_oracle_original'
        self.risk_errors=defaultdict(lambda:deque(maxlen=96))
        self.pending={};self.decisions=[];self.panels=[]
        self.collect_common=collect_common;self.common_events=[]
        self.bank={}
        if collect_common:
            self.bank={name:DeadlineCalibrator(scenario.initial_base,mode,dict(eta=.02,boundary_lambda=weight)) for name,mode,weight in
                       [('uncalibrated','uncalibrated',0.),('uniform','uniform',0.),('boundary1','boundary',1.),('boundary3','boundary',3.)]}

    def _mean_prediction(self,task,device):
        pred,scale,n=self.probcal.predict_point(task.context,device.spec.name,self.env.now)
        if self.mean_oracle:
            pred=self.scenario.base[(task.context,device.spec.name)]*math.exp(.08**2/2)
        return pred,scale
    def _guarded_prediction(self,task,device):
        # Same original task-priority risk rule for every legal mode.
        self.cal.set_now(self.env.now)
        raw,_=self.cal.predict(task.context,device.spec.name,task.qos,guarded=False)
        bound,sigma=self.cal.predict(task.context,device.spec.name,task.qos,guarded=True)
        if self.mean_oracle:
            mean=self.scenario.base[(task.context,device.spec.name)]*math.exp(.08**2/2)
            return mean*bound/max(raw,1e-12),sigma
        return bound,sigma
    def _best_predicted_time(self,task,devices):
        return min((self._guarded_prediction(task,d)[0] for d in self._compatible(task,devices)),default=float('inf'))
    def _probability(self,task,device):
        record=self.probcal.probability(task.context,device.spec.name,task.deadline-self.env.now,self.env.now,calibrated=self.mode in ('uniform','boundary'))
        p=record['p']
        if self.mode.startswith('fixed_margin'):
            p=float(task.deadline-self.env.now>=record['point']*math.exp(self.arm['margin']))
        elif self.mode=='rolling_cdf' and task.deadline>self.env.now:
            errors=self.risk_errors[record['model']];threshold=math.log((task.deadline-self.env.now)/record['point'])
            count=sum(e<=threshold for e in errors)
            p=(count+4*NORMAL.cdf(threshold/.30))/(len(errors)+4)
        record=dict(record,p=float(p));return record
    def _snapshot(self,task,idle):
        candidates=[]
        for dev in self._compatible(task,idle):
            rec=self._probability(task,dev);point=rec['point']
            if self.mean_oracle:point=self.scenario.base[(task.context,dev.spec.name)]*math.exp(.08**2/2)
            load=dev.served_work/max(self.total_work,1.)
            value=(old_score(point,self.env.now,task.arrival,task.deadline,load) if self.mode in ('window_original','mean_oracle_original') else score(point,rec['p'],self.env.now,task.arrival,task.deadline,load))
            candidates.append(dict(device=dev.spec.name,model=SIM.measured_device_type(dev.spec.name),point=point,load_fraction=load,score=value,probability=rec))
        candidates.sort(key=lambda r:(r['score'],r['device']))
        return candidates
    @staticmethod
    def boundary_sensitivity(task,candidates,now):
        """Dispatch-only +/-0.05 log perturbation; distinguish GPU models."""
        if task.deadline<=now:return 0.
        original=[]
        for row in candidates:
            rec=row['probability'];value=score(rec['point'],rec['p0'],now,task.arrival,task.deadline,row['load_fraction'])
            original.append((value,row['device'],row))
        best=min(original,key=lambda r:(r[0],r[1]))[2];best_model=best['model']
        remaining=task.deadline-now;level=LEVELS[task.qos]
        for direction in (-1.,1.):
            values=[]
            for _,_,row in original:
                rec=row['probability'];factor=math.exp(direction*BOUNDARY_LOG_PERTURB) if row['model']==best_model else 1.
                point=rec['point']*factor
                raw_p=NORMAL.cdf(math.log(remaining/point)/rec['scale'])
                p=min(1-EPS,max(EPS,raw_p))
                values.append((score(point,p,now,task.arrival,task.deadline,row['load_fraction']),row['device'],row['model']))
                if row['device']==best['device'] and (p>=level)!=(rec['p0']>=level):return 1.
            if min(values)[2]!=best_model:return 1.
        return 0.
    def _evaluation_probability(self,context,device,remaining):
        # Read-only analytic diagnostic. Never passed to calibration or selection.
        if remaining<=0:return 0.
        median=self.scenario.base[(context,device)]
        return NORMAL.cdf(math.log(remaining/median)/.08)
    def dispatch(self):
        self.now=self.env.now
        while self.queue:
            idle=[d for d in self.devices.values() if d.running is None and d.spec.capacity>0]
            feasible=[t for t in self.queue if self._compatible(t,idle)]
            if not idle or not feasible:return
            task=min(feasible,key=lambda t:(self._task_priority(t,idle),t.task_id))
            candidates=self._snapshot(task,idle);chosen=candidates[0];dev=self.devices[chosen['device']]
            sensitivity=self.boundary_sensitivity(task,candidates,self.env.now)
            remaining=task.deadline-self.env.now;record=chosen['probability']
            self.probcal.register_decision(task.task_id,task.context,dev.spec.name,self.env.now,remaining,record,sensitivity)
            self.pending[task.task_id]=dict(record=record,sensitivity=sensitivity,point=chosen['point'],remaining=remaining)
            bank_predictions={}
            for name,cal in self.bank.items():
                records={r['device']:cal.probability(task.context,r['device'],remaining,self.env.now,calibrated=name!='uncalibrated') for r in candidates}
                cal.register_decision(task.task_id,task.context,dev.spec.name,self.env.now,remaining,records[dev.spec.name],sensitivity)
                bank_predictions[name]=records
            self.queue.remove(task);task.start=self.env.now;task.device=dev.spec.name;task.predicted=chosen['point']
            task.prediction_bound,task.prediction_sigma=self._guarded_prediction(task,dev)
            # Physical service and evaluation below; legal decisions are complete.
            true_scores={r['device']:score(task.true_times[r['device']],float(task.true_times[r['device']]<=remaining),self.env.now,task.arrival,task.deadline,r['load_fraction']) for r in candidates}
            optimum=min(true_scores,key=lambda d:(true_scores[d],d))
            wrong=SIM.measured_device_type(optimum)!=chosen['model']
            compatible=self._compatible(task,list(self.devices.values()));fastest=min(compatible,key=lambda d:task.true_times[d.spec.name]).spec.name
            task.optimal_device=fastest;task.selection_regret=task.true_times[dev.spec.name]/task.true_times[fastest]
            self.selection_regrets.append(task.selection_regret);self.selection_count+=1;self.optimal_selections+=int(dev.spec.name==fastest)
            row=dict(task_id=task.task_id,now=self.env.now,arrival=task.arrival,deadline=task.deadline,qos=task.qos,context=json.dumps(task.context),device=dev.spec.name,model=chosen['model'],remaining=remaining,point=chosen['point'],p0=record['p0'],p=record['p'],scale=record['scale'],n=record['n'],boundary_sensitivity=sensitivity,wrong_model=int(wrong),candidates=candidates)
            self.decisions.append(row)
            if self.collect_common:
                expected={r['device']:self._evaluation_probability(task.context,r['device'],remaining) for r in candidates}
                self.common_events.append(dict(event='dispatch',now=self.env.now,task_id=task.task_id,remaining=remaining,selected=dev.spec.name,model=chosen['model'],boundary_sensitivity=sensitivity,predictions=bank_predictions,true_probabilities_evaluation_only=expected))
            duration=task.true_times[dev.spec.name];task.finish=self.env.now+duration;dev.running=task;dev.busy_until=task.finish
            self.env.process(self._complete_after(dev,task,duration))
    def _complete_after(self,device,task,duration):
        yield self.env.timeout(duration);self.now=self.env.now
        pending=self.pending.pop(task.task_id);rec=pending['record'];remaining=pending['remaining']
        learning=self.probcal.complete_feedback(task.task_id,duration,self.env.now)
        y=int(duration<=remaining);brier,logloss=probability_loss(rec['p'],y)
        # Always use dispatch prediction, not the updated Window estimate.
        self.risk_errors[rec['model']].append(math.log(duration/rec['point']))
        if task.qos>=2:self.bound_total+=1;self.bound_hits+=int(duration<=task.prediction_bound)
        safe=task.start+task.prediction_bound<=task.deadline;false=safe and task.finish>task.deadline
        self.predicted_safe_count+=int(safe);self.false_safe_count+=int(false)
        self.trace.append(dict(task_id=task.task_id,arrival=task.arrival,start=task.start,finish=task.finish,context=json.dumps(task.context),device=device.spec.name,qos=task.qos,deadline=task.deadline,predicted_s=task.predicted,observed_s=duration,p=rec['p'],p0=rec['p0'],remaining=remaining,on_time=y,prob_brier=brier,prob_logloss=logloss,sla_violation=int(task.finish>task.deadline),boundary_sensitivity=pending['sensitivity'],learning=learning))
        if self.collect_common:
            learnings={name:cal.complete_feedback(task.task_id,duration,self.env.now) for name,cal in self.bank.items()}
            self.common_events.append(dict(event='completion',now=self.env.now,task_id=task.task_id,observed_s=duration,on_time=y,learning=learnings))
        work=1.+.3*task.qos;device.served_work+=work;self.user_work[task.user]=self.user_work.get(task.user,0.)+work;self.total_work+=work
        self.completed.append(task);device.running=None;device.busy_until=self.env.now;self.dispatch()
    def _panels(self):
        previous=0.
        for at in (2.,6.,12.,24.,60.,120.,300.,600.):
            if at>self.horizon:break
            yield self.env.timeout(at-previous);previous=at
            cal=copy.deepcopy(self.probcal)
            for row in load_base_contexts():
                context,device=row['context'],row['worker'];initial=self.cal.base[(context,device)]
                for factor in (.80,1.,1.30):
                    remaining=initial*factor;rec=cal.probability(context,device,remaining,at,calibrated=self.mode in ('uniform','boundary'))
                    p=rec['p']
                    if self.mode.startswith('fixed_margin'):p=float(remaining>=rec['point']*math.exp(self.arm['margin']))
                    elif self.mode=='rolling_cdf':
                        errs=self.risk_errors[rec['model']];threshold=math.log(remaining/rec['point']);p=(sum(e<=threshold for e in errs)+4*NORMAL.cdf(threshold/.3))/(len(errs)+4)
                    truep=self._evaluation_probability(context,device,remaining);clipped=min(1-EPS,max(EPS,p))
                    median=self.scenario.base[(context,device)];point=median*math.exp(.08**2/2) if self.mean_oracle else rec['point']
                    self.panels.append(dict(at=at,context_id=row['context_id'],model=row['model'],budget_factor=factor,remaining=remaining,p=p,true_p_evaluation_only=truep,brier_expected=p*p-2*p*truep+truep,logloss_expected=-truep*math.log(clipped)-(1-truep)*math.log1p(-clipped),prob_squared_error=(p-truep)**2,point=point,point_ape=abs(point-median)/median,n=rec['n']))
    def run(self):
        self.env.process(self._panels());super().run();self.now=max(t.finish for t in self.completed)
        result=self.summary();positive=[r for r in self.trace if r['remaining']>0]
        if self.mean_oracle:result['prediction_mape']=statistics.mean(abs(r['predicted_s']-r['observed_s'])/r['observed_s'] for r in self.trace)
        result.update(prob_brier=statistics.mean(r['prob_brier'] for r in positive),prob_logloss=statistics.mean(r['prob_logloss'] for r in positive),prob_brier_all=statistics.mean(r['prob_brier'] for r in self.trace),expired_at_dispatch_fraction=1-len(positive)/len(self.trace),safe_false_positive=statistics.mean(r['p']>=LEVELS[r['qos']] and not r['on_time'] for r in self.trace),safe_false_negative=statistics.mean(r['p']<LEVELS[r['qos']] and r['on_time'] for r in self.trace),point_mape=result['prediction_mape'],decision_model_wrong_rate=statistics.mean(r['wrong_model'] for r in self.decisions),boundary_sensitive_fraction=statistics.mean(r['boundary_sensitivity'] for r in self.decisions),panel_prob_brier=statistics.mean(r['brier_expected'] for r in self.panels),panel_prob_logloss=statistics.mean(r['logloss_expected'] for r in self.panels),panel_prob_squared_error=statistics.mean(r['prob_squared_error'] for r in self.panels),panel_point_mape=statistics.mean(r['point_ape'] for r in self.panels),supervised_updates=getattr(self.probcal,'probability_updates',0),privileged_label_count=0,probe_count=0,probe_fraction=0.,total_actual_service_s=sum(t.true_times[t.device] for t in self.completed))
        for cut in (24.,120.):
            subset=[r for r in self.trace if r['arrival']<cut]
            if cut<=self.horizon:
                result[f'arrival_prefix{int(cut)}_sla']=statistics.mean(r['sla_violation'] for r in subset)
                result[f'arrival_prefix{int(cut)}_n']=len(subset)
        late=[r for r in self.trace if r['arrival']>=self.horizon*.8]
        result['supervised_updates']=self.probcal.calibration_updates
        result['calibration_clip_count']=self.probcal.parameter_clip_count
        result['final_calibration_a']=self.probcal.a
        for model,value in self.probcal.b.items():result['final_calibration_b_'+model]=value
        result['late20_sla']=statistics.mean(r['sla_violation'] for r in late) if late else result['sla_violation_rate']
        return result

def arm(mode,eta=0.,weight=0.,**extra):return dict(mode=mode,eta=eta,weight=weight,**extra)
def arm_id(a):return f"{a['mode']}__eta{a['eta']:g}__w{a['weight']:g}"
def development_arms():
    arms=[arm('window_original'),arm('uncalibrated'),arm('rolling_cdf'),arm('mean_oracle_original')]
    arms.extend(arm(f'fixed_margin_{value:g}',margin=value) for value in (.05,.15,.30))
    for eta in (.005,.02,.08):
        arms.append(arm('uniform',eta))
        arms.extend(arm('boundary',eta,weight) for weight in (1.,3.))
    return arms
def code_hashes():
    paths=['scripts/run_deadline.py','scripts/deadline_calibrator.py','scripts/analyze_deadline.py','scripts/check_artifacts.py','frozen_support/calibrator_core.py','frozen_support/profile_flows.py','00_protocol_sources/frozen_sources/hetero_qos_sim_n2.py','00_design_sources/实验协议.md']
    return {p:sha(ROOT/p) for p in paths}
def run_unit(args):
    stage,image,rho,seed,load,horizon,arms,fingerprint,common=args
    folder=ROOT/({'common':'01_common_feedback_diagnostics','validation':'02_development_validation','test':'03_heldout_closed_loop','smoke':'qa/smoke'}[stage])
    uid=f'image{image}_rho{rho:.2f}_task{seed}_load{load:.2f}_h{horizon:g}';rows=[];hashes=[]
    for a in arms:
        scenario=scenario_for(profile_path(image,rho),seed,load,horizon);runner=DeadlineSimulator(scenario,seed,a,common)
        plan=plan_hash(runner.tasks);hashes.append(plan);result=runner.run()
        if result['arrived']!=result['completed']:raise RuntimeError('Incomplete drain')
        result.update(stage=stage,image_seed=image,rho=rho,task_seed=seed,load=load,horizon=horizon,**a,arm_id=arm_id(a),run_id=uid+'__'+arm_id(a),plan_sha256=plan,code_fingerprint=fingerprint)
        path=folder/'raw/traces'/f"{result['run_id']}.jsonl.gz";path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt',encoding='utf-8',newline='\n') as out:
            for kind,records in [('decision',runner.decisions),('completion',runner.trace),('panel',runner.panels),('common_event',runner.common_events)]:
                for record in records:out.write(json.dumps(dict(kind=kind,**record),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        result['trace_sha256']=sha(path);result['trace_path']=str(path.relative_to(ROOT));rows.append(result)
        if a==arms[0]:
            # One owner per seed/load/horizon prevents parallel overwrites.
            if image==(DEV_IMAGES[0] if stage!='test' else TEST_IMAGES[0]) and rho==RHO_VALUES[0]:
                write_csv(folder/'raw/task_plans'/f'seed{seed}_load{load:.2f}_h{horizon:g}.csv',[dict(task_id=t.task_id,user=t.user,context=json.dumps(t.context),arrival=t.arrival,qos=t.qos,deadline=t.deadline,potential_times=json.dumps(t.true_times,sort_keys=True)) for t in runner.tasks])
    if len(set(hashes))!=1:raise RuntimeError('Unequal plans')
    payload=dict(complete=True,unit_id=uid,stage=stage,rows=rows,same_plan=True);atomic_json(folder/'raw/shards'/f'{uid}.json',payload);return payload
def select_and_freeze(rows):
    scores=defaultdict(list);lookup={}
    for row in rows:scores[row['arm_id']].append(row['sla_violation_rate']);lookup[row['arm_id']]={k:row[k] for k in ('mode','eta','weight')}
    ranking=[dict(lookup[aid],arm_id=aid,validation_sla=float(np.mean(values))) for aid,values in scores.items()]
    def pick(mode):
        rec=min((r for r in ranking if r['mode']==mode),key=lambda r:(r['validation_sla'],r['eta'],r['weight']));return {k:rec[k] for k in ('mode','eta','weight')}
    uniform,boundary=pick('uniform'),pick('boundary')
    fixed=min((r for r in ranking if r['mode'].startswith('fixed_margin')),key=lambda r:(r['validation_sla'],r['mode']))
    fixed_arm=arm(fixed['mode'],margin=float(fixed['mode'].split('_')[-1]))
    strongest=min((r for r in ranking if r['mode']=='window_original' or r['mode']=='rolling_cdf' or r['mode'].startswith('fixed_margin')),key=lambda r:(r['validation_sla'],r['mode']))
    strongest_arm=arm(strongest['mode'],**({'margin':float(strongest['mode'].split('_')[-1])} if strongest['mode'].startswith('fixed_margin') else {}))
    tests=[arm('window_original'),arm('uncalibrated'),arm('rolling_cdf'),fixed_arm,arm('mean_oracle_original'),uniform,boundary]
    comparisons=[dict(name=name,family='deadline_sla',arm_a=a,arm_b=b,primary_metrics=['sla_violation_rate']) for name,a,b in
                 [('boundary_vs_window',boundary,arm('window_original')),('boundary_vs_uniform',boundary,uniform),('uniform_vs_uncalibrated',uniform,arm('uncalibrated')),('uncalibrated_vs_window',arm('uncalibrated'),arm('window_original')),('boundary_vs_fixed',boundary,fixed_arm),('boundary_vs_rolling',boundary,arm('rolling_cdf'))]]
    # Gate applies only to the preselected boundary candidate, not a test-selected winner.
    strongest_comparison='boundary_vs_window' if strongest_arm['mode']=='window_original' else 'boundary_vs_rolling' if strongest_arm['mode']=='rolling_cdf' else 'boundary_vs_fixed'
    gate=dict(combiner='all',rules=[dict(comparison='boundary_vs_window',metric='sla_violation_rate',scope='overall',field='difference_a_minus_b',op='<=',value=-.3),dict(comparison='boundary_vs_window',metric='sla_violation_rate',scope='overall',field='ci_high',op='<',value=0.),dict(comparison='boundary_vs_window',metric='sla_violation_rate',scope='overall',field='p_holm',op='<=',value=.05),dict(comparison='boundary_vs_uniform',metric='sla_violation_rate',scope='overall',field='ci_high',op='<',value=0.),dict(comparison='boundary_vs_uniform',metric='sla_violation_rate',scope='overall',field='p_holm',op='<=',value=.05),dict(comparison=strongest_comparison,metric='sla_violation_rate',scope='overall',field='ci_high',op='<',value=0.),dict(comparison=strongest_comparison,metric='sla_violation_rate',scope='overall',field='p_holm',op='<=',value=.05)])
    freeze=dict(status='FROZEN',frozen_at_utc=datetime.now(timezone.utc).isoformat(),exploratory=True,code_sha256=code_hashes(),input_sha256={str(BASE_SOURCE):sha(BASE_SOURCE)},selected=dict(uniform=uniform,boundary=boundary,fixed=fixed_arm,strongest_simple=strongest_arm),test_arms=tests,validation_ranking=sorted(ranking,key=lambda r:r['validation_sla']),test_images=list(TEST_IMAGES),test_tasks=list(TEST_TASKS),analysis=dict(rate_columns_are='fraction',primary_comparisons=comparisons,diagnostic_comparisons=[dict(name='mean_oracle_vs_window',arm_a=arm('mean_oracle_original'),arm_b=arm('window_original'))],probe_gate=gate,design={'test':dict(image_seeds=list(TEST_IMAGES),task_seeds=list(TEST_TASKS),rho_levels=list(RHO_VALUES),load_levels=list(LOAD_VALUES),horizon_levels=list(TEST_HORIZONS))}),protocol_note='No online probes unless predeclared no-probe gate passes. Auxiliary cost/P99 review also required; no test retuning.')
    atomic_json(ROOT/'02_development_validation/参数选择.json',dict(ranking=freeze['validation_ranking'],selected=freeze['selected']));return freeze
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('common','validation','test','smoke'),required=True);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    if not 1<=args.workers<=3:raise ValueError('Use 1-3 workers')
    stage=args.stage
    if stage in ('validation','common','smoke'):
        images=DEV_IMAGES if stage!='smoke' else DEV_IMAGES[:1];seeds=DEV_TASKS if stage!='smoke' else DEV_TASKS[:1]
        horizons=DEV_HORIZONS if stage!='smoke' else (24.,);arms=development_arms() if stage=='validation' else [arm('window_original')]
    else:
        freeze=json.loads((ROOT/'00_design_sources/final_freeze.json').read_text(encoding='utf-8'))
        if freeze['code_sha256']!=code_hashes():raise RuntimeError('Frozen code changed')
        if freeze['input_sha256'][str(BASE_SOURCE)]!=sha(BASE_SOURCE):raise RuntimeError('Input changed')
        images=freeze['test_images'];seeds=freeze['test_tasks'];horizons=TEST_HORIZONS;arms=freeze['test_arms']
    rhos=RHO_VALUES if args.stage!='smoke' else (0.,);loads=LOAD_VALUES if args.stage!='smoke' else (.75,)
    folder=ROOT/({'common':'01_common_feedback_diagnostics','validation':'02_development_validation','test':'03_heldout_closed_loop','smoke':'qa/smoke'}[stage])
    if args.stage!='smoke' and (folder/'completion.json').exists():raise RuntimeError('Stage already completed; preserve results')
    moments=prepare_profiles(images);write_csv(folder/'profile_moments.csv',moments)
    fingerprint=hashlib.sha256(json.dumps(code_hashes(),sort_keys=True).encode()).hexdigest()
    jobs=[(stage,i,r,s,l,h,arms,fingerprint,stage in ('common','smoke')) for i in images for r in rhos for s in seeds for l in loads for h in horizons]
    atomic_json(folder/'config.json',dict(stage=args.stage,units=len(jobs),runs=len(jobs)*len(arms),arms=arms,workers=args.workers,source_fingerprint=fingerprint,horizons=list(horizons)))
    start=time.perf_counter();payloads=[]
    with Pool(args.workers) as pool:
        for payload in pool.imap_unordered(run_unit,jobs):
            payloads.append(payload)
            if len(payloads)%8==0 or len(payloads)==len(jobs):print(json.dumps(dict(stage=args.stage,completed_units=len(payloads),total_units=len(jobs),runs=len(payloads)*len(arms),elapsed_s=round(time.perf_counter()-start,2))),flush=True)
    rows=[r for payload in payloads for r in payload['rows']];write_csv(folder/'raw/per_run_results.csv',rows)
    atomic_json(folder/'completion.json',dict(complete=True,units=len(jobs),runs=len(rows),elapsed_s=time.perf_counter()-start,completed_at_utc=datetime.now(timezone.utc).isoformat()))
    if args.stage=='validation':atomic_json(ROOT/'00_design_sources/final_freeze.json',select_and_freeze(rows))
    print('DEADLINE '+args.stage.upper()+' COMPLETE',flush=True)
if __name__=='__main__':main()
