"""Run paired common-feedback CPU calibration; formal stages require explicit freeze.

Each stream is an atomic resume unit. All methods predict an event before any
method receives that event's feedback. Metrics use common actual own-event counts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import itertools
import json
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
from profile_flows import ROOT, SOURCE, FLOW_DEFINITION, atomic_json, atomic_npz, sha256, load_base_contexts, make_profile, make_stream, base_dict, prepare, late_mask
from calibrator_core import create_calibrator, PRIMARY_MODES, MODES, PARAM_GRID

FOLDERS=dict(smoke='01_mechanism_diagnostics/smoke_v2',pilot='01_mechanism_diagnostics',validation='02_development_validation',test='03_independent_prediction_test',drift='05_drift_tests')
SCHEDULES=dict(smoke=([11001],[0.,1.],[21001],60),pilot=(list(range(11001,11006)),[0.,.5,1.],list(range(21001,21004)),600),validation=(list(range(12001,12007)),[0.,.25,.5,.75,1.],list(range(22001,22004)),600),test=(list(range(13001,13021)),[0.,.25,.5,.75,1.],list(range(23001,23006)),1800),drift=(list(range(13001,13011)),[0.,.5,1.],list(range(23001,23004)),900))
DIAG_FIELDS=('local_n_window','local_updates_count','unique_contexts','effective_group_samples','heterogeneity','local_sampling_variance','age_s','heterogeneity_local_weight','local_freshness_factor','effective_local_weight','group_updates_count','group_z','point_z','raw_point_z','point_clipped','risk_sigma','risk_self_weight','guard_log','guarded_output_clipped','local_window_mean_clipped','point_borrowing_active','point_self_weight')

def configurations(stage, selected=None):
    if stage in ('smoke','pilot'):
        return [dict(label=mode,mode=mode,params={}) for mode in MODES]
    if stage == 'validation':
        configs=[dict(label=mode,mode=mode,params={}) for mode in PRIMARY_MODES if mode != 'adaptive_pool_local']
        for params in PARAM_GRID:
            s,h=params['shrink_scale'],params['freshness_half_life_s']
            configs.append(dict(label=f'adaptive_s{s:g}_h{h:g}',mode='adaptive_pool_local',params=params))
        return configs
    return [dict(label=mode,mode=mode,params=selected if mode=='adaptive_pool_local' else {}) for mode in PRIMARY_MODES]

def formal_freeze(stage, freeze_path):
    if stage not in ('test','drift'):
        return None, None
    if not freeze_path:
        raise RuntimeError('Formal stage requires --freeze after root independent QA and parameter freeze')
    frozen=json.loads(Path(freeze_path).read_text(encoding='utf-8-sig'))
    if frozen.get('status') != 'FROZEN' or frozen.get('qa_approved') is not True:
        raise RuntimeError('Freeze must have status=FROZEN and qa_approved=true')
    code_hashes=frozen.get('code_sha256')
    if not isinstance(code_hashes,dict) or not code_hashes:
        raise RuntimeError('Freeze must include nonempty code_sha256 with ROOT-relative paths')
    required={'scripts/profile_flows.py','scripts/run_fixed_feedback.py','scripts/fixed_statistics.py','scripts/calibrator_core.py'}
    normalized={str(path).replace('\\','/') for path in code_hashes}
    if not required.issubset(normalized):
        raise RuntimeError('Freeze code hashes must cover fixed runner, profile/flow, statistics, and calibrator scripts')
    for relative_path,expected in code_hashes.items():
        relative=Path(relative_path)
        if relative.is_absolute():
            raise RuntimeError('Freeze code paths must be relative: '+str(relative_path))
        target=(ROOT/relative).resolve()
        if not target.is_relative_to(ROOT.resolve()) or not target.is_file() or sha256(target)!=expected:
            raise RuntimeError('Frozen code hash mismatch: '+str(relative_path))
    selected=frozen.get('selected_params')
    if selected not in PARAM_GRID:
        raise RuntimeError('selected_params must exactly match one predeclared grid configuration')
    if stage=='drift' and not (ROOT/FOLDERS['test']/'completion.json').exists():
        raise RuntimeError('Drift sensitivity starts only after static formal test completion')
    return selected, dict(path=str(Path(freeze_path).resolve()),sha256=sha256(freeze_path),payload=frozen)

def atomic_csv(path, rows):
    rows=list(rows)
    if not rows:
        return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(temp,path)

def macro(ape, ids, mask):
    cells=np.unique(ids[mask])
    if not len(cells):
        return float('nan'),0,0
    return float(np.mean([ape[mask & (ids==cid)].mean() for cid in cells])),len(cells),int(mask.sum())

def metrics(predictions, stream, configs, metadata, diagnostic, feedback):
    ape=100.0*np.abs(predictions-stream['obs'][:,None])/stream['obs'][:,None]
    n=len(ape); ids=stream['context_ids']; own=stream['n_before']
    masks=dict(overall=np.ones(n,dtype=bool),cold_phase=np.arange(n)/max(1,n-1)<.2,pre_drift=np.arange(n)<n//2,post_drift=np.arange(n)>=n//2)
    masks.update({f'first{k}':own<k for k in (1,3,5,10)})
    masks.update(dict(own_n0=own==0,own_n1_3=(own>=1)&(own<=3),own_n4_7=(own>=4)&(own<=7),own_n8_plus=own>=8))
    out=[]
    for j,config in enumerate(configs):
        row=dict(metadata,mode=config['label'],core_mode=config['mode'],shrink_scale=config['params'].get('shrink_scale',''),freshness_half_life_s=config['params'].get('freshness_half_life_s',''))
        for label,mask in masks.items():
            value,ncontexts,nevents=macro(ape[:,j],ids,mask)
            row[label+'_macro_mape']=value
            row[label+'_micro_mape']=float(ape[mask,j].mean()) if nevents else float('nan')
            row[label+'_n_contexts']=ncontexts;row[label+'_n_events']=nevents
        row['p95_ape']=float(np.sort(ape[:,j])[int(np.ceil(.95*n))-1])
        row['prediction_point_clip_count']=int(diagnostic[:,j,DIAG_FIELDS.index('point_clipped')].sum())
        row['prediction_output_clip_count']=int(diagnostic[:,j,DIAG_FIELDS.index('guarded_output_clipped')].sum())
        row['prediction_local_window_clip_count']=int(diagnostic[:,j,DIAG_FIELDS.index('local_window_mean_clipped')].sum())
        for k,name in enumerate(feedback['fields']):
            row['feedback_'+name+'_count']=int(feedback['values'][:,j,k].sum())
        out.append(row)
    return out

def run_stream(stage, folder, image_seed, rho, task_seed, flow_kind, drift_type, n_events, configs, run_fingerprint, representative=False):
    name=f'i{image_seed}_r{rho:g}_t{task_seed}_{flow_kind}_{drift_type}'
    out=folder/'raw'/f'{name}.npz'; summary=folder/'per_stream'/f'{name}.csv';done=folder/'per_stream'/f'{name}.json'
    if done.exists():
        status=json.loads(done.read_text(encoding='utf-8'))
        if status['run_fingerprint']!=run_fingerprint:
            raise RuntimeError('Resume fingerprint changed: '+name)
        if out.exists() and summary.exists() and sha256(out)==status['npz_sha256'] and sha256(summary)==status['summary_sha256']:
            return False
        raise RuntimeError('Completed stream checksum mismatch: '+name)
    rows=load_base_contexts();profile=make_profile(image_seed,rho,rows,persist=False)
    stream=make_stream(profile,task_seed,flow_kind,n_events,drift_type,rows)
    cals=[create_calibrator(base_dict(profile),config['mode'],config['params']) for config in configs]
    pred=np.empty((n_events,len(cals)),dtype=float)
    diag=np.zeros((n_events,len(cals),len(DIAG_FIELDS)),dtype=float)
    feedback_fields=('local_innovation_clipped','local_state_clipped','pool_innovation_clipped','pool_state_clipped','pool_dtype_innovation_clipped','pool_dtype_state_clipped','pool_exposure_innovation_clipped','pool_exposure_state_clipped')
    fb=np.zeros((n_events,len(cals),len(feedback_fields)),dtype=np.uint8)
    for i,cid in enumerate(stream['context_ids']):
        ctx=rows[int(cid)]['context'];device=rows[int(cid)]['worker'];now=float(stream['now'][i]);obs=float(stream['obs'][i])
        for j,cal in enumerate(cals):
            cal.set_now(now)
            pred[i,j],_=cal.predict(ctx,device,qos=0,guarded=False)
            info=cal.diagnostics(ctx,device)
            for k,key in enumerate(DIAG_FIELDS):
                diag[i,j,k]=info.get(key,0.)
            assert info['local_updates_count']==stream['n_before'][i], 'Actual target feedback count mismatch'
            assert info['group_updates_count']==stream['group_n_before'][i], 'Actual group feedback count mismatch'
        for j,cal in enumerate(cals):
            flags=cal.observe(ctx,device,obs,float(pred[i,j]),now)
            for k,key in enumerate(feedback_fields):
                fb[i,j,k]=bool((flags or {}).get(key,False))
    assert np.isfinite(pred).all() and (pred>0).all()
    metadata=dict(stage=stage,image_seed=image_seed,rho=rho,task_seed=task_seed,flow_kind=flow_kind,drift_type=drift_type,n_events=n_events)
    atomic_npz(out,**stream,predictions=pred,mode_labels=np.asarray([x['label'] for x in configs]),diagnostics=diag,diagnostic_fields=np.asarray(DIAG_FIELDS),feedback_clip_flags=fb,feedback_clip_fields=np.asarray(feedback_fields),initial_s=profile['initial_s'],image_true_s=profile['true_s'],z_log=profile['z_log'],image_seed=np.asarray(image_seed),rho=np.asarray(rho),run_fingerprint=np.asarray(run_fingerprint))
    atomic_csv(summary,metrics(pred,stream,configs,metadata,diag,dict(fields=feedback_fields,values=fb)))
    if representative:
        trace=[]
        for i,cid in enumerate(stream['context_ids']):
            for j,config in enumerate(configs):
                row=dict(metadata,event_id=i,now=float(stream['now'][i]),context_id=int(cid),context_key=rows[int(cid)]['key'],mode=config['label'],initial_s=float(profile['initial_s'][cid]),true_median=float(stream['true_median'][i]),obs=float(stream['obs'][i]),prediction=float(pred[i,j]),ape=100*abs(pred[i,j]-stream['obs'][i])/stream['obs'][i],n_before=int(stream['n_before'][i]),group_n_before=int(stream['group_n_before'][i]))
                row.update({key:float(diag[i,j,k]) for k,key in enumerate(DIAG_FIELDS)})
                row.update({key:int(fb[i,j,k]) for k,key in enumerate(feedback_fields)})
                trace.append(row)
        atomic_csv(folder/'representative_trace.csv',trace)
    atomic_json(done,dict(run_fingerprint=run_fingerprint,npz_sha256=sha256(out),summary_sha256=sha256(summary),event_count=n_events,mode_count=len(configs),paired_input_sha256=hashlib.sha256(np.column_stack((stream['context_ids'],stream['obs'],stream['now'])).tobytes()).hexdigest()))
    return True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=FOLDERS,required=True);parser.add_argument('--freeze');parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,choices=(1,2,3),default=1);args=parser.parse_args()
    stage=args.stage;folder=ROOT/FOLDERS[stage];folder.mkdir(parents=True,exist_ok=True)
    selected,frozen=formal_freeze(stage,args.freeze);configs=configurations(stage,selected)
    images,rhos,tasks,n_events=SCHEDULES[stage]
    sources={str(path):sha256(path) for path in (SOURCE,Path(__file__),Path(__file__).parent/'profile_flows.py',Path(__file__).parent/'calibrator_core.py',ROOT/'00_protocol_sources'/'development_protocol.json',ROOT/'00_protocol_sources'/'frozen_sources'/'hetero_qos_sim_n2.py')}
    config=dict(stage=stage,image_seeds=images,rhos=rhos,task_seeds=tasks,n_events=n_events,flow_definition=FLOW_DEFINITION,configurations=configs,source_hashes=sources,freeze=frozen,processes=args.workers,primary_metric='context-macro MAPE over observed common context cells; report observed cell count',first_k='own prior actual feedback count n_before<k',diagnostic_fields=DIAG_FIELDS)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    config['run_fingerprint']=fingerprint
    prior=folder/'run_config.json'
    if prior.exists() and json.loads(prior.read_text(encoding='utf-8'))['run_fingerprint']!=fingerprint:
        raise RuntimeError('Stage config changed; preserve existing data and use a distinct output directory')
    atomic_json(prior,config)
    for source in sources:
        target=folder/'frozen_sources'/Path(source).name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    atomic_csv(folder/'profile_moments.csv',prepare(stage,images,rhos))
    drift_types=('group','context') if stage=='drift' else ('none',)
    schedule=list(itertools.product(images,rhos,tasks,('balanced','late_sparse'),drift_types))
    total=len(schedule);t0=time.monotonic();new=0
    active_schedule=schedule[:args.limit] if args.limit else schedule
    if args.workers==1:
        for index,(image,rho,task,flow,drift) in enumerate(active_schedule):
            new+=run_stream(stage,folder,image,rho,task,flow,drift,n_events,configs,fingerprint,representative=index==0)
            if (index+1)%5==0 or index==0 or index+1==total:
                print(json.dumps(dict(stage=stage,streams=index+1,total=total,elapsed_s=round(time.monotonic()-t0,2)),ensure_ascii=False),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures=[executor.submit(run_stream,stage,folder,image,rho,task,flow,drift,n_events,configs,fingerprint,index==0) for index,(image,rho,task,flow,drift) in enumerate(active_schedule)]
            for index,future in enumerate(as_completed(futures)):
                new+=future.result()
                if (index+1)%5==0 or index==0 or index+1==total:
                    print(json.dumps(dict(stage=stage,streams=index+1,total=total,workers=args.workers,elapsed_s=round(time.monotonic()-t0,2)),ensure_ascii=False),flush=True)
    all_rows=[]
    for path in sorted((folder/'per_stream').glob('*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as handle:
            all_rows.extend(csv.DictReader(handle))
    atomic_csv(folder/'stream_metrics.csv',all_rows)
    if not args.limit or args.limit>=total:
        assert len(list((folder/'per_stream').glob('*.json')))==total
        atomic_json(folder/'completion.json',dict(stage=stage,n_streams=total,n_events_per_stream=n_events,n_modes=len(configs),prediction_rows=total*n_events*len(configs),run_fingerprint=fingerprint,elapsed_s=time.monotonic()-t0,new_streams=new))
    print(json.dumps(dict(stage=stage,saved_summary=str(folder/'stream_metrics.csv'),completed_rows=len(all_rows)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
