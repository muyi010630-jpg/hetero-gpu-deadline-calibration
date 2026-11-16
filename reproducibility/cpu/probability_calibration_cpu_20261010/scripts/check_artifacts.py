"""Read-only provenance/data checks; only writes qa/artifact_checks.json."""
import argparse, csv, gzip, hashlib, json, math
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--all-trace-hashes',action='store_true');a=p.parse_args()
    freeze=json.loads((ROOT/'00_design_sources/final_freeze.json').read_text(encoding='utf-8'))
    for name,value in freeze['code_sha256'].items():assert sha(ROOT/name)==value,(name,'frozen code changed')
    for name,value in freeze['input_sha256'].items():assert sha(name)==value,(name,'input changed')
    result={'frozen_code_and_input_unchanged':True,'stages':{},'trace_hashes_all_checked':a.all_trace_hashes}
    required={'common':('01_common_feedback_diagnostics',96,96),'validation':('02_development_validation',96,1536),'test':('03_heldout_closed_loop',576,4032)}
    for stage,(folder,units,runs) in required.items():
        directory=ROOT/folder;status=json.loads((directory/'completion.json').read_text(encoding='utf-8'))
        assert status['complete'] and status['units']==units and status['runs']==runs
        with (directory/'raw/per_run_results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==runs and len({r['run_id'] for r in rows})==runs
        groups=defaultdict(set);plans=defaultdict(set);examples={};total_tasks=0
        for r in rows:
            key=tuple(r[k] for k in ('image_seed','rho','task_seed','load','horizon'));groups[key].add(r['plan_sha256'])
            plans[tuple(r[k] for k in ('task_seed','load','horizon'))].add(r['plan_sha256'])
            assert r['arrived']==r['completed'] and float(r['probe_count'])==0 and float(r['privileged_label_count'])==0
            total_tasks+=int(r['completed'])
            for metric in ('sla_violation_rate','prob_brier','prob_logloss','point_mape','p99_completion'):
                assert math.isfinite(float(r[metric]))
            path=ROOT/r['trace_path'];assert path.exists()
            if a.all_trace_hashes:assert sha(path)==r['trace_sha256'],str(path)
            if stage=='test':
                assert r['image_seed'] in {str(i) for i in freeze['test_images']}
                assert r['task_seed'] in {str(i) for i in freeze['test_tasks']}
                assert r['code_fingerprint']==hashlib.sha256(json.dumps(freeze['code_sha256'],sort_keys=True).encode()).hexdigest()
            examples.setdefault(tuple(r[k] for k in ('mode','rho','load','horizon')),r)
        assert len(groups)==units and all(len(x)==1 for x in groups.values())
        assert all(len(x)==1 for x in plans.values()),'Exogenous plan differs across image/rho'
        sampled=0
        for r in examples.values():
            decisions={};completed={};n_panels=0
            with gzip.open(ROOT/r['trace_path'],'rt',encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line)
                    if row['kind']=='decision':
                        assert row['task_id'] not in decisions;decisions[row['task_id']]=row
                    elif row['kind']=='completion':
                        assert row['task_id'] not in completed;completed[row['task_id']]=row
                    elif row['kind']=='panel':n_panels+=1
            assert len(decisions)==len(completed)==int(r['completed']) and n_panels>0
            updates=0
            for task,row in completed.items():
                d=decisions[task];assert row['device']==d['device'] and row['p']==d['p']
                assert row['remaining']==row['deadline']-row['start']
                assert row['on_time']==int(row['observed_s']<=row['remaining'])==1-row['sla_violation']
                feedback=row['learning']['calibration'];assert feedback['label']==row['on_time']
                updates+=int(feedback['updated'])
                if row['remaining']<=0:assert row['p']==0 and not feedback['updated']
            assert updates==int(r['supervised_updates']);sampled+=1
        result['stages'][stage]=dict(units=units,runs=runs,completed_tasks_across_runs=total_tasks,unique_task_plans=len(plans),same_plan_within_and_across_images=True,sampled_traces_fully_parsed=sampled,summary_sha256=sha(directory/'raw/per_run_results.csv'))
    result['passed']=True
    (ROOT/'qa/artifact_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
