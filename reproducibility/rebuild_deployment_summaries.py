"""Recompute deployment absolute summaries from saved 60-run metrics, without GPUs.

Original script nested a batch-prefilter around a two-batch loop. In the empty
second pass it relabeled the preceding six rows. This script groups directly by
batch/load/strategy and never relabels an already aggregated object.
"""
from pathlib import Path
import csv, statistics, hashlib, json
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'deployment/per_run_summary_10seed.csv'
OUT=ROOT/'recomputed'
METRICS=['mape_pct','p95_ape_pct','qos_bound_coverage_pct','false_safe_rate','gpu_sla_violation_rate','e2e_sla_violation_rate','strict_gpu_sla_violation_rate','mean_flow_time_s','p95_flow_time_s','p99_flow_time_s','device_jain_mean','controller_decision_p95_s','rtx5070_tasks','t4_tasks','a100_tasks']
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(rows,by_batch):
    keys=(['batch'] if by_batch else [])+['load','strategy']; groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for k,rs in sorted(groups.items()):
        record=dict(zip(keys,k));record['n_runs']=len(rs);record['tasks_total']=sum(int(r['success_tasks']) for r in rs)
        for m in METRICS:
            v=[float(r[m]) for r in rs];record[m+'_mean']=round(statistics.mean(v),6);record[m+'_std']=round(statistics.stdev(v),6) if len(v)>1 else 0.
        out.append(record)
    return out
def main():
    rows=read(SOURCE);assert len(rows)==60
    bybatch=summarize(rows,True);merged=summarize(rows,False)
    assert len(bybatch)==12 and all(r['n_runs']==5 for r in bybatch)
    assert len(merged)==6 and all(r['n_runs']==10 for r in merged)
    for name,actual in [('summary_by_load_strategy_by_batch_corrected.csv',bybatch),('summary_by_load_strategy_10seed_rebuilt.csv',merged)]:
        expected=read(ROOT/'deployment'/name)
        for a,b in zip(actual,expected):
            for key,value in a.items():
                if isinstance(value,(float,int)):assert abs(value-float(b[key]))<1e-9,(name,key,value,b[key])
                else:assert value==b[key]
        OUT.mkdir(exist_ok=True);write(OUT/name,actual)
    (OUT/'verification.json').write_text(json.dumps({'input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'n_runs':60,'batch_groups':12,'merged_groups':6,'saved_results_match':True},indent=2),encoding='utf-8')
    print('Verified: 60 runs; 12 batch groups; 6 merged absolute summaries match saved corrected tables.')
if __name__=='__main__':main()
