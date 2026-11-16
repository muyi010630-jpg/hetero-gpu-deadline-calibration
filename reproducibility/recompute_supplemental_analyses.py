"""Portable post-hoc statistical recomputation from packaged CSVs only.

Uses the numerical functions of the saved analysis scripts, with input/output
paths explicitly redirected. Does not execute simulation, update estimators,
read large NPZ shards, replace original plans, or pretend to be preregistration.
"""
from pathlib import Path
import csv, hashlib, importlib.util, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'recomputed/supplemental_analyses'
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def compare_frames(actual,expected,keys):
    a=pd.read_csv(actual).sort_values(keys).reset_index(drop=True)
    e=pd.read_csv(expected).sort_values(keys).reset_index(drop=True)
    assert list(a.columns)==list(e.columns),(actual,'columns')
    assert a.shape==e.shape,(actual,a.shape,e.shape)
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(e[c]):
            assert np.allclose(a[c],e[c],rtol=1e-12,atol=1e-12,equal_nan=True),(actual,c)
        else:assert a[c].fillna('').equals(e[c].fillna('')),(actual,c)

def recompute_rq2():
    src=ROOT/'supplemental_analyses/RQ2'
    m=module('saved_rq2',src/'reanalyse_rq2.py')
    base=ROOT/'cpu/heterogeneity_adaptive_cpu_20261009'
    m.SOURCES={'fixed':base/'03_independent_prediction_test/image_block_metrics.csv','closed':base/'04_closed_loop_scheduling/full/analysis/image_block_metrics.csv'}
    plan=json.loads((src/'analysis_plan.json').read_text(encoding='utf-8'))
    assert plan['comparison_specs']==m.specs()
    # CSV inputs contain no connection information; verify original recorded hashes.
    for key,p in m.SOURCES.items():assert m.digest(p)==plan['source_hashes'][key]['sha256'],('original input hash',key)
    indices={key:m.load_source(key) for key in m.SOURCES}
    records,blocks=[],[]
    for spec in plan['comparison_specs']:
        if spec['analysis']=='fixed_feedback_interaction':
            images,_,_,balanced=m.paired(indices['fixed'],spec['rho'],'balanced')
            images2,_,_,sparse=m.paired(indices['fixed'],spec['rho'],'late_sparse');assert images==images2
            d=sparse-balanced;pool_mean=window_mean=''
        else:
            key='fixed' if spec['analysis']=='fixed_direct' else 'closed'
            images,pool,window,d=m.paired(indices[key],spec['rho'],spec['condition'])
            pool_mean,window_mean=float(pool.mean()),float(window.mean())
        records.append({**spec,'pool_mean':pool_mean,'window_mean':window_mean,**m.infer(d,spec)})
        blocks.extend({'comparison_id':spec['id'],'image_seed':image,'difference_pp':float(v)} for image,v in zip(images,d))
    for r,p in zip(records,m.holm([r['p_raw'] for r in records])):r['p_holm_25']=p;r['reject_holm_0_05']=p<=.05
    target=OUT/'RQ2';target.mkdir(parents=True,exist_ok=True)
    m.write_csv(target/'rq2_supplementary_comparisons_25.csv',records)
    m.write_csv(target/'rq2_paired_image_differences.csv',blocks)
    compare_frames(target/'rq2_supplementary_comparisons_25.csv',src/'rq2_supplementary_comparisons_25.csv',['id'])
    compare_frames(target/'rq2_paired_image_differences.csv',src/'rq2_paired_image_differences.csv',['comparison_id','image_seed'])
    return dict(comparisons=25,paired_differences=500,original_input_hashes_match=True,saved_numbers_match=True)

def recompute_probability():
    src=ROOT/'supplemental_analyses/diagnostics'
    m=module('saved_diagnostics',src/'summarize_saved_diagnostics.py')
    m.PROB=ROOT/'cpu/probability_calibration_cpu_20261010'
    m.OUT=OUT/'diagnostics';m.OUT.mkdir(parents=True,exist_ok=True)
    # Call only the CSV function, deliberately excluding record_plan_and_inputs,
    # heterogeneity_summary and __main__, which read the omitted 1000 NPZ shards.
    m.probability_summary()
    for filename,keys in [('probability_horizon_image_blocks.csv',['horizon','image_seed','mode']),('probability_horizon_summary.csv',['horizon_s','arm','metric']),('probability_horizon_paired.csv',['horizon_s','arm_a','arm_b','metric'])]:
        compare_frames(m.OUT/filename,src/filename,keys)
    return dict(input_run_rows=1728,image_blocks_per_horizon=16,horizons=[24,120,600],saved_numbers_match=True,no_new_p_values=True)

def check_h_means():
    src=ROOT/'supplemental_analyses/diagnostics'
    streams=pd.read_csv(src/'heterogeneity_per_stream.csv');summary=pd.read_csv(src/'heterogeneity_weight_by_rho_flow.csv')
    assert len(streams)==1000 and len(summary)==10
    means=streams.groupby(['rho','flow_kind'])[['effective_weight_mean_all','no_history_fraction','h_zero_with_history_fraction']].mean()
    for _,r in summary.iterrows():
        got=means.loc[(r['rho'],r['flow_kind'])]
        assert abs(got['effective_weight_mean_all']-r['effective_local_weight_mean_all'])<1e-12
        assert abs(got['no_history_fraction']-r['no_history_fraction'])<1e-12
        assert abs(got['h_zero_with_history_fraction']-r['h_zero_with_history_fraction_all'])<1e-12
    return dict(streams=1000,conditions=10,mean_and_zero_fractions_match=True,event_quantiles_recomputed=False,reason='This CSV entry point checks means and fractions. Run recompute_h_event_diagnostics.py to recompute event quantiles from the included diagnostic extracts.')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=dict(status='verified_from_packaged_CSVs',analysis_nature='post_hoc_supplementary_after_original_holdout',original_estimators_changed=False,new_simulation=False,rq2=recompute_rq2(),probability=recompute_probability(),heterogeneity=check_h_means(),numpy=np.__version__,pandas=pd.__version__)
    (OUT/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
