"""Aggregate task streams within independent portrait blocks before inference.

Validation selects one frozen configuration. Pilot and drift are diagnostic.
Formal primary family is 3 comparators x 5 rho x 2 flow = 30 Holm tests.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from profile_flows import ROOT, atomic_json, sha256
from run_fixed_feedback import FOLDERS, atomic_csv

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def finite_mean(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float('nan')

def block_aggregate(rows):
    grouped=defaultdict(list)
    metrics=[key for key in rows[0] if key.endswith(('_macro_mape','_micro_mape','_count')) or key=='p95_ape']
    for row in rows:
        key=(int(row['image_seed']),float(row['rho']),row['flow_kind'],row['drift_type'],row['mode'])
        grouped[key].append(row)
    blocks=[]
    for (image,rho,flow,drift,mode),values in sorted(grouped.items()):
        out=dict(image_seed=image,rho=rho,flow_kind=flow,drift_type=drift,mode=mode,n_task_streams=len(values))
        out.update({metric:finite_mean([row[metric] for row in values]) for metric in metrics})
        blocks.append(out)
    return blocks,metrics

def condition_aggregate(blocks,metrics):
    grouped=defaultdict(list)
    for row in blocks:
        grouped[(row['rho'],row['flow_kind'],row['drift_type'],row['mode'])].append(row)
    result=[]
    for (rho,flow,drift,mode),values in sorted(grouped.items()):
        row=dict(rho=rho,flow_kind=flow,drift_type=drift,mode=mode,n_image_blocks=len(values))
        for metric in metrics:
            row[metric]=finite_mean([value[metric] for value in values])
        result.append(row)
    return result

def select_validation(blocks,folder):
    candidates=sorted({row['mode'] for row in blocks if row['mode'].startswith('adaptive_s')})
    objectives=[]
    config=json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
    param_by_label={value['label']:value['params'] for value in config['configurations']}
    for mode in candidates:
        # Complete rectangular image x rho x flow matrix; inner task average already done.
        values=[row for row in blocks if row['mode']==mode]
        assert len(values)==6*5*2 and all(row['n_task_streams']==3 for row in values)
        params=param_by_label[mode]
        objectives.append(dict(mode=mode,shrink_scale=params['shrink_scale'],freshness_half_life_s=params['freshness_half_life_s'],objective_macro_mape=float(np.mean([row['overall_macro_mape'] for row in values])),n_image_blocks=6,n_conditions=10,n_task_streams=180))
    objectives.sort(key=lambda row:(row['objective_macro_mape'],row['shrink_scale'],row['freshness_half_life_s']))
    atomic_csv(folder/'validation_candidates.csv',objectives)
    winner=objectives[0]
    selection=dict(status='VALIDATION_SELECTION_ONLY_NOT_FORMAL_FREEZE',selection_metric='all validation image/rho/flow equally weighted context-macro MAPE; inner task seeds equally weighted',selected_label=winner['mode'],selected_params={key:winner[key] for key in ('shrink_scale','freshness_half_life_s')},selected_objective_macro_mape=winner['objective_macro_mape'],tie_break='objective then parameter tuple ascending',source_stream_metrics_sha256=sha256(folder/'stream_metrics.csv'),run_fingerprint=config['run_fingerprint'],all_candidates=objectives)
    atomic_json(folder/'validation_selection.json',selection)
    return selection

def holm(rows,pkey='p_signflip'):
    ranked=sorted(range(len(rows)),key=lambda i:rows[i][pkey]);previous=0.
    for rank,index in enumerate(ranked):
        value=min(1.,(len(rows)-rank)*rows[index][pkey]);previous=max(previous,value)
        rows[index]['p_holm']=previous
        rows[index]['holm_family_size']=len(rows)
    return rows

def inference(blocks,stage,folder):
    rng=np.random.default_rng(20261009)
    by_key={(row['image_seed'],row['rho'],row['flow_kind'],row['drift_type'],row['mode']):row for row in blocks}
    images=sorted({row['image_seed'] for row in blocks});rhos=sorted({row['rho'] for row in blocks});flows=('balanced','late_sparse')
    drifts=('group','context') if stage=='drift' else ('none',)
    n=len(images);bootstrap_indices=rng.integers(0,n,size=(5000,n))
    # Use common random signs across all tests; independent portrait blocks are flipped.
    signs=(rng.integers(0,2,size=(100000,n),dtype=np.int8)*2-1)
    output=[]
    for rho,flow,drift,comparator in itertools.product(rhos,flows,drifts,('pool_ema','local_window','full_original')):
        adaptive=np.asarray([by_key[(image,rho,flow,drift,'adaptive_pool_local')]['overall_macro_mape'] for image in images])
        reference=np.asarray([by_key[(image,rho,flow,drift,comparator)]['overall_macro_mape'] for image in images])
        diff=adaptive-reference;point=float(diff.mean())
        boot=diff[bootstrap_indices].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
        null=(signs*diff).mean(axis=1)
        p=(1+np.count_nonzero(np.abs(null)>=abs(point)-1e-12))/(len(null)+1)
        output.append(dict(stage=stage,rho=rho,flow_kind=flow,drift_type=drift,metric='context-macro MAPE',candidate='adaptive_pool_local',comparator=comparator,candidate_mean=float(adaptive.mean()),comparator_mean=float(reference.mean()),candidate_minus_comparator_pp=point,ci_lower_pp=float(lo),ci_upper_pp=float(hi),p_signflip=float(p),n_image_blocks=n,n_negative_images=int(np.sum(diff<0)),n_positive_images=int(np.sum(diff>0)),bootstrap_draws=5000,signflip_draws=100000,bootstrap_rng_seed=20261009,ci_kind='marginal percentile image-block CI',negative_difference_means='candidate has lower error',interpretation='primary' if stage=='test' else 'exploratory drift sensitivity'))
    if stage=='test':
        assert n==20 and len(output)==30
    holm(output)
    atomic_csv(folder/('primary_paired_inference.csv' if stage=='test' else 'exploratory_paired_inference.csv'),output)
    differences=[]
    for row in output:
        for image in images:
            key=(image,row['rho'],row['flow_kind'],row['drift_type'])
            a=by_key[key+('adaptive_pool_local',)]['overall_macro_mape'];b=by_key[key+(row['comparator'],)]['overall_macro_mape']
            differences.append(dict(image_seed=image,rho=row['rho'],flow_kind=row['flow_kind'],drift_type=row['drift_type'],comparator=row['comparator'],adaptive_macro_mape=a,comparator_macro_mape=b,difference_pp=a-b))
    atomic_csv(folder/'paired_image_differences.csv',differences)
    return output

def write_report(stage,folder,blocks,conditions,selection=None,infer=None):
    modes=sorted({row['mode'] for row in blocks})
    lines=[f'# {stage} 固定反馈分析', '', '所有方法逐事件使用完全相同的观测反馈，先预测后更新。每个 stream 的主指标先在有观测的唯一 context-device 单位内计算 MAPE，再对这些单位等权平均；未观测单位不凭空填误差，其数量保存在逐流 CSV。随后对同一画像内的任务种子等权汇总，画像是推断单位。', '', 'balanced 为45单位随机排列循环；late_sparse 为由公共 SHA256 排序预定的22单位，前40%不出现，随后每单位权重为其他单位的1/4。任务序列和观测噪声跨画像/rho匹配。rho 是构造后的45维组间方差占比，不解释为 iid 相关系数。', '', 'first-k 采用目标单位实际累计反馈数 n_before<k；目标 n=0 时可能已有同组反馈。cold_phase 为流进度前20%。二者不是同一定义。P95 为逐流事件 APE 的 nearest rank 95分位，其画像汇总是各流P95平均，不是所有事件混合分位。', '', '| mode | 全条件等权 context-macro MAPE (%) |', '|---|---:|']
    for mode in modes:
        values=[row['overall_macro_mape'] for row in conditions if row['mode']==mode]
        lines.append(f'| {mode} | {np.mean(values):.8f} |')
    if selection:
        lines += ['', '验证集选择（尚不构成正式冻结）：`'+json.dumps(selection['selected_params'],ensure_ascii=False)+'`，全ρ/流/画像等权 macro MAPE='+f"{selection['selected_objective_macro_mape']:.8f}%。", '正式静态/漂移测试必须由 root 保存 FROZEN + qa_approved=true 的文件后才允许启动。候选选择不改变基线窗口、alpha、正式条件或测试种子。']
    if infer:
        lines += ['', '差值为 Adaptive−对照，负值表示候选误差较小。5000次画像块 paired percentile bootstrap；100000次画像块 paired signflip，p使用+1校正。置信区间为边际95%区间，不能视为同时区间。'+('30个主比较共同Holm校正。' if stage=='test' else '漂移比较为探索性单独Holm族，不与静态主族混合。'), 'Signflip 依赖配对零假设下画像块差值可交换/符号对称；独立重复来自随机受控画像，而非真实GPU硬件或独立context。']
    lines += ['', '裁剪计数、点残差与局部/组更新数保存在全部预测NPZ。原始窗口所有方法共同维护；风险估计采用公共原Full wrapper，但此处使用 unguarded point predictions，不据此推断SLA或guard覆盖率。', '', '此补实验是在先前结果之后设计的结构与候选验证；pilot 为机制诊断，不据其排名改变正式矩阵。画像基于同一套实测median，控制的是基准失配结构，不能代表未经建模的所有迁移环境。']
    (folder/'analysis_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=FOLDERS,required=True);args=parser.parse_args();stage=args.stage;folder=ROOT/FOLDERS[stage]
    if not (folder/'completion.json').exists():
        raise RuntimeError('Aggregate only complete stages')
    rows=read_csv(folder/'stream_metrics.csv');blocks,metrics=block_aggregate(rows);conditions=condition_aggregate(blocks,metrics)
    atomic_csv(folder/'image_block_metrics.csv',blocks);atomic_csv(folder/'condition_summary.csv',conditions)
    selection=select_validation(blocks,folder) if stage=='validation' else None
    infer=inference(blocks,stage,folder) if stage in ('test','drift') else None
    write_report(stage,folder,blocks,conditions,selection,infer)
    atomic_json(folder/'analysis_completion.json',dict(stage=stage,n_stream_method_rows=len(rows),n_image_condition_mode_rows=len(blocks),n_condition_mode_rows=len(conditions),statistics_script_sha256=sha256(__file__),input_sha256=sha256(folder/'stream_metrics.csv'),independent_unit='image block',selection=selection['selected_params'] if selection else None,n_inference_rows=len(infer) if infer else 0))
    print(json.dumps(dict(stage=stage,blocks=len(blocks),conditions=len(conditions),selected=selection['selected_params'] if selection else None),ensure_ascii=False))

if __name__=='__main__':
    main()
