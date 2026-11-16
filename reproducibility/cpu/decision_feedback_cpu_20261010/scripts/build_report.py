"""Create a readable decision report from frozen heldout summaries, no model changes."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES={'local_window':'局部窗口','pool_ema':'组级池化EMA','full_original':'原分层Full',
       'uniform_adapter':'窗口＋普通监督修正','decision_weighted':'窗口＋决策加权修正',
       'rank_history':'窗口＋历史排序修正','uniform_fullinfo':'充分反馈普通修正（特权）',
       'rank_fullinfo':'充分反馈排序修正（特权）','perfect_information':'同调度器真值参照（特权）'}
POLICIES={'no_probe':'不探测','random':'随机探测','uncertainty':'少样本/波动探测',
          'gated_random':'关键门控后随机','critical':'决策关键性探测'}
COMPARE={'decision_update_vs_window':'合法决策更新 − 局部窗口',
         'critical_feedback_vs_uncertainty':'窗口关键探测 − 窗口少样本/波动探测',
         'combined_vs_validation_probe_baseline':'联合机制 − 验证选出的最强窗口策略',
         'decision_objective_vs_uniform_adapter':'合法决策更新 − 普通监督修正',
         'full_information_ranking_vs_uniform':'充分反馈排序 − 充分反馈普通修正',
         'combined_vs_uniform_same_probe':'联合机制 − 普通修正＋同样关键探测'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    freeze=json.loads((ROOT/'00_design_sources/final_freeze.json').read_text(encoding='utf-8'))
    raw=pd.read_csv(ROOT/'02_heldout_closed_loop/raw/per_run_results.csv')
    pairs=pd.read_csv(ROOT/'analysis/test/paired_comparisons.csv')
    primary=pairs[pairs.primary.astype(str).str.lower().eq('true')].copy()
    combined=primary[primary.comparison=='combined_vs_validation_probe_baseline'].iloc[0]
    objective=primary[primary.comparison=='decision_objective_vs_uniform_adapter'].iloc[0]
    group=raw.groupby(['mode','policy','eta','ranking_weight','image_seed','rho','load'],as_index=False).mean(numeric_only=True)
    imgs=group.groupby(['mode','policy','eta','ranking_weight','image_seed'],as_index=False).mean(numeric_only=True)
    summary=imgs.groupby(['mode','policy','eta','ranking_weight'],as_index=False).mean(numeric_only=True)
    summary=summary.drop(columns=['image_seed','rho','load','seed','task_seed'],errors='ignore')
    selected=freeze['selected_legal']
    joint=summary[(summary['mode']==selected['mode']) & (summary.policy=='critical')].iloc[0]
    window=summary[(summary['mode']=='local_window') & (summary.policy=='no_probe')].iloc[0]
    oracle=summary[summary['mode']=='perfect_information'].iloc[0]
    cost_ratio=100*joint.probe_positive_service_cost_s/joint.total_actual_service_s
    p99_delta=joint.p99_completion-window.p99_completion
    # The protocol fixed the SLA screening threshold, but no numerical cost/P99 gate.
    # Cost and tail latency are reported for substantive judgment, not a post-hoc pass rule.
    promising=(combined.difference_a_minus_b<=-.3 and combined.ci_high<0)
    if promising:
        decision='达到预定SLA筛查门槛；是否继续真实GPU验证还须结合探测成本、尾延迟和多重比较结果判断。'
    elif combined.difference_a_minus_b<0 and combined.ci_high<0:
        decision='存在小幅积极信号，尚未达到预定实用收益门槛；先做有限机制诊断，不建议立即扩大云端实验。'
    else:
        decision='当前冻结实现未建立相对强简单基线的整体收益，不建议立即投入大规模真实GPU重跑；先查明瓶颈，再决定是否开发下一版。'
    rows=[]
    order=['local_window','pool_ema','full_original','uniform_adapter','decision_weighted','rank_history',
           'uniform_fullinfo','rank_fullinfo','perfect_information']
    for mode in order:
        for _,r in summary[summary['mode']==mode].sort_values('policy').iterrows():
            rows.append([NAMES[mode],POLICIES[r.policy],f'{100*r.sla_violation_rate:.3f}',f'{r.p99_completion:.4f}',
                         f'{100*r.prediction_mape:.3f}',f'{r.decision_score_regret_mean:.4f}',f'{100*r.probe_fraction:.2f}'])
    primary_rows=[[COMPARE[r.comparison],f'{r.difference_a_minus_b:+.3f}',
                   f'[{r.ci_low:+.3f}, {r.ci_high:+.3f}]',f'{r.p_holm:.5f}'] for _,r in primary.iterrows()]
    cost_rows=[]
    for _,r in summary[summary.policy!='no_probe'].iterrows():
        subset=raw[(raw['mode']==r['mode'])&(raw.policy==r.policy)]
        ratios=np.where(subset.probe_predicted_budget_s>0,subset.probe_positive_service_cost_s/subset.probe_predicted_budget_s,0)
        over=float(np.mean(ratios>1+1e-12))
        cost_rows.append([NAMES[r['mode']],POLICIES[r.policy],f'{r.probe_count:.2f}',f'{r.probe_budget_used_s:.4f}',
                          f'{r.probe_positive_service_cost_s:.4f}',f'{100*r.probe_positive_service_cost_s/r.total_actual_service_s:.2f}',f'{100*over:.1f}'])
    params=table(['模式','学习率eta','排序权重'],[[NAMES[k],v['eta'],v['ranking_weight']] for k,v in freeze['selected'].items()])
    text=f'''# 决策导向校准与有限预算反馈：CPU可行性报告

## 判断

**{decision}**

合法联合机制为“{NAMES[selected['mode']]}＋决策关键性探测”。相对验证阶段选出的最强Window策略，留出整体SLA差为{combined.difference_a_minus_b:+.3f}个百分点，画像块95%CI [{combined.ci_low:+.3f}, {combined.ci_high:+.3f}]，六项比较Holm p={combined.p_holm:.5f}。负差表示违约更少。决策更新相对普通监督adapter差为{objective.difference_a_minus_b:+.3f}个百分点，CI [{objective.ci_low:+.3f}, {objective.ci_high:+.3f}]，Holm p={objective.p_holm:.5f}，该项未通过校正检验。

联合方法的实际正探索服务增量/总实际服务的描述比值为{cost_ratio:.2f}%；相对Window不探索的P99差为{p99_delta:+.4f}秒。该成本是同一次选择相对默认设备的潜在服务差，不包含后续队列外部性；SLA/P99比较包含完整闭环影响。

本轮预定0.3个百分点实用筛查门槛，不能把小幅统计变化自动视为值得重构论文。方向的可行性与当前实现的有效性应区分；负结果不否定所有决策导向方法。

## 完成范围

仅使用本机CPU，3个工作进程；12逻辑CPU、31.06 GiB内存。独立开发验证48配对单元×29配置=1392运行，留出288单元×17配置=4896运行，合计6288次闭环运行。全部条件保留，测试开始后没有再改方法/参数/场景。真实GPU在线实验与论文Word均未改。

基础为45个唯一型号×上下文的既有实测画像；8个Worker是2个RTX5070、3个T4、3个A100的仿真副本。初始log失配均值0、标准差0.20，rho=0/0.5/1；24秒到达窗口，load=0.75/1.70，无实际漂移、无异常注入。随机种子与旧实验分离，但仍复用同一实测基础画像与任务生成器，不能视为新硬件域验证。

每方法使用相同外生任务计划、潜在设备服务时长和截止期。16画像是统计单位，先平均3任务种子，再平均6个预定条件。区间条件于这组实测基础画像与共同任务流，不能推广成所有真实GPU的概率保证。

## 留出整体结果

表中MAPE是实际被选设备的任务频率均值，选中任务集合内生改变，因此不同方法的MAPE不能单独作相同样本精度比较；公共固定面板误差在CSV中另列，面板不回流训练。评分差是当前空闲兼容候选中的实际评分遗憾，不是全局最优调度遗憾。

{table(['校准','反馈策略','SLA(%)','P99(s)','已选任务MAPE(%)','评分遗憾','探测比例(%)'],rows)}

## 六项预定主比较

差值单位为百分点；CI为16画像块的配对边际95%bootstrap区间，不是同时区间。全部六项整体SLA检验采用双侧精确sign-flip并共同Holm校正；探索性筛查不等于最终确认性试验。未检出差异不表示等效。

{table(['A−B','SLA差(pp)','95%CI','Holm p'],primary_rows)}

## 探测代价与预算兑现

预算规则对所有探测策略相同：派发比例上限5%，预计额外服务预留上限为累计参考预测服务的2%，每单元冷却1秒，预测慢化最多1.20倍；严格任务紧迫时停止。保留任务排序规则，探测只改当前任务设备；后续反馈、队列和任务排序可内生改变。少样本/波动对照是原RC优先级的受控设备探测版本，不是包含任务重排的原RC完整系统。

这些是预计成本约束，实际正探索增量可以超出预计预算。下表全部报告，不把预计上限写成真实GPU成本保证。平均探测次数是每流均值；成本份额为整体均值的描述比值，超预算比例是逐流指示值均值。

{table(['校准','探测','平均次数','预计预留(s)','实际正增量(s)','实际服务占比(%)','超预计预算流(%)'],cost_rows)}

独立轨迹核对发现，Window随机探测与门控随机在288/288个留出单元的完整解压轨迹相同。重建的4538个可行探测候选全部被评分范围重叠条件放行，门控改变集合0次。只允许预计慢化≤1.20倍的近邻，再用较宽的早期经验范围判断重叠，可能导致缺乏区分力；这里不是同型号副本直接触发门控，因为探测已排除默认型号。详见qa/门控退化诊断.md。

本设计下门控没有筛出不同探测对象，不能作为门控必要性证据，也不能推广成这两类算法普遍等价。该退化限制了当前关键性代理的辨别力，后续若开发须先修复并在新留出数据验证。

## 机制与信息边界

LocalWindow保留24条局部对数残差，风险规则沿用原Full。新增校准器在窗口预测上加有界log adapter（±0.25，最终log输出±0.70），普通监督log平方误差锚点与L2=0.01保持时间量纲。决策加权只用派发前QoS、评分接近程度和截止期歧义调权，最大权重5。

排序目标针对现有设备评分 projected +0.65×miss +0.08×load，而不只针对执行时长。平滑margin=0.02、temperature=0.05，按标签评分差封顶1加权，每次完成的pair梯度取均值，使用派发时保存的预测做延迟更新。本实现是轻量可行性代理，不是原版SPO+，也没有直接优化全局SLA。

合法历史排序只能使用选定设备完成观测和派发前其他候选已有的、年龄≤6秒的窗口均值；这些均值不是同任务的反事实标签。零历史时不能凭空生成正确排名。排序pair包含空闲Worker副本，pair次数不是独立样本量。

uniform_fullinfo与rank_fullinfo仅诊断潜在方向：任务完成后免费得到派发时所有兼容空闲设备的潜在时长，普通/排序模式具有同类型特权信息，但闭环反馈和候选集合会内生不同。其额外标签只更新adapter，不进入共同局部窗口和风险。不得把这些结果冒充真实在线方法。perfect_information也是同一启发式的真值参照，不是全局最优上界。

## 参数冻结与核查

{params}

开发按整体等权SLA选参，排序和普通监督各自从同一学习率网格选择。开发结束、留出开始前强化了对照：最强Window选择纳入不探测，并增加普通adapter＋相同关键探测。最终验证最强Window为不探测。旧配置和源码精确归档，模拟计算AST不变；该修正未使用留出结果。

冻结时间UTC：{freeze['frozen_at_utc']}。代码/协议/统计脚本和来源哈希见final_freeze.json。基线29个指标与原调度器精确一致；在线读取栅栏、loss有限差分、特权标签隔离、每派发预算、共同风险和完整数据核验结果在qa目录。

## 对论文定位的含义

只有胜过Window及普通监督修正，并在计入成本后改善系统指标，才支持将决策导向更新或反馈获取升级为方法贡献。若只有更复杂预测或特权标签版本有效，则应先解决信息获取/损失匹配，不能提前写“精度不是正确目标、我们找到了正确目标”。

真值参照整体SLA为{100*oracle.sla_violation_rate:.3f}%，Window为{100*window.sla_violation_rate:.3f}%，差为{100*(oracle.sla_violation_rate-window.sla_violation_rate):+.3f}个百分点。它看到当前任务各候选的潜在实际时长，包括抽样噪声，比可学习的条件均值更强，因此该差不是在线校准确定可回收的空间。这只说明该调度器的乐观信息敏感性；候选失败不能单独证明排队主导。

独立解读见[独立结果解读.md](./独立结果解读.md)。完整逐条件与配对结果见 analysis/test，所有原始任务计划、决策快照、完成轨迹及公共面板均保留。
'''
    (ROOT/'report/实验报告.md').write_text(text,encoding='utf-8')
    result=dict(decision=decision,exploratory=True,selected_legal=selected,combined_delta_pp=float(combined.difference_a_minus_b),
                combined_ci95_pp=[float(combined.ci_low),float(combined.ci_high)],combined_holm_p=float(combined.p_holm),
                objective_vs_uniform_delta_pp=float(objective.difference_a_minus_b),cost_positive_fraction_percent=float(cost_ratio),
                p99_delta_s=float(p99_delta),validation_runs=1392,test_runs=4896,all_runs=6288,
                overall=summary.to_dict('records'),primary=primary.to_dict('records'))
    (ROOT/'report/主要结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    summary.to_csv(ROOT/'report/整体汇总.csv',index=False,encoding='utf-8-sig')
    (ROOT/'README.md').write_text('# CPU可行性验证\n\n'+decision+'\n\n主要报告：[report/实验报告.md](report/实验报告.md)。\n\n'
        '数据：01_development_validation/raw 与 02_heldout_closed_loop/raw；统计：analysis/test；固定协议与哈希：00_design_sources；实现：scripts；核验：qa。\n\n'
        '复现：使用bundled Python，先 run_feasibility.py --stage validation --workers 3，再 refreeze_after_validation.py（仅首次、测试前），'
        '随后 run_feasibility.py --stage test --workers 3。已存在留出输出时不要运行refreeze；固定留出可直接重跑test保持原freeze，建议复制目录到新复现路径保留现有输出。\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('overall','primary')},ensure_ascii=False))
if __name__=='__main__':main()
