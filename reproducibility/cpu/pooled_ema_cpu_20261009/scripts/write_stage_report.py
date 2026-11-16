from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
STAGES={'replay':'02_scheduling_replay','structure':'03_mismatch_structure'}
LABELS={'full':'Full','pool_ema':'PoolEMA','local_matched':'LocalMatched','gated_pred':'GatedPoint'}


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=STAGES,required=True);args=ap.parse_args()
    output=ROOT/STAGES[args.stage]/'full';analysis=output/'analysis'
    completion=json.loads((output/'completion.json').read_text(encoding='utf-8'))
    if not completion['complete']:raise RuntimeError('Formal stage incomplete')
    means=read_csv(analysis/'summary_by_variant_mode.csv');tests=read_csv(analysis/'paired_seed_inference.csv')
    variants=sorted({r['variant'] for r in means})
    lines=['# '+('调度回放' if args.stage=='replay' else '失配结构')+'：组级池化EMA补充对照', '',
           '本阶段已正式完成；未改变原论文、母模拟器或原画像。所有对照只替换点预测残差，共用冻结的Full风险封装与调度器。', '',
           f"完成{completion['units']}个场景—种子单元、{completion['runs']}次运行。每个单元四种策略的外生计划与所有候选潜在执行时间SHA256完全一致；Full逐运行与原黄金结果的最大绝对差为{completion['full_golden_max_abs_delta']:.1e}。", '',
           '正式种子为601–640。调度回放135个场景；失配结构每个变体15个场景。每个种子内场景等权汇总，以40个种子作为推断单位。', '',
           '| 变体 | 点预测方法 | MAPE(%) | SLA违约率(%) | 严格QoS违约率(%) | 严格QoS上界覆盖率(%) | P99(s) |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in means:
        lines.append(f"| {row['variant']} | {LABELS[row['mode']]} | {float(row['prediction_mape'])*100:.4f} | {float(row['sla_violation_rate'])*100:.4f} | {float(row['strict_qos_violation_rate'])*100:.4f} | {float(row['qos_bound_coverage'])*100:.4f} | {float(row['p99_completion']):.6f} |")
    lines+=['','差值统一为“对照减Full”；负MAPE/SLA差表示对照优于Full。95%区间为5000次种子级配对bootstrap。p值采用100000次双侧配对符号翻转，Holm在每个变体内对三个对照×MAPE/SLA六项主要检验校正。','','| 变体 | 对照 | 指标 | 差(pp) | 95%CI(pp) | 原始p | Holm p |','| --- | --- | --- | ---: | --- | ---: | ---: |']
    for row in tests:
        if row['primary_test']!='True':continue
        lines.append(f"| {row['variant']} | {LABELS[row['mode']]} | {row['metric']} | {float(row['mean_delta']):+.4f} | [{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}] | {float(row['p_paired_signflip']):.6g} | {float(row['p_holm_primary6']):.6g} |")
    if args.stage=='structure' and (analysis/'holm_all24_sensitivity.csv').exists():
        lines+=['', '为跨四个结构的综合解读，另附`holm_all24_sensitivity.csv`，将四变体×三个对照×MAPE/SLA共24项一并作Holm校正。它是附加保守敏感性检验，保留上表既定的每变体六项主分析；不替换原区间或p值，也未重新运行模拟。']
    lines+=['','## 结果解释','']
    for variant in variants:
        for mode in ('pool_ema','gated_pred'):
            pair={r['metric']:r for r in tests if r['variant']==variant and r['mode']==mode and r['primary_test']=='True'}
            m=pair['prediction_mape'];s=pair['sla_violation_rate']
            lines.append(f"{variant}中，{LABELS[mode]}相对Full的MAPE差为{float(m['mean_delta']):+.4f}个百分点，SLA差为{float(s['mean_delta']):+.4f}个百分点。两项Holm校正p分别为{float(m['p_holm_primary6']):.6g}和{float(s['p_holm_primary6']):.6g}。")
            lines.append('')
    lines+=['## 归因范围与限制','',
            '- PoolEMA为设备型号×任务族的单状态直接EMA，按每条组内已完成反馈递推，不平均上下文EMA。按族步长与Full一致（0.35/0.45/0.22）；增量±0.35、状态±0.70。',
            '- 更高的组反馈更新频率、跨精度组共享与Full的精度优先邻居及跨型号回退存在结构差异；结果不能单独归因于融合权重这一项公式。',
            '- LocalMatched仅使用目标局部残差。GatedPoint在目标无已完成反馈时使用Full，随后仅停止点残差融合；Full风险尺度、自身权重及相应冷启动裕量始终保留，区别于原风险也门控的Gated。',
            '- 风险与调度规则相同，实际设备选择、完成顺序、反馈历史和风险尺度数值可随点预测变化。MAPE只统计实际选中执行，不能视作同一固定全候选测试集的精度。覆盖率按q≥2统计；假安全率以预测为安全的任务作分母。',
            '- 为共用Full风险封装，各模式保留局部残差状态；本实验不是最简组级部署的开销比较。',
            '- P0–P3沿用各一组固定画像；种子只随机化任务流，不代表40个独立迁移环境。P3初始失配幅度较小，不应作为同等幅度的直接优劣证据。',
            '- P2设计上共享偏差与独立噪声各占一半方差；原一次生成的P1/P2有限样本实际均值和方差未严格重标定，不主张严格等幅且唯一改变相关性的因果对照。',
            '- 分层是否优于直接池化应以正式表中估计和区间判断；不得将不利于分层的变体删去。未检出差异不等于证明等效。',
            '- 本地CPU估计器与回放对照不替代实体三主机在线闭环的PoolEMA对照。','',
            '## 文件','',
            '- `raw/per_run_results.csv`：全部运行汇总；`raw/shards`：可恢复原子分片。',
            '- `raw/representative_plans`与`raw/representative_traces`：代表任务计划及各策略执行轨迹，轨迹另列`point_log_residual_at_dispatch`为实际点修正。',
            '- `analysis/paired_cell_deltas_vs_full.csv`、`paired_seed_deltas_vs_full.csv`：逐场景及种子配对差。',
            '- `analysis/paired_seed_inference.csv`：全部指标区间与检验；次要指标未经多重校正。',
            '- `config.json`、`run_log.jsonl`、`completion.json`：配置、运行日志与完整性。',
            '- `00_configuration_sources/replay_source_manifest.json`：原始来源、冻结复制和SHA256。','']
    (analysis/'experiment_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(str(analysis/'experiment_report.md'))


if __name__=='__main__':main()
