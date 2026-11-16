"""Independent arithmetic/aggregation checks and manuscript-ready report."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from docx import Document

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "analysis_plan.json").read_text(encoding="utf-8"))
RESULTS = json.loads((HERE / "rq2_supplementary_comparisons_25.json").read_text(encoding="utf-8"))
ROOT = Path(r"$WORKSPACE")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def quantile_linear_manual(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


checks = []
source_rows = {key: read_csv(info["path"]) for key, info in PLAN["source_hashes"].items()}
differences = read_csv(HERE / "rq2_paired_image_differences.csv")
source_dir = Path(PLAN["source_hashes"]["fixed"]["path"]).parent
condition_summaries = {
    "fixed_direct": read_csv(source_dir / "condition_summary.csv"),
    "closed_direct": read_csv(Path(PLAN["source_hashes"]["closed"]["path"]).parent / "summary.csv"),
}
current_doc = Document(ROOT / "论文初稿.docx")
fixed_word_tables = [t for t in current_doc.tables if any(c.text == "19.930" for r in t.rows for c in r.cells)]
closed_word_tables = [t for t in current_doc.tables if any(c.text == "19.450" for r in t.rows for c in r.cells)]
assert len(fixed_word_tables) == len(closed_word_tables) == 1
word_data = {
    "fixed_direct": [[c.text for c in r.cells] for r in fixed_word_tables[0].rows][1:],
    "closed_direct": [[c.text for c in r.cells] for r in closed_word_tables[0].rows][1:],
}

for result in RESULTS:
    d = [float(row["difference_pp"]) for row in differences if row["comparison_id"] == result["id"]]
    assert len(d) == 20
    independent_mean = math.fsum(d) / len(d)
    assert abs(independent_mean - result["mean_difference_pp"]) < 1e-12
    rng = np.random.default_rng(result["bootstrap_seed"])
    sampled = rng.integers(0, 20, size=(5000, 20))
    bootstrap = [math.fsum(d[i] for i in index) / 20 for index in sampled]
    for fraction, field in [(0.025, "ci95_low_pp"), (0.975, "ci95_high_pp")]:
        assert abs(quantile_linear_manual(bootstrap, fraction) - result[field]) < 1e-12
    rng_sign = np.random.default_rng(result["signflip_seed"])
    signs = 2 * rng_sign.integers(0, 2, size=(100000, 20), dtype=np.int8) - 1
    long_means = np.sum(signs * np.array(d, dtype=np.longdouble), axis=1) / 20
    extreme = int(np.count_nonzero(np.abs(long_means) >= abs(independent_mean) - 1e-12))
    assert extreme == result["signflip_extreme_count"]
    if result["analysis"] != "fixed_feedback_interaction":
        fixed = result["analysis"] == "fixed_direct"
        condition_field = "flow_kind" if fixed else "load"
        metric = "overall_macro_mape" if fixed else "sla_violation_rate"
        for mode, mean_field, word_column in [("pool_ema", "pool_mean", 3), ("local_window", "window_mean", 4)]:
            selected = [row for row in condition_summaries[result["analysis"]] if float(row["rho"]) == result["rho"] and row["mode"] == mode and (row[condition_field] == result["condition"] if fixed else float(row[condition_field]) == result["condition"]) and (not fixed or row["drift_type"] == "none")]
            assert len(selected) == 1
            summary_value = float(selected[0][metric]) * (1 if fixed else 100)
            assert abs(summary_value - result[mean_field]) < 1e-12
            row_index = int(round(result["rho"] * 4)) * 2 + (0 if result["condition"] in ["balanced", 1.25] else 1)
            assert f"{result[mean_field]:.3f}" == word_data[result["analysis"]][row_index][word_column]
    checks.append({"id": result["id"], "mean_fsum": "pass", "bootstrap_manual_quantile": "pass", "signflip_longdouble": "pass", "source_summary_and_current_word_table": "pass" if result["analysis"] != "fixed_feedback_interaction" else "not_applicable"})

ordered = sorted(RESULTS, key=lambda row: row["p_raw"])
prior_adjusted = []
for rank, row in enumerate(ordered):
    prior_adjusted.append((25 - rank) * row["p_raw"])
    assert abs(min(1, max(prior_adjusted)) - row["p_holm_25"]) < 1e-15

report = {
    "status": "pass",
    "n_comparisons": 25,
    "checks": checks,
    "holm_independent_sorted_cumulative_max": "pass",
    "mean_unit_fixed": "percentage-point source values unchanged",
    "mean_unit_closed": "proportion multiplied by 100",
    "source_hashes_unchanged": {key: hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"] for key, info in PLAN["source_hashes"].items()},
    "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
assert all(report["source_hashes_unchanged"].values())
(HERE / "independent_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def signed(value):
    return f"{value:+.3f}".replace("-", "−")


def table(kind):
    direct = kind != "fixed_feedback_interaction"
    rows = [row for row in RESULTS if row["analysis"] == kind]
    if direct:
        lines = ["| ρ | 反馈条件 | PoolEMA | LocalWindow24 | Pool−Window差［边际95%区间］ | 补充分析Holm p |", "| ---: | --- | ---: | ---: | --- | ---: |"]
    else:
        lines = ["| ρ | 延迟稀疏差−均衡差［边际95%区间］ | Holm p | 正/负/零画像数 |", "| ---: | --- | ---: | --- |"]
    for row in rows:
        interval = f"{signed(row['mean_difference_pp'])}［{signed(row['ci95_low_pp'])},{signed(row['ci95_high_pp'])}］"
        if direct:
            condition = {"balanced": "均衡", "late_sparse": "延迟稀疏"}.get(row["condition"], f"负载{row['condition']}")
            lines.append(f"| {row['rho']:g} | {condition} | {row['pool_mean']:.3f} | {row['window_mean']:.3f} | {interval} | {row['p_holm_25']:.6f} |")
        else:
            count = f"{row['n_positive']}/{row['n_negative']}/{row['n_zero']}"
            lines.append(f"| {row['rho']:g} | {interval} | {row['p_holm_25']:.6f} | {count} |")
    return "\n".join(lines)


body = """# RQ2已有数据补充分析：入文结果与复现记录

本分析在2026-10-13根据模拟复审意见新增，属于事后补充分析，不属于原留出前冻结的主检验。分析只读取已有CSV，不运行模拟、不修改冻结校准器和原始数据。

## 可直接入正文的方法说明

另以20幅画像为块，对PoolEMA减LocalWindow24的10项固定反馈宏MAPE、10项闭环SLA及5项固定反馈差中差作补充分析；画像内5任务种子已平均，差中差为同一画像的“延迟稀疏差减均衡差”。25项统一Holm校正；边际95%区间以5000次画像块bootstrap计算，双侧配对符号翻转采用100000次Monte Carlo及加一修正。该分析在测试后提出，不追溯纳入原冻结检验。

## 可直接入正文的5.3结果

固定反馈下，PoolEMA在ρ≤0.75的8条件中宏MAPE均高于LocalWindow24；ρ=1时均衡流差为+0.124个百分点［0.109,0.141］，延迟稀疏流差为−0.576个百分点［−0.622,−0.528］，两者分别在20/20画像同向，校正p均为0.000250。ρ=1的反馈差中差为−0.700个百分点［−0.743,−0.657］，校正p=0.000250；其余4项差中差也均为负且通过25项校正。反馈条件能够改变所测完整配置的预测排序，但该证据不能分离共享粒度、反馈频率与滤波形式的独立作用。闭环中仅ρ=0、高负载条件检出PoolEMA违约更多，差+2.078个百分点［1.381,2.749］，校正p=0.000500；其余9条件未通过校正，包括ρ=1的两档负载。预测排序变化没有同时建立相应QoS优势，不提供仅由ρ决定的通用切换阈值。

## 可直接用于摘要/引言/讨论的短句

总体失配强度匹配的补充配对分析表明，PoolEMA与LocalWindow24的预测排序受反馈条件影响：ρ=1时，均衡反馈有利于窗口，延迟稀疏反馈有利于池化；闭环未建立对应的池化QoS优势。比较针对冻结完整配置，不单独识别共享机制效应。

## 表9（a）固定反馈：上下文宏MAPE（%）

""" + table("fixed_direct") + """

## 表9（b）闭环：SLA违约率（%）

""" + table("closed_direct") + """

表注：均值对20幅画像等权，每幅画像先平均5任务种子；差值单位均为百分点。固定反馈10项、闭环10项及5项固定反馈差中差共同构成25项补充分析Holm族。正差表示PoolEMA误差或违约更多。区间为画像块边际95%区间，不以是否跨零代替校正检验；未拒绝差异不表示等效。全条件方向计数及差中差另见复现材料，原自适应候选的30项冻结主检验独立保留。

## 固定反馈的全部差中差（复现材料）

""" + table("fixed_feedback_interaction") + """

## 推断口径与边界

配对符号翻转将画像作为独立推断块，依赖零假设下差值对符号翻转的对称性/可交换性，不是随机化机制的因果检验。bootstrap保持画像内配对及差中差结构。未对25个区间作同时覆盖校正；Holm用于25项检验。Monte Carlo p的最小报告分辨率是1/100001，不是精确全枚举p；所有随机流种子均列入分析方案。

## 已完成验证

- 全25项均值经独立math.fsum复核；bootstrap区间经手工线性分位数复核；符号翻转计数用longdouble求和复核；Holm用另一路排序累计最大值复核。
- 两面板的全部Pool和Window均值与原condition_summary/summary一致，并与当前Word表7的三位小数逐项一致。
- 20幅画像匹配、每画像5任务种子、无重复或缺失行和源文件SHA256均已检查；原始源表未改动。

## 文件与复现命令

计划：analysis_plan.json；完整统计：rq2_supplementary_comparisons_25.csv/json；500个画像差值：rq2_paired_image_differences.csv；独立核验：independent_validation.json。

```powershell
& '<USER_HOME>/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' '.\\reanalyse_rq2.py' --analyse
& '<USER_HOME>/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' '.\\validate_and_report_rq2.py'
```

首次方案使用--prepare-plan保存；已有方案不覆盖。脚本读取保存计划时校验源与实现哈希。若迁移环境，请保持原冻结脚本，另用路径包装器映射源表及输出目录并记录映射；不要改写原方案日期或将本分析称为预注册。
"""
(HERE / "RQ2_Supplementary_Results.md").write_text(body, encoding="utf-8")
print(json.dumps({"status": report["status"], "n_comparisons": len(checks), "all_original_table_means_match": True}, ensure_ascii=False))
