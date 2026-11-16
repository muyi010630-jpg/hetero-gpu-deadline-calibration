"""只读重分析已有留出记录；不调用模拟器、不更新冻结参数。"""
from pathlib import Path
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(r"$WORKSPACE")
OUT = Path(__file__).resolve().parent
PROB = ROOT / "修改补充实验/probability_calibration_cpu_20261010"
HET = ROOT / "修改补充实验/heterogeneity_adaptive_cpu_20261009"
BOOT_SEED = 2026101307
BOOT_REPS = 5000


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def record_plan_and_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [PROB / "03_heldout_closed_loop/raw/per_run_results.csv",
               PROB / "scripts/run_deadline.py", PROB / "scripts/deadline_calibrator.py",
               PROB / "03_heldout_closed_loop/config.json",
               HET / "scripts/calibrator_core.py", HET / "scripts/run_fixed_feedback.py",
               HET / "03_independent_prediction_test/run_config.json", Path(__file__)]
    records = []
    for path in sources:
        if path.exists():
            records.append(dict(path=str(path.relative_to(ROOT)), size=path.stat().st_size,
                                sha256=digest(path), category="configuration_or_aggregate"))
    npzs = sorted((HET / "03_independent_prediction_test/raw").glob("*.npz"))
    assert len(npzs) == 1000, len(npzs)
    for path in npzs:
        current_hash = digest(path)
        receipt = HET / "03_independent_prediction_test/per_stream" / (path.stem + ".json")
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        assert current_hash == metadata["npz_sha256"], str(path)
        records.append(dict(path=str(path.relative_to(ROOT)), size=path.stat().st_size,
                            sha256=current_hash, category="saved_prediction_diagnostics"))
    pd.DataFrame(records).to_csv(OUT / "source_sha256.csv", index=False, encoding="utf-8-sig")
    plan = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                nature="post hoc descriptive reanalysis of saved outputs; not preregistered before original holdout",
                prohibition="No simulation, cloud use, hyperparameter change, or new main significance test",
                probability=dict(arms=["uncalibrated", "uniform", "boundary"], horizons=[24, 120, 600],
                                 units="16 image blocks; each block averages two task seeds within rho x load, then six conditions",
                                 metrics=["panel_prob_brier", "sla_violation_rate"],
                                 comparisons=["uniform - uncalibrated", "boundary - uniform"],
                                 sla_scale="per_run fractions multiplied by 100, paired differences in percentage points",
                                 bootstrap_repetitions=BOOT_REPS, bootstrap_seed=BOOT_SEED,
                                 intervals="marginal 95% percentile intervals over image blocks; descriptive, no p values",
                                 caveat="Panel evaluation times differ by horizon: 24=[2,6,12,24],120 adds[60,120],600 adds[300,600]. Absolute horizon differences are not a controlled learning effect."),
                heterogeneity=dict(arm="adaptive_pool_local", params=dict(shrink_scale=0.5, freshness_half_life_s=1e9),
                                   conditions="rho=0,.25,.5,.75,1 x balanced/late_sparse; 20 images x 5 task seeds x 1800 prediction-before-update events each",
                                   local_weight="heterogeneity_local_weight = H/(H+0.5*v) only when n_local>0 and H>0, else 0",
                                   effective_weight="heterogeneity weight times saved local_freshness_factor",
                                   separations="n_local=0; n_local>0 and H=0; n_local>0 and H>0; insufficient observed group contexts also reported",
                                   means="stream event means, then five seeds per image, then twenty images equally; equal stream lengths make pooled mean identical",
                                   quantiles="pooled empirical quantiles of prediction-before-update events in each rho x flow; separate all-event and local-history-only denominators",
                                   inference="none; prediction events are not independent inferential units"),
                runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform()))
    write_json("analysis_plan.json", plan)
    print(f"Plan and source hashes recorded before numerical aggregation: {len(records)} sources", flush=True)
    return npzs


def boot_ci(values, salt):
    values = np.asarray(values, dtype=float)
    assert values.shape == (16,) and np.isfinite(values).all()
    rng = np.random.default_rng(BOOT_SEED + salt)
    draws = values[rng.integers(0, len(values), (BOOT_REPS, len(values)))].mean(axis=1)
    return list(map(float, np.quantile(draws, [.025, .975], method="linear")))


def probability_summary():
    df = pd.read_csv(PROB / "03_heldout_closed_loop/raw/per_run_results.csv")
    df = df[df["mode"].isin(["uncalibrated", "uniform", "boundary"])].copy()
    assert df["stage"].eq("test").all()
    assert len(df) == 1728
    assert not df.duplicated(["image_seed", "rho", "load", "task_seed", "horizon", "mode"]).any()
    # Identify actual run-level Brier as public analytic score, not selected labels.
    df["sla_violation_rate"] *= 100.0
    cells = df.groupby(["horizon", "image_seed", "rho", "load", "mode"])[["panel_prob_brier", "sla_violation_rate"]].mean()
    assert df.groupby(["horizon", "image_seed", "rho", "load", "mode"]).size().eq(2).all()
    assert cells.groupby(level=["horizon", "image_seed", "mode"]).size().eq(6).all()
    blocks = cells.groupby(level=["horizon", "image_seed", "mode"]).mean().reset_index()
    blocks.to_csv(OUT / "probability_horizon_image_blocks.csv", index=False, encoding="utf-8-sig")
    summaries, comparisons = [], []
    salt = 0
    for horizon in [24, 120, 600]:
        sub = blocks[blocks.horizon.eq(horizon)]
        for arm in ["uncalibrated", "uniform", "boundary"]:
            for metric in ["panel_prob_brier", "sla_violation_rate"]:
                values = sub[sub["mode"].eq(arm)].sort_values("image_seed")[metric].to_numpy()
                salt += 1
                lo, hi = boot_ci(values, salt)
                summaries.append(dict(horizon_s=horizon, arm=arm, metric=metric,
                                      mean=float(values.mean()), ci_low=lo, ci_high=hi, image_blocks=16))
        for a, b in [("uniform", "uncalibrated"), ("boundary", "uniform")]:
            for metric in ["panel_prob_brier", "sla_violation_rate"]:
                wide = sub.pivot(index="image_seed", columns="mode", values=metric).sort_index()
                delta = (wide[a] - wide[b]).to_numpy()
                salt += 1
                lo, hi = boot_ci(delta, salt)
                comparisons.append(dict(horizon_s=horizon, arm_a=a, arm_b=b, metric=metric,
                                        difference_a_minus_b=float(delta.mean()), ci_low=lo, ci_high=hi,
                                        negative_blocks=int((delta < 0).sum()), positive_blocks=int((delta > 0).sum()),
                                        zero_blocks=int((delta == 0).sum()), image_blocks=16,
                                        status="post_hoc_descriptive_no_new_significance_test"))
    pd.DataFrame(summaries).to_csv(OUT / "probability_horizon_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(OUT / "probability_horizon_paired.csv", index=False, encoding="utf-8-sig")
    # Cross-check the publication's pooled scores using the same block weighting.
    overall = blocks.groupby(["image_seed", "mode"])[["panel_prob_brier", "sla_violation_rate"]].mean().groupby("mode").mean()
    write_json("probability_horizon_results.json", dict(summaries=summaries, paired=comparisons,
                                                       overall_check=overall.reset_index().to_dict("records")))
    print("Probability horizon aggregation complete", flush=True)
    return summaries, comparisons


def heterogeneity_summary(npzs):
    stream_rows, bags = [], {}
    pattern = re.compile(r"i(\d+)_r([\d.]+)_t(\d+)_(balanced|late_sparse)_none")
    for j, path in enumerate(npzs):
        match = pattern.fullmatch(path.stem)
        assert match, path.name
        image, rho, task, flow = match.groups()
        image, rho, task = int(image), float(rho), int(task)
        with np.load(path, allow_pickle=False) as saved:
            modes = saved["mode_labels"].tolist()
            fields = saved["diagnostic_fields"].tolist()
            events = saved["diagnostics"][:, modes.index("adaptive_pool_local"), :]
            d = {name: events[:, fields.index(name)] for name in ["local_n_window", "heterogeneity", "heterogeneity_local_weight", "effective_local_weight", "unique_contexts", "local_freshness_factor", "local_sampling_variance"]}
        assert len(events) == 1800
        for name in d:
            assert np.isfinite(d[name]).all(), (path.name, name)
        no_history = d["local_n_window"] == 0
        history = ~no_history
        h_zero_history = history & (d["heterogeneity"] == 0)
        h_positive_history = history & (d["heterogeneity"] > 0)
        assert (no_history.astype(int) + h_zero_history + h_positive_history == 1).all()
        expected = np.zeros(len(events))
        expected[h_positive_history] = d["heterogeneity"][h_positive_history] / (d["heterogeneity"][h_positive_history] + .5 * d["local_sampling_variance"][h_positive_history])
        assert np.allclose(expected, d["heterogeneity_local_weight"], atol=1e-13, rtol=1e-12)
        assert np.allclose(d["effective_local_weight"], expected * d["local_freshness_factor"], atol=1e-13, rtol=1e-12)
        assert ((expected >= 0) & (expected <= 1)).all()
        row = dict(image_seed=image, rho=rho, task_seed=task, flow_kind=flow, n_events=1800,
                   no_history_fraction=float(no_history.mean()), h_zero_with_history_fraction=float(h_zero_history.mean()),
                   h_positive_with_history_fraction=float(h_positive_history.mean()),
                   heterogeneity_weight_mean_all=float(d["heterogeneity_local_weight"].mean()),
                   effective_weight_mean_all=float(d["effective_local_weight"].mean()),
                   effective_weight_zero_fraction_all=float((d["effective_local_weight"] == 0).mean()),
                   insufficient_group_contexts_with_history_fraction=float((history & (d["unique_contexts"] < 2)).mean()),
                   local_history_event_count=int(history.sum()), h_zero_history_event_count=int(h_zero_history.sum()))
        stream_rows.append(row)
        bags.setdefault((rho, flow), []).append({name: values for name, values in d.items()})
        if (j + 1) % 100 == 0:
            print(f"Saved diagnostic streams read: {j + 1}/1000", flush=True)
    stream_df = pd.DataFrame(stream_rows)
    assert stream_df.groupby(["rho", "flow_kind", "image_seed"]).size().eq(5).all()
    stream_df.to_csv(OUT / "heterogeneity_per_stream.csv", index=False, encoding="utf-8-sig")
    image_cols = [c for c in stream_df.columns if c.endswith("fraction") or c.endswith("fraction_all") or c.endswith("mean_all")]
    image_df = stream_df.groupby(["rho", "flow_kind", "image_seed"])[image_cols].mean().reset_index()
    image_df.to_csv(OUT / "heterogeneity_per_image.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (rho, flow), chunks in sorted(bags.items()):
        joined = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
        assert len(chunks) == 100
        history = joined["local_n_window"] > 0
        h_zero = history & (joined["heterogeneity"] == 0)
        n_history = int(history.sum())
        row = dict(rho=rho, flow_kind=flow, image_blocks=20, seeds_per_image=5,
                   prediction_events=180000, local_history_events=n_history,
                   no_history_fraction=float((~history).mean()),
                   h_zero_with_history_fraction_all=float(h_zero.mean()),
                   h_zero_fraction_given_history=float(h_zero.sum() / n_history),
                   h_positive_with_history_fraction_all=float((history & (joined["heterogeneity"] > 0)).mean()),
                   insufficient_group_contexts_with_history_fraction_all=float((history & (joined["unique_contexts"] < 2)).mean()))
        for name in ["heterogeneity_local_weight", "effective_local_weight"]:
            values = joined[name]
            row[name + "_mean_all"] = float(values.mean())
            row[name + "_mean_given_history"] = float(values[history].mean())
            row[name + "_zero_fraction_all"] = float((values == 0).mean())
            row[name + "_zero_fraction_given_history"] = float((values[history] == 0).mean())
            for mask_label, mask in [("all", np.ones(len(values), dtype=bool)), ("given_history", history)]:
                quantiles = np.quantile(values[mask], [.05, .25, .50, .75, .95], method="linear")
                for quantile_label, value in zip(["p05", "p25", "p50", "p75", "p95"], quantiles):
                    row[name + "_" + quantile_label + "_" + mask_label] = float(value)
        # Verify event-pooled mean equals the declared balanced block weighting.
        for name in ["heterogeneity_local_weight", "effective_local_weight"]:
            stream_col = ("heterogeneity_weight_mean_all" if name == "heterogeneity_local_weight" else "effective_weight_mean_all")
            block_mean = image_df[(image_df.rho == rho) & (image_df.flow_kind == flow)][stream_col].mean()
            assert abs(row[name + "_mean_all"] - block_mean) < 1e-12
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "heterogeneity_weight_by_rho_flow.csv", index=False, encoding="utf-8-sig")
    write_json("heterogeneity_weight_results.json", dict(scope="saved static fixed-feedback test only; not closed-loop or drift", rows=rows))
    return rows


def make_chinese_report(summaries, comparisons, h_rows):
    means = {(r["horizon_s"], r["arm"], r["metric"]): r["mean"] for r in summaries}
    pairs = {(r["horizon_s"], r["arm_a"], r["arm_b"], r["metric"]): r for r in comparisons}
    lines = ["# 已保存概率与异质性诊断的补充描述（2026-10-13）", "",
             "本分析只读取冻结留出结果；未运行新实验、未改估计器或选择参数。分析方案、输入 SHA256 和脚本已保存在本目录。属于原留出完成后的补充描述，不冒充原预设主分析。", "",
             "## 按任务到达时长重报概率与 SLA", "",
             "每时长以16幅画像为块：先平均同画像、ρ、负载的2个任务种子，再平均3个ρ×2个负载。区间为5000次画像块 percentile bootstrap 的边际95%区间；不追加p值或新主显著性判定。SLA差为前者减后者，单位百分点。", "",
             "| 到达时长（秒） | 未校准 Brier | 普通校准 Brier | 边界校准 Brier | 普通−未校准 SLA差 [95%区间] | 边界−普通 SLA差 [95%区间] |", "|---:|---:|---:|---:|---|---|"]
    for horizon in [24, 120, 600]:
        parts = []
        for a, b in [("uniform", "uncalibrated"), ("boundary", "uniform")]:
            r = pairs[(horizon, a, b, "sla_violation_rate")]
            parts.append(f"{r['difference_a_minus_b']:+.6f} [{r['ci_low']:+.6f},{r['ci_high']:+.6f}]")
        lines.append(f"| {horizon} | {means[(horizon,'uncalibrated','panel_prob_brier')]:.6f} | {means[(horizon,'uniform','panel_prob_brier')]:.6f} | {means[(horizon,'boundary','panel_prob_brier')]:.6f} | {parts[0]} | {parts[1]} |")
    lines += ["", "公共面板24秒条件的评价时刻为2、6、12、24秒；120秒另含60、120秒；600秒再含300、600秒。各时长内部方法使用同一面板，但跨时长的面板时刻集合不同，因此绝对 Brier 跨时长下降不能直接归因为在线学习。600秒条件不构成长期部署稳定性证明。", "", "## H 与局部混合权重", "",
              "仅使用共同反馈静态留出中的 adaptive_pool_local 冻结配置：λ=0.5，新鲜度半衰期10⁹秒。每条件包含20幅画像×5个种子×1800个预测前事件。均值先流内、再种子、再画像等权；由于每流事件数相同，它与条件内事件均值一致。分位数是条件内事件经验分位数，不把事件当独立样本做推断。", "",
              "H为零时局部权重为零；局部历史缺失也会令局部权重为零，但两者分开计数。下表“有历史H=0”以全部预测前事件为分母，“给定历史H=0”以n_local>0事件为分母。有效权重为 heterogeneity_local_weight×local_freshness_factor。", "",
              "| ρ | 反馈流 | 无局部历史% | 有历史H=0，占全部% | 给定历史H=0% | 有效局部权重均值（全部） | 有效局部权重中位数（有历史） | 有效局部权重P95（有历史） |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in h_rows:
        lines.append(f"| {r['rho']:.2f} | {r['flow_kind']} | {100*r['no_history_fraction']:.3f} | {100*r['h_zero_with_history_fraction_all']:.3f} | {100*r['h_zero_fraction_given_history']:.3f} | {r['effective_local_weight_mean_all']:.6f} | {r['effective_local_weight_p50_given_history']:.6f} | {r['effective_local_weight_p95_given_history']:.6f} |")
    r0 = [r for r in h_rows if r['rho'] == 0]
    r1 = [r for r in h_rows if r['rho'] == 1]
    lines += ["", "### 建议放正文的短段", "",
              f"在已保存共同反馈留出记录中，ρ=0时两种反馈流的有效局部权重均值为{r0[0]['effective_local_weight_mean_all']:.3f}和{r0[1]['effective_local_weight_mean_all']:.3f}，ρ=1时为{r1[0]['effective_local_weight_mean_all']:.3f}和{r1[1]['effective_local_weight_mean_all']:.3f}。在有局部历史的预测事件中，ρ=1下H归零比例分别为{100*r1[0]['h_zero_fraction_given_history']:.1f}%和{100*r1[1]['h_zero_fraction_given_history']:.1f}%。这些记录描述候选配置在所测反馈流中的混合行为，不建立H为无偏异质性估计量或独立机制收益；完整分位数和无历史比例见复现材料。", "",
              "均衡反馈与延迟稀疏反馈的顺序均为 balanced、late_sparse。原始H截零定义及冻结参数保持不变。", ""]
    (OUT / "Diagnostic_Results_and_Text_Notes.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    inputs = record_plan_and_inputs()
    probability_means, probability_pairs = probability_summary()
    heterogeneity_rows = heterogeneity_summary(inputs)
    make_chinese_report(probability_means, probability_pairs, heterogeneity_rows)
    write_json("completion.json", dict(status="complete", created_at_utc=datetime.now(timezone.utc).isoformat(),
                                       original_sources_modified=False, new_experiments=False,
                                       diagnostic_streams=1000, prediction_events=1800000,
                                       probability_image_blocks_per_horizon=16))
    print("Descriptive reanalysis complete", flush=True)
