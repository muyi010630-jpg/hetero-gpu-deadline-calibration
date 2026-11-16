"""Figure 3 from archived, predict-before-update shared-feedback records only.

No simulator, no GPU access, no parameter selection, and no predictor replay.
The only random operation is conditional stream bootstrap for uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parents[1]
PROJECT = IMAGES.parents[1]
SOURCE = PROJECT / "修改补充实验/pooled_ema_cpu_20261009/01_independent_calibration"
MODES = ("fixed", "family_ema", "hierarchical_ema", "pool_ema")
KS = tuple(range(1, 11))
SEEDS = tuple(range(101, 131))
BOOT_REPS = 5000
BOOT_SEED = 20261009
CAPTIONS = {
    "zh": "图3 共同反馈下目标单元的早期预测收敛。横轴为每个设备型号—任务上下文单元的前k次预测（k=1至10），纵轴为这些预测的累计MAPE。Fixed、LocalEMA（匹配步长）、分层EMA和PoolEMA对同一事件先预测，再使用相同反馈更新；三种在线校准方法采用匹配的任务族步长。每条任务流包含45个目标单元，曲线为30条任务流的等权均值，阴影为按任务流重采样5000次得到的逐点边际95%置信区间。首次目标预测时其他单元可能已提供共享历史。目标单元的前k次预测与表8定义的全流前20%冷阶段属于不同评价子集；区间条件于同一固定画像失配环境，不表示跨环境不确定性，预测误差改善也不能单独证明调度或SLA收益。",
    "en": "Figure 3. Early prediction convergence under shared feedback. The x-axis denotes the first k predictions for each device-model–task-context cell (k = 1–10), and the y-axis shows their cumulative MAPE. Fixed, LocalEMA (matched step sizes), Hierarchical EMA, and PoolEMA predict before receiving the same feedback; the three online calibration methods use matched task-family step sizes. Each stream contains 45 target cells. Lines show equally weighted means over 30 streams, and shading denotes pointwise marginal 95% confidence intervals from 5,000 stream-level bootstrap resamples. Other cells may already have supplied shared history before a target cell's first prediction. The first k target-cell predictions and the first 20% of the stream used to define the cold phase in Table 8 are different evaluation subsets. Intervals are conditional on one fixed profile-mismatch environment and do not quantify cross-environment uncertainty; prediction-error reductions alone do not establish scheduling or SLA gains."
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def extract():
    per_seed = []
    overall = {m: [] for m in MODES}
    cold = {m: [] for m in MODES}
    checked_rows = 0
    max_ape_delta_pp = 0.0
    manifest = []
    for seed in SEEDS:
        path = SOURCE / f"raw/per_seed/seed_{seed}_trace.csv"
        manifest.append({"role": "archived_prediction_trace", "path": str(path),
                         "bytes": path.stat().st_size, "sha256": sha256(path)})
        rows = read_csv(path)
        assert len(rows) == 1800 * len(MODES)
        counts = {}
        buckets = {(m, k): [] for m in MODES for k in KS}
        for event_id in range(1800):
            event = rows[4 * event_id:4 * event_id + 4]
            assert {r["strategy"] for r in event} == set(MODES)
            first = event[0]
            cell = (first["device"], first["context_key"])
            before = counts.get(cell, 0)
            shared = ("seed", "event_id", "device", "context_key", "observed_s",
                      "local_updates_seen", "local_observation_index", "group_updates_seen")
            for r in event:
                assert int(r["seed"]) == seed and int(r["event_id"]) == event_id
                assert all(r[f] == first[f] for f in shared)
                assert int(r["local_updates_seen"]) == before
                assert int(r["local_observation_index"]) == before + 1
                observed, predicted = float(r["observed_s"]), float(r["predicted_s"])
                assert observed > 0 and predicted > 0
                calculated = abs(predicted - observed) / observed * 100
                archived = float(r["ape"]) * 100
                delta = abs(calculated - archived)
                max_ape_delta_pp = max(max_ape_delta_pp, delta)
                assert delta < 1e-9
                mode = r["strategy"]
                overall[mode].append(archived)
                if event_id < 360:
                    assert r["phase_band"] == "cold_start"
                    cold[mode].append(archived)
                for k in KS:
                    if before < k:
                        buckets[(mode, k)].append(archived)
                checked_rows += 1
            counts[cell] = before + 1
        assert len(counts) == 45 and min(counts.values()) >= 10
        for mode in MODES:
            for k in KS:
                values = buckets[(mode, k)]
                assert len(values) == 45 * k
                per_seed.append({"seed": seed, "strategy": mode, "k": k,
                                 "target_cells": 45, "predictions": len(values),
                                 "mape_pct": statistics.fmean(values)})

    # Reuse a single paired stream-index bootstrap draw table for all curves/k.
    rng = random.Random(BOOT_SEED)
    draws = [[rng.randrange(30) for _ in range(30)] for _ in range(BOOT_REPS)]
    lookup = {(r["seed"], r["strategy"], r["k"]): r for r in per_seed}
    curve = []
    for mode in MODES:
        for k in KS:
            values = [lookup[(seed, mode, k)]["mape_pct"] for seed in SEEDS]
            samples = [statistics.fmean(values[i] for i in draw) for draw in draws]
            curve.append({"strategy": mode, "k": k, "n_streams": 30,
                          "target_cells_per_stream": 45, "predictions_total": 30 * 45 * k,
                          "mape_mean_pct": statistics.fmean(values),
                          "ci95_low_pct": percentile(samples, .025),
                          "ci95_high_pct": percentile(samples, .975)})

    checks = []
    curve_index = {(r["strategy"], r["k"]): r for r in curve}
    for r in read_csv(SOURCE / "analysis/summary_first_k.csv"):
        if r["scope"].endswith("_all_stream"):
            k = int(r["scope"].split("_")[1])
            recomputed = curve_index[(r["strategy"], k)]["mape_mean_pct"]
            delta = recomputed - float(r["mape_pct"])
            assert abs(delta) < 1e-10
            checks.append({"strategy": r["strategy"], "scope": r["scope"],
                           "archived_mape_pct": float(r["mape_pct"]),
                           "recomputed_mape_pct": recomputed, "difference_pp": delta})
    for name, scope, accumulator in (("summary_overall.csv", "overall", overall),
                                     ("summary_by_phase.csv", "cold_start", cold)):
        for r in read_csv(SOURCE / "analysis" / name):
            if r["scope"] == scope:
                recomputed = statistics.fmean(accumulator[r["strategy"]])
                delta = recomputed - float(r["mape_pct"])
                assert abs(delta) < 1e-10
                checks.append({"strategy": r["strategy"], "scope": scope,
                               "archived_mape_pct": float(r["mape_pct"]),
                               "recomputed_mape_pct": recomputed, "difference_pp": delta})
    for name in ("scripts/run_pool_ema_calibration.py", "analysis/summary_first_k.csv",
                 "analysis/summary_overall.csv", "analysis/summary_by_phase.csv",
                 "analysis/experiment_report.md", "analysis/golden_qa_full.json"):
        path = SOURCE / name
        manifest.append({"role": "source_protocol_or_validation", "path": str(path),
                         "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(HERE / "first_k_per_seed.csv", per_seed)
    write_csv(HERE / "first_k_curve.csv", curve)
    write_csv(HERE / "archived_summary_crosscheck.csv", checks)
    write_csv(HERE / "source_manifest_sha256.csv", manifest)
    (HERE / "validation.json").write_text(json.dumps({
        "passed": True, "source_prediction_rows_checked": checked_rows,
        "max_recomputed_ape_delta_pp": max_ape_delta_pp,
        "summary_crosschecks": len(checks), "summary_max_delta_pp": max(abs(r["difference_pp"]) for r in checks),
        "new_experiments": False, "predictor_reconstructed": False,
        "n_streams": 30, "events_per_stream": 1800, "target_cells_per_stream": 45,
        "k_range": [1, 10], "scope": "target first-k cumulative predictions over full archived stream",
        "bootstrap_reps": BOOT_REPS, "bootstrap_seed": BOOT_SEED,
        "confidence_interval": "pointwise percentile 95%, resampling task streams; conditional on one fixed mismatch environment"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return curve


def render(curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in (Path("<SYSTEM_FONTS>/arial.ttf"), Path("<SYSTEM_FONTS>/msyh.ttc")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Microsoft YaHei"],
                         "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                         "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "path", "axes.linewidth": .8,
                         "savefig.facecolor": "white"})
    styles = {
        "fixed": ("#606060", "--", "s"),
        "family_ema": ("#0072B2", "-.", "o"),
        "hierarchical_ema": ("#D55E00", ":", "^"),
        "pool_ema": ("#009E73", "-", "D")}
    labels = {
        "中文": {"fixed": "Fixed", "family_ema": "LocalEMA（匹配步长）", "hierarchical_ema": "分层EMA", "pool_ema": "PoolEMA"},
        "英文": {"fixed": "Fixed", "family_ema": "LocalEMA (matched step sizes)", "hierarchical_ema": "Hierarchical EMA", "pool_ema": "PoolEMA"}}
    for language in ("中文", "英文"):
        plt.rcParams["font.family"] = "Microsoft YaHei" if language == "中文" else "Arial"
        fig, ax = plt.subplots(figsize=(180 / 25.4, 100 / 25.4))
        fig.subplots_adjust(left=.11, right=.975, bottom=.16, top=.815)
        for mode in MODES:
            rows = [r for r in curve if r["strategy"] == mode]
            x = [r["k"] for r in rows]
            y = [r["mape_mean_pct"] for r in rows]
            low = [r["ci95_low_pct"] for r in rows]
            high = [r["ci95_high_pct"] for r in rows]
            color, linestyle, marker = styles[mode]
            ax.fill_between(x, low, high, color=color, alpha=.13, linewidth=0)
            ax.plot(x, y, label=labels[language][mode], color=color, linestyle=linestyle,
                    marker=marker, markevery=[0, 2, 4, 6, 9], linewidth=1.65,
                    markersize=4.0, markerfacecolor="white", markeredgewidth=1.0)
        ax.set_xlim(.8, 10.2)
        ax.set_ylim(0, 22)
        ax.set_xticks(KS)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_ylabel("MAPE（%）" if language == "中文" else "MAPE (%)")
        ax.set_xlabel("目标单元前 k 次预测（累计）" if language == "中文" else "First k predictions per target cell (cumulative)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3, width=.75)
        ax.grid(axis="y", color="#E5E5E5", linewidth=.55)
        ax.set_axisbelow(True)
        handles, legend_labels = ax.get_legend_handles_labels()
        legend_order = [0, 2, 1, 3]  # two rows: Fixed/LocalEMA, then Hierarchical/PoolEMA
        ax.legend([handles[i] for i in legend_order], [legend_labels[i] for i in legend_order],
                  loc="lower center", bbox_to_anchor=(.5, 1.025), ncol=2, frameon=False,
                  handlelength=2.5, columnspacing=2.3, labelspacing=.75, borderaxespad=0)
        basename = IMAGES / f"图3_冷启动收敛_{language}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(basename.with_suffix("." + ext), dpi=600)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, help="Optional isolated matplotlib dependency directory")
    parser.add_argument("--recompute", action="store_true", help="Recompute archived-data summaries; otherwise preserve CSVs and CI exactly")
    args = parser.parse_args()
    if args.packages:
        sys.path.insert(0, str(args.packages.resolve()))
    if args.recompute or not (HERE / "first_k_curve.csv").exists():
        curve = extract()
    else:
        curve = read_csv(HERE / "first_k_curve.csv")
        for row in curve:
            row["k"] = int(row["k"])
            for field in ("mape_mean_pct", "ci95_low_pct", "ci95_high_pct"):
                row[field] = float(row[field])
    (HERE / "captions.json").write_text(json.dumps(CAPTIONS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render(curve)
    outputs = [IMAGES / f"图3_冷启动收敛_{language}.{extension}"
               for language in ("中文", "英文") for extension in ("png", "pdf", "svg")]
    outputs.extend(p for p in HERE.iterdir() if p.is_file() and p.name != "output_manifest_sha256.csv")
    write_csv(HERE / "output_manifest_sha256.csv", [{"path": str(p), "bytes": p.stat().st_size,
                                                    "sha256": sha256(p)} for p in outputs])
    print(json.dumps({"figure_outputs": [str(p) for p in outputs[:6]], "verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
