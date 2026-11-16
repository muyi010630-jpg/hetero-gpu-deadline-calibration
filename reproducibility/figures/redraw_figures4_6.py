from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image
OUT=OUTPUT=Path(__file__).resolve().parent
RHOS=[0.,.25,.5,.75,1.]
GROUPS=[("A","full_vs_fixed"),("A","full_vs_ema"),("B","full_vs_fixed"),("B","full_vs_ema")]
BATCHES=["historical_5seed","current_runtime_5seed"]
COLORS={BATCHES[0]:"#0072B2",BATCHES[1]:"#D55E00"}
MARKERS={BATCHES[0]:"o",BATCHES[1]:"^"}

def draw4(comparisons, zh):
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = (["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"] if zh
                 else ["Arial", "Liberation Sans", "DejaVu Sans"])
    family = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": family, "font.size": 8.5,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.5, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "path", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 85 / 25.4))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.91, wspace=0.35)
    specs = [
        ("fixed_direct", ["balanced", "late_sparse"],
         ["均衡反馈", "延迟稀疏反馈"] if zh else ["Balanced feedback", "Delayed sparse feedback"],
         "(a)",
         "宏MAPE差（百分点）" if zh else "Macro MAPE difference (pp)", (-1.5, 14.0)),
        ("closed_direct", ["1.25", "1.7"],
         ["1.25 任务/(s·设备)", "1.70 任务/(s·设备)"] if zh else ["1.25 tasks/s/worker", "1.70 tasks/s/worker"],
         "(b)",
         "SLA违约率差（百分点）" if zh else "SLA violation difference (pp)", (-2.5, 3.2)),
    ]
    for ax, (kind, conditions, labels, title, ylabel, ylim) in zip(axes, specs):
        for condition, label, color, marker, ls in zip(conditions, labels, ["#0072B2", "#D55E00"], ["o", "s"], ["-", "--"]):
            rows = sorted([r for r in comparisons if r["analysis"] == kind and r["condition"] == condition],
                          key=lambda r: float(r["rho"]))
            x = np.array([float(r["rho"]) for r in rows])
            assert list(x) == RHOS
            y = np.array([float(r["mean_difference_pp"]) for r in rows])
            low = np.array([float(r["ci95_low_pp"]) for r in rows])
            high = np.array([float(r["ci95_high_pp"]) for r in rows])
            ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0, zorder=1)
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.4, label=label, zorder=3)
            for xx, yy, row in zip(x, y, rows):
                significant = float(row["p_holm_25"]) < 0.05
                ax.plot(xx, yy, color=color, marker=marker, linestyle="none", markersize=4,
                        markerfacecolor=color if significant else "white", markeredgewidth=1.0, zorder=4)
        ax.axhline(0, color="#505050", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        ax.set_title(title, loc="left", pad=10, fontweight="bold")
        ax.set_xlabel("共享比例 " + r"$\rho$" if zh else "Shared mismatch proportion " + r"$\rho$")
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(*ylim)
        ax.set_xticks(RHOS)
        ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
        if kind == "fixed_direct":
            ax.set_yticks([0, 4, 8, 12])
        else:
            ax.set_yticks([-2, -1, 0, 1, 2, 3])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3.2, width=0.8)
        ax.grid(axis="y", color="#DEDEDE", linestyle="-", linewidth=0.45, alpha=0.6, zorder=0)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.015, 0.19 if kind == "fixed_direct" else 0.025), ncol=1,
                  handlelength=2.1, handletextpad=0.6, borderaxespad=0, labelspacing=0.25)
    suffix = "中文" if zh else "英文"
    stem = OUT / ("figure4_sharing_zh" if zh else "figure4_sharing_en")
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=600, facecolor="white")
    plt.close(fig)
    return [str(stem.with_suffix(f".{fmt}")) for fmt in ["png", "pdf", "svg"]]

def draw6(table, points, language):
    zh = language == "中文"
    plt.rcParams.update({"font.family": "Microsoft YaHei" if zh else "Arial", "font.size": 8.5,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": 0.7, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), sharey=True)
    fig.subplots_adjust(left=0.23, right=0.975, bottom=0.12, top=0.86, hspace=0.52, wspace=0.28)
    plotted = ["mape_pct", "qos_bound_coverage_pct", "gpu_sla_violation_rate", "p99_flow_time_s"]
    titles = ["MAPE", "覆盖率" if zh else "Bound coverage", "GPU预算违约率" if zh else "GPU budget violations", "P99 完成时间" if zh else "P99 completion time"]
    limits = [(-15, 3), (-2, 43), (-9, 5), (-12, 5)]
    ticks = [[-15, -10, -5, 0], [0, 10, 20, 30, 40], [-8, -4, 0, 4], [-10, -5, 0, 5]]
    labels = [f"{load} / Full-Deploy\n−{'Fixed' if comparison == 'full_vs_fixed' else 'EMA'}" for load, comparison in GROUPS]
    for i, (ax, metric) in enumerate(zip(axes.flat, plotted)):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E4E7EA", linewidth=0.45)
        ax.axvline(0, color="#666666", linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_title(f"({chr(97 + i)}) {titles[i]}", fontsize=10, loc="left", pad=9)
        for row, (load, comparison) in enumerate(GROUPS):
            y = 3 - row
            ax.axhline(y, color="#EEEEEE", linewidth=0.45, zorder=0)
            for batch in BATCHES:
                subset = points[(points.load == load) & (points.comparison == comparison) & (points.metric == metric) & (points.batch == batch)].sort_values("seed")
                assert len(subset) == 5
                offsets = np.linspace(-0.23, -0.07, 5) if batch == BATCHES[0] else np.linspace(0.07, 0.23, 5)
                assert subset.delta.between(*limits[i]).all()
                ax.scatter(subset.delta, y + offsets, color=COLORS[batch], marker=MARKERS[batch], s=22,
                           edgecolors="white", linewidths=0.4, zorder=4)
            stat = table[(table.load == load) & (table.comparison == comparison) & (table.metric == metric)].iloc[0]
            ax.hlines(y, stat.ci_low, stat.ci_high, color="#202020", linewidth=1.1, zorder=3)
            ax.vlines([stat.ci_low, stat.ci_high], y - 0.035, y + 0.035, color="#202020", linewidth=0.8, zorder=3)
            ax.scatter([stat["mean"]], [y], marker="D", s=18, color="#202020", edgecolors="white", linewidths=0.35, zorder=5)
        ax.set_xlim(limits[i])
        ax.set_xticks(ticks[i])
        ax.set_ylim(-0.42, 3.42)
        ax.set_yticks([3, 2, 1, 0], labels)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", length=3, width=0.7)
        ax.set_xlabel("配对差（s）" if zh and metric == "p99_flow_time_s" else
                      "配对差（百分点）" if zh else "Paired difference (s)" if metric == "p99_flow_time_s" else "Paired difference (pp)", labelpad=6)
    legend_labels = ["部署批次1", "部署批次2", "均值及95% CI"] if zh else ["Deployment batch 1", "Deployment batch 2", "Mean and 95% CI"]
    handles = [Line2D([], [], color=COLORS[BATCHES[0]], marker="o", linestyle="none", markersize=5),
               Line2D([], [], color=COLORS[BATCHES[1]], marker="^", linestyle="none", markersize=5),
               Line2D([], [], color="#202020", marker="D", linewidth=1.0, markersize=4)]
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.55, 0.98), frameon=False,
               ncol=3, fontsize=8.5, handlelength=1.6, columnspacing=1.5, handletextpad=0.5)
    filename = OUTPUT / ("figure6_deployment_zh" if zh else "figure6_deployment_en")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(filename.with_suffix("." + suffix), dpi=600)
    plt.close(fig)
    with Image.open(filename.with_suffix(".png")) as im:
        assert im.size == (4260, 3240) and im.info["dpi"][0] >= 599
        return {"path": str(filename.with_suffix(".png")), "size": im.size, "dpi": im.info["dpi"]}

if __name__=='__main__':
    with (OUT/'figure4_plot_data.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    draw4(rows,True);draw4(rows,False)
    table=pd.read_csv(OUT/'figure6_statistics.csv');points=pd.read_csv(OUT/'figure6_points.csv')
    draw6(table,points,'中文');draw6(table,points,'英文')
