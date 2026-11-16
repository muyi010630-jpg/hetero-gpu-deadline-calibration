"""Reproduce Figure 3.1 from E1 measured execution-time medians.

Each context ratio is true_s(device) / true_s(RTX5070).  Family values
are equal-weight arithmetic means of those matched-context ratios.
The audit CSV retains all 45 context ratios and all 9 plotted means.
"""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image


BASE = Path(r"<SOURCE_WORKSPACE>/EMA/性能画像实验/00_统一汇总")
OUT = Path(__file__).resolve().parent
FAMILIES = ["compute", "memory", "inference"]
DEVICES = ["RTX5070", "T4", "A100"]
CONTEXT = ["family", "size", "batch", "dtype"]
COUNTS = {"compute": 7, "memory": 4, "inference": 4}
SOURCE = BASE / "profile_three_gpu.csv"


def prepare_data():
    source = pd.read_csv(SOURCE)
    assert len(source) == 45
    assert set(source.device) == set(DEVICES)
    assert source.groupby("device").size().to_dict() == dict.fromkeys(DEVICES, 15)
    assert len(source[CONTEXT].drop_duplicates()) == 15
    assert not source.duplicated(CONTEXT + ["device"]).any()
    assert np.isfinite(source.true_s).all() and (source.true_s > 0).all()
    reference = source.loc[source.device == "RTX5070", CONTEXT + ["true_s"]]
    reference = reference.rename(columns={"true_s": "reference_true_s"})
    ratios = source.merge(reference, on=CONTEXT, validate="many_to_one")
    ratios["ratio_vs_reference"] = ratios.true_s / ratios.reference_true_s
    assert reference.groupby("family").size().to_dict() == COUNTS
    grouped = ratios.groupby(["family", "device"], sort=False)["ratio_vs_reference"]
    stats = grouped.agg(contexts="count", mean_ratio="mean", median_ratio="median",
                        min_ratio="min", max_ratio="max").reset_index()

    summary = pd.read_csv(BASE / "ratio_summary_by_family.csv")
    merged_summary = summary.merge(stats, on=["family", "device"],
                                   suffixes=("_stored", "_computed"), validate="one_to_one")
    assert len(merged_summary) == 6
    summary_errors = {}
    for name in ["mean_ratio", "median_ratio", "min_ratio", "max_ratio"]:
        error = np.abs(merged_summary[name + "_stored"] - merged_summary[name + "_computed"])
        summary_errors[name] = float(error.max())
        assert (error <= 1e-6).all(), (name, error)
    assert (merged_summary.contexts_stored == merged_summary.contexts_computed).all()

    stored_ratios = pd.read_csv(BASE / "three_gpu_ratios_vs_5070.csv")
    merged_ratios = stored_ratios.merge(ratios, on=CONTEXT + ["device"],
                                       suffixes=("_stored", "_computed"), validate="one_to_one")
    assert len(merged_ratios) == 30
    context_error = np.abs(merged_ratios.ratio_vs_rtx5070 - merged_ratios.ratio_vs_reference)
    assert (context_error <= 1e-6).all()
    np.testing.assert_allclose(merged_ratios.true_s_stored, merged_ratios.true_s_computed,
                               atol=0, rtol=0)
    np.testing.assert_allclose(merged_ratios.rtx5070_true_s, merged_ratios.reference_true_s,
                               atol=0, rtol=0)

    matrix = stats.pivot(index="family", columns="device", values="mean_ratio")
    matrix = matrix.loc[FAMILIES, DEVICES]
    expected = np.array([[1, 3.186817, 0.506610], [1, 2.312595, 0.421787],
                         [1, 1.727860, 1.431297]])
    np.testing.assert_allclose(matrix.values, expected, atol=1e-6, rtol=0)

    context_audit = ratios[CONTEXT + ["device", "true_s", "reference_true_s",
                                    "ratio_vs_reference"]].copy()
    context_audit.insert(0, "record_type", "context_ratio")
    context_audit["contexts"] = 1
    mean_audit = stats.copy().rename(columns={"mean_ratio": "ratio_vs_reference"})
    mean_audit.insert(0, "record_type", "family_mean")
    audit = pd.concat([context_audit, mean_audit], ignore_index=True)
    audit["reference_device"] = "RTX5070"
    audit["source_file"] = SOURCE.name
    audit["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    audit = audit[["record_type", "family", "device", "reference_device", "size", "batch",
                   "dtype", "contexts", "true_s", "reference_true_s", "ratio_vs_reference",
                   "median_ratio", "min_ratio", "max_ratio", "source_file", "source_sha256"]]
    audit.to_csv(OUT / "图3-1_绘图数据.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    return matrix, summary_errors, float(context_error.max())


def render(matrix, language):
    cn = language == "中文"
    font_manager.fontManager.addfont(r"<SYSTEM_FONTS>/msyh.ttc")
    font_manager.fontManager.addfont(r"<SYSTEM_FONTS>/arial.ttf")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial"] if cn else ["Arial", "Microsoft YaHei"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "axes.unicode_minus": False, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(3.7, 2.65), dpi=150)
    ax = fig.add_axes([0.31, 0.215, 0.46, 0.645])
    cmap = LinearSegmentedColormap.from_list("blue_white_orange", ["#2166AC", "#F7F7F7", "#D97720"])
    heat = ax.imshow(np.log2(matrix.values), cmap=cmap, norm=Normalize(-2, 2),
                     interpolation="none", aspect="equal")
    ax.set_xticks(range(3), ["RTX 5070", "T4", "A100"])
    ax.set_yticks(range(3), [f"{family} (n={COUNTS[family]})" for family in FAMILIES])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=5)
    ax.set_ylabel("任务族" if cn else "Task family", labelpad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{matrix.iloc[row, col]:.2f}×", ha="center", va="center",
                    color="#161616", fontsize=9)
    cax = fig.add_axes([0.815, 0.24, 0.03, 0.55])
    colorbar = fig.colorbar(heat, cax=cax, ticks=[-2, -1, 0, 1, 2])
    colorbar.ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    colorbar.ax.tick_params(length=2, width=0.5, pad=3, labelsize=7.5)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.set_title("倍率" if cn else "Ratio", fontsize=8, pad=5)
    fig.text(0.56, 0.137, "执行时间倍率（相对 RTX 5070）" if cn else "Execution-time ratio relative to RTX 5070",
             ha="center", va="center", fontsize=8)
    fig.text(0.56, 0.065, "< 1：更快     > 1：更慢" if cn else "< 1: faster     > 1: slower",
             ha="center", va="center", fontsize=8)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < -0.5 or bounds.y0 < -0.5 or bounds.x1 > fig.bbox.width + 0.5 or bounds.y1 > fig.bbox.height + 0.5:
            outside.append((text.get_text(), tuple(bounds.bounds)))
    assert not outside, f"Text extends beyond figure canvas: {outside}"
    prefix = OUT / f"图3-1_任务设备性能画像热力图_{language}"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=600, transparent=False)
    plt.close(fig)
    with Image.open(prefix.with_suffix(".png")) as image:
        assert image.size == (2220, 1590), image.size
        assert abs(image.info["dpi"][0] - 600) < 0.1
        return {"language": language, "pixels": image.size, "dpi": image.info["dpi"],
                "width_mm": 3.7 * 25.4, "height_mm": 2.65 * 25.4,
                "text_bounds_checked": True}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix, summary_errors, context_error = prepare_data()
    images = [render(matrix, language) for language in ("中文", "英文")]
    print(json.dumps({"source_rows": 45, "matched_contexts": 15, "family_context_counts": COUNTS,
                      "arithmetic_mean_ratios": matrix.to_dict(),
                      "max_absolute_error_vs_family_summary": summary_errors,
                      "max_absolute_error_vs_context_ratios": context_error,
                      "audit_records": 54, "images": images, "output_directory": str(OUT)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
