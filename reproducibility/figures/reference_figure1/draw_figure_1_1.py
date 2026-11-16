"""Draw the introduction framework as an editable, journal-style vector figure."""
from pathlib import Path
import json
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"$WORKSPACE")
parser = argparse.ArgumentParser(description="绘制引言中的在线校准与QoS调度闭环图")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--qa-dir", type=Path, default=ROOT / "_scratch" / "figure1_1_academic_20261007")
args = parser.parse_args()
OUT, QA = args.output_dir.resolve(), args.qa_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
for name in ("msyh.ttc", "msyhbd.ttc", "arial.ttf"):
    font_manager.fontManager.addfont(Path(r"<SYSTEM_ROOT>/Fonts") / name)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "path",
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})
W, H = 194, 100


def draw(grayscale=False):
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.set_axis_off()
    dark = "#26323B"
    gray = "#79858F"
    border = "#B1BAC2"
    blue = "#365F82" if not grayscale else "#4B4B4B"
    tint = "#F0F5F9" if not grayscale else "#F2F2F2"
    light = "#FAFBFC"
    checks = []

    def text(x, y, label, size=8, color=dark, weight="normal", ha="center", box=None):
        artist = ax.text(x, y, label, fontsize=size, color=color,
                         fontweight=weight, ha=ha, va="center", linespacing=1.2)
        if box:
            checks.append((artist, box))
        return artist

    def rect(x, y, w, h, fill="white", edge=dark, lw=.8, dash=None):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                               linewidth=lw, linestyle=dash or "solid"))

    def arrow(points, feedback=False):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=7.2,
                                    linewidth=.95, color=blue if feedback else dark,
                                    linestyle=(0, (3.8, 2.5)) if feedback else "solid",
                                    capstyle="butt", joinstyle="miter"))

    # Separate preparation from task flow: the fixed model does not cause arrivals.
    rect(4, 74, 76, 21, edge=border, lw=.65, dash=(0, (4, 3)))
    rect(82, 74, 108, 21, edge=border, lw=.65, dash=(0, (4, 3)))
    rect(4, 7, 186, 66, edge=border, lw=.65, dash=(0, (4, 3)))
    text(8, 97.5, "离线准备", size=8.2, color=gray, weight="bold", ha="left")
    text(86, 97.5, "调度约束与状态", size=8.2, color=gray, weight="bold", ha="left")
    text(8, 69.4, "在线调度闭环", size=8.2, color=gray, weight="bold", ha="left")

    rect(8, 79, 31, 12, fill=light, edge=border, lw=.75)
    text(23.5, 87, "上下文性能画像", size=8.1, weight="bold", box=(8, 84, 31, 7))
    text(23.5, 81.7, "任务—设备重复测量", size=7.4, box=(8, 79, 31, 5))
    rect(45, 79, 31, 12, fill=light, edge=border, lw=.75)
    text(60.5, 87, "固定初始模型", size=8.1, weight="bold", box=(45, 84, 31, 7))
    text(60.5, 81.5, r"$T_0(x,g)$", size=9.4, box=(45, 79, 31, 5))
    arrow([(39, 85), (45, 85)])
    arrow([(60.5, 79), (60.5, 64)])

    rect(86, 79, 100, 12, fill=light, edge=border, lw=.75)
    text(136, 87.2, r"QoS等级 $q_j$ · 截止期 · 用户服务欠额 · 队列状态", size=8.1, box=(86, 84, 100, 7))
    text(136, 81.5, "显存容量 · 设备可用性", size=8.1, box=(86, 79, 100, 5))
    # The QoS level also determines the empirical risk margin.
    arrow([(92, 79), (92, 67.5), (70.5, 67.5), (70.5, 64)])
    text(81.0, 69.4, r"$q_j$", size=8.1, color=dark)
    arrow([(97.5, 79), (97.5, 64)])
    arrow([(134.5, 79), (134.5, 64)])

    xs = (8, 45, 82, 119, 156)
    titles = ("任务输入", "校准预测", "QoS评分", "设备分配", "GPU执行")
    bodies = (
        ((49.4, "任务到达", 8.0), (41.8, "上下文提取 " + r"$x$", 8.0)),
        ((52.1, "候选设备服务时间", 7.7),
         (47.1, "点预测 " + r"$\hat{T}_t(x,g)$", 8.0),
         (41.8, "经验上界 " + r"$U_t(x,g,q_j)$", 8.0)),
        ((52.1, "经验上界 " + r"$U_t$", 8.0),
         (47.1, "截止期风险与任务排序", 7.7),
         (41.8, "用户服务欠额", 8.0)),
        ((52.1, "点预测 " + r"$\hat{T}_t$", 8.0),
         (47.1, "可行设备检查", 8.0),
         (41.8, "选择执行设备", 8.0)),
        ((49.4, "非抢占任务执行", 8.0), (41.8, "采集GPU执行耗时", 8.0)),
    )
    for i, (x, title, rows) in enumerate(zip(xs, titles, bodies)):
        rect(x, 38, 31, 26, fill=tint if i == 1 else "white",
             edge=blue if i == 1 else dark, lw=.85)
        ax.plot([x, x + 31], [55.5, 55.5], color=border, linewidth=.55)
        text(x + 15.5, 59.7, title, size=9.1, weight="bold",
             color=blue if i == 1 else dark, box=(x, 55.5, 31, 8.5))
        for y, label, size in rows:
            text(x + 15.5, y, label, size=size,
                 box=(x, y - 2.3, 31, 4.6))
        if i < 4:
            arrow([(x + 31, 51), (xs[i + 1], 51)])

    # The point prediction reaches device selection separately from risk scoring.
    arrow([(69.5, 38), (69.5, 34.1), (134.5, 34.1), (134.5, 38)])
    text(103.0, 31.5, "点预测用于设备选择", size=7.3,
         box=(71, 29.7, 64, 3.6))

    # The feedback updates only residual state. The offline T0 remains fixed.
    rect(45, 10, 56, 19, fill=tint, edge=blue, lw=.85)
    text(73, 25.3, "分层在线校准", size=9.1, color=blue, weight="bold", box=(45, 22, 56, 7))
    text(73, 19.1, "EMA残差更新", size=8.0,
         color=blue, box=(45, 16.3, 56, 5.7))
    text(73, 13.1, "稀疏观测时借用相关历史", size=8.0,
         color=blue, box=(45, 10, 56, 6.3))
    rect(131, 10, 56, 19, fill=light, edge=blue, lw=.85)
    text(159, 25.3, "执行反馈", size=9.1, color=blue, weight="bold", box=(131, 22, 56, 7))
    text(159, 19.1, "已完成任务的GPU耗时", size=8.0,
         color=blue, box=(131, 16.3, 56, 5.7))
    text(159, 12.7, r"$z_n=\log\!\left(T_{\mathrm{obs},n}/T_0\right)$", size=9.2,
         color=blue, box=(131, 10, 56, 6.3))
    arrow([(171.5, 38), (171.5, 29)], feedback=True)
    arrow([(131, 19.5), (101, 19.5)], feedback=True)
    text(116, 24.7, "对数残差", size=7.8, color=blue, box=(101, 22, 30, 7))
    arrow([(60.5, 29), (60.5, 38)], feedback=True)
    text(58.1, 33.4, "后续预测", size=7.0, color=blue, ha="right")

    arrow([(8, 3.4), (17, 3.4)])
    text(19, 3.4, "前向处理", size=7.4, color=gray, ha="left")
    arrow([(51, 3.4), (60, 3.4)], feedback=True)
    text(62, 3.4, "反馈更新", size=7.4, color=gray, ha="left")
    text(93, 3.4, "图示为回放主实现；真实闭环采用部署变体", size=7.2,
         color=gray, ha="left", box=(92, 1.0, 98, 4.8))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, bounds in checks:
        extent = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        x, y, w, h = bounds
        assert x + .5 <= extent.x0 and extent.x1 <= x + w - .5, (artist.get_text(), extent)
        assert y <= extent.y0 and extent.y1 <= y + h, (artist.get_text(), extent)
    return fig


def main():
    stem = "图1-1_在线校准与QoS调度闭环_学术版"
    fig = draw()
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=600, pad_inches=0)
    fig.savefig(QA / "preview.png", dpi=180)
    plt.close(fig)
    gray = draw(grayscale=True)
    gray.savefig(QA / "grayscale.png", dpi=180)
    plt.close(gray)
    with Image.open(OUT / f"{stem}.png") as raster:
        report = {"size_mm": [W, H], "png_pixels": raster.size,
                  "dpi": raster.info.get("dpi"), "text_bounds": "passed",
                  "feedback_target": "hierarchical residual state; fixed T0 remains unchanged",
                  "prediction_roles": {"point": "device selection", "empirical_upper": "deadline risk and task ordering"},
                  "implementation_scope": "replay main implementation; real closed-loop is a deployment variant",
                  "outputs": [str(OUT / f"{stem}.{s}") for s in ("png", "svg")]}
    (QA / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
