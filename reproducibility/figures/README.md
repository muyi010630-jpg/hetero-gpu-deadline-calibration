# 主文图表对应关系

`manuscript_figure_1.png` 至 `manuscript_figure_6.png` 是直接从本次最终主稿提取的图像，与正文嵌入对象一致。

| 图 | 保存数据与实现 | 本次处理 |
| --- | --- | --- |
| 1 闭环框架 | `reference_figure1/` 中的现存框架绘图代码 | 保留原图，主稿图注明确限于回放流程。该代码为可审阅参考，未证明与原嵌入图逐字节同版。 |
| 2 性能热力图 | `reference_figure2/` 的绘图代码及45单元画像 | 保留原图；倍率由同上下文耗时比再按任务族算术平均。 |
| 3 早期预测收敛 | `reference_figure3/first_k_curve.csv`、`first_k_per_seed.csv` 与原脚本 | 保留原图与保存的均值、区间。原脚本还依赖原始事件记录及旧目录层级，不作为包内完整重跑入口。 |
| 4 共享比例比较 | `figure4_plot_data.csv`、`figure4_paired_image_differences.csv`、`redraw_figures4_6.py` | 删除真实ρ标线，保留原点值、区间及校正统计量。 |
| 5 覆盖率与假安全率 | `reference_figure5/figure5_plot_data.csv`、两份场景均值CSV与原脚本 | 保留原图。 |
| 6 三主机配对差 | `figure6_statistics.csv`、`figure6_points.csv`、`redraw_figures4_6.py` | 统一Full-Deploy标签，保留两批点值及既定合并区间。 |

`redraw_figures4_6.py` 只读取已保存统计量，输出中英文PNG/SVG/PDF，不重做统计检验。依赖NumPy、pandas、Matplotlib和Pillow；当前中文/英文字体路径为Windows的微软雅黑/Arial，其他平台需配置等价字体。`reference_figure*/` 保存历史实现以供核对，可能含 `$WORKSPACE` 或旧相对层级，不能直接视为已验证的独立执行入口。历史图注和核验记录中的表号属于当时版本，以本次主稿编号为准。
