# 图3：共同反馈下目标单元的早期预测收敛

插入位置：第5.2节，表8之后。输出中文、英文的600 dpi PNG及矢量PDF、SVG，图幅180 mm × 100 mm。图内仅保留坐标轴、刻度和两行方法图例；样本数与置信区间说明放在caption中。`captions.json`只包含可直接入论文的`zh`和`en`文本。

本图直接使用2026-10-09已保存的四方法逐事件预测记录，未重新采集GPU数据，未运行新实验，未重建或更新预测器，未重新选择参数。原记录在三类GPU实测画像基础上构造受控共同反馈流，因此应表述为“共同反馈受控实验”，不可表述为30次真实GPU部署。

## 中文图注（可直接使用）

图3 共同反馈下目标单元的早期预测收敛。横轴为每个设备型号—任务上下文单元的前k次预测（k=1至10），纵轴为这些预测的累计MAPE。Fixed、LocalEMA（匹配步长）、分层EMA和PoolEMA对同一事件先预测，再使用相同反馈更新；三种在线校准方法采用匹配的任务族步长。每条任务流包含45个目标单元，曲线为30条任务流的等权均值，阴影为按任务流重采样5000次得到的逐点边际95%置信区间。首次目标预测时其他单元可能已提供共享历史。目标单元的前k次预测与表8定义的全流前20%冷阶段属于不同评价子集；区间条件于同一固定画像失配环境，不表示跨环境不确定性，预测误差改善也不能单独证明调度或SLA收益。

## English caption

Figure 3. Early prediction convergence under shared feedback. The x-axis denotes the first k predictions for each device-model–task-context cell (k = 1–10), and the y-axis shows their cumulative MAPE. Fixed, LocalEMA (matched step sizes), Hierarchical EMA, and PoolEMA predict before receiving the same feedback; the three online calibration methods use matched task-family step sizes. Each stream contains 45 target cells. Lines show equally weighted means over 30 streams, and shading denotes pointwise marginal 95% confidence intervals from 5,000 stream-level bootstrap resamples. Other cells may already have supplied shared history before a target cell's first prediction. The first k target-cell predictions and the first 20% of the stream used to define the cold phase in Table 8 are different evaluation subsets. Intervals are conditional on one fixed profile-mismatch environment and do not quantify cross-environment uncertainty; prediction-error reductions alone do not establish scheduling or SLA gains.

## 数据与口径

- 数据源：`<SOURCE_WORKSPACE>/pooled_ema_cpu_20261009/01_independent_calibration/raw/per_seed/seed_101_trace.csv`至`seed_130_trace.csv`，共30条流，每流1800事件、每事件4种方法。
- 原始标识：`fixed`→Fixed；`family_ema`→LocalEMA（匹配步长）/LocalEMA (matched step sizes)（这是任务族匹配步长的局部状态估计器，不是组级池化）；`hierarchical_ema`→分层EMA/Hierarchical EMA；`pool_ema`→PoolEMA。
- 目标单元：`(device, context_key)`，每流45个；图中k依据各方法共有的`local_observation_index`，不是PoolEMA的组更新次数，也不是Fixed恒为0的估计器状态次数。
- 第k个图点纳入每个目标单元的第1至k次预测，全流内筛选，不限制事件进度；每种子每方法恰为45k个预测，30流共1350k个预测。因此流等权均值与这些记录的事件池化MAPE一致。
- APE为`100 × abs(predicted_s − observed_s) / observed_s`。原始`ape`已逐行重新计算核验。
- 置信区间：30条任务流为重采样单位，5000次百分位bootstrap，`random.Random(20261009)`；四方法各k共享同一批流索引抽样，非同时区间，不添加显著性星号，不把逐事件记录视作独立样本。
- 源实验先预测后更新的逻辑见归档`run_pool_ema_calibration.py`的`execute_seed`；制图核验共同事件、反馈、局部预测序号及更新前计数，共检查216000行。源文件SHA256见`source_manifest_sha256.csv`。

## 与表8及归档结果核对

总体MAPE依次为20.475172%、7.628711%、7.495057%、6.954112%；前20%冷阶段MAPE依次为19.835221%、10.040609%、9.431534%、7.310411%，均与表8来源汇总逐数核对。

图中k=1的累计MAPE依次为19.439091%、19.439091%、15.212875%、10.778908%；k=10依次为19.490791%、9.171827%、8.637209%、6.927148%。k=1/3/5/10全部16个均值与原归档`summary_first_k.csv`的`first_k_all_stream`口径核对；总体及冷阶段另核对8个均值，数值差不超过1e-10个百分点。图曲线并不要求收敛到表8全流均值，因为前10次目标预测与全流事件是不同评价子集。

## 复现

```powershell
& '<USER_HOME>/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' '<LOCAL_WORKDIR>/论文撰写/图片/制图数据_20261014/图3/build_figure3.py' --packages '<LOCAL_WORKDIR>/_scratch/figure1_1_bilingual_20261011/python-deps'
```

输出`first_k_per_seed.csv`保留1200个流—方法—k均值；`first_k_curve.csv`保留40个图点及区间；`archived_summary_crosscheck.csv`保留24项汇总核对；`validation.json`保留逐行与汇总验证结果。输出及代码校验值见`output_manifest_sha256.csv`。

默认复现命令只读取已保存的`first_k_curve.csv`并重新导出图片，保持原数据和CI文件不变。仅当显式增加`--recompute`时才重新汇总已有预测记录；不会生成新实验。
