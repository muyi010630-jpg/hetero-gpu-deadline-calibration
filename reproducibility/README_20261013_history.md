# 论文本地复现材料，2026-10-13

本目录是定稿阶段整理的本地材料，尚未向外网发布、未获得公开仓库地址或 DOI。不能将本目录的存在表述为“公开材料已可访问”。对外获取渠道、许可证和作者信息需要作者在投稿前确定。

## 当前材料能完成什么

- 复算三主机两批共 60 次运行的绝对指标：按运行等权；两批各 5 种子、2 负载、3 策略。保留 12 行批次汇总及 6 行合并汇总。
- 从包内已保存 CSV 独立复算 RQ2 的 25 项补充比较（500 个画像内配对差）、概率诊断 24/120/600 秒分层均值与区间，并核对异质性诊断的条件均值及零权重比例。结果与本轮保存结果逐项一致。
- 查阅四组 CPU 补充验证的冻结代码、输入画像、开发/留出配置、画像块配对表、概率与决策统计输出。目录名称和相对实验结构保留，避免混淆不同批次或统计族。
- 核验每次部署配置、输入画像校验值及 Worker 就绪驱动记录；查阅统计检验及公平性统计的保存结果。
- 对照源文件和包内文件的分别校验值。包内文本为去除端点、用户目录与绝对工作路径后的副本，部分序列化发生规范化；不能用包内校验值冒充历史源文件校验值。

## 已验证的无 GPU 复算命令

依赖 Python 3.10 或以上，仅标准库。解压后在本目录运行：

```text
python rebuild_deployment_summaries.py
```

结果写入 `recomputed/`，脚本同时核对保存的修复表。运行输出应报告 60 个运行、12 个批次组和 6 个合并组。此处从逐运行指标复算，不重新读取任务级原始日志，也不重新估计任务级 P99。原运行 P99 按运行内最近秩计算，当前脚本仅对这些 P99 跨运行等权平均。

以下命令复算本轮补分析，依赖 NumPy 2.3.5、pandas 3.0.1（已在 Python 3.12.14 环境核验）：

```text
python recompute_supplemental_analyses.py
```

输入全部位于本包：RQ2 共同反馈与闭环画像块 CSV；概率诊断逐运行 CSV；H 逐流及条件汇总 CSV。输出位于 `recomputed/supplemental_analyses/`。脚本使用保存补分析脚本的数值函数和已记录种子；不调用模拟器，不重新选择参数，不更新估计器。

`supplemental_analyses/RQ2/`、`supplemental_analyses/diagnostics/` 保存分析方案、原始脚本、CSV/JSON/MD结果、交叉核对与来源哈希。这些分析均发生在原留出完成之后，身份为补充分析，不属于留出前预注册。RQ2 的 25 比较单独构成 Holm 校正族；概率时长分层属于描述性区间，不追加主显著性检验。

早期交付仅保留 H 的来源校验、事件级统计、1000 行逐流汇总和20画像分块结果。20261109交付已在 supplemental_analyses/diagnostics/h_events/ 补入全部1000流的七个必要诊断字段，共180万条事件，并逐项核对原始分片哈希。运行 recompute_h_event_diagnostics.py 可复算事件均值、比例及分位数。完整原始NPZ还含其他数组，未随本包提供。source_to_extract_sha256.csv分别标识完整来源与字段提取文件。

CPU 实验实现保存在 `cpu/`。原实验环境为 Python 3.12.14、NumPy 2.3.5；相邻实验的基础画像和冻结支持代码已经同时纳入。代码中的本机绝对路径已替为 `$WORKSPACE`，这些副本用于实现核查；原研究目录中的方法代码及冻结参数没有改动。尚未完成所有实验入口的迁移及完整重跑测试，因此不宣称本包能够直接独立重跑全部 CPU 实验。大规模 raw/traces 分片、缓存和 GPU 连接信息均未纳入。

## 部署材料与批次标错修复

`deployment/per_run_summary_10seed.csv` 的历史 batch 字段正确。原 `summary_by_load_strategy_by_batch.csv` 在双重批次循环的空子集中改写了前六行的标签；本包改为直接按 batch、load、strategy 分组，修复输出存于 `deployment/summary_by_load_strategy_by_batch_corrected.csv`。原实验文件没有覆盖或修改；修复后合并绝对均值与原正确 10 种子均值逐项一致。

- `deployment/deployment_run_index.csv`：两批 60 个运行与实跑配置、输入和 Worker 脚本校验值对应。
- `deployment/runs/`：逐运行配置及驱动就绪记录。连接地址、用户名、端口等字段已经移除。
- `deployment/profiles/P1.csv`、`P2.csv`：批次输入画像。原文件校验值分别为 `643704fa8b8f1efe2115692b643a461ea7c514641ebcd941f70e0e561e2da116`、`6ef6a9268e99022c6cd02405f7f060f4302f64a4271b38dffb93f6a16b10bf79`。包内序列化校验值以 manifest 为准。
- `deployment/controller_traceability.csv`：现存两份控制器源码的校验值及追溯限制。运行配置保存了 Worker 脚本校验值，却未记录完整控制器校验值；现存源码本身不足以证明每次历史运行使用了逐字节一致的控制器快照。Worker 脚本 hash 也不代表整体系统 hash。

## 目录与证据索引

| 目录 | 论文证据 | 核验边界 |
| --- | --- | --- |
| `cpu/pooled_ema_cpu_20261009/` | 基本在线校准、池化对照、共同风险点估计对照 | 冻结配置比较，不独立归因共享机制 |
| `cpu/heterogeneity_adaptive_cpu_20261009/` | 共享比例、反馈分布、公共面板与画像块分析 | 多幅受控失配输入，不能当新增实体环境 |
| `cpu/probability_calibration_cpu_20261010/` | 概率校准、Brier、三时长诊断及六项 SLA 比较 | 公共面板与所选反馈分开，多个统计族分开 |
| `cpu/decision_feedback_cpu_20261010/` | 决策六比较及探测诊断 | 历史、额外标签与合法探测的信息权限不同 |
| `deployment/` | Fixed、EMA、部署 Full 的绝对指标和种子内配对 | 每运行 100 任务，两批环境和参数存在差异 |

本包未包含作者单位、作者贡献、基金编号或通信作者联系渠道，均不可由实验目录推定。后续三主机 PoolEMA/LocalWindow24 实验、完整可移植重跑入口及公开归档不属于已完成材料。

`package_manifest_sha256.csv` 是最终包内文件完整索引；`source_and_package_manifest_sha256.csv` 同时列原证据文件与规范化包内副本校验值。验证状态保存于 `recomputed/verification.json` 和 `recomputed/supplemental_analyses/verification.json`。外部获取入口待作者确认，不用本机绝对路径替代投稿时的可获取渠道。
