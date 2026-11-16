# Reproducibility Data and Code

The accompanying data and code are submitted for review and will be published as Supplementary Materials with the article upon acceptance. Author-created source code is licensed under the MIT License in LICENSE_CODE.txt. Author-created data and explanatory documentation are licensed under CC BY 4.0 in LICENSE_DATA.txt. Third-party material and external dependencies retain their original terms. Cite the article identified above and add its DOI when available.

This portable delivery accompanies Manuscript.docx and Supplementary_Materials.pdf. Reproducibility_Guide.pdf gives the current article figure/table mappings. PATH_RENAMING_MAP.csv and PORTABILITY_NOTES.md describe the delivery names, relative paths, and checksum scopes. The dated source notes below retain historical context; use the current guide when a historical section or figure number differs.

## RTX5070 driver metadata correction

The author-confirmed RTX5070 driver version is 616.64 for all experiments in the accompanying study. DRIVER_METADATA_CORRECTION.md and DRIVER_METADATA_CORRECTIONS.csv record the author's confirmation on 10 November 2026 and link the conflicting recorded values to their source-file hashes. For the RTX5070 driver version, this correction takes precedence over the environment files and dated notes. Original recorded metadata are retained for traceability.

## 已验证的复算入口

解压后运行以下命令，Python 3.10或以上，仅使用标准库：

```text
python recompute_revision.py
python rebuild_deployment_summaries.py
```

第一条从包内6000条任务记录和6000条任务计划核对60次运行的任务集合、成功状态和违约标志，重新计算运行内最近秩P99、两批绝对均值与种子内配对差；从两侧原始画像样本核对45单元的样本量、中位数、波动及描述性平方和分解。结果写入 `recomputed/revision/`，应报告PASS、60运行、6000成功任务、12行分批指标、45画像单元。

第二条从保存的逐运行指标重建12个批次组与6个合并组，修复历史汇总的批次标签错误；它不重新估计任务级P99。两条复算均不新建显著性检验、不重选统计族。

```text
python recompute_supplemental_analyses.py
```

此命令依赖NumPy和pandas，从已保存CSV复算RQ2的25项补充比较及概率时长诊断等，结果写入 `recomputed/supplemental_analyses/`。依赖记录见 `requirements_recompute.txt`。本次验证输出一并保存；不会调用GPU、模拟器或重新选参。

## 文件与论文位置

| 位置 | 文件 | 内容 |
| --- | --- | --- |
| 主文3.6、补充S3 | `deployment/calibrator_reference.py`、`verified_parameters.csv` | Fixed/EMA/Full-Deploy完整校准定义和60份配置参数；参考源码仅提取校准类，不含连接/派发操作。 |
| 主文表6、补充S4 | `deployment/runs/`、`deployment_run_index.csv`、`controller_traceability.csv` | 批次、画像、驱动与现存控制器追溯范围。 |
| 主文表13、补充S5/S6 | `deployment/task_records.csv`、`task_plans.csv`、`task_status_audit.csv`、`metrics_from_tasks.csv`、`batch_absolute_and_paired.csv` | 任务集合、分母、运行内P99及分批绝对/配对结果；率差用百分点，时延差用秒。 |
| 主文4.1/5.6、补充S7 | `profiles/` | 三设备原始测量、2026-09-21诊断画像、每单元样本统计、环境及迁移分解。 |
| 主文5.2/5.4 | `main_experiments/` | 主回放、独立预测保存的结果及配置。 |
| 主文5.2—5.5、补充S8/S9 | `cpu/` 四个实验目录 | 冻结方法、开发与留出配置、输入画像、代表Worker面板实现和统计结果。 |
| 补充表S1/S2 | `supplemental_analyses/`、`methods_evidence_source_20261015.docx` | 事后直接比较及描述性时长诊断；保持原统计身份。 |
| 图1—6 | `figures/README.md` | 正文嵌入图、保存制图数据、图4/6重绘脚本及其他图的实现参考。 |

## 来源与范围

两批部署分别为种子20260911—15和20260916—20，每批5种子×2配置×3策略。种子数字不作为运行日期证明。A/B同时改变到达压力与截止期；Full在源数据字段中仍用`full`，在论文中称Full-Deploy。同批设置一致不证明历史选参与评测独立，且两批不是同环境10次重复。

原 `summary_by_load_strategy_by_batch.csv` 存在批次标签错误；本包使用正确的逐运行batch字段及任务记录重新分组。修复表为 `deployment/summary_by_load_strategy_by_batch_corrected.csv`，此次关键指标表为 `batch_absolute_and_paired.csv`。原研究文件未被覆盖。

离线画像每单元2—10个有效样本，运行期诊断每单元8个样本；后者是2026-09-21独立诊断测量，不是全部部署任务。45单元等权的71.72%/28.28%是描述性平方和占比，不是随机效应方差比例、真实ρ或可回收误差比例。推理CSV旧标签size=224对应实际32×32输入。

两份现存控制器的校准定义一致，但逐运行配置未记录完整控制器校验值；现存源码哈希、Worker脚本哈希均不证明每次历史运行采用相同完整系统快照。参数来源及未追溯的早期选参过程在补充S4如实说明。

CPU原实验部分入口仍保留 `$WORKSPACE` 或旧目录结构，未验证全部实验能够独立一键重跑。其他大规模raw/traces分片未全部纳入。H诊断现已补入1000流的七个必要字段，运行recompute_h_event_diagnostics.py可复算事件均值、比例和分位数；完整原始NPZ还含其他数组，未随包提供。详见 `README_20261013_history.md` 与 `supplemental_analyses/diagnostics/source_sha256.csv`。PoolEMA/LocalWindow24真机对比、全Worker敏感性和画像确认实验不属于此次完成内容。

部分原有材料是移除连接地址、用户目录与绝对工作路径后的整理副本。`source_and_package_manifest_sha256.csv` 保留旧材料的来源/副本分别哈希，`figures/source_and_package_sha256_20261015.csv` 保留本次新增图表参考来源；`deployment/task_status_audit.csv` 保留任务计划、轨迹和配置源哈希。`package_manifest_sha256.csv` 是本次全部包内文件校验清单（不列自身）；其哈希是当前整理版本的标识，不能冒充历史运行快照。


## Retained H event data in the current delivery

The 20261109 delivery includes the seven diagnostic fields from all 1000 fixed-feedback streams, with 1800000 events and verified source-to-extract hashes, under supplemental_analyses/diagnostics/h_events/. Run python recompute_h_event_diagnostics.py to recompute the reported H-event means, fractions, and quantiles using the archived aggregation function. Complete source NPZ files include other arrays and are outside this delivery.
