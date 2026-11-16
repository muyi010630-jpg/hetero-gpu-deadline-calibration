# TF32 Diagnostic Data and Code

The accompanying data and code are submitted for review and will be published as Supplementary Materials with the article upon acceptance. Author-created source code is licensed under the MIT License in LICENSE_CODE.txt. Author-created data and explanatory documentation are licensed under CC BY 4.0 in LICENSE_DATA.txt. Third-party material and external dependencies retain their original terms. Cite the article identified above and add its DOI when available.

This portable delivery accompanies the article and Supplementary_Materials.pdf. See Reproducibility_Guide.pdf and PORTABILITY_NOTES.md for current names, analysis commands, and checksum scopes. The dated collection notes below are retained as historical records.

2026-10-18完成。A100、T4、RTX 5070各3个新进程轮次，按预定ABA/BAB/ABA切换；每轮4个上下文、3阶段，每阶段20个窗口，每窗口10次GEMM。正式数据共2160个计时窗口、21600次被计时GEMM，另有计时外预热。

## 阅读入口

- `analysis/`：最终分析报告、校验结果、阶段及轮次汇总、中英文图与图注。
- `raw/A100/round_01..03`、`raw/T4/round_01..03`、`raw/RTX5070/round_01..03`：全部正式原始记录。
- 每轮的 `windows.csv` 为窗口计时；`stages.csv` 为阶段统计；`round_summary.csv` 为轮次耗时比、恢复比及数值差异；`numerical_differences.csv` 为CPU float64计算的相对Frobenius差异；`output_checks.csv` 记录输出有限性与哈希。
- 每轮 `environment.json`、`configuration.json`、`input_manifest.json` 保存实际环境、设置及输入哈希；`telemetry.jsonl` 保存阶段边界遥测；三份进程归属文件保存初始化前后查询及归属依据。
- `transfer/`：云端上传脚本及回传文件的SHA256清单。两台云端的全部回传文件已在本机复核。
- `collection_status.json`：汇总完成状态与2160条计时计数。`postflight_*.json`：云端实验结束状态；实验进程已退出，两卡无残留计算任务。
- `protocol/experiment_plan_original.md`：执行依据的原始方案；`scripts/protocol.json` 与 `scripts/windows_v4/protocol.json` 是实际冻结配置。

## 实际环境

A100 and T4 used PyTorch 2.5.1+cu124, CUDA 12.4, and driver 550.90.07. RTX 5070 used PyTorch 2.11.0+cu128 and CUDA 12.8. The author-confirmed RTX5070 driver version is 616.64. DRIVER_METADATA_CORRECTION.md and DRIVER_METADATA_CORRECTIONS.csv link the recorded driver values to the author's confirmation on 10 November 2026. Round environment JSON files retain the recorded metadata. Interpret their RTX5070 driver entries using the author correction. Other environment fields follow the per-round records.

## 完整保留的实施调整

正式测量前发现跨NumPy版本的正态采样输入哈希不一致，故统一为PCG64原始位流高24位构造精确FP32输入；三卡CPU输入检查一致后才启动正式测量。没有根据加速幅度选择上下文、尺寸、种子或数学模式。

云端NVML显示宿主PID而Python显示容器PID，因此在成功空基线、自己的输入复制到GPU并同步后，要求所选GPU只有一个新PID并固定登记；这一对应关系明确标为操作性推断。保留全部原始查询，不能称为内核层直接证明的命名空间映射。

Windows v3在空进程基线门禁处失败，计时数为0，完整保留在`attempts/RTX5070_v3_pre_cuda_gate`；云端正在运行的v3没有中断或重跑。Windows专用v4只调整桌面进程库存和自身PID直接匹配规则，测量设计及输入不变。失败尝试未进入2160条正式数据。正式Windows查询库存为空，而此前提升权限的查询列出桌面程序（包括WallpaperEngine），这种可见性差异不能被解释成无桌面负载。RTX 5070结果限定于当时实际桌面环境，不能完全排除背景干扰。

`smoke_round_*`只属于一次接口与输出检查，`input_check_round_*`仅为CPU输入核验；两者均不属于正式统计。旧版脚本及失败协调记录也已保留。未按快慢删点、未用重跑替换正式数据；未运行调度器、未注入人工耗时倍率、未修改原有论文实验数据或服务器功耗/频率设置。

## 解释边界

R=T_off/T_on；R>1表示允许TF32时更快。恢复比为阶段3/阶段1，三个轮次只作描述性重复，不将20个窗口当作独立硬件样本，不给出伪精确的显著性推断。数值差异以禁止TF32的输出为比较参考，不把它称作数学真值。

本实验检验软件数学设置是否可能使固定性能画像失效。它不追溯历史部署偏差，不确认具体Tensor Core内核，不隔离不同GPU之间的架构效应，也不直接证明PoolEMA优越或调度收益提高。详细结论以`analysis/`内的最终报告为准。

云端原始目录仍保留：`<REMOTE_WORKDIR>/ei_tf32_mechanism_20261018`。本交付不含服务器密码。
