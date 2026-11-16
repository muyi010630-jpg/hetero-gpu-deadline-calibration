# TF32软件计算路径与GPU画像失配：实测结果

本报告只描述本次独立机制诊断，不改动历史数据或论文。

完整设计核验：**通过**；36个设备×上下文×轮次配对比较，2160个计时窗口。每张设备只有1张实体GPU，三轮为该设备的重复测量。

正式采集沿用v2冻结的输入构造：CPU端以PCG64原始uint64高24位构造[-1,1)上的精确float32均匀网格，再按上下文转为FP16；同轮同上下文跨设备A/B字节SHA256一致。云端A100/T4采用冻结v3，Windows RTX 5070采用专用v4；两者的输入、阶段、预热、计时和统计量相同，区别在于进程归属核验与桌面背景库存记录。保留的前置烟测、纯CPU输入检查和零计时门禁失败均不进入正式计时统计。

## 耗时比、恢复程度与输出差异

R>1表示允许TF32时测得耗时较短。恢复比为同一轮阶段3/阶段1；delta在CPU float64上以禁止TF32输出为参考计算，参考输出不称为数学真值。

| GPU | 上下文 | 第1轮R | 第2轮R | 第3轮R | 三轮中位R | 恢复比范围 | delta范围 |
|---|---|---:|---:|---:|---:|---:|---:|
| A100 | FP32, N=2048 | 4.78778 | 4.80109 | 4.78504 | 4.78778 | 0.99970–1.02074 | 0.000261052–0.00026117 |
| A100 | FP32, N=4096 | 5.70068 | 5.73724 | 5.73309 | 5.73309 | 1.00339–1.01476 | 0.000261251–0.000261275 |
| A100 | FP32, N=6144 | 6.01859 | 6.04458 | 6.05818 | 6.04458 | 1.00057–1.01178 | 0.00026166–0.000261689 |
| A100 | FP16, N=4096 | 1.00836 | 1.00637 | 0.99843 | 1.00637 | 0.99470–1.02559 | 0–0 |
| T4 | FP32, N=2048 | 0.99810 | 1.00170 | 0.98867 | 0.99810 | 1.00990–1.06286 | 0–0 |
| T4 | FP32, N=4096 | 0.99647 | 0.99732 | 0.98789 | 0.99647 | 1.01340–1.06413 | 0–0 |
| T4 | FP32, N=6144 | 0.99493 | 0.99558 | 0.99249 | 0.99493 | 0.99888–1.06178 | 0–0 |
| T4 | FP16, N=4096 | 0.98747 | 0.99744 | 0.98543 | 0.98747 | 1.01659–1.06379 | 0–0 |
| RTX5070 | FP32, N=2048 | 1.32944 | 1.37835 | 1.37904 | 1.37835 | 0.99604–1.03281 | 0.00026102–0.000261143 |
| RTX5070 | FP32, N=4096 | 1.48051 | 1.52106 | 1.50749 | 1.50749 | 1.00018–1.01268 | 0.000261118–0.000261142 |
| RTX5070 | FP32, N=6144 | 1.66938 | 1.68205 | 1.64365 | 1.66938 | 0.98156–1.00248 | 0.000261351–0.000261381 |
| RTX5070 | FP16, N=4096 | 0.99484 | 0.98678 | 0.99339 | 0.99339 | 1.02500–1.02847 | 0–0 |

## 与机制解释的关系

- **A100**：三个FP32规模均在全部三轮出现R>1；FP32三轮中位R范围为4.78778–6.04458，FP16对照的三轮中位R为1.00637。这描述方向与量级，不等价于显著性结论。
- **T4**：FP32耗时比方向存在跨轮或跨规模变化；FP32三轮中位R范围为0.99493–0.99810，FP16对照的三轮中位R为0.98747。这描述方向与量级，不等价于显著性结论。
- **RTX5070**：三个FP32规模均在全部三轮出现R>1；FP32三轮中位R范围为1.37835–1.66938，FP16对照的三轮中位R为0.99339。这描述方向与量级，不等价于显著性结论。

**有限机制判断：本次结果支持后端数学模式切换能够使固定GPU、矩阵规模和输入输出dtype下的执行时间发生可重复偏离。** A100和RTX 5070的全部18个FP32轮次R均大于1；对应恢复比分别为0.99970–1.02074和0.98156–1.03281，恢复漂移相对各自FP32切换量级较小。A100提供较清楚的设备内支持，RTX 5070提供同向观察但须保留桌面背景干扰限制。T4全部上下文的R为0.98543–1.00170，且其恢复变化最大为6.41%；不能把接近1的小幅变化解释为清楚的TF32设置效应。三卡FP16对照的R为0.98543–1.00836，delta范围为0–0。两台支持TF32的设备在FP32下delta范围为0.00026102–0.000261689，性能变化伴随内部精度变化，不能表述为无条件等精度加速。该结论不延伸至历史失配归因或调度收益。

36个轮次比较中，16个的恢复绝对对数变化不小于设置耗时比的绝对对数变化，具体行见round_ratios.csv。该比较只是并列描述漂移与切换量级，不是事后合格阈值，不用于删点。

解释时应结合恢复比、阴性设备对照和FP16任务对照。若设置切换量级与恢复漂移相当，本诊断不足以清楚隔离设置影响；若设备内FP32变化跨轮一致、恢复漂移相对较小而对照较弱，则与软件数学模式和硬件能力交互影响耗时的解释一致。没有运行内核计数器，不能声称直接观察到Tensor Core利用率变化。

## 适用范围

1. 这是已知设置的主动干预，只验证所测设备、输入和软件环境；不用于追溯历史部署失配的原因。
2. 不把历史71.72%的描述性共享成分解释为TF32贡献；不测定模拟共享比例rho。
3. 不运行调度器、不比较校准器，因此不能证明PoolEMA、Full或其他策略具有QoS收益。
4. 输入输出dtype相同不等于内部有效精度相同；数值差异需要结合应用容差解释，本实验不证明应用精度等价。
5. 已知数学模式应优先写入画像元数据或触发重新画像；本实验不支持以在线校准替代显式配置管理。
6. 不同设备的软件栈不同，效应幅度差异不能单独归因于架构。

## 文件

- all_timing_windows.csv：全部原始字段、源文件与规范化字段。
- stage_summary.csv：108个阶段×上下文中位数及窗口波动描述。
- round_ratios.csv：36个每轮R、恢复比、输出差异。
- device_context_summary.csv：12个设备×上下文的三轮R及中位数。
- output_differences_CPU_float64.csv：原始输出差异字段与来源。
- validation.json：完整设计与数值检查。
- figure_captions.json：中英文论文图注。
- source_sha256.csv / output_sha256.csv：输入与输出校验清单。

图形纵轴：log2；无截断点、无显著性标记、无置信区间。

## 环境记录

| GPU | PyTorch | CUDA构建 | NumPy | 精度接口 |
|---|---|---|---|---|
| NVIDIA A100-PCIE-40GB | 2.5.1+cu124 | 12.4 | 2.1.3 | legacy_allow_tf32 |
| Tesla T4 | 2.5.1+cu124 | 12.4 | 2.1.3 | legacy_allow_tf32 |
| NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | 2.5.2 | fp32_precision |

驱动查询、其他后端设置和逐轮环境详见environment_summary.csv及原始environment.json。各设备内三轮的软件、GPU标识和其他数学设置已逐项核对；跨设备输入矩阵以SHA256核对。

## 采集状态与警告

九个正式轮次中，状态记录标记遥测查询失败的轮次为0，标记新外部计算进程的轮次为0，使用显式推断的宿主/容器PID关联的轮次为6。边界快照不等于连续监控，未标记新进程不等于证明无干扰。
RTX 5070正式三轮报告的背景库存条目数为[0, 0, 0]；保留的零计时前置查询观测到32个背景进程，包括WallpaperEngine。两次执行上下文的查询可见性并不一致，不能用正式库存为空证明后台进程已停止或未使用GPU，RTX 5070结果须附桌面干扰限制。
- A100 round 1: NVML host PID 482693 is operationally associated with container Python PID 1755 from an empty baseline and sole post-context process. This is an explicitly inferred namespace association, not a kernel-proven PID mapping.
- A100 round 2: NVML host PID 483139 is operationally associated with container Python PID 1853 from an empty baseline and sole post-context process. This is an explicitly inferred namespace association, not a kernel-proven PID mapping.
- A100 round 3: NVML host PID 483709 is operationally associated with container Python PID 1947 from an empty baseline and sole post-context process. This is an explicitly inferred namespace association, not a kernel-proven PID mapping.
- T4 round 1: NVML host PID 810484 is operationally associated with container Python PID 1886 from an empty baseline and sole post-context process. This is an explicitly inferred namespace association, not a kernel-proven PID mapping.
- T4 round 2: NVML host PID 811462 is operationally associated with container Python PID 1969 from an empty baseline and sole post-context process. This is an explicitly inferred namespace association, not a kernel-proven PID mapping.
- T4 round 3: NVML host PID 812752 is operationally associated with container Python PID 2073 from an empty baseline and sole post-context process. This is an explicitly inferred namespace association, not a kernel-proven PID mapping.
- RTX5070 was measured in the actual Windows desktop environment. A preserved zero-timing preformal query observed 32 background GPU process entries, including WallpaperEngine if listed in validation.windows_desktop_context; formal round inventory counts are [0, 0, 0]. Empty formal inventories do not establish an interference-free GPU or absence of desktop activity.