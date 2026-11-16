# 图5 数据来源与图注

中文图注：图5 风险上界的覆盖—假安全权衡。（a）完整风险与移除上界配置的上界覆盖率；（b）相应假安全率。数据来自风险组件对照，135个场景、每场景20个种子（601—620）；先在场景内对种子取均值，再对场景等权平均。覆盖率仅对存在严格QoS任务的运行有定义，12个无严格任务的场景—种子单元不参与覆盖率场景内平均；假安全率以预测安全任务为条件，保留全部2700单元。条形为描述性均值。移除上界后，MAPE由12.281644%变为12.280687%，SLA违约率由19.621786%变为19.617898%，总体近乎不变，不表示严格相等。风险识别改善不等同于实际SLA收益或概率保证。

English caption: Figure 5. Coverage–false-safe tradeoff of the risk upper bound. (a) Upper-bound coverage and (b) false-safe rate for the full-risk and no-upper-bound configurations. The risk-component ablation covers 135 scenarios with 20 seeds per scenario (601–620). Seeds are averaged within each scenario, followed by equal weighting across scenarios. Coverage is defined only for runs containing strict-QoS tasks; 12 scenario–seed cells without such tasks are excluded from within-scenario coverage means. The false-safe rate is conditional on predicted-safe tasks and retains all 2,700 cells. Bars show descriptive means. Removing the upper bound changes MAPE from 12.281644% to 12.280687% and the SLA violation rate from 19.621786% to 19.617898%; both overall means remain nearly unchanged, rather than exactly equal. Better risk identification does not imply an observed SLA benefit or a probabilistic guarantee.

插入位置：第5.4节“风险组件对照”说明之后。中文/英文版数据和布局相同。PNG为600dpi，PDF/SVG为矢量格式。

原始数据：`<LOCAL_WORKDIR>/修补实验2_deadline重设_20260907/experiment_C2_replay_deadline_fixed/per_seed_results.csv`。
程序：`generate_figure5.py`。数据：`figure5_plot_data.csv`；各策略场景均值及完整数值核查见同目录CSV/JSON。
