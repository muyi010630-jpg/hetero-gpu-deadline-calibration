图4 共享失配比例与PoolEMA相对LocalWindow24的表现差异
（a）共同反馈宏MAPE差；（b）闭环SLA差。差值均为PoolEMA−LocalWindow24，正值表示池化误差或违约更多。每点为20幅画像等权均值，画像内先平均5任务种子；阴影为5000次画像块bootstrap的边际95%区间。实心点表示25项补充比较统一Holm校正后p<0.05，空心点表示未通过。连线仅辅助阅读。

Figure 4 Differences between PoolEMA and LocalWindow24 across controlled shared-mismatch proportions
(a) Common-feedback macro-MAPE differences. (b) Closed-loop SLA violation differences. All differences are PoolEMA minus LocalWindow24; positive values favor LocalWindow24. Points average 20 image blocks after averaging five task seeds within each block. Shading shows marginal 95% intervals from 5000 paired image-block bootstrap replicates. Filled markers indicate Holm-adjusted p<0.05 within the 25-comparison supplemental family; hollow markers do not imply equivalence. Lines connect tested proportions for readability only.

图6 三主机闭环的种子内配对差
（a）MAPE；（b）覆盖率；（c）GPU预算违约率；（d）P99完成时延。A、B分别为宽松和紧张配置；差值均为Full-Deploy减对应基线。圆点和三角形表示部署批次1和2，菱形与横线表示合并均值和边际95%区间。端到端违约及校正检验见正文表13。

Figure 6 Within-seed paired differences in the three-host deployment
(a) MAPE. (b) Bound coverage. (c) GPU-budget violations. (d) P99 completion latency. A and B denote relaxed and tight configurations, respectively. Differences are Full-Deploy minus the indicated baseline. Circles and triangles identify deployment batches 1 and 2; diamonds and horizontal lines show pooled means and marginal 95% intervals. End-to-end violations and multiplicity-adjusted tests are reported in Table 13.
