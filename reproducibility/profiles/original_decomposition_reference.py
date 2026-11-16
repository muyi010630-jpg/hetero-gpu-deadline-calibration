# -*- coding: utf-8 -*-
"""
真实系统失配的方差结构分解 —— 回答"仿真的族级共享偏差设定是否真实"
数据源：
  A) canonical 离线标定画像  EMA/性能画像实验/00_统一汇总/profile_three_gpu.csv
  B) 部署环境运行时画像      实验补充_20260921/07_N6_real_online/diagnostics/profile_runtime_20260921/profile_current_runtime.csv
两文件均为真机重复测量（true_s = 重复测量中位数）。
纯标准库实现（本机无 numpy/pandas）。
"""
import csv, math, statistics as st

CANON = r"reproducibility/deployment/profiles/P1.csv"
RUNT  = r"reproducibility/deployment/profiles/P2.csv"
KEY = ("device", "family", "size", "batch", "dtype")

def load(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            k = tuple(row[c].strip() for c in KEY)
            out[k] = {"true_s": float(row["true_s"]),
                      "initial_s": float(row["initial_s"])}
    return out

def anova(records):
    """records: list of (device, family, value). 返回各层平方和占比。"""
    n = len(records)
    grand = sum(v for _, _, v in records) / n
    sst = sum((v - grand) ** 2 for _, _, v in records)
    devices = sorted({d for d, _, _ in records})
    fams = sorted({f for _, f, _ in records})
    # device 主效应
    dev_mean = {d: sum(v for dd, _, v in records if dd == d) / sum(1 for dd, _, _ in records if dd == d) for d in devices}
    fam_mean = {f: sum(v for _, ff, v in records if ff == f) / sum(1 for _, ff, _ in records if ff == f) for f in fams}
    ss_dev = sum(len([1 for dd, _, _ in records if dd == d]) * (dev_mean[d] - grand) ** 2 for d in devices)
    ss_fam = sum(len([1 for _, ff, _ in records if ff == f]) * (fam_mean[f] - grand) ** 2 for f in fams)
    # interaction (device x family cell)
    ss_int = 0.0
    ss_res = 0.0
    for d in devices:
        for f in fams:
            vals = [v for dd, ff, v in records if dd == d and ff == f]
            if not vals:
                continue
            cm = sum(vals) / len(vals)
            ss_int += len(vals) * (cm - dev_mean[d] - fam_mean[f] + grand) ** 2
            ss_res += sum((v - cm) ** 2 for v in vals)
    return dict(n=n, sst=sst, ss_dev=ss_dev, ss_fam=ss_fam, ss_int=ss_int, ss_res=ss_res,
                devices=devices, fams=fams)

def pct(a):
    s = a["sst"] if a["sst"] > 0 else 1
    return [a["ss_dev"] / s * 100, a["ss_fam"] / s * 100, a["ss_int"] / s * 100, a["ss_res"] / s * 100]

canon, runt = load(CANON), load(RUNT)
keys = sorted(set(canon) & set(runt))
print("common contexts:", len(keys))

# ---------- A. 真实环境迁移偏差 r = ln(runtime_true / canonical_true) ----------
recs_env = []
for k in keys:
    if canon[k]["true_s"] > 0 and runt[k]["true_s"] > 0:
        recs_env.append((k[0], k[1], math.log(runt[k]["true_s"] / canon[k]["true_s"])))
a_env = anova(recs_env)
p = pct(a_env)
print("\n=== [A] 真实环境迁移偏差 r_env = ln(运行时true / 离线true) ===")
print("n =", a_env["n"], " SST =", round(a_env["sst"], 6))
print("  device       %6.2f%%" % p[0])
print("  family       %6.2f%%" % p[1])
print("  device*fa    %6.2f%%" % p[2])
print("  单元内(上下文) %6.2f%%" % p[3])
print("  族级共享成分(device+family+dev*fa) = %.2f%%" % (p[0] + p[1] + p[2]))

# 量级与簇内散布
lo_env = [math.log(runt[k]["true_s"] / canon[k]["true_s"]) for k in keys]
absr = sorted(abs(x) for x in lo_env)
print("\n  |r_env| 中位 %.4f (%.2f%%)  均值 %.4f  max %.4f (%.2f%%)" %
      (st.median(absr), (math.exp(st.median(absr)) - 1) * 100,
       sum(absr) / len(absr), absr[-1], (math.exp(absr[-1]) - 1) * 100))
print("  单元内 SD（同 设备×族 内跨上下文）:")
for d in sorted({k[0] for k in keys}):
    for f in sorted({k[1] for k in keys}):
        vs = [math.log(runt[k]["true_s"] / canon[k]["true_s"]) for k in keys if k[0] == d and k[1] == f]
        if len(vs) >= 2:
            print("    %-9s %-10s n=%d  sd=%.4f" % (d, f, len(vs), st.pstdev(vs)))

# 仿真注入偏差的量级（供对照）
print("\n  仿真注入 |ln bias| 取值: RTX5070 0.199~0.328 / T4 0.105~0.199 / A100 0.0953")
print("\n  分设备平均迁移 (%):")
for d in sorted({k[0] for k in keys}):
    vs = [runt[k]["true_s"] / canon[k]["true_s"] - 1 for k in keys if k[0] == d]
    print("    %-9s mean %+7.2f%%  min %+7.2f%%  max %+7.2f%%  n=%d"
          % (d, sum(vs) / len(vs) * 100, min(vs) * 100, max(vs) * 100, len(vs)))

# ---------- B. 仿真画像偏差：initial_s = true_s * bias(device,family) ----------
recs_sim = []
for k in keys:
    if runt[k]["true_s"] > 0 and runt[k]["initial_s"] > 0:
        recs_sim.append((k[0], k[1], math.log(runt[k]["initial_s"] / runt[k]["true_s"])))
a_sim = anova(recs_sim)
q = pct(a_sim)
print("\n=== [B] 仿真画像偏差 r_sim = ln(initial_s / true_s)（=论文注入的失配）===")
print("  device       %6.2f%%" % q[0])
print("  family       %6.2f%%" % q[1])
print("  device*fa    %6.2f%%" % q[2])
print("  单元内(上下文) %6.2f%%" % q[3])

# ---------- C. 两者叠加后的结构 ----------
recs_all = [(k[0], k[1], math.log(runt[k]["initial_s"] / canon[k]["true_s"])) for k in keys]
a_all = anova(recs_all)
r = pct(a_all)
print("\n=== [C] 端到端 r = ln(initial_s / 离线true)（静态模型相对部署环境的真实总误差）===")
print("  device       %6.2f%%" % r[0])
print("  family       %6.2f%%" % r[1])
print("  device*fa    %6.2f%%" % r[2])
print("  单元内(上下文) %6.2f%%" % r[3])
print("  族级共享成分 = %.2f%%" % (r[0] + r[1] + r[2]))

# 量级
lo = [math.log(runt[k]["initial_s"] / canon[k]["true_s"]) for k in keys]
print("\n  |r| 中位 %.4f (%.2f%%)  max %.4f (%.2f%%)" %
      (st.median([abs(x) for x in lo]),
       (math.exp(st.median([abs(x) for x in lo])) - 1) * 100,
       max(abs(x) for x in lo),
       (math.exp(max(abs(x) for x in lo)) - 1) * 100))
