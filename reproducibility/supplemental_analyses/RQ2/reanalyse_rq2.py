"""Post hoc paired RQ2 analysis; reads existing block summaries only.

Usage: python reanalyse_rq2.py --prepare-plan
       python reanalyse_rq2.py --analyse
No simulation and no mutation of the original evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(r"$WORKSPACE")
HERE = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "修改补充实验" / "heterogeneity_adaptive_cpu_20261009"
SOURCES = {
    "fixed": SOURCE_ROOT / "03_independent_prediction_test" / "image_block_metrics.csv",
    "closed": SOURCE_ROOT / "04_closed_loop_scheduling" / "full" / "analysis" / "image_block_metrics.csv",
}
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
FLOWS = ["balanced", "late_sparse"]
LOADS = [1.25, 1.7]
SEED = 20261013
N_BOOT = 5000
N_SIGN = 100000
PLAN = HERE / "analysis_plan.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def specs():
    result = []
    for rho in RHOS:
        for flow in FLOWS:
            result.append({"id": f"fixed_r{rho:g}_{flow}", "analysis": "fixed_direct", "rho": rho, "condition": flow})
    for rho in RHOS:
        for load in LOADS:
            result.append({"id": f"closed_r{rho:g}_load{load:g}", "analysis": "closed_direct", "rho": rho, "condition": load})
    for rho in RHOS:
        result.append({"id": f"fixed_interaction_r{rho:g}", "analysis": "fixed_feedback_interaction", "rho": rho, "condition": "late_sparse_minus_balanced"})
    for spec in result:
        spec["bootstrap_seed"] = int.from_bytes(hashlib.sha256(f"{SEED}|{spec['id']}|bootstrap".encode()).digest()[:8], "little")
        spec["signflip_seed"] = int.from_bytes(hashlib.sha256(f"{SEED}|{spec['id']}|signflip".encode()).digest()[:8], "little")
    return result


def prepare_plan():
    HERE.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        raise RuntimeError("Plan already exists; preserve its original timestamp instead of overwriting it.")
    plan = {
        "status": "post_hoc_supplementary_analysis",
        "not_original_preregistration": True,
        "prepared_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "base_seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_method": "paired image-block percentile bootstrap, numpy quantile(method=linear), marginal 95% intervals",
        "signflip_replicates": N_SIGN,
        "signflip_method": "two-sided absolute mean statistic; independently flip each of 20 image differences; p=(extreme_count+1)/(100000+1)",
        "multiple_testing": "Holm step-down over all 25 comparisons, alpha=0.05",
        "estimand": "equal-weight mean across 20 images of PoolEMA minus LocalWindow24; 5 task seeds already averaged within each image in source CSV",
        "interaction_estimand": "for a matched image and rho: (Pool-Window in late_sparse) minus (Pool-Window in balanced)",
        "units": {"fixed_direct": "MAPE percentage points", "closed_direct": "SLA violation percentage points", "fixed_feedback_interaction": "MAPE percentage points"},
        "assumptions": "image is independent resampling/inference block; within-image outcomes remain paired; sign-flip inference uses symmetry/exchangeability of paired differences under null and is not a randomized-mechanism causal test",
        "source_hashes": {key: {"path": str(path), "sha256": digest(path)} for key, path in SOURCES.items()},
        "script_sha256": digest(Path(__file__)),
        "source_filter": {"fixed": "drift_type == none", "closed": "supplied full analysis no-drift holdout"},
        "comparison_modes": ["pool_ema", "local_window"],
        "comparison_specs": specs(),
        "zero_definition": "abs(paired difference) <= 1e-12 in percentage-point units",
        "scope": "read-only reanalysis of already completed tests; no new simulation; preserve original 30-comparison adaptive primary families separately",
    }
    dump(PLAN, plan)
    print(json.dumps({"plan": str(PLAN), "n_comparisons": len(plan["comparison_specs"]), "source_sha256": plan["source_hashes"]}, ensure_ascii=False))


def load_source(key):
    with SOURCES[key].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {}
    for row in rows:
        if row["mode"] not in ("pool_ema", "local_window"):
            continue
        if key == "fixed" and row["drift_type"] != "none":
            continue
        image = int(row["image_seed"])
        rho = float(row["rho"])
        condition = row["flow_kind"] if key == "fixed" else float(row["load"])
        idx = (rho, condition, row["mode"], image)
        if idx in index:
            raise AssertionError(f"Duplicate source row {key}: {idx}")
        expected_count = "n_task_streams" if key == "fixed" else "n_streams"
        assert int(row[expected_count]) == 5, (key, idx, row[expected_count])
        metric = "overall_macro_mape" if key == "fixed" else "sla_violation_rate"
        value = float(row[metric]) * (1.0 if key == "fixed" else 100.0)
        assert math.isfinite(value), (key, idx)
        index[idx] = value
    assert len(index) == 400, (key, len(index))
    return index


def paired(index, rho, condition):
    pool_images = sorted(key[3] for key in index if key[:3] == (rho, condition, "pool_ema"))
    window_images = sorted(key[3] for key in index if key[:3] == (rho, condition, "local_window"))
    assert pool_images == window_images == list(range(13001, 13021)), (rho, condition, pool_images, window_images)
    pool = np.array([index[(rho, condition, "pool_ema", image)] for image in pool_images])
    window = np.array([index[(rho, condition, "local_window", image)] for image in pool_images])
    return pool_images, pool, window, pool - window


def infer(d, spec):
    rng_boot = np.random.default_rng(spec["bootstrap_seed"])
    indices = rng_boot.integers(0, len(d), size=(N_BOOT, len(d)))
    boot_means = np.mean(d[indices], axis=1)
    ci = np.quantile(boot_means, [0.025, 0.975], method="linear")
    rng_sign = np.random.default_rng(spec["signflip_seed"])
    signs = 2 * rng_sign.integers(0, 2, size=(N_SIGN, len(d)), dtype=np.int8) - 1
    null_means = np.mean(signs * d, axis=1)
    extreme = int(np.sum(np.abs(null_means) >= abs(float(np.mean(d))) - 1e-12))
    return {
        "mean_difference_pp": float(np.mean(d)),
        "ci95_low_pp": float(ci[0]),
        "ci95_high_pp": float(ci[1]),
        "p_raw": (extreme + 1) / (N_SIGN + 1),
        "signflip_extreme_count": extreme,
        "n_negative": int(np.sum(d < -1e-12)),
        "n_positive": int(np.sum(d > 1e-12)),
        "n_zero": int(np.sum(np.abs(d) <= 1e-12)),
        "n_images": len(d),
        "min_difference_pp": float(np.min(d)),
        "max_difference_pp": float(np.max(d)),
        "sd_difference_pp": float(np.std(d, ddof=1)),
    }


def holm(p):
    order = sorted(range(len(p)), key=lambda i: (p[i], i))
    adjusted = [0.0] * len(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def write_csv(path, records):
    fields = list(records[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def analyse():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["comparison_specs"] == specs()
    assert plan["script_sha256"] == digest(Path(__file__)), "Implementation changed after saving plan; investigate instead of silently changing it."
    for key, path in SOURCES.items():
        assert digest(path) == plan["source_hashes"][key]["sha256"], f"Source {key} changed after plan."
    sources = {key: load_source(key) for key in SOURCES}
    output, block_records = [], []
    for spec in plan["comparison_specs"]:
        if spec["analysis"] == "fixed_feedback_interaction":
            images, _, _, balanced = paired(sources["fixed"], spec["rho"], "balanced")
            images2, _, _, sparse = paired(sources["fixed"], spec["rho"], "late_sparse")
            assert images == images2
            differences = sparse - balanced
            pool_mean = window_mean = ""
        else:
            key = "fixed" if spec["analysis"] == "fixed_direct" else "closed"
            images, pool, window, differences = paired(sources[key], spec["rho"], spec["condition"])
            pool_mean, window_mean = float(np.mean(pool)), float(np.mean(window))
        record = {**spec, "pool_mean": pool_mean, "window_mean": window_mean, **infer(differences, spec)}
        output.append(record)
        for image, diff in zip(images, differences):
            block_records.append({"comparison_id": spec["id"], "image_seed": image, "difference_pp": float(diff)})
    for record, p_adj in zip(output, holm([row["p_raw"] for row in output])):
        record["p_holm_25"] = p_adj
        record["reject_holm_0_05"] = p_adj <= 0.05
    write_csv(HERE / "rq2_supplementary_comparisons_25.csv", output)
    write_csv(HERE / "rq2_paired_image_differences.csv", block_records)
    dump(HERE / "rq2_supplementary_comparisons_25.json", output)
    metadata = {"completed_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "plan_sha256": digest(PLAN), "script_sha256": digest(Path(__file__)), "numpy": np.__version__, "n_comparisons": len(output), "n_block_differences": len(block_records)}
    dump(HERE / "analysis_completion.json", metadata)
    print(json.dumps({"completion": metadata, "results": output}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-plan", action="store_true")
    group.add_argument("--analyse", action="store_true")
    args = parser.parse_args()
    if args.prepare_plan:
        prepare_plan()
    else:
        analyse()
