"""Offline descriptive analysis of the prespecified three-GPU TF32 diagnostic.

This program does not import torch, contact a server, or launch GPU work. It
consumes measurement CSVs, validates the complete design, and reports paired
descriptive ratios. There are no p values or confidence intervals; twenty
within-phase windows are never treated as independent hardware replicates.

Run with the bundled CPU Python, for example:
  python analyze_tf32.py --input-root ../raw --output-dir ../analysis
Explicit --timing-file / --difference-file flags can be repeated when names
do not follow the measurement script's defaults.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
import sys

HERE = Path(__file__).resolve().parent

def portable_path(value):
    """Record package-relative sources and output basenames without machine paths."""
    p = Path(value).resolve()
    try:
        return p.relative_to(HERE.parent).as_posix()
    except ValueError:
        return p.name

DEVICES = ("A100", "T4", "RTX5070")
CONTEXTS = (("float32", 2048), ("float32", 4096), ("float32", 6144), ("float16", 4096))
ROUNDS = (1, 2, 3)
PHASES = (1, 2, 3)
MODE_PLAN = {1: ("ieee", "allow_tf32", "ieee"),
             2: ("allow_tf32", "ieee", "allow_tf32"),
             3: ("ieee", "allow_tf32", "ieee")}
FROZEN_SOURCES = {
    "A100": {"directory": HERE, "script": "d43d4e04e8df995ebf97e2b87e8aac38c7e0076f4f9c22d346d24ee6252d3c58",
             "protocol": "9bddfb17ad534779fe013bb5ef8a82a237015e3db7ca344d25734ed86bfb7935"},
    "T4": {"directory": HERE, "script": "d43d4e04e8df995ebf97e2b87e8aac38c7e0076f4f9c22d346d24ee6252d3c58",
           "protocol": "9bddfb17ad534779fe013bb5ef8a82a237015e3db7ca344d25734ed86bfb7935"},
    "RTX5070": {"directory": HERE / "windows_v4", "script": "37d149180c23cdaad20d5ea59c09caee103b52a6b227311bf9e027977970c8c3",
                "protocol": "39fec9e023c11ce1a5386bdc857a8a97ec8e4ba1964479294ceb89ec6e2ff63a"},
}


def normalized_uuid(value):
    value = str(value).lower().strip()
    return value[4:] if value.startswith("gpu-") else value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{**r, "source_file": portable_path(path), "source_csv_row": i + 2}
                for i, r in enumerate(csv.DictReader(f))]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def field(row: dict, *keys: str, required: bool = True, default=None):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return row[k]
    if required:
        raise ValueError(f"Missing field {keys}; file={row.get('source_file')}, row={row.get('source_csv_row')}")
    return default


def yes(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok", "success", "succeeded", "passed"}


def device_name(value: str) -> str:
    key = str(value).upper().replace(" ", "").replace("-", "").replace("_", "")
    for dev in DEVICES:
        if dev in key:
            return dev
    raise ValueError(f"Unknown device: {value!r}")


def dtype_name(value: str) -> str:
    key = str(value).lower().replace("torch.", "")
    if key in {"fp32", "float32", "float"}:
        return "float32"
    if key in {"fp16", "float16", "half"}:
        return "float16"
    raise ValueError(f"Unknown dtype: {value!r}")


def mode_name(value: str) -> str:
    key = str(value).lower().strip()
    if key in {"ieee", "off", "false", "0", "forbid_tf32", "tf32_off"}:
        return "ieee"
    if key in {"allow_tf32", "tf32", "on", "true", "1", "tf32_on"}:
        return "allow_tf32"
    raise ValueError(f"Unknown mode: {value!r}")


def base_record(row: dict) -> dict:
    ctx = field(row, "context", required=False)
    contexts = {"fp32_n2048": ("float32", 2048), "fp32_n4096": ("float32", 4096),
                "fp32_n6144": ("float32", 6144), "fp16_n4096": ("float16", 4096)}
    fallback_dtype, fallback_n = contexts.get(ctx, (None, None))
    return {"device": device_name(field(row, "device", "device_label", "device_tag", "gpu_label", "gpu")),
            "round": int(field(row, "round", "round_id", "round_index")),
            "dtype": dtype_name(field(row, "dtype", "input_dtype", required=fallback_dtype is None, default=fallback_dtype)),
            "n": int(field(row, "n", "N", "matrix_n", "matrix_size", "size", required=fallback_n is None, default=fallback_n)),
            "seed": int(field(row, "seed", "input_seed"))}


def normalize_timing(row: dict) -> dict:
    out = base_record(row)
    out.update({"phase": int(field(row, "phase", "phase_id", "stage", "stage_id")),
                "mode": mode_name(field(row, "mode", "math_mode", "tf32_mode")),
                "window": int(field(row, "window", "window_id", "window_index")),
                "calls_per_window": int(field(row, "calls_per_window", "window_calls", "calls", "inner_iters", "calls_per_timing_window")),
                "success": yes(field(row, "success", "status"))})
    per = field(row, "per_gemm_ms", "gemm_ms", "gpu_time_ms_per_gemm", "avg_ms", "gpu_ms_per_call", "gpu_per_call_ms", "gpu_time_ms", required=False)
    total = field(row, "window_ms", "window_gpu_ms", "total_gpu_ms", "elapsed_ms", "gpu_window_ms", required=False)
    if per is None and total is None:
        raise ValueError(f"No timing field found in {row['source_file']}")
    out["per_gemm_ms"] = float(per) if per is not None else float(total) / out["calls_per_window"]
    out["window_ms"] = float(total) if total is not None else out["per_gemm_ms"] * out["calls_per_window"]
    out["_raw"] = row
    return out


def normalize_difference(row: dict) -> dict:
    out = base_record(row)
    # The measurement script explicitly scans finite output elements before the
    # CPU float64 calculation. Its valid field jointly certifies both checks.
    arithmetic = str(field(row, "arithmetic", required=False, default=""))
    valid = field(row, "valid", required=False)
    out.update({"delta": float(field(row, "delta", "relative_frobenius_difference", "relative_frobenius_delta", "relative_frobenius")),
                "off_finite": yes(field(row, "off_finite", "C_off_finite", "c_off_finite", "finite_off", required=valid is None, default=valid)),
                "on_finite": yes(field(row, "on_finite", "C_on_finite", "c_on_finite", "finite_on", required=valid is None, default=valid)),
                "reference_norm": float(field(row, "reference_norm", "off_frobenius_norm", "norm_off", "C_off_norm", "c_off_norm", "off_frobenius")),
                "difference_norm": float(field(row, "difference_norm", "difference_frobenius_norm", "norm_diff", "diff_norm", "difference_frobenius")),
                "computation_device": str(field(row, "computation_device", "difference_device", "analysis_device", "delta_device", required=not arithmetic, default="cpu" if "CPU" in arithmetic else "unknown")),
                "computation_dtype": str(field(row, "computation_dtype", "difference_dtype", "analysis_dtype", "delta_dtype", required=not arithmetic, default="float64" if "float64 before subtraction" in arithmetic else "unknown")),
                "success": yes(field(row, "success", "status", "valid")), "_raw": row})
    return out


def discover(root: Path, explicit: list[str] | None, names: tuple[str, ...], output: Path) -> list[Path]:
    if explicit:
        paths = sorted({Path(p).resolve() for p in explicit})
        if any(p.parent.name not in {"round_01", "round_02", "round_03"} for p in paths):
            raise ValueError("Only round_01..round_03 formal CSV directories may be analyzed; smoke/input_check are excluded.")
        return paths
    return sorted({p.resolve() for name in names for p in root.rglob(name)
                   if output.resolve() not in p.resolve().parents
                   and p.parent.name in {"round_01", "round_02", "round_03"}
                   and not any(part.startswith(("smoke_", "input_check_", "preformal_")) for part in p.parts)})


def validate(timings: list[dict], differences: list[dict]) -> dict:
    errors, warnings = [], []
    check_items = []

    def check(name: str, passed: bool, detail=None):
        check_items.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(name)

    expected_keys = set(itertools.product(DEVICES, ROUNDS, PHASES, CONTEXTS))
    check("2160 timing windows", len(timings) == 2160, len(timings))
    count_by_device = dict(Counter(t["device"] for t in timings))
    check("720 timing windows per device", count_by_device == {d: 720 for d in DEVICES}, count_by_device)
    groups = defaultdict(list)
    for t in timings:
        groups[(t["device"], t["round"], t["phase"], (t["dtype"], t["n"]))].append(t)
    check("All 108 expected device-round-phase-context cells", set(groups) == expected_keys,
          {"missing": [str(k) for k in sorted(expected_keys - set(groups))],
           "unexpected": [str(k) for k in sorted(set(groups) - expected_keys)]})
    window_keys = [(t["device"], t["round"], t["phase"], t["dtype"], t["n"], t["window"]) for t in timings]
    check("No duplicate timing windows", len(window_keys) == len(set(window_keys)))
    invalid_windows, bad_modes, bad_seeds, bad_times = [], [], [], []
    for key, rows in groups.items():
        idx = sorted(r["window"] for r in rows)
        if idx not in [list(range(20)), list(range(1, 21))]:
            invalid_windows.append(str(key))
        if any(r["mode"] != MODE_PLAN.get(r["round"], (None,) * 3)[r["phase"] - 1] for r in rows if 1 <= r["phase"] <= 3):
            bad_modes.append(str(key))
        if len({r["seed"] for r in rows}) != 1:
            bad_seeds.append(str(key))
        for r in rows:
            if not (math.isfinite(r["per_gemm_ms"]) and r["per_gemm_ms"] > 0 and
                    math.isfinite(r["window_ms"]) and r["window_ms"] > 0 and
                    math.isclose(r["window_ms"], r["per_gemm_ms"] * r["calls_per_window"], rel_tol=1e-6, abs_tol=1e-8)):
                bad_times.append(str(key))
    check("20 unique contiguous windows per cell", not invalid_windows, invalid_windows)
    check("Prespecified ABA/BAB/ABA modes", not bad_modes, bad_modes)
    check("Fixed input seed within each cell", not bad_seeds, bad_seeds)
    check("10 GEMM calls in every timing window", all(t["calls_per_window"] == 10 for t in timings))
    check("Finite positive consistent event timings", not bad_times, bad_times)
    check("No failed timing windows", all(t["success"] for t in timings))
    round_seeds = defaultdict(set)
    for t in timings:
        round_seeds[t["round"]].add(t["seed"])
    check("Same seed across stages, contexts and devices within each round",
          all(len(v) == 1 for v in round_seeds.values()), {str(k): sorted(v) for k, v in round_seeds.items()})

    dkeys = [(d["device"], d["round"], (d["dtype"], d["n"])) for d in differences]
    expected_dkeys = set(itertools.product(DEVICES, ROUNDS, CONTEXTS))
    check("36 output-difference records", len(differences) == 36, len(differences))
    check("No missing or duplicate output-difference records", len(dkeys) == len(set(dkeys)) and set(dkeys) == expected_dkeys,
          {"missing": [str(k) for k in sorted(expected_dkeys - set(dkeys))]})
    check("All paired outputs finite", all(d["off_finite"] and d["on_finite"] for d in differences))
    check("All differences calculated on CPU float64",
          all(d["computation_device"].lower() == "cpu" and dtype_name_for_double(d["computation_dtype"]) for d in differences))
    check("Output-difference records succeeded", all(d["success"] for d in differences))
    check("Finite nonzero reference and consistent delta",
          all(math.isfinite(d["reference_norm"]) and d["reference_norm"] > 0 and math.isfinite(d["difference_norm"]) and d["difference_norm"] >= 0
              and math.isfinite(d["delta"]) and d["delta"] >= 0
              and math.isclose(d["delta"], d["difference_norm"] / d["reference_norm"], rel_tol=1e-8, abs_tol=1e-15)
              for d in differences))
    check("Output-difference seeds match timing inputs",
          all(d["seed"] in round_seeds.get(d["round"], set()) for d in differences))
    return {"complete_design_passed": not errors, "errors": errors, "warnings": warnings,
            "checks": check_items, "n_timing_records": len(timings), "n_difference_records": len(differences),
            "measurement_unit": "one CUDA Event window containing 10 GEMM calls; timing divided by 10",
            "inference": "descriptive only; no significance tests, no confidence intervals",
            "round_sequences": MODE_PLAN}


def dtype_name_for_double(value: str) -> bool:
    return value.lower().replace("torch.", "").replace("numpy.", "") in {"float64", "double"}


def quantile_sorted(x: list[float], q: float) -> float:
    x = sorted(x)
    pos = (len(x) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[lo] + (x[hi] - x[lo]) * (pos - lo)


def summaries(timings: list[dict], differences: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groups = defaultdict(list)
    for t in timings:
        groups[(t["device"], t["round"], t["phase"], t["dtype"], t["n"])].append(t)
    stage_rows, stage_index = [], {}
    for key in sorted(groups):
        data = groups[key]
        device, rnd, phase, dtype, n = key
        values = [r["per_gemm_ms"] for r in data]
        rec = {"device": device, "round": rnd, "phase": phase, "dtype": dtype, "n": n,
               "seed": data[0]["seed"], "mode": data[0]["mode"], "n_windows": len(values),
               "median_ms": median(values), "mean_ms": mean(values),
               "min_ms": min(values), "max_ms": max(values),
               "q25_ms": quantile_sorted(values, .25), "q75_ms": quantile_sorted(values, .75),
               "window_cv_percent": stdev(values) / mean(values) * 100}
        stage_rows.append(rec)
        stage_index[key] = rec
    diff_idx = {(d["device"], d["round"], d["dtype"], d["n"]): d for d in differences}
    round_rows = []
    for device, (dtype, n), rnd in itertools.product(DEVICES, CONTEXTS, ROUNDS):
        stages = [stage_index[(device, rnd, phase, dtype, n)] for phase in PHASES]
        off = [s["median_ms"] for s in stages if s["mode"] == "ieee"]
        on = [s["median_ms"] for s in stages if s["mode"] == "allow_tf32"]
        t_off = math.exp(mean(math.log(v) for v in off))
        t_on = math.exp(mean(math.log(v) for v in on))
        ratio = t_off / t_on
        recovery = stages[2]["median_ms"] / stages[0]["median_ms"]
        diff = diff_idx[(device, rnd, dtype, n)]
        round_rows.append({"device": device, "dtype": dtype, "n": n, "round": rnd, "seed": stages[0]["seed"],
                           "sequence": "ABA" if rnd in (1, 3) else "BAB", "A": "ieee", "B": "allow_tf32",
                           "stage1_ms": stages[0]["median_ms"], "stage2_ms": stages[1]["median_ms"], "stage3_ms": stages[2]["median_ms"],
                           "t_off_ms": t_off, "t_on_ms": t_on, "R_off_over_on": ratio,
                           "recovery_T3_over_T1": recovery,
                           "recovery_abs_log": abs(math.log(recovery)), "switch_abs_log": abs(math.log(ratio)),
                           "recovery_to_switch_abs_log": abs(math.log(recovery)) / abs(math.log(ratio)) if ratio != 1 else "undefined_R_equals_1",
                           "underestimate_fraction_if_on_profile_and_R_gt_1": 1 - 1 / ratio if ratio > 1 else "not_applicable",
                           "delta_CPU_float64": diff["delta"], "C_off_finite": diff["off_finite"], "C_on_finite": diff["on_finite"]})
    summary_rows = []
    for device, (dtype, n) in itertools.product(DEVICES, CONTEXTS):
        selected = [r for r in round_rows if (r["device"], r["dtype"], r["n"]) == (device, dtype, n)]
        rs = [r["R_off_over_on"] for r in selected]
        recs = [r["recovery_T3_over_T1"] for r in selected]
        deltas = [r["delta_CPU_float64"] for r in selected]
        summary_rows.append({"device": device, "dtype": dtype, "n": n, "n_rounds": 3,
                             **{f"R_round{i}": rs[i - 1] for i in ROUNDS},
                             "R_median": median(rs), "R_min": min(rs), "R_max": max(rs),
                             "recovery_min": min(recs), "recovery_max": max(recs),
                             "max_abs_recovery_change_percent": max(abs(x - 1) * 100 for x in recs),
                             "delta_median": median(deltas), "delta_min": min(deltas), "delta_max": max(deltas),
                             "all_R_above_1": all(x > 1 for x in rs), "all_R_below_1": all(x < 1 for x in rs),
                             "n_rounds_recovery_abs_log_ge_switch_abs_log": sum(r["recovery_abs_log"] >= r["switch_abs_log"] for r in selected)})
    return stage_rows, round_rows, summary_rows


def audit_auxiliary(t_files: list[Path], timings: list[dict], differences: list[dict], stage_rows: list[dict],
                    round_rows: list[dict], validation: dict) -> tuple[list[Path], list[dict]]:
    """Independently compare raw windows with archived summaries and run metadata."""
    sources, environment_rows = [], []
    input_hashes = defaultdict(list)
    stage_index = {(r["device"], r["round"], r["phase"], r["dtype"], r["n"]): r for r in stage_rows}
    round_index = {(r["device"], r["round"], r["dtype"], r["n"]): r for r in round_rows}
    diff_index = {(r["device"], r["round"], r["dtype"], r["n"]): r for r in differences}
    env_by_device = defaultdict(list)
    protocol_hashes, script_hashes = defaultdict(set), defaultdict(set)
    readback_errors = []

    def check(name, passed, detail=None):
        validation["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            validation["errors"].append(name)

    def warning(message):
        validation["warnings"].append(message)

    run_dirs = sorted({p.parent for p in t_files})
    check("Nine formal round directories", len(run_dirs) == 9, len(run_dirs))
    for run_dir in run_dirs:
        label = str(run_dir)
        required = ["environment.json", "configuration.json", "status.json", "input_manifest.json",
                    "stages.csv", "round_summary.csv", "output_checks.csv", "telemetry.jsonl",
                    "process_ownership.json", "pre_cuda_baseline.json", "post_context_processes.json"]
        missing = [name for name in required if not (run_dir / name).is_file()]
        check(f"All auxiliary records present: {label}", not missing, missing)
        if missing:
            continue
        sources.extend(run_dir / name for name in required)
        env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        cfg = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        dev = device_name(env["device_label"])
        rnd = int(status["round"])
        check(f"Round completed with full counts and formal protocol: {label}",
              status["state"].startswith("completed") and not status["smoke"] and not cfg["smoke"]
              and status.get("input_check") is False
              and status["window_rows"] == 240 and status["completed_context_phases"] == 12
              and status["paired_output_differences"] == 4
              and cfg["windows_per_context_phase"] == 20 and cfg["calls_per_window"] == 10)
        if status.get("external_compute_process_seen"):
            warning(f"{dev} round {rnd}: external GPU compute process observed; retain data and qualify attribution.")
        if status.get("telemetry_query_failure_seen"):
            warning(f"{dev} round {rnd}: at least one telemetry query failed; external interference cannot be fully excluded.")
        ownership = json.loads((run_dir / "process_ownership.json").read_text(encoding="utf-8"))
        baseline = json.loads((run_dir / "pre_cuda_baseline.json").read_text(encoding="utf-8"))
        after = json.loads((run_dir / "post_context_processes.json").read_text(encoding="utf-8"))
        check(f"Successful fully parsed pre/post process queries: {label}",
              baseline["query"].get("returncode") == 0 and after["query"].get("returncode") == 0
              and not baseline["parse_errors"] and not after["parse_errors"])
        uuid = normalized_uuid(env["gpu_uuid"])
        selected_post = [r for r in after["processes"] if normalized_uuid(r["gpu_uuid"]) == uuid]
        selected_baseline = [r for r in baseline["processes"] if normalized_uuid(r["gpu_uuid"]) == uuid]
        owned = int(ownership["owned_nvml_pid"])
        python_pid = int(ownership["benchmark_pid"])
        check(f"Ownership UUID/fixed PID recorded: {label}", ownership["fixed_for_entire_round"] is True
              and normalized_uuid(ownership["selected_gpu_uuid"]) == uuid and python_pid == int(env["pid"]))
        background = set()
        if dev == "RTX5070":
            background = {int(r["pid"]) for r in selected_baseline}
            check(f"Windows exact Python PID ownership and frozen inventory: {label}",
                  not ownership["inferred"] and owned == python_pid
                  and {int(x) for x in ownership["background_inventory_pids"]} == background
                  and owned not in background
                  and sum(int(r["pid"]) == owned for r in selected_post) == 1
                  and all(int(r["pid"]) == owned or int(r["pid"]) in background for r in selected_post))
            if ownership.get("background_inventory_present"):
                names = sorted({r["process_name"].replace("\\", "/").rsplit("/", 1)[-1] for r in selected_baseline})
                warning(f"RTX5070 round {rnd}: Windows background GPU inventory contains {len(background)} preexisting PIDs, including {names}. Baseline presence does not establish graphics-only behavior or absence of GPU work; interference from these processes cannot be excluded.")
        else:
            check(f"Cloud empty-baseline/sole-post-context fixed PID ownership: {label}",
                  not baseline["processes"] and len(selected_post) == 1
                  and int(selected_post[0]["pid"]) == owned
                  and bool(ownership["inferred"]) == (owned != python_pid))
            if ownership["inferred"]:
                warning(f"{dev} round {rnd}: NVML host PID {owned} is operationally associated with container Python PID {python_pid} from an empty baseline and sole post-context process. This is an explicitly inferred namespace association, not a kernel-proven PID mapping.")
        check(f"No invalid output flag: {label}", not status.get("nonfinite_output_seen", False))
        protocol_hashes[dev].add(cfg["protocol_sha256"])
        script_hashes[dev].add(cfg["script_sha256"])
        plan = cfg["round_plan"]
        check(f"Recorded round seed and sequence follow protocol: {label}",
              int(plan["round"]) == rnd and int(plan["seed"]) == 2026101800 + rnd
              and tuple(plan["phases"]) == MODE_PLAN[rnd])
        check(f"No TF32 environment override: {label}",
              all(env["environment_variables"].get(k) in (None, "") for k in
                  ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")))
        env_by_device[dev].append(env)
        environment_rows.append({"device": dev, "round": rnd, "gpu_name": env["gpu_name"],
                                 "gpu_uuid": env.get("gpu_uuid"), "compute_capability": json.dumps(env["compute_capability"]),
                                 "torch": env["torch"], "torch_cuda_build": env["torch_cuda_build"],
                                 "numpy": env["numpy"], "python": env["python"], "platform": env["platform"],
                                 "precision_api": env["precision_api"],
                                 "other_matmul_settings": json.dumps(env["other_matmul_settings"], sort_keys=True),
                                 "driver_query": json.dumps(env["driver_query"], ensure_ascii=False),
                                 "external_compute_process_seen": status.get("external_compute_process_seen"),
                                 "telemetry_query_failure_seen": status.get("telemetry_query_failure_seen"),
                                 "ownership_inferred": ownership["inferred"], "benchmark_pid": python_pid,
                                 "owned_nvml_pid": owned, "background_inventory_count": len(background),
                                 "background_interference_limitation": ownership.get("background_interference_limitation", ""),
                                 "source_file": portable_path(run_dir / "environment.json")})
        for manifest in json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8")):
            input_hashes[(rnd, manifest["context"])].append((dev, manifest["a_sha256"], manifest["b_sha256"]))
            check(f"Contiguous input: {dev} {rnd} {manifest['context']}", manifest["contiguous"] is True)
            check(f"Frozen explicit PCG64 input generator: {dev} {rnd} {manifest['context']}",
                  manifest.get("generator") == "PCG64 raw uint64 high24 / 2**23 - 1")
        archived_stages = read_csv(run_dir / "stages.csv")
        check(f"Twelve stage summary rows: {label}", len(archived_stages) == 12)
        for row in archived_stages:
            b = base_record(row)
            key = (b["device"], b["round"], int(row["phase"]), b["dtype"], b["n"])
            rec = stage_index.get(key)
            check(f"Independent median/min/max reproduce archived stage {key}", rec is not None and
                  all(math.isclose(float(row[k]), rec[k], rel_tol=1e-12, abs_tol=1e-12) for k in ("median_ms", "min_ms", "max_ms")))
            check(f"Prespecified warmup completed {key}", int(row["warmup_calls"]) >= 20 and float(row["warmup_seconds"]) >= 3)
        archived_rounds = read_csv(run_dir / "round_summary.csv")
        check(f"Four archived round summary rows: {label}", len(archived_rounds) == 4)
        for row in archived_rounds:
            b = base_record(row)
            key = (b["device"], b["round"], b["dtype"], b["n"])
            rec = round_index.get(key)
            mapping = {"R_off_over_on": "R_off_over_on", "recovery_T3_over_T1": "recovery_T3_over_T1",
                       "T_off_ms": "t_off_ms", "T_on_ms": "t_on_ms", "delta": "delta_CPU_float64"}
            check(f"Independent R/recovery/delta reproduce archived round {key}", rec is not None and
                  all(math.isclose(float(row[k]), rec[v], rel_tol=1e-12, abs_tol=1e-12) for k, v in mapping.items()))
        checks = read_csv(run_dir / "output_checks.csv")
        check(f"Twelve finite full-size output records: {label}", len(checks) == 12 and
              all(yes(r["all_finite"]) and int(r["output_elements"]) == int(r["finite_elements"])
                  and int(r["output_elements"]) == base_record(r)["n"] ** 2 for r in checks))
        for row in checks:
            if int(row["phase"]) > 2:
                continue
            b = base_record(row)
            rec = diff_index[(b["device"], b["round"], b["dtype"], b["n"])]["_raw"]
            setting = "off" if mode_name(row["mode"]) == "ieee" else "on"
            check(f"Paired output identity matches difference record: {dev} {rnd} {row['context']} {setting}",
                  row["output_sha256"] == rec[f"{setting}_sha256"] and int(row["phase"]) == int(rec[f"{setting}_phase"]))
        snaps = [json.loads(line) for line in (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        check(f"Six boundary telemetry snapshots: {label}", len(snaps) == 6 and
              {(int(s["phase"]), s["position"]) for s in snaps} == set(itertools.product(PHASES, ("before", "after"))))
        for snap in snaps:
            check(f"Fixed NVML ownership retained: {label} phase{snap['phase']} {snap['position']}",
                  int(snap["owned_nvml_pid"]) == owned and int(snap["benchmark_pid"]) == python_pid
                  and normalized_uuid(snap["selected_gpu_uuid"]) == uuid)
            if dev == "RTX5070":
                check(f"Windows background inventory never expanded: {label} phase{snap['phase']} {snap['position']}",
                      {int(x) for x in snap["background_inventory_pids"]} == background)
            for query in ("gpu_state", "compute_processes"):
                if snap[query].get("returncode") != 0:
                    warning(f"{dev} round {rnd} phase {snap['phase']} {snap['position']}: telemetry {query} query failed.")
            if snap.get("compute_parse_errors"):
                warning(f"{dev} round {rnd} phase {snap['phase']} {snap['position']}: compute query parse errors present.")
            if snap["compute_processes"].get("returncode") == 0:
                unknown = []
                for fields in csv.reader(snap["compute_processes"].get("stdout", "").splitlines()):
                    if len(fields) < 3 or normalized_uuid(fields[0]) != uuid:
                        continue
                    pid = int(fields[1].strip())
                    if pid != owned and pid not in background:
                        unknown.append(pid)
                if unknown:
                    warning(f"{dev} round {rnd} phase {snap['phase']} {snap['position']}: independently observed unknown NVML PIDs {unknown}.")
                check(f"External PID flag agrees with independently parsed query: {label} phase{snap['phase']} {snap['position']}",
                      bool(snap["external_compute_processes"]) == bool(unknown))
    for row in timings:
        raw = row["_raw"]
        readback = json.loads(raw["readback_json"])
        api = raw["precision_api"]
        if api == "legacy_allow_tf32":
            ok = readback == {"matmul_allow_tf32": row["mode"] == "allow_tf32"}
        elif api == "fp32_precision":
            ok = readback == {"global_fp32_precision": "ieee", "matmul_fp32_precision": "tf32" if row["mode"] == "allow_tf32" else "ieee"}
        else:
            ok = False
        if not ok:
            readback_errors.append((row["device"], row["round"], row["phase"], row["dtype"], row["n"]))
    check("Every window records matching precision-interface readback", not readback_errors, readback_errors[:10])
    for dev, envs in env_by_device.items():
        keys = ("gpu_uuid", "gpu_name", "compute_capability", "torch", "torch_cuda_build", "numpy", "precision_api", "other_matmul_settings")
        stable = all(all(e[k] == envs[0][k] for k in keys)
                     and e["driver_query"].get("stdout") == envs[0]["driver_query"].get("stdout") for e in envs)
        check(f"Fixed GPU and software settings across three rounds: {dev}", len(envs) == 3 and stable)
        if any(e["driver_query"].get("returncode") != 0 for e in envs):
            warning(f"{dev}: driver/UUID query failed in at least one round; use available GPU properties and preserve that identification limit.")
    identical = all(len(records) == 3 and {r[0] for r in records} == set(DEVICES)
                    and len({(r[1], r[2]) for r in records}) == 1 for records in input_hashes.values())
    check("Exact CPU input hashes match across devices for all 12 round-context pairs", len(input_hashes) == 12 and identical)
    core_protocols = []
    core_keys = ("contexts", "rounds", "input_generation", "operator", "inference_context", "warmup", "measurement", "precision_api", "outputs", "estimands")
    for dev in DEVICES:
        frozen = FROZEN_SOURCES[dev]
        current_protocol = frozen["directory"] / "protocol.json"
        current_measure = frozen["directory"] / "tf32_measure.py"
        sources.extend([current_protocol, current_measure])
        check(f"Per-device runs match exact frozen sources: {dev}",
              protocol_hashes[dev] == {frozen["protocol"]} and script_hashes[dev] == {frozen["script"]}
              and sha256(current_protocol) == frozen["protocol"] and sha256(current_measure) == frozen["script"],
              {"protocol_sha256": sorted(protocol_hashes[dev]), "measurement_script_sha256": sorted(script_hashes[dev])})
        proto = json.loads(current_protocol.read_text(encoding="utf-8"))
        core_protocols.append({k: proto[k] for k in core_keys})
    check("Identical timing/input/estimand protocol across cloud-v3 and Windows-v4", all(p == core_protocols[0] for p in core_protocols))
    desktop_gate_dir = HERE.parent / "attempts" / "RTX5070_v3_pre_cuda_gate" / "round_01"
    desktop_gate_baseline = desktop_gate_dir / "pre_cuda_baseline.json"
    desktop_gate_status = desktop_gate_dir / "status.json"
    if desktop_gate_baseline.exists() and desktop_gate_status.exists():
        sources.extend([desktop_gate_baseline, desktop_gate_status])
        preflight_inventory = json.loads(desktop_gate_baseline.read_text(encoding="utf-8"))
        preflight_status = json.loads(desktop_gate_status.read_text(encoding="utf-8"))
        names = sorted({r["process_name"].replace("\\", "/").rsplit("/", 1)[-1] for r in preflight_inventory["processes"]})
        check("Preserved failed Windows-v3 gate contains zero timing windows", preflight_status["window_rows"] == 0)
        formal_counts = [e["background_inventory_count"] for e in environment_rows if e["device"] == "RTX5070"]
        validation["windows_desktop_context"] = {
            "preserved_preformal_inventory_count": len(preflight_inventory["processes"]),
            "preserved_preformal_process_names": names,
            "formal_round_inventory_counts": formal_counts,
            "wallpaper_engine_observed_preformal": any("wallpaper" in n.lower() for n in names),
            "limitation": "Windows desktop processes were visible in the preserved preformal query. Empty formal inventories do not prove their absence or inactivity; query visibility differs across execution contexts and transient activity is not continuously monitored."}
        warning("RTX5070 was measured in the actual Windows desktop environment. A preserved zero-timing preformal query observed "
                f"{len(preflight_inventory['processes'])} background GPU process entries, including WallpaperEngine if listed in validation.windows_desktop_context; "
                f"formal round inventory counts are {formal_counts}. Empty formal inventories do not establish an interference-free GPU or absence of desktop activity.")
    else:
        warning("RTX5070 uses a Windows desktop environment. Process inventory snapshots cannot establish absence of background GPU activity or exclude transient interference.")
    validation["complete_design_passed"] = not validation["errors"]
    validation["input_identity_pairs"] = {f"round{k[0]}_{k[1]}": rows for k, rows in input_hashes.items()}
    return sources, environment_rows


def make_figures(round_rows: list[dict], summary_rows: list[dict], out: Path) -> dict:
    os.environ["MPLCONFIGDIR"] = str(out / "mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    all_r = [r["R_off_over_on"] for r in round_rows] + [1]
    log_scale = max(all_r) / min(all_r) > 3
    if log_scale:
        lo = math.floor(math.log2(min(all_r)) * 2) / 2 - .15
        hi = math.ceil(math.log2(max(all_r)) * 2) / 2 + .15
        limits = [2 ** lo, 2 ** hi]
    else:
        span = max(max(all_r) - min(all_r), .2)
        limits = [max(0, min(min(all_r) - .08 * span, .9)), max(max(all_r) + .08 * span, 1.1)]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "s", "^"]
    exports = []
    for lang in ("中文", "英文"):
        zh = lang == "中文"
        available = {f.name for f in font_manager.fontManager.ttflist}
        preferred = (["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"] if zh
                     else ["Arial", "Liberation Sans", "DejaVu Sans"])
        family = next((name for name in preferred if name in available), "DejaVu Sans")
        plt.rcParams.update({"font.family": family, "font.size": 8,
                             "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                             "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
                             "svg.fonttype": "path", "axes.unicode_minus": False})
        fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 75 / 25.4), sharey=True)
        fig.subplots_adjust(left=.085, right=.99, bottom=.22, top=.79, wspace=.18)
        for panel, (device, ax) in enumerate(zip(DEVICES, axes)):
            ax.axhline(1, color="#777777", ls="--", lw=.8, zorder=1)
            if log_scale:
                ax.set_yscale("log", base=2)
                exp_min = math.ceil(math.log2(limits[0]))
                exp_max = math.floor(math.log2(limits[1]))
                ax.yaxis.set_major_locator(FixedLocator([2 ** k for k in range(exp_min, exp_max + 1)]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.minorticks_off()
            ax.set_ylim(*limits)
            for x, (dtype, n) in enumerate(CONTEXTS):
                rows = sorted([r for r in round_rows if (r["device"], r["dtype"], r["n"]) == (device, dtype, n)], key=lambda r: r["round"])
                for j, row in enumerate(rows):
                    ax.scatter(x + (j - 1) * .12, row["R_off_over_on"], s=24, marker=markers[j],
                               color=colors[j], edgecolors="white", linewidths=.35, zorder=3)
                r_med = median(r["R_off_over_on"] for r in rows)
                ax.hlines(r_med, x - .29, x + .29, color="black", lw=1.5, zorder=4)
            ax.set_xlim(-.5, 3.5)
            ax.set_xticks(range(4), ["FP32\n2048", "FP32\n4096", "FP32\n6144", "FP16\n4096"])
            ax.set_xlabel("矩阵阶数 N" if zh else "Matrix dimension N", labelpad=4)
            ax.text(.02, 1.06, f"({chr(97 + panel)}) {'RTX 5070' if device == 'RTX5070' else device}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="out", width=.7, length=3)
        ylabel = ("耗时比 R = T_off / T_on" if zh else "Timing ratio R = T_off / T_on")
        if log_scale:
            ylabel += "（log2轴）" if zh else " (log2 axis)"
        axes[0].set_ylabel(ylabel, labelpad=6)
        handles = [Line2D([], [], ls="none", marker=markers[j], color=colors[j], markersize=4,
                          label=f"轮次 {j + 1}" if zh else f"Round {j + 1}") for j in range(3)]
        handles.append(Line2D([], [], color="black", lw=1.5, label="三轮中位数" if zh else "Median of three rounds"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995), ncol=4, frameon=False,
                   handletextpad=.5, columnspacing=1.5, fontsize=7.5)
        stem = out / ("figure7_tf32_zh" if zh else "figure7_tf32_en")
        for ext in ("png", "pdf", "svg"):
            path = stem.with_suffix("." + ext)
            fig.savefig(path, dpi=600, facecolor="white")
            exports.append(portable_path(path))
        plt.close(fig)
    return {"paths": exports, "width_mm": 180, "height_mm": 75, "png_dpi": 600,
            "common_y_scale": "log2" if log_scale else "linear", "common_y_limits": limits,
            "each_point": "one paired measurement round, not a timing window",
            "summary_mark": "median of three round-level R values", "confidence_intervals": "none"}


def captions(fig_info: dict) -> dict:
    scale_zh = "纵轴为以2为底的对数轴。" if fig_info["common_y_scale"] == "log2" else "三个面板采用相同的线性纵轴范围。"
    scale_en = "The y axis uses a base-2 logarithmic scale. " if fig_info["common_y_scale"] == "log2" else "The panels share a common linear y-axis range. "
    return {
        "zh": "软件数学模式切换下的配对GEMM耗时比。(a) A100；(b) T4；(c) RTX 5070。每台设备内固定矩阵输入、规模及输入输出dtype，仅切换矩阵乘法的TF32许可设置。三个轮次分别采用禁止—允许—禁止、允许—禁止—允许和禁止—允许—禁止的阶段顺序。每阶段20个窗口，每窗口10次GEMM，以CUDA Event测得的窗口时间除以10；先取阶段窗口中位数，再将同一设置的重复阶段取几何平均，得到R=T_off/T_on。三个标记为三轮R，黑色短线为三轮中位数；虚线R=1表示两设置耗时相同，R>1表示允许TF32时更快。" + scale_zh + "窗口用于描述同轮稳定性，不作为独立硬件重复；图中不提供显著性推断。各设备的软件环境不同，跨设备效应幅度不能单独归因于架构。RTX 5070在实际Windows桌面环境测量，进程库存快照不能排除后台GPU活动。TF32许可设置不等于直接确认某个内核被执行；恢复比与CPU float64输出差异另行报告。",
        "en": "Paired GEMM timing ratios under changes in the backend mathematical mode. (a) A100; (b) T4; (c) RTX 5070. Within each device, matrix inputs, dimensions and input/output dtypes were fixed while only the TF32 permission for matrix multiplication was changed. The three rounds used off–on–off, on–off–on and off–on–off sequences. Each phase comprised 20 windows of 10 GEMM calls; CUDA Event window times were divided by 10. Phase medians were computed first, and repeated phases with the same setting were combined geometrically to obtain R=T_off/T_on. Symbols show the three round-level ratios and black segments their median. The dashed line at R=1 indicates equal timings; R>1 indicates faster execution when TF32 was permitted. " + scale_en + "Windows describe within-round stability and are not independent hardware replicates; no significance inference is shown. Software stacks differ across devices, so between-device effect magnitudes do not isolate architecture. RTX 5070 measurements used the actual Windows desktop environment; process-inventory snapshots cannot exclude background GPU activity. Permitting TF32 does not directly confirm execution of a particular kernel. Recovery ratios and CPU float64 output differences are reported separately."}


def generate_report(out: Path, validation: dict, round_rows: list[dict], summary_rows: list[dict], sources: list[dict], fig_info: dict, environments: list[dict]) -> None:
    lines = ["# TF32软件计算路径与GPU画像失配：实测结果", "", "本报告只描述本次独立机制诊断，不改动历史数据或论文。", "",
             f"完整设计核验：**{'通过' if validation['complete_design_passed'] else '未通过'}**；{len(round_rows)}个设备×上下文×轮次配对比较，2160个计时窗口。每张设备只有1张实体GPU，三轮为该设备的重复测量。", "",
             "正式采集沿用v2冻结的输入构造：CPU端以PCG64原始uint64高24位构造[-1,1)上的精确float32均匀网格，再按上下文转为FP16；同轮同上下文跨设备A/B字节SHA256一致。云端A100/T4采用冻结v3，Windows RTX 5070采用专用v4；两者的输入、阶段、预热、计时和统计量相同，区别在于进程归属核验与桌面背景库存记录。保留的前置烟测、纯CPU输入检查和零计时门禁失败均不进入正式计时统计。", "",
             "## 耗时比、恢复程度与输出差异", "", "R>1表示允许TF32时测得耗时较短。恢复比为同一轮阶段3/阶段1；delta在CPU float64上以禁止TF32输出为参考计算，参考输出不称为数学真值。", "",
             "| GPU | 上下文 | 第1轮R | 第2轮R | 第3轮R | 三轮中位R | 恢复比范围 | delta范围 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        context = f"{'FP32' if row['dtype'] == 'float32' else 'FP16'}, N={row['n']}"
        lines.append(f"| {row['device']} | {context} | {row['R_round1']:.5f} | {row['R_round2']:.5f} | {row['R_round3']:.5f} | {row['R_median']:.5f} | {row['recovery_min']:.5f}–{row['recovery_max']:.5f} | {row['delta_min']:.6g}–{row['delta_max']:.6g} |")
    lines.extend(["", "## 与机制解释的关系", ""])
    for dev in DEVICES:
        fp32 = [r for r in summary_rows if r["device"] == dev and r["dtype"] == "float32"]
        control = next(r for r in summary_rows if r["device"] == dev and r["dtype"] == "float16")
        direction = "三个FP32规模均在全部三轮出现R>1" if all(r["all_R_above_1"] for r in fp32) else "三个FP32规模均在全部三轮出现R<1" if all(r["all_R_below_1"] for r in fp32) else "FP32耗时比方向存在跨轮或跨规模变化"
        lines.append(f"- **{dev}**：{direction}；FP32三轮中位R范围为{min(r['R_median'] for r in fp32):.5f}–{max(r['R_median'] for r in fp32):.5f}，FP16对照的三轮中位R为{control['R_median']:.5f}。这描述方向与量级，不等价于显著性结论。")
    comparable = [r for r in round_rows if r["recovery_abs_log"] >= r["switch_abs_log"]]
    a100_fp32 = [r for r in round_rows if r["device"] == "A100" and r["dtype"] == "float32"]
    rtx_fp32 = [r for r in round_rows if r["device"] == "RTX5070" and r["dtype"] == "float32"]
    t4_all = [r for r in round_rows if r["device"] == "T4"]
    fp16_all = [r for r in round_rows if r["dtype"] == "float16"]
    if all(r["R_off_over_on"] > 1 for r in a100_fp32 + rtx_fp32):
        a100_rec = [r["recovery_T3_over_T1"] for r in a100_fp32]
        rtx_rec = [r["recovery_T3_over_T1"] for r in rtx_fp32]
        lines.extend(["", "**有限机制判断：本次结果支持后端数学模式切换能够使固定GPU、矩阵规模和输入输出dtype下的执行时间发生可重复偏离。** "
                      f"A100和RTX 5070的全部18个FP32轮次R均大于1；对应恢复比分别为{min(a100_rec):.5f}–{max(a100_rec):.5f}和{min(rtx_rec):.5f}–{max(rtx_rec):.5f}，"
                      "恢复漂移相对各自FP32切换量级较小。A100提供较清楚的设备内支持，RTX 5070提供同向观察但须保留桌面背景干扰限制。"
                      f"T4全部上下文的R为{min(r['R_off_over_on'] for r in t4_all):.5f}–{max(r['R_off_over_on'] for r in t4_all):.5f}，"
                      f"且其恢复变化最大为{max(abs(r['recovery_T3_over_T1'] - 1) * 100 for r in t4_all):.2f}%；不能把接近1的小幅变化解释为清楚的TF32设置效应。"
                      f"三卡FP16对照的R为{min(r['R_off_over_on'] for r in fp16_all):.5f}–{max(r['R_off_over_on'] for r in fp16_all):.5f}，"
                      f"delta范围为{min(r['delta_CPU_float64'] for r in fp16_all):.6g}–{max(r['delta_CPU_float64'] for r in fp16_all):.6g}。"
                      f"两台支持TF32的设备在FP32下delta范围为{min(r['delta_CPU_float64'] for r in a100_fp32 + rtx_fp32):.6g}–{max(r['delta_CPU_float64'] for r in a100_fp32 + rtx_fp32):.6g}，"
                      "性能变化伴随内部精度变化，不能表述为无条件等精度加速。该结论不延伸至历史失配归因或调度收益。"])
    lines.extend(["", f"36个轮次比较中，{len(comparable)}个的恢复绝对对数变化不小于设置耗时比的绝对对数变化，具体行见round_ratios.csv。该比较只是并列描述漂移与切换量级，不是事后合格阈值，不用于删点。", "",
                  "解释时应结合恢复比、阴性设备对照和FP16任务对照。若设置切换量级与恢复漂移相当，本诊断不足以清楚隔离设置影响；若设备内FP32变化跨轮一致、恢复漂移相对较小而对照较弱，则与软件数学模式和硬件能力交互影响耗时的解释一致。没有运行内核计数器，不能声称直接观察到Tensor Core利用率变化。", "",
                  "## 适用范围", "", "1. 这是已知设置的主动干预，只验证所测设备、输入和软件环境；不用于追溯历史部署失配的原因。", "2. 不把历史71.72%的描述性共享成分解释为TF32贡献；不测定模拟共享比例rho。", "3. 不运行调度器、不比较校准器，因此不能证明PoolEMA、Full或其他策略具有QoS收益。", "4. 输入输出dtype相同不等于内部有效精度相同；数值差异需要结合应用容差解释，本实验不证明应用精度等价。", "5. 已知数学模式应优先写入画像元数据或触发重新画像；本实验不支持以在线校准替代显式配置管理。", "6. 不同设备的软件栈不同，效应幅度差异不能单独归因于架构。", "",
                  "## 文件", "", "- all_timing_windows.csv：全部原始字段、源文件与规范化字段。", "- stage_summary.csv：108个阶段×上下文中位数及窗口波动描述。", "- round_ratios.csv：36个每轮R、恢复比、输出差异。", "- device_context_summary.csv：12个设备×上下文的三轮R及中位数。", "- output_differences_CPU_float64.csv：原始输出差异字段与来源。", "- validation.json：完整设计与数值检查。", "- figure_captions.json：中英文论文图注。", "- source_sha256.csv / output_sha256.csv：输入与输出校验清单。", "", "图形纵轴：" + fig_info["common_y_scale"] + "；无截断点、无显著性标记、无置信区间。", "",
                  "## 环境记录", "", "| GPU | PyTorch | CUDA构建 | NumPy | 精度接口 |", "|---|---|---|---|---|"])
    for dev in DEVICES:
        env = next(e for e in environments if e["device"] == dev)
        lines.append(f"| {env['gpu_name']} | {env['torch']} | {env['torch_cuda_build']} | {env['numpy']} | {env['precision_api']} |")
    lines.extend(["", "驱动查询、其他后端设置和逐轮环境详见environment_summary.csv及原始environment.json。各设备内三轮的软件、GPU标识和其他数学设置已逐项核对；跨设备输入矩阵以SHA256核对。", "", "## 采集状态与警告", ""])
    query_failed_rounds = sum(bool(e["telemetry_query_failure_seen"]) for e in environments)
    external_rounds = sum(bool(e["external_compute_process_seen"]) for e in environments)
    inferred_rounds = sum(bool(e["ownership_inferred"]) for e in environments)
    lines.append(f"九个正式轮次中，状态记录标记遥测查询失败的轮次为{query_failed_rounds}，标记新外部计算进程的轮次为{external_rounds}，使用显式推断的宿主/容器PID关联的轮次为{inferred_rounds}。边界快照不等于连续监控，未标记新进程不等于证明无干扰。")
    desktop = validation.get("windows_desktop_context")
    if desktop:
        lines.append(f"RTX 5070正式三轮报告的背景库存条目数为{desktop['formal_round_inventory_counts']}；保留的零计时前置查询观测到{desktop['preserved_preformal_inventory_count']}个背景进程，{'包括WallpaperEngine' if desktop['wallpaper_engine_observed_preformal'] else '名称保留在validation.json'}。两次执行上下文的查询可见性并不一致，不能用正式库存为空证明后台进程已停止或未使用GPU，RTX 5070结果须附桌面干扰限制。")
    if validation["warnings"]:
        lines.extend("- " + w for w in validation["warnings"])
    else:
        lines.append("已保存的测量状态记录未标记外部计算进程或遥测查询失败；这只说明边界快照和脚本检查的结果，不能排除快照之间所有瞬时干扰。")
    (out / "Results_and_Interpretation_Limits.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--timing-file", action="append")
    ap.add_argument("--difference-file", action="append")
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_files = discover(args.input_root, args.timing_file, ("windows.csv", "timings.csv", "timing_windows.csv", "timings_*.csv"), out)
    d_files = discover(args.input_root, args.difference_file, ("output_differences.csv", "numerical_differences.csv", "output_differences_*.csv"), out)
    if not t_files or not d_files:
        write_json(out / "validation.json", {"complete_design_passed": False,
                    "errors": ["Input timing and output-difference CSV files are required"],
                    "timing_files": [portable_path(p) for p in t_files], "difference_files": [portable_path(p) for p in d_files]})
        raise SystemExit("No complete measurement inputs found; no numerical results or figures generated.")
    raw_t = [r for p in t_files for r in read_csv(p)]
    raw_d = [r for p in d_files for r in read_csv(p)]
    try:
        timing = [normalize_timing(r) for r in raw_t]
        differences = [normalize_difference(r) for r in raw_d]
        validation = validate(timing, differences)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        write_json(out / "validation.json", {"complete_design_passed": False, "errors": [str(exc)]})
        raise
    validation["analyzed_at_utc"] = datetime.now(timezone.utc).isoformat()
    validation["analysis_script_sha256"] = sha256(Path(__file__))
    validation["timing_files"] = [portable_path(p) for p in t_files]
    validation["difference_files"] = [portable_path(p) for p in d_files]
    write_json(out / "validation.json", validation)
    write_csv(out / "all_timing_windows.csv", [{**r["_raw"], **{f"normalized_{k}": v for k, v in r.items() if k != "_raw"}} for r in timing])
    write_csv(out / "output_differences_CPU_float64.csv", [{**r["_raw"], **{f"normalized_{k}": v for k, v in r.items() if k != "_raw"}} for r in differences])
    if not validation["complete_design_passed"]:
        print(json.dumps({"complete_design_passed": False, "errors": validation["errors"]}, ensure_ascii=False, indent=2))
        return 2
    stage, rounds, summary = summaries(timing, differences)
    extra_sources, environments = audit_auxiliary(t_files, timing, differences, stage, rounds, validation)
    write_json(out / "validation.json", validation)
    inputs = sorted(set(t_files + d_files + extra_sources + [Path(__file__).resolve()]))
    source_manifest = [{"path": portable_path(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs]
    write_csv(out / "source_sha256.csv", source_manifest)
    if not validation["complete_design_passed"]:
        print(json.dumps({"complete_design_passed": False, "errors": validation["errors"]}, ensure_ascii=False, indent=2))
        return 2
    write_csv(out / "environment_summary.csv", environments)
    write_csv(out / "stage_summary.csv", stage)
    write_csv(out / "round_ratios.csv", rounds)
    write_csv(out / "device_context_summary.csv", summary)
    fig_info = make_figures(rounds, summary, out)
    write_json(out / "figure_export_metadata.json", fig_info)
    cap = captions(fig_info)
    write_json(out / "figure_captions.json", cap)
    (out / "Figure_Captions.md").write_text("# 中文\n\n" + cap["zh"] + "\n\n# English\n\n" + cap["en"] + "\n", encoding="utf-8")
    generate_report(out, validation, rounds, summary, source_manifest, fig_info, environments)
    output_files = [p for p in out.iterdir() if p.is_file() and p.name != "output_sha256.csv"]
    write_csv(out / "output_sha256.csv", [{"path": portable_path(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output_files)])
    print(json.dumps({"complete_design_passed": True, "n_windows": len(timing), "n_round_ratios": len(rounds), "output_dir": portable_path(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
