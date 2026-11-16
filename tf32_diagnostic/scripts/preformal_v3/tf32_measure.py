#!/usr/bin/env python3
"""Frozen, standalone TF32 diagnostic. Invoke once per round in a fresh process.

Only Python's standard library is imported before argument parsing; --help is
usable without torch/numpy. --smoke is explicitly segregated from formal data.
No GPU settings other than the selected PyTorch TF32 permission are changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = SCRIPT_DIR / "protocol.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class IncrementalCSV:
    def __init__(self, path, fields):
        self.stream = open(path, "x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()
        self.stream.flush()

    def append(self, row):
        self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def run_readonly(command):
    started = utc_now()
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=15, check=False)
        return {"command": command, "started_utc": started,
                "completed_utc": utc_now(), "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:
        return {"command": command, "started_utc": started,
                "completed_utc": utc_now(), "error": repr(exc)}


def query_compute_processes():
    query = run_readonly([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"])
    rows, errors = [], []
    if query.get("returncode") == 0:
        for row in csv.reader(query.get("stdout", "").splitlines()):
            if not row or all(not part.strip() for part in row):
                continue
            if len(row) < 3:
                errors.append({"row": row, "reason": "too_few_fields"})
                continue
            gpu_uuid, pid, process_name = (part.strip() for part in row[:3])
            try:
                pid_value = int(pid)
                if pid_value <= 0 or not gpu_uuid:
                    raise ValueError("Invalid PID or UUID")
            except ValueError:
                errors.append({"row": row, "reason": "invalid_pid_or_uuid"})
                continue
            rows.append({"gpu_uuid": gpu_uuid, "pid": pid_value,
                         "process_name": process_name,
                         "used_gpu_memory": row[3].strip() if len(row) > 3 else None})
    return {"query": query, "processes": rows, "parse_errors": errors}


def establish_ownership(baseline, after_context, selected_uuid, benchmark_pid):
    """Conservative PID association; never infer through multiple/unknown PIDs."""
    if baseline["query"].get("returncode") != 0 or baseline["parse_errors"]:
        raise RuntimeError("Pre-CUDA compute-process query failed or was malformed")
    if baseline["processes"]:
        raise RuntimeError("Pre-CUDA compute-process baseline is not empty")
    if after_context["query"].get("returncode") != 0 or after_context["parse_errors"]:
        raise RuntimeError("Post-context compute-process query failed or was malformed")
    selected = [row for row in after_context["processes"]
                if normalize_gpu_uuid(row["gpu_uuid"]) == normalize_gpu_uuid(selected_uuid)]
    if len(selected) != 1:
        raise RuntimeError("Post-context selected GPU must have exactly one compute process; observed " + str(len(selected)))
    observed_pid = selected[0]["pid"]
    direct = observed_pid == benchmark_pid
    return {"benchmark_pid": benchmark_pid, "owned_nvml_pid": observed_pid,
            "selected_gpu_uuid": selected_uuid, "inferred": not direct,
            "ownership_basis": "direct NVML PID equals Python PID" if direct else
            "sole process appearing after own CUDA initialization with empty baseline",
            "pre_cuda_baseline": baseline, "post_context_query": after_context,
            "fixed_for_entire_round": True, "created_utc": utc_now()}


def telemetry_snapshot(path, phase, position, selected_uuid, owned_nvml_pid):
    result = {"utc": utc_now(), "phase": phase, "position": position,
              "benchmark_pid": os.getpid(), "selected_gpu_uuid": selected_uuid,
              "owned_nvml_pid": owned_nvml_pid}
    result["gpu_state"] = run_readonly([
        "nvidia-smi", "--query-gpu=timestamp,index,uuid,name,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"])
    parsed_query = query_compute_processes()
    result["compute_processes"] = parsed_query["query"]
    result["compute_parse_errors"] = parsed_query["parse_errors"]
    external = []
    query = result["compute_processes"]
    if query.get("returncode") == 0:
        for row in csv.reader(query.get("stdout", "").splitlines()):
            if len(row) < 3:
                continue
            gpu_uuid, pid, process_name = (part.strip() for part in row[:3])
            if selected_uuid and normalize_gpu_uuid(gpu_uuid) != normalize_gpu_uuid(selected_uuid):
                continue
            try:
                own = int(pid) == owned_nvml_pid
            except ValueError:
                own = False
            if not own:
                external.append({"gpu_uuid": gpu_uuid, "pid": pid,
                                 "process_name": process_name})
    result["external_compute_processes"] = external
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def normalize_gpu_uuid(value):
    """torch properties may omit the GPU- prefix that nvidia-smi prints."""
    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


class PrecisionControl:
    """Use exactly one generation of TF32 API per process."""
    OTHER_ATTRIBUTES = ("allow_fp16_reduced_precision_reduction",
                        "allow_bf16_reduced_precision_reduction",
                        "allow_fp16_accumulation")

    def __init__(self, torch):
        self.torch = torch
        version_match = re.match(r"(\d+)\.(\d+)", torch.__version__)
        version = tuple(map(int, version_match.groups())) if version_match else (0, 0)
        self.api = "legacy_allow_tf32"
        if (version >= (2, 9) and hasattr(torch.backends, "fp32_precision")
                and hasattr(torch.backends.cuda.matmul, "fp32_precision")):
            self.api = "fp32_precision"
            torch.backends.fp32_precision = "ieee"
        self.other_initial = self.other_settings()

    def other_settings(self):
        settings = {}
        for name in self.OTHER_ATTRIBUTES:
            try:
                settings[name] = getattr(self.torch.backends.cuda.matmul, name)
            except (AttributeError, RuntimeError):
                settings[name] = "unsupported"
        return settings

    def set_mode(self, mode):
        allow = mode == "allow_tf32"
        if self.api == "fp32_precision":
            self.torch.backends.cuda.matmul.fp32_precision = "tf32" if allow else "ieee"
            readback = {"global_fp32_precision": self.torch.backends.fp32_precision,
                        "matmul_fp32_precision": self.torch.backends.cuda.matmul.fp32_precision}
            expected = "tf32" if allow else "ieee"
            if readback != {"global_fp32_precision": "ieee", "matmul_fp32_precision": expected}:
                raise RuntimeError("New precision interface did not read back the requested mode: " + repr(readback))
        else:
            self.torch.backends.cuda.matmul.allow_tf32 = allow
            readback = {"matmul_allow_tf32": self.torch.backends.cuda.matmul.allow_tf32}
            if readback["matmul_allow_tf32"] != allow:
                raise RuntimeError("Legacy precision interface did not read back the requested mode")
        if self.other_settings() != self.other_initial:
            raise RuntimeError("Other matmul settings changed during the experiment")
        return readback


def hash_array(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def generate_cpu_inputs(np, protocol, round_plan):
    """Use a defined raw-bit-to-float transform, independent of NumPy samplers."""
    contexts = {item["id"]: item for item in protocol["contexts"]}
    host_inputs, input_manifest = {}, []
    for context_id in round_plan["context_order"]:
        context = contexts[context_id]
        bit_generator = np.random.PCG64(np.random.SeedSequence(
            [round_plan["seed"], context["input_stream_id"]]))

        def next_matrix():
            raw = bit_generator.random_raw(context["n"] * context["n"])
            raw >>= np.uint64(40)
            matrix = raw.astype(np.float32)
            matrix *= np.float32(2 ** -23)
            matrix -= np.float32(1.0)
            return matrix.reshape(context["n"], context["n"]).astype(
                np.dtype(context["dtype"]), copy=False)

        matrix_a, matrix_b = next_matrix(), next_matrix()
        host_inputs[context_id] = (matrix_a, matrix_b)
        input_manifest.append({"context": context_id, "round_seed": round_plan["seed"],
                               "input_stream_id": context["input_stream_id"],
                               "shape": list(matrix_a.shape), "dtype": str(matrix_a.dtype),
                               "a_sha256": hash_array(matrix_a), "b_sha256": hash_array(matrix_b),
                               "contiguous": bool(matrix_a.flags.c_contiguous and matrix_b.flags.c_contiguous),
                               "generator": "PCG64 raw uint64 high24 / 2**23 - 1",
                               "numpy_version": np.__version__})
    return host_inputs, input_manifest


def output_check(array, np):
    finite_count = 0
    flat = array.reshape(-1)
    for offset in range(0, flat.size, 1_000_000):
        finite_count += int(np.isfinite(flat[offset:offset + 1_000_000]).sum())
    return {"output_sha256": hash_array(array), "output_elements": int(flat.size),
            "finite_elements": finite_count, "all_finite": finite_count == flat.size,
            "output_dtype": str(array.dtype), "output_shape": json.dumps(list(array.shape))}


def numerical_difference(on, off, np):
    """Convert each chunk to CPU float64 BEFORE subtraction/squaring."""
    if on.shape != off.shape:
        raise RuntimeError("Paired output shapes differ")
    on_flat, off_flat = on.reshape(-1), off.reshape(-1)
    numerator_chunks, denominator_chunks = [], []
    for offset in range(0, off_flat.size, 1_000_000):
        off64 = off_flat[offset:offset + 1_000_000].astype(np.float64)
        on64 = on_flat[offset:offset + 1_000_000].astype(np.float64)
        if not (bool(np.isfinite(off64).all()) and bool(np.isfinite(on64).all())):
            return {"valid": False, "reason": "nonfinite_output", "delta": None,
                    "off_frobenius": None, "difference_frobenius": None}
        difference64 = on64 - off64
        numerator_chunks.append(float(np.sum(difference64 * difference64, dtype=np.float64)))
        denominator_chunks.append(float(np.sum(off64 * off64, dtype=np.float64)))
    numerator = math.sqrt(math.fsum(numerator_chunks))
    denominator = math.sqrt(math.fsum(denominator_chunks))
    if not (math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0):
        return {"valid": False, "reason": "nonfinite_norm_or_nonpositive_denominator",
                "delta": None, "off_frobenius": denominator if math.isfinite(denominator) else None,
                "difference_frobenius": numerator if math.isfinite(numerator) else None}
    return {"valid": True, "reason": "", "delta": numerator / denominator,
            "off_frobenius": denominator, "difference_frobenius": numerator}


def environment(torch, np, precision, args):
    props = torch.cuda.get_device_properties(0)
    gpu_uuid = getattr(props, "uuid", None)
    gpu_uuid = str(gpu_uuid) if gpu_uuid is not None else None
    return {"created_utc": utc_now(), "device_label": args.device_label,
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "hostname": platform.node(),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "numpy": np.__version__, "cudnn": torch.backends.cudnn.version(),
            "cuda_device_index": 0, "gpu_name": props.name, "gpu_uuid": gpu_uuid,
            "compute_capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory,
            "cuda_visible_device_count": torch.cuda.device_count(),
            "pid": os.getpid(), "precision_api": precision.api,
            "other_matmul_settings": precision.other_initial,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "environment_variables": {name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_LAUNCH_BLOCKING", "PYTORCH_CUDA_ALLOC_CONF")},
            "driver_query": run_readonly(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Separate interface/output check; never formal data")
    parser.add_argument("--input-check", action="store_true", help="CPU NumPy input hashes only; no torch import, CUDA initialization or GEMM")
    args = parser.parse_args()
    if args.smoke and args.input_check:
        parser.error("--smoke and --input-check are mutually exclusive")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    round_plan = next(item for item in protocol["rounds"] if item["round"] == args.round)
    contexts = {item["id"]: item for item in protocol["contexts"]}
    windows = 1 if args.smoke else protocol["measurement"]["windows_per_context_phase"]
    calls = 1 if args.smoke else protocol["measurement"]["calls_per_window"]
    warmup_calls = 2 if args.smoke else protocol["warmup"]["minimum_calls"]
    warmup_seconds = 0.0 if args.smoke else protocol["warmup"]["minimum_seconds"]
    warmup_batch = 1 if args.smoke else protocol["warmup"]["batch_calls"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = "input_check_round_" if args.input_check else ("smoke_round_" if args.smoke else "round_")
    run_dir = args.output_dir / (run_prefix + f"{args.round:02d}")
    run_dir.mkdir(exist_ok=False)
    status = {"protocol_id": protocol["protocol_id"], "state": "preparing",
              "device_label": args.device_label, "round": args.round,
              "smoke": args.smoke, "input_check": args.input_check,
              "started_utc": utc_now(), "pid": os.getpid(),
              "window_rows": 0, "completed_context_phases": 0,
              "expected_window_rows": 0 if args.input_check else 12 * windows,
              "expected_context_phases": 0 if args.input_check else 12,
              "paired_output_differences": 0,
              "external_compute_process_seen": False, "telemetry_query_failure_seen": False,
              "nonfinite_output_seen": False, "run_directory": str(run_dir.resolve())}
    write_json(run_dir / "status.json", status)
    csvs = []
    try:
        import numpy as np
        if args.input_check:
            host_inputs, input_manifest = generate_cpu_inputs(np, protocol, round_plan)
            write_json(run_dir / "input_manifest.json", input_manifest)
            host_inputs.clear()
            status.update({"state": "completed", "completed_utc": utc_now(),
                           "numpy": np.__version__, "torch_imported": False,
                           "cuda_initialized": False, "gemm_calls": 0,
                           "input_contexts": len(input_manifest),
                           "script_sha256": sha256_file(__file__),
                           "protocol_sha256": sha256_file(PROTOCOL_PATH)})
            write_json(run_dir / "status.json", status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return 0
        baseline = query_compute_processes()
        write_json(run_dir / "pre_cuda_baseline.json", baseline)
        if baseline["query"].get("returncode") != 0 or baseline["parse_errors"] or baseline["processes"]:
            raise RuntimeError("A successful empty pre-CUDA compute-process baseline is required")
        overrides = {name: os.environ.get(name) for name in
                     ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")}
        if any(value is not None and value != "" for value in overrides.values()):
            raise RuntimeError("Nonempty TF32 override environment variable; preserving it and aborting: " + repr(overrides))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; no formal GPU measurements were performed")
        torch.cuda.set_device(0)
        precision = PrecisionControl(torch)
        env = environment(torch, np, precision, args)
        write_json(run_dir / "environment.json", env)
        write_json(run_dir / "configuration.json", {
            "protocol": protocol, "round_plan": round_plan, "smoke": args.smoke,
            "windows_per_context_phase": windows, "calls_per_window": calls,
            "minimum_warmup_calls": warmup_calls, "minimum_warmup_seconds": warmup_seconds,
            "script_sha256": sha256_file(__file__), "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "created_utc": utc_now()})

        # All inputs are generated on CPU before ANY input is copied to GPU.
        host_inputs, input_manifest = generate_cpu_inputs(np, protocol, round_plan)
        write_json(run_dir / "input_manifest.json", input_manifest)
        gpu_inputs = {key: (torch.from_numpy(pair[0]).to("cuda"), torch.from_numpy(pair[1]).to("cuda"))
                      for key, pair in host_inputs.items()}
        torch.cuda.synchronize()
        host_inputs.clear()

        # Establish ownership only after real inputs occupy the selected GPU.
        after_context = query_compute_processes()
        write_json(run_dir / "post_context_processes.json", after_context)
        if not env["gpu_uuid"]:
            raise RuntimeError("Selected CUDA GPU UUID is unavailable; ownership cannot be assigned safely")
        ownership = establish_ownership(baseline, after_context, env["gpu_uuid"], os.getpid())
        ownership["namespace_status"] = None
        if platform.system() == "Linux":
            try:
                ownership["namespace_status"] = Path("/proc/self/status").read_text(encoding="utf-8")
            except OSError as exc:
                ownership["namespace_status_error"] = repr(exc)
        write_json(run_dir / "process_ownership.json", ownership)
        env["process_ownership"] = {key: ownership[key] for key in (
            "benchmark_pid", "owned_nvml_pid", "inferred", "ownership_basis")}
        write_json(run_dir / "environment.json", env)

        common = ["device", "round", "phase", "context", "seed", "mode", "precision_api", "readback_json"]
        window_csv = IncrementalCSV(run_dir / "windows.csv", common + ["window", "calls", "gpu_window_ms", "gpu_per_call_ms", "success", "completed_utc"])
        stage_csv = IncrementalCSV(run_dir / "stages.csv", common + ["n", "dtype", "windows", "calls_per_window", "median_ms", "min_ms", "max_ms", "warmup_calls", "warmup_seconds", "completed_utc"])
        output_csv = IncrementalCSV(run_dir / "output_checks.csv", common + ["output_sha256", "output_elements", "finite_elements", "all_finite", "output_dtype", "output_shape", "paired_reference_role"])
        diff_csv = IncrementalCSV(run_dir / "numerical_differences.csv", ["device", "round", "context", "seed", "off_phase", "on_phase", "off_sha256", "on_sha256", "valid", "reason", "delta", "off_frobenius", "difference_frobenius", "arithmetic"])
        summary_csv = IncrementalCSV(run_dir / "round_summary.csv", ["device", "round", "context", "seed", "phase_sequence", "stage1_median_ms", "stage2_median_ms", "stage3_median_ms", "T_off_ms", "T_on_ms", "R_off_over_on", "recovery_T3_over_T1", "delta", "numerical_valid", "external_compute_process_seen", "smoke"])
        csvs = [window_csv, stage_csv, output_csv, diff_csv, summary_csv]
        stage_results, output_pairs, paired_differences = {}, {}, {}
        status["state"] = "measuring"
        write_json(run_dir / "status.json", status)

        def take_snapshot(phase, position):
            snap = telemetry_snapshot(run_dir / "telemetry.jsonl", phase, position,
                                      env["gpu_uuid"], ownership["owned_nvml_pid"])
            status["external_compute_process_seen"] |= bool(snap["external_compute_processes"])
            status["telemetry_query_failure_seen"] |= (any(snap[name].get("returncode") != 0 for name in ("gpu_state", "compute_processes")) or bool(snap["compute_parse_errors"]))
            write_json(run_dir / "status.json", status)

        with torch.inference_mode():
            for phase, mode in enumerate(round_plan["phases"], 1):
                torch.cuda.synchronize()
                readback = precision.set_mode(mode)
                take_snapshot(phase, "before")
                for context_id in round_plan["context_order"]:
                    context = contexts[context_id]
                    a, b = gpu_inputs[context_id]
                    metadata = {"device": args.device_label, "round": args.round,
                                "phase": phase, "context": context_id, "seed": round_plan["seed"],
                                "mode": mode, "precision_api": precision.api,
                                "readback_json": json.dumps(readback, sort_keys=True)}
                    status.update({"current_phase": phase, "current_context": context_id, "current_mode": mode})
                    write_json(run_dir / "status.json", status)
                    torch.cuda.synchronize()
                    warmup_start, warmed = time.perf_counter(), 0
                    while warmed < warmup_calls or time.perf_counter() - warmup_start < warmup_seconds:
                        for _ in range(warmup_batch):
                            output = torch.mm(a, b)
                        torch.cuda.synchronize()
                        warmed += warmup_batch
                    warmup_elapsed = time.perf_counter() - warmup_start
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    times = []
                    for window in range(1, windows + 1):
                        start_event.record()
                        for _ in range(calls):
                            output = torch.mm(a, b)
                        end_event.record()
                        end_event.synchronize()
                        elapsed_ms = float(start_event.elapsed_time(end_event))
                        per_call_ms = elapsed_ms / calls
                        if not (math.isfinite(per_call_ms) and per_call_ms > 0):
                            raise RuntimeError("Nonpositive or nonfinite CUDA Event measurement")
                        times.append(per_call_ms)
                        window_csv.append({**metadata, "window": window, "calls": calls,
                                           "gpu_window_ms": elapsed_ms, "gpu_per_call_ms": per_call_ms,
                                           "success": True, "completed_utc": utc_now()})
                        status["window_rows"] += 1
                    stage_median = statistics.median(times)
                    stage_results[(phase, context_id)] = stage_median
                    stage_csv.append({**metadata, "n": context["n"], "dtype": context["dtype"],
                                      "windows": windows, "calls_per_window": calls,
                                      "median_ms": stage_median, "min_ms": min(times), "max_ms": max(times),
                                      "warmup_calls": warmed, "warmup_seconds": warmup_elapsed,
                                      "completed_utc": utc_now()})
                    # Copy and inspect the LAST measured output after the timing window.
                    cpu_output = output.cpu().numpy()
                    checks = output_check(cpu_output, np)
                    status["nonfinite_output_seen"] |= not checks["all_finite"]
                    role = "first_mode_output" if phase <= 2 else "recovery_hash_only"
                    output_csv.append({**metadata, **checks, "paired_reference_role": role})
                    if phase <= 2:
                        pair_entry = output_pairs.setdefault(context_id, {})
                        pair_entry[mode] = {"array": cpu_output, "phase": phase, "sha256": checks["output_sha256"]}
                        if "ieee" in pair_entry and "allow_tf32" in pair_entry:
                            off, on = pair_entry["ieee"], pair_entry["allow_tf32"]
                            difference = numerical_difference(on["array"], off["array"], np)
                            paired_differences[context_id] = difference
                            diff_csv.append({"device": args.device_label, "round": args.round,
                                             "context": context_id, "seed": round_plan["seed"],
                                             "off_phase": off["phase"], "on_phase": on["phase"],
                                             "off_sha256": off["sha256"], "on_sha256": on["sha256"],
                                             **difference, "arithmetic": "CPU float64 before subtraction; chunked Frobenius norm"})
                            del output_pairs[context_id]
                            del off, on, pair_entry
                            status["paired_output_differences"] += 1
                    del cpu_output
                    status["completed_context_phases"] += 1
                    write_json(run_dir / "status.json", status)
                    print(json.dumps({"event": "context_completed", "device": args.device_label,
                                      "round": args.round, "phase": phase, "context": context_id,
                                      "mode": mode, "median_ms": stage_median,
                                      "window_rows": status["window_rows"]}), flush=True)
                take_snapshot(phase, "after")

        for context_id in round_plan["context_order"]:
            stage_times = [stage_results[(phase, context_id)] for phase in (1, 2, 3)]
            mode_times = {}
            for mode in ("ieee", "allow_tf32"):
                selected = [value for value, stage_mode in zip(stage_times, round_plan["phases"]) if stage_mode == mode]
                mode_times[mode] = math.exp(math.fsum(math.log(value) for value in selected) / len(selected))
            difference = paired_differences[context_id]
            summary_csv.append({"device": args.device_label, "round": args.round,
                                "context": context_id, "seed": round_plan["seed"],
                                "phase_sequence": "/".join(round_plan["phases"]),
                                "stage1_median_ms": stage_times[0], "stage2_median_ms": stage_times[1],
                                "stage3_median_ms": stage_times[2], "T_off_ms": mode_times["ieee"],
                                "T_on_ms": mode_times["allow_tf32"],
                                "R_off_over_on": mode_times["ieee"] / mode_times["allow_tf32"],
                                "recovery_T3_over_T1": stage_times[2] / stage_times[0],
                                "delta": difference["delta"], "numerical_valid": difference["valid"],
                                "external_compute_process_seen": status["external_compute_process_seen"],
                                "smoke": args.smoke})
        if status["window_rows"] != status["expected_window_rows"] or status["completed_context_phases"] != 12 or status["paired_output_differences"] != 4:
            raise RuntimeError("Incomplete measurement counts")
        if status["nonfinite_output_seen"] or any(not item["valid"] for item in paired_differences.values()):
            status["state"] = "completed_with_invalid_numerics"
        elif status["external_compute_process_seen"]:
            status["state"] = "completed_with_external_compute_warning"
        else:
            status["state"] = "completed"
        status["completed_utc"] = utc_now()
        write_json(run_dir / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        status.update({"state": "failed", "failed_utc": utc_now(), "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()})
        write_json(run_dir / "status.json", status)
        print(json.dumps({"event": "failed", "run_directory": str(run_dir),
                          "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1
    finally:
        for incremental in csvs:
            incremental.close()


if __name__ == "__main__":
    sys.exit(main())
