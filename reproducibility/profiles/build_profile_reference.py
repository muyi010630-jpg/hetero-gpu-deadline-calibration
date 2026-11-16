from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_BIASES = {
    "RTX5070": {"compute": 1.35, "memory": 0.72, "inference": 1.22},
    "T4": {"compute": 0.82, "memory": 1.22, "inference": 0.90},
}


def family_to_context(row: Dict[str, str]) -> Tuple[str, int, int, str]:
    family = row["family"]
    size = int(row["size"])
    batch = int(row["batch"])
    dtype = row["dtype"]
    if family == "gemm":
        return "compute", size, batch, "fp16" if dtype == "float16" else "fp32"
    if family == "saxpy":
        return "memory", size, batch, "fp32"
    if family == "cifar10_resnet18":
        return "inference", size, batch, "fp16"
    raise ValueError(f"Unsupported family: {family}")


def device_profile_name(path: Path) -> str:
    label = " / ".join(path.parts).lower()
    if "5070" in label:
        return "RTX5070"
    if "t4" in label:
        return "T4"
    return path.stem


def build_rows(paths: Iterable[Path]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, Tuple[str, int, int, str]], List[float]] = {}
    devices: Dict[str, Tuple[str, float]] = {}
    for path in paths:
        label = device_profile_name(path)
        if "5070" in label:
            vendor, memory_gb = "NVIDIA", 11.94
        elif label == "T4":
            vendor, memory_gb = "NVIDIA", 15.00
        else:
            vendor, memory_gb = "NVIDIA", 0.0
        devices[label] = (vendor, memory_gb)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("success", "True") != "True":
                    continue
                if row.get("mode") != "gpu":
                    continue
                ctx = family_to_context(row)
                grouped.setdefault((label, ctx), []).append(float(row["gpu_ms"]) / 1000.0)
    if not grouped:
        raise RuntimeError("No usable rows found")

    rows: List[Dict[str, object]] = []
    for (device, ctx), samples in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        family, size, batch, dtype = ctx
        true_s = statistics.median(samples)
        bias = DEFAULT_BIASES.get(device, {}).get(family, 1.10)
        rows.append({
            "device": device,
            "vendor": devices[device][0],
            "memory_gb": devices[device][1],
            "family": family,
            "size": size,
            "batch": batch,
            "dtype": dtype,
            "initial_s": round(true_s * bias, 9),
            "true_s": round(true_s, 9),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a code_1 profile CSV from benchmark_raw.csv files")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    rows = build_rows(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["device", "vendor", "memory_gb", "family", "size", "batch", "dtype", "initial_s", "true_s"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
