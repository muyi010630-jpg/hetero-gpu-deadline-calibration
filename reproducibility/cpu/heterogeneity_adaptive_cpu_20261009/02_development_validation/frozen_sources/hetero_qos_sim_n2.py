#!/usr/bin/env python3
"""Measurement-driven discrete-event simulation for RC-QH-DRF.

The simulator is dependency-free (Python standard library only) so that the
algorithm can be inspected and executed before installing a larger framework.
It models heterogeneous accelerators, contextual service-time errors, drift,
QoS deadlines, user fairness, robust residual calibration and controlled
probing.  Results are written as CSV/JSON for paper plots and statistics.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import simpy  # type: ignore
except ImportError:  # pragma: no cover - used only in offline/minimal environments
    class _Timeout:
        def __init__(self, delay: float):
            self.delay = delay

    class _MiniEnvironment:
        """Small SimPy-compatible subset used when packages cannot be fetched."""
        def __init__(self):
            self.now = 0.0
            self._queue: List[Tuple[float, int, object]] = []
            self._seq = 0

        def timeout(self, delay: float) -> _Timeout:
            return _Timeout(delay)

        def process(self, generator):
            self._seq += 1
            heapq.heappush(self._queue, (self.now, self._seq, generator))
            return generator

        def run(self):
            while self._queue:
                when, _, generator = heapq.heappop(self._queue)
                self.now = when
                try:
                    event = next(generator)
                except StopIteration:
                    continue
                self._seq += 1
                heapq.heappush(self._queue, (self.now + event.delay, self._seq, generator))

    class _MiniSimpy:
        Environment = _MiniEnvironment

    simpy = _MiniSimpy()


Context = Tuple[str, int, int, str]  # family, size, batch, dtype
FAMILY_ALPHA = {"compute": 0.35, "inference": 0.22, "memory": 0.45}


def measured_device_type(name: str) -> str:
    """Return the measured GPU type for a possibly replicated device name."""
    return name.split("__", 1)[0]


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    vendor: str
    capacity: int
    speed: Dict[str, float]
    memory_gb: float


@dataclass
class Task:
    task_id: int
    user: str
    context: Context
    arrival: float
    qos: int
    deadline: float
    true_times: Dict[str, float]
    start: Optional[float] = None
    finish: Optional[float] = None
    device: Optional[str] = None
    predicted: Optional[float] = None
    prediction_bound: Optional[float] = None
    prediction_sigma: Optional[float] = None
    probed: bool = False
    optimal_device: Optional[str] = None
    selection_regret: Optional[float] = None


@dataclass
class DeviceState:
    spec: DeviceSpec
    running: Optional[Task] = None
    busy_until: float = 0.0
    served_work: float = 0.0


@dataclass
class CalibrationCell:
    zhat: float = 0.0
    sigma: float = 0.0
    samples: List[float] = field(default_factory=list)
    updates: int = 0
    last_update: float = 0.0
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    drift_boost_remaining: int = 0
    drift_alarm_count: int = 0
    last_alpha: float = 0.0
    last_effective_z: float = 0.0
    last_effective_sigma: float = 0.0
    last_neighbor_count: int = 0
    last_self_weight: float = 1.0


class OnlineCalibrator:
    """Context/device residual model used by EMA and RC policies."""

    def __init__(self, base: Dict[Tuple[Context, str], float], strategy: str,
                 window: int = 20, ema_rho: float = 0.12,
                 rc_rho: float = 0.20, z_min: float = -0.7,
                 z_max: float = 0.7, min_samples: int = 4,
                 kappa_qos: float = 1.25,
                 hier_tau: float = 4.0,
                 drift_alpha_high: float = 0.65,
                 drift_cusum_h: float = 0.080,
                 drift_cusum_k: float = 0.015,
                 drift_boost_steps: int = 8,
                 risk_beta: float = 1.35,
                 cold_uncertainty: float = 0.30):
        self.base = base
        self.strategy = strategy
        self.window = window
        self.is_rc_family = strategy.startswith("rc")
        self.ema_like_strategies = {
            "ema", "ema_qos", "family_ema_qos", "hier_qos", "ema_risk",
            "full_qos", "full_no_hier", "full_no_drift", "full_no_risk", "drift_qos",
        }
        self.innovation_strategies = {
            "hier_qos", "ema_risk", "drift_qos", "full_qos",
            "full_no_hier", "full_no_drift", "full_no_risk",
        }
        self.updates_enabled = strategy in {*self.ema_like_strategies, "rc", "rc_noprobe", "rc_noguard", "rc_norobust"}
        self.robust_enabled = strategy in {"rc", "rc_noprobe", "rc_noguard"}
        self.guard_enabled = strategy in {
            "ema", "ema_qos", "family_ema_qos", "rc", "rc_noprobe", "rc_norobust",
            "hier_qos", "drift_qos", "full_qos", "full_no_hier", "full_no_drift", "full_no_risk",
        }
        self.hierarchy_enabled = strategy in {"hier_qos", "full_qos", "full_no_drift", "full_no_risk"}
        self.drift_enabled = strategy in {"drift_qos"}
        self.risk_upper_enabled = strategy in {"ema_risk", "full_qos", "full_no_hier", "full_no_drift"}
        self.family_alpha_enabled = strategy in {
            "family_ema_qos", "hier_qos", "ema_risk", "full_qos",
            "full_no_hier", "full_no_drift", "full_no_risk", "drift_qos",
        }
        # RC needs a slightly faster response than the plain EMA baseline;
        # otherwise a regime change is repeatedly rejected as an outlier.
        self.rho = rc_rho if self.is_rc_family else ema_rho
        self.z_min = z_min
        self.z_max = z_max
        self.min_samples = min_samples
        self.kappa_qos = kappa_qos
        self.hier_tau = max(1e-6, hier_tau)
        self.drift_alpha_high = max(self.rho, min(0.95, drift_alpha_high))
        self.drift_cusum_h = max(1e-6, drift_cusum_h)
        self.drift_cusum_k = max(0.0, drift_cusum_k)
        self.drift_boost_steps = max(1, drift_boost_steps)
        self.risk_beta = max(0.0, risk_beta)
        self.cold_uncertainty = max(0.0, cold_uncertainty)
        self.cells: Dict[Tuple[Context, str], CalibrationCell] = {}
        self.prediction_errors: List[float] = []
        self.prediction_abs_pct: List[float] = []
        self.anomaly_count = 0
        self.drift_alarm_count = 0

    def _cell(self, key: Tuple[Context, str]) -> CalibrationCell:
        return self.cells.setdefault(key, CalibrationCell())

    def _alpha_for_context(self, context: Context) -> float:
        return FAMILY_ALPHA.get(context[0], self.rho) if self.family_alpha_enabled else self.rho

    def _neighbor_stats(self, context: Context, device: str) -> Tuple[float, float, int]:
        """Return prior residual information from same-family neighboring cells.

        The lookup uses only already-updated calibration cells, so it is usable
        for cold-start borrowing without looking ahead in the replay stream.
        Same measured GPU type receives full trust; cross-device same-family
        information is retained as a weaker fallback when no sibling exists.
        """
        family, _, _, dtype = context
        exact: List[Tuple[float, float, int]] = []
        fallback: List[Tuple[float, float, int]] = []
        target_type = measured_device_type(device)
        for (ctx, other_device), other in self.cells.items():
            if ctx == context and other_device == device:
                continue
            if other.updates <= 0 or ctx[0] != family:
                continue
            weight = max(1, other.updates)
            item = (other.zhat, max(other.sigma, 1e-6), weight)
            if measured_device_type(other_device) == target_type and ctx[3] == dtype:
                exact.append(item)
            else:
                fallback.append(item)
        source = exact if exact else fallback
        if not source:
            return 0.0, 0.0, 0
        scale = 1.0 if exact else 0.55
        total = sum(w for _, _, w in source)
        mean = sum(z * w for z, _, w in source) / total
        var = sum(w * ((z - mean) ** 2 + sigma ** 2) for z, sigma, w in source) / total
        return mean * scale, math.sqrt(max(0.0, var)) * scale, total

    def _effective_state(self, context: Context, device: str) -> Tuple[float, float, int, float]:
        key = (context, device)
        cell = self._cell(key)
        z = cell.zhat
        sigma = max(cell.sigma, 0.25 / math.sqrt(len(cell.samples) + 1))
        neighbor_count = 0
        self_weight = 1.0
        if self.hierarchy_enabled:
            neighbor_z, neighbor_sigma, neighbor_count = self._neighbor_stats(context, device)
            if neighbor_count > 0 and cell.updates < max(2 * self.min_samples, 2 * self.hier_tau):
                self_weight = cell.updates / (cell.updates + self.hier_tau)
                neighbor_trust = neighbor_count / (neighbor_count + self.hier_tau)
                z = self_weight * cell.zhat + (1.0 - self_weight) * neighbor_trust * neighbor_z
                cold_sigma = self.cold_uncertainty / math.sqrt(cell.updates + 1)
                sigma = max(sigma, (1.0 - self_weight) * neighbor_sigma, cold_sigma)
        cell.last_effective_z = z
        cell.last_effective_sigma = sigma
        cell.last_neighbor_count = neighbor_count
        cell.last_self_weight = self_weight
        return z, sigma, neighbor_count, self_weight

    def predict(self, context: Context, device: str, qos: int,
                guarded: bool = True) -> Tuple[float, float]:
        key = (context, device)
        base = self.base[key]
        cell = self._cell(key)
        if not self.updates_enabled:
            z, sigma = 0.0, 0.0
        elif self.strategy == "oracle":
            # Oracle is replaced by the actual sample by the scheduler.
            z, sigma = cell.zhat, max(cell.sigma, 0.25 / math.sqrt(len(cell.samples) + 1))
        elif self.strategy in self.innovation_strategies:
            z, sigma, _, self_weight = self._effective_state(context, device)
        else:
            z, sigma = cell.zhat, max(cell.sigma, 0.25 / math.sqrt(len(cell.samples) + 1))
            self_weight = 1.0
        guard = 0.0
        if guarded and self.risk_upper_enabled and qos >= 1:
            qos_beta = self.risk_beta * (1.0 + 0.35 * qos)
            cold_bonus = (1.0 - self_weight) * self.cold_uncertainty
            guard = qos_beta * max(sigma, cold_bonus)
        elif guarded and self.guard_enabled and qos >= 2:
            guard = self.kappa_qos * sigma
        return base * math.exp(max(self.z_min, min(self.z_max, z + guard))), sigma

    def observe(self, context: Context, device: str, observed: float,
                predicted: float, now: float) -> None:
        if predicted > 0:
            err = abs(predicted - observed)
            self.prediction_errors.append(err)
            self.prediction_abs_pct.append(err / observed if observed else 0.0)
        if not self.updates_enabled:
            return
        key = (context, device)
        base = self.base[key]
        z = math.log(max(observed, 1e-9) / max(base, 1e-9))
        cell = self._cell(key)
        cell.samples.append(z)
        if len(cell.samples) > self.window:
            cell.samples.pop(0)
        if self.strategy in self.ema_like_strategies:
            series = list(cell.samples)
            if len(series) >= 12:
                recent = series[-4:]
                reference = series[-12:-4]
                reference_sigma = statistics.pstdev(reference) if len(reference) > 1 else 0.0
                innovation = statistics.mean(recent) - statistics.mean(reference)
                adaptive_h = max(self.drift_cusum_h, 1.35 * reference_sigma / math.sqrt(len(recent)))
            else:
                innovation = 0.0
                adaptive_h = self.drift_cusum_h
            alarm = False
            if self.drift_enabled and len(series) >= 12:
                cell.cusum_pos = max(0.0, cell.cusum_pos + innovation - self.drift_cusum_k)
                cell.cusum_neg = min(0.0, cell.cusum_neg + innovation + self.drift_cusum_k)
                if cell.cusum_pos > adaptive_h or -cell.cusum_neg > adaptive_h:
                    alarm = True
                    cell.drift_alarm_count += 1
                    self.drift_alarm_count += 1
                    cell.drift_boost_remaining = self.drift_boost_steps
                    cell.cusum_pos = 0.0
                    cell.cusum_neg = 0.0
            alpha_base = self._alpha_for_context(context)
            alpha = self.drift_alpha_high if (self.drift_enabled and cell.drift_boost_remaining > 0) else alpha_base
            if cell.drift_boost_remaining > 0:
                cell.drift_boost_remaining -= 1
            delta = max(-0.35, min(0.35, z - cell.zhat))
            previous = cell.zhat
            cell.zhat = max(self.z_min, min(self.z_max, cell.zhat + alpha * delta))
            inst_sigma = abs(z - previous)
            cell.sigma = max(0.0, min(0.7, (1.0 - alpha) * cell.sigma + alpha * inst_sigma))
            cell.updates += 1
            cell.last_update = now
            cell.last_alpha = alpha
            self.anomaly_count += int(alarm)
            return
        if len(cell.samples) < self.min_samples:
            return
        sample = list(cell.samples)
        if self.robust_enabled:
            # Use a recent sub-window for the location estimate. The full
            # window remains useful for robust dispersion, but using its
            # median as the target makes sustained drift look like an outlier.
            recent = sample[-max(5, len(sample) // 3):]
            med = statistics.median(recent)
            mad = statistics.median(abs(v - med) for v in recent)
            scale = 1.4826 * mad
            if scale > 1e-9:
                filtered = [v for v in recent if abs(v - med) <= 3.0 * scale]
                self.anomaly_count += int(abs(z - med) > 3.0 * scale)
            else:
                filtered = recent
            # MAD filtering rejects isolated outliers, but the update target
            # uses the filtered mean so sustained drift is not mistaken for
            # noise and ignored for too long.
            target = statistics.mean(filtered) if filtered else med
            sigma = statistics.pstdev(filtered) if len(filtered) > 1 else 0.0
        else:
            target = statistics.mean(sample)
            sigma = statistics.pstdev(sample) if len(sample) > 1 else 0.0
        delta = max(-0.2, min(0.2, target - cell.zhat))
        cell.zhat = max(self.z_min, min(self.z_max, cell.zhat + self.rho * delta))
        cell.sigma = max(0.0, min(0.6, (1.0 - self.rho) * cell.sigma + self.rho * sigma))
        cell.updates += 1
        cell.last_update = now


class Scenario:
    def __init__(self, rng: random.Random, horizon: float = 600.0,
                 arrival_rate: float = 0.22, drift: bool = True,
                 profile_csv: Optional[str] = None, drift_level: float = 1.0,
                 mismatch_level: float = 1.0, strict_qos_prob: float = 0.20,
                 deadline_scale: float = 1.0, runtime_anomaly_rate: float = 0.0,
                 runtime_anomaly_slowdown: float = 2.0,
                 burst_multiplier: float = 2.5,
                 deadline_anchor: str = "best",
                 deadline_best_weight: float = 1.0,
                 deadline_jitter_low: float = 1.0,
                 deadline_jitter_high: float = 1.0,
                 deadline_min_budget_s: float = 0.0,
                 deadline_additive_s: float = 0.0,
                 deadline_qos_factors: Sequence[float] | None = None):
        self.rng = rng
        self.horizon = horizon
        self.arrival_rate = arrival_rate
        self.drift = drift
        self.drift_level = max(0.0, drift_level)
        self.mismatch_level = max(0.0, mismatch_level)
        self.strict_qos_prob = max(0.0, min(0.95, strict_qos_prob))
        self.deadline_scale = max(0.25, deadline_scale)
        self.runtime_anomaly_rate = max(0.0, min(0.5, runtime_anomaly_rate))
        self.runtime_anomaly_slowdown = max(1.0, runtime_anomaly_slowdown)
        self.burst_multiplier = max(1.0, burst_multiplier)
        self.deadline_anchor = deadline_anchor
        self.deadline_best_weight = max(0.0, min(1.0, deadline_best_weight))
        self.deadline_jitter_low = max(0.01, deadline_jitter_low)
        self.deadline_jitter_high = max(self.deadline_jitter_low, deadline_jitter_high)
        self.deadline_min_budget_s = max(0.0, deadline_min_budget_s)
        self.deadline_additive_s = max(0.0, deadline_additive_s)
        self.deadline_qos_factors = tuple(deadline_qos_factors or (5.0, 3.2, 2.0))
        self.devices = [
            DeviceSpec("RTX5070", "NVIDIA", 1, {"compute": 1.0, "memory": 1.0, "inference": 1.0}, 12),
            DeviceSpec("T4", "NVIDIA", 1, {"compute": 0.48, "memory": 0.55, "inference": 0.62}, 16),
            DeviceSpec("MI210", "AMD", 1, {"compute": 0.82, "memory": 1.18, "inference": 0.78}, 64),
            DeviceSpec("L4", "NVIDIA", 1, {"compute": 0.70, "memory": 0.78, "inference": 0.83}, 24),
        ]
        self.contexts: List[Context] = [
            ("compute", 2048, 1, "fp16"), ("compute", 4096, 1, "fp16"),
            ("compute", 4096, 1, "fp32"), ("memory", 256, 1, "fp32"),
            ("memory", 1024, 1, "fp32"), ("inference", 224, 1, "fp16"),
            ("inference", 224, 8, "fp16"), ("inference", 224, 32, "fp16"),
        ]
        if profile_csv:
            self.devices, self.contexts, self.base, self.initial_base = self._load_profile(Path(profile_csv))
        else:
            self.base = self._make_base_times()
            self.initial_base = self._make_initial_times()

    @staticmethod
    def _load_profile(path: Path):
        required = {"device", "vendor", "memory_gb", "family", "size", "batch",
                    "dtype", "initial_s", "true_s"}
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows or not required.issubset(rows[0]):
            missing = required - set(rows[0] if rows else [])
            raise ValueError(f"Invalid profile CSV; missing columns: {sorted(missing)}")
        device_meta: Dict[str, Tuple[str, float]] = {}
        contexts = set()
        base: Dict[Tuple[Context, str], float] = {}
        initial: Dict[Tuple[Context, str], float] = {}
        for row in rows:
            ctx: Context = (row["family"], int(row["size"]), int(row["batch"]), row["dtype"])
            name = row["device"]
            device_meta[name] = (row["vendor"], float(row["memory_gb"]))
            contexts.add(ctx)
            base[(ctx, name)] = float(row["true_s"])
            initial[(ctx, name)] = float(row["initial_s"])
        missing_pairs = [(ctx, name) for ctx in contexts for name in device_meta if (ctx, name) not in base]
        if missing_pairs:
            raise ValueError(f"Profile must contain every context/device pair; missing {len(missing_pairs)} pairs")
        devices = [DeviceSpec(name, vendor, 1, {}, memory)
                   for name, (vendor, memory) in sorted(device_meta.items())]
        return devices, sorted(contexts), base, initial

    @staticmethod
    def memory_required_gb(ctx: Context) -> float:
        family, size, batch, dtype = ctx
        bytes_per = 2 if dtype in {"fp16", "bf16"} else 4
        if family == "compute":
            return 3 * size * size * bytes_per / (1024 ** 3)
        if family == "memory":
            return 3 * size / 1024.0
        return 1.0 + 0.15 * batch

    def _make_base_times(self) -> Dict[Tuple[Context, str], float]:
        base: Dict[Tuple[Context, str], float] = {}
        for ctx in self.contexts:
            family, size, batch, dtype = ctx
            if family == "compute":
                work = (size / 2048.0) ** 3 * (1.0 if dtype == "fp16" else 1.8)
                # Internal time unit is seconds; 0.018 is a nominal 18 ms.
                nominal = 0.018 * work
            elif family == "memory":
                nominal = 0.008 * (size / 256.0)
            else:
                nominal = 0.007 * math.sqrt(batch)
            for d in self.devices:
                base[(ctx, d.name)] = nominal / d.speed[family]
        return base

    def _make_initial_times(self) -> Dict[Tuple[Context, str], float]:
        # The scheduler starts from a deliberately imperfect but plausible
        # profile. Biases are device/family dependent, so device rankings can
        # be wrong even when the nominal hardware table looks reasonable.
        bias = {
            # Deliberate, bounded cold-start mismatch. It changes the
            # predicted best device for several task families, which is the
            # condition under which online calibration can affect scheduling.
            "RTX5070": {"compute": 1.35, "memory": 0.72, "inference": 1.22},
            "T4": {"compute": 0.82, "memory": 1.22, "inference": 0.90},
            "MI210": {"compute": 0.86, "memory": 1.28, "inference": 1.25},
            "L4": {"compute": 1.24, "memory": 0.96, "inference": 0.88},
        }
        return {(ctx, d.name): value * (bias[d.name][ctx[0]] ** self.mismatch_level)
                for (ctx, name), value in self.base.items()
                for d in self.devices if d.name == name}

    def _choose_context(self, phase: float) -> Context:
        groups: Dict[str, List[Context]] = {"compute": [], "memory": [], "inference": []}
        for ctx in self.contexts:
            groups[ctx[0]].append(ctx)
        if phase < 0.35:
            family = self.rng.choices(["compute", "inference", "memory"], weights=[0.50, 0.30, 0.20])[0]
        elif phase < 0.70:
            family = self.rng.choices(["compute", "inference", "memory"], weights=[0.25, 0.55, 0.20])[0]
        else:
            family = self.rng.choices(["compute", "inference", "memory"], weights=[0.40, 0.25, 0.35])[0]
        candidates = groups.get(family) or self.contexts
        return self.rng.choice(candidates)

    def generate_tasks(self) -> List[Task]:
        tasks: List[Task] = []
        t, i = 0.0, 0
        users = ["u0", "u1", "u2", "u3"]
        while t < self.horizon:
            phase = t / self.horizon
            burst = self.burst_multiplier if (0.25 <= phase < 0.35 or 0.72 <= phase < 0.80) else 1.0
            t += self.rng.expovariate(self.arrival_rate * burst)
            if t >= self.horizon:
                break
            ctx = self._choose_context(phase)
            non_strict = 1.0 - self.strict_qos_prob
            qos0 = non_strict * (0.45 / 0.80)
            qos1 = non_strict * (0.35 / 0.80)
            qos = self.rng.choices([0, 1, 2], weights=[qos0, qos1, self.strict_qos_prob])[0]
            # QoS classes have progressively tighter relative deadlines.
            # The default `best` anchor preserves the original experiment C
            # rule. Alternative anchors are used only by deadline sensitivity
            # reruns, where deadlines should not be determined solely by the
            # fastest measured device for a context.
            base_times = [self.base[(ctx, d.name)] for d in self.devices]
            best = min(base_times)
            median = statistics.median(base_times)
            if self.deadline_anchor == "median":
                nominal = median
            elif self.deadline_anchor == "mix":
                w = self.deadline_best_weight
                nominal = w * best + (1.0 - w) * median
            else:
                nominal = best
            jitter = self.rng.uniform(self.deadline_jitter_low, self.deadline_jitter_high)
            factor = self.deadline_qos_factors[min(qos, len(self.deadline_qos_factors) - 1)]
            budget = max(self.deadline_min_budget_s, nominal * factor * self.deadline_scale * jitter + self.deadline_additive_s)
            deadline = t + budget
            true: Dict[str, float] = {}
            for d in self.devices:
                measured_type = measured_device_type(d.name)
                noise = self.rng.lognormvariate(0.0, 0.08)
                drift_factor = 1.0
                if self.drift and t > self.horizon * 0.48:
                    if measured_type in {"T4", "L4"}:
                        drift_factor *= 1.0 + 0.18 * self.drift_level
                    if measured_type == "RTX5070" and ctx[0] == "memory":
                        drift_factor *= max(0.55, 1.0 - 0.10 * self.drift_level)
                if self.drift and self.horizon * 0.70 < t < self.horizon * 0.82 and measured_type == "MI210":
                    drift_factor *= 1.0 + 0.30 * self.drift_level
                if self.runtime_anomaly_rate and self.rng.random() < self.runtime_anomaly_rate:
                    drift_factor *= self.runtime_anomaly_slowdown
                true[d.name] = self.base[(ctx, d.name)] * noise * drift_factor
            user = self.rng.choices(users, weights=[0.40, 0.30, 0.20, 0.10])[0]
            tasks.append(Task(i, user, ctx, t, qos, deadline, true))
            i += 1
        return tasks


class EventSimulator:
    def __init__(self, scenario: Scenario, strategy: str, seed: int,
                 probe_rate: float = 0.03, cal_window: int = 20,
                 min_samples: int = 4, ema_rho: float = 0.12,
                 rc_rho: float = 0.20, kappa_qos: float = 1.25,
                 priority_risk_weight: float = 1.0,
                 fairness_weight: float = 0.5,
                 short_job_weight: float = 0.22,
                 probe_max_slowdown: float = 1.20,
                 hier_tau: float = 4.0,
                 drift_alpha_high: float = 0.65,
                 drift_cusum_h: float = 0.080,
                 drift_cusum_k: float = 0.015,
                 drift_boost_steps: int = 8,
                 risk_beta: float = 1.35,
                 cold_uncertainty: float = 0.30):
        self.scenario = scenario
        self.strategy = strategy
        self.seed = seed
        self.rng = random.Random(seed + 991)
        self.tasks = scenario.generate_tasks()
        self.users = sorted({task.user for task in self.tasks})
        self.devices = {d.name: DeviceState(d) for d in scenario.devices}
        self.cal = OnlineCalibrator(
            scenario.initial_base, strategy, window=cal_window,
            ema_rho=ema_rho, rc_rho=rc_rho, min_samples=min_samples,
            kappa_qos=kappa_qos, hier_tau=hier_tau,
            drift_alpha_high=drift_alpha_high,
            drift_cusum_h=drift_cusum_h,
            drift_cusum_k=drift_cusum_k,
            drift_boost_steps=drift_boost_steps,
            risk_beta=risk_beta,
            cold_uncertainty=cold_uncertainty,
        )
        self.env = simpy.Environment()
        self.queue: List[Task] = []
        self.now = 0.0
        self.completed: List[Task] = []
        self.probe_rate = 0.0 if strategy == "rc_noprobe" else probe_rate
        self.is_rc_family = strategy.startswith("rc")
        self.priority_risk_weight = priority_risk_weight
        self.fairness_weight = fairness_weight
        self.short_job_weight = short_job_weight
        self.probe_max_slowdown = max(1.0, probe_max_slowdown)
        self.user_work: Dict[str, float] = {}
        self.total_work = 0.0
        self.trace: List[Dict[str, object]] = []
        self.bound_hits = 0
        self.bound_total = 0
        self.false_safe_count = 0
        self.predicted_safe_count = 0
        self.selection_regrets: List[float] = []
        self.optimal_selections = 0
        self.selection_count = 0

    @staticmethod
    def _pct(values: Sequence[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
        return ordered[idx]

    @staticmethod
    def _jain(values: Sequence[float]) -> float:
        if not values:
            return 1.0
        denom = sum(v * v for v in values)
        return (sum(values) ** 2 / (len(values) * denom)) if denom else 0.0

    def _dynamic_fairness_metrics(self) -> Dict[str, float]:
        users = self.users or sorted({task.user for task in self.completed})
        if not users:
            return {
                "window_jain_fairness_mean": 1.0,
                "window_jain_fairness_min": 1.0,
                "user_slowdown_mean": 0.0,
                "user_slowdown_max": 0.0,
                "user_p95_wait_max": 0.0,
                "max_wait_s": 0.0,
                "min_user_service_share": 0.0,
            }
        horizon = max(self.now, max((t.finish or 0.0) for t in self.completed), 1e-9)
        window = max(1.0, horizon / 10.0)
        jains: List[float] = []
        start = 0.0
        while start < horizon:
            end = min(horizon, start + window)
            active = [u for u in users if any(t.user == u and t.arrival <= end and (t.finish or horizon) >= start for t in self.tasks)]
            if active:
                served = []
                for user in active:
                    work = sum(1.0 + 0.3 * t.qos for t in self.completed if t.user == user and t.finish is not None and start <= t.finish < end)
                    served.append(work)
                jains.append(self._jain(served))
            start = end
        waits_by_user: Dict[str, List[float]] = {u: [] for u in users}
        slowdowns_by_user: Dict[str, List[float]] = {u: [] for u in users}
        service_by_user: Dict[str, float] = {u: 0.0 for u in users}
        for task in self.completed:
            if task.start is None or task.finish is None or task.device is None:
                continue
            wait = max(0.0, task.start - task.arrival)
            service = max(task.true_times[task.device], 1e-9)
            flow = max(0.0, task.finish - task.arrival)
            waits_by_user[task.user].append(wait)
            slowdowns_by_user[task.user].append(flow / service)
            service_by_user[task.user] += 1.0 + 0.3 * task.qos
        p95_waits = [self._pct(vals, 0.95) for vals in waits_by_user.values() if vals]
        mean_slowdowns = [statistics.mean(vals) for vals in slowdowns_by_user.values() if vals]
        total_service = max(sum(service_by_user.values()), 1e-9)
        active_service_shares = [service_by_user[u] / total_service for u in users if any(t.user == u for t in self.tasks)]
        return {
            "window_jain_fairness_mean": statistics.mean(jains) if jains else 1.0,
            "window_jain_fairness_min": min(jains) if jains else 1.0,
            "user_slowdown_mean": statistics.mean(mean_slowdowns) if mean_slowdowns else 0.0,
            "user_slowdown_max": max(mean_slowdowns) if mean_slowdowns else 0.0,
            "user_p95_wait_max": max(p95_waits) if p95_waits else 0.0,
            "max_wait_s": max((w for vals in waits_by_user.values() for w in vals), default=0.0),
            "min_user_service_share": min(active_service_shares) if active_service_shares else 0.0,
        }

    def _compatible(self, task: Task, devices: Sequence[DeviceState]) -> List[DeviceState]:
        return [d for d in devices
                if self.scenario.memory_required_gb(task.context) <= d.spec.memory_gb]

    def _best_predicted_time(self, task: Task, devices: Sequence[DeviceState]) -> float:
        compatible = self._compatible(task, devices)
        if not compatible:
            return float("inf")
        if self.strategy == "oracle":
            return min(task.true_times[d.spec.name] for d in compatible)
        return min(self.cal.predict(task.context, d.spec.name, task.qos)[0] for d in compatible)

    def _mean_prediction(self, task: Task, device: DeviceState) -> Tuple[float, float]:
        if self.strategy == "oracle":
            return task.true_times[device.spec.name], 0.0
        return self.cal.predict(task.context, device.spec.name, task.qos, guarded=False)

    def _guarded_prediction(self, task: Task, device: DeviceState) -> Tuple[float, float]:
        if self.strategy == "oracle":
            return task.true_times[device.spec.name], 0.0
        return self.cal.predict(task.context, device.spec.name, task.qos, guarded=True)

    def _urgency_terms(self, task: Task, best_pred: float) -> Tuple[float, float, float, float]:
        budget = max(task.deadline - task.arrival, 1e-6)
        elapsed = max(0.0, self.env.now - task.arrival)
        completion_ratio = max(0.0, self.env.now + best_pred - task.arrival) / budget
        remaining_slack = task.deadline - self.env.now - best_pred
        lateness = max(0.0, -remaining_slack / budget)
        warning_start = [0.90, 0.78, 0.62][task.qos]
        risk = max(0.0, (completion_ratio - warning_start) / max(1e-6, 1.0 - warning_start))
        return min(2.5, risk), lateness, best_pred / budget, min(1.0, elapsed / budget)

    def _score(self, task: Task, device: DeviceState) -> Tuple[float, float]:
        predicted, sigma = self._guarded_prediction(task, device)
        available = max(self.env.now, device.busy_until)
        completion = available + predicted
        slack = task.deadline - completion
        user = task.user
        # Dominant-share fairness deficit, normalized by completed work.
        shares = [v / max(self.total_work, 1e-9) for v in self.user_work.values()]
        min_share = min(shares, default=0.0)
        own_share = self.user_work.get(user, 0.0) / max(self.total_work, 1e-9)
        deficit = max(0.0, min_share + 0.25 - own_share)
        budget = max(task.deadline - task.arrival, 1e-6)
        normalized_completion = max(0.0, completion - task.arrival) / budget
        qos_weight = 1.0 + 0.8 * task.qos
        if self.strategy == "aqh":
            # Adaptive-QH baseline: increase deadline pressure as the
            # normalized slack becomes negative, without online calibration.
            urgency = max(0.0, min(1.0, -slack / max(task.deadline - task.arrival, 1e-9)))
            qos_weight += 0.6 * urgency * task.qos
        # Device choice must have an observable effect. The service-time term
        # is normalized by this task's deadline budget instead of being a
        # small absolute coefficient (which previously made calibration
        # almost irrelevant under queue pressure).
        score = (
            qos_weight * max(0.0, -slack) / budget
            + 0.9 * normalized_completion
            - 0.6 * deficit
        )
        if self.strategy == "fcfs":
            score = task.arrival
        elif self.strategy == "edf":
            score = task.deadline
        # `predict` already applies the uncertainty guard for strict QoS
        # tasks. Keep the raw selected prediction here for error accounting.
        return score, predicted

    def _task_priority(self, task: Task, idle: Sequence[DeviceState]) -> float:
        """QoS/fairness priority with prediction-aware deadline risk.

        The first implementation used calibrated predictions only for device
        selection. That made RC improve model diagnostics but muted its effect
        on SLA and mean completion time. This priority brings predicted
        completion risk into queue ordering while keeping fairness bounded.
        """
        compatible = self._compatible(task, idle)
        if not compatible:
            return float("inf")
        best_pred = self._best_predicted_time(task, idle)
        risk, lateness, service_ratio, age = self._urgency_terms(task, best_pred)
        shares = [v / max(self.total_work, 1e-9) for v in self.user_work.values()]
        min_share = min(shares, default=0.0)
        own_share = self.user_work.get(task.user, 0.0) / max(self.total_work, 1e-9)
        deficit = max(0.0, min_share + 0.25 - own_share)
        if self.strategy == "fcfs":
            return task.arrival
        if self.strategy == "edf":
            return task.deadline
        # Lower is earlier. Strict QoS and lateness are dominant; fairness is
        # a bounded tie-breaker. The short-job term is deliberately small; it
        # lets better predictions improve mean completion without starving
        # long-running tasks.
        risk_weight = self.priority_risk_weight * (1.0 + 0.35 * task.qos)
        aging_bonus = 0.22 * age
        if self.strategy == "aqh":
            # AQH is an adaptive fixed-model baseline: it reacts to aging and
            # predicted risk but does not update the service model online.
            risk_weight *= 1.20
            aging_bonus = 0.35 * age
        return (
            -2.45 * task.qos
            - risk_weight * risk
            - 1.15 * lateness
            - self.fairness_weight * deficit
            - aging_bonus
            + self.short_job_weight * service_ratio
            + task.arrival * 1e-6
        )

    def _select_device(self, task: Task, idle: Sequence[DeviceState]) -> Tuple[DeviceState, float]:
        compatible = self._compatible(task, idle)
        # NOTE(repair-2, oracle symmetry fix): the oracle branch previously used
        # a separate rule `dev = min(compatible, key=now + true_times[d])` that
        # omitted the load-balancing term `0.08 * load` applied to every other
        # strategy. Replicated devices of the same measured type share identical
        # service times, so ties were always broken by device name and all tasks
        # of a type piled onto the first replica. That inflated Oracle queueing
        # alone and made the perfect-information reference the *worst* strategy
        # on SLA violation. Oracle is defined as "identical scheduler, perfect
        # service-time information", so it must follow the same scoring path;
        # `_mean_prediction` already returns true_times when strategy == oracle.
        scored = []
        budget = max(task.deadline - task.arrival, 1e-6)
        for d in compatible:
            pred, _ = self._mean_prediction(task, d)
            projected = max(0.0, self.env.now + pred - task.arrival) / budget
            miss = max(0.0, (self.env.now + pred - task.deadline) / budget)
            # Mild load-balancing term avoids sending every task to one device
            # when predicted times are nearly tied.
            load = d.served_work / max(self.total_work, 1.0)
            scored.append((projected + 0.65 * miss + 0.08 * load, d, pred))
        _, dev, pred = min(scored, key=lambda x: (x[0], x[1].spec.name))
        return dev, pred

    def _urgent_strict_waiting(self, tasks: Sequence[Task], idle: Sequence[DeviceState]) -> bool:
        for task in tasks:
            if task.qos < 2:
                continue
            best_pred = self._best_predicted_time(task, idle)
            if not math.isfinite(best_pred):
                continue
            risk, lateness, _, _ = self._urgency_terms(task, best_pred)
            if lateness > 0.0 or risk >= 0.75:
                return True
        return False

    def _select_probe(self, tasks: Sequence[Task], idle: Sequence[DeviceState]) -> Optional[Tuple[float, Task, DeviceState, float]]:
        probe_candidates: List[Tuple[int, float, float, int, str, float, Task, DeviceState, float]] = []
        for task in tasks:
            if task.qos >= 2:
                continue
            best_pred = self._best_predicted_time(task, idle)
            if not math.isfinite(best_pred):
                continue
            task_priority = self._task_priority(task, idle)
            for device in self._compatible(task, idle):
                pred, sigma = self._mean_prediction(task, device)
                if pred > best_pred * self.probe_max_slowdown:
                    continue
                if task.qos == 1 and self.env.now + pred > task.deadline:
                    continue
                cell = self.cal._cell((task.context, device.spec.name))
                probe_candidates.append((
                    len(cell.samples),
                    -sigma,
                    task_priority,
                    task.task_id,
                    device.spec.name,
                    task_priority,
                    task,
                    device,
                    pred,
                ))
        if not probe_candidates:
            return None
        _, _, _, _, _, score, task, device, pred = min(probe_candidates)
        return score, task, device, pred

    def dispatch(self) -> None:
        self.now = self.env.now
        while self.queue:
            # A queue may start with a task that does not fit on a particular
            # idle device; compatibility is enforced per candidate below.
            all_idle = [d for d in self.devices.values() if d.running is None and d.spec.capacity > 0]
            idle = all_idle
            if not idle:
                return
            known = {"fixed", "aqh", "ema", "ema_qos", "family_ema_qos", "hier_qos", "ema_risk", "drift_qos",
                     "full_qos", "full_no_hier", "full_no_drift", "full_no_risk",
                     "rc", "rc_noprobe", "rc_noguard", "rc_norobust",
                     "oracle", "fcfs", "edf"}
            if self.strategy in known:
                feasible = [t for t in self.queue if any(
                    self.scenario.memory_required_gb(t.context) <= d.spec.memory_gb for d in idle)]
                if not feasible:
                    return
                probe = None
                if (self.is_rc_family and self.probe_rate > 0.0
                        and self.rng.random() < self.probe_rate
                        and not self._urgent_strict_waiting(feasible, idle)):
                    probe = self._select_probe(feasible, idle)
                if probe is not None:
                    score, task, dev, pred = probe
                    task.probed = True
                else:
                    task = min(feasible, key=lambda t: (self._task_priority(t, idle), t.task_id))
                    dev, pred = self._select_device(task, idle)
                    score = self._task_priority(task, idle)
                candidates = [(score, task, dev, pred)]
            else:
                candidates = []
            candidates = [(self._score(t, d)[0], t, d, self._score(t, d)[1])
                          for t in self.queue for d in idle
                          if self.scenario.memory_required_gb(t.context) <= d.spec.memory_gb] if not candidates else candidates
            if not candidates:
                return
            score, task, dev, pred = min(candidates, key=lambda x: (x[0], x[1].task_id, x[2].spec.name))
            self.queue.remove(task)
            task.start = self.now
            task.device = dev.spec.name
            task.predicted = pred
            bound, sigma = self._guarded_prediction(task, dev)
            task.prediction_bound = bound
            task.prediction_sigma = sigma
            compatible = [d for d in self.devices.values()
                          if self.scenario.memory_required_gb(task.context) <= d.spec.memory_gb]
            best = min(compatible, key=lambda d: task.true_times[d.spec.name])
            task.optimal_device = best.spec.name
            task.selection_regret = task.true_times[dev.spec.name] / task.true_times[best.spec.name]
            self.selection_regrets.append(task.selection_regret)
            self.selection_count += 1
            self.optimal_selections += int(dev.spec.name == best.spec.name)
            duration = task.true_times[dev.spec.name]
            task.finish = self.now + duration
            dev.running = task
            dev.busy_until = task.finish
            self.env.process(self._complete_after(dev, task, duration))

    def _complete_after(self, device: DeviceState, task: Task, duration: float):
        yield self.env.timeout(duration)
        self.now = self.env.now
        observed = task.true_times[device.spec.name]
        self.cal.observe(task.context, device.spec.name, observed,
                         task.predicted or observed, self.env.now)
        if task.qos >= 2 and task.prediction_bound is not None:
            self.bound_total += 1
            self.bound_hits += int(observed <= task.prediction_bound)
        predicted_safe = bool(
            task.start is not None
            and task.prediction_bound is not None
            and task.start + task.prediction_bound <= task.deadline
        )
        false_safe = bool(predicted_safe and task.finish is not None and task.finish > task.deadline)
        self.predicted_safe_count += int(predicted_safe)
        self.false_safe_count += int(false_safe)
        cell = self.cal._cell((task.context, device.spec.name))
        self.trace.append({
            "seed": self.seed, "strategy": self.strategy,
            "task_id": task.task_id, "user": task.user,
            "family": task.context[0], "size": task.context[1],
            "batch": task.context[2], "dtype": task.context[3],
            "qos": task.qos, "arrival": task.arrival,
            "start": task.start, "finish": task.finish,
            "device": device.spec.name, "predicted_s": task.predicted,
            "prediction_bound_s": task.prediction_bound,
            "observed_s": observed, "prediction_sigma": task.prediction_sigma,
            "deadline": task.deadline,
            "sla_violation": int(bool(task.finish and task.finish > task.deadline)),
            "predicted_safe": int(predicted_safe),
            "false_safe": int(false_safe),
            "probed": int(task.probed),
            "optimal_device": task.optimal_device,
            "selection_regret": task.selection_regret,
            "calibration_updates": cell.updates,
            "drift_alarm_count": cell.drift_alarm_count,
            "last_alpha": cell.last_alpha,
            "hier_neighbor_count": cell.last_neighbor_count,
            "hier_self_weight": cell.last_self_weight,
        })
        work = 1.0 + 0.3 * task.qos
        device.served_work += work
        self.user_work[task.user] = self.user_work.get(task.user, 0.0) + work
        self.total_work += work
        self.completed.append(task)
        device.running = None
        device.busy_until = self.env.now
        self.dispatch()

    def _arrival_process(self):
        previous = 0.0
        for task in self.tasks:
            yield self.env.timeout(max(0.0, task.arrival - previous))
            previous = task.arrival
            self.queue.append(task)
            self.dispatch()

    def run(self) -> Dict[str, object]:
        self.env.process(self._arrival_process())
        self.env.run()
        self.now = self.env.now
        return self.summary()

    def summary(self) -> Dict[str, object]:
        if not self.completed:
            return {"strategy": self.strategy, "seed": self.seed, "completed": 0}
        comp = [t.finish - t.arrival for t in self.completed if t.finish is not None]
        wait = [t.start - t.arrival for t in self.completed if t.start is not None]
        violations = [t for t in self.completed if t.finish and t.finish > t.deadline]
        strict = [t for t in violations if t.qos >= 2]
        shares = list(self.user_work.values())
        denom = sum(v * v for v in shares)
        jain = (sum(shares) ** 2 / (len(shares) * denom)) if shares and denom else 1.0
        fairness = self._dynamic_fairness_metrics()
        def pct(values: Sequence[float], p: float) -> float:
            if not values:
                return 0.0
            a = sorted(values)
            idx = min(len(a) - 1, max(0, math.ceil(p * len(a)) - 1))
            return a[idx]
        return {
            "strategy": self.strategy, "seed": self.seed,
            "arrived": len(self.tasks), "completed": len(self.completed),
            "mean_completion": statistics.mean(comp), "p95_completion": pct(comp, .95),
            "p99_completion": pct(comp, .99), "mean_wait": statistics.mean(wait),
            "sla_violation_rate": len(violations) / len(self.completed),
            "strict_qos_violation_rate": len(strict) / max(1, sum(t.qos >= 2 for t in self.completed)),
            "throughput": len(self.completed) / max(self.now, 1e-9), "jain_fairness": jain,
            **fairness,
            "prediction_mae": statistics.mean(self.cal.prediction_errors) if self.cal.prediction_errors else 0.0,
            "prediction_mape": statistics.mean(self.cal.prediction_abs_pct) if self.cal.prediction_abs_pct else 0.0,
            "qos_bound_coverage": self.bound_hits / self.bound_total if self.bound_total else 0.0,
            "false_safe_rate": self.false_safe_count / self.predicted_safe_count if self.predicted_safe_count else 0.0,
            "false_safe_all_rate": self.false_safe_count / len(self.completed),
            "predicted_safe_rate": self.predicted_safe_count / len(self.completed),
            "anomaly_count": self.cal.anomaly_count,
            "drift_alarm_count": self.cal.drift_alarm_count,
            "probe_count": sum(t.probed for t in self.completed),
            "device_regret": statistics.mean(self.selection_regrets) if self.selection_regrets else 0.0,
            "optimal_device_rate": self.optimal_selections / self.selection_count if self.selection_count else 0.0,
        }


def run_experiments(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    rows: List[Dict[str, object]] = []
    traces: List[Dict[str, object]] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for strategy in strategies:
            scenario = Scenario(random.Random(seed), args.horizon, args.arrival_rate,
                                not args.no_drift, getattr(args, "profile_csv", None),
                                getattr(args, "drift_level", 1.0),
                                getattr(args, "mismatch_level", 1.0),
                                getattr(args, "strict_qos_prob", 0.20),
                                getattr(args, "deadline_scale", 1.0),
                                getattr(args, "runtime_anomaly_rate", 0.0),
                                getattr(args, "runtime_anomaly_slowdown", 2.0),
                                getattr(args, "burst_multiplier", 2.5),
                                getattr(args, "deadline_anchor", "best"),
                                getattr(args, "deadline_best_weight", 1.0),
                                getattr(args, "deadline_jitter_low", 1.0),
                                getattr(args, "deadline_jitter_high", 1.0),
                                getattr(args, "deadline_min_budget_s", 0.0),
                                getattr(args, "deadline_additive_s", 0.0),
                                getattr(args, "deadline_qos_factors", None))
            simulator = EventSimulator(
                scenario, strategy, seed, args.probe_rate,
                getattr(args, "cal_window", 20),
                getattr(args, "min_samples", 4),
                getattr(args, "ema_rho", 0.12),
                getattr(args, "rc_rho", 0.20),
                getattr(args, "kappa_qos", 1.25),
                getattr(args, "priority_risk_weight", 1.0),
                getattr(args, "fairness_weight", 0.5),
                getattr(args, "short_job_weight", 0.22),
                getattr(args, "probe_max_slowdown", 1.20),
                getattr(args, "hier_tau", 4.0),
                getattr(args, "drift_alpha_high", 0.65),
                getattr(args, "drift_cusum_h", 0.080),
                getattr(args, "drift_cusum_k", 0.015),
                getattr(args, "drift_boost_steps", 8),
                getattr(args, "risk_beta", 1.35),
                getattr(args, "cold_uncertainty", 0.30),
            )
            result = simulator.run()
            rows.append(result)
            traces.extend(simulator.trace)
    grouped: List[Dict[str, object]] = []
    for strategy in strategies:
        subset = [r for r in rows if r["strategy"] == strategy]
        numeric = [k for k, v in subset[0].items() if isinstance(v, (int, float)) and k not in {"seed"}]
        out: Dict[str, object] = {"strategy": strategy, "seeds": len(subset)}
        for key in numeric:
            vals = [float(r[key]) for r in subset]
            out[key + "_mean"] = statistics.mean(vals)
            out[key + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        grouped.append(out)
    return rows, grouped, traces


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Discrete-event heterogeneous accelerator QoS-fair scheduling simulation")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--horizon", type=float, default=120.0)
    p.add_argument("--arrival-rate", type=float, default=18.0)
    p.add_argument("--probe-rate", type=float, default=0.03)
    p.add_argument("--no-drift", action="store_true")
    p.add_argument("--drift-level", type=float, default=1.0,
                   help="Multiplier for the piecewise performance drift")
    p.add_argument("--mismatch-level", type=float, default=1.0,
                   help="Exponent applied to cold-start model bias; >1 stresses initial mismatch")
    p.add_argument("--strict-qos-prob", type=float, default=0.20,
                   help="Probability that a generated task uses the strict QoS class")
    p.add_argument("--deadline-scale", type=float, default=1.0,
                   help="Multiplier for all generated task deadline budgets")
    p.add_argument("--deadline-anchor", choices=["best", "median", "mix"], default="best",
                   help="Service-time anchor used to generate deadline budgets")
    p.add_argument("--deadline-best-weight", type=float, default=1.0,
                   help="Best-device weight used when --deadline-anchor=mix")
    p.add_argument("--deadline-jitter-low", type=float, default=1.0,
                   help="Lower uniform jitter multiplier for generated deadline budgets")
    p.add_argument("--deadline-jitter-high", type=float, default=1.0,
                   help="Upper uniform jitter multiplier for generated deadline budgets")
    p.add_argument("--deadline-min-budget-s", type=float, default=0.0,
                   help="Minimum generated deadline budget in seconds")
    p.add_argument("--deadline-additive-s", type=float, default=0.0,
                   help="Additive slack in seconds for generated deadline budgets")
    p.add_argument("--runtime-anomaly-rate", type=float, default=0.0,
                   help="Probability of an occasional per-device service-time slowdown")
    p.add_argument("--runtime-anomaly-slowdown", type=float, default=2.0,
                   help="Slowdown multiplier for injected runtime anomalies")
    p.add_argument("--burst-multiplier", type=float, default=2.5,
                   help="Arrival-rate multiplier during burst intervals")
    p.add_argument("--cal-window", type=int, default=20,
                   help="Residual samples retained per context/device calibration cell")
    p.add_argument("--min-samples", type=int, default=4,
                   help="Minimum samples before a calibration cell can update")
    p.add_argument("--ema-rho", type=float, default=0.12,
                   help="EMA baseline residual update rate")
    p.add_argument("--rc-rho", type=float, default=0.20,
                   help="RC residual update rate")
    p.add_argument("--kappa-qos", type=float, default=1.25,
                   help="Uncertainty guard multiplier for strict QoS predictions")
    p.add_argument("--priority-risk-weight", type=float, default=1.0,
                   help="Weight of prediction-aware deadline risk in queue ordering")
    p.add_argument("--fairness-weight", type=float, default=0.5,
                   help="Bounded fairness-deficit weight in queue ordering")
    p.add_argument("--short-job-weight", type=float, default=0.22,
                   help="Small positive service-ratio penalty favoring shorter predicted work")
    p.add_argument("--probe-max-slowdown", type=float, default=1.20,
                   help="Maximum predicted slowdown allowed for RC probing relative to the best device")
    p.add_argument("--hier-tau", type=float, default=4.0,
                   help="Shrinkage strength for hierarchical cold-start borrowing")
    p.add_argument("--drift-alpha-high", type=float, default=0.65,
                   help="Temporary EMA update rate after a residual drift alarm")
    p.add_argument("--drift-cusum-h", type=float, default=0.080,
                   help="CUSUM threshold for residual mean-shift detection")
    p.add_argument("--drift-cusum-k", type=float, default=0.015,
                   help="CUSUM slack parameter for residual mean-shift detection")
    p.add_argument("--drift-boost-steps", type=int, default=8,
                   help="Number of observations using the high alpha after a drift alarm")
    p.add_argument("--risk-beta", type=float, default=1.35,
                   help="Risk upper-bound multiplier for uncertainty-aware QoS scheduling")
    p.add_argument("--cold-uncertainty", type=float, default=0.30,
                   help="Prior uncertainty used for low-sample hierarchical cells")
    p.add_argument("--strategies", default="fcfs,edf,fixed,aqh,ema,rc,oracle")
    p.add_argument("--profile-csv", default=None,
                   help="Optional measured profile CSV; see README for schema")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = p.parse_args()
    raw, summary, traces = run_experiments(args)
    out = Path(args.out_dir)
    write_csv(out / "per_seed_results.csv", raw)
    write_csv(out / "summary_results.csv", summary)
    write_csv(out / "task_trace.csv", traces)
    config = vars(args) | {"devices": ["RTX5070", "T4", "MI210", "L4"], "contexts": 8}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results written to: {out}")


if __name__ == "__main__":
    main()
