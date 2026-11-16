from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Sequence
ContextKey = Tuple[str, int, int, str]

@dataclass
class OnlineCell:
    z_hat: float = 0.0
    updates: int = 0
    residual_logs: List[float] = field(default_factory=list)

def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, math.ceil(len(arr) * p) - 1))
    return arr[idx]

class OnlineCalibrator:
    def __init__(
        self,
        strategy: str,
        base: Dict[Tuple[str, ContextKey], float],
        alpha: float = 0.30,
        window: int = 24,
        min_samples: int = 4,
        hier_tau: float = 1.5,
        risk_beta: float = 0.80,
        cold_uncertainty: float = 0.30,
        fixed_safety: float = 1.10,
        ema_kappa: float = 1.30,
    ) -> None:
        self.strategy = strategy
        self.base = base
        self.alpha = alpha
        self.window = window
        self.min_samples = min_samples
        self.hier_tau = hier_tau
        self.risk_beta = risk_beta
        self.cold_uncertainty = cold_uncertainty
        self.fixed_safety = fixed_safety
        self.ema_kappa = ema_kappa
        self.cells: Dict[Tuple[str, ContextKey], OnlineCell] = {}
        self.family_cells: Dict[Tuple[str, str], OnlineCell] = {}
        self.worker_cells: Dict[str, OnlineCell] = {}

    def _cell(self, worker_id: str, key: ContextKey) -> OnlineCell:
        return self.cells.setdefault((worker_id, key), OnlineCell())

    def _family_cell(self, worker_id: str, family: str) -> OnlineCell:
        return self.family_cells.setdefault((worker_id, family), OnlineCell())

    def _worker_cell(self, worker_id: str) -> OnlineCell:
        return self.worker_cells.setdefault(worker_id, OnlineCell())

    def fixed(self, worker_id: str, key: ContextKey) -> float:
        return self.base[(worker_id, key)]

    def _context_distance(self, a: ContextKey, b: ContextKey) -> float:
        if a[0] != b[0]:
            return 99.0
        family = a[0]
        dist = 0.0
        if family in {"compute", "memory"}:
            dist += abs(math.log2(max(a[1], 1) / max(b[1], 1)))
        if family == "inference":
            dist += abs(math.log2(max(a[2], 1) / max(b[2], 1)))
        if a[3] != b[3]:
            dist += 0.65
        return dist

    def _neighbor_z(self, worker_id: str, key: ContextKey) -> Tuple[float, float, int]:
        weighted_z = 0.0
        total_w = 0.0
        total_n = 0
        for (wid, other_key), cell in self.cells.items():
            if wid != worker_id or other_key == key or cell.updates <= 0:
                continue
            dist = self._context_distance(key, other_key)
            if dist > self.hier_tau:
                continue
            w = math.exp(-dist / max(self.hier_tau, 1e-9)) * min(cell.updates, self.min_samples)
            weighted_z += w * cell.z_hat
            total_w += w
            total_n += cell.updates
        if total_w <= 0:
            return 0.0, 0.0, 0
        return weighted_z / total_w, total_w, total_n

    def _full_z(self, worker_id: str, key: ContextKey) -> Tuple[float, str, int]:
        direct = self._cell(worker_id, key)
        fam = self._family_cell(worker_id, key[0])
        worker = self._worker_cell(worker_id)
        neighbor_z, neighbor_w, neighbor_n = self._neighbor_z(worker_id, key)
        pieces: List[Tuple[float, float, str]] = [(0.0, 1.0, "prior")]
        if direct.updates > 0:
            pieces.append((direct.z_hat, min(direct.updates, self.min_samples) + 0.5, "direct"))
        if fam.updates > 0:
            pieces.append((fam.z_hat, min(fam.updates, self.min_samples) * 0.75, "family"))
        if neighbor_w > 0:
            pieces.append((neighbor_z, min(neighbor_w, self.min_samples) * 0.55, "neighbor"))
        if worker.updates > 0:
            pieces.append((worker.z_hat, min(worker.updates, self.min_samples) * 0.20, "worker"))
        if direct.updates >= self.min_samples:
            pieces = [(direct.z_hat, 4.0, "direct"), (fam.z_hat, 0.85, "family"), (neighbor_z, 0.35, "neighbor")]
            pieces = [(z, w, name) for z, w, name in pieces if w > 0 and not math.isnan(z)]
        total_w = sum(w for _, w, _ in pieces)
        z = sum(z * w for z, w, _ in pieces) / max(total_w, 1e-9)
        names = "+".join(name for _, _, name in pieces)
        support_n = direct.updates + fam.updates + neighbor_n + worker.updates
        return z, names, support_n

    def _collect_residuals(self, worker_id: str, key: ContextKey) -> List[float]:
        direct = self._cell(worker_id, key)
        fam = self._family_cell(worker_id, key[0])
        worker = self._worker_cell(worker_id)
        values = list(direct.residual_logs) + list(fam.residual_logs[-self.window :]) + list(worker.residual_logs[-self.window :])
        for (wid, other_key), cell in self.cells.items():
            if wid != worker_id or other_key == key or cell.updates <= 0:
                continue
            if self._context_distance(key, other_key) <= self.hier_tau:
                values.extend(cell.residual_logs[-max(3, self.window // 3) :])
        return values[-(self.window * 3) :]

    def predict(self, worker_id: str, key: ContextKey, qos: int) -> Tuple[float, float, Dict[str, Any]]:
        base = self.fixed(worker_id, key)
        if self.strategy == "fixed":
            pred = base
            bound = pred * self.fixed_safety
            return pred, bound, {"z_hat": 0.0, "updates": 0, "borrow_source": "static", "support_n": 0, "risk_margin_log": math.log(self.fixed_safety)}
        cell = self._cell(worker_id, key)
        if self.strategy == "ema":
            pred = base * math.exp(cell.z_hat)
            bound = pred * self.ema_kappa
            return pred, bound, {"z_hat": cell.z_hat, "updates": cell.updates, "borrow_source": "direct", "support_n": cell.updates, "risk_margin_log": math.log(self.ema_kappa)}
        z, source, support_n = self._full_z(worker_id, key)
        pred = base * math.exp(z)
        residuals = [abs(v) for v in self._collect_residuals(worker_id, key)]
        empirical = percentile(residuals, 0.80) if residuals else 0.0
        cold = self.cold_uncertainty / math.sqrt(1.0 + max(cell.updates, 0))
        qos_scale = 1.0 + 0.12 * int(qos)
        margin_log = qos_scale * (self.risk_beta * empirical + cold)
        margin_log = min(1.10, max(0.05, margin_log))
        bound = pred * math.exp(margin_log)
        return pred, bound, {"z_hat": z, "updates": cell.updates, "borrow_source": source, "support_n": support_n, "risk_margin_log": margin_log}

    def observe(self, worker_id: str, key: ContextKey, observed_s: float, predicted_s: float) -> Tuple[float, float, int]:
        if self.strategy == "fixed":
            return 0.0, 0.0, 0
        cell = self._cell(worker_id, key)
        before = cell.z_hat
        base = max(self.fixed(worker_id, key), 1e-9)
        z_obs = math.log(max(observed_s, 1e-9) / base)
        z_obs = max(-1.5, min(1.5, z_obs))
        pred_err_log = math.log(max(observed_s, 1e-9) / max(predicted_s, 1e-9))
        pred_err_log = max(-1.5, min(1.5, pred_err_log))
        self._update_cell(cell, z_obs, pred_err_log, self.alpha)
        if self.strategy == "full":
            self._update_cell(self._family_cell(worker_id, key[0]), z_obs, pred_err_log, self.alpha * 0.85)
            self._update_cell(self._worker_cell(worker_id), z_obs, pred_err_log, self.alpha * 0.55)
        return before, cell.z_hat, cell.updates

    def _update_cell(self, cell: OnlineCell, z_obs: float, pred_err_log: float, alpha: float) -> None:
        cell.z_hat = (1.0 - alpha) * cell.z_hat + alpha * z_obs
        cell.updates += 1
        cell.residual_logs.append(pred_err_log)
        if len(cell.residual_logs) > self.window:
            del cell.residual_logs[: len(cell.residual_logs) - self.window]
