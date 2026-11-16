"""Frozen Full risk with historical point-estimator controls and an EB-style candidate.

No mode receives true service profiles, correlation parameters, or future events.
New raw windows and direct pool states are updated identically for every mode.
Only the point estimator selected at prediction time differs.
"""
from __future__ import annotations

import importlib.util
import math
import statistics
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN_PATH = ROOT / '00_protocol_sources' / 'frozen_sources' / 'hetero_qos_sim_n2.py'
spec = importlib.util.spec_from_file_location('adaptive_frozen_simulator', FROZEN_PATH)
sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)
OriginalCalibrator = sim.OnlineCalibrator

PRIMARY_MODES = ('full_original', 'local_ema', 'pool_ema', 'gated_point',
                 'local_window', 'adaptive_pool_local')
DIAGNOSTIC_MODES = ('hierarchy_common_scope', 'pool_matched_dtype',
                    'full_no_neighbor_shrink', 'full_fresh_neighbors',
                    'pool_exposure_matched', 'adaptive_fixed_mix',
                    'adaptive_no_freshness')
MODES = PRIMARY_MODES + DIAGNOSTIC_MODES
WINDOW = 24
NOISE_VARIANCE_FLOOR = 0.0025
DIAGNOSTIC_FRESHNESS_HALF_LIFE_S = 6.0
PARAM_GRID = tuple(dict(shrink_scale=s, freshness_half_life_s=h)
                   for s in (0.5, 1.0, 2.0) for h in (1.5, 6.0, 1e9))


def _clip(value, lo=-0.70, hi=0.70):
    return max(lo, min(hi, value))


def _window_stats(values):
    n = len(values)
    if not n:
        return 0, 0.0, NOISE_VARIANCE_FLOOR, 0.0
    mean = statistics.mean(values)
    variance = max(NOISE_VARIANCE_FLOOR,
                   statistics.variance(values) if n >= 2 else NOISE_VARIANCE_FLOOR)
    return n, mean, variance, variance / n


class Calibrator(OriginalCalibrator):
    def __init__(self, base, mode, params=None):
        if mode not in MODES:
            raise ValueError('Unknown point mode: ' + str(mode))
        selected = dict(shrink_scale=1.0, freshness_half_life_s=6.0)
        if params:
            unknown = set(params) - set(selected)
            if unknown:
                raise ValueError('Unsupported/oracle parameter keys: ' + repr(sorted(unknown)))
            selected.update(params)
        selected = {key: float(value) for key, value in selected.items()}
        if selected['shrink_scale'] not in (0.5, 1.0, 2.0):
            raise ValueError('shrink_scale must belong to the frozen three-value grid')
        if selected['freshness_half_life_s'] not in (1.5, 6.0, 1e9):
            raise ValueError('freshness_half_life_s must belong to the frozen three-value grid')
        super().__init__(base, 'full_qos', window=24, ema_rho=0.30, rc_rho=0.20,
                         z_min=-0.70, z_max=0.70, min_samples=4,
                         kappa_qos=0.0, hier_tau=1.5, drift_alpha_high=0.65,
                         drift_cusum_h=0.080, drift_cusum_k=0.015,
                         drift_boost_steps=8, risk_beta=0.80, cold_uncertainty=0.30)
        self.mode = mode
        self.params = selected
        self.now = 0.0
        self.raw_local = {}
        self.raw_unique_context = {}
        self.raw_local_last_update = {}
        self.group_z = {}
        self.group_updates = {}
        self.dtype_group_z = {}
        self.dtype_group_updates = {}
        self.exposure_group_z = {}
        self.exposure_group_updates = {}
        self.group_members = {}
        self.group_local_cells = {}
        for context, device in base:
            self.group_members.setdefault(self.group_key(context, device), set()).add(context)
            self.group_local_cells.setdefault(self.group_key(context, device), set()).add((context, device))
        self.last_prediction = {}
        self.last_observation = {}

    @staticmethod
    def group_key(context, device):
        return sim.measured_device_type(device), context[0]

    @staticmethod
    def unique_context_key(context, device):
        return sim.measured_device_type(device), context

    def set_now(self, now):
        now = float(now)
        if not math.isfinite(now) or now < self.now - 1e-12:
            raise ValueError('Prediction/feedback time must be finite and nondecreasing')
        self.now = max(self.now, now)

    def _heterogeneity(self, context, device):
        group = self.group_key(context, device)
        records = []
        for (model, other_context), values in self.raw_unique_context.items():
            if (model, other_context[0]) != group:
                continue
            n, mean, variance, sampling_variance = _window_stats(values)
            if n:
                records.append((n, mean, variance, sampling_variance))
        total = sum(rec[0] for rec in records)
        if not total:
            return dict(heterogeneity=0.0, unique_contexts=0, effective_group_samples=0,
                        between_window_mean_variance=0.0, mean_sampling_variance=0.0)
        grand = sum(n * mean for n, mean, _, _ in records) / total
        between = sum(n * (mean - grand) ** 2 for n, mean, _, _ in records) / total
        sampling = sum(n * sampling_var for n, _, _, sampling_var in records) / total
        # One observed unique context cannot identify cross-context heterogeneity.
        heterogeneity = max(0.0, between - sampling) if len(records) >= 2 else 0.0
        return dict(heterogeneity=heterogeneity, unique_contexts=len(records),
                    effective_group_samples=total, between_window_mean_variance=between,
                    mean_sampling_variance=sampling)

    def _point_neighbors(self, context, device, common_scope=False, freshness=False):
        family, _, _, dtype = context
        target_type = sim.measured_device_type(device)
        exact, fallback = [], []
        for (other_context, other_device), other in self.cells.items():
            if (other_context, other_device) == (context, device):
                continue
            if other.updates <= 0 or other_context[0] != family:
                continue
            same_type = sim.measured_device_type(other_device) == target_type
            if common_scope and not same_type:
                continue
            weight = float(max(1, other.updates))
            if freshness:
                age = max(0.0, self.now - other.last_update)
                weight *= 2.0 ** (-age / DIAGNOSTIC_FRESHNESS_HALF_LIFE_S)
            item = (other.zhat, max(other.sigma, 1e-6), weight)
            if common_scope or (same_type and other_context[3] == dtype):
                exact.append(item)
            else:
                fallback.append(item)
        source = exact if exact else fallback
        if not source:
            return 0.0, 0.0, 0.0
        total = sum(item[2] for item in source)
        if total <= 0.0:
            return 0.0, 0.0, 0.0
        mean = sum(z * weight for z, _, weight in source) / total
        variance = sum(weight * ((z - mean) ** 2 + sigma ** 2)
                       for z, sigma, weight in source) / total
        scale = 1.0 if exact else 0.55
        return mean * scale, math.sqrt(max(0.0, variance)) * scale, total

    def _hierarchy_point(self, context, device, common_scope=False,
                         freshness=False, no_neighbor_shrink=False):
        cell = self._cell((context, device))
        neighbor_z, _, count = self._point_neighbors(context, device, common_scope, freshness)
        active = count > 0 and cell.updates < max(2 * self.min_samples, 2 * self.hier_tau)
        self_weight = cell.updates / (cell.updates + self.hier_tau) if active else 1.0
        trust = (1.0 if no_neighbor_shrink else count / (count + self.hier_tau)) if active else 0.0
        z = self_weight * cell.zhat + (1.0 - self_weight) * trust * neighbor_z
        return z, dict(point_neighbor_count=count, point_neighbor_trust=trust,
                       point_self_weight=self_weight, point_borrowing_active=active)

    def _point_estimate(self, context, device, full_z, risk_self_weight):
        key = context, device
        cell = self._cell(key)
        group = self.group_key(context, device)
        local_n, local_mean, local_var, local_sampling_var = _window_stats(self.raw_local.get(key, ()))
        bounded_local_mean = _clip(local_mean)
        h = self._heterogeneity(context, device)
        last_update = self.raw_local_last_update.get(key)
        age = max(0.0, self.now - last_update) if last_update is not None else None
        freshness = 2.0 ** (-age / self.params['freshness_half_life_s']) if age is not None else 0.0
        local_weight = (h['heterogeneity'] / (h['heterogeneity'] + self.params['shrink_scale'] * local_sampling_var)
                        if local_n and h['heterogeneity'] > 0.0 else 0.0)
        detail = dict(h, local_n_window=local_n, local_updates_count=cell.updates,
                      local_mean=local_mean, local_bounded_mean=bounded_local_mean,
                      local_noise_variance=local_var, local_sampling_variance=local_sampling_var,
                      age_s=age, local_freshness_factor=freshness,
                      heterogeneity_local_weight=local_weight,
                      group_updates_count=self.group_updates.get(group, 0),
                      group_z=self.group_z.get(group, 0.0),
                      known_unique_contexts_in_group=len(self.group_members.get(group, ())),
                      known_local_cells_in_group=len(self.group_local_cells.get(group, ())),
                      point_neighbor_count=0.0,
                      point_neighbor_trust=0.0, point_self_weight=1.0,
                      point_borrowing_active=False,
                      effective_local_weight=None)
        if self.mode == 'full_original':
            z = full_z
            detail.update(point_neighbor_count=cell.last_neighbor_count,
                          point_neighbor_trust=(cell.last_neighbor_count / (cell.last_neighbor_count + self.hier_tau)
                                                if risk_self_weight < 1.0 else 0.0),
                          point_self_weight=risk_self_weight,
                          point_borrowing_active=risk_self_weight < 1.0)
        elif self.mode == 'local_ema':
            z = cell.zhat
        elif self.mode == 'pool_ema':
            z = self.group_z.get(group, 0.0)
        elif self.mode == 'gated_point':
            z = full_z if cell.updates == 0 else cell.zhat
            detail['point_borrowing_active'] = cell.updates == 0 and risk_self_weight < 1.0
            if detail['point_borrowing_active']:
                detail.update(point_neighbor_count=cell.last_neighbor_count,
                              point_neighbor_trust=cell.last_neighbor_count / (cell.last_neighbor_count + self.hier_tau),
                              point_self_weight=risk_self_weight)
        elif self.mode == 'local_window':
            z = local_mean if local_n else 0.0
        elif self.mode == 'pool_matched_dtype':
            z = self.dtype_group_z.get(group + (context[3],), 0.0)
        elif self.mode == 'pool_exposure_matched':
            z = self.exposure_group_z.get(group, 0.0)
        elif self.mode in ('hierarchy_common_scope', 'full_no_neighbor_shrink', 'full_fresh_neighbors'):
            z, point_detail = self._hierarchy_point(
                context, device, common_scope=self.mode == 'hierarchy_common_scope',
                freshness=self.mode == 'full_fresh_neighbors',
                no_neighbor_shrink=self.mode == 'full_no_neighbor_shrink')
            detail.update(point_detail)
        else:
            if self.mode == 'adaptive_fixed_mix':
                local_weight = local_n / (local_n + self.hier_tau) if local_n else 0.0
            if self.mode == 'adaptive_no_freshness':
                freshness = 1.0 if local_n else 0.0
            effective_weight = local_weight * freshness
            z = effective_weight * bounded_local_mean + (1.0 - effective_weight) * self.group_z.get(group, 0.0)
            detail.update(effective_local_weight=effective_weight,
                          point_self_weight=effective_weight,
                          point_borrowing_active=(effective_weight < 1.0 and self.group_updates.get(group, 0) > 0),
                          local_freshness_factor=freshness, mixing_local_weight=local_weight,
                          local_window_mean_clipped=(local_mean != bounded_local_mean))
        bounded = _clip(z)
        detail.update(raw_point_z=z, point_z=bounded, point_clipped=(bounded != z))
        return bounded, detail

    def predict(self, context, device, qos=0, guarded=False):
        # Do not override _neighbor_stats/_effective_state: these are the common
        # frozen Full risk wrapper, including sigma and its self-weight cold term.
        full_z, sigma, neighbor_count, self_weight = OriginalCalibrator._effective_state(self, context, device)
        point_z, detail = self._point_estimate(context, device, full_z, self_weight)
        guard = 0.0
        if guarded and qos >= 1:
            guard = self.risk_beta * (1.0 + 0.35 * qos) * max(sigma, (1.0 - self_weight) * self.cold_uncertainty)
        output_z = _clip(point_z + guard)
        detail.update(mode=self.mode, now=self.now, risk_full_z=full_z, risk_sigma=sigma,
                      risk_neighbor_count=neighbor_count, risk_self_weight=self_weight,
                      guard_log=guard, output_log_residual=output_z,
                      guarded_output_clipped=(output_z != point_z + guard))
        self.last_prediction[(context, device)] = detail
        return self.base[(context, device)] * math.exp(output_z), sigma

    @staticmethod
    def _update_direct(state, counts, key, residual, alpha):
        previous = state.get(key, 0.0)
        raw_delta = residual - previous
        delta = max(-0.35, min(0.35, raw_delta))
        unbounded = previous + alpha * delta
        updated = _clip(unbounded)
        state[key] = updated
        counts[key] = counts.get(key, 0) + 1
        return dict(previous=previous, residual=residual, alpha=alpha, delta=delta,
                    state=updated, innovation_clipped=(delta != raw_delta),
                    state_clipped=(updated != unbounded))

    def observe(self, context, device, obs, pred, now):
        self.set_now(now)
        if not math.isfinite(float(obs)) or obs <= 0.0:
            raise ValueError('Observed execution time must be finite and positive')
        key = context, device
        residual = math.log(max(obs, 1e-9) / max(self.base[key], 1e-9))
        cell = self._cell(key)
        alpha = self._alpha_for_context(context)
        previous = cell.zhat
        raw_delta = residual - previous
        delta = max(-0.35, min(0.35, raw_delta))
        unbounded = previous + alpha * delta
        # Parent local EMA, sigma, samples, update counter and risk are unchanged.
        OriginalCalibrator.observe(self, context, device, obs, pred, now)
        self.raw_local.setdefault(key, deque(maxlen=WINDOW)).append(residual)
        unique = self.unique_context_key(context, device)
        self.raw_unique_context.setdefault(unique, deque(maxlen=WINDOW)).append(residual)
        self.raw_local_last_update[key] = float(now)
        group = self.group_key(context, device)
        direct = self._update_direct(self.group_z, self.group_updates, group, residual, alpha)
        dtype = self._update_direct(self.dtype_group_z, self.dtype_group_updates,
                                    group + (context[3],), residual, alpha)
        m = max(1, len(self.group_local_cells.get(group, ())))
        exposure_alpha = 1.0 - (1.0 - alpha) ** (1.0 / m)
        exposure = self._update_direct(self.exposure_group_z, self.exposure_group_updates,
                                       group, residual, exposure_alpha)
        detail = dict(now=self.now, residual=residual, local_alpha=alpha,
                      local_innovation_clipped=(delta != raw_delta),
                      local_state_clipped=(_clip(unbounded) != unbounded),
                      pool_innovation_clipped=direct['innovation_clipped'],
                      pool_state_clipped=direct['state_clipped'],
                      pool_dtype_innovation_clipped=dtype['innovation_clipped'],
                      pool_dtype_state_clipped=dtype['state_clipped'],
                      pool_exposure_alpha=exposure_alpha,
                      pool_exposure_innovation_clipped=exposure['innovation_clipped'],
                      pool_exposure_state_clipped=exposure['state_clipped'])
        self.last_observation[key] = detail
        return dict(detail)

    def diagnostics(self, context, device):
        """Snapshot last prediction and feedback flags; no truth/stream access."""
        key = context, device
        result = dict(self.last_prediction.get(key, {}))
        result.update({'last_' + field: value for field, value in self.last_observation.get(key, {}).items()})
        return result


def create_calibrator(base, mode, params=None):
    return Calibrator(base, mode, params)
