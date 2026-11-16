"""Point-estimator replacements with the frozen Full risk and scheduler rules.

Local cells are deliberately retained ONLY for the common Full risk wrapper.
PoolEMA's point state is one directly updated EMA per (measured GPU type, family),
not an average of context EMA snapshots. Its faster group event clock and broader
precision scope are inherent structural differences, not additional observations.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / '00_configuration_sources' / 'frozen_sources'
spec = importlib.util.spec_from_file_location('pool_frozen_simulator', FROZEN / 'hetero_qos_sim_n2.py')
sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)
OriginalCalibrator = sim.OnlineCalibrator
MODES = ('full', 'pool_ema', 'local_matched', 'gated_pred')


class SharedRiskCalibrator(OriginalCalibrator):
    mode = 'full'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.strategy != 'full_qos':
            raise ValueError('All modes must use the frozen full_qos scheduler configuration')
        self.point_mode = type(self).mode
        self.group_z = {}
        self.group_updates = {}

    @staticmethod
    def group_key(context, device):
        return sim.measured_device_type(device), context[0]

    def predict(self, context, device, qos, guarded=True):
        if self.point_mode == 'full':
            return super().predict(context, device, qos, guarded=guarded)
        cell = self._cell((context, device))
        full_z, sigma, _, self_weight = super()._effective_state(context, device)
        if self.point_mode == 'pool_ema':
            z = self.group_z.get(self.group_key(context, device), 0.0)
        elif self.point_mode == 'local_matched':
            z = cell.zhat
        elif self.point_mode == 'gated_pred':
            z = full_z if cell.updates == 0 else cell.zhat
        else:
            raise ValueError(self.point_mode)
        # Keep the Full uncertainty and self-weight-dependent cold bonus intact.
        guard = 0.0
        if guarded and qos >= 1:
            guard = self.risk_beta * (1.0 + 0.35 * qos) * max(
                sigma, (1.0 - self_weight) * self.cold_uncertainty)
        corrected_z = max(self.z_min, min(self.z_max, z + guard))
        return self.base[(context, device)] * math.exp(corrected_z), sigma

    def observe(self, context, device, observed, predicted, now):
        # The same feedback reaches the frozen local EMA/risk statistics.
        super().observe(context, device, observed, predicted, now)
        if self.point_mode != 'pool_ema':
            return
        group = self.group_key(context, device)
        old = self.group_z.get(group, 0.0)
        z = math.log(max(observed, 1e-9) / max(self.base[(context, device)], 1e-9))
        delta = max(-0.35, min(0.35, z - old))
        self.group_z[group] = max(self.z_min, min(self.z_max, old + self._alpha_for_context(context) * delta))
        self.group_updates[group] = self.group_updates.get(group, 0) + 1


sim.OnlineCalibrator = SharedRiskCalibrator


def task_fingerprint(tasks):
    """Hash all exogenous inputs, including all candidate potential service times."""
    import hashlib
    import json
    plan = [(t.task_id, t.user, t.context, t.arrival, t.qos, t.deadline,
             sorted(t.true_times.items())) for t in tasks]
    return hashlib.sha256(json.dumps(plan, separators=(',', ':'), ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def make_runner(profile, mode, seed, pool, load, deadline_scale=.80, strict_prob=.25):
    import random
    SharedRiskCalibrator.mode = mode
    scenario = sim.Scenario(random.Random(seed), 12.0, pool * load, True,
                            str(profile), 1.0, 1.0, strict_prob, deadline_scale,
                            0.006, 1.60, 2.00, 'median', 1.0, .92, 1.08,
                            0.0, 0.0, (1.60, 1.30, 1.10))
    return sim.EventSimulator(scenario, 'full_qos', seed, 0.0, 24, 4,
                              .30, .20, 0.0, 1.0, .50, .22, 1.20,
                              1.5, .65, .080, .015, 8, .80, .30)
