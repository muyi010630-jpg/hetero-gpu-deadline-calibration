"""Historical Window controls and bounded decision-related log adapters.

This is a feasibility adapter, not SPO+ or an estimator of global scheduling
regret. Legal modes see selected completions and dispatch-time history only.
The two ``*_fullinfo`` modes are explicitly privileged mechanism diagnostics.
All modes retain the frozen Full risk state and selected-only raw windows.

Runner contract: register_decision at dispatch, then complete_feedback exactly
once at completion. Do not also call observe for that completion. Inherited
observe can be used to preload a common historical feedback sequence in QA.
"""
from __future__ import annotations

import math
from collections import defaultdict

from frozen_support import calibrator_core as core

sim = core.sim
OriginalCalibrator = core.OriginalCalibrator

MODES = ('local_window', 'pool_ema', 'full_original', 'uniform_adapter',
         'decision_weighted', 'rank_history', 'uniform_fullinfo', 'rank_fullinfo')
BASELINE_MODES = frozenset(('local_window', 'pool_ema', 'full_original'))
PRIVILEGED_MODES = frozenset(('uniform_fullinfo', 'rank_fullinfo'))
ADAPTER_BOUND = 0.25
ADAPTER_L2 = 0.01
RANK_TEMPERATURE = 0.05
RANK_MARGIN = 0.02
HISTORY_MAX_AGE_S = 6.0


def _finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + ' must be finite')
    return result


def _positive(value, name):
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(name + ' must be positive')
    return result


def _sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


def _softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def scheduler_score(service_s, *, now, arrival, deadline, load_fraction):
    """The idle-device score used by the frozen scheduler, including lateness."""
    budget = max(deadline - arrival, 1e-6)
    projected = max(0.0, now + service_s - arrival) / budget
    miss = max(0.0, now + service_s - deadline) / budget
    return projected + 0.65 * miss + 0.08 * load_fraction


def score_log_derivative(service_s, *, now, arrival, deadline):
    """d scheduler_score / d log(service), away from the two hinge knots."""
    budget = max(deadline - arrival, 1e-6)
    return service_s / budget * (
        float(now + service_s > arrival) + 0.65 * float(now + service_s > deadline))


class DecisionCalibrator(core.Calibrator):
    def __init__(self, base, mode, params=None):
        if mode not in MODES:
            raise ValueError('Unknown decision-calibration mode: ' + str(mode))
        selected = dict(eta=0.05, ranking_weight=0.1)
        if params:
            unknown = set(params) - set(selected)
            if unknown:
                raise ValueError('Unsupported/oracle parameter keys: ' + repr(sorted(unknown)))
            selected.update(params)
        selected = {name: _finite(value, name) for name, value in selected.items()}
        if not 0.0 < selected['eta'] <= 1.0:
            raise ValueError('eta must be in (0, 1]')
        if selected['ranking_weight'] < 0.0:
            raise ValueError('ranking_weight must be nonnegative')
        super().__init__(base, mode if mode in BASELINE_MODES else 'local_window')
        # Keep the inherited adaptive parameters, used by common diagnostics.
        self.adapter_params = selected
        self.mode = mode
        self.adapter_z = {}
        self.pending_decisions = {}
        self.completed_decisions = set()
        self.supervised_updates = 0
        self.pair_updates = 0  # Number of non-tied selected-vs-other label pairs.
        self.privileged_label_count = 0  # Non-selected vector labels consumed.
        self.completed_feedback_count = 0
        self.adapter_clip_count = 0
        self.skipped_history_pairs = 0
        self.supervised_updates_by_key = defaultdict(int)
        self.pair_updates_by_key = defaultdict(int)
        self.supervised_weight_sum = 0.0
        self.supervised_loss_sum = 0.0
        self.ranking_loss_sum = 0.0
        self.last_learning = {}

    def _point_estimate(self, context, device, full_z, risk_self_weight):
        if self.mode in BASELINE_MODES:
            return super()._point_estimate(context, device, full_z, risk_self_weight)
        # Reuse the exact original Window location and diagnostic calculation.
        # Calibrators are mutable, sequential objects, as in the original core.
        requested_mode = self.mode
        self.mode = 'local_window'
        try:
            window_z, detail = super()._point_estimate(context, device, full_z, risk_self_weight)
        finally:
            self.mode = requested_mode
        adjustment = self.adapter_z.get((context, device), 0.0)
        raw = window_z + adjustment
        bounded = core._clip(raw)
        detail.update(window_point_z=window_z, adapter_z=adjustment,
                      raw_point_z=raw, point_z=bounded, point_clipped=(raw != bounded))
        return bounded, detail

    def history_snapshot(self, context, device):
        """Raw selected-feedback log-Window mean, excluding adapters/clips."""
        key = (context, device)
        count, mean, _, _ = self._local_window_stats(key)
        last = self.raw_local_last_update.get(key)
        return dict(updates=self._cell(key).updates,
                    history_mean_s=(self.base[key] * math.exp(mean) if count else None),
                    history_age_s=(max(0.0, self.now - last) if last is not None else None))

    def register_decision(self, task_id, context, selected_device, snapshot):
        """Copy only permitted dispatch-time fields; retain no Task/Scenario."""
        if task_id in self.pending_decisions or task_id in self.completed_decisions:
            raise ValueError('Task decision already registered/completed')
        context = tuple(context)
        now = _finite(snapshot['now'], 'dispatch now')
        arrival = _finite(snapshot['arrival'], 'arrival')
        deadline = _finite(snapshot['deadline'], 'deadline')
        if now < arrival or deadline <= arrival:
            raise ValueError('Dispatch must follow arrival and budget must be positive')
        if abs(now - self.now) > 1e-9:
            raise ValueError('Call set_now(dispatch now) before registering the decision')
        qos = int(snapshot['qos'])
        if qos not in (0, 1, 2) or qos != snapshot['qos']:
            raise ValueError('qos must be 0, 1, or 2')
        criticality = max(0.0, min(1.0, _finite(snapshot['criticality'], 'criticality')))
        candidates = {}
        for candidate in snapshot['candidates']:
            device = candidate['device']
            key = (context, device)
            if key not in self.base or device in candidates:
                raise ValueError('Candidates must be unique known context/device keys')
            pred = _positive(candidate['pred_s'], 'candidate prediction')
            load = _finite(candidate['load_fraction'], 'load_fraction')
            if load < 0.0:
                raise ValueError('load_fraction must be nonnegative')
            updates = int(candidate.get('updates', 0))
            if updates < 0 or updates != candidate.get('updates', 0):
                raise ValueError('updates must be a nonnegative integer')
            history_mean = candidate.get('history_mean_s')
            history_age = candidate.get('history_age_s')
            if history_mean is not None:
                history_mean = _positive(history_mean, 'history_mean_s')
            if history_age is not None:
                history_age = _finite(history_age, 'history_age_s')
                if history_age < 0.0:
                    raise ValueError('history_age_s must be nonnegative')
            point_z = math.log(pred / self.base[key])
            # The clip derivative is zero at/outside the frozen output bounds.
            derivative = float(-0.70 < point_z < 0.70)
            candidates[device] = dict(
                device=device, pred_s=pred, load_fraction=load, updates=updates,
                history_mean_s=history_mean, history_age_s=history_age,
                point_derivative=derivative)
        if selected_device not in candidates:
            raise ValueError('Selected device must belong to the dispatch candidate set')
        self.pending_decisions[task_id] = dict(
            context=context, selected_device=selected_device, now=now,
            arrival=arrival, deadline=deadline, qos=qos,
            criticality=criticality, candidates=candidates)

    @staticmethod
    def _score(record, service_s, candidate):
        return scheduler_score(service_s, now=record['now'], arrival=record['arrival'],
                               deadline=record['deadline'],
                               load_fraction=candidate['load_fraction'])

    def _pair_terms(self, record, labels):
        """Delayed pair loss from saved predictions; no post-feedback prediction."""
        selected_device = record['selected_device']
        selected = record['candidates'][selected_device]
        selected_label = self._score(record, labels[selected_device], selected)
        selected_pred = self._score(record, selected['pred_s'], selected)
        terms = []
        for device, label in labels.items():
            if device == selected_device:
                continue
            other = record['candidates'][device]
            label_gap = self._score(record, label, other) - selected_label
            if abs(label_gap) <= 1e-12:
                continue  # No arbitrary ordering of equal label scores.
            target_sign = 1.0 if label_gap > 0.0 else -1.0
            pred_gap = self._score(record, other['pred_s'], other) - selected_pred
            argument = (RANK_MARGIN - target_sign * pred_gap) / RANK_TEMPERATURE
            regret_weight = min(1.0, abs(label_gap))
            multiplier = regret_weight * _sigmoid(argument) * target_sign
            common = dict(now=record['now'], arrival=record['arrival'], deadline=record['deadline'])
            grad_selected = (multiplier * score_log_derivative(selected['pred_s'], **common)
                             * selected['point_derivative'])
            grad_other = (-multiplier * score_log_derivative(other['pred_s'], **common)
                          * other['point_derivative'])
            terms.append(dict(device=device, grad_selected=grad_selected,
                              grad_other=grad_other, regret_weight=regret_weight,
                              loss=regret_weight * RANK_TEMPERATURE * _softplus(argument)))
        return terms

    def complete_feedback(self, task_id, observed, now, full_vector=None):
        """Consume one selected observation; update all gradients simultaneously."""
        if task_id not in self.pending_decisions:
            raise ValueError('Completion has no pending dispatch decision')
        record = self.pending_decisions[task_id]
        context, selected_device = record['context'], record['selected_device']
        observed = _positive(observed, 'observed execution time')
        now = _finite(now, 'completion now')
        if now < record['now'] or now < self.now - 1e-12:
            raise ValueError('Completion time must be nondecreasing and follow dispatch')
        privileged = self.mode in PRIVILEGED_MODES
        if not privileged and full_vector is not None:
            raise ValueError('Legal/baseline modes must not receive counterfactual full_vector')
        labels = {selected_device: observed}
        if privileged:
            if full_vector is None or set(full_vector) != set(record['candidates']):
                raise ValueError('Privileged vector must exactly match the dispatch candidate set')
            labels = {device: _positive(value, 'privileged execution time')
                      for device, value in full_vector.items()}
            if not math.isclose(labels[selected_device], observed, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Privileged selected label must equal actual selected observation')
        gradients = defaultdict(float)
        regression_weight = 1.0
        supervised_labels = labels if privileged else {selected_device: observed}
        pair_terms = []
        supervised_loss = 0.0
        if self.mode not in BASELINE_MODES:
            if self.mode == 'decision_weighted':
                qos_weight = 1.0 + 0.5 * record['qos'] if record['qos'] >= 1 else 1.0
                regression_weight = min(5.0, (1.0 + 2.0 * record['criticality']) * qos_weight)
            for device, label in supervised_labels.items():
                candidate = record['candidates'][device]
                error = math.log(candidate['pred_s'] / label)
                gradients[(context, device)] += regression_weight * error * candidate['point_derivative']
                supervised_loss += 0.5 * regression_weight * error ** 2
            if self.mode == 'rank_history':
                proxy_labels = {selected_device: observed}
                for device, candidate in record['candidates'].items():
                    if device == selected_device:
                        continue
                    mean, age = candidate['history_mean_s'], candidate['history_age_s']
                    if candidate['updates'] > 0 and mean is not None and age is not None and age <= HISTORY_MAX_AGE_S:
                        proxy_labels[device] = mean
                    else:
                        self.skipped_history_pairs += 1
                pair_terms = self._pair_terms(record, proxy_labels)
            elif self.mode == 'rank_fullinfo':
                pair_terms = self._pair_terms(record, labels)
            # Average pair gradients within this completion to avoid a hidden
            # learning-rate increase when more devices happen to be idle.
            if pair_terms:
                scale = self.adapter_params['ranking_weight'] / len(pair_terms)
                for term in pair_terms:
                    gradients[(context, selected_device)] += scale * term['grad_selected']
                    gradients[(context, term['device'])] += scale * term['grad_other']
            for key in gradients:
                gradients[key] += ADAPTER_L2 * self.adapter_z.get(key, 0.0)
        # Common local EMA, sigma, raw windows, pools and risk statistics receive
        # exactly the selected observation, irrespective of privileged labels.
        flags = super().observe(context, selected_device, observed,
                                record['candidates'][selected_device]['pred_s'], now)
        clips = 0
        for key, gradient in gradients.items():
            raw = self.adapter_z.get(key, 0.0) - self.adapter_params['eta'] * gradient
            bounded = max(-ADAPTER_BOUND, min(ADAPTER_BOUND, raw))
            self.adapter_z[key] = bounded
            clips += int(raw != bounded)
        if self.mode not in BASELINE_MODES:
            self.supervised_updates += len(supervised_labels)
            self.supervised_weight_sum += regression_weight * len(supervised_labels)
            for device in supervised_labels:
                self.supervised_updates_by_key[(context, device)] += 1
            self.pair_updates += len(pair_terms)
            for term in pair_terms:
                self.pair_updates_by_key[(context, selected_device)] += 1
                self.pair_updates_by_key[(context, term['device'])] += 1
        if privileged:
            self.privileged_label_count += len(labels) - 1
        self.completed_feedback_count += 1
        self.adapter_clip_count += clips
        self.supervised_loss_sum += supervised_loss
        ranking_loss = (sum(term['loss'] for term in pair_terms) / len(pair_terms) if pair_terms else 0.0)
        self.ranking_loss_sum += ranking_loss
        self.last_learning = dict(task_id=task_id, now=now, regression_weight=regression_weight,
                                  supervised_labels=(len(supervised_labels) if self.mode not in BASELINE_MODES else 0),
                                  usable_pairs=len(pair_terms), privileged_labels=(len(labels) - 1 if privileged else 0),
                                  supervised_loss=supervised_loss, ranking_loss=ranking_loss,
                                  adapter_clips=clips,
                                  label_kind=('privileged_vector' if privileged else
                                              'selected_plus_historical_proxy' if self.mode == 'rank_history' else 'selected_only'))
        del self.pending_decisions[task_id]
        self.completed_decisions.add(task_id)
        result = dict(flags)
        result['learning'] = dict(self.last_learning)
        return result

    def learning_diagnostics(self):
        return dict(mode=self.mode, eta=self.adapter_params['eta'],
                    ranking_weight=self.adapter_params['ranking_weight'],
                    supervised_updates=self.supervised_updates, pair_updates=self.pair_updates,
                    privileged_label_count=self.privileged_label_count,
                    completed_feedback_count=self.completed_feedback_count,
                    adapter_clip_count=self.adapter_clip_count,
                    skipped_history_pairs=self.skipped_history_pairs,
                    supervised_weight_sum=self.supervised_weight_sum,
                    supervised_loss_sum=self.supervised_loss_sum,
                    ranking_loss_sum=self.ranking_loss_sum,
                    pending_decisions=len(self.pending_decisions))

    def diagnostics(self, context, device):
        result = super().diagnostics(context, device)
        key = (context, device)
        result.update(adapter_z=self.adapter_z.get(key, 0.0),
                      supervised_updates=self.supervised_updates,
                      pair_updates=self.pair_updates,
                      privileged_label_count=self.privileged_label_count,
                      key_supervised_updates=self.supervised_updates_by_key.get(key, 0),
                      key_pair_updates=self.pair_updates_by_key.get(key, 0))
        return result


Calibrator = DecisionCalibrator


def create_calibrator(base, mode, params=None):
    return DecisionCalibrator(base, mode, params)
