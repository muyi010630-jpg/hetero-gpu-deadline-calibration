"""Selected-feedback deadline-probability calibration over frozen Window points.

The lognormal base CDF and positive-slope calibration layer are empirical
models, not coverage guarantees. Only an actual dispatched device supplies a
training label. Dispatch features, probabilities and parameters are frozen for
delayed BCE gradients; current parameters supply identity regularization.

Runner contract: register_decision once at dispatch and complete_feedback once
at completion. complete_feedback includes the single selected core.observe.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from frozen_support import calibrator_core as core

sim = core.sim
OriginalCalibrator = core.OriginalCalibrator
NORMAL = statistics.NormalDist()
MODES = ('uncalibrated', 'uniform', 'boundary')
ETA_GRID = (0.005, 0.02, 0.08)
BOUNDARY_LAMBDA_GRID = (0.0, 1.0, 3.0)
IDENTITY_L2 = 0.02
LOG_A_BOUND = math.log(2.0)
B_BOUND = 1.0
PROBABILITY_EPSILON = 1e-6
COLD_VARIANCE = 0.09
VARIANCE_FLOOR = 0.0025
VARIANCE_PRIOR_SAMPLES = 4.0


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(name + ' must be positive')
    return value


def _clip_probability(value):
    return max(PROBABILITY_EPSILON, min(1.0 - PROBABILITY_EPSILON, value))


def _logit(value):
    return math.log(value) - math.log1p(-value)


def _sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def bce_loss_and_gradient(log_a, b, logit0, label, weight=1.0):
    """Pure data-loss helper for finite-difference QA; excludes L2."""
    a = math.exp(log_a)
    score = a * logit0 + b
    probability = _sigmoid(score)
    # Stable logistic BCE, including numerically saturated probabilities.
    loss = weight * (max(score, 0.0) - label * score + math.log1p(math.exp(-abs(score))))
    error = weight * (probability - label)
    return loss, error * a * logit0, error


class DeadlineCalibrator:
    def __init__(self, base, mode='uncalibrated', params=None):
        if mode not in MODES:
            raise ValueError('Unknown probability-calibration mode: ' + str(mode))
        selected = dict(eta=0.02, boundary_lambda=(1.0 if mode == 'boundary' else 0.0))
        if params:
            unknown = set(params) - set(selected)
            if unknown:
                raise ValueError('Unsupported/oracle parameter keys: ' + repr(sorted(unknown)))
            selected.update(params)
        selected = {name: _finite(value, name) for name, value in selected.items()}
        if selected['eta'] not in ETA_GRID:
            raise ValueError('eta must belong to the frozen three-value grid')
        if selected['boundary_lambda'] not in BOUNDARY_LAMBDA_GRID:
            raise ValueError('boundary_lambda must belong to the frozen three-value grid')
        if mode != 'boundary' and selected['boundary_lambda'] != 0.0:
            raise ValueError('Only boundary mode may use a nonzero boundary_lambda')
        base = {key: _positive(value, 'base service time') for key, value in base.items()}
        self.core = core.Calibrator(base, mode='local_window')
        self.mode = mode
        self.params = selected
        self.log_a = 0.0
        self.b = {sim.measured_device_type(device): 0.0 for _, device in base}
        self.pending_decisions = {}
        self.completed_decisions = set()
        self.completed_feedback_count = 0
        self.calibration_updates = 0
        self.skipped_nonpositive_budget = 0
        self.parameter_clip_count = 0
        self.updates_by_model = defaultdict(int)
        self.weight_sum = 0.0
        self.weighted_bce_sum = 0.0
        self.brier_sum = 0.0
        self.last_calibration = {}

    def __getattr__(self, name):
        # Expose original summary/history properties without forwarding Python
        # copy/pickle protocols; ordinary deepcopy copies this wrapper's state.
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, 'core'), name)

    @property
    def a(self):
        return math.exp(self.log_a)

    def set_now(self, now):
        self.core.set_now(now)

    def predict_point(self, context, device, now):
        self.core.set_now(now)
        point, sigma = self.core.predict(context, device, qos=0, guarded=False)
        n = self.core._local_window_stats((context, device))[0]
        return point, sigma, n

    def _distribution(self, context, device, now, point_override=None):
        self.core.set_now(now)
        if point_override is None:
            point, _, n = self.predict_point(context, device, now)
        else:
            if (context, device) not in self.core.base:
                raise ValueError('Unknown context/device key')
            point = _positive(point_override, 'point_override')
            n = self.core._local_window_stats((context, device))[0]
        _, _, sample_variance, _ = self.core._local_window_stats((context, device))
        sample_variance = max(VARIANCE_FLOOR, sample_variance)
        if n == 0:
            blend, blended_variance, predictive_scale = 0.0, COLD_VARIANCE, 0.30
        else:
            blend = n / (n + VARIANCE_PRIOR_SAMPLES)
            blended_variance = blend * sample_variance + (1.0 - blend) * COLD_VARIANCE
            predictive_scale = math.sqrt(blended_variance * (1.0 + 1.0 / (n + 1)))
        return dict(point=point, scale=predictive_scale, n=n,
                    sample_variance=sample_variance, blend_weight=blend,
                    blended_variance=blended_variance)

    def probability(self, context, device, remaining, now,
                    calibrated=True, point_override=None):
        remaining = _finite(remaining, 'remaining service budget')
        distribution = self._distribution(context, device, now, point_override)
        model = sim.measured_device_type(device)
        log_a = self.log_a if calibrated else 0.0
        b = self.b[model] if calibrated else 0.0
        if remaining <= 0.0:
            raw_p0, p0, p, logit0 = 0.0, 0.0, 0.0, None
        else:
            z = (math.log(remaining) - math.log(distribution['point'])) / distribution['scale']
            raw_p0 = NORMAL.cdf(z)
            p0 = _clip_probability(raw_p0)
            logit0 = _logit(p0)
            # Exact identity comparison, while retaining the same analytic
            # positive-slope transform used by calibration gradients.
            p = p0 if log_a == 0.0 and b == 0.0 else _sigmoid(math.exp(log_a) * logit0 + b)
        return dict(distribution, now=float(now), remaining_s=remaining,
                    model=model, raw_p0=raw_p0, p0=p0, p=p, logit0=logit0,
                    log_a_at_dispatch=log_a, a_at_dispatch=math.exp(log_a),
                    b_at_dispatch=b, calibrated=bool(calibrated),
                    trainable=(remaining > 0.0))

    def quantile(self, context, device, level, now, kind='uncalibrated'):
        """Return a service-time quantile, not a confidence bound.

        Because the computational CDF clips p0 tails, inversion is restricted
        to levels attainable by that clipped CDF. All levels in [.01, .99]
        remain attainable under the frozen a/b bounds.
        """
        if kind not in ('uncalibrated', 'calibrated'):
            raise ValueError('kind must be uncalibrated or calibrated')
        level = _finite(level, 'quantile level')
        if not 0.0 < level < 1.0:
            raise ValueError('quantile level must be in (0, 1)')
        distribution = self._distribution(context, device, now)
        model = sim.measured_device_type(device)
        target_p0 = level
        if kind == 'calibrated':
            target_p0 = _sigmoid((_logit(level) - self.b[model]) / self.a)
        if target_p0 < PROBABILITY_EPSILON - 1e-15 or target_p0 > 1.0 - PROBABILITY_EPSILON + 1e-15:
            raise ValueError('Level is outside the attainable clipped probability range')
        target_p0 = _clip_probability(target_p0)
        return math.exp(math.log(distribution['point']) + distribution['scale'] * NORMAL.inv_cdf(target_p0))

    def probability_bounds(self, context, device, level, now):
        """Paired uncalibrated/calibrated service quantiles; no coverage claim."""
        return dict(level=float(level),
                    uncalibrated_s=self.quantile(context, device, level, now, kind='uncalibrated'),
                    calibrated_s=self.quantile(context, device, level, now, kind='calibrated'))

    def register_decision(self, task_id, context, device, now, remaining,
                          probability_record, boundary_sensitivity):
        if task_id in self.pending_decisions or task_id in self.completed_decisions:
            raise ValueError('Task decision already registered/completed')
        context = tuple(context)
        if (context, device) not in self.core.base:
            raise ValueError('Unknown selected context/device key')
        now = _finite(now, 'dispatch now')
        remaining = _finite(remaining, 'dispatch remaining budget')
        if abs(now - self.core.now) > 1e-9:
            raise ValueError('Call probability/set_now at dispatch now before registering')
        if boundary_sensitivity not in (0, 1) or not isinstance(boundary_sensitivity, (int, float)):
            raise ValueError('boundary_sensitivity must be the dispatch-time binary 0/1 feature')
        forbidden = [key for key in probability_record
                     if key.lower() in ('truth', 'true_s', 'true_times', 'rho', 'full_vector',
                                        'observed', 'outcome', 'future')]
        if forbidden:
            raise ValueError('Forbidden outcome/oracle dispatch fields: ' + repr(forbidden))
        model = sim.measured_device_type(device)
        if probability_record['model'] != model:
            raise ValueError('Probability model must match selected GPU model')
        if not math.isclose(_finite(probability_record['now'], 'record now'), now, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('Probability record must come from this dispatch time')
        if not math.isclose(_finite(probability_record['remaining_s'], 'record remaining'), remaining,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Probability record must use this dispatch budget')
        record = dict(now=now, remaining_s=remaining, model=model,
                      context=context, device=device,
                      point=_positive(probability_record['point'], 'dispatch point'),
                      p0=_finite(probability_record['p0'], 'dispatch p0'),
                      p=_finite(probability_record['p'], 'dispatch p'),
                      logit0=probability_record['logit0'],
                      log_a_at_dispatch=_finite(probability_record['log_a_at_dispatch'], 'dispatch log_a'),
                      b_at_dispatch=_finite(probability_record['b_at_dispatch'], 'dispatch b'),
                      boundary_sensitivity=int(boundary_sensitivity))
        if not 0.0 <= record['p0'] <= 1.0 or not 0.0 <= record['p'] <= 1.0:
            raise ValueError('Dispatch probabilities must belong to [0, 1]')
        if remaining > 0.0:
            record['logit0'] = _finite(record['logit0'], 'dispatch logit0')
            if not probability_record['calibrated'] and self.mode != 'uncalibrated':
                raise ValueError('Trainable modes require their calibrated dispatch probability')
        elif record['p'] != 0.0 or record['p0'] != 0.0:
            raise ValueError('Nonpositive budget must have probability zero')
        self.pending_decisions[task_id] = record

    def complete_feedback(self, task_id, observed, now, **forbidden):
        if forbidden:
            raise ValueError('Only selected observed/now feedback is allowed: ' + repr(sorted(forbidden)))
        if task_id not in self.pending_decisions:
            raise ValueError('Completion has no pending dispatch decision')
        observed = _positive(observed, 'selected observed service time')
        now = _finite(now, 'completion now')
        record = self.pending_decisions[task_id]
        if now < record['now'] or now < self.core.now - 1e-12:
            raise ValueError('Completion must be nondecreasing and follow dispatch')
        remaining, model = record['remaining_s'], record['model']
        label = int(observed <= remaining)
        weight = (1.0 + self.params['boundary_lambda'] * record['boundary_sensitivity']
                  if self.mode == 'boundary' else 1.0)
        trainable = remaining > 0.0
        data_loss, data_gradient_log_a, data_gradient_b = 0.0, 0.0, 0.0
        if trainable:
            # The probability and slope below come from dispatch, even when
            # intervening completions have updated the global calibration layer.
            error = weight * (record['p'] - label)
            data_gradient_log_a = error * math.exp(record['log_a_at_dispatch']) * record['logit0']
            data_gradient_b = error
            score = math.exp(record['log_a_at_dispatch']) * record['logit0'] + record['b_at_dispatch']
            data_loss = weight * (max(score, 0.0) - label * score + math.log1p(math.exp(-abs(score))))
        before_log_a, before_b = self.log_a, dict(self.b)
        penalty = 0.5 * IDENTITY_L2 * (before_log_a ** 2 + sum(value ** 2 for value in before_b.values()))
        # Original local/raw-window/Full-risk statistics see this observation once.
        flags = self.core.observe(record['context'], record['device'], observed, record['point'], now)
        clips = 0
        if trainable and self.mode != 'uncalibrated':
            eta = self.params['eta']
            raw_log_a = before_log_a - eta * (data_gradient_log_a + IDENTITY_L2 * before_log_a)
            self.log_a = max(-LOG_A_BOUND, min(LOG_A_BOUND, raw_log_a))
            clips += int(raw_log_a != self.log_a)
            for gpu_model, intercept in before_b.items():
                gradient = IDENTITY_L2 * intercept + (data_gradient_b if gpu_model == model else 0.0)
                raw_b = intercept - eta * gradient
                self.b[gpu_model] = max(-B_BOUND, min(B_BOUND, raw_b))
                clips += int(raw_b != self.b[gpu_model])
            self.calibration_updates += 1
            self.updates_by_model[model] += 1
        if not trainable:
            self.skipped_nonpositive_budget += 1
        self.completed_feedback_count += 1
        self.parameter_clip_count += clips
        if trainable:
            self.weight_sum += weight
            self.weighted_bce_sum += data_loss
            self.brier_sum += (record['p'] - label) ** 2
        self.last_calibration = dict(task_id=task_id, now=now, model=model, label=label,
                                     remaining_s=remaining, probability_at_dispatch=record['p'],
                                     base_probability_at_dispatch=record['p0'],
                                     boundary_sensitivity=record['boundary_sensitivity'], weight=weight,
                                     trainable=trainable, updated=(trainable and self.mode != 'uncalibrated'),
                                     data_bce=data_loss, identity_penalty=penalty,
                                     data_gradient_log_a=data_gradient_log_a,
                                     data_gradient_b=data_gradient_b,
                                     log_a_at_dispatch=record['log_a_at_dispatch'],
                                     b_at_dispatch=record['b_at_dispatch'],
                                     parameter_clips=clips, log_a=self.log_a, a=self.a, b=dict(self.b))
        del self.pending_decisions[task_id]
        self.completed_decisions.add(task_id)
        result = dict(flags)
        result['calibration'] = dict(self.last_calibration)
        return result

    def calibration_diagnostics(self):
        return dict(mode=self.mode, eta=self.params['eta'], boundary_lambda=self.params['boundary_lambda'],
                    log_a=self.log_a, a=self.a, b=dict(self.b),
                    completed_feedback_count=self.completed_feedback_count,
                    calibration_updates=self.calibration_updates,
                    skipped_nonpositive_budget=self.skipped_nonpositive_budget,
                    parameter_clip_count=self.parameter_clip_count,
                    updates_by_model=dict(self.updates_by_model), weight_sum=self.weight_sum,
                    weighted_bce_sum=self.weighted_bce_sum, brier_sum=self.brier_sum,
                    pending_decisions=len(self.pending_decisions))

    def diagnostics(self, context, device):
        result = self.core.diagnostics(context, device)
        result.update(self.calibration_diagnostics())
        return result


def create_calibrator(base, mode='uncalibrated', params=None):
    return DeadlineCalibrator(base, mode, params)
