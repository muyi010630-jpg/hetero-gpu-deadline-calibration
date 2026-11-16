"""Numerical equivalence only; no experimental data or benchmark selection."""
from __future__ import annotations
import hashlib
import json
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / 'scripts' / 'calibrator_core.py'
BEFORE = Path(__file__).with_name('calibrator_core_pilot.py')


def load_source(path, name):
    module = types.ModuleType(name)
    # The preserved source retains its original relative import contract.
    module.__file__ = str(CURRENT)
    sys.modules[name] = module
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module


def main():
    before = load_source(BEFORE, 'cache_check_before')
    after = load_source(CURRENT, 'cache_check_after')
    calls = {'before': 0, 'after': 0}
    for module, label in ((before, 'before'), (after, 'after')):
        original = module._window_stats
        def counted(values, fn=original, key=label):
            calls[key] += 1
            return fn(values)
        module._window_stats = counted
    contexts = [(family, 1024 + i * 1024, 1 + i, dtype)
                for family in ('compute', 'memory', 'inference')
                for i, dtype in enumerate(('fp16', 'fp32'))]
    devices = ('RTX5070__01', 'RTX5070__02', 'T4__01', 'T4__02')
    base = {(ctx, dev): 1.0 + .03*i + .02*j
            for i, ctx in enumerate(contexts) for j, dev in enumerate(devices)}
    keys = list(base)
    cases = [(mode, params) for mode in after.MODES
             for params in (after.PARAM_GRID if mode.startswith('adaptive_') else (None,))]
    predictions = diagnostics = observations = 0
    max_delta = 0.0
    tested_modes = set()

    def compare_dict(a, b):
        nonlocal diagnostics, max_delta
        if set(a) != set(b):
            raise AssertionError(('diagnostic keys', set(a)^set(b)))
        for key in a:
            if a[key] != b[key]:
                raise AssertionError((key, a[key], b[key]))
            if isinstance(a[key], (float, int)):
                max_delta = max(max_delta, abs(a[key]-b[key]))
        diagnostics += 1

    for mode, params in cases:
        tested_modes.add(mode)
        a = before.create_calibrator(base, mode, params)
        b = after.create_calibrator(base, mode, params)

        def query(ctx, dev, now, qos, guarded):
            nonlocal predictions, max_delta
            a.set_now(now); b.set_now(now)
            pa, sa = a.predict(ctx, dev, qos, guarded)
            pb, sb = b.predict(ctx, dev, qos, guarded)
            if (pa, sa) != (pb, sb):
                raise AssertionError((mode, params, now, pa, pb, sa, sb))
            max_delta = max(max_delta, abs(pa-pb), abs(sa-sb))
            compare_dict(a.diagnostics(ctx, dev), b.diagnostics(ctx, dev))
            predictions += 1
            return pa, pb

        for i in range(180):
            key = keys[0] if i < 40 or i >= 136 else keys[(i-40) % len(keys)]
            ctx, dev = key
            now = .071*i
            pa, pb = query(ctx, dev, now, 0, False)
            query(ctx, dev, now, 2, True)  # repeated state: reuse caches
            query(contexts[(i+1) % len(contexts)], devices[(i+1) % len(devices)],
                  now+.004, i % 3, bool(i % 2))
            residual = 1.20 if i < 40 else (-1.20 if i >= 136 else .40*math.sin(i*.41)+.08*(i%5-2))
            obs = base[key] * math.exp(residual)
            da = a.observe(ctx, dev, obs, pa, now+.01)
            db = b.observe(ctx, dev, obs, pb, now+.01)
            compare_dict(da, db)
            compare_dict(a.diagnostics(ctx, dev), b.diagnostics(ctx, dev))
            observations += 1
            query(ctx, dev, now+.012, 1, True)  # invalidation after feedback
            query(ctx, dev, now+.03, 0, False)  # age change, unchanged H
        for j, (ctx, dev) in enumerate(keys):
            query(ctx, dev, 40.+j*.01, j%3, bool(j%2))
    result = dict(passed=True, tested_modes=sorted(tested_modes),
                  mode_parameter_cases=len(cases), completed_feedback_per_case=180,
                  prediction_pairs=predictions, diagnostic_dict_pairs=diagnostics,
                  observation_pairs=observations, maximum_absolute_numeric_difference=max_delta,
                  equality='Exact equality for predictions, sigma, every diagnostic key and feedback flags',
                  coverage=['empty and singleton windows', 'worker clones', 'precision/model scopes',
                            'local/unique window rollover', 'positive and negative clipping',
                            'repeated predictions', 'feedback invalidation', 'time-only changes',
                            'all nine parameter settings for all three adaptive modes'],
                  window_statistics_function_calls=calls,
                  before_sha256=hashlib.sha256(BEFORE.read_bytes()).hexdigest(),
                  after_sha256=hashlib.sha256(CURRENT.read_bytes()).hexdigest(),
                  check_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  status='No formal experiment run; pure cache equivalence check')
    destination = ROOT / '00_protocol_sources' / '缓存优化等价性检查.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
