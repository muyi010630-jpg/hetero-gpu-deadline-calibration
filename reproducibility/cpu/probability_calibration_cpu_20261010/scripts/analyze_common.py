"""Read-only, common-feedback probability diagnostics; never choose parameters.

Each bank learns from exactly the selected feedback of one Window trajectory.
Analytic probabilities are evaluation labels only. Candidate metrics average
devices within a GPU model, then models within a dispatch, then dispatches.
Inference is descriptive and uses four independent mismatch-image blocks.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BANKS = ('uncalibrated', 'uniform', 'boundary1', 'boundary3')
IMAGE_SEEDS = tuple(range(51001, 51005))
TASK_SEEDS = (61001, 61002)
RHOS = (0.0, 0.5, 1.0)
LOADS = (0.75, 1.70)
HORIZONS = (24.0, 120.0)
SHARED_KEYS = ('point', 'scale', 'n', 'sample_variance', 'blend_weight',
               'blended_variance', 'now', 'remaining_s', 'model', 'raw_p0',
               'p0', 'logit0', 'trainable')
SCOPES = ('all_idle_candidates_expected_positive',
          'selected_expected_positive', 'selected_outcome_positive',
          'selected_outcome_all')
METRICS = ('brier', 'logloss', 'prob_squared_error')
EPS = 1e-6
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20261010


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_equal(a, b, message):
    if a != b:
        raise ValueError(message + ': ' + repr(a) + ' != ' + repr(b))


def assert_close(a, b, message):
    if not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(message + ': ' + repr(a) + ' != ' + repr(b))


def probability_losses(p, q):
    if not (math.isfinite(p) and math.isfinite(q) and 0 <= p <= 1 and 0 <= q <= 1):
        raise ValueError('Nonfinite/out-of-range probability')
    clipped = min(1 - EPS, max(EPS, p))
    return dict(brier=p * p - 2 * p * q + q,
                logloss=-q * math.log(clipped) - (1 - q) * math.log1p(-clipped),
                prob_squared_error=(p - q) ** 2)


def average_losses(losses):
    if not losses:
        raise ValueError('Empty positive-budget diagnostic scope')
    return {metric: float(np.mean([item[metric] for item in losses])) for metric in METRICS}


def analyze_events(events, ordinary_completions=None):
    """Validate legal common feedback and return bank/scope run means."""
    pending = {}
    dispatched = set()
    finished = set()
    losses = {bank: {scope: [] for scope in SCOPES} for bank in BANKS}
    now_previous = -math.inf
    sensitive_positive = 0
    positive = 0
    candidate_device_counts = []
    candidate_model_counts = []
    selected_models = defaultdict(int)
    gradient_norm_sum = defaultdict(float)
    clip_counts = defaultdict(int)
    regular_completion = {}
    if ordinary_completions is not None:
        for row in ordinary_completions:
            if row['task_id'] in regular_completion:
                raise ValueError('Duplicate physical completion')
            regular_completion[row['task_id']] = row
    for event in events:
        now = float(event['now'])
        if not math.isfinite(now) or now < now_previous - 1e-9:
            raise ValueError('Common feedback events are not chronological')
        now_previous = now
        task = event['task_id']
        if event['event'] == 'dispatch':
            if task in dispatched:
                raise ValueError('Duplicate dispatch')
            dispatched.add(task)
            assert_equal(set(event['predictions']), set(BANKS), 'Unexpected banks')
            predictions = event['predictions']
            baseline = predictions['uncalibrated']
            candidates = set(baseline)
            if not candidates or event['selected'] not in candidates:
                raise ValueError('No legal selected candidate')
            assert_equal(set(event['true_probabilities_evaluation_only']), candidates,
                         'Analytic evaluation panel does not match candidates')
            for bank in BANKS:
                assert_equal(set(predictions[bank]), candidates, 'Different candidate sets')
                for device in candidates:
                    record = predictions[bank][device]
                    reference = baseline[device]
                    for key in SHARED_KEYS:
                        assert_equal(record[key], reference[key], 'Shared dispatch feature ' + key)
                    assert_close(record['remaining_s'], event['remaining'], 'Dispatch remaining budget')
                    assert_close(record['now'], now, 'Dispatch timestamp')
                    if not (math.isfinite(record['p']) and 0 <= record['p'] <= 1):
                        raise ValueError('Invalid online probability')
                    if event['remaining'] <= 0:
                        assert_equal(record['p'], 0.0, 'Expired probability must be structural zero')
            assert_equal(baseline[event['selected']]['model'], event['model'], 'Selected model')
            assert_equal(baseline[event['selected']]['p'], baseline[event['selected']]['p0'],
                         'Uncalibrated identity probability')
            if event['boundary_sensitivity'] not in (0, 1):
                raise ValueError('Invalid pre-dispatch boundary feature')
            pending[task] = event
            selected_models[event['model']] += 1
            candidate_device_counts.append(len(candidates))
            candidate_model_counts.append(len({baseline[d]['model'] for d in candidates}))
            if event['remaining'] > 0:
                positive += 1
                sensitive_positive += event['boundary_sensitivity']
                for bank in BANKS:
                    by_model = defaultdict(list)
                    for device in candidates:
                        rec = predictions[bank][device]
                        by_model[rec['model']].append(probability_losses(
                            rec['p'], event['true_probabilities_evaluation_only'][device]))
                    model_means = [average_losses(values) for values in by_model.values()]
                    losses[bank]['all_idle_candidates_expected_positive'].append(average_losses(model_means))
                    chosen = event['selected']
                    losses[bank]['selected_expected_positive'].append(probability_losses(
                        predictions[bank][chosen]['p'], event['true_probabilities_evaluation_only'][chosen]))
        elif event['event'] == 'completion':
            if task not in pending or task in finished:
                raise ValueError('Completion without a unique pending decision')
            dispatch = pending.pop(task)
            finished.add(task)
            observed = float(event['observed_s'])
            if not math.isfinite(observed) or observed <= 0:
                raise ValueError('Invalid selected observation')
            assert_close(now, dispatch['now'] + observed, 'Physical completion timestamp')
            label = int(observed <= dispatch['remaining'])
            assert_equal(event['on_time'], label, 'Completion label')
            assert_equal(set(event['learning']), set(BANKS), 'Incomplete feedback bank')
            shared_flags = None
            for bank in BANKS:
                result = event['learning'][bank]
                calibration = result['calibration']
                record = dispatch['predictions'][bank][dispatch['selected']]
                assert_equal(calibration['task_id'], task, 'Learning task')
                assert_equal(calibration['label'], label, 'Shared selected feedback label')
                assert_equal(calibration['model'], dispatch['model'], 'Learning model')
                assert_close(calibration['remaining_s'], dispatch['remaining'], 'Learning dispatch budget')
                assert_close(calibration['now'], now, 'Learning completion timestamp')
                assert_close(calibration['probability_at_dispatch'], record['p'], 'Saved delayed probability')
                assert_close(calibration['base_probability_at_dispatch'], record['p0'], 'Saved base probability')
                assert_equal(calibration['boundary_sensitivity'], dispatch['boundary_sensitivity'], 'Shared boundary feature')
                expected_weight = 1.0 + {'uncalibrated': 0.0, 'uniform': 0.0,
                                         'boundary1': 1.0, 'boundary3': 3.0}[bank] * dispatch['boundary_sensitivity']
                assert_close(calibration['weight'], expected_weight, 'Frozen pre-dispatch weight')
                assert_equal(calibration['trainable'], dispatch['remaining'] > 0, 'Trainable budget')
                assert_equal(calibration['updated'], dispatch['remaining'] > 0 and bank != 'uncalibrated',
                             'Selected-only probability update')
                flags = {key: value for key, value in result.items() if key != 'calibration'}
                if shared_flags is None:
                    shared_flags = flags
                else:
                    assert_equal(flags, shared_flags, 'Shared Window feedback state/flags')
                gradient_norm_sum[bank] += math.hypot(calibration['data_gradient_log_a'], calibration['data_gradient_b'])
                clip_counts[bank] += calibration['parameter_clips']
                outcome = probability_losses(record['p'], float(label))
                outcome['prob_squared_error'] = None  # True probability is not an outcome label.
                losses[bank]['selected_outcome_all'].append(outcome)
                if dispatch['remaining'] > 0:
                    losses[bank]['selected_outcome_positive'].append(outcome)
            if ordinary_completions is not None:
                if task not in regular_completion:
                    raise ValueError('Shadow feedback absent from physical trace')
                physical = regular_completion[task]
                assert_equal(physical['device'], dispatch['selected'], 'Shared physical selected device')
                assert_close(physical['observed_s'], observed, 'Shared physical selected duration')
                assert_close(physical['remaining'], dispatch['remaining'], 'Shared physical dispatch budget')
                assert_equal(physical['on_time'], label, 'Shared physical completion label')
        else:
            raise ValueError('Unknown common feedback event')
    if pending or dispatched != finished:
        raise ValueError('Incomplete common-feedback drain')
    if ordinary_completions is not None:
        assert_equal(set(regular_completion), finished, 'Every physical feedback supplied exactly once')
    if not positive:
        raise ValueError('No positive-budget decisions')
    rows = []
    for bank in BANKS:
        for scope in SCOPES:
            values = losses[bank][scope]
            row = dict(bank=bank, scope=scope, observations=len(values),
                       dispatches=len(dispatched), positive_dispatches=positive,
                       positive_boundary_sensitive_fraction=sensitive_positive / positive,
                       mean_candidate_devices=float(np.mean(candidate_device_counts)),
                       mean_candidate_models=float(np.mean(candidate_model_counts)),
                       data_gradient_norm_sum=gradient_norm_sum[bank],
                       parameter_clip_count=clip_counts[bank])
            for metric in METRICS:
                row[metric] = (float(np.mean([v[metric] for v in values]))
                               if all(v[metric] is not None for v in values) else '')
            rows.append(row)
    verification = dict(dispatches=len(dispatched), completions=len(finished),
                        positive_dispatches=positive, structural_zero_dispatches=len(dispatched) - positive,
                        sensitive_positive_dispatches=sensitive_positive,
                        selected_models=dict(selected_models), banks=list(BANKS),
                        shared_point_p0_scale_n=True, shared_selected_feedback=True,
                        analytic_truth_used_for_evaluation_only=True)
    return rows, verification


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError('Empty analysis table')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def block_rows(run_rows, image_seeds=IMAGE_SEEDS, task_seeds=TASK_SEEDS,
               rhos=RHOS, loads=LOADS, horizons=HORIZONS):
    keyed = {}
    for row in run_rows:
        key = (row['image_seed'], row['rho'], row['task_seed'], row['load'], row['horizon'], row['bank'], row['scope'])
        if key in keyed:
            raise ValueError('Duplicate common diagnostic cell')
        keyed[key] = row
    expected = set(itertools.product(image_seeds, rhos, task_seeds, loads, horizons, BANKS, SCOPES))
    assert_equal(set(keyed), expected, 'Complete common-feedback design')
    blocks = []
    for image, bank, scope in itertools.product(image_seeds, BANKS, SCOPES):
        for slice_name, slice_horizons in [('overall', horizons)] + [('horizon_' + str(int(h)), (h,)) for h in horizons]:
            output = dict(image_seed=image, bank=bank, scope=scope, slice=slice_name,
                          independent_block_count=1, condition_cells=len(rhos) * len(loads) * len(slice_horizons))
            for metric in METRICS:
                conditions = []
                for rho, load, horizon in itertools.product(rhos, loads, slice_horizons):
                    values = [keyed[(image, rho, task, load, horizon, bank, scope)][metric] for task in task_seeds]
                    conditions.append(float(np.mean(values)) if all(v != '' for v in values) else '')
                output[metric] = float(np.mean(conditions)) if all(v != '' for v in conditions) else ''
            blocks.append(output)
    return blocks


def paired_diagnostics(blocks, image_seeds=IMAGE_SEEDS):
    keyed = {(r['image_seed'], r['bank'], r['scope'], r['slice']): r for r in blocks}
    scopes_slices = sorted({(r['scope'], r['slice']) for r in blocks})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(image_seeds), size=(BOOTSTRAPS, len(image_seeds)))
    pairs, means = [], []
    for scope, slice_name in scopes_slices:
        for metric in METRICS:
            reference = [keyed[(i, 'uncalibrated', scope, slice_name)][metric] for i in image_seeds]
            if any(v == '' for v in reference):
                continue
            reference = np.asarray(reference, dtype=float)
            for bank in BANKS:
                values = np.asarray([keyed[(i, bank, scope, slice_name)][metric] for i in image_seeds], dtype=float)
                means.append(dict(bank=bank, scope=scope, slice=slice_name, metric=metric,
                                  mean=float(values.mean()), independent_image_blocks=len(image_seeds)))
                if bank != 'uncalibrated':
                    delta = values - reference
                    distribution = delta[indices].mean(axis=1)
                    low, high = np.quantile(distribution, (.025, .975))
                    pairs.append(dict(bank_a=bank, bank_b='uncalibrated', scope=scope,
                                      slice=slice_name, metric=metric, difference_a_minus_b=float(delta.mean()),
                                      ci_low=float(low), ci_high=float(high),
                                      bootstrap_resamples=BOOTSTRAPS, independent_image_blocks=len(image_seeds),
                                      unit='dimensionless', inference='descriptive_development_only'))
    return pairs, means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / '01_common_feedback_diagnostics'
    completion = json.loads((folder / 'completion.json').read_text(encoding='utf-8'))
    assert_equal(completion['complete'], True, 'Formal stage completed')
    assert_equal(completion['units'], 96, 'Formal common units')
    assert_equal(completion['runs'], 96, 'Formal common runs')
    with (folder / 'raw/per_run_results.csv').open(encoding='utf-8-sig', newline='') as source:
        metadata = list(csv.DictReader(source))
    assert_equal(len(metadata), 96, 'Formal common result rows')
    units = set()
    run_rows, checks = [], []
    fingerprints = set()
    for row in metadata:
        assert_equal(row['stage'], 'common', 'Diagnostic stage')
        assert_equal(row['mode'], 'window_original', 'Common fixed physical policy')
        unit = (int(row['image_seed']), float(row['rho']), int(row['task_seed']), float(row['load']), float(row['horizon']))
        if unit in units:
            raise ValueError('Duplicate common physical run')
        units.add(unit)
        fingerprints.add(row['code_fingerprint'])
        trace_path = (root / row['trace_path']).resolve()
        if root not in trace_path.parents:
            raise ValueError('Trace path escapes experiment directory')
        assert_equal(sha(trace_path), row['trace_sha256'], 'Raw trace hash')
        events, physical = [], []
        with gzip.open(trace_path, 'rt', encoding='utf-8') as trace:
            for line in trace:
                event = json.loads(line)
                if event['kind'] == 'common_event':
                    events.append(event)
                elif event['kind'] == 'completion':
                    physical.append(event)
        summaries, verification = analyze_events(events, physical)
        assert_equal(verification['dispatches'], int(row['arrived']), 'All arrivals drained')
        assert_equal(verification['completions'], int(row['completed']), 'All selected completions observed')
        common = dict(image_seed=unit[0], rho=unit[1], task_seed=unit[2], load=unit[3], horizon=unit[4],
                      run_id=row['run_id'], trace_sha256=row['trace_sha256'], plan_sha256=row['plan_sha256'])
        run_rows.extend(dict(common, **summary) for summary in summaries)
        checks.append(dict(common, **verification))
    assert_equal(units, set(itertools.product(IMAGE_SEEDS, RHOS, TASK_SEEDS, LOADS, HORIZONS)), 'Formal diagnostic grid')
    assert_equal(len(fingerprints), 1, 'Common code fingerprint')
    blocks = block_rows(run_rows)
    pairs, means = paired_diagnostics(blocks)
    output = folder / 'analysis'
    write_csv(output / 'per_run_probability.csv', run_rows)
    write_csv(output / 'image_block_probability.csv', blocks)
    write_csv(output / 'paired_diagnostic.csv', pairs)
    write_csv(output / 'bank_summary.csv', means)
    write_json(output / 'verification.json', dict(status='PASS', checked_at_utc=datetime.now(timezone.utc).isoformat(),
               units=len(units), runs=len(metadata), shared_feedback_banks=list(BANKS),
               source_result_sha256=sha(folder / 'raw/per_run_results.csv'),
               analyzer_sha256=sha(Path(__file__)), code_fingerprint=next(iter(fingerprints)),
               bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAPS,
               all_runs_checked=checks, forbidden_use='No parameter selection or closed-loop QoS claim'))
    lines = ['# 共同反馈概率诊断', '',
             '本诊断属于开发阶段。四个影子校准器接收同一条 Window 调度轨迹的选中设备完成反馈，学习率固定为 0.02；边界加权系数分别为 1 和 3。影子预测不改变真实调度选择。解析真实概率仅用于评价，未进入训练。', '',
             '已经逐条核对候选集合、point/p0/scale/n、完成标签、保存的派发概率、边界权重及 Window 更新标志。96 个流全部完整完成。', '',
             '全部空闲候选的期望损失先在 GPU 型号内平均设备，再对型号等权平均；随后在单流内平均派发。选中设备的真实结果单独评价。正剩余截止期评价排除派发时已过期的任务；全任务结果另列，避免结构零概率稀释比较。', '',
             '统计先平均两个任务随机种子，再对 rho、负载和回放时长条件等权平均，使用四个失配画像作为独立块。以下区间为 5,000 次画像块配对自助法的描述性 95% 区间，样本量仅四块，不作为确认性检验；未进行参数选择或多重检验后的显著性主张。Brier、Logloss 均为无量纲损失，负差值表示改善。', '',
             '|评价范围|影子校准器|指标|相对未校准的差值|描述性95%区间|',
             '|---|---|---|---:|---|']
    scope_labels = {'all_idle_candidates_expected_positive': '全部空闲候选（正剩余截止期）',
                    'selected_expected_positive': '选中设备期望损失（正剩余截止期）',
                    'selected_outcome_positive': '选中设备真实结果（正剩余截止期）',
                    'selected_outcome_all': '选中设备真实结果（全部任务）'}
    for row in pairs:
        if row['slice'] == 'overall' and row['metric'] in ('brier', 'logloss'):
            lines.append(f"|{scope_labels[row['scope']]}|{row['bank_a']}|{row['metric']}|{row['difference_a_minus_b']:.6f}|[{row['ci_low']:.6f}, {row['ci_high']:.6f}]|")
    lines += ['', '本诊断可以区分相同观测条件下概率质量的变化，不能证明未选中设备得到了充分校准，也不能证明 SLA 收益。不同影子模型拥有相同点预测，实际服务时间与调度轨迹相同。是否改善闭环 QoS，必须由冻结参数后的独立留出实验判断。', '']
    (output / '共同反馈诊断说明.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status='PASS', units=96, summary_rows=len(run_rows), image_block_rows=len(blocks), paired_rows=len(pairs)), ensure_ascii=False))


if __name__ == '__main__':
    main()
