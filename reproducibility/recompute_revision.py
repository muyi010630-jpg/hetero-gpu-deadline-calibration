"""Recompute the revision's deployment and profile evidence using only Python's standard library.
Run after extracting the archive: python recompute_revision.py
No simulator, GPU access, parameter search or new statistical tests are performed.
"""
from pathlib import Path
from collections import defaultdict
from statistics import mean, median, stdev
import csv, json, math, hashlib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'recomputed' / 'revision'
OUT.mkdir(parents=True, exist_ok=True)

def read(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def close(actual, saved, tol=1e-8):
    assert abs(float(actual) - float(saved)) <= tol, (actual, saved)

tasks, plans = defaultdict(list), defaultdict(list)
for row in read('deployment/task_records.csv'):
    tasks[row['run_id']].append(row)
for row in read('deployment/task_plans.csv'):
    plans[row['run_id']].append(row)
assert len(tasks) == len(plans) == 60 and set(tasks) == set(plans)
audit = {r['run_id']: r for r in read('deployment/task_status_audit.csv')}
metrics = []
for run_id, records in sorted(tasks.items()):
    assert len(records) == len(plans[run_id]) == 100
    assert len({r['task_id'] for r in records}) == 100
    assert {r['task_id'] for r in records} == {r['task_id'] for r in plans[run_id]}
    assert all(r['runner_ok'] == '1' for r in records)
    for key in ('planned', 'attempt_records', 'unique_completed', 'success'):
        assert int(audit[run_id][key]) == 100
    for key in ('execution_failed', 'missing', 'duplicate_attempts'):
        assert int(audit[run_id][key]) == 0
    seed = int(records[0]['seed'])
    batch = '1' if 20260911 <= seed <= 20260915 else '2'
    assert batch == records[0]['deployment_batch'] == audit[run_id]['batch']
    assert 20260911 <= seed <= 20260920
    for row in records:
        assert int(row['gpu_sla_violation']) == int(float(row['observed_gpu_s']) > float(row['gpu_deadline_budget_s']))
        assert int(row['e2e_sla_violation']) == int(float(row['controller_receive_elapsed_s']) > float(row['e2e_deadline_s']))
    metrics.append(dict(batch=batch, configuration=records[0]['configuration'], strategy=records[0]['strategy'],
        seed=str(seed), gpu_sla_violation_rate=mean(int(r['gpu_sla_violation']) for r in records),
        e2e_sla_violation_rate=mean(int(r['e2e_sla_violation']) for r in records),
        p99_flow_time_s=sorted(float(r['controller_flow_time_s']) for r in records)[98]))

key = lambda r: (r['batch'], r['configuration'], r['strategy'], r['seed'])
saved_metrics = {key(r): r for r in read('deployment/metrics_from_tasks.csv')}
for row in metrics:
    for field in ('gpu_sla_violation_rate', 'e2e_sla_violation_rate', 'p99_flow_time_s'):
        close(row[field], saved_metrics[key(row)][field], 5.1e-7)
write('metrics_from_tasks.csv', metrics)

batch_rows = []
for saved in read('deployment/batch_absolute_and_paired.csv'):
    field = saved['metric']
    factor = 1 if field == 'p99_flow_time_s' else 100
    values = {strategy: {r['seed']: float(r[field]) * factor for r in metrics
        if r['batch'] == saved['batch'] and r['configuration'] == saved['configuration'] and r['strategy'] == strategy}
        for strategy in ('fixed', 'ema', 'full')}
    assert len(values['fixed']) == 5 and set(values['fixed']) == set(values['ema']) == set(values['full'])
    row = dict(batch=saved['batch'], configuration=saved['configuration'], metric=field,
        fixed=mean(values['fixed'].values()), ema=mean(values['ema'].values()), full_deploy=mean(values['full'].values()),
        full_minus_fixed=mean(values['full'][s] - values['fixed'][s] for s in values['full']),
        full_minus_ema=mean(values['full'][s] - values['ema'][s] for s in values['full']), paired_seeds=5)
    for name in ('fixed', 'ema', 'full_deploy', 'full_minus_fixed', 'full_minus_ema'):
        close(row[name], saved[name], 1e-7)
    batch_rows.append(row)
write('batch_absolute_and_paired.csv', batch_rows)

canonical_raw, runtime_raw = defaultdict(list), defaultdict(list)
for device in ('RTX5070', 'T4', 'A100'):
    for row in read(f'profiles/raw/{device}.csv'):
        if row['mode'] != 'gpu' or row.get('success', 'True') != 'True':
            continue
        family = {'gemm': 'compute', 'saxpy': 'memory', 'cifar10_resnet18': 'inference'}[row['family']]
        dtype = ('fp16' if row['dtype'] == 'float16' else 'fp32') if family == 'compute' else ('fp16' if family == 'inference' else 'fp32')
        canonical_raw[(device, family, int(row['size']), int(row['batch']), dtype)].append(float(row['gpu_ms']) / 1000)
for row in read('profiles/runtime_measurements.csv'):
    if row['success'] == '1':
        runtime_raw[(row['worker_id'], row['family'], int(row['size']), int(row['batch']), row['dtype'])].append(float(row['gpu_s']))
samples = read('profiles/profile_samples_and_migration.csv')
assert len(canonical_raw) == len(runtime_raw) == len(samples) == 45
for row in samples:
    cell = (row['device'], row['family'], int(row['size_label']), int(row['batch']), row['dtype'])
    for prefix, values in (('canonical', canonical_raw[cell]), ('runtime', runtime_raw[cell])):
        assert len(values) == int(row[prefix + '_n'])
        close(median(values), row[prefix + '_median_s'], 5.1e-10)
        for name, value in (('min', min(values)), ('max', max(values)), ('sd', stdev(values))):
            close(value, row[f'{prefix}_{name}_s'])
    close(math.log(float(row['runtime_median_s']) / float(row['canonical_median_s'])), row['log_ratio'])
    row['log_ratio'] = float(row['log_ratio'])
grand = mean(r['log_ratio'] for r in samples)
dm = {d: mean(r['log_ratio'] for r in samples if r['device'] == d) for d in ('RTX5070', 'T4', 'A100')}
fm = {f: mean(r['log_ratio'] for r in samples if r['family'] == f) for f in ('compute', 'memory', 'inference')}
gm = {(d, f): mean(r['log_ratio'] for r in samples if (r['device'], r['family']) == (d, f)) for d in dm for f in fm}
ss = dict(total=sum((r['log_ratio'] - grand) ** 2 for r in samples),
    device=sum((dm[r['device']] - grand) ** 2 for r in samples),
    family=sum((fm[r['family']] - grand) ** 2 for r in samples),
    interaction=sum((gm[r['device'], r['family']] - dm[r['device']] - fm[r['family']] + grand) ** 2 for r in samples),
    within_group=sum((r['log_ratio'] - gm[r['device'], r['family']]) ** 2 for r in samples))
close(ss['total'], sum(v for k, v in ss.items() if k != 'total'))
saved = json.loads((ROOT / 'profiles/decomposition.json').read_text(encoding='utf-8'))
for name, value in ss.items():
    close(value, saved['sums_of_squares'][name])
shares = {k: 100 * v / ss['total'] for k, v in ss.items() if k != 'total'}
for name, value in shares.items():
    close(value, saved['percentage_shares'][name])

report = dict(status='PASS', runs=60, planned_tasks=6000, successful_tasks=6000,
    failed_missing_or_duplicate=0, batch_metric_rows=len(batch_rows), profile_cells=45,
    descriptive_percentage_shares=shares, scope='Saved task records, saved configurations and profile samples; no new experiments or tests.')
(OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
