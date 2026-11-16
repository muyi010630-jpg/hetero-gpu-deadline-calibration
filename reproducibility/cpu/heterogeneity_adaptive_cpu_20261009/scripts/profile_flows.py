"""Deterministic controlled portraits and common CPU feedback streams.

rho is the realized between-group variance fraction over 45 unique cells,
not an iid context correlation. Repeated device instances are not replicates.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parent / 'pooled_ema_cpu_20261009' / '00_configuration_sources' / 'profiles_P0' / 'cluster_08_profile.csv'
CACHE = ROOT / '00_protocol_sources' / 'profile_flow_cache'
FLOW_VERSION = 'controlled_flows_v1'
FLOW_DEFINITION = dict(version=FLOW_VERSION, balanced='independent randomized permutations of all 45 unique cells', late_count=22, late_onset_fraction=0.4, late_per_cell_weight=0.25, noise_log_sd=0.08, now_end_s=12.0, drift_onset_fraction=0.5, drift_log_amplitude=0.15, rng='numpy.PCG64', source=str(SOURCE))

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)

def load_base_contexts():
    unique = {}
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            model = row['device'].split('__')[0]
            key = (model, row['family'], int(row['size']), int(row['batch']), row['dtype'])
            truth = float(row['true_s'])
            if key in unique:
                assert abs(unique[key]['true_s'] - truth) < 1e-12, 'Clones disagree in measured truth'
            else:
                unique[key] = dict(model=model, worker=model+'__01', family=row['family'], size=int(row['size']), batch=int(row['batch']), dtype=row['dtype'], true_s=truth, vendor=row['vendor'], memory_gb=float(row['memory_gb']))
    rows = [unique[key] for key in sorted(unique)]
    groups = sorted({(row['model'], row['family']) for row in rows})
    for i, row in enumerate(rows):
        row['context_id'] = i
        row['group_id'] = groups.index((row['model'], row['family']))
        row['context'] = (row['family'], row['size'], row['batch'], row['dtype'])
        row['key'] = '|'.join(map(str, (row['model'],) + row['context']))
    assert len(rows) == 45 and len(groups) == 9
    return rows

def standardized_components(image_seed, rows=None):
    rows = load_base_contexts() if rows is None else rows
    gids = np.asarray([row['group_id'] for row in rows], dtype=np.int16)
    rng = np.random.default_rng(image_seed)
    a = rng.normal(size=9)[gids]
    a -= a.mean(); a /= np.sqrt(np.mean(a*a))
    e = rng.normal(size=len(rows))
    for gid in range(9):
        mask = gids == gid; e[mask] -= e[mask].mean()
    e /= np.sqrt(np.mean(e*e))
    assert abs(a.mean()) < 1e-12 and abs(e.mean()) < 1e-12 and abs(np.mean(a*e)) < 1e-12
    return a, e

def make_profile(image_seed, rho, rows=None, persist=True):
    rows = load_base_contexts() if rows is None else rows
    a, e = standardized_components(image_seed, rows)
    z = 0.20 * (math.sqrt(rho)*a + math.sqrt(1.0-rho)*e)
    truth = np.asarray([row['true_s'] for row in rows], dtype=float)
    gids = np.asarray([row['group_id'] for row in rows], dtype=np.int16)
    group_mean = np.asarray([z[gids == gid].mean() for gid in range(9)])
    between = float(np.mean(group_mean[gids]**2))
    assert abs(z.mean()) < 1e-12 and abs(np.mean(z*z)-0.04) < 1e-12 and abs(between/0.04-rho) < 1e-12
    profile = dict(image_seed=np.asarray(image_seed), rho=np.asarray(rho), z_log=z, a=a, e=e, initial_s=truth*np.exp(-z), true_s=truth, group_ids=gids, context_ids=np.arange(len(rows), dtype=np.int16), keys=np.asarray([row['key'] for row in rows]), models=np.asarray([row['model'] for row in rows]), workers=np.asarray([row['worker'] for row in rows]), families=np.asarray([row['family'] for row in rows]), sizes=np.asarray([row['size'] for row in rows], dtype=np.int64), batches=np.asarray([row['batch'] for row in rows], dtype=np.int16), dtypes=np.asarray([row['dtype'] for row in rows]), source_sha256=np.asarray(sha256(SOURCE)), version=np.asarray(FLOW_VERSION))
    if persist:
        atomic_npz(CACHE/'profiles'/f'image_{image_seed}_rho_{rho:g}.npz', **profile)
    return profile

def late_mask(rows=None):
    rows = load_base_contexts() if rows is None else rows
    ranked = sorted(range(len(rows)), key=lambda i: hashlib.sha256(('late_sparse_v1|'+rows[i]['key']).encode()).hexdigest())
    mask = np.zeros(len(rows), dtype=bool); mask[ranked[:22]] = True
    return mask

def drift_direction(image_seed, drift_type, rows=None):
    rows = load_base_contexts() if rows is None else rows
    gids = np.asarray([row['group_id'] for row in rows])
    rng = np.random.default_rng(np.random.SeedSequence([image_seed, 918271]))
    if drift_type == 'group':
        direction = rng.normal(size=9)[gids]; direction -= direction.mean()
    elif drift_type == 'context':
        direction = rng.normal(size=45)
        for gid in range(9):
            mask = gids == gid; direction[mask] -= direction[mask].mean()
    else:
        return np.zeros(45)
    direction /= np.sqrt(np.mean(direction*direction))
    return direction

def make_stream(profile, task_seed, flow_kind, n_events, drift_type='none', rows=None):
    rows = load_base_contexts() if rows is None else rows
    late = late_mask(rows)
    # Context selection and observation noise have separate deterministic RNGs.
    # Image/rho/drift type are absent from these seeds: common task stream.
    selector = np.random.default_rng(np.random.SeedSequence([task_seed, 51973, 0 if flow_kind == 'balanced' else 1]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([task_seed, 71933]))
    if flow_kind == 'balanced':
        ids = np.concatenate([selector.permutation(45) for _ in range(math.ceil(n_events/45))])[:n_events]
    elif flow_kind == 'late_sparse':
        cut = int(math.floor(0.4*n_events))
        early_ids = np.flatnonzero(~late)
        weights = np.where(late, 0.25, 1.0); weights /= weights.sum()
        ids = np.concatenate((selector.choice(early_ids, size=cut), selector.choice(45, size=n_events-cut, p=weights)))
        assert not np.any(late[ids[:cut]])
    else:
        raise ValueError('Unknown flow: '+flow_kind)
    ids = ids.astype(np.int16)
    noise = noise_rng.normal(0.0, 0.08, n_events)
    direction = drift_direction(int(profile['image_seed']), drift_type, rows)
    active = np.arange(n_events) >= int(n_events*0.5)
    true_median = profile['true_s'][ids] * np.exp(0.15*direction[ids]*active)
    obs = true_median*np.exp(noise)
    gids = profile['group_ids'][ids]
    own_count = np.zeros(45, dtype=np.int32); group_count = np.zeros(9, dtype=np.int32)
    n_before = np.empty(n_events, dtype=np.int32); group_before = np.empty(n_events, dtype=np.int32)
    for i, cid in enumerate(ids):
        n_before[i] = own_count[cid]; group_before[i] = group_count[gids[i]]
        own_count[cid] += 1; group_count[gids[i]] += 1
    return dict(context_ids=ids, obs=obs, true_median=true_median, noise_log=noise, now=np.linspace(0.0, 12.0, n_events), n_before=n_before, group_n_before=group_before, group_ids=gids, late_context_mask=late, drift_direction=direction, drift_type=np.asarray(drift_type), task_seed=np.asarray(task_seed), flow_kind=np.asarray(flow_kind), n_events=np.asarray(n_events))

def base_dict(profile):
    return {((str(profile['families'][i]), int(profile['sizes'][i]), int(profile['batches'][i]), str(profile['dtypes'][i])), str(profile['workers'][i])): float(profile['initial_s'][i]) for i in range(45)}

def build_image(image_seed, rho):
    """Shared closed-loop API; clone rows only after this unique-cell construction."""
    rows = load_base_contexts()
    # Parallel closed-loop workers must not repeatedly replace a shared NPZ.
    # The fixed driver owns cache persistence in its sequential prepare phase.
    profile = make_profile(image_seed, rho, rows, persist=False)
    return [dict(row, initial_s=float(profile['initial_s'][i]), z_log=float(profile['z_log'][i]), shared_a=float(profile['a'][i]), within_e=float(profile['e'][i])) for i,row in enumerate(rows)]

def prepare(stage, image_seeds, rhos, persist=True):
    rows = load_base_contexts(); summaries = []
    for image_seed in image_seeds:
        for rho in rhos:
            profile = make_profile(image_seed, rho, rows, persist)
            z = profile['z_log']; gids=profile['group_ids']
            within = float(np.mean((z-np.asarray([z[gids==g].mean() for g in range(9)])[gids])**2))
            summaries.append(dict(stage=stage,image_seed=image_seed,rho=rho,n_unique=45,n_groups=9,mean_z=float(z.mean()),sd_z=float(z.std()),between_fraction=1-within/0.04,a_e_inner_product=float(np.mean(profile['a']*profile['e']))))
    return summaries

if __name__ == '__main__':
    checks=prepare('smoke',[11001],[0,.25,.5,.75,1])
    atomic_json(CACHE/'generation_definition.json',dict(**FLOW_DEFINITION,source_sha256=sha256(SOURCE),profile_script_sha256=sha256(__file__),late_context_ids=np.flatnonzero(late_mask()).tolist(),smoke_moments=checks))
    print(json.dumps(checks,ensure_ascii=False))
