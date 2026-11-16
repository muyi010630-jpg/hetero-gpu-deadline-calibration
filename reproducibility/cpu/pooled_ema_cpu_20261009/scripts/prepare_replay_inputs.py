from __future__ import annotations
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent.parent
SOURCE = WORKSPACE / '实验补充_20260921' / '重做实验_20260921'
CONFIG = ROOT / '00_configuration_sources'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pairs = []
    for filename in ('hetero_qos_sim_n2.py', 'run_n2_cpu.py', 'run_n2_cpu_par.py'):
        pairs.append((SOURCE / 'N2_scheduler_baselines/scripts' / filename,
                      CONFIG / 'frozen_sources' / filename))
    for filename in ('run_n9_profile_robustness.py', 'build_profile_variants.py'):
        pairs.append((SOURCE / 'N9_profile_robustness/scripts' / filename,
                      CONFIG / 'frozen_sources' / filename))
    pairs.append((SOURCE / 'N2_scheduler_baselines/raw/full/per_seed_results.csv',
                  CONFIG / 'golden/n2_per_seed_results.csv'))
    pairs.append((SOURCE / 'N9_profile_robustness/raw/per_seed_results.csv',
                  CONFIG / 'golden/n9_per_seed_results.csv'))
    profile_root = WORKSPACE / '修补实验2_deadline重设_20260907/experiment_C2_replay_deadline_fixed/profiles_expanded'
    for size in (2, 4, 8, 16, 32):
        pairs.append((profile_root / f'cluster_{size:02d}_profile.csv',
                      CONFIG / 'profiles_replay' / f'cluster_{size:02d}_profile.csv'))
        for variant in ('P0', 'P1', 'P2', 'P3'):
            pairs.append((SOURCE / 'N9_profile_robustness' / f'profiles_{variant}' / f'cluster_{size:02d}_profile.csv',
                          CONFIG / f'profiles_{variant}' / f'cluster_{size:02d}_profile.csv'))
    manifest = []
    for original, frozen in pairs:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists() and sha256(frozen) != sha256(original):
            raise RuntimeError(f'Existing frozen source differs: {frozen}')
        shutil.copy2(original, frozen)
        manifest.append(dict(source=str(original), frozen=str(frozen), sha256=sha256(frozen), bytes=frozen.stat().st_size))
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / 'replay_source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    with (CONFIG / 'replay_source_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys()); w.writeheader(); w.writerows(manifest)
    print(json.dumps({'frozen_files': len(manifest), 'all_hashes_verified': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
