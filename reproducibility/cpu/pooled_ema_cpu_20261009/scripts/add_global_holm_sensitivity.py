"""Additional conservative 24-test Holm sensitivity; retain the frozen primary analysis."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    analysis = ROOT / '03_mismatch_structure' / 'full' / 'analysis'
    source = analysis / 'paired_seed_inference.csv'
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['primary_test'] == 'True']
    if len(rows) != 24 or len({row['variant'] for row in rows}) != 4:
        raise RuntimeError('Need all four variants and all 24 original primary tests')
    previous = 0.0
    for i, row in enumerate(sorted(rows, key=lambda item: float(item['p_paired_signflip']))):
        adjusted = min(1.0, max(previous, (len(rows)-i) * float(row['p_paired_signflip'])))
        row['p_holm_all24_sensitivity'] = adjusted
        row['significant_all24_05_sensitivity'] = adjusted < 0.05
        previous = adjusted
    destination = analysis / 'holm_all24_sensitivity.csv'
    with destination.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (analysis / 'holm_all24_sensitivity_method.json').write_text(json.dumps({
        'role': 'Additional conservative sensitivity analysis, not a replacement for frozen within-variant primary Holm correction',
        'family': 'All four fixed mismatch variants x three point-estimator contrasts x MAPE/SLA = 24 tests',
        'source_csv': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'p_values': 'Original paired sign-flip p-values; no rerun or new bootstrap',
        'intervals': 'Original marginal confidence intervals unchanged',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved conservative all-24 Holm sensitivity')


if __name__ == '__main__':
    main()
