"""Recompute the published H-event summaries from retained diagnostic fields."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('recomputed/h_events'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    event_root = root / 'supplemental_analyses/diagnostics/h_events'
    output = args.output_dir.resolve()
    if output.is_relative_to(root / 'supplemental_analyses'):
        parser.error('Choose an output directory outside the supplied diagnostic records.')
    output.mkdir(parents=True, exist_ok=True)
    with (event_root / 'source_to_extract_sha256.csv').open(encoding='utf-8-sig', newline='') as stream:
        receipts = list(csv.DictReader(stream))
    paths = sorted((event_root / 'raw').glob('*.npz'))
    if len(paths) != 1000 or len(receipts) != 1000:
        raise ValueError('Expected exactly 1000 H-event extracts and receipts.')
    for receipt in receipts:
        path = event_root / receipt['extract_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['extract_sha256']:
            raise ValueError(f'Extract checksum mismatch for {path.name}')
    spec = importlib.util.spec_from_file_location(
        'saved_diagnostics', root / 'supplemental_analyses/diagnostics/summarize_saved_diagnostics.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = output
    module.heterogeneity_summary(paths)
    for name in ['heterogeneity_per_stream.csv', 'heterogeneity_per_image.csv', 'heterogeneity_weight_by_rho_flow.csv']:
        expected = pd.read_csv(root / 'supplemental_analyses/diagnostics' / name)
        actual = pd.read_csv(output / name)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    result = {
        'status': 'PASS',
        'streams': len(paths),
        'prediction_events': len(paths) * 1800,
        'condition_rows': 10,
        'reference_tables_matched': 3,
        'analysis': 'descriptive recomputation of retained events using the archived aggregation function',
    }
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
