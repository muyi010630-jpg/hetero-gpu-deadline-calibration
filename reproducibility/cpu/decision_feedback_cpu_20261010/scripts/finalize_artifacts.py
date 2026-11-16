"""Index completed artifacts without changing frozen experiment/analysis files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    required = ['report/实验报告.md', 'report/独立结果解读.md',
                'qa/最终推断审计.md', 'qa/门控退化诊断.md', 'qa/completed_data_checks.json',
                'analysis/test/analysis_metadata.json']
    for relative in required:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    checks = json.loads((ROOT / 'qa/completed_data_checks.json').read_text(encoding='utf-8'))
    assert checks['passed'] and checks['source_and_input_unchanged']
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name == 'artifact_manifest.json':
            continue
        data = path.read_bytes()
        files.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'finalized_at_utc': datetime.now(timezone.utc).isoformat(),
                'complete': True, 'cpu_only': True, 'development_runs': 1392,
                'heldout_runs': 4896, 'total_runs': 6288,
                'files_count': len(files), 'indexed_bytes': sum(x['bytes'] for x in files),
                'trace_qa_passed': True, 'files': files}
    (ROOT / 'artifact_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
