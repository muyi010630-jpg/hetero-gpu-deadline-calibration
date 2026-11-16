"""Strengthen baseline controls before any heldout execution, without rerunning outcomes."""
import json
import shutil
import sys
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
import run_feasibility as run
if (root/'02_heldout_closed_loop/completion.json').exists() or (root/'02_heldout_closed_loop/raw').exists():
    raise RuntimeError('Cannot change comparisons after heldout execution begins')
archive=root/'01_development_validation/pretest_control_corrections';archive.mkdir(parents=True,exist_ok=True)
for source in (root/'00_design_sources/final_freeze.json',root/'01_development_validation/参数选择.json',
               root/'01_development_validation/config.json'):
    shutil.copy2(source,archive/source.name)
rows=pd.read_csv(root/'01_development_validation/raw/per_run_results.csv').to_dict('records')
freeze=run.select_params(rows)
freeze['pretest_control_amendment']=dict(no_heldout_execution=True,
    validation_original_code_fingerprint=rows[0]['code_fingerprint'],
    validation_results_sha256=run.sha(root/'01_development_validation/raw/per_run_results.csv'),
    note='Only selectors, comparison list, ordinary-adapter probe control and source locking extended; all validation trajectories preserved.')
run.atomic_json(root/'00_design_sources/final_freeze.json',freeze)
print(json.dumps(dict(selected=freeze['selected'],test_arms=len(freeze['test_arms']),
                     primary_comparisons=len(freeze['analysis']['primary_comparisons'])),ensure_ascii=False))
