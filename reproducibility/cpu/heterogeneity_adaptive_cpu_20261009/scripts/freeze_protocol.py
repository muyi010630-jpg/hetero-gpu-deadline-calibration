"""Freeze validation-selected parameters and formal code before heldout runs."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    path=ROOT/'00_protocol_sources/final_freeze.json'
    if path.exists():raise RuntimeError('Existing formal freeze must be preserved')
    selection_path=ROOT/'02_development_validation/validation_selection.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    complete=json.loads((ROOT/'02_development_validation/completion.json').read_text(encoding='utf-8'))
    assert complete['n_streams']==180 and complete['n_modes']==14
    qa=json.loads((ROOT/'qa/design_checks.json').read_text(encoding='utf-8'));assert qa['passed']
    for rel,value in qa['source_sha256'].items():assert sha(ROOT/rel)==value
    from profile_flows import SOURCE
    from calibrator_core import PARAM_GRID
    assert selection['selected_params'] in PARAM_GRID
    code=['scripts/calibrator_core.py','scripts/profile_flows.py','scripts/run_fixed_feedback.py',
          'scripts/fixed_statistics.py','scripts/run_closed_loop.py','scripts/freeze_protocol.py',
          'qa/verify_design.py','qa/verify_completed_data.py',
          '00_protocol_sources/frozen_sources/hetero_qos_sim_n2.py','00_protocol_sources/development_protocol.json']
    payload=dict(status='FROZEN',qa_approved=True,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                 selected_params=selection['selected_params'],selection_source=str(selection_path),
                 validation_selection_sha256=sha(selection_path),
                 validation_metric=selection['selection_metric'],
                 validation_objective=selection['selected_objective_macro_mape'],
                 validation_fingerprint=complete['run_fingerprint'],
                 code_sha256={p:sha(ROOT/p) for p in code},input_sha256={str(SOURCE):sha(SOURCE)},
                 qa_evidence_sha256=sha(ROOT/'qa/design_checks.json'),
                 interpretation='Selected on static validation only; age decay improvement not established',
                 formal_matrix='Unchanged development_protocol.json: static1000, closed1000 units/6000runs, drift360',
                 prohibition='No test-driven change to candidate, seeds, rho levels, endpoints, baselines or selection rule')
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(freeze_path=str(path),selected_params=payload['selected_params'],freeze_sha256=sha(path)),ensure_ascii=False))
if __name__=='__main__':main()
