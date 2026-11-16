"""Reconcile retained setup attempts with completed data; verify local copies."""
from pathlib import Path
import json,csv,hashlib,datetime,shutil
import orchestrate_tf32 as o

def main():
    previous=json.loads((o.ROOT/'collection_status.json').read_text(encoding='utf-8'))
    archive=o.ROOT/'collection_attempts'/'formal_coordinator_local_gate_failure.json'
    assert not archive.exists()
    o.save(archive,previous)
    summary={'state':'completed','completed_utc':o.now(),'expected_formal_timing_rows':2160,'devices':{},'retained_attempts':'Pre-formal interface/input adjustments and the Windows v3 empty-baseline failure are retained. The Windows failed attempt collected zero timing rows; cloud v3 jobs were never restarted. Windows v4 contains the same measurement design with a desktop-baseline-specific ownership gate.','windows_environment_limitation':'Actual desktop environment; earlier elevated WDDM process inventory included desktop applications. Empty or unchanged later query results cannot prove absence of desktop load or interference.'}
    total=0
    for label in ('A100','T4','RTX5070'):
        folder=o.RAW/label
        batch=json.loads((folder/'batch_status.json').read_text(encoding='utf-8'))
        assert batch['state']=='completed' and batch['completed_rounds']==[1,2,3],batch
        rows=0;rounds=[]
        for round_id in (1,2,3):
            d=folder/f'round_{round_id:02d}'
            status=json.loads((d/'status.json').read_text(encoding='utf-8'))
            assert status['state'] in ('completed','completed_with_background_inventory','completed_with_external_compute_warning'),status
            with (d/'windows.csv').open(encoding='utf-8-sig',newline='') as f:n=sum(1 for _ in csv.DictReader(f))
            assert n==240 and status['window_rows']==240
            rows+=n;rounds.append(status)
        assert rows==720
        if label in o.PORTS:
            manifest=json.loads((o.ROOT/'transfer'/f'{label}_download_manifest.json').read_text(encoding='utf-8'))
            for entry in manifest['files']:
                p=folder/entry['path']
                assert p.stat().st_size==entry['bytes'] and o.sha(p)==entry['sha256'],str(p)
        summary['devices'][label]={'formal_timing_rows':rows,'round_statuses':rounds,'all_three_rounds_collected':True}
        total+=rows
    assert total==2160
    summary['actual_formal_timing_rows']=total
    o.save(o.ROOT/'collection_status.json',summary)
    protocol_dir=o.ROOT/'protocol';protocol_dir.mkdir(exist_ok=True)
    source=Path(r'<LOCAL_WORKDIR>/审稿意见/软件计算路径与GPU画像失配_小型机制验证方案_20261018.md')
    target=protocol_dir/'experiment_plan_original.md'
    if not target.exists():shutil.copy2(source,target)
    o.emit({'state':'completed','formal_timing_rows':total,'cloud_download_checksums_verified':True})

if __name__=='__main__':main()
