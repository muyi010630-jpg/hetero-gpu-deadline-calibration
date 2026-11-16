"""Preserve a zero-timing setup failure, then collect the planned Windows rounds."""
from pathlib import Path
import json,shutil,subprocess,sys,os,datetime
import orchestrate_tf32 as o

def main():
    raw=o.RAW/'RTX5070';archive=o.ROOT/'attempts'/'RTX5070_v3_pre_cuda_gate'
    status=json.loads((raw/'round_01'/'status.json').read_text(encoding='utf-8'))
    assert status['state']=='failed' and status['window_rows']==0 and status['completed_context_phases']==0
    assert not (raw/'round_02').exists() and not (raw/'round_03').exists()
    assert archive.resolve().is_relative_to(o.ROOT.resolve()) and raw.resolve().is_relative_to(o.ROOT.resolve())
    archive.mkdir(parents=True,exist_ok=False)
    for name in ('round_01','batch_status.json','batch.log','round_01.log'):
        src=raw/name
        if src.exists():
            assert src.resolve().is_relative_to(raw.resolve())
            shutil.move(str(src),str(archive/name))
    text=(o.SCRIPTS/'run_all_rounds.py').read_text(encoding='utf-8')
    text=text.replace("result['state']!='completed'","result['state'] not in ('completed','completed_with_background_inventory','completed_with_external_compute_warning')")
    runner=o.SCRIPTS/'windows_v4'/'run_all_rounds.py'
    runner.write_text(text,encoding='utf-8')
    o.save(o.ROOT/'windows_setup_recovery.json',{'utc':o.now(),'preserved_failed_attempt':str(archive),'failed_timing_rows':0,'recovery_reason':'Windows WDDM enumerates desktop applications in compute-process query; use explicit baseline inventory and direct own PID association','no_timing_selected_or_replaced':True,'formal_windows_measurement':str(o.SCRIPTS/'windows_v4'/'tf32_measure.py'),'background_limitation':'Baseline app inventory cannot prove graphics-only or absence of compute interference. Keep all subsequent measured rounds, including any later external-process warning.'})
    with (raw/'batch.log').open('xb') as stream:
        p=subprocess.Popen([o.LOCAL_PYTHON,'-u',str(runner),'--device-label','RTX5070','--output-dir',str(raw)],stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    o.emit({'event':'windows_v4_started','launcher_pid':p.pid,'raw_directory':str(raw)})
    return 0

if __name__=='__main__':raise SystemExit(main())
