"""Run the frozen three rounds in three fresh child processes, without retries."""
from pathlib import Path
import argparse,sys,subprocess,json,os,datetime,hashlib

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(temp,p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device-label',required=True);ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    state_file=a.output_dir/'batch_status.json'
    if state_file.exists():raise RuntimeError('Existing batch status: preserve rather than overwrite')
    state={'device':a.device_label,'state':'starting','started_utc':now(),'runner_pid':os.getpid(),'completed_rounds':[]}
    save(state_file,state)
    try:
        for r in (1,2,3):
            command=[sys.executable,'-u',str(Path(__file__).with_name('tf32_measure.py')),'--round',str(r),'--device-label',a.device_label,'--output-dir',str(a.output_dir)]
            with open(a.output_dir/f'round_{r:02d}.log','xb') as log:
                flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,**flags)
                state.update(state='running',current_round=r,child_pid=p.pid);save(state_file,state)
                rc=p.wait()
            result=json.loads((a.output_dir/f'round_{r:02d}'/'status.json').read_text(encoding='utf-8'))
            if rc or result['state'] not in ('completed','completed_with_background_inventory','completed_with_external_compute_warning'):
                raise RuntimeError(f'Round {r} requires inspection: exit={rc}, state={result["state"]}')
            state['completed_rounds'].append(r);save(state_file,state)
        state.update(state='completed',completed_utc=now());save(state_file,state)
        files=[]
        for f in sorted(a.output_dir.rglob('*')):
            if f.is_file() and f.name!='remote_manifest.json':
                files.append({'path':f.relative_to(a.output_dir).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
        save(a.output_dir/'remote_manifest.json',{'created_utc':now(),'device':a.device_label,'files':files})
        return 0
    except BaseException as e:
        state.update(state='failed',error=repr(e),completed_utc=now());save(state_file,state);raise

if __name__=='__main__':sys.exit(main())
