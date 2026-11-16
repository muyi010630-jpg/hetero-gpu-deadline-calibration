"""Authorized GPU collection. Passwords enter hidden stdin and are never saved."""
from pathlib import Path
import sys,os,json,getpass,base64,hashlib,time,subprocess,shlex,concurrent.futures,datetime,traceback,argparse
import paramiko

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts';RAW=ROOT/'raw'
LOCAL_PYTHON=os.environ.get('TF32_LOCAL_PYTHON',sys.executable)
REMOTE=os.environ.get('TF32_REMOTE_WORKDIR','tf32_remote')
REMOTE_PYTHON=os.environ.get('TF32_REMOTE_PYTHON','python3')
HOST='region-9.autodl.pro';PORTS={'A100':41673,'T4':50209}
FINGERPRINT='SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0'
KEYS=Path(os.environ.get('TF32_HOST_KEYS',str(ROOT/'protocol'/'observed_host_keys.json')))

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def emit(v):print(json.dumps(v,ensure_ascii=False),flush=True)
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(client,cmd,timeout=60):
    _,out,err=client.exec_command(cmd,timeout=timeout)
    text=out.read().decode('utf-8','replace');error=err.read().decode('utf-8','replace');rc=out.channel.recv_exit_status()
    if rc:raise RuntimeError(f'Remote exit {rc}: {error[-2000:]} {text[-1000:]}')
    return text
def pycmd(code):return REMOTE_PYTHON+' -c '+shlex.quote(code)
def connect(label,password):
    record=next(x for x in json.loads(KEYS.read_text(encoding='utf-8')) if x['hostname']==HOST and x['port']==PORTS[label])
    assert record['fingerprint']==FINGERPRINT
    key=paramiko.Ed25519Key(data=base64.b64decode(record['key_base64']))
    c=paramiko.SSHClient();c.get_host_keys().add(f'[{HOST}]:{PORTS[label]}',key.get_name(),key)
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect(HOST,port=PORTS[label],username='root',password=password,timeout=20,auth_timeout=20,banner_timeout=20,look_for_keys=False,allow_agent=False)
    return c
def fetch_json(client,path):return json.loads(command(client,pycmd('from pathlib import Path; print(Path('+repr(path)+').read_text())')))
def upload(client,label):
    command(client,pycmd('from pathlib import Path; p=Path('+repr(REMOTE)+'); p.mkdir(exist_ok=False); (p/"scripts").mkdir(); (p/"raw").mkdir()'))
    s=client.open_sftp()
    try:
        for name in ['tf32_measure.py','protocol.json','run_all_rounds.py']:
            s.put(str(SCRIPTS/name),REMOTE+'/scripts/'+name)
    finally:s.close()
    remote_hashes=json.loads(command(client,pycmd('from pathlib import Path; import hashlib,json; p=Path('+repr(REMOTE+'/scripts')+'); print(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}))')))
    assert all(sha(SCRIPTS/n)==v for n,v in remote_hashes.items())
    save(ROOT/'transfer'/f'{label}_uploaded_script_hashes.json',remote_hashes)
def update_preformal(client,label):
    code='from pathlib import Path\nimport shutil\np=Path('+repr(REMOTE)+')\nassert not (p/"raw"/"batch_status.json").exists() and not list((p/"raw").glob("round_*"))\narchive=p/"scripts"/"preformal_v1"\narchive.mkdir(exist_ok=True)\nfor name in ("tf32_measure.py","protocol.json"):\n src=p/"scripts"/name; dst=archive/name\n if not dst.exists():shutil.copy2(src,dst)'
    command(client,pycmd(code))
    s=client.open_sftp()
    try:
        for name in ['tf32_measure.py','protocol.json','run_all_rounds.py']:
            s.put(str(SCRIPTS/name),REMOTE+'/scripts/'+name)
    finally:s.close()
    remote_hashes=json.loads(command(client,pycmd('from pathlib import Path; import hashlib,json; p=Path('+repr(REMOTE+'/scripts')+'); print(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}))')))
    assert all(sha(SCRIPTS/n)==v for n,v in remote_hashes.items())
    save(ROOT/'transfer'/f'{label}_formal_uploaded_script_hashes.json',remote_hashes)
def input_check(client,label):
    if client:
        cmd=REMOTE_PYTHON+' -u '+shlex.quote(REMOTE+'/scripts/tf32_measure.py')+' --round 1 --device-label '+label+' --output-dir '+shlex.quote(REMOTE+'/raw')+' --input-check'
        command(client,cmd,timeout=180)
        download_all(client,label)
    else:
        result=subprocess.run([LOCAL_PYTHON,'-u',str(SCRIPTS/'tf32_measure.py'),'--round','1','--device-label',label,'--output-dir',str(RAW/label),'--input-check'],capture_output=True,timeout=180)
        assert result.returncode==0,result.stderr.decode('utf-8','replace')[-2000:]
    result=json.loads((RAW/label/'input_check_round_01'/'status.json').read_text(encoding='utf-8'))
    assert result['state']=='completed' and result['gemm_calls']==0 and result['torch_imported']==False,result
    emit({'event':'cpu_input_check_completed','device':label,'contexts':result['input_contexts']})
    return json.loads((RAW/label/'input_check_round_01'/'input_manifest.json').read_text(encoding='utf-8'))
def download_all(client,label):
    code='from pathlib import Path\nimport json,hashlib\np=Path('+repr(REMOTE+'/raw')+')\nb=json.loads((p/"batch_status.json").read_text()) if (p/"batch_status.json").exists() else {}\nterminal=b.get("state") in ("completed","failed")\nready={"round_%02d"%r for r in b.get("completed_rounds",[])}\nfor d in list(p.glob("smoke_round_*"))+list(p.glob("input_check_round_*")):\n if (d/"status.json").exists() and json.loads((d/"status.json").read_text()).get("state")=="completed":ready.add(d.name)\nrows=[]\nfor f in sorted(p.rglob("*")):\n if f.is_file() and (terminal or f.relative_to(p).parts[0] in ready):rows.append({"path":f.relative_to(p).as_posix(),"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()})\nprint(json.dumps(rows))'
    listing=json.loads(command(client,pycmd(code)))
    dest=RAW/label;dest.mkdir(parents=True,exist_ok=True);s=client.open_sftp()
    try:
        for row in listing:
            p=dest/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and sha(p)==row['sha256']:continue
            tmp=p.with_name(p.name+'.transfer');s.get(REMOTE+'/raw/'+row['path'],str(tmp))
            assert sha(tmp)==row['sha256'],f'Transfer mismatch: {label}/{row["path"]}'
            os.replace(tmp,p)
    finally:s.close()
    save(ROOT/'transfer'/f'{label}_download_manifest.json',{'verified_utc':now(),'files':listing})
    return len(listing)
def local_run(smoke):
    d=RAW/'RTX5070';d.mkdir(parents=True,exist_ok=True)
    log=d/('smoke.log' if smoke else 'batch.log')
    args=[LOCAL_PYTHON,'-u',str(SCRIPTS/('tf32_measure.py' if smoke else 'run_all_rounds.py')),'--device-label','RTX5070','--output-dir',str(d)]
    if smoke:args+=['--round','1','--smoke']
    with log.open('xb') as stream:
        p=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    return p
def remote_smoke(client,label):
    cmd=REMOTE_PYTHON+' -u '+shlex.quote(REMOTE+'/scripts/tf32_measure.py')+' --round 1 --device-label '+label+' --output-dir '+shlex.quote(REMOTE+'/raw')+' --smoke > '+shlex.quote(REMOTE+'/raw/smoke.log')+' 2>&1'
    command(client,cmd,timeout=180)
    result=fetch_json(client,REMOTE+'/raw/smoke_round_01/status.json')
    download_all(client,label)
    return result
def start_remote(client,label):
    code='import subprocess,json; from pathlib import Path; p=Path('+repr(REMOTE)+'); log=open(p/"raw"/"batch.log","xb"); child=subprocess.Popen(['+repr(REMOTE_PYTHON)+',"-u",str(p/"scripts"/"run_all_rounds.py"),"--device-label",'+repr(label)+',"--output-dir",str(p/"raw")],stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print(json.dumps({"pid":child.pid}))'
    return json.loads(command(client,pycmd(code)))
def poll_remote(client):
    code='from pathlib import Path; import json,os; p=Path('+repr(REMOTE+'/raw')+'); b=json.loads((p/"batch_status.json").read_text()) if (p/"batch_status.json").exists() else {"state":"starting"}; r=b.get("current_round"); f=p/("round_%02d"%r)/"status.json" if r else None; s=json.loads(f.read_text()) if f and f.exists() else {}; print(json.dumps({"batch":b,"round":s}))'
    value=json.loads(command(client,pycmd(code)))
    if value['batch'].get('state')=='completed':
        ready=json.loads(command(client,pycmd('import json; from pathlib import Path; print(json.dumps(Path('+repr(REMOTE+'/raw/remote_manifest.json')+').exists()))')))
        if not ready:value['batch']['state']='finalizing'
    return value
def poll_local():
    p=RAW/'RTX5070';f=p/'batch_status.json';b=json.loads(f.read_text(encoding='utf-8')) if f.exists() else {'state':'starting'};r=b.get('current_round');s=p/f'round_{r:02d}'/'status.json' if r else None
    if b.get('state')=='completed' and not (p/'remote_manifest.json').exists():b['state']='finalizing'
    return {'batch':b,'round':json.loads(s.read_text(encoding='utf-8')) if s and s.exists() else {}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume-after-smoke',action='store_true');args=ap.parse_args()
    previous=None
    if (ROOT/'collection_status.json').exists():
        previous=json.loads((ROOT/'collection_status.json').read_text(encoding='utf-8'))
        if not args.resume_after_smoke:raise RuntimeError('Existing collection status: inspect rather than restart')
        assert previous['state']=='needs_inspection' and previous['devices']=={},'Only resume verified pre-formal smoke failure'
        assert not (RAW/'RTX5070'/'batch_status.json').exists(),'Existing formal local batch must be inspected, never restarted'
        archive=ROOT/'collection_attempts'/('attempt_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
        save(archive,previous)
    passwords={label:getpass.getpass(label+' SSH password: ') for label in PORTS}
    clients={};status={'started_utc':now(),'state':'connecting','remote_root':REMOTE,'devices':{}}
    save(ROOT/'collection_status.json',status)
    try:
        for label in PORTS:clients[label]=connect(label,passwords.pop(label))
        emit({'event':'connected','host_identity':'pinned user-approved key'})
        if not args.resume_after_smoke:
            for label,c in clients.items():upload(c,label)
        else:
            for label,c in clients.items():
                absent=json.loads(command(c,pycmd('from pathlib import Path; import json; p=Path('+repr(REMOTE+'/raw')+'); print(json.dumps(not (p/"batch_status.json").exists() and not list(p.glob("round_*"))))')))
                assert absent,f'{label} already has a formal batch; inspect rather than restart'
                update_preformal(c,label)
        status['state']='smoke_checks';save(ROOT/'collection_status.json',status)
        if not args.resume_after_smoke:
            lp=local_run(True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                future={label:pool.submit(remote_smoke,c,label) for label,c in clients.items()}
                for label,f in future.items():
                    result=f.result();assert result['state']=='completed',(label,result)
                    assert not result['telemetry_query_failure_seen'],(label,'telemetry failed')
                    emit({'event':'smoke_completed','device':label,'windows':result['window_rows']})
            assert lp.wait()==0,'Local smoke failed; inspect preserved log'
        else:
            for label in clients:
                result=fetch_json(clients[label],REMOTE+'/raw/smoke_round_01/status.json')
                assert result['state']=='completed' and not result['telemetry_query_failure_seen'],result
                emit({'event':'existing_smoke_verified','device':label,'windows':result['window_rows']})
        result=json.loads((RAW/'RTX5070'/'smoke_round_01'/'status.json').read_text(encoding='utf-8'));assert result['state']=='completed',result
        assert not result['telemetry_query_failure_seen'],'Local telemetry failed'
        emit({'event':'smoke_completed','device':'RTX5070','windows':result['window_rows']})
        status['state']='cpu_input_identity_check';save(ROOT/'collection_status.json',status)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(input_check,clients.get(d),d) for d in ['A100','T4','RTX5070']]
            manifests=[f.result() for f in futures]
        norm=lambda rows:sorted((x['context'],x['a_sha256'],x['b_sha256']) for x in rows)
        assert norm(manifests[0])==norm(manifests[1])==norm(manifests[2]),'Cross-device input mismatch; stop before formal collection'
        save(ROOT/'smoke_validation.json',{'passed':True,'utc':now(),'cross_device_inputs_equal':True,'identity_source':'CPU-only frozen formal-generator input checks; v1 smoke inputs differed by NumPy sampler version','devices':['A100','T4','RTX5070'],'smoke_excluded_from_formal':True})
        status['state']='formal_measurement';save(ROOT/'collection_status.json',status)
        for label,c in clients.items():status['devices'][label]={'launch':start_remote(c,label)}
        lp=local_run(False);status['devices']['RTX5070']={'launch':{'pid':lp.pid}};save(ROOT/'collection_status.json',status)
        emitted={};downloaded={d:set() for d in clients};terminal_downloaded=set()
        while True:
            results={}
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs={d:pool.submit(poll_remote,c) for d,c in clients.items()}
                for d,f in fs.items():results[d]=f.result()
            results['RTX5070']=poll_local()
            for d,v in results.items():
                state=v['batch'].get('state');detail=v['round'];progress=(state,v['batch'].get('current_round'),detail.get('completed_context_phases'),detail.get('window_rows'))
                status['devices'][d]['progress']=v
                if emitted.get(d)!=progress:
                    emit({'event':'progress','device':d,'batch_state':state,'round':progress[1],'context_phases':progress[2],'window_rows':progress[3]});emitted[d]=progress
                if d in clients:
                    done=set(v['batch'].get('completed_rounds',[]))
                    if done-downloaded[d] or (state in ('completed','failed') and d not in terminal_downloaded):
                        n=download_all(clients[d],d);downloaded[d]=done
                        emit({'event':'downloaded_verified','device':d,'files':n,'completed_rounds':sorted(done)})
                        if state in ('completed','failed'):terminal_downloaded.add(d)
                if state=='failed':raise RuntimeError(f'{d} batch failed; preserved all files')
            status['updated_utc']=now();save(ROOT/'collection_status.json',status)
            if all(v['batch'].get('state')=='completed' for v in results.values()):break
            if lp.poll() is not None and results['RTX5070']['batch'].get('state')!='completed':raise RuntimeError('Local batch exited without completion')
            time.sleep(12)
        assert lp.wait()==0
        status.update(state='completed',completed_utc=now());save(ROOT/'collection_status.json',status)
        emit({'event':'all_collected','local_root':str(ROOT),'state':'completed'})
        return 0
    except BaseException as e:
        status.update(state='needs_inspection',error=repr(e),traceback=traceback.format_exc(),updated_utc=now());save(ROOT/'collection_status.json',status)
        for label,c in clients.items():
            try:download_all(c,label)
            except Exception:pass
        emit({'event':'needs_inspection','error':repr(e),'note':'Do not restart; inspect live remote/local handles and preserved files'})
        raise
    finally:
        passwords.clear()
        for c in clients.values():c.close()

if __name__=='__main__':sys.exit(main())
