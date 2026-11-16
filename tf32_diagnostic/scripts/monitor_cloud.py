"""Reconnect to existing authorized runs, collect data, never launch a job."""
import getpass,datetime,time,json,concurrent.futures
import orchestrate_tf32 as o

def main():
    clients={}
    try:
        for label in o.PORTS:
            clients[label]=o.connect(label,getpass.getpass(label+' SSH password: '))
        last={};done=set()
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                future={d:pool.submit(o.poll_remote,c) for d,c in clients.items()}
                results={d:f.result() for d,f in future.items()}
            for d,v in results.items():
                progress=(v['batch'].get('state'),v['batch'].get('current_round'),v['round'].get('completed_context_phases'),v['round'].get('window_rows'))
                if last.get(d)!=progress:
                    o.emit({'device':d,'progress':progress});last[d]=progress
                complete=tuple(v['batch'].get('completed_rounds',[]))
                key=(d,complete,progress[0] in ('completed','failed'))
                if key not in done:
                    count=o.download_all(clients[d],d);done.add(key)
                    o.emit({'device':d,'verified_files':count,'completed_rounds':complete})
            o.save(o.ROOT/'cloud_monitor_status.json',{'updated_utc':o.now(),'devices':results})
            if all(v['batch'].get('state') in ('completed','failed') for v in results.values()):
                for d,c in clients.items():o.download_all(c,d)
                o.emit({'event':'cloud_terminal','states':{d:v['batch']['state'] for d,v in results.items()}})
                return 0 if all(v['batch']['state']=='completed' for v in results.values()) else 1
            time.sleep(12)
    finally:
        for c in clients.values():c.close()

if __name__=='__main__':raise SystemExit(main())
