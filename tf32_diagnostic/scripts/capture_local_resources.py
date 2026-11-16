from pathlib import Path
import ctypes,os,platform,shutil,subprocess,json,datetime
class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total_phys',ctypes.c_ulonglong),('available_phys',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),('available_page',ctypes.c_ulonglong),('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),('available_extended',ctypes.c_ulonglong)]
m=MemoryStatus();m.length=ctypes.sizeof(m)
ok=bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)))
root=Path(__file__).resolve().parents[1]
d=shutil.disk_usage(root)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
r={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Native Windows GlobalMemoryStatusEx + Python stdlib fallback because optional psutil is absent','logical_cpu_count':os.cpu_count(),'processor':platform.processor(),'platform':platform.platform(),'memory':{'total_bytes':m.total_phys,'available_bytes':m.available_phys,'percent_used':m.load} if ok else None,'disk':{'total_bytes':d.total,'free_bytes':d.free},'gpu_query':{'returncode':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr}}
(root/'local_resources.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r))
