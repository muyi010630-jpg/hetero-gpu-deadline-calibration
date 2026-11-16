import ctypes
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (key, ctypes.c_ulonglong) for key in ('total_physical', 'available_physical',
        'total_pagefile', 'available_pagefile', 'total_virtual', 'available_virtual', 'extended')]

root = Path(__file__).resolve().parents[1]
mem = MemoryStatus(); mem.length = ctypes.sizeof(mem)
ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
disk = shutil.disk_usage(root)
payload = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
               os=platform.platform(), python=platform.python_version(),
               logical_cpu_count=os.cpu_count(), processor=os.environ.get('PROCESSOR_IDENTIFIER'),
               memory_total_gb=mem.total_physical/2**30 if ok else None,
               memory_available_gb=mem.available_physical/2**30 if ok else None,
               disk_free_gb=disk.free/2**30, workers_selected=3, cpu_only=True,
               detection_note='Skill detector lacked psutil; native Windows memory API and Python standard library used. CIM unavailable. No GPU computations requested or performed.')
(root/'00_design_sources/本机资源.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(payload, ensure_ascii=False))
