"""Create and independently read back a versioned local data archive."""
from pathlib import Path
import json,hashlib,datetime,zipfile
import orchestrate_tf32 as o

def main():
    state=json.loads((o.ROOT/'collection_status.json').read_text(encoding='utf-8'))
    assert state['state']=='completed' and state['actual_formal_timing_rows']==2160
    validation=json.loads((o.ROOT/'analysis'/'validation.json').read_text(encoding='utf-8'))
    assert validation['complete_design_passed'] and not validation['errors'],validation.keys()
    assert (o.ROOT/'analysis'/'Independent_Checks_20261018.md').exists()
    entries=[]
    for p in sorted(o.ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or 'mplconfig' in p.parts or p.name=='delivery_manifest.json' or p.suffix in ('.tmp','.transfer','.pyc'):
            continue
        entries.append({'path':p.relative_to(o.ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':o.sha(p)})
    o.save(o.ROOT/'delivery_manifest.json',{'created_utc':o.now(),'formal_timing_rows':2160,'exclusions':['Python bytecode/cache','matplotlib font cache','temporary/transfer fragments'],'files':entries})
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive=o.ROOT.parent/f'TF32机制验证_完整实验数据_{stamp}.zip'
    assert not archive.exists() and archive.resolve().is_relative_to(o.ROOT.parent.resolve())
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for item in entries:
            z.write(o.ROOT/item['path'],o.ROOT.name+'/'+item['path'])
        z.write(o.ROOT/'delivery_manifest.json',o.ROOT.name+'/delivery_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in entries:
            value=z.read(o.ROOT.name+'/'+item['path'])
            assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256'],item['path']
    digest=o.sha(archive)
    archive.with_suffix('.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    o.save(archive.with_suffix('.verified.json'),{'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'files_verified':len(entries)+1,'formal_timing_rows':2160,'zip_crc_passed':True,'all_archived_file_sha256_passed':True,'verified_utc':o.now()})
    o.emit({'archive':str(archive),'sha256':digest,'files':len(entries)+1,'bytes':archive.stat().st_size})

if __name__=='__main__':main()
