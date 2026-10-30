"""Build a compact, byte-equivalent NIfTI resource bank without editing sources."""
import gzip,hashlib,json,shutil
from pathlib import Path


def digest(path, decompressed=False):
    h=hashlib.sha256()
    opener=gzip.open if decompressed and str(path).endswith('.nii.gz') else open
    with opener(path,'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_templates(source,destination,progress=lambda _:None):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or destination.is_relative_to(source):
        raise ValueError('Packed resources must be separate from original resources')
    destination.mkdir(parents=True,exist_ok=True)
    originals=sorted(p for p in source.rglob('*') if p.is_file() and p.name!='.resource_index.json')
    mapping={};canonical={};stream_cache={};records=[]
    before=sum(p.stat().st_size for p in originals)
    for i,path in enumerate(originals):
        relative=path.relative_to(source).as_posix()
        file_hash=digest(path)
        if file_hash not in stream_cache:stream_cache[file_hash]=digest(path,True) if path.name.endswith('.nii.gz') else file_hash
        content_hash=stream_cache[file_hash]
        # Keep the historical atlas at its exact public path.
        key=(content_hash,'atlas' if relative=='aal.nii' else 'nifti' if path.name.endswith(('.nii','.nii.gz')) else 'other')
        parent=destination/Path(relative).parent;parent.mkdir(parents=True,exist_ok=True)
        (parent/'.resource_directory').touch()
        if key not in canonical:
            stored=relative+'.gz' if path.suffix=='.nii' and relative!='aal.nii' else relative
            target=destination/stored
            if stored!=relative:
                with path.open('rb') as src,target.open('wb') as dst:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0,compresslevel=6) as zipped:
                        shutil.copyfileobj(src,zipped,1024*1024)
            else:shutil.copyfile(path,target)
            assert digest(target,True)==content_hash,(relative,stored)
            canonical[key]=stored
        mapping[relative]=canonical[key]
        records.append({'original':relative,'stored':canonical[key],'file_sha256':file_hash,'content_sha256':content_hash})
        if i==0 or (i+1)%25==0 or i+1==len(originals):progress(f'Packed templates {i+1}/{len(originals)}; unique payloads {len(canonical)}')
    (destination/'.resource_index.json').write_text(json.dumps({'version':1,'files':mapping},ensure_ascii=False,indent=2),encoding='utf-8')
    after=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    return {'source':str(source),'destination':str(destination),'original_bytes':before,'packed_bytes':after,
            'logical_files':len(mapping),'unique_payloads':len(canonical),'records':records,'all_unique_payloads_verified':True}


if __name__=='__main__':
    import os
    stage=Path(os.environ['TEMP'])/'neuroguide_release_20261013'
    report=prepare_templates('F:/NeuroGuide/sample_data',stage/'compact_sample_data',lambda message:print(message,flush=True))
    (stage/'resource_compression.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Template resources MiB',report['original_bytes']/2**20,'->',report['packed_bytes']/2**20,flush=True)
