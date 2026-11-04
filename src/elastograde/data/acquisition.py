from pathlib import Path
import urllib.request, shutil, zipfile, stat, re, os
from elastograde.reporting.provenance import sha256

SOURCES={
 'impact-acquired':'https://users.encs.concordia.ca/~impact/ultrasound-elastography-dataset-for-unsupervised-training/',
 'impact-simulation':'https://users.encs.concordia.ca/~impact/ultrasound-elastography-simulation-database/'}

def instructions(dataset):
    if dataset not in SOURCES:raise ValueError('Unknown dataset')
    return {'dataset':dataset,'landing_page':SOURCES[dataset],'license':'not_verified; review source terms before download or redistribution','instructions':'Follow the official Download link. For acquired data obtain every volume and extract Data_121706.zip.001 once, not each volume. Preserve README and checksums. Inspect keys and verify axes before filling configs/datasets/impact.yaml.','verification_date':'2026-09-22'}

def download(url,destination,expected_sha256=None,max_bytes=20*1024**3):
    if not url.startswith('https://'):raise ValueError('HTTPS URL required')
    p=Path(destination);p.parent.mkdir(parents=True,exist_ok=True);part=p.with_suffix(p.suffix+'.part')
    offset=part.stat().st_size if part.exists() else 0
    req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(req,timeout=30) as r:
        if 'text/html' in r.headers.get('Content-Type','').lower():raise ValueError('Received HTML rather than data')
        resumed=offset>0 and r.status==206
        if resumed and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Invalid resume range')
        total=int(r.headers.get('Content-Length',0))+(offset if resumed else 0)
        if total>max_bytes or shutil.disk_usage(p.parent).free<max(total-offset,0):raise ValueError('Download exceeds size/free-space limit')
        with part.open('ab' if resumed else 'wb') as f:
            first=True
            while chunk:=r.read(1024*1024):
                if first and chunk.lstrip().lower().startswith((b'<!doctype html',b'<html')):raise ValueError('HTML response')
                first=False;f.write(chunk)
                if f.tell()>max_bytes:raise ValueError('Download size limit reached')
    if expected_sha256 and sha256(part)!=expected_sha256:raise ValueError('Checksum mismatch')
    part.replace(p);return {'path':str(p),'sha256':sha256(p)}

def extract_archive(path,destination,expected_volumes=None):
    p=Path(path);dest=Path(destination)
    if dest.exists() and any(dest.iterdir()):raise ValueError('Extraction requires empty destination')
    combined=None
    if re.search(r'\.\d{3}$',p.name):
        if not p.name.endswith('.001') or expected_volumes is None:raise ValueError('Start at .001 and supply verified expected volume count')
        volumes=[p.with_name(p.name[:-3]+f'{i:03d}') for i in range(1,expected_volumes+1)]
        if any(not x.exists() for x in volumes):raise ValueError('Missing multipart volume')
        if shutil.disk_usage(p.parent).free<2*sum(x.stat().st_size for x in volumes):raise ValueError('Insufficient assembly space')
        combined=p.with_name(p.name+'.assembled.zip')
        if combined.exists():raise ValueError('Assembly target already exists')
        with combined.open('xb') as f:
            for v in volumes:
                with v.open('rb') as src:shutil.copyfileobj(src,f)
        p=combined
    try:
        with zipfile.ZipFile(p) as archive:
            root=dest.resolve();infos=archive.infolist()
            if shutil.disk_usage(dest.parent).free<sum(i.file_size for i in infos):raise ValueError('Insufficient extraction space')
            names=set()
            for item in infos:
                target=(root/item.filename).resolve()
                if root not in target.parents or stat.S_ISLNK(item.external_attr>>16):raise ValueError('Unsafe archive entry')
                if target in names:raise ValueError('Duplicate archive path')
                names.add(target)
            dest.mkdir(parents=True,exist_ok=True);archive.extractall(dest)
    finally:
        if combined and combined.exists():combined.unlink()
    return str(dest)
