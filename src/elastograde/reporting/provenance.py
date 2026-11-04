import hashlib, json, platform, importlib.metadata
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

def environment():
    names=['numpy','scipy','torch','scikit-learn','matplotlib','typer','textual','nbformat','nbclient','h5py']
    out={'python':platform.python_version(),'platform':platform.platform()}
    for n in names:
        try:out[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:out[n]='not_installed'
    return out

def code_hash():
    root=Path(__file__).parents[1]
    return digest({str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py'))})
