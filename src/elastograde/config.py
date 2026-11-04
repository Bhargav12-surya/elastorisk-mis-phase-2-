from pathlib import Path
import json, yaml

def load(path):
    p=Path(path)
    data=yaml.safe_load(p.read_text())
    if not isinstance(data,dict): raise ValueError('Configuration must be a mapping')
    return data

def save_json(path, data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    return p
