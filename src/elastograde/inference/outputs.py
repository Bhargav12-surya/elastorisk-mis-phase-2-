from pathlib import Path
import numpy as np
from elastograde.config import save_json

def export(out,result):
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(p/'maps.npz',**arrays)
    summary={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    save_json(p/'summary.json',summary);return summary
