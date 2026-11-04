from pathlib import Path
import time,os,uuid,subprocess,sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from elastograde.config import save_json

def execute(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);results=[]
    for p in sorted(root.glob('*.ipynb')):
        start=time.perf_counter();nb=nbformat.read(p,as_version=4)
        try:
            nbformat.validate(nb)
            if os.environ.get('ELASTOGRADE_NOTEBOOK_BACKEND')=='process':
                proc=subprocess.run([sys.executable,'-m','elastograde.notebook_process',str(p),str(out/p.name),str(root.parent)],capture_output=True,text=True,timeout=180)
                nb=nbformat.read(out/p.name,4) if (out/p.name).exists() else nb
                if proc.returncode:raise RuntimeError(proc.stderr or 'Cell execution failed; inspect executed notebook')
            else:
                NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root.parent)}}).execute()
            status='passed';error=None
        except Exception as e:status='failed';error=str(e)
        nbformat.write(nb,out/p.name);results.append({'notebook':p.name,'status':status,'seconds':time.perf_counter()-start,'error':error})
    result={'backend':os.environ.get('ELASTOGRADE_NOTEBOOK_BACKEND','jupyter_kernel'),'passed':sum(r['status']=='passed' for r in results),'failed':sum(r['status']=='failed' for r in results),'results':results};save_json(out/'execution_summary.json',result)
    if result['failed']:raise RuntimeError(f"{result['failed']} notebook(s) failed; see {out}/execution_summary.json")
    return result
