from pathlib import Path
import json,sys
import typer
from elastograde.config import load,save_json
app=typer.Typer(help='ElastoGrade Research: mechanics first, clinical grading gated.',no_args_is_help=True)
data=typer.Typer();app.add_typer(data,name='data')
pgm=typer.Typer();app.add_typer(pgm,name='pgm')
report=typer.Typer();app.add_typer(report,name='report')
risk=typer.Typer();app.add_typer(risk,name='risk')
clinical=typer.Typer();app.add_typer(clinical,name='clinical')
phenotype=typer.Typer();app.add_typer(phenotype,name='phenotype')
notebooks=typer.Typer();app.add_typer(notebooks,name='notebooks')

def emit(value):typer.echo(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
@app.command()
def doctor():
    from elastograde.reporting.provenance import environment
    import torch
    emit(dict(environment(),cuda_available=torch.cuda.is_available(),mode='mechanics',clinical_grading='unvalidated'))
@data.command()
def instructions(dataset:str='impact-acquired'):
    from elastograde.data.acquisition import instructions
    emit(instructions(dataset))
@data.command()
def inspect(root:Path=typer.Option(...),out:Path=typer.Option(Path('reports/inventory'))):
    from elastograde.data.inventory import inventory
    v=inventory(root);save_json(out/'data_inventory.json',v);save_json(out/'dataset_capabilities.json',{'rf_axes':'not_verified','pathology_grades':'not_verified','temporal_sequences':'not_verified'});emit(v)
@data.command()
def prepare(config:Path=typer.Option(...)):
    c=load(config)
    from elastograde.data.impact_acquired import prepare as acquired
    from elastograde.data.impact_simulation import prepare as simulation
    r=(simulation if c.get('dataset_id')=='impact-simulation' else acquired)(c);save_json(Path(c['output'])/'manifest.json',r);emit({'prepared':len(r)})
@data.command(name='split')
def split_data(config:Path=typer.Option(...)):
    from elastograde.data.splits import grouped_split
    c=load(config);r=grouped_split(json.loads(Path(c['manifest']).read_text()),c.get('group_key','geometry_id'),c.get('seed',42));save_json(c['output'],r);emit({'samples':len(r),'output':c['output']})
@data.command()
def audit(manifest:Path=typer.Option(...),group_key:str='geometry_id'):
    from elastograde.data.audit import audit
    emit(audit(json.loads(manifest.read_text()),group_key))
@data.command()
def download(url:str=typer.Option(...),destination:Path=typer.Option(...),checksum:str|None=None):
    from elastograde.data.acquisition import download
    emit(download(url,destination,checksum))
@data.command()
def extract(archive:Path=typer.Option(...),destination:Path=typer.Option(...),expected_volumes:int|None=None):
    from elastograde.data.acquisition import extract_archive
    emit({'destination':extract_archive(archive,destination,expected_volumes)})
@app.command()
def demo(out:Path=Path('reports/demo'),seed:int=42):
    from elastograde.reporting.reports import demo
    emit(demo(out,seed))
@app.command()
def train(config:Path=typer.Option(...),resume:Path|None=None,dry_run:bool=False):
    c=load(config)
    if dry_run:emit({'config':c,'status':'not_run'});return
    from elastograde.training.engine import train
    emit(train(c,resume,callback=lambda row:typer.echo(json.dumps(row))))
@app.command()
def evaluate(run:Path=typer.Option(...),split:str='test',method:str='checkpoint'):
    from elastograde.reporting.reports import evaluate_run
    emit(evaluate_run(run,split,method))
@app.command()
def infer(input:Path=typer.Option(...),checkpoint:Path|None=None,method:str='ncc',mode:str='mechanics',out:Path=Path('reports/inference')):
    if mode!='mechanics':raise ValueError('Use clinical readiness first; no clinical model is included')
    from elastograde.data.schemas import load_sample
    from elastograde.inference.pipeline import infer
    from elastograde.inference.outputs import export
    from elastograde.reporting.figures import panel
    s=load_sample(input);r=infer(s,method,checkpoint);emit(export(out,r));panel(s,r,out/'panel.png')
@pgm.command(name='demo')
def pgm_demo(topic:str='bayesian-quality'):
    from elastograde.pgm.bayesian_quality import illustrative_network
    from elastograde.pgm.moralization import verify
    from elastograde.pgm.hmm import illustrative_hmm
    from elastograde.pgm.decisions import choose
    if topic=='bayesian-quality':emit({'p_reliable':illustrative_network().query('reliable',{'residual':0,'cycle':0}),'provenance':'assumed educational CPDs'})
    elif topic=='moralization':emit(verify(illustrative_network()))
    elif topic=='hmm':emit(illustrative_hmm().infer([0,0,1,1,2,0],ordered=True))
    elif topic=='decision':emit(choose(.8))
    else:raise ValueError('Topics: bayesian-quality, moralization, hmm, decision')
@report.command(name='build')
def report_build(run:Path=typer.Option(...),format:str='html'):
    if format not in ('html','markdown'):raise ValueError('format html or markdown')
    from elastograde.reporting.reports import build_report
    emit({'report':build_report(run)})
@risk.command(name='analyze')
def risk_analyze(run:Path=typer.Option(...)):
    s=json.loads((run/'summary.json').read_text());emit({'action':s['decision'],'quality':s['quality'],'educational_bn':s['educational_bn'],'clinical_risk':'not_estimable_without_clinical_validation'})
@clinical.command(name='readiness')
def readiness(manifest:Path=typer.Option(...),checkpoint_metadata:Path|None=None):
    from elastograde.capabilities import clinical_readiness
    r=clinical_readiness(json.loads(manifest.read_text()),json.loads(checkpoint_metadata.read_text()) if checkpoint_metadata else None);emit(r)
    if not r['eligible']:raise typer.Exit(2)
@phenotype.command(name='fit')
def phenotype_fit(config:Path=typer.Option(...)):
    import numpy as np,pickle
    from elastograde.models.representations import MechanicalPhenotypes
    c=load(config);d=np.load(c['features'],allow_pickle=False);roles=d['roles'].tolist();m=MechanicalPhenotypes().fit(d['features'],roles,components=c.get('components',[2,3]));p=Path(c['output']);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('wb') as f:pickle.dump(m,f)
    emit({'components':m.model.n_components,'output':str(p),'warning':'Load only your own trusted pickle artifacts'})
@notebooks.command(name='execute')
def notebooks_execute(mode:str='demo',root:Path=Path('notebooks'),out:Path=Path('reports/executed_notebooks')):
    if mode!='demo':raise ValueError('CLI notebook runner is bounded demo; use notebook parameters for real data')
    from elastograde.notebook_runner import execute
    emit(execute(root,out))
@app.command()
def tui():
    from elastograde.tui.app import ElastoApp
    ElastoApp().run()

def main():
    try:app()
    except (ValueError,FileNotFoundError,KeyError,RuntimeError) as e:
        typer.echo(json.dumps({'status':'failed','error':str(e)}),err=True);raise SystemExit(1)
if __name__=='__main__':main()
