from pathlib import Path
import json,html,csv
from elastograde.config import save_json
from elastograde.reporting.provenance import sha256

def build_report(run):
    p=Path(run);summary=json.loads((p/'summary.json').read_text()) if (p/'summary.json').exists() else json.loads((p/'status.json').read_text())
    text=json.dumps(summary,indent=2);image='<img src="panel.png" alt="Measured displacement, strain and quality maps">' if (p/'panel.png').exists() else ''
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>ElastoGrade Research report</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#f5f7fa;color:#152333}pre{white-space:pre-wrap;background:white;padding:22px;border:1px solid #ccd}img{max-width:100%}h1{color:#12566b}</style><h1>ElastoGrade Research</h1><p>Research software output. Data origin: '+html.escape(str(summary.get('data_origin','unknown')))+'. Histopathological grading is not validated by this report.</p>'+image+'<h2>Recorded output and provenance</h2><pre>'+html.escape(text)+'</pre></html>'
    (p/'report.html').write_text(page);(p/'report.md').write_text('# ElastoGrade Research report\n\n```json\n'+text+'\n```\n')
    save_json(p/'report_manifest.json',{'files':{f.name:sha256(f) for f in p.iterdir() if f.is_file() and f.name!='report_manifest.json'},'data_origin':summary.get('data_origin')})
    return str(p/'report.html')

def demo(out='reports/demo',seed=42):
    import numpy as np
    from elastograde.data.synthetic import make_pair
    from elastograde.data.schemas import save_sample
    from elastograde.inference.pipeline import infer
    from elastograde.inference.outputs import export
    from elastograde.reporting.figures import panel
    from elastograde.evaluation.metrics import displacement_metrics,strain_metrics
    from elastograde.mechanics.strain import strain
    p=Path(out);p.mkdir(parents=True,exist_ok=True);s=make_pair(seed,kind='compression');save_sample(p/'sample.npz',s)
    result=infer(s);result['metrics']=displacement_metrics(result['flow'],s.displacement,result['valid']);result['strain_metrics']=strain_metrics(result['axial_strain'],strain(s.displacement,s.metadata['spacing_mm'],'least_squares')['axial'],result['valid'])
    export(p,result);panel(s,result,p/'panel.png');build_report(p)
    return {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}

def evaluate_run(run,split='test',method='checkpoint'):
    import numpy as np
    from elastograde.data.schemas import load_sample
    from elastograde.inference.pipeline import infer
    from elastograde.evaluation.metrics import displacement_metrics
    from elastograde.evaluation.grouped_statistics import grouped_bootstrap
    p=Path(run);records=json.loads((p/'manifest.json').read_text());rows=[]
    for r in records:
        if r['split']!=split:continue
        s=load_sample(r['path']);result=infer(s,method='ncc' if method=='checkpoint' else method,checkpoint=p/'best.pt' if method=='checkpoint' else None)
        row={'sample_id':r['sample_id'],'group':r.get('geometry_id') or r.get('patient_id') or r.get('acquisition_id'),'data_origin':s.metadata.get('data_origin'),'residual':result['quality']['residual_mean']}
        if s.displacement is not None:row.update(displacement_metrics(result['flow'],s.displacement,result['valid']))
        rows.append(row)
    if not rows:raise ValueError('No eligible samples in requested split')
    summary={'split':split,'method':method,'rows':rows,'reference_metrics':'available' if 'endpoint_error' in rows[0] else 'not_available'}
    if all('endpoint_error' in r for r in rows) and len(set(r['group'] for r in rows))>1:summary['endpoint_error_ci']=grouped_bootstrap([r['endpoint_error'] for r in rows],[r['group'] for r in rows])
    save_json(p/f'evaluation_{split}_{method}.json',summary)
    with (p/f'evaluation_{split}_{method}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return summary
