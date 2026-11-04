from pathlib import Path
R=Path(__file__).resolve().parents[1]
import nbformat as nbf
# A source-controlled generator makes each notebook reproducible, editable, and executable.
chapters=[]
def add(name,question,theory,code,interpret,exercise):chapters.append((name,question,theory,code,interpret,exercise))
add('00_Project_Map_and_Claims','What can this project scientifically claim?',r'''The computation has three distinct layers: motion reconstruction, mechanical characterization, and an optional pathology-linked grading model. The first two can be investigated with phantom data. Neither an inclusion label nor a stiffness ordering defines a histological grade. Label-free motion learning uses the RF pair itself as a supervision signal. Simulation ground-truth supervision is a separate experiment. A pathology-linked grading head is supervised even when its encoder was learned without pathology labels.

The software gate checks data and model provenance; it does not establish clinical validity. A successful smoke test means the implementation executed on fixtures. It does not show superiority to published systems.''',"""from elastograde.capabilities import clinical_readiness
from elastograde.reporting.provenance import environment
readiness=clinical_readiness({'records':[]})
print(json.dumps(readiness,indent=2))
print(environment())""",'The blocked result is expected. No pathology-linked patient data or clinically calibrated checkpoint is bundled. The following notebooks demonstrate mechanics and mathematics using synthetic fixtures.','Explain three meanings of label-free. Answer: absent displacement labels during motion training; absent pathology labels during representation learning; absent pathology measurements at inference. These are different experimental claims.')
add('01_Environment_and_Reproducibility','Can another machine reproduce the fixture?',r'''A seed determines pseudo-random numbers, but reproducibility also depends on numerical libraries, hardware and deterministic kernels. A run records its dependency versions, source hash, configuration and source checksums. GPU interpolation gradients can be nondeterministic. Reproducibility should therefore specify a numerical tolerance rather than promise identical bytes across all platforms.''',"""import torch
from elastograde.reporting.provenance import environment,code_hash
from elastograde.data.synthetic import make_pair
s1=make_pair(seed);s2=make_pair(seed)
assert np.array_equal(s1.pre,s2.pre)
print(environment());print('Code hash:',code_hash());print('CUDA:',torch.cuda.is_available())
plt.imshow(s1.pre,cmap='gray',aspect='auto');plt.title('synthetic_fixture: seeded RF-like texture');plt.show()""",'The repeated fixture matches exactly in this environment. This is a data-generator property, not evidence of scanner realism.','Why retain environment metadata when a seed is fixed? Answer: library algorithms and hardware rounding can change the numeric trajectory.')
add('02_Dataset_Acquisition_and_Inventory','Which metadata must be verified before importing real RF?',r'''Archive filenames are not a schema. Inspect every candidate array key, shape and dtype; identify axes from source documentation; record displacement direction and units. Legacy MATLAB and HDF5-backed MATLAB use different storage routes. Canonical files are 2D pairs with separate metadata, and their content checksums detect accidental changes. The real IMPACT adapters require an explicit mapping; an empty mapping fails instead of guessing.''',"""from elastograde.data.acquisition import instructions
from elastograde.data.inventory import inventory
from elastograde.data.schemas import save_sample
from elastograde.data.synthetic import make_pair
p=save_sample(output_dir/'inventory_fixture.npz',make_pair(seed))
print(instructions('impact-acquired'))
print(json.dumps(inventory(output_dir),indent=2)[:5000])""",'The inventory here describes a generated fixture only. Real acquisition groups, archive keys, license conditions and source spacing still need verification from downloaded IMPACT files.','Can a long axis automatically be called axial? Answer: no; array dimensions alone cannot establish acquisition orientation.')
add('03_RF_Physics_and_Preprocessing','How do RF, envelope and B-mode differ?',r'''RF is a signed oscillatory signal. An analytic signal along the axial direction is $s_a=s+i\mathcal H(s)$, and its envelope is $|s_a|$. Display B-mode is $20\log_{10}(A/A_{max})$ clipped to a chosen dynamic range. These transformations are not interchangeable: envelope and display compression discard information useful for phase-sensitive motion estimation. Pulse-echo axial spacing is $c/(2f_s)$ only when the raw sampling model applies; lateral spacing is independent metadata.''',"""from elastograde.data.synthetic import make_pair
from elastograde.data.preprocessing import envelope,bmode,normalize_pair
from elastograde.mechanics.coordinates import axial_spacing
s=make_pair(seed);a,b=normalize_pair(s.pre,s.post)
fig,axes=plt.subplots(1,3,figsize=(11,3))
for ax,img,title in zip(axes,[s.pre,envelope(s.pre),bmode(s.pre)],['RF-like signed fixture','Envelope','B-mode dB']):
 ax.imshow(img,aspect='auto',cmap='gray');ax.set_title(title)
plt.suptitle('synthetic_fixture');plt.tight_layout();plt.show()
print('Illustrative 40 MHz spacing in mm:',axial_spacing(40e6))""",'B-mode is for visual inspection, not a replacement for signed RF training inputs. The calculated spacing is an assumption-based worked example, not a recovered metadata value.','Why is an ultrasound screenshot unsuitable as RF input? Answer: rendering, compression and color mapping have destroyed the signed sampled waveform.')
add('04_Grouped_Splits_and_Leakage_Audit','How do we prevent dependent pairs from crossing splits?',r'''Statistical independence belongs to an acquisition or patient, not a patch. One geometry can produce many scatterer realizations, compression levels and frame pairs. For a geometry-generalization question, all descendants remain in one partition. If true group metadata is absent, duplicate detection does not prove independence. Separate train, validation, calibration and test roles protect fitting, model selection, uncertainty tuning and final evaluation.''',"""from elastograde.data.splits import grouped_split
from elastograde.data.audit import audit
records=[{'sample_id':f's{i}','geometry_id':f'g{i//2}','frame_ids':[f'f{i}']} for i in range(24)]
split=grouped_split(records,seed=seed)
print(audit(split));print({role:sum(r['split']==role for r in split) for role in ['train','validation','calibration','test']})
a=split[0];b=next(r for r in split if r['split']!=a['split']);b['frame_ids']=a['frame_ids']
try:audit(split)
except ValueError as e:print('Expected rejection:',e)""",'The deliberate shared-frame leak is rejected. This protects declared lineage; it cannot recover unrecorded acquisition identity.','Why split before patch generation? Answer: otherwise nearly identical patches from the same source can appear in training and testing.')
add('05_Classical_Displacement_Baselines','Does local correlation recover known subpixel motion?',r'''Windowed normalized cross-correlation subtracts local means and divides covariance by local standard deviations. An integer displacement search locates the strongest correlation peak. A parabolic fit around that peak provides a local subpixel estimate, but its assumptions can fail with repeated RF phase peaks or decorrelation. The baseline searches signed correlation and declares invalid borders rather than interpreting padded values as tissue.''',"""from elastograde.data.synthetic import make_pair
from elastograde.baselines.ncc import estimate
from elastograde.baselines.regularized import refine
from elastograde.evaluation.metrics import displacement_metrics
s=make_pair(seed);u,c,m=estimate(s.pre,s.post);v=refine(u,c)
print('NCC:',displacement_metrics(u,s.displacement,m))
print('Regularized:',displacement_metrics(v,s.displacement,m))
fig,axes=plt.subplots(1,2,figsize=(9,3))
axes[0].imshow(u[1],cmap='coolwarm');axes[0].set_title('Axial displacement pixels')
axes[1].imshow(c,cmap='viridis');axes[1].set_title('NCC proxy confidence')
plt.suptitle('synthetic_fixture');plt.show()""",'An accurate constant translation does not establish strain accuracy. Strain differentiates motion, amplifying small spatial errors. The regularizer is an original simple quadratic smoother, not a reproduction of a named published solver.','Why can high correlation still select the wrong displacement? Answer: narrowband RF can have multiple similar phase peaks.')
add('06_Strain_and_Mechanical_Features','What physical quantities can displacement identify?',r'''For small deformation, $\epsilon_{zz}=\partial u_z/\partial z$, $\epsilon_{xx}=\partial u_x/\partial x$ and $\epsilon_{xz}=\frac12(\partial u_x/\partial z+\partial u_z/\partial x)$. Convert displacement and coordinates consistently. Engineering shear is twice tensor shear. A least-squares derivative reduces noise at the cost of spatial resolution. Freehand strain without a known stress field cannot determine an absolute Young modulus.''',"""from elastograde.data.synthetic import make_pair
from elastograde.mechanics.strain import strain
from elastograde.mechanics.features import mechanical_features
s=make_pair(seed,kind='compression');v=strain(s.displacement,s.metadata['spacing_mm'],'least_squares')
print('Expected compression strain -0.025; observed median:',np.median(v['axial']))
print(mechanical_features(v['axial'],s.valid,s.roi,s.metadata['spacing_mm']))
plt.imshow(v['axial'],cmap='coolwarm',vmin=-.03,vmax=0);plt.colorbar();plt.title('synthetic_fixture: analytic axial strain');plt.show()""",'This fixture has uniform analytic compression. Its elliptical ROI is illustrative, not a physically simulated hard inclusion. A strain-ratio calculation is a software demonstration only.','If displacement is in pixels but the derivative denominator is in millimeters, what happens? Answer: units are inconsistent and the strain is scaled incorrectly.')
add('07_Self_Supervised_Learning','Can a compact flow network train without motion labels?',r'''A backward sampling field reconstructs reference RF as $I_m(x+u(x))$. The self-supervised objective combines RF/envelope similarity, forward-backward consistency, anisotropic smoothness and a Jacobian penalty. None uses reference motion when the supervised weight is zero. A low reconstruction loss can still accompany incorrect motion; validate on independent known-motion data. The candidate uses shared encoders, local feature correlation and coarse-to-fine residual refinement.''',"""from elastograde.config import load
from elastograde.training.engine import train
cfg=load(ROOT/'configs/demo.yaml');cfg.update(output=str(output_dir/'training'),epochs=1,samples=8)
summary=train(cfg)
print(summary)
rows=[json.loads(x) for x in (output_dir/'training/history.jsonl').read_text().splitlines()]
plt.plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],'o-',label='train');plt.plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],'s-',label='validation');plt.legend();plt.title('synthetic_fixture: one-epoch smoke training');plt.show()""",'This is a genuinely executed tiny training job. One epoch verifies execution and gradient flow; it is not a converged model or a state-of-the-art result.','Why keep validation groups independent even for self-supervision? Answer: selecting hyperparameters from dependent examples can still overestimate generalization.')
add('08_Simulation_Supervision_and_Adaptation','How should training regimes be compared?',r'''Pure self-supervision uses observed pairs. Simulation supervision uses reference displacement generated by a simulator. Hybrid learning starts from a simulation-supervised checkpoint and adapts using unlabeled acquired RF. Calling all three label-free would hide a substantial difference in information access. Evaluate them on identical held-out groups with comparable tuning budgets.''',"""from elastograde.config import load
from elastograde.training.engine import train
base=load(ROOT/'configs/demo.yaml');base.update(samples=8,epochs=1)
supervised=dict(base,output=str(output_dir/'supervised'),regime='simulation_supervised',loss=dict(base['loss'],rf=0.,env=0.,sup=1.))
a=train(supervised)
hybrid=dict(base,output=str(output_dir/'hybrid'),regime='hybrid',pretrained=str(output_dir/'supervised/best.pt'))
b=train(hybrid)
print({'supervised':a['status'],'adaptation':b['status'],'data_origin':'synthetic_fixture','real_domain_adaptation':'not_evaluated'})""",'Both stages run here on toy fixtures to exercise checkpoint transfer. This is not an IMPACT simulation-to-acquisition domain adaptation result. Real experiments require the verified manifest.','Does a self-supervised adaptation stage erase the supervised pretraining history? Answer: no; the full training provenance must remain visible.')
add('09_Mechanical_Representations_and_Clustering','Do mechanical clusters remain stable and free of nuisance confounding?',r'''Standardization and dimensionality reduction are fitted on training features. A Gaussian mixture supplies soft membership in descriptive mechanical clusters. BIC balances fit and model complexity; it does not discover the true number of clinical grades. Bootstrap adjusted Rand agreement measures clustering stability under resampling. A nuisance-prediction probe asks whether embeddings encode scanner or compression conditions.''',"""from elastograde.models.representations import MechanicalPhenotypes,nuisance_probe
rng=np.random.default_rng(seed);X=np.r_[rng.normal(-1,.4,(25,4)),rng.normal(1,.4,(25,4))]
m=MechanicalPhenotypes().fit(X,['train']*len(X),components=[2,3]);p=m.predict(X)
print(m.stability(X,repeats=4));print(p['names'][:4])
plt.scatter(X[:,0],X[:,1],c=p['probabilities'].argmax(1),cmap='viridis');plt.title('synthetic_fixture: descriptive clusters, not grades');plt.show()
print(nuisance_probe(X[:40],X[:40,0],X[40:],X[40:,0]))""",'The nuisance probe intentionally predicts a feature-derived nuisance, demonstrating how a representation can contain an unwanted shortcut. Cluster colors are arbitrary labels.','Can two well-separated clusters be called benign and malignant? Answer: not without independently verified diagnosis labels and an evaluated mapping.')
add('10_Uncertainty_and_Selective_Prediction','Does rejection reduce error while preserving useful coverage?',r'''Residual and cycle mismatch are uncalibrated quality proxies. Epistemic disagreement needs multiple independently fitted models. Calibration needs held-out groups. Selective risk is the error among accepted cases; coverage is the accepted fraction. Reporting error alone rewards rejecting nearly everything. Conformal group-level intervals require a clear coverage event and exchangeability; their coverage is not guaranteed under arbitrary scanner shift.''',"""from elastograde.uncertainty.selective import risk_coverage
from elastograde.uncertainty.calibration import conformal_radius,TemperatureScaler,reliability
rng=np.random.default_rng(seed);errors=rng.uniform(0,1,30);uncertainty=errors+rng.normal(0,.15,30)
r=risk_coverage(errors,uncertainty)
plt.plot(r['coverage'],r['risk']);plt.xlabel('Retained independent groups');plt.ylabel('Mean toy error');plt.title('synthetic_fixture: selective risk');plt.show()
print('Group nonconformity radius:',conformal_radius(errors,.1))
logits=rng.normal(size=(40,3));labels=rng.integers(0,3,40)
t=TemperatureScaler().fit(logits,labels,['calibration']*40)
print('Temperature:',t.temperature);print(reliability(t.predict(logits),labels))""",'The uncertainty-error association is deliberately generated for teaching, not measured model calibration. Evaluating the fitted temperature on the same calibration sample is a fit diagnostic, not held-out validation.','What should accompany an accepted-case accuracy? Answer: coverage, rejection criteria and performance on the rejected group when reference labels exist.')
add('11_MISV_Bayesian_Networks','How do factors combine evidence about reliability?',r'''A Bayesian network factorizes a joint distribution as $P(V)=\prod_iP(V_i\mid Pa_i)$. In the illustrative quality graph, domain influences quality and reliability, while quality influences residual, cycle evidence and uncertainty. Inference sums over unobserved variables. Variable elimination reuses factors instead of enumerating every assignment. The graph is an assumed probabilistic structure, not a demonstrated causal model.

Representation learning can introduce a latent quality state explaining several observed binary features. EM alternates posterior membership and parameter updates; latent state names can permute, so state 1 is not intrinsically good quality.''',"""from elastograde.pgm.bayesian_quality import illustrative_network
from elastograde.pgm.latent_quality import BernoulliMixture
bn=illustrative_network();e={'residual':0,'cycle':0}
print('Variable elimination:',bn.query('reliable',e));print('Enumeration:',bn.enumerate_query('reliable',e))
rng=np.random.default_rng(seed);latent=rng.integers(0,2,80);X=(rng.random((80,3))<np.where(latent[:,None]==1,.85,.2)).astype(int)
em=BernoulliMixture().fit(X,['train']*80)
plt.plot(em.history);plt.title('synthetic_fixture: latent mixture fit');plt.ylabel('Mean log likelihood');plt.show()
print('Learned emissions:',em.p)""",'Network CPDs are assumed educational numbers. EM emissions are learned from synthetic observations. Neither output is a clinically calibrated reliability estimate.','If residual and cycle share a hidden parent, are they marginally independent? Answer: generally not. They can be conditionally independent given that parent in the assumed graph.')
add('12_MISV_Markov_Networks_and_MRF','When does spatial consistency erase useful structure?',r'''An undirected model represents a distribution with clique potentials and a partition function. Graph separation states conditional independencies under the appropriate model assumptions. A Potts energy adds a penalty when neighboring labels disagree. If unaries depend on measured strain, the implemented posterior is a conditional random-field formulation. ICM finds a local optimum; exact enumeration is reserved for tiny graphs. Increasing smoothness can remove genuine small inclusions.''',"""from elastograde.pgm.markov_network import separated
from elastograde.pgm.mrf import icm,exact_map,energy
print('a independent of c given b:',separated([('a','b'),('b','c')],{'a'},{'c'},{'b'}))
rng=np.random.default_rng(seed);truth=np.zeros((18,18),int);truth[7:11,7:11]=1
obs=truth+rng.normal(0,.35,truth.shape);unary=np.stack([obs**2,(obs-1)**2],-1)
labels,history=icm(unary,beta=.3)
fig,axes=plt.subplots(1,3,figsize=(10,3))
for ax,img,title in zip(axes,[truth,unary.argmin(-1),labels],['Toy reference','Independent unary','Conditional Potts']):ax.imshow(img,cmap='viridis');ax.set_title(title)
plt.suptitle('synthetic_fixture');plt.show();print('Energy:',history)
small=unary[:2,:2];print('Tiny exact MAP:',exact_map(small)[1])""",'Improved visual smoothness is not sufficient evidence: evaluate small-region sensitivity and boundary error. Large beta can favor a uniform image even when a small true inclusion exists.','Does zero correlation prove Markov conditional independence? Answer: no; conditional independence is a statement about the full conditional distribution.')
add('13_MISV_Hidden_Markov_Models','Why do filtering, smoothing and Viterbi differ?',r'''The HMM assumes a hidden Markov state and conditionally independent observations. Filtering uses observations up to time $t$; smoothing uses the whole sequence; Viterbi selects the most probable complete state path. They answer different questions. Sequence boundaries must reset the initial state. A collection of unrelated compression pairs cannot be converted into a temporal experiment by sorting filenames.''',"""from elastograde.pgm.hmm import illustrative_hmm
h=illustrative_hmm();obs=[0,0,0,1,1,1,2,2,0,0];r=h.infer(obs,ordered=True)
plt.plot(r['filtered'][:,0],label='Filtered stable');plt.plot(r['smoothed'][:,0],label='Smoothed stable');plt.step(range(len(obs)),r['viterbi']/2,label='Viterbi state /2');plt.legend();plt.title('synthetic_fixture: ordered categorical sequence');plt.show()
print('Sequence likelihood:',r['log_likelihood'])
try:h.infer(obs)
except ValueError as e:print('Expected gate:',e)""",'These are synthetic ordered observations with assumed transitions. True IMPACT sequence availability remains unverified. No real temporal reliability claim follows.','Which inference can be used online without future frames? Answer: filtering; smoothing uses future observations.')
add('14_MISV_Moralization_and_Decision_Networks','How can the same joint distribution support risk-sensitive decisions?',r'''Moralization connects co-parents and removes arrow directions. Assigning each Bayesian CPD to its family clique preserves the factor product and therefore the joint distribution. The undirected graph can lose some independencies; moralization is not triangulation or causal discovery. A decision network adds actions and utilities to uncertain states. The Bayes action minimizes expected loss: $a^*=\arg\min_a\sum_sL(a,s)P(s\mid e)$.

For reliability probability 0.8, the illustrative acceptance loss is $10(1-0.8)=2$, reacquisition loss is 1, and review loss is $2(0.2)+0.5(0.8)=0.8$. Review wins under these assumed costs.''',"""from elastograde.pgm.bayesian_quality import illustrative_network
from elastograde.pgm.moralization import verify
from elastograde.pgm.decisions import choose,sensitivity
print(verify(illustrative_network()));print(choose(.8))
ps=np.linspace(0,1,40);rows=sensitivity(ps,[5,10,20]);names={'accept_mechanics':0,'reacquire':1,'expert_review':2}
for cost in [5,10,20]:plt.step(ps,[names[r['action']] for r in rows if r['false_accept_cost']==cost],label=f'False-accept cost {cost}')
plt.yticks([0,1,2],list(names));plt.xlabel('Assumed reliability probability');plt.legend();plt.title('Educational decision sensitivity');plt.show()""",'Exact probability agreement validates the conversion code on the small graph. Decision changes reflect utility assumptions, not a clinical treatment recommendation.','Why can the moral graph contain more edges? Answer: parent variables become neighbors because their shared child CPD forms a joint family factor.')
add('15_Clinical_Ordinal_Grading_Readiness','What changes when genuine histopathology-linked grades become available?',r'''Ordinal grades have ordered categories, so a proportional-odds model uses shared feature effects with ordered thresholds. Cumulative probabilities must be monotone so class probabilities stay nonnegative. Patient-grouped fitting and independent calibration remain mandatory. An organ-specific grade is not interchangeable with malignancy, imaging risk score or stage. Synthetic class labels test the model interface only.''',"""from elastograde.capabilities import clinical_readiness
from elastograde.models.ordinal import OrdinalLogistic,OrdinalHead
from elastograde.evaluation.metrics import clinical_metrics
rng=np.random.default_rng(seed);X=rng.normal(size=(120,3));y=np.digitize(X[:,0]+rng.normal(0,.15,120),[-.5,.5])
m=OrdinalLogistic().fit(X[:80],y[:80],['train']*80)
p=m.predict_proba(X[80:]);print('Synthetic interface metrics:',clinical_metrics(p,y[80:]))
print('Real clinical readiness:',clinical_readiness({'records':[]}))
plt.imshow(clinical_metrics(p,y[80:])['confusion_matrix'],cmap='Blues');plt.title('synthetic_fixture: ordinal interface only');plt.xlabel('Predicted');plt.ylabel('Generated target');plt.show()""",'These metrics concern mathematically generated labels, not tumors. A real study must supply linked reference pathology and validate the model on independent patients.','Does a successful synthetic ordinal classifier prove tumor grading? Answer: no; it proves that optimization and probability interfaces work on a constructed problem.')
add('16_Benchmarks_Ablations_and_Robustness','Which added components actually improve performance?',r'''Ablations remove one contribution at a time while holding data access and tuning budgets fixed. Paired comparisons use the same independent groups. Group bootstrap confidence intervals resample geometries or patients, not dependent pixels. Report computational cost alongside accuracy. Robustness perturbations are stress tests; they do not recreate every scanner domain shift.''',"""from elastograde.data.synthetic import make_pair
from elastograde.evaluation.robustness import corrupt
from elastograde.inference.pipeline import infer
from elastograde.evaluation.metrics import displacement_metrics
from elastograde.evaluation.grouped_statistics import grouped_bootstrap
rows=[]
for i in range(4):
 s=make_pair(seed+i)
 for severity in [0,.2,.5]:
  c=corrupt(s,'noise',severity,seed+i);r=infer(c)
  rows.append({'group':i,'severity':severity,**displacement_metrics(r['flow'],s.displacement,r['valid'])})
for severity in [0,.2,.5]:
 vals=[r['endpoint_error'] for r in rows if r['severity']==severity]
 print(severity,grouped_bootstrap(vals,list(range(4)),repeats=200))
plt.plot([r['severity'] for r in rows],[r['endpoint_error'] for r in rows],'o');plt.title('synthetic_fixture: NCC noise stress');plt.xlabel('Noise fraction');plt.ylabel('EPE pixels');plt.show()""",'Only four toy groups are used here. Confidence intervals are illustrative and small-sample uncertainty is substantial. Real benchmark runs are not implied.','Why compare models on identical groups? Answer: paired differences reduce variation caused by unequal test difficulty and make the scientific comparison interpretable.')
add('17_Risk_Analysis_and_Failure_Atlas','Which failures can the system detect, and which remain invisible?',r'''A quality detector can identify some failure evidence, such as high residual or cycle mismatch. It cannot infer missing pathology or solve modulus identifiability. The research risk register links each hazard to a verification procedure and a residual limitation. Hand-assigned costs and CPDs are educational assumptions. They must not become calibrated medical risk probabilities through a change of label.''',"""from elastograde.data.synthetic import make_pair
from elastograde.evaluation.robustness import corrupt
from elastograde.inference.pipeline import infer
s=make_pair(seed,kind='compression');rows=[]
for name,kind,severity in [('clean','noise',0),('noisy','noise',1),('dropout','dropout',.3)]:
 r=infer(corrupt(s,kind,severity));rows.append((name,r['quality'],r['decision']))
print(json.dumps(rows,indent=2))
plt.bar([r[0] for r in rows],[r[1]['residual_mean'] for r in rows]);plt.title('synthetic_fixture: observed quality failures');plt.ylabel('Mean RF residual');plt.show()""",'Threshold actions are demonstrative. A missed failure is useful research evidence and should stay in the failure atlas instead of being hidden by removing a difficult sample.','Can a low residual guarantee correct physical displacement? Answer: no; ambiguous texture and model shortcuts may produce a good reconstruction with wrong motion.')
add('18_End_to_End_CLI_and_Research_Report','Can the documented CLI generate inspectable artifacts?',r'''Reproducible software exposes the same computational services to notebooks, CLI and terminal UI. The report records arrays, numeric summaries, software provenance and data origin. HTML rendering escapes text and links to local figures. A report is evidence only for its recorded run; it does not transform a fixture demonstration into a clinical experiment.''',"""import subprocess,sys
run=output_dir/'cli_demo'
p=subprocess.run([sys.executable,'-m','elastograde','demo','--out',str(run),'--seed',str(seed)],capture_output=True,text=True,check=True)
summary=json.loads(p.stdout);print('EPE pixels:',summary['metrics']['endpoint_error']);print('Data origin:',summary['data_origin'])
from IPython.display import Image,display
if (run/'panel.png').exists():display(Image(filename=str(run/'panel.png')))
print('Report:',run/'report.html')
assert summary['grade_probabilities'] is None""",'The command has executed and produced maps and a report from synthetic data. Use the inventory, explicit adapter, split audit and training commands for a verified real-data study.','What should accompany every performance table? Answer: data origin, independent sample counts, split protocol, units, model provenance and the actual run artifacts.')
setup="""from pathlib import Path
import os,json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path.cwd()
if not (ROOT/'configs').exists():ROOT=ROOT.parent
mode='demo'
data_root=ROOT/'data/raw'
config_path=ROOT/'configs/demo.yaml'
seed=42
max_samples=8
device='cpu'
output_dir=ROOT/'reports/notebooks'/NOTEBOOK_NAME
output_dir.mkdir(parents=True,exist_ok=True)
print('Mode:',mode,'| Data origin: synthetic_fixture')
"""
for name,question,theory,code,interpret,exercise in chapters:
 nb=nbf.v4.new_notebook();nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
 nb.cells=[nbf.v4.new_markdown_cell('# '+name[3:].replace('_',' ')+'\n\n**Question:** '+question+'\n\n**Prerequisites:** install the project with notebook extras; run from the repository root. This notebook is self-contained in bounded demo mode.'),nbf.v4.new_markdown_cell('## Theory and assumptions\n\n'+theory),nbf.v4.new_code_cell(setup.replace('NOTEBOOK_NAME',repr(name))),nbf.v4.new_markdown_cell('## Experiment\n\nInputs: seeded synthetic fixtures or explicitly identified educational probability tables. Outputs: printed numerical evidence, saved run artifacts where applicable, and plots. Real-data-only claims remain unavailable until source archives and metadata are verified.'),nbf.v4.new_code_cell(code),nbf.v4.new_markdown_cell('## Interpretation and failure cases\n\n'+interpret),nbf.v4.new_markdown_cell('## Check your understanding\n\n'+exercise),nbf.v4.new_markdown_cell('## Evidence statement\n\nThis execution demonstrates the computations shown above. It does not establish clinical accuracy, generalization to patients, or state-of-the-art performance. Record the environment and split protocol before a full experiment.')]
 p=R/'notebooks'/f'{name}.ipynb';p.parent.mkdir(exist_ok=True);nbf.write(nb,p)
