> Archived original modular README. Relative paths below were originally relative to the repository root. Use the root README.md for the standalone pipeline.

# ElastoGrade Research

**Ultrasound Elastography for Label-Free Tissue Tumor Grading**

Physics-guided label-free elastography, probabilistic tissue characterization, and a clinically gated grading extension.

This repository is an executable research workbench for RF motion estimation, strain calculation, tissue-mechanics features, uncertainty analysis and probabilistic reasoning. It includes 19 educational notebooks, a Python package, a command-line interface, an interactive terminal application, synthetic verification data, tests and local research reports. It is intended to support a carefully designed study; it is not a trained clinical tumor-grading product.

The requested Concordia IMPACT resources describe acquired and simulated **phantom** data. They support studying displacement estimation and mechanical characterization. Their published descriptions do not establish pathology-linked tumor grades. This repository therefore keeps pathology outputs unavailable until a separate clinical dataset, grading definition, patient-level validation and compatible model provenance are supplied. Mechanical clusters are never renamed as cancer grades.

## Quick start

Use Python 3.11 or later; the included execution evidence was produced with Python 3.12 on CPU. From the extracted repository folder:

```bash
python -m venv .venv
```

Activate on Linux/macOS with `source .venv/bin/activate`, or on Windows PowerShell with `.venv\Scripts\Activate.ps1`. If PowerShell policy prevents activation, invoke `.venv\Scripts\python.exe` directly rather than modifying a machine-wide policy. Install CPU PyTorch and the project:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[notebooks,tui,dev]"
python -m elastograde doctor
python -m elastograde demo --out reports/demo
python -m pytest -q
python -m elastograde tui
```

Open `reports/demo/report.html` locally. It contains a measured synthetic-fixture result, not a mock clinical prediction. For a short genuine neural training run:

```bash
python -m elastograde train --config configs/demo.yaml
python -m elastograde evaluate --run reports/training --split test
python scripts/execute_notebooks.py
```

The `elastograde` console command and `python -m elastograde` call the same application. The module invocation is convenient when a user-level script directory is not on PATH. Network access is only required to install dependencies or acquire external datasets; the installed demo does not download weights or data.

## Contents

1. [Research purpose and evidence](#research-purpose-and-evidence)
2. [Ultrasound and mechanical theory](#ultrasound-and-mechanical-theory)
3. [Data acquisition and contracts](#data-acquisition-and-contracts)
4. [Learning and inference](#learning-and-inference)
5. [Representations and clinical extension](#representations-and-clinical-extension)
6. [Uncertainty and selective prediction](#uncertainty-and-selective-prediction)
7. [MIS-V mathematical guide](#mis-v-mathematical-guide)
8. [Experimental design and statistics](#experimental-design-and-statistics)
9. [Software operation](#software-operation)
10. [Risk, limitations and validation](#risk-limitations-and-validation)
11. [Learning path and notebook guide](#learning-path-and-notebook-guide)
12. [FAQ](#faq)
13. [Glossary and symbols](#glossary-and-symbols)
14. [References and contribution guide](#references-and-contribution-guide)

## Research purpose and evidence

### The central question

Can a compact, acquisition-aware motion model improve strain fidelity and selective reliability under domain shift, while retaining interpretable mechanical features? Answering this requires more than building an attention network. It requires a defined population of RF pairs, independent evaluation groups, a defensible reference, strong comparison methods and an honest account of uncertainty. The project supplies those engineering components and an initial protocol. Whether the proposed architecture improves real-data results remains an empirical question.

The main mechanics endpoint is displacement endpoint error on verified reference fields, aggregated by independent geometry or acquisition. Axial strain error is a key secondary endpoint. On acquired phantom RF without reference motion, reconstruction residual, correlation and cycle consistency are explicitly proxies. They are useful failure signals but cannot become ground-truth accuracy merely because they are easy to compute. A separate clinical study would need pathology-linked endpoints.

### Three meanings of label-free

A model may learn displacement without supplied displacement labels because a reconstructed RF frame supplies its training signal. A representation encoder may learn without pathology grades, yet later feed a supervised grading head. At inference, a system may operate without a new biopsy even though development required biopsy labels. These statements concern different parts of the workflow. The README, configuration and output provenance keep them separate.

The `self_supervised` regime disallows a nonzero supervised loss. The `simulation_supervised` regime requires a supervised term and verified simulated reference fields. The `hybrid` regime requires a pretraining checkpoint and records adaptation separately. A synthetic fixture has analytic motion, but that does not turn it into a validated acoustic simulation. All fixture-derived outputs retain `data_origin: synthetic_fixture`.

### Modes and claims

| Mode | Inputs | Permitted research output | Unsupported claim |
| --- | --- | --- | --- |
| Mechanics | RF pair and verified metadata | Displacement, strain, quality, descriptive features | Histological grade |
| Simulation supervision | RF and simulation reference motion | Supervised motion model and controlled error analysis | Clinical generalization |
| Clinical research extension | Linked imaging, pathology and patient groups | Readiness assessment and ordinal-model research | Validated clinical deployment without a clinical study |

Implementation means code exists. Execution means a recorded run succeeded. Empirical support means a prespecified evaluation supports a claim. Clinical validity requires appropriate clinical evidence. These are not interchangeable milestones. `docs/claims_ledger.md` and `IMPLEMENTATION_STATUS.md` identify their current boundaries. Synthetic classification accuracy only verifies an interface on generated labels; it is never presented as tumor-grading accuracy.

### Candidate contribution and related work

The candidate combines shared anisotropic feature extraction, local correlation, coarse-to-fine motion refinement, bidirectional constraints and interpretable quality reasoning. Existing ultrasound research has already explored pyramidal displacement estimation and semi-supervised adaptation. Consequently, this repository does not claim those broad ideas as novel. A contribution could lie in a carefully validated interaction among RF/envelope inputs, acquisition metadata, confidence modeling and selective reliability, but only if comparisons establish an advantage.

The appropriate comparison begins with zero motion and windowed NCC, continues to a compact U-Net-like registration model, and then tests the pyramid model. Match independent groups, valid masks and tuning budgets. Publish negative findings: if attention adds cost but not robustness, remove it. If a simple quality threshold performs as well as a Bayesian layer, report that rather than treating mathematical complexity as evidence of value.

## Ultrasound and mechanical theory

### What an ultrasound measurement contains

An ultrasound transducer emits an acoustic pulse and receives echoes from spatial variations in acoustic properties. The recorded waveform contains time-varying signed values. Under a pulse-echo model, later arrival times correspond approximately to deeper structures, but the conversion depends on assumed sound speed and acquisition processing. A 2D RF array represents many axial samples across lateral scan lines; it is not simply an ordinary grayscale photograph.

Speckle arises from interference among echoes from unresolved scatterers. When tissue moves coherently, local echo patterns can be tracked. If scatterers move out of the imaging plane, acquisition conditions change, or compression is too large, patterns decorrelate. A registration algorithm may then find a mathematically plausible match that is physically wrong. This is why acquisition quality and validity masks belong in the pipeline rather than being added only after model training.

### RF, IQ, envelope and B-mode

RF is the sampled signed waveform. IQ is a complex baseband representation when available from the acquisition system. The Hilbert analytic signal can provide an envelope from real RF along a verified axial axis. Envelope magnitude removes the signed carrier oscillation. B-mode applies logarithmic compression and often additional display processing. Every transformation changes the information available to the estimator.

For real RF `r(z,x)`, define the analytic signal `a=r+i H_z(r)` and envelope `A=|a|`. A display convention is `B=20 log10(max(A,epsilon)/Amax)`, clipped to a fixed dynamic range such as 60 dB. That range is a visualization choice. The software preserves original RF and keeps display clipping out of numeric strain metrics. An exported color elastogram or screenshot cannot be treated as equivalent to RF because the original phase and sampling values are unavailable.

### Compression versus shear-wave elastography

Compression elastography estimates deformation between pre- and post-compression RF acquisitions. The local displacement field is differentiated to obtain strain. Under comparable loading, a stiffer inclusion may deform less than surrounding material, which motivates relative mechanical characterization. Stress is usually not known everywhere in freehand compression, so absolute modulus recovery is generally not identifiable from strain alone.

Shear-wave elastography instead analyzes wave propagation induced by a suitable excitation. Wave-speed-to-modulus relationships depend on density, material assumptions, geometry and the wave model. Applying such a relationship to a pair of quasi-static compression frames is a category error. This repository does not compute kPa values from compression strain or pretend that an ultrasound RF carrier frequency is a shear-wave propagation measurement.

### Coordinate and sampling contract

Arrays use `[z,x]`: axial coordinate first, lateral second. Flow uses two channels `[ux,uz]` in pixel units. This differs from array indexing order, and the distinction is deliberate: grid-sampling coordinates are ordered lateral then axial. A flow vector specifies where to sample the moving image for a given reference location. Thus `warped(x)=moving(x+u(x))`; a positive axial component samples deeper in the moving frame.

If the sampling frequency is 40 MHz and an assumed sound speed is 1540 m/s, raw pulse-echo axial spacing is 0.01925 mm before accounting for decimation or other processing. This is a worked example, not a hard-coded universal metadata value. Lateral spacing must come from acquisition geometry. Resampling changes spacing and displacement scaling. Canonical loaders never infer axes by selecting the longer dimension or silently transpose all MATLAB arrays.

### Composition and reference frames

Suppose `a` maps reference coordinates into an intermediate image and `b` maps intermediate coordinates into a final image. Their composed displacement is `a(x)+b(x+a(x))`. Adding fields at the same indices without resampling is generally not composition. The code implements this operation and a known-motion test. The same issue arises when generating arbitrary frame pairs from displacement fields measured relative to a first frame.

The IMPACT simulation description mentions linear motion, but the archive's precise coordinate convention must still be verified. If both reference fields are defined on the same original grid, subtraction describes relative material-point motion on that grid. A target defined on the first member of a new image pair may require regridding. The import adapter therefore accepts only an explicitly verified reference-to-moving convention in pixel units; unsupported representations must be converted deliberately.

### Why subpixel estimation matters

Axial RF sampling can be fine compared with the displacements of interest. Restricting estimates to integer samples causes quantization and unstable derivatives. Windowed NCC finds integer peaks and refines them with a local parabola. This assumes a smooth, well-resolved peak. Repeated phase peaks, low signal variance or a maximum on the search boundary weaken that assumption.

The search radius must cover the plausible displacement range without inviting excessive false matches or expensive computation. Window size balances precision and spatial resolution: large windows stabilize noisy correlations but average across mechanically distinct regions; small windows preserve boundaries but have weaker statistical support. These are validation-selected parameters. A single image that looks smooth cannot determine the best operating point.

### Strain tensor and a numeric example

For small deformations, axial strain is `d uz / dz`, lateral strain is `d ux / dx`, and tensor shear is half the sum of the cross derivatives. Compression is negative under the implemented convention. If an axial displacement decreases by 0.2 mm over 10 mm, the average axial strain is `-0.02`, or -2%. Multiplying by 100 changes only the displayed percentage, not the stored dimensionless quantity.

Pixel flow can yield normal strain because matching axis spacing cancels, but physical shear mixes two axes and requires their spacing ratio. The software returns physical shear only when spacing is known. A displacement array expressed in millimeters must not be differentiated against pixel coordinates and labeled dimensionless. The `to_mm` conversion scales the lateral channel by lateral spacing and the axial channel by axial spacing.

### Differentiation, regularization and boundaries

Differentiation amplifies high-frequency noise. A displacement map with small pointwise errors can produce an unusable strain map if those errors vary rapidly. The repository compares ordinary finite differences with a local least-squares line derivative implemented through a first-order Savitzky-Golay filter. The latter reduces derivative variance while introducing a smoothing scale that can blur small inclusions.

Derivative settings belong to the experiment protocol. Do not select a new smoothing window separately for each test image because it looks better. Mask invalid borders, retain raw values for metrics and use fixed comparable color limits for figures. Finite-strain interpretations require a different tensor when deformation is large; the default small-strain analysis is not silently generalized to arbitrarily large compression.

### Stress, stiffness and identifiability

In a simple uniaxial linear elastic setting, modulus relates stress to strain. A spatially heterogeneous imaging problem involves boundary conditions, loading, constitutive assumptions and potentially unobserved out-of-plane motion. If the applied stress is unknown, changing both stress and modulus scale can leave an observed strain pattern compatible with multiple explanations. A neural network cannot remove this ambiguity merely by learning a plausible image-to-number mapping.

Simulation material parameters describe the simulator's configured mechanics, not necessarily an observed per-pixel modulus image supplied in the archive. Real phantom nominal material properties are useful contextual information but not automatically pixelwise reference maps. An inverse-modulus extension would require a stated forward model, appropriate boundary/load evidence and an identifiability study. It is intentionally outside the default inference outputs.

### Mechanical features and reference regions

The implemented features include median strain, median absolute deviation, lower and upper quantiles, invalid fraction, inclusion/background contrast, CNR, a background-over-ROI absolute-strain ratio, boundary-gradient summary and ROI area when spacing is known. CNR uses the absolute difference of regional means divided by the square root of the sum of regional variances. A zero denominator produces an unavailable value rather than an infinite performance claim.

Reference regions must be specified before comparing ratios. Different depths, loading conditions and shadowing can confound interpretation. A mask derived from the model's own strain map is not an independent reference for validating that map. The synthetic demo uses an illustrative ellipse and explicitly identifies it as such; its regional feature calculation exercises software, not a physical hard-inclusion experiment.

## Data acquisition and contracts

### Official resources

Start with the [acquired phantom dataset](https://users.encs.concordia.ca/~impact/ultrasound-elastography-dataset-for-unsupervised-training/) and [simulation dataset](https://users.encs.concordia.ca/~impact/ultrasound-elastography-simulation-database/). Their descriptions were checked on 22 September 2026. Follow the current Download links, preserve all accompanying READMEs and inspect the archive contents before writing mappings. The download folders were reachable during preparation, but their actual array layouts and file-level metadata were not available for verification here.

The acquired resource uses RF pairs from a physical phantom and a multipart archive. Obtain all volumes and begin extraction at `Data_121706.zip.001`; do not independently extract every volume. The simulation resource includes reference motion. Data licenses and redistribution permissions must be checked at the source. Public accessibility alone does not make the data part of this repository's MIT code license.

### Acquisition commands and manual route

```bash
python -m elastograde data instructions --dataset impact-acquired
python -m elastograde data instructions --dataset impact-simulation
python -m elastograde data inspect --root data/raw/impact-acquired --out reports/inventory
```

A verified HTTPS file URL can be used with `data download --url URL --destination PATH --checksum HASH`. The helper detects HTML responses, supports range-based continuation and checks a supplied checksum. If a provider requires interactive access, download manually and place the authorized archives in the local raw-data directory. Do not paste a login page URL and assume its saved HTML is a valid archive.

The extraction helper checks archive paths, symbolic-link entries, duplicate entries and space before extraction into an empty destination. For multipart archives, supply the verified expected volume count. Unknown counts are not guessed from the highest filename because a missing final volume would be invisible to that rule. Raw archives remain unmodified, and the project does not upload their contents anywhere.

### Inspect, map and prepare

The inventory lists source names, sizes, checksums and numeric array metadata. Legacy MAT inventory uses `whosmat`; HDF5 inventory walks dataset keys. Canonical import requires named `pre` and `post` arrays and can optionally map `ux`, `uz`, validity and ROI arrays. Arrays must represent explicit 2D pairs. Higher-dimensional datasets should be separated according to verified frame indexing before importing; silently squeezing a time or channel dimension could destroy meaning.

Edit `configs/datasets/impact.yaml` after inspection. Set `axes` to the verified value, map the exact keys, list source files and sample IDs, and supply verified group identifiers. The provided config intentionally uses `not_verified` and empty records so accidental training cannot proceed with guesses. This is a visible data dependency, not a fake default dataset. The simulation config additionally requires motion units and reference convention.

### Example mapping shape

```yaml
dataset_id: impact-acquired
root: data/raw/impact-acquired
output: data/processed/impact-acquired
axes: zx
spacing_mm: null
mapping:
  pre: VERIFIED_PRE_KEY
  post: VERIFIED_POST_KEY
records:
  - sample_id: acquisition_001_pair_001
    path: VERIFIED_RELATIVE_FILE.mat
    acquisition_id: VERIFIED_ACQUISITION_ID
    frame_ids: [VERIFIED_PRE_FRAME_ID, VERIFIED_POST_FRAME_ID]
```

The capitalized entries are explicit placeholders to replace using the actual archive. They are not claimed file keys. Keep unknown spacing null. Use geometry groups for simulations and acquisition groups for physical phantom sessions where those identities are verified. A source-file checksum protects file identity; it does not prove that two different files are independent acquisitions.

### Canonical schema and provenance

Each canonical pair is an NPZ file plus JSON metadata. The metadata includes schema version, sample ID, dataset/domain, source path/checksum, geometry/acquisition/sequence/patient IDs where applicable, frame lineage, frequency, spacing, labels if supplied, preprocessing configuration hash and coordinate conventions. Arrays are loaded with pickle disabled. A checksum mismatch fails clearly. The raw files are not rewritten.

The default NPZ storage favors transparent per-pair access and small fixtures. HDF5 source arrays are read by explicit keys. For very large studies, a chunked cache is available as an optional utility; choose chunk sizes from actual access patterns and benchmark throughput. A cache should depend on source checksum, processing configuration and code version. Changing normalization or array orientation must invalidate a cache rather than silently reuse old data.

### Preprocessing and augmentation

Pair-consistent normalization uses a shared mean and energy scale. This avoids treating pre/post intensity differences as entirely unrelated transformations. Envelope extraction runs along the verified axial axis. A bandpass filter requires a declared sampling rate and cutoff frequencies below Nyquist. Axial resampling uses polyphase filtering, updates axial spacing and rescales the axial displacement channel.

A horizontal flip negates lateral motion; an axial flip negates axial motion. Masks and coordinates must transform with images. Arbitrary rotations can exchange physical axes or invalidate probe geometry, so they are not a casual natural-image augmentation. Noise and gain stress tests are explicitly parameterized. Excessive artificial corruption is labeled as a stress test rather than asserted to represent a scanner population.

### Independent splits and lineage

The principal simulation protocol groups all descendants of a verified geometry. Train, validation, calibration and test roles are assigned before patch generation. Shared source frames, ancestor IDs, geometry IDs, patient IDs and acquisition IDs cannot cross partitions. The audit raises an error if the declared lineage reveals overlap. Source hash overlap is also checked conservatively; if a source container holds independent units, document and construct a trustworthy sample-level lineage rather than weakening the audit casually.

For patient data, group every lesion and session from one patient in an outer partition. Fit normalizers, PCA, clustering, CPDs and calibration only in their declared roles. A method using unlabeled final-test images during adaptation is transductive and must be reported separately from the primary inductive experiment. “No labels used” does not make test-image exposure irrelevant.

### Missing metadata and scope limits

The system distinguishes unavailable, unverified and inapplicable capabilities. Patient IDs are inapplicable to phantoms; temporal ordering may be unverified for a folder of pairs; pathology grades are not established by the requested dataset descriptions. Unknown values remain null instead of acquiring a plausible synthetic value. Explicit metadata is a necessary part of scientific reproducibility.

If independent acquisition identity cannot be recovered, qualify the experiment as within-phantom performance on the available grouping assumptions. Similarity clustering can detect duplicates but cannot prove independence. If temporal ordering is unknown, do not run the real-data HMM. If spacing is missing, retain clearly labeled pixel-motion outputs and withhold physical shear and area claims that depend on spacing.

## Learning and inference

### Baseline ladder

Zero motion provides a sanity floor and can be surprisingly competitive on very small deformations. NCC provides an interpretable classical reference with explicit search and window settings. Confidence-weighted quadratic smoothing supplies a simple regularized baseline. It is not represented as GLUE, OVERWIND or another published algorithm. A compact U-Net-like network establishes whether learning itself helps under the chosen data regime.

The pyramid candidate adds local feature correlation and multi-scale refinement. Compare it against the U-Net under matched training access. Simulation-supervised and hybrid variants reuse the same architecture so changes in information access are visible. External optical-flow models are optional future integrations requiring verified implementations and licenses; no downloaded foundation model is needed for the core project.

### Tensor walkthrough

For a single RF input, each frame enters as `[B,1,H,W]`. Fusion uses two channels per frame: normalized RF and envelope. The shared encoder produces feature maps at full, half and quarter resolution with a configurable channel width. At each scale, normalized feature correlations are computed for local offsets within a bounded radius. A radius of one creates nine offset channels.

The refinement block receives reference features, the correlation volume, current two-channel flow, a validity channel and four metadata channels. Those metadata channels encode scaled center/sample frequency plus two missingness indicators. Missing frequency is not represented as a known zero-frequency acquisition. Flow is upsampled between scales with component-specific pixel scaling. The output remains a full-resolution sampling field; strain is calculated afterward using a separate derivative estimator.

### Optional attention and confidence

Low-resolution cross-attention is configurable and disabled in the small default profile. It has a token-budget guard because global attention can make large RF arrays impractical. Its scientific value must be established by ablation. The model should not be advertised as more advanced merely because the architecture contains an attention layer.

An optional log-variance head supports a heteroscedastic Gaussian residual objective. The objective contains both an inverse-variance residual term and a log-variance penalty, preventing trivial unlimited uncertainty from reducing all residual costs. Bounded log variance adds numerical protection. This estimate describes the specified reconstruction likelihood, not a calibrated probability of displacement correctness or malignancy. Evaluate variance collapse, intensity dependence and correlation with reference error.

### Objective components

The default objective is a weighted sum of RF local-correlation loss, envelope correlation loss, forward/backward cycle mismatch, anisotropic smoothness and a folding penalty. Simulation supervision can add a robust displacement loss when the target reference is verified. Every component is logged separately. The loss configuration is saved alongside the checkpoint so a later figure can be traced to the actual training objective.

Local correlation uses masked moments and support normalization. Warp masks include both reference validity and sampled moving validity. Zero support raises an error. Float32 warping and loss calculation remain available when mixed precision is enabled for model operations. Nonfinite objectives fail instead of being silently replaced by zeros. This is important because a supposedly stable run can otherwise conceal invalid geometry or exploding gradients.

### Bidirectionality and mechanical assumptions

Forward/backward consistency checks whether `u_f(x)+u_b(x+u_f(x))` is near zero where both fields are supported. It helps identify inconsistent correspondences but is not a full physical law. Two mutually consistent wrong fields can still satisfy the constraint. The Jacobian determinant of `x+u(x)` indicates local folding, but a numerical folding warning is not a diagnosis of tissue pathology.

The smoothness term weights axial and lateral gradients differently. Those weights express a model preference, not an established universal material law. A 2D zero-divergence condition is not imposed as strict incompressibility because unobserved out-of-plane motion can violate a naive 2D formulation. A physics loss must match the imaging and material assumptions rather than simply borrow an attractive equation.

### Training and checkpoints

The training engine uses AdamW, gradient clipping, optional accumulation, optional CUDA mixed precision and deterministic seeds. The default CPU demonstration uses tiny arrays and a compact 14,766-parameter pyramid configuration. This number is measured for the supplied default architecture; changing width, radius, inputs or attention changes it. Full research training requires an explicit command rather than occurring on import.

Checkpoint files contain model state, optimizer state, epoch, best validation objective, configuration, random-number state and provenance. Resume rejects changes to critical model/regime/loss/seed settings. Only load trusted local checkpoints because optimizer and random-state objects require a richer serialization path. A new experiment should use a new output directory so its history is not confused with a previous run's append-only log.

### Inference and tiling

Classical inference estimates both directions, constructs validity, computes strain, extracts features and records quality. Neural inference loads the declared model configuration and checkpoint. Results are arrays plus a JSON summary, with no automatic pathology grade. Tiled prediction supports overlap blending for larger arrays, and a constant-field test verifies coverage and blending arithmetic.

Tiling is not guaranteed to match whole-image inference exactly: receptive-field boundaries and normalization may change predictions. Inspect seam-region motion and strain error on manageable images before adopting a tile size. Overlap should be chosen relative to the receptive field and expected displacement. Do not call a large tiled image validated merely because every output pixel has a numeric value.

## Representations and clinical extension

### Mechanical phenotype learning

The feature baseline combines robust strain summaries and quality measures. StandardScaler and optional PCA fit only on training rows. Gaussian mixtures produce soft membership, descriptive names and likelihood-based outlier flags. BIC selects among configured component counts, subject to enough training samples. A component name is an arbitrary index; label swapping does not indicate a biological change.

Bootstrap stability reports adjusted Rand agreement after resampling. Use one feature row per independent group or explicitly aggregate repeated measurements first. Stable clusters can still encode nuisance structure, such as compression level, frequency or geometry. A held-out nuisance-prediction probe checks this possibility. Attractive two-dimensional embedding plots are exploratory figures rather than independent evidence of clinical discrimination.

### Learned embeddings

The motion network exposes pooled encoder features. These can support a representation experiment once motion estimation is functioning. A variance-preserving loss utility is provided for paired representations, but it is not enabled as an unsupported claim that a new self-supervised foundation model has been trained. Compare learned embeddings against the interpretable feature baseline and evaluate whether they add information beyond nuisance variables.

Any feature selection, PCA dimension choice, mixture count or ordinal head tuning belongs within the training/validation protocol. Never fit a representation on all subjects and then call a split of its features an independent test. If a researcher deliberately uses transductive representations, report that design and compare it separately. Label-free does not mean free of information leakage.

### Ordinal logistic and neural models

The proportional-odds model uses a linear score and ordered cutpoints. Each cumulative probability is a sigmoid of score minus cutpoint. Consecutive differences produce class probabilities. Cutpoint increments are parameterized with a positive softplus transform so their order cannot reverse. Optimization minimizes negative log likelihood with a small coefficient penalty.

The neural ordinal head uses the same shared-score, ordered-threshold concept and a cumulative binary loss. This is preferable to treating arbitrary grade integers as the sole MSE target. The head is deliberately small because realistic clinical cohorts may contain few independent patients. High-capacity models cannot be justified by counting many correlated RF patches as if they were separate people.

### Clinical readiness contract

The gate requires an organ, histological grading system, pathology reference, acquisition protocol, ethics provenance, label definition and complete linked records. Every record must identify patient and lesion, imaging/pathology dates, verified linkage and label provenance. The manifest must declare classes and a minimum independent-patient count per class and role. This minimum is a protocol prerequisite, not a power calculation or a guarantee of adequacy.

A compatible checkpoint metadata object must match organ, grading system and the manifest hash, and reference calibration and validation evidence. Passing those checks establishes only software eligibility for research. It does not certify that the evidence is medically adequate. In the delivered repository, no real clinical dataset or clinically calibrated grade checkpoint is included, so default grade outputs remain null.

### Clinical evaluation

When suitable data is available, compare feature-only ordinal logistic regression, embedding-only prediction and fusion on identical patient partitions. Report confusion matrices, macro-F1, balanced accuracy, class recall, ordinal MAE, quadratic weighted kappa, NLL and Brier score. Include patient-level confidence intervals and representative errors. If a class is absent from a fold, state which metrics are undefined rather than turning them into favorable zeros.

Binary benign/malignant labels support a diagnosis experiment, not histological grading. BI-RADS-like imaging categories and stage labels have their own definitions and must not be renamed as grade. A prospective or external cohort can address transport to new patients and scanners. Phantom validation remains useful upstream engineering evidence but cannot replace this clinical reference standard.

## Uncertainty and selective prediction

### Different uncertainty sources

Measurement quality captures dropout, saturation or decorrelation. Model disagreement reflects sensitivity across independently trained models. A likelihood variance is tied to a particular residual model. Out-of-distribution scoring identifies feature patterns unlike the training reference. Pathology uncertainty concerns the reference diagnosis or grading process itself. Combining these under one unlabeled “confidence” value obscures what was actually measured.

The mechanics pipeline reports residual and cycle mismatch as proxies. A Ledoit-Wolf feature covariance model supplies a simple Mahalanobis OOD score; its threshold is training-derived and should be recalibrated/validated for the intended protocol. The ensemble utility requires at least two fields and does not pretend that one model's repeated deterministic output is an ensemble. Calibration should be tested against reference errors whenever those are available.

### Probability calibration

Temperature scaling fits a scalar on independent calibration logits and labels. It can adjust confidence without changing the argmax class ordering. Reliability diagrams group predictions by confidence and compare average confidence with observed accuracy. NLL and Brier score complement the diagram because binning alone can hide problems or change the visual impression.

Evaluating a temperature on its own calibration subset is a fitting diagnostic, not a final performance result. Keep a separate patient test set. Calibration in one domain does not guarantee calibration after scanner or population shift. A probability with two decimal places is not necessarily accurate to two decimal places; its provenance and empirical validation matter more than its formatting.

### Conformal interpretation

The optional conformal utility selects a finite-sample quantile of group-level nonconformity scores. The score definition must specify the event to cover: for example, a maximum image-level motion error or a patient-level class set. Pixel dependence prevents treating millions of RF pixels as millions of independent calibration patients. The number of independent groups controls quantile granularity.

For very small calibration sets, the requested coverage can require an infinite conservative radius; the utility returns infinity rather than silently selecting an unjustified finite value. Exchangeability is an assumption, not an observed fact guaranteed by calling the function. Under arbitrary domain shift, the ordinary marginal coverage guarantee no longer follows. Any interval figure should state its calibration population and coverage event.

### Selective prediction and decisions

Sort independent cases by a declared uncertainty score, retain increasing fractions and compute mean error among accepted cases. Plot risk against coverage. A model accepting only easy cases may achieve low risk but serve few acquisitions. Report overall rejection rate, reason categories, class/site coverage and errors in both accepted and rejected sets when references exist.

The default threshold policy is illustrative and says whether mechanical estimates appear acceptable under specified proxy limits. The educational Bayesian decision is displayed separately, with assumed CPDs and costs identified. Neither is a clinically calibrated treatment or biopsy policy. A research deployment should learn or justify thresholds on training/calibration data and evaluate their utility on independent data.

## MIS-V mathematical guide

### Course alignment

The supplied course is **23MAT303, Mathematics for Intelligent Systems 5**, Semester 5, L-T-P-C 2-0-2-3. Its three units cover probabilistic graphical models and Bayesian inference, Markov networks and HMMs, and MRFs, decision networks and BN-to-Markov conversion. The project integrates these topics into mechanics quality reasoning and supplies separate small mathematical demonstrations where real-data capabilities are missing.

CO1 concerns modeling complex systems with PGMs; CO2 concerns Bayesian foundations and applications; CO3 concerns directed and undirected models; CO4 concerns real-world graphical-model applications. `docs/mis_v_mapping.md` links every topic and outcome to code, notebooks and exercises. The intent is to explain why a graphical model is appropriate, what independence assumptions it makes and how its numerical inference can be checked.

### Bayesian networks and factorization

A Bayesian network is a directed acyclic graph whose nodes represent variables and whose edges specify parent sets for conditional distributions. The joint distribution factorizes as the product of each variable's CPD given its parents. The educational network uses domain, quality, residual, cycle mismatch, uncertainty and mechanical reliability. The graph expresses assumed dependence, not proven causation.

Quality serves as a shared parent of several evidence variables. Residual and cycle mismatch can therefore be associated marginally even if conditionally independent given quality in this model. Adding every correlated indicator as an independent cause of reliability can double-count evidence. The graph and CPDs should reflect a reasoned model, and alternatives should be tested on held-out likelihood or calibrated reliability prediction where true error labels exist.

### CPDs and supervised learning

Binary CPDs are arrays whose final axis indexes the child's state. Each parent configuration must sum to one. The constructor verifies dimensions, positivity bounds and normalization. Supervised CPD fitting uses counts with smoothing to avoid assigning exact zero probability to an event merely because a small training sample did not contain it. Discretization thresholds also fit only on training observations.

A reliable/unreliable target could be defined by a prespecified physical-error tolerance on reference simulation fields. A low residual alone cannot establish a true reliability label on real RF with unknown displacement. Transporting learned simulation reliability to acquired data is an additional modeling assumption that requires evaluation. The delivered illustrative CPDs are explicitly assumed educational tables, not estimated clinical probabilities.

### Exact inference and factor operations

Posterior inference sums out variables that are neither queried nor observed. The implementation includes factor conditioning, multiplication, marginalization and variable elimination. A separate exhaustive enumeration implementation serves as a transparent check on the small network. Agreement to numerical tolerance is a meaningful software test because the algorithms take different computational routes to the same posterior.

Elimination order affects intermediate factor size. A sparse graph can become expensive when elimination creates large cliques. This motivates concepts such as induced width and junction-tree methods for larger networks. The project does not claim its binary educational implementation is a production-scale general PGM solver. It is intentionally small enough for students to inspect every CPD and compare every posterior calculation.

### Latent representation learning and EM

A latent quality variable can explain several observed indicators. In a Bernoulli mixture, each state has a probability for each binary observation and a mixture weight. The E-step calculates posterior membership; the M-step updates weights and emission probabilities using those memberships. Smoothing prevents exact-zero estimates and improves numerical behavior on small samples.

Latent state identities are not fixed by the likelihood: swapping state indices leaves the model equivalent. Thus component zero cannot automatically mean healthy, stable or reliable. Interpret states through observed behavior and independent references where available. A neural embedding used as input to a Bayesian network is a different form of representation integration; it should not be described as learning the Bayesian graph's latent representation unless that learning actually occurs.

### Markov networks and independencies

An undirected graphical model expresses a joint distribution proportional to a product of nonnegative clique potentials. The normalizing partition function sums that product over all assignments. Potentials need not individually be normalized CPDs. Graph separation encodes conditional independencies under the appropriate model conditions; it is a property of distributions, not simply of a sample correlation matrix.

The executable separation example uses a chain `a-b-c`: conditioning on `b` separates `a` and `c`. Exact factor enumeration computes a partition function on a tiny graph. Such examples let students verify both graph reasoning and numeric probabilities. A Gaussian special case can connect precision-matrix zeros to conditional independence, but that relationship should not be generalized to arbitrary nonlinear data without its assumptions.

### Spatial MRF and conditional Potts model

A Potts energy sums unary label costs and penalties for disagreement between neighbors. Spatial coherence can suppress isolated noise and produce more stable mechanical phenotype regions. When unary costs are derived from observed strain or features, the implemented conditional model is a CRF-style formulation rather than an unconditional generative image model. The repository identifies this distinction explicitly while teaching the shared MRF energy machinery.

ICM updates one label at a time to reduce local energy. It is computationally simple but can become trapped in local minima. Exact enumeration on very small grids supplies a reference check. Energy reduction is not equivalent to better segmentation: high pairwise weight can erase a small true inclusion. Evaluate boundary retention and small-region sensitivity in addition to visual smoothness and aggregate overlap.

### Hidden Markov models

An HMM combines an initial-state distribution, a transition matrix and an emission model. The hidden state at the next time depends on the current state, and each observation depends on the current state. For acquisition quality, illustrative states can be stable, decorrelated and recovering. These are research modeling choices, not disease progression labels.

Filtering uses past and present observations. Smoothing incorporates future observations. Viterbi chooses the most probable complete state path and need not match independent marginal argmax states. The implementation uses log-domain calculations for numerical stability and validates probability normalization. Real HMM analysis requires true sequence ordering; without it, the notebook uses a visibly labeled synthetic sequence. Training resets boundaries rather than treating separate acquisitions as one continuous trajectory.

### Moralization and probability preservation

To moralize a BN, connect every pair of parents sharing a child and remove edge directions. Each CPD becomes a factor over its child and parents. Their product still gives the original joint distribution. The conversion test enumerates every assignment and verifies that the normalized undirected factor product equals the BN joint to numerical tolerance.

This distributional equivalence does not mean the undirected graph expresses every independence of the original directed graph. Some graph-level information is lost. Moralization is also not triangulation: building a junction tree may require additional fill-in edges and an elimination order. Neither operation discovers causal direction from RF measurements. These distinctions are part of the MIS-V explanation rather than optional footnotes.

### Decision networks and expected loss

A decision network adds available actions and a utility or loss function to uncertain states. In this project, actions concern research workflow: accept a mechanics estimate, reacquire, or request expert review. With reliability probability `p`, a decision uses `[1-p,p]` over unreliable/reliable states and multiplies that vector by an action-state loss table. The minimum expected-loss action wins.

For the illustrative table, acceptance has loss 10 when unreliable and zero when reliable; reacquisition costs one in either state; review costs two or 0.5. At `p=0.8`, expected losses are two, one and 0.8, so review is selected. Changing costs can change the action without changing the posterior. Sensitivity plots make those value assumptions visible instead of hiding them inside an unexplained “risk score.”

## Experimental design and statistics

### Primary protocol

Declare one primary metric, its independent unit and a model-selection rule before final testing. For simulation mechanics, endpoint error in physical units is preferred when spacing is verified; pixel units remain explicit otherwise. Use strain error as a secondary measure. For real phantom data without reference fields, use a separately named proxy endpoint and avoid claiming true displacement accuracy.

Keep geometry, acquisition or patient groups intact. A 60/15/10/rest split is only a convenient demonstration, not a universal study design. Small cohorts may need grouped nested cross-validation, with a distinct calibration strategy. Never force a stratified split that breaks patient groups. Record excluded samples, invalid-pixel fractions and the reasons an endpoint could not be computed.

### Metric definitions

Endpoint error is the Euclidean norm of the difference between estimated and reference displacement at valid locations, averaged under a declared aggregation rule. Axial/lateral MAE and RMSE reveal directional error differences hidden by a single EPE. Strain MAE and RMSE compare the declared derivative estimators. If reference strain is differentiated from reference motion, report that provenance.

CNR and regional contrast describe image/mechanics separability, not diagnostic accuracy. Dice and IoU require independent reference masks. Correlation, residual and cycle mismatch are registration-quality proxies. Clinical metrics require eligible pathology labels. Every table should identify units, masks, group counts, model/checkpoint, tuning budget and metric direction. Missing endpoints are unavailable, not zero.

### Bootstrap and paired comparisons

The grouped bootstrap first aggregates repeated measurements to independent-group means, then resamples groups. This avoids falsely narrow intervals from treating thousands of pixels or patches as independent observations. The delivered utility uses equal group weighting and reports that choice. Patient-weighted and lesion-weighted questions are different estimands; choose one deliberately.

Paired comparisons subtract method errors on the same groups before bootstrapping. Training-seed variation is separate from data-sampling uncertainty and should be reported separately. Multiple exploratory comparisons can inflate false discoveries; Holm adjustment is provided for a collection of p-values when relevant. Confidence intervals from four toy groups remain educational; they do not establish a precise clinical effect.

### Ablations and robustness

The configuration generator covers the compact baseline, pyramid candidate, envelope/fusion inputs, cycle removal, smoothness removal, confidence weighting and simulation supervision. Hybrid adaptation is a second-stage experiment requiring a prior simulation checkpoint. Spatial smoothing, quality thresholds versus BN reasoning and rejection versus full coverage are tested through their modules and notebooks.

Stress tests include noise, gain, dropout and missing metadata. Additional frequency and scanner shifts require real metadata and held-out domains. Keep corruption seeds and severity visible. A robust model should be assessed on true reference error, not only a residual objective it was trained to optimize. Failed stress conditions belong in the failure atlas even when they make the overall story less impressive.

### Runtime and memory

Parameter count, device, software versions and measured elapsed time are recorded. Training and inference timing depend on shape, batch size, preprocessing and warm-up. On CUDA, use synchronization around timing intervals and reset peak-memory counters for the measurement being reported. A CPU smoke run on a 48-by-32 fixture cannot support a claim about full-resolution real-time imaging.

The 8 GB profile is a starting configuration, not a demonstrated maximum-memory guarantee. Use acquisition-aware patches, small batches and accumulation; profile representative full samples before planning large runs. Increasing a correlation radius grows the number of cost channels quadratically. Adding attention increases memory much faster with token count. Prefer an evidence-backed compact model over complexity that cannot be tested reliably.

## Software operation

### Environment and dependencies

`pyproject.toml` defines portable dependency ranges; `requirements-lock.txt` records the versions actually used for verification. CPU and GPU Torch wheels differ. Install the correct wheel for the target hardware before installing project extras. The package does not automatically rewrite CUDA configuration or download pretrained checkpoints. `doctor` reports available versions and CUDA availability.

Notebook, TUI and development dependencies are separated into extras. Windows paths are handled with `pathlib`; no author-specific data path is needed. Large jobs belong in the CLI so a notebook smoke run stays bounded. Always use a new output folder for a new scientific run, and retain the configuration with its artifacts. The test environment's full package list is not presented as a universally portable operating-system lock.

### CLI reference

| Command | Purpose | Required prerequisite |
| --- | --- | --- |
| `doctor` | Environment report | Installed package |
| `data instructions` | Official access guidance | Dataset name |
| `data inspect` | Keys, shapes, checksums | Existing local folder |
| `data download` | Authorized file transfer | Verified HTTPS URL |
| `data extract` | Safe archive extraction | Complete archive/volumes |
| `data prepare` | Canonical pair conversion | Verified explicit mapping |
| `data split` | Grouped role assignment | Verified independent IDs |
| `data audit` | Leakage checks | Split manifest |
| `demo` | Offline mechanics example | No external data |
| `train` | Model optimization | Demo config or real manifest |
| `evaluate` | Held-out mechanics metrics | Run manifest and checkpoint |
| `infer` | Mechanics outputs | Canonical pair |
| `phenotype fit` | Training-only mixtures | Feature matrix and roles |
| `pgm demo` | Executable mathematics | Topic selection |
| `risk analyze` | Quality and decision evidence | Inference summary |
| `report build` | Local HTML/Markdown report | Completed run |
| `notebooks execute` | Clean-kernel notebook demos | Notebook extras |
| `clinical readiness` | Clinical prerequisite audit | Clinical manifest |
| `tui` | Interactive terminal app | TUI extra |

Commands print machine-readable JSON for summaries and nonzero exit codes for invalid operations. `train --dry-run` shows the requested configuration without starting work. Use `--help` on each subcommand for arguments. Missing data produces an explicit error; no command silently switches a real experiment to synthetic data.

### Real-data sequence

After downloading and verifying the data, inspect it, fill the appropriate adapter config, prepare canonical pairs, generate grouped splits, audit them and start training. The acquisition config and split config are separate because source layout and scientific grouping answer different questions. Use an acquisition group key for real phantom sessions and geometry group key for simulation generalization, as supported by verified metadata.

```bash
python -m elastograde data prepare --config configs/datasets/simulation.yaml
python -m elastograde data split --config configs/experiments/splits.yaml
python -m elastograde data audit --manifest artifacts/manifest.json
python -m elastograde train --config configs/gpu_8gb.yaml --dry-run
python -m elastograde train --config configs/gpu_8gb.yaml
```

These commands require the edited mappings and manifests. They do not imply the distributed configs are ready to run against an unseen archive without inspection. Retain the verified source README and field map in the study's data card.

### Terminal UI walkthrough

The dashboard displays research mode and provides an offline demo button. The Data tab inventories a local folder. Experiment accepts a configuration path, previews it and starts training. Training exposes a stop request and optional resume checkpoint. Inference accepts a canonical pair and an optional trusted checkpoint. Reasoning lets the user inspect how reliability and false-acceptance cost affect the expected-loss decision. Results reads run summaries and builds reports. Help documents keyboard navigation and scientific limits.

Work occurs in a background worker so navigation remains responsive. Stop is cooperative and completes the current epoch before ending. Logs explicitly show failed operations rather than replacing them with default predictions. Detailed maps are saved as PNG and HTML because terminal image support varies. Keyboard navigation and text statuses do not depend only on color.

### Configurations and output artifacts

The demo config uses synthetic fixtures, CPU, a small model and a bounded epoch count. The CPU profile increases epochs while retaining a modest workload. The GPU profile selects CUDA and a real split manifest, with accumulation and optional mixed precision. Model fields control input representation, width, correlation radius, confidence and attention. Loss fields control each objective component.

Training writes configuration, split manifest, JSONL history, `last.pt`, `best.pt` and a status summary. Inference writes `maps.npz`, `summary.json` and a visual panel. Reporting writes HTML, Markdown and a hash manifest. The arrays retain raw numeric values and validity masks; summaries identify units and data origin. A missing grade is represented by null with a reason, never by a default low grade.

### Tests and continuous integration

Tests cover identity and subpixel warps, analytic compression strain, vector flipping, composition, spacing requirements, gradients, masks, grouped leakage, source corruption, archive safety, explicit adapters, calibration roles, BN inference, moralization, HMM ordering, MRF energy, decision arithmetic, ordinal probabilities, checkpoint resume, CLI operations and TUI mounting. Notebook validation checks all 19 files and clean-kernel execution supplies a separate integration result.

The CI workflow installs a CPU environment, runs tests and executes notebook demos. Local evidence is stored in `reports/verification/` and executed notebooks. Tests use known analytic expectations rather than arbitrary excellent real-data scores. A passing numerical test verifies the corresponding operation; it does not prove that a trained model has learned medically useful behavior.

### Troubleshooting

| Symptom | Likely cause | Next step |
| --- | --- | --- |
| `No module named elastograde` | Wrong environment or not installed | Activate environment; install editable package |
| CLI script absent | Scripts directory outside PATH | Use `python -m elastograde` |
| CUDA unavailable | CPU wheel or incompatible machine | Use CPU profile or correct official GPU wheel |
| MAT mapping fails | Unverified key/shape | Run inventory and inspect source README |
| Checksum mismatch | Source changed after conversion | Revalidate and regenerate canonical pair |
| No valid support | Large motion, masks or small image | Inspect search bounds and masks |
| Negative/implausible strain | Sign/axis mistake or noisy derivative | Run analytic fixture tests; inspect units |
| Very smooth but incorrect maps | Excessive regularization | Compare boundary error and ablations |
| Notebook kernel fails | Missing kernel dependencies | Install notebook extras in active interpreter |
| Clinical readiness blocked | Missing pathology evidence | Supply the required verified clinical manifest |
| Resume rejected | Critical configuration changed | Start a new run or restore original config |
| Multipart extraction fails | Missing volume or wrong count | Obtain all volumes and verified total |

## Risk, limitations and validation

### Risk register philosophy

A useful risk analysis connects a hazard to a mechanism, observable evidence, mitigation and remaining limitation. Wrong axis metadata can reverse or rescale mechanics despite apparently plausible images. Shared patients or source frames can inflate evaluation. Decorrelation can produce confident-looking false correspondences. Aggressive smoothing can erase small inclusions. Calibration can fail under distribution shift. Missing stress information makes absolute modulus underdetermined.

The full register lists these hazards and the specific tests or audits that address them. Detection does not eliminate risk: a low residual cannot reveal every wrong correspondence, and a lineage audit cannot prove identities that were never recorded. Hand-assigned FMEA scores would only prioritize review; they are not empirical harm probabilities. This project uses explicit expected-loss examples instead of presenting arbitrary scores as calibrated clinical risk.

### Observed failure atlas

The stress notebook records clean, noisy and dropout examples and compares their residuals and actions. Keep every example's seed, condition, reference, metric mask and rejection reason. Review false acceptance and false rejection separately. A threshold tuned to catch a dramatic corruption may still miss subtle phase ambiguity. A model that rejects all difficult cases can hide poor generalization behind a small accepted subset.

Future real-data failure categories should include compression range, frequency, acquisition site, region depth, out-of-plane motion evidence and reference-label uncertainty. Do not infer sensitive patient attributes from images merely to populate a fairness table. Evaluate relevant recorded subgroups only when the data and study approvals support them, and acknowledge when the dataset cannot answer a subgroup question.

### What remains externally dependent

The source archives, original README files, verified per-file mappings and independent acquisition identifiers have not been supplied in this package. Consequently, the adapters are explicit and testable on known fixtures, but the exact IMPACT archive-to-canonical mapping remains a data-access task. The package does not contain a real-data trained model, a published benchmark reproduction or a clinical grade classifier.

True real-sequence ordering must be established before applying the HMM to acquired data. Absolute modulus estimation requires additional mechanical information. Clinical grading requires a separately acquired pathology-linked cohort and external validation. These are scientific dependencies; substituting invented metadata or labels would make the repository less complete as research, even if it made every button appear enabled.

### Milestones and exit criteria

First, verify the environment and run the offline demo, tests and all notebooks. Second, inventory real archives and approve axes, units, reference convention and lineage. Third, reproduce classical baseline metrics on independent geometry/acquisition groups. Fourth, train the compact models and compare pure self-supervision, simulation supervision and adaptation under a frozen protocol. Fifth, evaluate uncertainty and rejection against reference error and document failure cases.

Only after those stages should the project pursue a clinical grading study. Define organ and grade, secure appropriate data governance, verify lesion/pathology linkage, plan independent-patient sample sizes and reserve external evaluation. The final milestone is a reproducible result whose scientific claim matches its evidence. A large repository or an elaborate UI is useful engineering work but is not itself the evidence for state-of-the-art tumor grading.

## Learning path and notebook guide

Run notebooks in numeric order for a full course, or open individual files after installation because each bounded demo is self-contained. The parameter cell records mode, seed, data root, configuration path, sample limit, device and output directory. Each notebook has theory, an experiment, interpretation, failure cases, an exercise and an evidence statement. Executed copies are placed in `reports/executed_notebooks` so the source notebooks remain clean.

The first seven notebooks establish claims, environment, data inventory, RF physics, split design, classical tracking and strain. Notebooks 07–10 address learning, adaptation, phenotype representations and uncertainty. Notebooks 11–14 cover the MIS-V mathematical core. Notebooks 15–18 address clinical readiness, comparative evaluation, risk and the complete CLI/report path. None requires an unverified real archive to execute its educational demonstration.


## FAQ

**1. Can this repository grade a real tumor immediately?** No. It has motion, mechanics and ordinal research components, but it does not include a clinically validated pathology-grade model. The default inference result deliberately has a null grade. Supplying a real RF pair does not change that evidence boundary.

**2. Why not assign grades using stiffness thresholds?** A mechanical measurement and a histological grade describe different properties. Their association can depend on organ, tumor type, loading, fibrosis, necrosis and acquisition. A threshold requires an appropriate labeled clinical study. An intuitive ordering is not a validated grading rule.

**3. Why are the phantom datasets still valuable?** They allow controlled investigation of motion and mechanics, including known simulated displacement. Such upstream verification can uncover coordinate errors, quantify domain shifts and compare estimators. It is a strong engineering foundation even though it cannot establish patient-level grading accuracy.

**4. What does the synthetic demo prove?** It proves the listed software path executes and can be checked against known toy motion. The fixture is band-limited texture transformed by an analytic field. It does not reproduce acoustic scattering, finite-element tissue mechanics or clinical disease diversity.

**5. Why can a good displacement image have poor strain?** Strain differentiates displacement. Small alternating errors can have a large derivative even when mean displacement error is small. Inspect strain-specific metrics, regularization bias and boundary fidelity, not only displacement appearance.

**6. Why cannot every RF pair count as a separate patient?** Multiple pairs may come from the same phantom region, acquisition or patient and share substantial information. Their errors are correlated. Counting them independently exaggerates sample size and can make confidence intervals unrealistically narrow.

**7. Does unsupervised learning avoid leakage?** No. Fitting on final-test images can make the experiment transductive. Selecting hyperparameters on dependent validation frames can also inflate performance. The split protocol applies to unlabeled representation learning as well as supervised models.

**8. What is the difference between simulation-supervised and hybrid?** Simulation-supervised training uses reference motion targets. Hybrid training transfers that model and adapts it using another objective on acquired data. The adaptation step does not erase the label information used during pretraining.

**9. Why does the HMM require sequences?** Its transition model describes temporal dependence. Unrelated pairs have no justified transition relationship. Sorting filenames creates an order but not a physical time process. The course demonstration therefore uses a declared synthetic sequence until real ordering is verified.

**10. Are the Bayesian reliability probabilities calibrated?** The default CPDs are educational assumptions. Their posteriors are mathematically valid under that assumed model but are not empirical clinical probabilities. Real reliability modeling needs reference-error targets, training-only estimation and independent calibration checks.

**11. Why is moralization not causal discovery?** Moralization transforms a supplied graph into an undirected factor representation. It does not infer the original graph or establish causal effects. The numeric test verifies preservation of the supplied joint distribution, not the truth of its causal interpretation.

**12. What should I change for an 8 GB GPU?** Start with the provided profile, verify representative array sizes and use a modest patch size, batch size and accumulation. Measure actual peak memory. Do not assume a configuration tested on tiny CPU fixtures will fit every full RF acquisition.

**13. Can I train using B-mode alone?** An envelope/image baseline can be useful if labeled correctly, but it loses signed RF information and answers a different input-modality question. A screenshot with display annotations or a color map is not equivalent to the underlying RF pair.

**14. Why are some data configs intentionally incomplete?** The actual archive field names and axes were not available for verification. A clear required mapping is safer and more scientifically honest than a loader that guesses plausible keys and silently produces wrong tensors. The inventory utility supplies the information needed to fill it.

**15. How do I resume training?** Use `train --config YOUR_CONFIG --resume reports/YOUR_RUN/last.pt` with compatible model, regime, loss and seed settings. Increase the total epoch count deliberately. Starting a changed model is a new experiment, not a resume of the same trajectory.

**16. Why can correlation be high but the motion wrong?** RF signals can contain repeated phase structure; several offsets may align similar oscillations. Texture ambiguity and decorrelation can also create false matches. Bidirectional consistency and uncertainty help diagnose failures but do not guarantee uniqueness.

**17. What does an OOD rejection mean?** It means the chosen feature model considers an observation unusual relative to its training reference. It does not diagnose a rare tumor or prove the input is invalid. OOD scores require validation on relevant shifts and can miss unfamiliar examples.

**18. Can uncertainty guarantee safety?** No. An uncertainty method can itself be miscalibrated or insensitive to a new failure mechanism. Evaluate how well it predicts observed errors, especially confident failures, and report the accepted coverage. A conservative-looking visualization is not a guarantee.

**19. Why is absolute modulus unavailable?** The default compression data do not supply all loading information required to identify it. Inferring a kPa value from strain alone would hide an inverse-problem ambiguity. A future module must specify its forward model, boundary evidence and scale assumptions.

**20. What is needed for a state-of-the-art claim?** A current related-work review, a comparable benchmark, verified independent splits, strong and fairly tuned baselines, repeated experiments, uncertainty intervals, compute reporting and reproducible artifacts. Passing the included tests or producing attractive figures is not sufficient.

**21. Do I need to run all notebooks in order?** Numeric order is the recommended learning path, but each bounded demo is designed to be self-contained after installation. Outputs from one notebook are not silently required by another. Full real-data experiments remain explicit CLI jobs with recorded manifests.

**22. Can I use all available data for calibration?** No. Calibration parameters fitted on the final test set make subsequent performance optimistic. Use a declared calibration partition or a carefully designed grouped cross-validation strategy. Calibration must respect the same independent unit as the prediction target.

**23. Why are cluster names neutral?** Unsupervised component indices have no intrinsic medical meaning and can permute between fits. Calling a component high grade would invent a semantic mapping. Interpret mechanical clusters using independent evidence while retaining descriptive labels until a validated association exists.

**24. How should I present this in a viva?** Start with the scientific scope, demonstrate a known-motion fixture, show the coordinate and split checks, explain one learning objective, then work through the BN and decision example. State precisely which results use synthetic, simulated, acquired phantom or clinical data.

## Glossary and symbols

| Symbol or term | Meaning and caution |
| --- | --- |
| RF | Signed radio-frequency echo samples |
| IQ | Complex baseband signal representation |
| B-mode | Log-compressed display image, not raw RF |
| `z`, `x` | Axial and lateral coordinates |
| `u=[ux,uz]` | Reference-to-moving sampling displacement |
| `dz`, `dx` | Verified physical spacing along each axis |
| `epsilon_zz` | Axial small strain, dimensionless |
| `epsilon_xz` | Tensor shear; engineering shear is twice this |
| NCC | Normalized cross-correlation |
| EPE | Endpoint displacement error |
| CNR | Declared regional contrast-to-noise ratio |
| CPD | Conditional probability distribution |
| BN | Bayesian network, directed acyclic model |
| MRF | Markov random field, undirected model |
| CRF | Conditional random field with observed-input conditioning |
| HMM | Hidden Markov model for ordered sequences |
| MAP | Most probable assignment under the stated model |
| OOD | Out of distribution relative to a specified reference |
| Coverage | Fraction of cases retained by a selective predictor |
| Calibration | Agreement between predicted probability and frequency under a defined population |
| Provenance | Traceable source, processing and model history |
| Geometry group | Common simulated structure whose descendants remain together |
| Histological grade | Pathology-defined category under an organ-specific system |
| Phenotype cluster | Descriptive statistical grouping without automatic clinical semantics |

## References and contribution guide

Official source pages are linked in the data section. The principal research anchors are Tehrani and Rivaz, *Displacement Estimation in Ultrasound Elastography using Pyramidal Convolutional Neural Network*; Tehrani, Mirzaei and Rivaz, *Semi-Supervised Training of Optical Flow Convolutional Neural Networks in Ultrasound Elastography*, MICCAI 2020; Tehrani and colleagues, *Bi-Directional Semi-Supervised Training of Convolutional Neural Networks for Ultrasound Elastography Displacement Estimation*, 2022; and Mirzaei, Asif and Rivaz, *Combining Total Variation Regularization with Window-Based Time Delay Estimation in Ultrasound Elastography*, 2019. Verify exact publisher metadata before a final manuscript bibliography. The included BibTeX avoids inventing unverified DOI strings.

PyTorch's official `grid_sample` documentation defines normalized grid coordinates, interpolation and `align_corners` behavior. The implementation uses the half-pixel convention for `align_corners=False`, and numerical tests compare it with a separate NumPy/SciPy sampler. Software versions are recorded in run metadata because future API changes can alter installation or execution behavior.

Contributions should add a focused numerical test, preserve declared input contracts and explain their scientific purpose. A new loss must name its assumptions and failure cases. A new dataset adapter must include verified source mappings and a data card. A new performance claim must reference an actual run and independent evaluation. Do not add hard-coded benchmark scores, private data, generated pathology labels masquerading as real labels, or external code without its license and attribution.

For orientation, reusable algorithms live under `src/elastograde`; configurations live under `configs`; the 19 teaching notebooks live under `notebooks`; tests live under `tests`; supplementary protocols and cards live under `docs`; and generated evidence lives under `reports`. The original generation specification is retained as `GENERATION_SPECIFICATION.md` so future work can compare implementation against the requested scope without relying on conversational memory.

## Notebook file index

- [00_Project_Map_and_Claims.ipynb](notebooks/00_Project_Map_and_Claims.ipynb)
- [01_Environment_and_Reproducibility.ipynb](notebooks/01_Environment_and_Reproducibility.ipynb)
- [02_Dataset_Acquisition_and_Inventory.ipynb](notebooks/02_Dataset_Acquisition_and_Inventory.ipynb)
- [03_RF_Physics_and_Preprocessing.ipynb](notebooks/03_RF_Physics_and_Preprocessing.ipynb)
- [04_Grouped_Splits_and_Leakage_Audit.ipynb](notebooks/04_Grouped_Splits_and_Leakage_Audit.ipynb)
- [05_Classical_Displacement_Baselines.ipynb](notebooks/05_Classical_Displacement_Baselines.ipynb)
- [06_Strain_and_Mechanical_Features.ipynb](notebooks/06_Strain_and_Mechanical_Features.ipynb)
- [07_Self_Supervised_Learning.ipynb](notebooks/07_Self_Supervised_Learning.ipynb)
- [08_Simulation_Supervision_and_Adaptation.ipynb](notebooks/08_Simulation_Supervision_and_Adaptation.ipynb)
- [09_Mechanical_Representations_and_Clustering.ipynb](notebooks/09_Mechanical_Representations_and_Clustering.ipynb)
- [10_Uncertainty_and_Selective_Prediction.ipynb](notebooks/10_Uncertainty_and_Selective_Prediction.ipynb)
- [11_MISV_Bayesian_Networks.ipynb](notebooks/11_MISV_Bayesian_Networks.ipynb)
- [12_MISV_Markov_Networks_and_MRF.ipynb](notebooks/12_MISV_Markov_Networks_and_MRF.ipynb)
- [13_MISV_Hidden_Markov_Models.ipynb](notebooks/13_MISV_Hidden_Markov_Models.ipynb)
- [14_MISV_Moralization_and_Decision_Networks.ipynb](notebooks/14_MISV_Moralization_and_Decision_Networks.ipynb)
- [15_Clinical_Ordinal_Grading_Readiness.ipynb](notebooks/15_Clinical_Ordinal_Grading_Readiness.ipynb)
- [16_Benchmarks_Ablations_and_Robustness.ipynb](notebooks/16_Benchmarks_Ablations_and_Robustness.ipynb)
- [17_Risk_Analysis_and_Failure_Atlas.ipynb](notebooks/17_Risk_Analysis_and_Failure_Atlas.ipynb)
- [18_End_to_End_CLI_and_Research_Report.ipynb](notebooks/18_End_to_End_CLI_and_Research_Report.ipynb)

See [architecture diagrams](docs/architecture.md), [requirement coverage](docs/requirements_coverage.md), [MIS-V traceability](docs/mis_v_mapping.md), and [risk register](docs/risk_register.md).


## Execution environment note

The build environment prohibits local socket binding, so normal Jupyter kernel startup was unavailable here. All notebook cells are additionally verified in a fresh Python process per notebook using IPython capture, with embedded figures and outputs. This backend is explicitly recorded as `process` in the execution summary; it is not represented as a successful Jupyter-kernel run. On a normal workstation, `python scripts/execute_notebooks.py` uses the standard Jupyter backend. To select the socket-free fallback, set `ELASTOGRADE_NOTEBOOK_BACKEND=process` for that command. Both routes validate nbformat and fail on cell errors.

Training uses the configured shape as a crop budget: random training crops and deterministic central validation crops retain parent-group identities. This makes memory use more predictable but changes the evaluated region. Use whole-image or tiled held-out inference for a full-image benchmark and report any difference in support. All stored motion remains referenced to the corresponding cropped pair, with crop origin recorded.
