# Master generation prompt: Ultrasound Elastography for Label-Free Tissue Tumor Grading

Prepared 21 September 2026. Paste the complete prompt below into a coding agent with filesystem access. This deliverable is a generation specification, not an already trained model or completed repository.

## Read this before using the prompt

The two requested Concordia IMPACT datasets support elastography development and phantom validation. Their published descriptions do not establish the availability of histopathology-linked tumor grades. The scientifically defensible core is label-free displacement learning and uncertainty-aware mechanical tissue characterization; clinical grading is a conditional extension. “Label-free” must specify whether it means no displacement labels during training, no pathology labels during representation learning, or no pathology labels at inference. These are different claims.

Verified source anchors:

- [IMPACT acquired phantom dataset](https://users.encs.concordia.ca/~impact/ultrasound-elastography-dataset-for-unsupervised-training/): 2,200 RF pairs from a CIRS 059 phantom; 10 MHz probe center frequency and 40 MHz sampling. Its multipart archive must be assembled by extracting the first volume with all volumes present.
- [IMPACT simulation dataset](https://users.encs.concordia.ca/~impact/ultrasound-elastography-simulation-database/): 24 models at 5 MHz, with additional 10 MHz data for 19 models; axial and lateral displacement references. Verify the archive's actual hierarchy and counts before constructing samples.
- [Tehrani, Mirzaei and Rivaz, MICCAI 2020](https://arxiv.org/abs/2007.01421): source for simulation-assisted learning and unsupervised adaptation in ultrasound displacement estimation.

Do not confuse mechanical inclusion classes, malignancy classification, imaging scores, tumor stage, and histopathological grade. This distinction is a design requirement throughout this prompt.

---

# BEGIN PROMPT TO THE CODING AGENT

## 1. Mission and working standard

Build a complete, executable, carefully documented research software repository for:

**Ultrasound Elastography for Label-Free Tissue Tumor Grading**

Use the working software name **ElastoGrade Research**. Add a visible subtitle:

**Physics-guided label-free elastography, probabilistic tissue characterization, and a clinically gated grading extension.**

Act as a research engineer with expertise in ultrasound RF processing, inverse problems, self-supervised motion estimation, probabilistic graphical models, experimental design, Python packaging, Jupyter notebooks and terminal applications.

Create actual files. Do not merely describe what someone else should create. Implement a complete small-data path first, then extend it to configurable full experiments. Prioritize correctness, runnable code, reproducibility and informative failure messages over an oversized collection of untested models.

The deliverable must contain:

1. An installable Python package with reusable research modules.
2. A substantial set of valid, educational `.ipynb` notebooks.
3. A scriptable command-line interface and an interactive terminal UI.
4. A comprehensive `README.md` that serves as an ultimate theory and implementation guide.
5. Dataset adapters, manifests, grouped splits and dataset cards.
6. Classical and learning-based baselines with a meaningful ablation framework.
7. Uncertainty estimation, explicit out-of-distribution checks and abstention.
8. Integrated MIS-V probabilistic graphical-model demonstrations.
9. Automated numerical, data-leakage, CLI and notebook smoke tests.
10. Publication-oriented plots, machine-readable metrics and report generation.
11. A genuinely executable offline demonstration that uses clearly identified synthetic fixtures.
12. A clinical extension that is implemented and tested using synthetic fixtures but cannot issue real clinical claims without suitable real data and validation.

Never promise state-of-the-art performance, novelty, clinical validity or publication acceptance before evidence exists. Set a research ambition and evaluate it against reproducible baselines. Label the proposed architecture as a candidate. Do not invent benchmark results, citations, filenames inside downloaded archives, or trained checkpoints.

Work autonomously on routine engineering choices. Do not repeatedly ask whether to continue. If data access is blocked, finish all code that can be built and verified, document the exact missing dependency and provide a manual acquisition route. Never bypass access controls or accept legal terms on the user's behalf.

## 2. Scientific scope and capability gates

Implement three explicit operating modes.

### Mode A: mechanical characterization, default

Available using the two IMPACT datasets, subject to archive verification. Outputs may include displacement, strain, quality measures, mechanical feature vectors, descriptive mechanical phenotype clusters, and uncertainty. Do not output pathology grades or malignancy probabilities in this mode.

### Mode B: simulation-supervised mechanics, explicitly labeled

Use verified simulation ground truth to train or calibrate displacement or strain models. Label this training accurately as simulation-supervised or hybrid; it is not fully label-free. Maintain an independent purely self-supervised training arm for fair comparison.

### Mode C: clinical grading research, conditional

Require a separately verified dataset with an explicitly identified organ, histological grading system, tissue diagnosis, patient and lesion linkage, acquisition information and pathology reference. Learn the mapping from representations to grades using training data, calibrate on separate groups, and evaluate on held-out patients. The whole pipeline then has supervised components even if its encoder is label-free.

Clinical mode must be disabled by default. A boolean flag alone must not enable it: validate a capability manifest and checkpoint metadata. Missing labels, invalid grading-system metadata, or incompatible calibration must produce a structured refusal to generate a grade, with a useful explanation.

Unsupervised clusters must retain neutral names such as `mechanical_cluster_0`. Do not call them grade I/II/III, benign/malignant, or low/high cancer risk. Mechanical stiffness is not a universal monotonic proxy for tumor grade.

Define a claims ledger containing each claim, required evidence, available evidence, validation status and relevant run IDs. Keep four distinctions explicit: implemented; executed; empirically supported; clinically validated. Passing tests establishes only the relevant software behavior.

## 3. Research question and candidate contribution

Primary question:

Can acquisition-aware, confidence-weighted self-supervised displacement estimation improve strain fidelity and selective reliability under phantom-domain shift, while retaining interpretable mechanical features?

Secondary questions:

- Does adding simulation supervision improve or weaken generalization to acquired phantom data?
- Do RF and envelope representations contribute differently under increasing decorrelation?
- Does uncertainty identify displacement/strain failures better than raw reconstruction residual alone?
- Can an MRF improve spatial consistency without erasing small stiff inclusions?
- Can a Bayesian network combine correlated quality evidence more usefully than a simple threshold baseline?
- Does an HMM stabilize acquisition-quality assessment when true ordered sequences exist?
- When pathology-linked data exists, do mechanical representations improve ordinal grading over simpler image and feature baselines?

Candidate contribution: a compact multi-scale displacement network with acquisition-aware inputs, bidirectional consistency, mechanically motivated regularization and a probabilistic quality/decision layer. This is a hypothesis, not a claim of new mathematics. Produce a related-work comparison before claiming novelty.

Predefine primary endpoints, split units and model-selection criteria before final testing. Select one primary mechanics endpoint for simulation and one explicitly qualified real-phantom quality endpoint. Real reconstruction quality is a proxy unless reference displacement is available.

## 4. Dataset acquisition and verification

Use these official landing pages as the starting points:

- `https://users.encs.concordia.ca/~impact/ultrasound-elastography-dataset-for-unsupervised-training/`
- `https://users.encs.concordia.ca/~impact/ultrasound-elastography-simulation-database/`

Recheck the pages and linked READMEs during implementation. Record retrieval date, links, published citation requests, access conditions, redistribution permissions and any unknown license status. Free download does not automatically permit redistribution. Keep original data outside the repository and do not bundle restricted data.

The acquired dataset is described as RF pairs from a physical phantom. The simulation source supplies motion references. Neither description establishes pathology-grade labels. Record these as distinct domains.

Implement an acquisition helper that:

- prints exact verified acquisition instructions;
- downloads only where an accessible authorized link and terms permit;
- resumes downloads and validates sizes/checksums when supplied;
- detects HTML error/login pages masquerading as archive files;
- checks disk space and all multipart volumes before extraction;
- extracts safely without path traversal or overwriting unrelated files;
- supports manual data placement without network access;
- records archive checksums and source provenance;
- reports missing volumes clearly;
- never substitutes generated data for missing real data without an explicit demo mode.

The acquired dataset page specifies extracting `Data_121706.zip.001` with every archive volume present. Do not extract each volume independently. Discover actual simulation archive names instead of guessing them.

Inventory archives before writing final adapters. Inspect small representative files and enumerate keys, nested structures, shapes, dtypes, coordinate conventions and units. Handle MATLAB legacy MAT and v7.3 HDF5 when encountered. Do not indiscriminately squeeze dimensions, transpose arrays or infer axial orientation solely from array aspect ratio.

Write `data_inventory.json`, `dataset_capabilities.json` and a human-readable acquisition report. Unknown metadata must remain unknown. Distinguish published metadata from per-file verified metadata.

## 5. Canonical schema and provenance

Design a versioned canonical sample schema including:

- sample, dataset and domain IDs;
- source relative path and SHA-256 checksum;
- acquisition, sequence, frame, pair, physical-phantom, region and geometry IDs when known;
- scatterer realization and simulation condition IDs when known;
- patient, lesion, study, site and scanner IDs for future clinical data;
- pre/post RF arrays or lazy references;
- optional IQ and envelope representations with provenance;
- frame ordering and timestamps when present;
- axial/lateral dimensions and coordinate orientation;
- sample rate, center frequency, axial/lateral spacing and units;
- compression metadata and loading information when available;
- displacement reference, validity mask and its coordinate frame;
- optional strain, region masks and their provenance;
- pathology labels and grading-system provenance only when supplied;
- quality flags, exclusion reasons and split assignment.

Use `null` for unavailable fields. Distinguish `not_available`, `not_applicable` and `not_verified` in capability metadata. Do not invent patient IDs for phantoms. A stable sample ID is not a verified independent acquisition ID.

Prefer chunked local storage suited to measured data access, such as HDF5 or Zarr with a pinned compatible version. Preserve raw data unmodified. Avoid loading the whole dataset into RAM. Make cache keys depend on source checksum, preprocessing configuration and code version.

## 6. Splitting and leakage prevention

Split at the highest independent identifiable unit before generating patches, augmentations, frame pairs or cached derivatives.

For simulations, group all strain levels and scatterer realizations of a geometry into a common geometry group for the principal generalization experiment. Link corresponding geometry families across frequencies when identifiable. A frequency-shift experiment using a previously seen geometry answers a different question; report it separately.

For acquired phantom data, group by verified acquisition/session/sequence and region where possible. If independent groups cannot be reconstructed, do not claim independent clinical or phantom generalization. Similarity analysis can flag duplicates, but cannot prove acquisition independence. Explicitly qualify all within-phantom evaluations.

For clinical data, keep every lesion, session, frame and augmentation of one patient in a single outer partition. Support a held-out site when feasible.

Use separate training, validation, uncertainty-calibration and test roles. With small samples, use grouped nested cross-validation or a documented split that preserves usable independent groups. Never force class stratification when it breaks groups or creates invalid folds.

Any preprocessing fit, normalization population statistic, PCA, feature selection, representation model, clustering model, pseudo-label rule, calibration parameter, threshold and grade mapping must use only its permitted training/calibration partition. If unsupervised training uses test covariates, call the experiment transductive and do not present it as the primary inductive result.

Generate an overlap audit based on IDs, source hashes, shared source frames and derivative lineage. Fail execution on known leakage. Include tests demonstrating that two pairs sharing a source frame cannot cross splits under the selected protocol.

## 7. Ultrasound preprocessing and coordinate conventions

Teach and implement the distinction among signed RF, complex IQ, envelope amplitude and log-compressed B-mode. Never use a screenshot or color elastogram as if it were RF.

Define coordinates explicitly, for example axial `z`, lateral `x`, array `[z,x]`, displacement channels `[u_x,u_z]` with a documented conversion between array order and grid-sampling order. State forward versus backward warping and reference frame conventions once and enforce them everywhere.

If metadata supports conversion, derive axial sample spacing as `c/(2 f_s)` for pulse-echo time samples; account for decimation, scan conversion and any resampling. The assumed sound speed and applicable acquisition model must be recorded. Lateral spacing comes from verified acquisition metadata, not axial sample spacing.

Implement:

- robust finite-value and saturation checks;
- optional physically justified DC removal and bandpass filtering;
- Hilbert-envelope extraction along the verified axial dimension for RF;
- log compression for visualization, with fixed documented dynamic range;
- pair-consistent normalization preserving relevant RF phase and relative information;
- signal quality and decorrelation summaries;
- valid-region masks for padding, warping and signal dropout;
- transformations that update coordinates, displacement vectors, masks and spacing consistently;
- metadata-aware resampling with anti-aliasing and motion-unit conversion;
- configurable patch extraction with acquisition-aware patch dimensions.

Do not resize RF like a natural image without examining aliasing and carrier effects. Avoid unrestricted rotations or axis swaps that invalidate axial/lateral semantics. Flips must transform vector components and physical metadata. Keep amplitude/noise augmentations plausible and explicitly parameterized.

For displacement ground truth relative to the first frame, derive new pair targets with coordinate-frame awareness. Simple subtraction is valid only under the relevant shared-reference convention; a target defined in a different frame may require composition/regridding. Verify the published linear-motion assumption and archive conventions before pair synthesis.

## 8. Offline synthetic fixtures

Provide tiny deterministic fixtures for software verification, never represented as IMPACT data or validated ultrasound simulation.

Include zero motion, known subpixel translation, analytic affine compression, smooth heterogeneous displacement, dropout, corrupted metadata and missing-spacing examples. A band-limited speckle-like image generator with analytic warping is sufficient for tests, but must be labeled a toy RF-like fixture rather than a Field II or FEM substitute.

Separate any optional physically richer FEM/acoustic simulator from the default demo. Document dependencies and licensing. Never claim that a simple Gaussian texture warp reproduces ultrasound scattering physics.

Every demo figure, report and metric must carry `data_origin: synthetic_fixture`. Synthetic grade labels can verify software plumbing only, not diagnostic capability or clinical calibration.

## 9. Baselines and experimental fairness

Implement a hierarchy with equal data and split access:

1. Zero-motion sanity baseline.
2. Windowed normalized cross-correlation with subpixel refinement and confidence.
3. A regularized classical displacement refinement, clearly identified as your implementation unless reproducing an exact published algorithm.
4. A compact U-Net-like displacement regressor trained with self-supervised registration losses.
5. A compact pyramidal cost-volume model as the principal candidate.
6. The same candidate with simulation supervision only.
7. The same candidate with simulation pretraining plus real-phantom self-supervised adaptation.
8. Optional externally maintained optical-flow baseline only after license, dependency and RF suitability checks.

Do not call an approximate baseline GLUE, OVERWIND, RAFT or another named method without reproducing its relevant implementation or using verified author code. Clearly distinguish inspired implementation from faithful reproduction.

Match preprocessing and evaluation masks. Report tuning budgets and compute. Include a baseline using quality thresholds alone to test whether the Bayesian quality layer adds value. Do not compare a heavily tuned candidate against deliberately weak defaults.

## 10. Candidate architecture

Implement a compact network rather than a collection of unrelated foundation models.

Inputs: normalized pre/post RF pairs, optionally envelope pairs, validity masks and verified acquisition metadata. Missing metadata uses explicit missingness indicators, not invented values.

Core components:

- a shared multi-scale encoder;
- anisotropic receptive fields respecting axial/lateral sampling;
- local correlation/cost volumes with bounded displacement search;
- coarse-to-fine residual displacement refinement;
- optional lightweight cross-feature attention at low resolution;
- forward and reverse displacement estimation using shared weights;
- confidence/variance outputs only when trained with a valid objective;
- full-resolution displacement and a separate derivative-based strain stage.

Make RF-only, envelope-only and fused variants configuration options. Explain why each is included. Attention is an ablation, not a mandatory novelty claim. Avoid full-resolution global attention that violates the compute budget.

For each block document tensor dimensions, input assumptions, parameter count and approximate memory effects. Include shape tests and gradient checks on tiny tensors.

Support tiled inference with overlap, blending and valid border handling. Strain estimation near patch seams must be checked against whole-image inference on manageable examples.

## 11. Losses and numerical safeguards

Define the sampling convention explicitly. For a backward sampler mapping reference coordinates into the moving image:

`I_warp(x) = I_moving(x + u(x))`.

Keep code, equations and plots consistent. Implement a configurable objective with separate logged components:

`L = λ_rf L_rf + λ_env L_env + λ_fb L_fb + λ_s L_smooth + λ_j L_jacobian + λ_sup L_supervised`.

Use only terms supported by the selected training mode.

- RF similarity: robust local normalized correlation or another justified RF-sensitive criterion. Explain phase wrapping, narrowband ambiguity and decorrelation failure modes.
- Envelope similarity: local correlation or robust normalized residual; SSIM is optional and must have a documented data range.
- Bidirectional consistency: `u_f(x) + u_b(x + u_f(x))`, evaluated only where both fields are valid.
- Smoothness: anisotropic robust first- or second-order regularization; preserve inclusion boundaries and justify any edge weights. B-mode appearance may not identify mechanical boundaries.
- Jacobian penalty: discourage implausible foldings of `φ(x)=x+u(x)` without equating every numerical violation to clinical pathology.
- Supervised displacement loss: use verified reference fields and masks, with axial/lateral units handled consistently.

If uncertainty weights the residual, include a properly normalized likelihood or regularization term to prevent the model from setting every pixel uncertain. Add an ablation showing whether uncertainty collapses, merely tracks intensity, or predicts actual error on labeled simulation data.

Always normalize masked losses by valid support, check zero-support cases, prevent divide-by-zero and report excluded areas. Do not allow NaN replacement to conceal unstable training. Float32 should be available for correlation, warping and derivatives even when mixed precision is enabled elsewhere.

Do not impose strict two-dimensional zero divergence as a universal incompressibility law: out-of-plane deformation may be unobserved. Mechanical constraints must match the dimensional and material assumptions.

Optional equilibrium or inverse-elasticity losses require explicit boundary conditions, constitutive assumptions and identifiability analysis. They are not a default route to absolute modulus.

## 12. Strain and mechanical features

Compute small-strain components from displacement in physical coordinates when physical spacing is known:

- `ε_zz = ∂u_z/∂z`
- `ε_xx = ∂u_x/∂x`
- `ε_xz = 0.5(∂u_x/∂z + ∂u_z/∂x)`

Document that engineering shear strain is `2 ε_xz`. State the compression sign convention. When deformation is not small, discuss whether a finite-strain tensor is needed rather than silently reusing the small-strain interpretation.

Compare finite differences with a local least-squares derivative estimator. Include the bias/variance trade-off of smoothing and use fixed validation-selected settings. Preserve unclipped numerical outputs; visualization clipping must not modify metrics.

Extract ROI-level features when masks are available or an explicitly identified segmentation procedure supplies them:

- median and robust spread of axial strain;
- strain ratio relative to a declared reference region;
- heterogeneity, quantiles and spatial variation;
- inclusion/background contrast and CNR with explicit formulas;
- boundary strain gradients and displacement discontinuity measures;
- ROI area and boundary features with physical units when possible;
- RF/envelope quality and residual summaries;
- uncertainty and invalid-pixel fractions;
- stability across compression levels or ordered acquisitions.

Specify numerator/denominator for every strain ratio. Use a minimum denominator and report unstable ratios as invalid. Avoid comparing reference regions at incompatible depths/loading conditions without qualification.

Freehand compression strain alone does not identify absolute Young's modulus. Do not convert strain maps to kPa or apply shear-wave speed formulas to compression datasets. Simulation modulus metadata can characterize simulated materials but is not automatically an observed per-pixel modulus map. A future inverse-modulus module must report assumptions, boundary/load information and scale ambiguity.

## 13. Label-free representations and phenotypes

Build a modest representation-learning extension after displacement estimation works:

- physically meaningful mechanical feature baseline;
- compact encoder embeddings from displacement/strain with validity masks;
- optional variance-preserving self-supervised representation objective;
- train-only standardization and optional PCA;
- Gaussian mixture or another explicitly selected clustering model;
- bootstrap cluster stability, posterior membership and outlier diagnostics;
- sensitivity to cluster count and nuisance variables.

Cluster count is selected with training/validation criteria, not to match an expected number of clinical grades. Avoid a circular result where thresholding a feature generates labels and the same feature is evaluated as a successful classifier.

Inspect whether clusters track geometry, frequency, compression, gain or noise rather than tissue mechanics. Test this with nuisance-prediction probes and controlled simulation contrasts. Plot embeddings for interpretation but do not use attractive UMAP/t-SNE separation as proof of clinical value.

Ordering clusters by strain or simulated stiffness produces a descriptive mechanical ordering only. Preserve this naming in the CLI and reports.

## 14. Optional pathology-linked ordinal grading

Implement an extension interface with strict requirements:

- verified patient/lesion-level reference labels;
- a documented organ-specific grading definition and label provenance;
- sufficient independent examples per category for the proposed evaluation;
- date and lesion linkage between imaging and reference pathology;
- missing or uncertain reference labels handled explicitly;
- patient-grouped training, calibration and test partitions.

Provide ordinal logistic regression on mechanical features as a strong small-data baseline. Add a small ordinal neural head with monotonic cumulative probabilities and a clearly derived loss. Do not use arbitrary integer MSE as the only grading objective.

Compare mechanical features, learned embeddings and their fusion. An optional B-mode baseline requires actual corresponding images and must not stand in for RF-based elastography. Binary diagnosis labels support binary classification, not histological grading. BI-RADS and stage labels must not be relabeled as grade.

Report macro-F1, balanced accuracy, confusion matrices, per-class recall, ordinal MAE, quadratic weighted kappa and probability metrics where applicable. Include patient-level confidence intervals and failure examples. With insufficient independent clinical data, produce a readiness report instead of misleading performance tables.

## 15. Uncertainty, calibration and selective prediction

Separate:

- measurement quality/decorrelation;
- model disagreement or epistemic uncertainty;
- aleatoric estimates tied to a likelihood;
- out-of-distribution evidence;
- clinical label uncertainty when available.

Begin with residual, forward/backward mismatch and ensemble disagreement baselines. Train a small ensemble only if compute allows; otherwise use a documented economical approximation and do not claim equivalent behavior.

Validate uncertainty against observed displacement/strain errors on simulation. Evaluate whether uncertainty remains useful under frequency and corruption shifts. A high confidence estimate is not evidence of correctness without calibration.

For clinical classification, fit temperature scaling or another chosen calibration method on independent calibration groups. Show reliability diagrams, NLL and Brier score. Do not call mechanics confidence a probability of malignancy.

Conformal prediction is optional but must use an appropriate held-out exchangeable calibration unit. For patient-level targets calibrate at the patient/lesion unit justified by the protocol, not on millions of dependent pixels. Explain marginal coverage, finite-sample granularity and loss of guarantees under arbitrary distribution shift. If calibrating image-level mechanics intervals, define the aggregation/nonconformity score and coverage event precisely.

Selective prediction must show error versus retained coverage, rejection reasons and error among accepted cases. Prevent a system that looks excellent only by rejecting nearly every case from being presented as generally accurate.

## 16. MIS-V syllabus integration: exact coverage

The user's course is **23MAT303 — Mathematics for Intelligent Systems 5**, Semester 5, L-T-P-C 2-0-2-3.

Use these supplied syllabus topics as the authoritative scope:

- Unit 1: introduction to probabilistic graphical models; probabilistic AI; introduction to Bayesian networks; representation learning in Bayesian networks; inference in Bayesian networks.
- Unit 2: Markov networks; independencies in Markov networks; hidden Markov models.
- Unit 3: Markov random fields; decision networks; conversion from Bayesian networks to Markov networks.

Course outcomes: model complex systems with PGMs; develop Bayesian-network foundations and applications; develop directed and undirected graphical models; apply graphical models to real-world problems.

Integrate these models into the actual mechanics-quality workflow where appropriate, and also provide transparent educational examples. Do not attach a superficial risk-analysis paragraph and claim full syllabus coverage.

### Unit 1: Bayesian quality network

Define a small interpretable network over domain, acquisition quality, reconstruction residual, cycle inconsistency, uncertainty and mechanical-estimate reliability. Choose a DAG and justify each conditional dependence. A possible generative structure is domain → quality, quality → residual/cycle evidence, and quality/domain → reliability. Treat the graph as an assumed probabilistic model rather than an established causal graph.

Define reliability targets using a predeclared physical-error tolerance where simulation reference exists. Real data without reference cannot supply true reliability labels merely because its residual is small. Model transport from simulation must be evaluated and qualified.

Provide a transparent discrete BN baseline and, if justified, a learned emission/latent-state extension. Learn discretization boundaries on training data only. Estimate CPDs with smoothing, validate normalization and compare held-out likelihood and reliability prediction.

For representation learning, show how a latent categorical quality state can explain multiple observations; compare a small latent-variable/EM example with a fully observed model. Explain identifiability and state-permutation ambiguity. A neural embedding feeding a BN is not automatically the same as learning a Bayesian-network representation.

Teach factorization, marginalization, Bayes' rule, conditional independence, variable elimination and posterior queries. Implement exact enumeration for a tiny network as a check against the chosen library. Show missing-evidence inference and a sensitivity analysis for prior/CPD assumptions.

### Unit 2: Markov networks and HMM

Construct an undirected graph for spatial quality/phenotype variables. Explain clique potentials, positivity assumptions, partition function and local/global conditional independence. Demonstrate graph separation with small executable examples. Do not equate zero correlation with conditional independence.

Use an HMM only for verified temporally ordered acquisitions. Hidden states can be stable, decorrelated and recovering acquisition quality; observations may include residual, motion and dropout statistics. Implement forward-backward smoothing, online filtering and Viterbi decoding; explain why these outputs differ.

Fit transitions/emissions on training sequences, reset at sequence boundaries and prevent test-sequence parameter fitting. If only independent RF pairs are available, the real-data temporal module must remain unavailable; use an explicitly labeled synthetic sequence for course demonstrations. Never sort unrelated pairs by filename and call them a temporal sequence.

### Unit 3: MRF and decision network

Implement a small Potts-style spatial MRF:

`E(y) = Σ_i ψ_i(y_i) + β Σ_(i,j) w_ij 1[y_i ≠ y_j]`.

Unaries can come from mechanical phenotype likelihoods; pairwise terms promote local coherence. Define how quality affects unary strength and how neighborhood weights are computed. Distinguish an MRF joint model from a CRF conditioned on observed images. Choose and name the actual model correctly.

Implement a tractable mean-field or ICM baseline, explain limitations and compare with no spatial smoothing. Validate on a tiny graph with enumeration where feasible. Measure boundary preservation and small-inclusion loss, not just visual smoothness.

Convert the educational BN to an undirected representation by moralizing: connect co-parents, remove directions, and assign CPDs as clique factors. Show that the factor product retains the BN distribution, while the graph can lose some conditional-independence information. Moralization is not triangulation. Explain induced width and when junction-tree construction additionally needs triangulation. Check normalized joint probabilities and posterior queries numerically before and after conversion.

Create a decision network using posterior reliability, available actions and a transparent loss/utility table. Primary research actions: accept mechanical estimate, request a new acquisition, or abstain/refer for expert review. Compute:

`a* = argmin_a Σ_s L(a,s) P(s | evidence)`.

The utility table is a configurable educational/research assumption, not a validated clinical policy. Include sensitivity to false-acceptance cost and reacquisition burden. A pathology-dependent utility must remain in the clinical extension and use validated probabilities.

### Required syllabus traceability

Create a table mapping every topic and CO1–CO4 to its equation, source module, notebook, executable experiment and assessment question. Include short worked numerical examples, graph figures, interpretation questions and answers. Make clear which experiments use real data and which demonstrate mathematics on fixtures.

## 17. Risk analysis with quantitative links

Create a research risk register with hazard, mechanism, impact, detectable evidence, mitigation, verification and residual limitation. Include:

- patient or geometry leakage;
- incorrect RF axis or displacement sign;
- wrong physical spacing or units;
- phase ambiguity and speckle decorrelation;
- excessive compression and out-of-plane motion;
- oversmoothing small inclusions;
- domain shift across frequency/scanner/site;
- absent stress information and non-identifiable modulus;
- poor uncertainty calibration and confident failures;
- spurious phenotype/grade associations;
- missing pathology provenance;
- biased exclusion and selective-prediction coverage;
- insufficient independent groups;
- corrupted input and software errors.

Use Bayesian inference to combine evidence about reliability and decision theory to examine risk-sensitive actions. Do not present hand-assigned risk scores as empirically estimated probabilities. If a simple FMEA-style score is included, label it ordinal prioritization rather than a calibrated measure of harm.

Create an error taxonomy and link observed failures to run IDs. Quantify robustness where data permits. Include a limitations section and a concrete clinical-validation roadmap without claims of regulatory compliance.

## 18. Evaluation protocol

Metrics must state formula, aggregation unit, mask, physical units, direction of improvement and eligibility conditions.

Displacement: axial/lateral MAE and RMSE, endpoint error, valid-pixel fraction, and error stratified by motion/compression. Do not mix physical and pixel units.

Strain: MAE/RMSE against verified reference strain or an explicitly documented derivative of reference displacement; contrast/CNR for verified ROIs; boundary preservation; strain noise in defined homogeneous regions. Do not call contrast a segmentation score or a clinical accuracy measure.

Registration: valid-region correlation and residual, cycle mismatch and Jacobian summaries. Report these as proxies when no ground truth exists.

Segmentation: Dice/IoU only when independent reference masks exist. Masks derived from the same predicted strain cannot independently validate that strain.

Phenotypes: stability, nuisance association and outlier fraction. Silhouette alone is insufficient.

Uncertainty: error-uncertainty relationship, coverage/width for intervals where defined, risk-coverage curves and calibration metrics appropriate to the target.

Clinical metrics: enabled only after the clinical gate; use group-level uncertainty intervals.

Compute efficiency: parameter count, peak memory, batch size, full-image shape, synchronized GPU timings, warm-up, preprocessing and inference latency separately. Report hardware/software details. Do not extrapolate toy CPU timing to production GPU throughput.

Use bootstrap at the independent geometry/acquisition/patient unit. Do not bootstrap pixels as independent patients. Report seeds separately from data-sampling uncertainty. Use paired group-level comparisons and account for many comparisons where applicable. Never tune thresholds on final test results.

For missing metrics use `not_applicable` or `not_evaluated`, not zero. Export numerators/denominators and exclusion counts where practical.

## 19. Ablation and robustness matrix

Required priority experiments:

- classical NCC vs compact self-supervised baseline vs candidate;
- self-supervised only vs simulation-supervised vs hybrid adaptation;
- RF only vs envelope only vs fusion;
- with/without bidirectional consistency;
- with/without anisotropic smoothness;
- with/without confidence weighting;
- no spatial regularization vs MRF;
- threshold quality rule vs BN-based quality inference;
- no rejection vs selective prediction;
- seen-domain vs held-out geometry and frequency conditions.

Optional experiments, after core runs succeed:

- attention ablation;
- derivative-estimator comparison;
- representation/clustering variants;
- HMM vs independent-frame quality inference on true sequences;
- ensemble size and compute trade-offs;
- clinical feature-only vs embedding-only vs fused ordinal prediction.

Stress tests: plausible RF noise, gain changes, dropout, compression/decorrelation changes, reduced sampling, metadata loss, unsupported array shapes and held-out simulation geometries. Corruptions must respect acquisition semantics, and severe unrealistic corruptions must be labeled stress-only.

Create a prioritized experiment plan with compute tiers. A factorial explosion is not required; explain which interactions matter and why.

## 20. Compute and environment design

Provide profiles for CPU demo, an 8 GB CUDA GPU and larger optional experiments. Treat 8 GB as a design target to measure, not a guaranteed memory claim.

Use a compact model, small batches, patching, gradient accumulation and optional mixed precision. Profile memory before scaling. Provide deterministic settings and explain operations whose exact reproducibility is limited.

Keep CPU smoke tests short and bounded. Full training requires an explicit command; notebook smoke runs must not accidentally train for hours. Do not download giant models on import.

Pin a tested dependency set. Use current official documentation when selecting version-sensitive APIs. Provide a portable base environment and optional extras for notebooks, TUI, visualization and extended models. Avoid pinning a CUDA wheel incompatible with the user's machine; document CPU and appropriate GPU installation routes.

Support Windows PowerShell, Linux bash and hosted notebook environments. Use `pathlib`, configurable data roots and safe process spawning. Never hard-code the author's paths. Avoid shell-only assumptions inside Python.

## 21. Repository architecture

Create this structure, adapting small details when justified:

```text
elastograde-research/
  README.md
  pyproject.toml
  requirements-lock.txt
  CITATION.cff
  LICENSE
  THIRD_PARTY_NOTICES.md
  .gitignore
  configs/
    demo.yaml
    cpu.yaml
    gpu_8gb.yaml
    datasets/
    models/
    experiments/
    pgm/
  src/elastograde/
    __init__.py
    cli.py
    config.py
    capabilities.py
    data/
      acquisition.py
      inventory.py
      schemas.py
      impact_acquired.py
      impact_simulation.py
      synthetic.py
      preprocessing.py
      splits.py
      audit.py
    mechanics/
      coordinates.py
      warping.py
      strain.py
      features.py
    baselines/
      ncc.py
      regularized.py
    models/
      compact_unet.py
      pyramid_flow.py
      representations.py
      ordinal.py
    training/
      losses.py
      engine.py
      checkpoints.py
    inference/
      pipeline.py
      tiling.py
      outputs.py
    uncertainty/
      quality.py
      calibration.py
      ood.py
      selective.py
    pgm/
      bayesian_quality.py
      latent_quality.py
      markov_network.py
      hmm.py
      mrf.py
      moralization.py
      decisions.py
    evaluation/
      metrics.py
      grouped_statistics.py
      robustness.py
      ablations.py
    reporting/
      figures.py
      reports.py
      provenance.py
    tui/
      app.py
      screens.py
  notebooks/
  scripts/
    generate_notebooks.py
    execute_notebooks.py
  tests/
    fixtures/
    test_coordinates.py
    test_splits.py
    test_metrics.py
    test_pgm.py
    test_capabilities.py
    test_cli.py
    test_notebooks.py
  docs/
    theory.md
    dataset_cards/
    model_card.md
    clinical_readiness.md
    mis_v_mapping.md
    risk_register.md
    experiment_protocol.md
    troubleshooting.md
    references.bib
    claims_ledger.md
  reports/
  data/README.md
  artifacts/README.md
```

Do not check datasets, secrets, huge caches or private patient metadata into Git. Do not invent an author identity or put restrictive third-party data under the code license. Explain the selected code license and keep external licensing separate.

## 22. Notebook contract

Create real nbformat-valid `.ipynb` files, not Python scripts renamed as notebooks. Use reusable package imports rather than duplicating every algorithm in cells.

Every notebook must have a question, prerequisites, theory, equations, input/output descriptions, configuration, reproducible execution, plots, interpretation, failure cases and a concluding evidence statement. Include a small concept explanation for an undergraduate reader without sacrificing mathematical precision.

Use parameters such as `mode`, `data_root`, `config_path`, `seed`, `max_samples`, `device` and `output_dir`. A notebook should run in demo mode without real data. Where a real-data-only section is unavailable, explicitly skip it with a reason and never create a fake result.

Required notebooks:

| File | Required scope and output |
| --- | --- |
| `00_Project_Map_and_Claims.ipynb` | Scope, label taxonomy, mode gates, architecture and claims ledger |
| `01_Environment_and_Reproducibility.ipynb` | Device detection, dependency versions, deterministic fixture run |
| `02_Dataset_Acquisition_and_Inventory.ipynb` | Verified access steps, archive checks, schema inspection and capabilities |
| `03_RF_Physics_and_Preprocessing.ipynb` | RF/envelope/B-mode, sampling, axes, phase and preprocessing comparisons |
| `04_Grouped_Splits_and_Leakage_Audit.ipynb` | Group lineage, overlap checks and limits of independence |
| `05_Classical_Displacement_Baselines.ipynb` | NCC, subpixel refinement, error maps and failure examples |
| `06_Strain_and_Mechanical_Features.ipynb` | Derivatives, units, strain ratio, CNR and identifiability |
| `07_Self_Supervised_Learning.ipynb` | Warping, candidate architecture, losses and short genuine training run |
| `08_Simulation_Supervision_and_Adaptation.ipynb` | Accurate training-regime comparison and domain gap |
| `09_Mechanical_Representations_and_Clustering.ipynb` | Feature baselines, latent features, stability and confounding |
| `10_Uncertainty_and_Selective_Prediction.ipynb` | Residual/ensemble baselines, interval assumptions and risk-coverage |
| `11_MISV_Bayesian_Networks.ipynb` | Factorization, representation learning, CPDs, exact inference and missing evidence |
| `12_MISV_Markov_Networks_and_MRF.ipynb` | Independencies, potentials, spatial inference and smoothing failures |
| `13_MISV_Hidden_Markov_Models.ipynb` | Filtering, smoothing, Viterbi and true-sequence capability gate |
| `14_MISV_Moralization_and_Decision_Networks.ipynb` | BN conversion, probability checks, utilities and Bayes actions |
| `15_Clinical_Ordinal_Grading_Readiness.ipynb` | Real-data eligibility audit and clearly marked synthetic interface demonstration |
| `16_Benchmarks_Ablations_and_Robustness.ipynb` | Grouped evaluation, comparative plots and missing-run handling |
| `17_Risk_Analysis_and_Failure_Atlas.ipynb` | Quantitative failures, risk register and decision sensitivity |
| `18_End_to_End_CLI_and_Research_Report.ipynb` | Invoke actual CLI, load outputs, generate report and provenance |

Add compact worked exercises and answers for the MIS-V notebooks. Execute all notebooks in bounded demo mode from clean kernels. Clear stale outputs or regenerate them; never retain outputs from a different code version as current evidence.

## 23. CLI design

Use a coherent framework such as Typer plus Rich, with compatible pinned versions. Make CLI commands thin wrappers around tested package functions.

Implement commands equivalent to:

```bash
elastograde doctor
elastograde data instructions --dataset impact-acquired
elastograde data inspect --root DATA_ROOT --out reports/inventory
elastograde data prepare --config configs/datasets/impact.yaml
elastograde data split --config configs/experiments/splits.yaml
elastograde data audit --manifest artifacts/manifest.json
elastograde demo --out reports/demo --seed 42
elastograde train --config configs/demo.yaml
elastograde evaluate --run RUN_DIRECTORY --split test
elastograde infer --input SAMPLE_PATH --checkpoint CHECKPOINT_PATH --mode mechanics
elastograde phenotype fit --config CONFIG_PATH
elastograde pgm demo --topic bayesian-quality
elastograde pgm demo --topic moralization
elastograde risk analyze --run RUN_DIRECTORY
elastograde report build --run RUN_DIRECTORY --format html
elastograde notebooks execute --mode demo
elastograde clinical readiness --manifest CLINICAL_MANIFEST
elastograde tui
```

Make each documented command actually work, or show its exact missing-data prerequisite. Generate the referenced example configs; do not document paths that do not exist.

Provide help, clear argument validation, progress, cancellation, resumable checkpoints, structured logs, exit codes and `--json` where useful. Support dry-run for expensive operations. Do not swallow exceptions behind a success message.

Inference outputs should include:

- physical or pixel-coordinate displacement with units;
- strain and validity maps;
- mechanical summary features;
- quality evidence and uncertainty method;
- descriptive phenotype posterior when a fitted model exists;
- action and rejection reasons;
- source and checkpoint provenance;
- `grade_status` and `grade_probabilities: null` when unsupported.

The UI must not convert a refused grade into a default grade. Missing spacing blocks physical-unit claims but can permit explicitly labeled pixel-unit operations.

## 24. Interactive terminal UI

Use a maintainable terminal framework such as Textual if available. The app must reuse the CLI/package services rather than implementing a separate research pipeline.

Screens:

1. Dashboard: mode, dataset capability, environment and latest runs.
2. Data browser: provenance, schema, quality and split summaries.
3. Experiment builder: validated presets and compute estimate/status.
4. Training monitor: losses, metrics, resource usage, checkpoints and stop/resume.
5. Inference: select pair and checkpoint, inspect status and outputs.
6. Probabilistic reasoning: evidence, posterior quality, decision costs and chosen action.
7. Results: comparable runs, plots, exports and claims readiness.
8. Help: physics glossary, MIS-V links and troubleshooting.

Support keyboard navigation, readable contrast, narrow terminal fallback and non-color status indicators. Avoid requiring terminal image protocols: save detailed figures as PNG/HTML and offer opening them locally. No browser server or cloud upload is needed for a CLI-based UI.

Long-running work must not freeze navigation. Cancellation must leave a consistent run state. Show “not run”, “running”, “failed”, “completed” and “blocked by missing data” distinctly. Never present synthetic runs as clinical results.

## 25. Massive README specification

Write a self-contained educational README, approximately 10,000–15,000 useful words if the implementation scope supports it. Length is subordinate to accuracy and clarity: do not pad it with repeated claims. Keep the quick start near the top and use a navigable contents list. Put exhaustive API details in linked docs while retaining substantial theory in the README itself.

Required coverage:

1. Project purpose, research question and honest current status.
2. Title interpretation and the three meanings of label-free.
3. What these datasets support and what clinical grading still requires.
4. A newcomer-friendly explanation of ultrasound and elastography.
5. RF, IQ, envelope and B-mode; why raw signal access matters.
6. Strain elastography versus shear-wave elastography.
7. Coordinate systems, units, sampling and displacement conventions.
8. Speckle tracking, decorrelation, compression and out-of-plane motion.
9. Displacement-to-strain derivation with a small numeric example.
10. Stress, strain, modulus and inverse-problem identifiability.
11. Dataset cards, acquisition instructions, citations and license status.
12. Canonical schema, preprocessing and provenance.
13. Grouped splitting, independent units and leakage examples.
14. Architecture diagrams and tensor-shape walkthrough.
15. Each baseline and why it is necessary.
16. Each objective term, its assumptions and failure modes.
17. Pure self-supervised, simulation-supervised and hybrid protocols.
18. Mechanical features and reference-region choices.
19. Representation learning, clustering and why clusters are not grades.
20. Ordinal grading extension, data gates and reference standards.
21. Uncertainty, calibration, OOD detection and selective prediction.
22. Bayesian-network fundamentals, CPDs and posterior inference.
23. Latent quality representation and identifiability.
24. Markov networks, independencies and partition functions.
25. HMM filtering, smoothing and decoding.
26. MRF spatial coherence and oversmoothing risks.
27. Moralization, clique factors and triangulation distinctions.
28. Decision networks, loss matrices and worked expected-loss calculation.
29. Complete MIS-V topic/CO mapping.
30. Risk register and observed failure atlas.
31. Installation on Windows/Linux and CPU/GPU profiles.
32. Five-minute offline demo with expected types of outputs.
33. Real-data setup and commands from raw data to a report.
34. Notebook-by-notebook learning guide.
35. CLI command reference and terminal UI walkthrough.
36. Configuration reference with tested examples.
37. Training, resuming, checkpointing and inference.
38. Experiment matrix, baselines and ablation priorities.
39. Metric definitions, statistical protocol and reporting rules.
40. Actual results if run, otherwise explicit not-run tables.
41. Runtime and memory measurements with their conditions.
42. Reproducibility, seeds, versions and artifact lineage.
43. Numerical tests, data audits and continuous integration.
44. Troubleshooting matrix for common data and CUDA failures.
45. Limitations, domain shift and validation roadmap.
46. Research contribution hypotheses and evidence needed for each.
47. A practical milestone plan with exit criteria.
48. FAQ, glossary and symbol table.
49. Verified bibliography and citation instructions.
50. File tree, code map and contribution guide.

Use meaningful diagrams: branched pipeline, split/data lineage, BN, MRF and decision network. Mathematical graphs must match code. Include explanatory tables and units. Use illustrative numbers only when visibly labeled as worked examples, never as experiment results.

Add at least 20 substantive FAQ entries, including: Can I grade real tumors now? Why not use stiffness thresholds? Why does a phantom not represent clinical validity? Why do good registration images still fail strain estimation? Why cannot every RF pair become a separate patient? Why can a probability be confident but wrong? Why does the HMM require sequences? Why is moralization not causal discovery? What changes with an 8 GB GPU? What evidence is needed to claim state of the art?

## 26. Reports and publication artifacts

Generate a local HTML research report and Markdown summary from real run artifacts. Include mode, data origin, independent group counts, split hash, configuration, checkpoint checksum, code version, units, metrics, failure examples and limitations.

Create plots only from recorded arrays or metrics. Use consistent axes and color scales for comparisons, physical scale bars where justified, colorblind-friendly palettes and visible invalid regions. Separate uncertainty and strain color maps. Plot titles must identify synthetic fixtures when used.

Include actual-vs-predicted displacement, strain panels, residuals, calibration/risk-coverage plots, group-level metric distributions, ablation comparisons, PGM figures and decision sensitivity. Skip unsupported plots with reasons.

Export CSV/JSON metrics, figure provenance and a machine-readable report manifest. Build comparison tables from completed eligible runs, never hard-coded performance values. No claimed confidence interval may come from fabricated observations.

## 27. Meaningful tests and acceptance criteria

Implement tests that detect real scientific or software errors:

- zero motion warps to identity;
- known subpixel translations verify sign, direction and units;
- affine compression has the expected analytic strain;
- vector augmentation transforms coordinates correctly;
- reference-frame composition is checked on a known deformation;
- masked similarity remains finite and rejects empty support;
- raw and processed caches preserve provenance;
- parent-frame/geometry overlap is rejected;
- normalization and clustering do not fit on test groups;
- constant/NaN/corrupt arrays fail clearly;
- physical metrics reject missing spacing;
- BN CPDs normalize and exact inference matches enumeration;
- moralized factors reproduce the toy BN joint distribution;
- MRF inference agrees with a tiny exact example where applicable;
- HMM probabilities are normalized and sequence boundaries reset state;
- expected-loss actions match a hand-computed example;
- clinical output cannot activate without verified prerequisites;
- demo outputs are marked synthetic;
- CLI errors return nonzero codes;
- checkpoint save/resume preserves configuration and model state;
- valid notebooks execute in smoke mode.

Do not prescribe arbitrary excellent accuracy thresholds on real data. Numerical fixture tolerances are justified by known analytic solutions. Test installation in a fresh environment when feasible and record what was actually executed.

## 28. Execution phases and completion protocol

Phase 1: inspect environment and source documentation; create a facts/unknowns table, evidence ledger, repository structure and dependency strategy.

Phase 2: implement canonical schema, fixtures, preprocessing, coordinates, grouped split audit and classical baseline; run a complete mechanics demo.

Phase 3: implement compact learning model, training loop, losses, checkpointing and tiny overfit/smoke validation; add the real-data adapters once formats are inspected.

Phase 4: implement strain features, phenotype modeling, uncertainty and selective prediction. Preserve mode boundaries.

Phase 5: implement MIS-V modules and exact numerical demonstrations; connect suitable models to the quality workflow.

Phase 6: implement conditional clinical interfaces and strict capability gates; test with labeled synthetic fixtures without clinical claims.

Phase 7: implement CLI and terminal UI, report generator, notebooks and extensive documentation.

Phase 8: execute tests and bounded notebooks, fix defects, audit claims and package the deliverable.

For tasks spanning context windows, maintain `IMPLEMENTATION_STATUS.md` with completed files, tests, unresolved evidence requirements and exact next commands. Do not stop after a plan if execution tools are available. Do not pretend full training occurred when only smoke tests ran.

At completion, provide:

- repository location or downloadable archive;
- exact installation and offline-demo commands;
- exact real-data acquisition/placement instructions;
- notebook execution order;
- test and notebook pass/fail counts;
- executed experiments and their conditions;
- unexecuted expensive experiments and exact launch commands;
- clinical readiness status;
- concrete remaining blockers;
- concise next research milestone.

The offline path must genuinely work. The full real-data path must either work on supplied data or state precisely which unverified adapter details remain. Unknown archive layouts are not solved by a plausible-looking hard-coded loader.

## 29. Final self-audit before delivery

Answer these questions with evidence, not reassurance:

1. Are the two datasets correctly identified as physical/simulated phantom data?
2. Is every output mode named honestly?
3. Are pathology grades absent from unsupported real-data outputs?
4. Is simulation supervision separated from fully self-supervised training?
5. Are reference frame, vector order, spacing and units consistent?
6. Is absolute modulus withheld when it is not identifiable?
7. Are independent groups protected across all derived samples?
8. Does every documented CLI example reference implemented commands/configs?
9. Can the demo run offline without downloaded weights?
10. Do the notebooks explain results and failure cases rather than merely display code?
11. Is every MIS-V topic connected to an executable example?
12. Is the HMM limited to actual sequences or clearly marked synthetic demonstrations?
13. Are BN probabilities learned/assumed with explicit provenance?
14. Do risk decisions disclose utility assumptions?
15. Are empirical results traceable to completed runs?
16. Are missing results marked honestly?
17. Does the README distinguish hypothesis, implementation and validation?
18. Have current external sources and software licenses been checked where required?
19. Is the system small enough to debug and profile on the stated compute tier?
20. Is the next clinical-validation requirement concrete?

Now implement the repository and verify it. Do not respond with another generic project outline.

# END PROMPT
