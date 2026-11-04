# Clinical readiness


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

