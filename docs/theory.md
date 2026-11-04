# Theory guide


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

