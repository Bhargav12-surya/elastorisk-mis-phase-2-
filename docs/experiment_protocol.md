# Prespecified experiment protocol


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

