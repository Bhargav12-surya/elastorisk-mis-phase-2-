# Model card

## Intended research use
RF-pair motion estimation and mechanically descriptive features. The compact pyramid is a candidate model, not a validated diagnostic device. Default inputs are signed RF; optional envelope/fusion variants and acquisition metadata are implemented.

## Outputs and constraints
Reference-to-moving flow [ux,uz] in pixels, validity, strain, quality evidence and pooled features. Physical conversions require spacing. Histological grades and absolute modulus are not supplied by default. Optional variance models a reconstruction likelihood, not clinical confidence.

## Training evidence
Included smoke checkpoints use analytic synthetic fixtures only. Two-epoch demonstration and notebook jobs verify optimization, not convergence or real-data accuracy. Regime, code hash, manifest hash, environment and validation loss are recorded with each run.

## Evaluation
Use geometry/acquisition/patient groups, matched valid masks and reference motion where available. See experiment_protocol.md. Source IMPACT archives and clinical data were not available for file-level validation. GPU memory and throughput are not measured here.

## Limitations
RF phase ambiguity; out-of-plane motion; decorrelation; small inclusion oversmoothing; unknown loading; acquisition shift; uncalibrated default quality probabilities. Synthetic ROIs are illustrative and not finite-element hard inclusions.

## Model selection
A validation objective selects best.pt. Test arrays are not used by training. Calibration roles remain separate. Never select a checkpoint from final test accuracy.
