# Research risk register

No hand-assigned score here is presented as a probability of clinical harm.

| Hazard | Mechanism | Impact | Evidence | Mitigation | Verification | Residual limitation |
|---|---|---|---|---|---|---|
| Leakage | Shared frames/groups | Inflated evaluation | Lineage overlap audit | Split before augmentation | test_splits.py | Unrecorded identities remain unknown |
| Axis/sign error | Misinterpreted RF convention | Wrong motion and strain | Analytic translation/compression tests | Explicit mapping | test_coordinates.py | Source documentation required |
| Wrong spacing | Pixel/physical unit mixing | Incorrect shear and area | Spacing validation | Require positive metadata | test_coordinates.py | Scanner metadata may itself be wrong |
| Phase ambiguity | Repeated carrier peaks | False correspondence | Residual/cycle/reference error | NCC bounds and bidirectionality | test_coordinates.py | Consistent wrong fields possible |
| Decorrelation / out-of-plane motion | Changed scatterer pattern | Unreliable strain | Stress and quality evidence | Reject/reacquire research policy | notebook 17 | Proxy thresholds uncalibrated |
| Excessive smoothing | High regularization | Small inclusion lost | Boundary and tiny-region tests | Ablate beta/window | notebook 12 | No real lesion masks yet |
| Domain shift | Frequency/scanner/site change | Confident errors | Held-out domains and OOD scores | Independent evaluation | notebook 16 | Arbitrary shift not covered |
| Non-identifiable modulus | Unknown stress/load | False kPa claims | Output/schema audit | Withhold absolute modulus | test_capabilities.py | Additional mechanics needed |
| Poor uncertainty | Misfit likelihood or priors | False acceptance | Risk-coverage and calibration | Independent calibration groups | notebook 10 | Unseen failures may evade detection |
| Spurious grades | Nuisance-driven clusters | Misleading disease inference | Label provenance gate | Neutral cluster names | test_capabilities.py | Clinical evidence absent |
| Missing pathology | No linked reference | Invalid grading study | Clinical readiness reasons | Block grade output | test_capabilities.py | Gate is not medical validation |
| Selective bias | Reject difficult subgroups | Optimistic accepted accuracy | Coverage and rejection reasons | Report all eligible groups | notebook 10 | Small subgroup counts |
| Too few groups | Many dependent samples | Unstable intervals | Group count | Grouped bootstrap and qualified claims | test_metrics.py | No power analysis supplied |
| Corrupted input | NaN/checksum/archive errors | Crash or wrong estimates | Validation and checksum | Fail clearly | test_data.py | All format variants cannot be anticipated |
