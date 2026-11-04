# Requirement coverage

The original specification is retained in GENERATION_SPECIFICATION.md. This table identifies the implementation route for every numbered section. “Implemented” is not a claim that data-dependent research has been performed.

| Section | Implementation route | Evidence / dependency |
|---|---|---|
| 1. Mission and working standard | README.md; pyproject.toml | Software and synthetic demonstrations; see verification report. |
| 2. Scientific scope and capability gates | capabilities.py; docs/clinical_readiness.md | Software and synthetic demonstrations; see verification report. |
| 3. Research question and candidate contribution | README.md; docs/experiment_protocol.md | Software and synthetic demonstrations; see verification report. |
| 4. Dataset acquisition and verification | data/acquisition.py; data/inventory.py | Exact IMPACT source mapping and independent IDs require actual archives; explicit fixtures tested. |
| 5. Canonical schema and provenance | data/schemas.py; data/cache.py | Exact IMPACT source mapping and independent IDs require actual archives; explicit fixtures tested. |
| 6. Splitting and leakage prevention | data/splits.py; data/audit.py | Exact IMPACT source mapping and independent IDs require actual archives; explicit fixtures tested. |
| 7. Ultrasound preprocessing and coordinate conventions | data/preprocessing.py; mechanics/coordinates.py | Software and synthetic demonstrations; see verification report. |
| 8. Offline synthetic fixtures | data/synthetic.py | Software and synthetic demonstrations; see verification report. |
| 9. Baselines and experimental fairness | baselines/; models/ | Software and synthetic demonstrations; see verification report. |
| 10. Candidate architecture | models/pyramid_flow.py | Software and synthetic demonstrations; see verification report. |
| 11. Losses and numerical safeguards | training/losses.py | Software and synthetic demonstrations; see verification report. |
| 12. Strain and mechanical features | mechanics/strain.py; mechanics/features.py | Software and synthetic demonstrations; see verification report. |
| 13. Label-free representations and phenotypes | models/representations.py | Software and synthetic demonstrations; see verification report. |
| 14. Optional pathology-linked ordinal grading | models/ordinal.py; capabilities.py | Ordinal models implemented; real clinical inference blocked by missing pathology data and validated checkpoint. |
| 15. Uncertainty, calibration and selective prediction | uncertainty/ | Software and synthetic demonstrations; see verification report. |
| 16. MIS-V syllabus integration: exact coverage | pgm/; docs/mis_v_mapping.md | All syllabus topics implemented; HMM real-data use needs verified sequences. |
| 17. Risk analysis with quantitative links | docs/risk_register.md; notebook 17 | Software and synthetic demonstrations; see verification report. |
| 18. Evaluation protocol | evaluation/ | Protocol, utilities and toy stress experiments provided; full real-data benchmarks not run. |
| 19. Ablation and robustness matrix | evaluation/ablations.py; notebook 16 | Protocol, utilities and toy stress experiments provided; full real-data benchmarks not run. |
| 20. Compute and environment design | configs/; README.md | CPU executed; 8 GB CUDA profile supplied but GPU profiling unavailable. |
| 21. Repository architecture | src/elastograde/ | Software and synthetic demonstrations; see verification report. |
| 22. Notebook contract | notebooks/00–18 | Software and synthetic demonstrations; see verification report. |
| 23. CLI design | cli.py | Software and synthetic demonstrations; see verification report. |
| 24. Interactive terminal UI | tui/ | Software and synthetic demonstrations; see verification report. |
| 25. Massive README specification | README.md | Software and synthetic demonstrations; see verification report. |
| 26. Reports and publication artifacts | reporting/; reports/ | Software and synthetic demonstrations; see verification report. |
| 27. Meaningful tests and acceptance criteria | tests/ | Software and synthetic demonstrations; see verification report. |
| 28. Execution phases and completion protocol | IMPLEMENTATION_STATUS.md | Software and synthetic demonstrations; see verification report. |
| 29. Final self-audit before delivery | docs/claims_ledger.md | Software and synthetic demonstrations; see verification report. |

Optional scope choices: external pretrained optical-flow baselines, FEM/acoustic simulator, inverse-modulus solver and a separately trained representation foundation model are not required by the core and are not claimed as supplied. Core implementations expose small educational algorithms rather than falsely labeling them production clinical models.
