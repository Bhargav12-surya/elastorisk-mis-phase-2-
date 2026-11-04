# Troubleshooting and operations


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

