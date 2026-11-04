# Data

The local folder contains acquired phantom RF pairs in `phantom/`, simulation RF sequences and displacement companions in `simulation/`, and RF stacks of unverified provenance in `test/`.

The standalone `elastography_mis.py` pipeline uses both phantom and simulation RF data for self-supervised motion learning. It inventories `test/` and keeps its arrays out of paired training. Initial flow estimates and simulation displacement companions are excluded from supervision until their reference conventions are verified. Pathology grades are not supplied.

See the root [README.md](../README.md) for observed schemas, axis assumptions, acquisition grouping, cache preparation, and GPU commands. Raw MAT files are preserved; caches are written under `artifacts/` by default. Preserve the original dataset provenance and usage terms.
