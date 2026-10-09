# `xms` package

Planned namespaces follow the executable boundary:

- `ingest` and `observations` produce time-aligned evidence without rig semantics.
- `solve` consumes observations and publishes motion through `animation`.
- `animation` owns the `AnimationBundle` contract, IO, validation, sampling, and FK.
- `profiles` resolves versioned character and scene metadata.
- `geometry` and `qa` measure constraints and acceptance.
- `render` consumes a validated bundle through Blender adapters.
- `assembly` and `report` produce media and evidence.
- `export` provides optional derived outputs.

Implementation begins in Stage 1 after the Stage 0 runtime smoke. Empty namespace directories are placeholders for that work.

