# RenderPipeline

RenderPipeline is the clean implementation repository for Xandra Motion Studio: an offline pipeline that turns a source MP4 into observations, an executable `AnimationBundle`, animation on the Xandra rig, and a rendered MP4.

The repository starts from the contracts and execution plan in [`docs/development`](docs/development/). Historical projects such as `clip02_blender_kit`, `clip_full`, and `dance` are research inputs only. Runtime code must not import them or depend on their precomputed animation.

## Core boundary

```text
MP4 -> observations -> solver -> AnimationBundle -> Blender adapter -> rendered MP4
```

`AnimationBundle` is the only motion contract consumed by Blender/render. Tracking, fusion, IK, contacts, and collision correction happen before the bundle is published.

## Repository layout

- `src/xms/` — Python package, split by pipeline boundary.
- `blender/` — Blender process entry points; no tracking or solving.
- `profiles/` — versioned character and scene registration data.
- `schemas/` — machine-readable contracts.
- `experiments/` — bounded Stage 0 experiments only.
- `tests/` — unit and integration verification.
- `benchmarks/` — manifests, annotations, policies, and measured results.
- `examples/` — declarative jobs and compact, provenance-marked reference data.
- `assets/` — policy and manifests; large assets stay outside Git.
- `environments/` — verified runtime locks when a stage needs them.
- `docs/development/` — architecture and implementation stages.
- `docs/research/` — reports about legacy attempts and reusable findings.

## Current state

The repository is scaffolded but production implementation has not started. The next step is Stage 0 in [`STAGES.md`](docs/development/STAGES.md).

