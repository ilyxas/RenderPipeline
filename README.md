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

Stages 0–7 are the frozen functional baseline. Stage 8 adds an experimental
single-window body optimizer (`--solver temporal`); baseline remains the default.
The temporal experiment reduces reprojection error and body jitter, but fails
frozen fast-gesture/continuity gates and regresses dance foot drift. It is not
motion-quality acceptance. Work stops after Stage 8.

See the [Stage 8 handoff and comparison videos](docs/development/STAGE8_HANDOFF.md),
the [Stages 5–7 checkpoint](docs/development/STAGES_5_7_HANDOFF.md), and the
preserved [Stages 0–4 checkpoint](docs/development/STAGES_0_4_HANDOFF.md).
Contacts, collision solving and window stitching remain later-stage work.

The temporal backend uses the optional `temporal` dependency group (SciPy) and
accepts one 2–4 second window. Frozen benchmark reproduction uses existing
observations and saved calibration, without rerunning tracking:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/run_temporal.py sing --phase solve
PYTHONPATH=src .venv/bin/python benchmarks/run_temporal.py sing --phase render
```

The harness requires a fresh `runs/stage8/sing` directory; it never overwrites a
published bundle. Repeat for `dance` and `non_neutral` with the same parameters.
