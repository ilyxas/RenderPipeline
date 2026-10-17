# CLAUDE.md — RenderPipeline (Xandra Motion Studio)

Read this first in every session. If anything here disagrees with the code, the code wins — then fix this file.

**Then open [`docs/status/`](docs/status/README.md)** — current status (`STATUS.md`), confirmed problems to fix with evidence (`TODO.md`) and per-session logs (`sessions/`). At the end of every session add `sessions/YYYY-MM-DD.md` and update `STATUS.md` / `TODO.md`.

## What this is and why

An offline macOS pipeline: a source MP4 of a real person → the same motion performed by **Xandra** (a professionally built, anatomically detailed rigged character) → rendered MP4.

Why it exists: generative video gives a result but no control. Xandra gives full control over every bone and morph, so the same body can later do anything — paired acrobalance, an obstacle course, yoga lessons. RenderPipeline expands her motion vocabulary beyond hand-made clips. Results are also meant to flow back into HumanoidAlpha (bundle → animated GLB → Animation Mode); that link is not yet verified end to end.

This is an exploratory, for-fun research project. Owners: Ilya and Claude. Earlier work was done by Codex, Grok Build and Grok Bot.

## Current state

Not kept here, so it cannot go stale: the current state is in [`docs/status/STATUS.md`](docs/status/STATUS.md) (what works, which solver, which character/scene profile and assets are in use), open problems with evidence in [`docs/status/TODO.md`](docs/status/TODO.md), history in [`docs/status/sessions/`](docs/status/sessions/). Keep those up to date in the same session as the change.

## R&D mode: the approach is a hypothesis

Nobody has proven that the current approach will reach the goal. Treat it as one candidate, not the answer. Alternative ways to get motion from video (for example DeepMotion, other video-mocap tools or models) are legitimate options to evaluate and compare. Proposing "this approach is a dead end, try X" is a valid outcome of work.

What is stable is the **goal** (source motion faithfully performed by Xandra) and the **diagnostic video set** (below). Any approach is judged by the same clips.

### Current approach (as implemented)

```
MP4 → observations → solver → AnimationBundle → Blender adapter → MP4
```

While working inside this approach, keep its design consistent:
- `AnimationBundle` is the motion contract Blender consumes; published bundles are immutable.
- Blender renders; tracking, solving, IK and fusion happen before the bundle.
- Profiles (character / scene) are versioned and bound to asset hashes.

Changing these is allowed when the evidence calls for it — but as an explicit, discussed decision, not a side effect.

### Always

- No clip-specific timecodes or hand-tuned per-clip constants in runtime code.
- Legacy projects (`clip02_blender_kit`, `clip_full`, `dance`) are research input only; never import them.
- Large assets stay outside Git and are registered in `.xms/config.json` (local, ignored).

## Diagnostic video set — `base video/`

Critical asset. Used both for **regression** (nothing that worked may break) and for **new checks** (each new capability is proven here first).

- Starts with eight minimal clips: one person, static camera, ~6 s, one simple motion each.
- Will grow over time: more complex motions, harder starting poses. Do not assume a fixed count or fixed content.
- A change is not "better" until it is compared on this set against the previous result.
- Never edit or delete clips in this set without Ilya.

## How we work

1. **Discuss before acting.** Agree on the goal, the "done" criterion and the "fail" criterion before changing code. No rushing into implementation.
2. **Your own result is not acceptance.** Always state which level was reached: *implemented* / *functionally verified* / *quality accepted*. A render that exits 0 proves nothing about motion quality.
3. **Call failure failure.** If the output looks wrong to a human, it is wrong, whatever the metrics say. Do not search for a metric that passes.
4. **Do not stack on an unproven foundation.** If a lower layer is not verified, do not add stages on top of it — verify the base first.
5. **Judge physical plausibility, not only fit to the source.** Centre of mass over support, pelvis/torso tilt, feet on the floor and not crossing, no body penetration. These cannot be gamed by matching 2D landmarks.
6. **Agreement in numbers is not correctness.** Python FK matching Blender only shows both sides compute the same thing; both may share the same wrong axis or rest-pose convention.
7. **Don't burn resources blindly.** No repeated full renders or benchmark campaigns without a concrete hypothesis.
8. Frozen benchmarks and acceptance thresholds (`benchmarks/acceptance.v1.json`) are not changed to make results pass.

## Environment

- macOS, Apple Silicon. Runtime in `.venv`; `./xms` refuses to run without it.
- Blender 4.5 LTS (EEVEE and Cycles), FFmpeg, MediaPipe Tasks — paths registered in `.xms/config.json`.
- Character/scene profile and `.blend` pair: change only through the registry (`./xms assets status|use|register`, Studio card «00 Персонаж и сцена»), never by editing the pair in `.xms/config.json`; profiles are immutable, a changed asset is a new version.
- Render engines, samples and timings: `docs/status/STATUS.md` → «Рендер». Prefer preview for experiments.
- Viggle: key only from `VIGGLE_API_KEY`; uploads are paid and external — ask Ilya before any upload; results are cached in `~/.cache/xms/viggle/`.
- Every run writes `runs/<id>/` (ignored by Git): video, GLB, bundle, observations, manifest, diagnostics, logs, compare, report. Frame PNGs are deleted after a successful encode; `./xms clean` handles older runs.
- Tests: `tests/` (unittest; pytest is not installed in `.venv`): `PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`. `tools/validate_base_videos.py` runs the eight diagnostic clips and keeps evidence without auto-acceptance.

## Where to look

- `docs/status/` — status, TODO with evidence, session logs (start here; the source of truth for current state).
- `README.md` — commands and current solver behaviour (most up to date).
- `docs/development/ARCHITECTURE.md`, `STAGES.md` — design and stage plan. **Status headers there are stale.**
- `docs/development/*HANDOFF*.md` — per-stage results; the `video` solver is not described in any handoff.
- `docs/research/` — analysis of the legacy Grok Bot pipelines.
- `base video/` — the diagnostic video set (see above).

## Git

- One logical change per commit with a clear message. No history rewriting.
- Do not commit renders or `runs/`. Assets (`.blend`) only through Git LFS and only when Ilya asks; textures packed into the `.blend` so a clone renders the same.
