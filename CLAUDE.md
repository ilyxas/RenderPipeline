# CLAUDE.md — RenderPipeline (Xandra Motion Studio)

Read this first in every session. If anything here disagrees with the code, the code wins — then fix this file.

## What this is and why

An offline macOS pipeline: a source MP4 of a real person → the same motion performed by **Xandra** (a professionally built, anatomically detailed rigged character) → rendered MP4.

Why it exists: generative video gives a result but no control. Xandra gives full control over every bone and morph, so the same body can later do anything — paired acrobalance, an obstacle course, yoga lessons. RenderPipeline expands her motion vocabulary beyond hand-made clips. Results are also meant to flow back into HumanoidAlpha (bundle → animated GLB → Animation Mode); that link is not yet verified end to end.

This is an exploratory, for-fun research project. Owners: Ilya and Claude. Earlier work was done by Codex, Grok Build and Grok Bot.

## Current state (2026-10-11)

- Infrastructure works: `./xms run`, `./xms gui`, run folders, compare video, reports.
- **Core reconstruction is not proven.** On the first eight diagnostic clips (`base video/`) poses are wrong: tilted body, unstable stance, arms that do not reach targets.
- Observed pattern: in-plane motion (arms up/sideways, knee raise) transfers roughly; **depth motion and self-contact** (crossing arms, hands to face, head yaw, forward kick) fail or are attenuated.
- Next step: a synthetic `AnimationBundle` test — neutral pose plus one moving joint, no video — judged by meaning ("right arm forward 45°" looks like that), not only by numeric agreement. If it fails, the bug is after reconstruction; if it passes, look at observations / solving / retargeting.
- Default solver is `video` (see README). `temporal` has **not** passed acceptance; Stages 9–16 were built on top of it and are functional, not accepted.

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
- Blender 4.5 LTS (EEVEE), FFmpeg, MediaPipe Tasks — paths registered in `.xms/config.json`.
- Preview: 540×960, 16 samples. Final: 1080×1920, 64 samples. Prefer preview for experiments.
- Every run writes `runs/<id>/` (ignored by Git): video, bundle, observations, manifest, diagnostics, logs, compare, report.
- Tests: `tests/` (unittest; pytest is not installed in `.venv`): `PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`. `tools/validate_base_videos.py` runs the eight diagnostic clips and keeps evidence without auto-acceptance.

## Where to look

- `README.md` — commands and current solver behaviour (most up to date).
- `docs/development/ARCHITECTURE.md`, `STAGES.md` — design and stage plan. **Status headers there are stale.**
- `docs/development/*HANDOFF*.md` — per-stage results; the `video` solver is not described in any handoff.
- `docs/research/` — analysis of the legacy Grok Bot pipelines.
- `base video/` — the diagnostic video set (see above).

## Git

- One logical change per commit with a clear message. No history rewriting.
- Do not commit assets, renders or `runs/`.
