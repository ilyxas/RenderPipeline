# Stages 0–4 engineering handoff

Status: **Stage 4 architecture checkpoint passed; stopped for review.**
Result quality: `baseline_preview`, not production acceptance. No Stage 5+ work.
Date: 2026-10-09. Real run: `runs/20261009-175723-14b81716`.

## Delivered artifacts

- [Rendered MP4](../../runs/20261009-175723-14b81716/outputs/video.mp4)
- [AnimationBundle archive](../../runs/20261009-175723-14b81716/outputs/AnimationBundle.zip)
- [New observations, timeline and overlays](../../runs/20261009-175723-14b81716/outputs/observations.zip)
- [Run manifest](../../runs/20261009-175723-14b81716/manifest.json)
- [Known limitations](../../runs/20261009-175723-14b81716/limitations.md)
- [Video review frames decoded from MP4](../../runs/20261009-175723-14b81716/outputs/video-review.jpg)
- Stage 2 visual diagnostics: [body](../../runs/stage2/diagnostics/body-review.jpg),
  [wrists](../../runs/stage2/diagnostics/wrists-review.jpg),
  [head and facial morphs](../../runs/stage2/diagnostics/face-review.jpg).

## What was implemented

Stage 0: bounded smoke harness, isolated repository runtime, source/model hashes,
pose inference and Blender two-key action evidence. The full character existed in
the adjacent repository's local LFS cache; the initial report missed that cache.
It is now used directly by configured path, without download or source modification.

Stage 1: executable versioned AnimationBundle, one immutable-publication writer,
independent reader/validator, TRS FK and quaternion sampling, schema, CLI and small
synthetic fixture. No runtime dependency on historical scripts.

Stage 2: Xandra registration (341 bones, 51 semantic face channels), hash-bound
character/scene profiles, Blender consumer, idempotent material look profile and
17 numerical/visual diagnostics. The right-smile alias is explicit. Canonical
natural teeth are retained; the historical soft-lips geometry change is not enabled.

Stage 3: source probing, rational timeline, actual PTS checks against decode,
sequential MediaPipe pose extraction, raw observation contract and independent
reader, labelled overlays. World landmark estimates are not treated as ground truth.

Stage 4: bounded segment-direction body solver with fixed rig proportions, root
estimate and head prior; sequential CLI run, Blender EEVEE frames, H.264/AAC assembly,
JSON provenance manifest, full MP4 decode verification and independent render-only.

## Which contracts were verified

- Metadata/NPZ round trip preserves arrays; invalid hierarchy, NaN, nonmonotonic
  time, nonunit quaternions, conflicting ownership, invalid ranges, unsupported
  versions and profile mismatch are rejected. Published paths cannot be overwritten.
- Analytical rest/root/90° FK and SLERP pass. Nonidentity armature TRS and nonuniform
  rest-scale order are checked. Rotation uses `R_rest × R_delta`; scale follows it.
- On the actual rig (armature scale approximately 0.01), Python FK and evaluated
  Blender matrices agree within **0.00201 mm** position and **0.0000249°** rotation.
  Morph weight error is at most **1.20e-8**. Root motion applies once. Head rotation
  and jaw morph diagnostics are isolated; left/right arms, legs, wrists, blinks and
  smiles are visually distinct. Numerical evidence includes every registered bone.
- Timeline fixtures cover 24, 30 and 30000/1001 FPS plus nonzero video/audio starts.
  Empty detections have invalid masks and zero confidence. Actual decoded PTS match
  the source timeline. Observations read independently without importing MediaPipe.
- The real run newly extracted **60/60 valid pose frames**, source interval
  **[3.0, 5.5)**. The solver bundle records their exact NPZ hash; the render records
  the same bundle hash. No archived animation was used.
- MP4 fully decodes: **60 frames, 540×960, 24 FPS, 2.500 s**, with source audio
  beginning at output 0 and lasting 2.500 s. Duration error is zero at reported
  precision, within the one-frame tolerance. Run elapsed time: **52.02 s**.
- Independent render-only used the saved bundle, no source-video argument and a
  nonexistent pose-model path. Frames 0, 30 and 59 were **pixel-identical**. PNG file
  hashes differ solely because Blender embeds Date/RenderTime metadata.
- Character, scene, source MP4 and pose model hashes remained unchanged.

The FK scale-order clarification prompted only the affected mathematical tests and
numerical Blender checks to be repeated. Diagnostic stills were not rerendered.
Other successful checks were not repeatedly run.

## What actually works

One CLI command creates new observations, an executable bundle and a viewable,
audible MP4 of Xandra moving in the bedroom. Arm gestures, head movement, root
motion and changing leg stance are visible in frames decoded from the actual MP4.
The same saved bundle can be rendered independently of ingestion/tracking. Numerical
and visual diagnostics establish local-rest rotation, coordinate conversion and
representative facial controls before tracking is connected to them.

## What does not work yet

- Face, jaw, gaze, wrists and fingers remain neutral/unobserved in the MP4. There is
  no singing articulation or lip sync; Stage 2 facial motion is synthetic diagnostics.
- No temporal solver, contact/foot anchoring or collision correction. Foot sliding,
  height variation, pose jitter and implausible overlap can occur.
- Monocular depth/root translation are approximate. Root depth is fixed and marked
  unobserved in calibration; head motion is a low-confidence body prior.
- No robust occlusion recovery, multiple-person identity tracking or shot handling.
- Wide gestures approach/cross the fixed camera boundary. Hair and shadows retain
  preview sampling noise. These quality limitations were recorded, not tuned away.
- Full silent/VFR/rotated-media integration QA belongs to Stage 5 and has not been
  performed. Only the Stage 3 timing unit fixtures and this actual source were checked.
- No database, cache, resume, UI, export, installer or later-stage infrastructure.

## Deviations from approved architecture

None. The LFS-cache asset path is a local configuration choice. The supported look
is explicitly versioned `xandra_kit_natural_v1`; no canonical geometry is modified.
The overall Stage 0–4 boundaries remain MP4 → observations → solver → AnimationBundle
→ Blender/render. Runtime and paths are documented in [usage](STAGES_0_4_USAGE.md).

## Recommended next step

Review the MP4 and Stage 2 diagnostic sheets, then explicitly approve Stage 5 if
satisfied with the architecture checkpoint. Stage 5 should add the planned timing
matrix and comparison/report evidence. Motion-quality optimization remains deferred
until its approved stages. **Do not proceed beyond Stage 4 without review.**
