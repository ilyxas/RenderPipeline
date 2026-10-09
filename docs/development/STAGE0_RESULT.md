# Stage 0 result — passed (local LFS asset located)

Date: 2026-10-09. Stage 0 now passes. The original blocked evidence below is retained.

The full character was already present in the adjacent repository’s local Git LFS
cache. Its SHA-256 matches the pointer OID. Blender 4.5.13 loaded it and evaluated
the two-key pelvis action (maximum matrix element change 0.3461609). A new
`RenderPipeline/.venv` with Python 3.13.15, NumPy 2.5.3 and MediaPipe 1.1.0
produced 33 image and 33 world landmarks from the previously decoded frame.
The resolved runtime is recorded in `environments/stage0-macos-arm64.txt`.
No existing venv or canonical asset was changed; no asset download occurred.
Successful FFmpeg checks were reused.

Current evidence: [resolved-manifest.json](../../runs/stage0-20261009/resolved-manifest.json).
Proceed to Stage 1 under the approved scope. No architecture deviation.

## Original blocked pass (superseded)

## Implementation and evidence

Added `experiments/runtime_smoke.py` and its invocation documentation. The harness
accepts all paths explicitly, records hashes/versions/exit statuses, decodes one
frame, attempts one pose inference, tests a two-key Blender action in memory,
and exercises the preview encoding prerequisite. It never saves Blender assets,
reads GLB, activates existing venvs, or imports legacy pipeline code.

Evidence: [`manifest.json`](../../runs/stage0-20261009/manifest.json) and sibling
subprocess logs. Outputs are Stage 0 diagnostics, not a rendered character video,
observations contract, or AnimationBundle. No Stage 4 deliverable exists yet.

Selected binaries/runtime:

- `/opt/homebrew/opt/python@3.13/bin/python3.13`: Python 3.13.15, outside existing
  venvs; **not yet a compatible tracking runtime**, because dependencies are absent.
- `/Applications/Blender.app/Contents/MacOS/Blender`: 4.5.13 LTS,
  build `daeeeca98fb0`.
- `/opt/homebrew/bin/ffmpeg`, `/opt/homebrew/bin/ffprobe`: 9.0.2.
- Local pose model: `/Users/ilya/codex/dance/pipeline/models/pose_landmarker_heavy.task`.
  Found without reading existing venvs or GLB. No model download is necessary.

## Focused verification

- FFprobe succeeded: source H.264 video 720×1280, 24/1 FPS, 361 frames,
  video start 0; stereo AAC 48 kHz, audio start 0, duration 15 seconds.
- One frame decoded successfully to RGB24 with the expected 2,764,800 bytes.
- One synthetic 540×960 frame encoded successfully using libx264/yuv420p.
  This is a codec prerequisite check; it is not the Stage 4 output.
- NumPy and MediaPipe imports failed in Python 3.13.15. The bounded checks of
  system Python 3.9.6 and Homebrew 3.14.6 also found neither package. Pose output
  shape and inference compatibility remain unverified.
- Sandboxed Blender process exited -11 during startup. The required retry outside
  the sandbox exited 1 with: `Error: File format is not supported in file` at the
  character path. The two-key action never ran; rig, shape keys, inheritance,
  action API behavior, and evaluated transforms remain unverified.
- All four supplied input file hashes remained unchanged, including after the
  outside-sandbox retry. Existing venvs and GLB were not opened or changed.

## Blocking evidence and necessary fix

`/Users/ilya/codex/clip02_blender_kit/blend/Xandra_Animations.blend` is **134 bytes**
and contains a Git LFS pointer rather than Blender data:

```text
version https://git-lfs.github.com/spec/v1
oid sha256:b20530fe8acf51d659028beb1abb0eda13bf57ab4d3020dc95a068931885e801
size 161569909
```

The only asset fix needed is to make that real 161,569,909-byte character available
at an explicitly supplied read-only path. Options: provide an already hydrated
local copy, or explicitly authorize fetching this single LFS object into a separate
asset-store location. Do not replace it with GLB, precomputed animation, or a mock
rig. No asset download, archive modification, or alternate character investigation
was attempted.

Separately, create a **new** isolated environment and install only NumPy and
MediaPipe plus their required transitive dependencies. Establish compatible wheels
and record resolved versions before treating the environment as verified. No SciPy,
Demucs, audio separation stack, installer, or full final dependency stack is needed
for this smoke check. No packages were installed during this blocked pass.

## Input hashes

| Role | SHA-256 of currently supplied file |
| --- | --- |
| Character **pointer** | `88303cf2154ff9e2dac4176ac7fd4f5cc84f9dfb24ea1416986dcc5e6390791c` |
| Bedroom scene | `09da5370a2d72ae544fa5b9ac3c2b123f85392e4f67b00988e91a946f2f37fdc` |
| sing.mp4 | `20c16af4c286190b9a2e2fd33780f71000c885cf3c835a377f41dd6dae3ce8cb` |
| Pose task | `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b` |

The character pointer hash must not be confused with the real asset's LFS OID.

Recommended next step: resolve the canonical character availability and minimal
runtime setup, then run the blocked pose/Blender checks with evidence in a new run.
After Stage 0 passes, continue Stages 1–4 under the existing authorization, including
Stage 2 left/right limb, head, wrist, and facial visual diagnostics and the Stage 4
new-observations MP4 checkpoint. Stop for review after Stage 4.
