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

## Run a new video

### Graphical launcher

On macOS, double-click **Xandra Studio.command** in the repository, or run:

```sh
./xms gui
```

The local interface opens in your browser. Select a video through the system file
dialog, then choose the start time and duration (5 seconds by default; **Весь ролик**
selects the remaining video). You can preview and seek the source before running.
Resolution (540×960 through 2160×3840) and render quality are independent.
The interface exposes solver selection and every Useful option below, including
body-only processing, surface checks, refinement, audio cues, camera mode and face
close-up. Optional output and asset configuration paths are under additional settings.

**Run** starts the existing pipeline. Progress shows actual completed frames for
tracking and rendering; stages without frame counts show an indeterminate bar.
**Stop** terminates the CLI and its Blender/FFmpeg descendants; partial files and
logs remain in the run folder. Buttons open the video, comparison, report and bundle.
Keep the launcher terminal open; Ctrl+C shuts down the server and any active run.
The server listens only on 127.0.0.1 and requires no new dependencies.

Custom resolution also works from the command line:

```sh
./xms run input.mp4 --start 0 --end 5 --quality preview --resolution 720x1280
```

### Command line

The current machine's Xandra, bedroom, tracker models, Blender and FFmpeg are
registered in `.xms/config.json` (local, ignored by Git). No intermediate commands
are required. From this repository:

```sh
./xms run input.mp4 --character xandra --quality preview
./xms run input.mp4 --start 3 --end 8 --output runs/my-xandra.mp4
./xms run input.mp4 --solver temporal --quality preview --face-closeup
./xms run input.mp4 --quality final --output runs/my-final.mp4
```

For the exact `xms` command, add this repository to your shell path:

```sh
export PATH="$PWD:$PATH"
xms run input.mp4 --character xandra --quality preview
```

Omitting `--start`/`--end` processes the whole video. Every run gets a unique
`runs/<id>/` directory with `outputs/video.mp4`, immutable `animation/`, observations,
`manifest.json`, solver diagnostics, logs, comparison video and an HTML report.
`outputs/animation.glb` is the same bundle as an animated GLB (skinned Xandra, bones and
ARKit face morphs in one animation `XMS_Bundle`), exported by `blender/export_glb.py`
from the read-only canonical asset and checked against bundle FK (`glb_export` in the
manifest). Its failure is a warning and does not discard the MP4.
`--output` copies the main MP4 to a chosen path and refuses to overwrite it.
The terminal prints both MP4 and bundle paths. A successful render exits 0;
motion/surface/visual acceptance is reported separately and can remain `needs_review`.

`video` is the default: calibrated image-plane joint trajectories, a smoothed
bone-length depth prior and one world-to-local rotation conversion across the rig.
A neutral interval is used only when observed limb directions match the rest pose.
Palm orientation uses that same calibration interval; observed phalanx directions
are transferred inside the solved palm frame. Distal finger joints use hinges;
knuckles have bounded splay (15°, thumb 40°). Anatomical limits and rate corrections
are recorded separately. Clavicles share arm elevation, following the working
legacy retarget rather than keeping the shoulder fixed.
Elbow bend planes and forearm axial orientation preserve the solved joint positions.
No second arm IK or palm
flip correction runs after this solve. Calibration and degraded assumptions are
recorded in `image_retarget_diagnostics` and `hand_diagnostics`. Ambiguous palm and
finger estimates have declared angular rate fallbacks (900°/s and 1000°/s);
correction counts appear in diagnostics and the pipeline log. These can degrade
very fast hand articulation, so a smooth render is not visual acceptance.

Rest-derived sole proxies align to Y=0 on supported frames; short low-speed
support intervals use leg IK while the free leg stays raised. Root correction is
bounded to 45cm. Missing feet use a 250ms hold and then remain uncorrected.
Shared image translation is **not** sufficient evidence of flight: automatic
jump recovery is currently unsupported and the floor prior can suppress jumps.
This assumes an upright visible person and flat floor. Slopes, stairs, moving
cameras and long occlusions remain unsupported; skin sole geometry is approximate.

Monocular depth and complex occlusion remain estimates; this is not broad visual acceptance. `--solver baseline` uses the
preserved baseline backend. `--solver temporal` adds experimental overlapping body
optimization and contact/collision costs; it uses SciPy, reports exhausted-window
baseline fallbacks, and has **not passed motion-quality acceptance**. Version 3
profiles bound optimizer increments to 5° and 3cm; this helps retain the initializer's
fast movement but does not establish reconstruction accuracy.

`--solver viggle` (experimental, R&D comparison only) retargets an external Viggle
Motion 3D extraction (MetaHuman-body rig) onto Xandra by rotations. Body and head come
from Viggle; `--channels full` adds Viggle fingers and the video face pipeline's ARKit
morphs and audio lip cues, `--channels body` leaves fingers and face neutral. Feet are
kept on the floor every sample (camera drift in the source reads as root drift); jumps
are not modelled. The key is read only from `VIGGLE_API_KEY`. Each source video is
extracted once and cached by content hash and fps in `~/.cache/xms/viggle/`
(`XMS_VIGGLE_CACHE` to override); a cache miss uploads the whole video to a paid external
service and therefore fails unless `XMS_VIGGLE_ALLOW_UPLOAD=1` is set. Arm limits are
reported, not applied. It has **not passed motion-quality acceptance**.

Useful options:

- `--allow-degraded-final`: explicitly override the catastrophic motion gate.
  Without it, large rotation jumps or deep collisions stop `final` before rendering;
  the AnimationBundle and diagnostics remain available. Preview remains available.
- `--channels body`: skip face/hand tracking for a faster body experiment.
- `--surface-qa sampled` (default), `all`, or `off`: evaluated hand–garment checks.
  Sampled coverage is diagnostic. One bounded refinement can be attempted;
  `--no-refine` keeps the initial reconstruction. Unresolved defects are retained.
- `--audio off|envelope|speech|singing`: envelope is a low-confidence energy cue.
  Speech optionally uses Rhubarb; singing optionally uses a separate Demucs runtime.
  Missing/failing backends fall back and log warnings. Original audio is the soundtrack.
- `--camera fixed`: keep the registered camera; default `fit` frames the whole
  skeletal trajectory with a stable FOV. The solver calibration camera is unchanged.
- `--face-closeup`: optional same-bundle face video; its failure does not discard
  the main MP4. Diagnostic views do not run by default.

Preview is 540×960/16 EEVEE samples. Final is a **render-quality candidate** at
1080×1920/64 samples, preceded by a whole-clip preview and codec/timing verification;
these sample counts have not passed flicker/artistic acceptance. Existing bedroom
lighting, material and exposure defaults are preserved.

## Setup on another machine

Use a new compatible runtime, the recorded dependency locks in `environments/`,
and your own registered asset/model paths. Do not reuse protected legacy environments
or animated GLB files. The package provides a standard console script after install:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[tracking,temporal]'
mkdir -p .xms
cp examples/local-config.example.json .xms/config.json
# Edit the absolute paths in .xms/config.json to your registered assets and binaries.
./xms run input.mp4 --quality preview
```

You can also pass `--config /absolute/path/config.json` or set `XMS_CONFIG`.
No models or large assets are downloaded automatically. Source assets are read-only.
Missing input, model, asset or codec errors identify the path or log to fix.

## Working and experimental features

Working: source PTS/segment mapping, body/face/hand observations, baseline reconstruction,
AnimationBundle contracts, Blender adapter, original soundtrack, preview encoding,
comparison/report, default asset configuration and optional close-up. New policies
have focused functional checks and one fresh end-to-end demonstration.

Experimental/partial: temporal motion quality, monocular contact/floor estimates,
proxy collision correction, bilateral palms, audio articulation and bounded surface
refinement. Surface QA covers declared hand–top/shorts pairs, not the whole character;
hair, finger–finger and other skin intersections remain unchecked. Long tracking gaps
fall back to neutral; arbitrary occlusion, kneeling and complex posture recognition
are outside scope. Long/full-resolution videos use temporary disk-backed RGB decoding
and can require substantial disk/render time. No Stage 17–21 cache, scheduler or resume.

[Stages 9–16 record](docs/development/STAGES_9_16_HANDOFF.md) distinguishes functional
verification from acceptance and records the remaining work. Frozen benchmark
thresholds and version 2 profiles remain unchanged. The [Stage 8 handoff](docs/development/STAGE8_HANDOFF.md)
retains its failed gesture/continuity evidence.

Representative demonstration on this machine:

```sh
./xms run /Users/ilya/codex/clip02_blender_kit/input/sing.mp4 \
  --start 2 --end 4.5 --quality preview --face-closeup \
  --output runs/stage16/xandra-demo.mp4 --runs-root runs/stage16/jobs
```

The demonstrated output is [Xandra MP4](runs/stage16/xandra-demo.mp4).
Use a different output filename to repeat it.
