# RenderPipeline

RenderPipeline is the implementation repository for Xandra Motion Studio: an offline
macOS pipeline that turns a source MP4 of a person into the same motion performed by
Xandra, a rendered MP4 and an animated GLB.

```text
MP4 -> observations -> solver -> AnimationBundle -> Blender adapter -> MP4 (+ GLB)
```

`AnimationBundle` is the only motion contract consumed by Blender. Tracking, external
mocap, retargeting, limits and contacts happen before the bundle is published; Blender
only applies and renders it. Legacy projects (`clip02_blender_kit`, `clip_full`, `dance`)
are research inputs only and are never imported.

## Status in one paragraph

Infrastructure works end to end (CLI, GUI, run folders, comparison video, report, GLB).
Motion quality is **not accepted** for any solver. The default `video` solver loses
depth, self-contact and limb twist; the experimental `viggle` backend (external Viggle
mocap) is clearly better on the diagnostic set in side view and is the current choice for
looking at results — see [Viggle vs video](docs/research/VIGGLE_COMPARISON.md) and
[What is still broken](#what-is-still-broken).

## Run a new video

### Graphical launcher

Double-click **Xandra Studio.command**, or from the repository:

```sh
./xms gui
```

Pick a video (or paste its path under *Файлы и дополнительные настройки*), set start and
duration, resolution, quality and the reconstruction method, then **Run**. Progress shows
the current stage; buttons open the video, comparison, report, run folder and bundle.

- **Viggle · внешний мокап** shows whether the selected video is already in the Viggle
  cache (free) or needs a paid upload; only the *Разрешить платную отправку* tick allows
  the upload for that run. The key must be in the environment of the terminal that
  started the GUI (`export VIGGLE_API_KEY=…`).
- **Stop** ends the run and every Blender/FFmpeg process under it; the run folder is
  kept and its manifest says `cancelled`. Closing the launcher's Terminal window or
  pressing Ctrl+C there also stops an active run.
- The server listens only on 127.0.0.1 and needs no extra dependencies.

### Command line

```sh
./xms run input.mp4                                   # video solver, preview
./xms run input.mp4 --start 3 --end 8 --resolution 720x1280
./xms run input.mp4 --solver viggle --channels full --quality final --resolution 720x1280
./xms run input.mp4 --quality final --output ~/Movies/xandra.mp4
```

Omitting `--start/--end` processes the whole video. A successful render exits 0; motion
and visual acceptance are reported separately (`needs_review`). Ctrl+C, SIGTERM or
closing the terminal ends a run as `cancelled` (exit 130).

Useful options:

- `--solver video|viggle|temporal|baseline` — see [Solvers](#solvers).
- `--channels full|body` — `body` skips face and hand channels.
- `--quality preview|final`, `--resolution WxH` (even, 540×960 … 2160×3840).
- `--audio off|envelope|speech|singing` — lip cues from the soundtrack; `speech` needs
  Rhubarb and `singing` a Demucs runtime, otherwise they fall back to `envelope` with a
  warning. The original audio is always the soundtrack.
- `--face-closeup` — extra close-up video of the face from the same bundle.
- `--camera fit|fixed`, `--output PATH` (never overwrites), `--keep-frames`.
- `--surface-qa sampled|all|off`, `--no-refine` — hand–garment checks and one bounded
  refinement (diagnostic).
- `--allow-degraded-final` — render `final` even when the motion gate blocks it.
- `--job FILE` — `xms.baseline_job.v1` job file (see `examples/jobs`).

### What a run produces

Every run gets `runs/<id>/`:

- `outputs/video.mp4` (with the source audio), `outputs/animation.glb`,
  optional `outputs/face.mp4`, `comparison/compare.mp4` (source next to Xandra);
- `animation/` — the immutable AnimationBundle; `observations/`, `manifest.json`,
  `solve-diagnostics.json`, `logs/`, `report/report.html`.

`animation.glb` is the same bundle as an animated GLB: every render-visible mesh skinned
to the armature, bones and ARKit face morphs in one animation `XMS_Bundle`, checked
against bundle FK (`glb_export` in the manifest). Its failure is a warning only.

PNG frame sequences are encoder input only and are deleted after a successful encode
(`--keep-frames` keeps them; failed and cancelled runs keep theirs). For older run
folders:

```sh
./xms clean            # dry run: what would be freed
./xms clean --apply    # move frame PNGs to the Trash (results stay)
```

### Render settings

Preview: 540×960, 16 EEVEE samples. Final: 1080×1920 by default, 32 samples, preceded by
a whole-clip 540×960 preview and codec/timing checks. Shadows use Blender's defaults
(1 ray, 6 steps) instead of the asset's 4/16 — same look, about 2.8× faster overall
([measurements](docs/research/RENDER_SETTINGS.md)). On an M4 MacBook Air the final pass of
a 15 s clip at 720×1280 takes about 7 minutes, plus about 6 for the whole-clip preview.

## Solvers

**`viggle`** (experimental, R&D) retargets an external Viggle Motion 3D extraction
(MetaHuman-body rig) onto Xandra by rotations. Body and head come from Viggle;
`--channels full` adds Viggle fingers and the video face pipeline's ARKit morphs and
audio lip cues. Feet stay on the floor every sample (camera drift in the source reads as
root drift). Arm limits are reported, not applied.

- The key is read only from `VIGGLE_API_KEY` and never stored. A source video is
  uploaded once — the whole file, about $0.05 per second — and the result is cached by
  content hash and fps in `~/.cache/xms/viggle/` (`XMS_VIGGLE_CACHE` to override).
- A cache miss uploads to an external paid service and therefore fails unless
  `XMS_VIGGLE_ALLOW_UPLOAD=1` is set (the GUI tick does this for one run). Deleting the
  cache means paying again for those videos.

**`video`** (default) uses calibrated image-plane joint trajectories, a smoothed
bone-length depth prior and one world-to-local rotation conversion across the rig. A
neutral interval (frames whose 2D limb directions match Xandra's rest pose) calibrates
per-bone offsets; `XMS_NEUTRAL_CALIBRATION=0` disables it for comparison. Palm
orientation uses the same interval; distal finger joints are hinges and knuckles have
bounded splay. After the solve, arm limits from HumanoidAlpha's joint data apply (elbow
hinge, two-axis wrist, forearm roll within ±90° with out-of-range tracker samples
bridged), and arm roll is distributed over the skinned twist bones. Rest-derived sole
proxies align to Y=0 on supported frames, with leg IK on short support intervals.

**`temporal`** adds experimental overlapping body optimization and contact/collision
costs; **`baseline`** is the preserved first backend. Neither has passed acceptance.

## Characters and assets

The local `.xms/config.json` (ignored by Git) registers Blender, FFmpeg, the MediaPipe
models, the character asset and profile, and the bedroom scene. Profiles are bound to the
asset's SHA-256; a different `.blend` is refused until it is registered.

| Profile | Asset | Look |
|---|---|---|
| `profiles/characters/xandra/v3` | `blender/Xandra_Animations.blend` (Git LFS) | blue top, black shorts |
| `profiles/characters/xandra/v4` | `Xandra_BlackLace_Fitting_v01.blend` (local, outside Git) | Black Lace fitting prototype |

Both use the identical 341-bone skeleton, so bundles, arm limits and the Viggle map apply
to either (they are bound to `rig.npz`, not to the asset file). To switch, point
`character_asset` and `character_profile` in `.xms/config.json` to the pair you want.
The bedroom scene `clip02_scene.blend` is also a local asset outside Git.

## What is still broken

- **No arm–arm or arm–body collision.** Crossed arms intersect; surface QA only samples
  hand–garment pairs and is reported as unavailable on open garments.
- **Viggle estimate errors pass through**: e.g. the knee raise over-flexes the knee
  (146°); the retarget copies Viggle's rotations as they are.
- **Jumps are not modelled** (floor support assumes a planted foot); slopes, stairs and
  long occlusions are unsupported. The scene has no props (a table the person leans on
  becomes air).
- **`video` solver**: depth is a guess (raised arms drift forward), hand tracker palm
  flips survive into wrist pose, shoulder roll follows a noisy bend plane on straight
  arms, and the right knee can twitch when contact switches.
- Shoulder/hip anatomical limits are not ported; arm limits are not applied to Viggle.
- `--audio speech/singing` need Rhubarb / Demucs, which are not installed here.
- The animated GLB does not yet contain the eight clips HumanoidAlpha's Animation Mode
  expects; it plays there as a one-shot clip.
- Final still renders a whole-clip preview first (~15–20 % of final time).

## Setup on another machine

```sh
python3.13 -m venv .venv          # pyproject requires >=3.11; 3.13 is the verified runtime
.venv/bin/python -m pip install -r environments/stage0-macos-arm64.txt scipy==1.18.1
.venv/bin/python -m pip install -e '.[tracking,temporal]'
mkdir -p .xms && cp examples/local-config.example.json .xms/config.json
# Edit the absolute paths in .xms/config.json (assets, models, Blender 4.5, FFmpeg).
./xms run "base video/<clip>.mp4"
```

`--config PATH` or `XMS_CONFIG` select another configuration. No models or large assets
are downloaded automatically; source assets are read-only.

Tests: `PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`.

## Repository layout

- `src/xms/` — Python package by pipeline boundary (`observations`, `solve`, `animation`,
  `render`, `export`, `qa`, `gui`, `clean`).
- `blender/` — Blender process entry points (render, GLB export, surface checks); no solving.
- `profiles/` — versioned character/scene registrations, arm limits, Viggle map.
- `base video/` — the diagnostic clip set every change is compared on.
- `benchmarks/` — frozen acceptance policy and measured baselines (not edited to pass).
- `docs/research/` — findings: Viggle comparison, render settings, legacy reports.
- `docs/development/` — architecture and stage handoffs (stage status headers are historical).
- `schemas/`, `examples/`, `environments/`, `experiments/`, `tests/`.
