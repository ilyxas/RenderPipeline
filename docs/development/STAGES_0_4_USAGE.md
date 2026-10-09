# Stages 0–4 checkpoint usage

This implementation supports a body-only `baseline_preview`. Later stages are
not implemented. Local binaries, models and source assets are explicitly supplied
in a JSON config (see `examples/jobs/local-config.example.json`). Profiles are
hash-bound to the registered Xandra and bedroom assets. A full local Git LFS object
can be used directly as the read-only character asset; `.blend` filename suffix is
not required by Blender.

The verified environment is `.venv` created for this repository, Python 3.13.15.
Exact tracking dependencies are recorded in `environments/stage0-macos-arm64.txt`.
Commands below use the package directly. Installing this package from `pyproject.toml`
provides the equivalent `xms` console entry point.

```sh
PYTHONPATH=src .venv/bin/python -m xms.cli bundle validate tests/fixtures/synthetic_bundle

PYTHONPATH=src .venv/bin/python -m xms.cli observe /path/sing.mp4 \
  --start 3 --end 5.5 --out runs/new-observations --config runs/local-config.json

PYTHONPATH=src .venv/bin/python -m xms.cli run /path/sing.mp4 \
  --character xandra --scene bedroom --quality preview --solver baseline \
  --start 3 --end 5.5 --config runs/local-config.json

PYTHONPATH=src .venv/bin/python -m xms.cli run --job /path/baseline-job.json

PYTHONPATH=src .venv/bin/python -m xms.cli render runs/RUN/animation \
  --fps 24 --out runs/render-only --config runs/local-config.json
```

`run` always creates a unique output directory and newly extracts observations.
It executes observe → solve → validate/publish bundle → encode preflight → render
→ audio/video assembly → full MP4 decode verification. A manifest and logs retain
hashes, paths, parameters, versions, timing, exit status and known limitations.
There is no database, scheduler, cache, resume, UI, export, or installer.

Render-only takes the saved bundle and registered character/scene. It does not
read the source MP4, run MediaPipe, or require the pose model path to exist.
`--frame-indices 0,30,59` selects diagnostic frames. It writes PNGs and its own
bundle-hash-bound manifest. FFmpeg assembly is independently callable through
`xms.assembly.encode`.

The published bundle directory contains `metadata.json` and `arrays.npz`.
NPZ is read with `allow_pickle=False`; publishing to an existing bundle path is
rejected. Matrices use metres, right-handed Y-up/+Z-forward coordinates and XYZW
quaternions. Local TRS uses `R_rest × R_delta`; scale remains on the right of the
rotation. World root offset applies exactly once to the pelvis subtree. Profile
hashes include rig, face map and look profile hashes. Blender checks both the
profile binding and actual rig arrays before applying animation.

The observation folder contains source probe data, an explicit timeline, raw
image/world landmarks with visibility/presence and masks, source PTS, crop/display
transforms and overlays. It contains no Xandra bone or morph data. World landmarks
remain tracker estimates. Multiple detections are rejected; robust identity and
tracking recovery are not claimed.

The baseline uses observed segment directions and registered rest translations.
Local rotation angle caps bound the solve. Head motion is a limited body prior.
Root translation uses a documented image-plane estimate with a clip-median origin;
root depth is unobserved. Face, gaze, jaw, wrists and finger deltas are neutral with
zero confidence and `unobserved` provenance. No temporal quality optimization,
contact solve, collision correction or clip-specific timestamp adjustment occurs.

Stage 2 diagnostics are in `runs/stage2/diagnostics`, with labelled body, wrist and
face review sheets. Tooth geometry remains canonical under `xandra_kit_natural_v1`.
The archival soft-lips tuck is explicitly a separate, unverified variant and is
not applied. Hair tint/skin material corrections are versioned and idempotent.

Verification entry points:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=src .venv/bin/python tests/integration/test_pose_observations.py runs/RUN/observations
PYTHONPATH=src .venv/bin/python tests/integration/test_vertical_slice.py runs/RUN
```

Stage 2 fixture creation and Blender verification are separate commands documented
in `tests/integration/test_blender_bundle.py` and `blender/apply_bundle.py --help`
(the latter runs under Blender). Diagnostic evidence is local and ignored by Git;
source code, schemas, profiles and the small synthetic bundle fixture are tracked
project deliverables.
