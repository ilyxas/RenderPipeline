# Experiments

This directory is reserved for bounded experiments named by a stage. Stage 0 may add `runtime_smoke.py` and its result. An experiment must state its question, input, observable result, and stop condition before code is added.

## Stage 0 runtime smoke

Question: can this Mac decode the source, infer one pose, and evaluate a newly
created two-key action on the canonical rig without changing source assets?
Inputs: explicit character/scene/video/pose-model and binary paths. Run with a
compatible Python outside all existing venvs. No downloads or installation occur.
Observable result: a JSON manifest with hashes, versions, exit statuses, one RGB
frame, pose landmarks when available, and Blender transforms/morph/action evidence.
Stop condition: one focused pass; missing dependencies/models block Stage 1.
No clip rendering, legacy animation ingestion, or asset saving is performed.

```sh
python experiments/runtime_smoke.py \
  --character /path/Xandra_Animations.blend --scene /path/clip02_scene.blend \
  --video /path/sing.mp4 --pose-model /path/pose_landmarker.task \
  --blender /Applications/Blender.app/Contents/MacOS/Blender \
  --ffmpeg /opt/homebrew/bin/ffmpeg --ffprobe /opt/homebrew/bin/ffprobe \
  --out runs/stage0-unique-id
```

The output directory must be new. Exit 0 means all prerequisites passed; exit 1
means blocked. `manifest.json` and individual subprocess logs retain the evidence.

