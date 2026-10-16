# altline — isolated alternative pipeline (R&D)

A from-scratch 2D-video → 3D-skeleton motion retargeting pipeline for a static camera.
It does not read or import anything from `src/xms/`, `blender/` or `tools/`; every stage
lives here and hands off to the next one through files only.

Status: **implemented and functionally verified on synthetic data** (exact round trip, see
tests). Motion quality on real video is **not accepted**; see `reports/` for the reviewer's
findings per run.

```
MP4 ─► observations.py ─► calibration.py ─► solver.py ─► limits.py ─► animation.py ─► motion_bundle.json
        raw_pose.json      actor_profile.json  raw_rotations  clean_rotations             (+ preview.py)
                           calibrated_pose.json
```

## Run

```bash
python3 -m pip install mediapipe opencv-python-headless matplotlib numpy   # once
python3 -m altline.run "base video/<clip>.mp4" --run-id my_run              # all stages + preview
python3 -m altline.observations "<clip>.mp4" --run-id my_run                # or stage by stage:
python3 -m altline.calibration --run-id my_run
python3 -m altline.solver      --run-id my_run
python3 -m altline.limits      --run-id my_run
python3 -m altline.animation   --run-id my_run
python3 -m altline.preview     --run-id my_run
python3 -m unittest discover -s altline/tests -t . -p 'test_*.py'
```

The MediaPipe Pose Landmarker model (`pose_landmarker_full.task`) is downloaded once to
`~/.cache/altline/` (or set `ALTLINE_POSE_MODEL`). On headless Linux MediaPipe needs
`libegl1`/`libgles2`. Runs go to `altline/runs/<run_id>/` (ignored by Git); reviewer
reports go to `altline/reports/<run_id>/` (committed).

## Stages

| Stage | File | Output | What it does |
|---|---|---|---|
| 1 | `observations.py` | `observations/raw_pose.json` | MediaPipe Pose (Tasks API, VIDEO mode). Per frame `joints` = world landmarks in meters (MediaPipe axes, raw), `image_joints` = normalized image coords. Format `{"frames": [{"frame_index", "joints": {name: [x, y, z, visibility]}}]}` plus `meta`. |
| 2 | `calibration.py` | `calibration/actor_profile.json`, `calibrated_pose.json` | Converts to canonical space; finds the rest-pose baseline (straight-ish limbs, symmetric abduction, minimal motion); actor segment lengths (median), target/actor ratios; rest alignment (upright, facing +Z); head/neck reference pitch; elbow/knee *hinge bias*. Per-frame `length_ratio` keeps the actor's length variance as a diagnostic only. |
| 3 | `solver.py` | `solver/raw_rotations.json` | Analytic aim + twist per bone, root to leaf; output local basis quaternions. Uses only unit directions, so rig bone lengths stay constant. |
| 4 | `limits.py` | `limits/clean_rotations.json`, `limits_report.json` | Swing-twist limits: elbow hinge 0..140°, forearm roll ±90°, wrist ±90°, knee hinge 0..150° (twist ±30°). Spikes/flips/missing frames are bridged with slerp; no frame is dropped. |
| 5 | `animation.py` | `animation/motion_bundle.json` | `{format, fps, bones, rotation_mode, frames: [{t, q: [[w,x,y,z] per bone]}]}`. |
| – | `preview.py` | `preview/preview.mp4` | Source | front | side; rig bones vs tracker landmarks (grey dots). |

All thresholds: `config.py`. No per-clip constants anywhere.

## Conventions

- **Canonical space** (after calibration): +X actor left, +Y up, +Z actor forward, meters.
  MediaPipe world → canonical: `(x, -y, -z)`, then the rest alignment rotation.
- **Bone frames** (anatomical): Y along the bone, Z = anatomical front (forward for trunk,
  arms, legs; up for feet), X = Y × Z. Elbow flexion = +rotation about the forearm's X;
  knee flexion = −rotation about the shin's X.
- **Pose model** (Blender's): `pose_world[b] = pose_world[parent] @ rest_local[b] @ basis[b]`,
  `rest_local[b] = rest[parent]^T rest[b]`. `basis` is what the bundle stores, i.e. what
  Blender keeps in `pose_bone.rotation_quaternion`.

## Decisions worth knowing

- **Absolute directions, relative scale.** The solver aims each bone at the observed
  direction; calibration only removes the actor's scale, the camera's tilt/yaw, and two
  tracker biases measured in the rest baseline. It does *not* force the rig's rest pose onto
  the actor's first pose.
- **Hinge bias.** MediaPipe reports 20–35° of elbow and knee bend for visibly straight
  standing limbs. When a rest baseline is found, that bend is treated as tracker bias and
  removed by remapping the bend `[bias, 180] → [0, 180]`, keeping the chord
  (shoulder→wrist, hip→ankle) and the bend plane. Disable with
  `CalibrationConfig.hinge_bias_correction = False`.
- **Twist sources.** Upper arm / thigh twist come from the elbow / knee bend plane, weighted
  by how bent the limb is (`SolverConfig.bend_twist_*`): a straight limb has no bend plane,
  so its twist is inherited. Forearm and hand twist come from pinky→index; shin twist from
  heel→toe. The pose model gives only one hand-plane reference, so wrist twist relative to
  the forearm is not observable separately (it is all forearm roll).
- **No root translation in v1.** MediaPipe world landmarks are hip-centred; global travel
  (jumps, steps) is not observed.

## Replacing the reference rig

`skeleton.reference_skeleton()` is a simplified 20-bone A-pose rig (≈1.68 m). The real
Xandra rig plugs in as data, without code changes:

1. Export from the real armature a `rig.npz` with, for the 20 driven bones under the
   altline names (`pelvis, spine, chest, neck, head, clavicle_l/r, upper_arm_l/r,
   forearm_l/r, hand_l/r, thigh_l/r, shin_l/r, foot_l/r`):
   `names` (N,), `parents` (N,) int with −1 for the root and parents before children,
   `heads`, `tails` (N, 3) in canonical space (+X left, +Y up, +Z forward; from Blender's
   Z-up/−Y-forward armature space: `canon = (x, z, −y)`), and optionally `rest` (N, 3, 3):
   the rig's own bone frames (Blender `bone.matrix_local` rotation, columns = bone axes, in
   canonical space).
2. Run with `--rig rig.npz`. Lengths come from the rig; anatomical frames are rebuilt from
   heads/tails, and when `rest` is present the per-bone offset `O = anat^T rest` converts
   every quaternion to the rig's own bone rolls (`basis_rig = O^T basis O`), covered by
   `tests/test_pipeline.py::EndToEnd::test_rig_with_other_bone_rolls`.
3. Bones of the 341-bone rig that altline does not drive keep identity rotation. Where the
   real rig splits a segment (e.g. several spine bones, twist bones), the export picks one
   bone per altline segment, or the split is a later, explicit step.
