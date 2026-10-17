# altline — isolated alternative pipeline (R&D)

A from-scratch 2D-video → 3D-skeleton motion retargeting pipeline for a static camera.
It does not read or import anything from `src/xms/`, `blender/` or `tools/`; every stage
lives here and hands off to the next one through files only.

Status: **implemented and functionally verified on synthetic data** (exact round trip, see
tests). v2/v3 = the reviewer's v1/v2 action items; v4 = Phase 2 (Butterworth filter, flip
guard, temporal solver, acceleration damper, Viterbi depth signs). Motion quality on real
video is **not accepted**; see `reports/` for the reviewer's findings per run.

```
MP4 ─► observations ─► filters ─► calibration ─► flip_guard ─► solver ─► solver_temporal ─► limits ─► damper ─► animation
       raw_pose        filtered_   actor_profile  guarded_pose  frame_     raw_rotations      clean_    damped_   motion_bundle
                       pose        calibrated_pose              rotations                     rotations rotations (+ preview)
```

## Run

```bash
python3 -m pip install mediapipe opencv-python-headless matplotlib numpy   # once
python3 -m altline.run "base video/<clip>.mp4" --run-id my_run              # all stages + preview
python3 -m altline.observations "<clip>.mp4" --run-id my_run                # or stage by stage:
python3 -m altline.calibration --run-id my_run       # reads filtered_pose.json when present
python3 -m altline.flip_guard  --run-id my_run
python3 -m altline.solver      --run-id my_run
python3 -m altline.solver_temporal --run-id my_run
python3 -m altline.limits      --run-id my_run
python3 -m altline.damper      --run-id my_run
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
| 1b | `filters.py` (Task 6) | `observations/filtered_pose.json`, `filter_report.json` | Zero-phase Butterworth (`filtfilt`, 2nd order per pass) on every joint's world and image track. Per-joint cutoff by Winter's residual analysis, clamped to a band per group: core 2–6 Hz, elbows/knees 2.5–6 Hz, distal (wrists, fingers, ankles, feet) 2–5 Hz. Visibility is not filtered. |
| 2 | `calibration.py` (+ `depth.py`) | `calibration/actor_profile.json`, `calibrated_pose.json` | Converts to canonical space; finds the rest-pose baseline; actor segment lengths; **rebuilds depth from 2D points + constant lengths** with depth signs chosen over the whole clip (Viterbi); (One-Euro filter, off since Phase 2); rest alignment (upright, facing +Z); rest biases of head frame, neck pitch, foot pitch; elbow/knee *hinge bias*. Per-frame `length_ratio` of the raw tracker is kept as a diagnostic. |
| 2b | `flip_guard.py` (Task 7) | `calibration/guarded_pose.json`, `flip_report.json` | A limb vector (upper arm, forearm, hand, palm normal; thigh, shin, foot, heel) turning > 45° from the last clean frame marks a corrupted run; runs up to 8 frames that end are replaced by a cubic Hermite spline through ±3 clean neighbours (on joint offsets, so the limb root stays put). Longer jumps are kept as real motion and reported. |
| 3 | `solver.py` | `solver/frame_rotations.json` | Analytic aim + twist per bone, root to leaf, every frame on its own; output local basis quaternions. Uses only unit directions, so rig bone lengths stay constant. |
| 3b | `solver_temporal.py` (Task 8) | `solver/raw_rotations.json`, `temporal_report.json` | Per bone and frame t, window t−2..t+2 in the tangent space at the per-frame solution: minimise Σ c_k‖x_k − y_k‖² + λ Σ‖x_{k−1} − 2x_k + x_{k+1}‖² (λ = 2, c = solver confidence), keep the centre. Constant angular velocity passes unchanged. |
| 4 | `limits.py` | `limits/clean_rotations.json`, `limits_report.json` | Swing-twist limits: elbow hinge 0..140°, forearm roll ±90°, wrist ±90°, knee hinge 0..150° (twist ±30°). Clamped root to leaf, children keep their world orientation; excess roll of a straight limb goes to the upper arm / thigh. Spikes/flips/missing frames are bridged with slerp; no frame is dropped. |
| 4b | `damper.py` (Task 9) | `damper/damped_rotations.json`, `damper_report.json` | Per bone, a follower with angular acceleration soft-capped at the stability gate (3000°/s²): untouched below 0.7×gate, exponential approach to the gate above it; catch-up speed planned so it can brake (no ringing); forward and backward passes averaged. **The output is under the gate by construction**, so the report keeps the input acceleration, frames damped and the deviation from the input. |
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
- **Depth from 2D + lengths.** MediaPipe flattens limbs that point at the camera. For
  segments that lie in the image plane at rest (trunk, widths, clavicles, neck, ears, arm
  and leg segments) depth is rebuilt as `|dz| = sqrt(L² − d²)` (weak perspective). `L` is the
  segment's projection in the rest baseline (v2's 98th percentile over the clip invented
  depth). The sign is chosen per segment over the whole clip by Viterbi: disagreeing with
  the tracker costs |tracker dz|/L per frame, a depth jump costs 10×|Δdz|/L. A rigid segment
  changes sign only through the image plane, where switching is free, so tracker sign
  flips of a few frames are ignored. Within ~14° of the image plane that formula is
  ill-conditioned, so the tracker's own depth is kept there (blended out by 32°). Forward
  pointing pieces (nose, fingers, heel, toes) keep the tracker depth.
- **Hinge bias.** MediaPipe reports elbow and knee bend for visibly straight standing
  limbs. When a rest baseline is found, the bend measured there is removed from small
  bends only (fully up to the bias, fading out by twice the bias), keeping the chord
  (shoulder→wrist, hip→ankle) and the bend plane.
- **Twist sources.** Upper arm / thigh twist come from the elbow / knee bend plane, weighted
  by how bent the limb is (`SolverConfig.bend_twist_*`, `knee_twist_*`): a straight arm
  inherits its twist, a straight leg takes it from the foot heading. Forearm and hand twist come from pinky→index; shin twist from
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
