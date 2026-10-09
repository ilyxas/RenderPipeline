# Stages 5–7 engineering checkpoint

Stages 5–7 are implemented. Stop here for engineering review; Stage 8 has not started. This is a functional baseline, not motion-quality acceptance. Original Stage 4 `runs/20261009-175723-14b81716` and accepted Stage 6 inventories were verified unchanged byte-for-byte.

## Functionality and contracts

- Stage 5: PTS-aligned source/render comparison, SAR-preserving letterbox, HTML/JSON QA, explicit unavailable metrics, result statuses/exit codes, subprocess wall time and RSS. Silent, rational FPS, VFR, rotation, nonzero PTS, SAR, short and invalid media were exercised. Duration and A/V checks retain the one-frame limit. Invalid input produces an error report before rendering.
- Stage 6: newly extracted named face observations and hand landmarks feed baseline head, facial morph, wrist and finger solving. The compositor publishes one AnimationBundle consumed unchanged by the main and diagnostic renders. Face owns head where available; body is fallback. Jaw/eyes remain morph-owned. Wrist rotation compensates the solved forearm. Missing observations retain explicit masks and neutral fallback.
- Stage 7: deterministic reliable-window calibration with fixed rest geometry, confidence and frozen reference camera; source-level tuning/holdout split; reprojection, angular velocity/acceleration, scalar limits, root trajectory, stance bone-drift proxies and coverage. Fixed-input repeat solves produced exactly identical arrays on all three clips (tolerance 1e-10).
- Verified synthetic contracts: named mapping and `_neutral`, duplicate aliases, mirrored L/R assignment, forearm compensation, exclusive head ownership, unavailable QA, known orthographic projection and angular velocity. Saved calibration rejects mismatched source bindings. All three Stage 7 bundles pass the independent contract reader.

## Actual rendered artifacts

Paths below are relative to the repository root. Large media/observations are local and ignored by Git.

Accepted Stage 6 run: `runs/stage6/roi-runs/20261009-184331-e2bf5411/`

- `outputs/video.mp4`: 2.5 seconds, 60 frames, 24 fps, 540×960; body, head, face and hands/fingers through the same bundle.
- `comparison/compare.mp4`: source alongside the actual render, sampled through the recorded source PTS map.
- `outputs/face.mp4`, `outputs/hand_l.mp4`, `outputs/hand_r.mp4`: diagnostic cameras, same bundle and timeline.
- `animation/`, `observations/`, `manifest.json`, `report/report.html`, `report/quality.json`: executable data and provenance.
- `runs/stage6/*-review.jpg`: contact sheets inspected from actual encoded MP4s.

Stage 5 review of immutable Stage 4: `runs/stage5/stage4-review/report.html` and `comparison/compare.mp4`.

Stage 7 measurements/calibration: `benchmarks/baseline/{sing,dance,non_neutral}/`; executable bundles and HTML reports: `runs/stage7/<id>/baseline/`. Stage 7 bundles use the new calibration; the Stage 6 video deliberately retains its original calibration. No claim is made that the Stage 6 render depicts the later Stage 7 bundle.

## Frozen measurements

All windows are 2.5 seconds / 60 output samples. Reprojection units are visible skeletal height, a source-landmark proxy; the cropped holdout has no full-body height normalization.

| Measurement | sing (tuning) | dance (tuning) | non_neutral (holdout) |
|---|---:|---:|---:|
| Reprojection median / p95 | .04668 / .09798 | .04174 / .10028 | .08452 / .14130 |
| Body angular acceleration p95, deg/s² | 9885 | 20348 | 15801 |
| Observed adjacent rotation max, deg | 74.44 **fail** | 177.22 **fail** | 81.66 **fail** |
| Root adjacent step max, m | .01649 | .06324 | .03573 |
| Face coverage | 93.3% | 93.3% | 100% |
| Left / right hand coverage | 46.7% / 100% | 43.3% / 61.7% | 93.3% / 100% |
| Stance foot-bone drift median / p95, m | .24293 / .33897 | .20187 / .21399 | unavailable |
| Calibration confidence | .4496 | .4500 | .2925 |
| Full solve wall time, s | 5.75 | 4.33 | 7.93 |
| Python process peak RSS, MiB | 40.70 | 40.47 | 40.22 |

Scalar-cap violation fraction is zero; this does not establish anatomical validity. All three fail the unchanged 45° adjacent-rotation gate. Root steps pass the 0.10 m gate. The holdout starts kneeling and leaning, needs no manual A-pose, and has no visible feet/full-height calibration. No neutral candidate was found in any selected window; confidence is reduced accordingly.

Sing has two manually bracketed mouth events: only one channel crossing matched, with .167 s error to the annotated bracket. This is a jaw-channel proxy, not lip-sync measurement. Metric 3D depth, geometric lip closure/audio synchronization, sole skating, surface penetration, independently annotated hand swap counts and exhaustive uncropped head support are unavailable. No confident flight interval was observed; none was fabricated.

Frozen criteria: `benchmarks/acceptance.v1.json`, SHA256 `1c4346a7c90a7e0c9b1854caaebc1266097a95438fcc7ef76145bc4e01af221c`. These were written before baseline measurements and any temporal experiment. `benchmarks/baseline/manifest.json` records runtime, code/artifact hashes, verification and unavailable reasons. Performance budgets are limits for the next experiment, not temporal-solver measurements.

## Known limitations and architecture deviations

Motion is visibly imperfect: abrupt rotations, foot drift, hand losses and neutral returns, conspicuous teeth and imperfect singing shapes. Some fingertips leave the diagnostic camera framing. Face confidence is an availability/ROI-size heuristic, not a calibrated probability. Camera/global orientation are monocular priors, not recovered metric camera geometry. Manual annotations cover short intervals only.

No architectural boundary deviation or later-stage infrastructure was introduced. Character/scene v2 profiles preserve v1 and encode ownership/diagnostic cameras. The first full-frame face attempt produced 0/60 detections and is preserved as rejected evidence; the accepted extraction uses a single body-derived head ROI with unchanged detector thresholds. The VFR nearest-PTS residual was corrected from an invented test gate to a reported sampling measurement; architecture duration/A-V thresholds were not changed. No contact solver, IK, temporal optimizer, collision correction, scheduler or cache was added.

## Reproduction and next step

Use the existing local `.venv` and asset configuration `runs/stage6/config.json`; runtime versions are in `environments/stages5-7-runtime.json`. Large assets remain external. No legacy animation or exported animated GLB is consumed.

```sh
PYTHONPATH=src .venv/bin/python -m xms.cli run /Users/ilya/codex/clip02_blender_kit/input/sing.mp4 --start 3 --end 5.5 --config runs/stage6/config.json --channels full --runs-root runs/reproduction
PYTHONPATH=src .venv/bin/python benchmarks/run_baseline.py non_neutral --observations runs/stage7/non_neutral/observations --profile profiles/characters/xandra/v2 --out runs/reproduction/non-neutral-baseline
```

The full run now uses Stage 7 calibration; use the saved Stage 6 bundle to reproduce its original motion. The baseline command requires a new output directory and preserves checked-in frozen reference measurements. CLI exit code 3 means needs_review, not process/render failure. On this Mac, Blender and `/usr/bin/time -l` require execution outside the restricted sandbox.

Ready for a bounded Stage 8 implementation after review: fixed inputs, camera, calibration, reference metrics and regression gates exist. Quality acceptance is not achieved; the future solver must demonstrate improvements under unchanged criteria, while unavailable metrics remain unavailable. Recommended next step: approve this checkpoint, then implement only the Stage 8 body window solver and compare against these frozen baselines.
