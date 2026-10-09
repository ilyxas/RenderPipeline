# Stage 8 engineering handoff

9 October 2026, Asia/Jerusalem. Stage 8 experiment completed; **motion-quality acceptance failed**. Baseline remains the default. Stop after Stage 8.

The simultaneous window optimizer improves body trajectories and reprojection on all three clips, and the rendered frame sequences show more coherent body progression. It fails frozen fast-gesture preservation and full-animation continuity gates; dance also regresses stance foot-bone drift. Therefore meaningful overall motion-quality improvement under the requested conditions is **not demonstrated**. Reduced derivative metrics are not sufficient to call this successful.

## Algorithm and architecture

The existing untracked Stage 8 draft was reused and its sequential joint solves replaced with one SciPy `least_squares` solve. All 60 samples share one objective and sparse Jacobian. Variables are root XY increments and exponential-coordinate increments for pelvis, upper/lower arms, thighs, calves and feet (11 joint trajectories). Root depth is fixed at its frozen unobserved gauge. The Stage 7 body backend supplies the initializer; the saved calibration and camera remain fixed. No per-frame optimization or output smoothing is used.

Each residual evaluation performs vectorized canonical `rest_local @ delta` FK through the relevant ancestor closure. It fits the 12 named landmark projections and weak 3D segment-direction priors, with visibility/presence weights and explicit validity masks. Temporal terms penalize geodesic local rotation velocity and acceleration over adjacent samples, plus root trajectory derivatives. Rotation derivatives account for sample-time spacing; coefficients use 24 Hz normalization. Weak initializer anchors and soft scalar-cap residuals complete the objective. Final cap projection is recorded in the published objective diagnostics; caps retain the baseline values. These are scalar swing limits, not anatomical limits.

Robust loss is `soft_l1`, scale .015. Velocity/acceleration weights decrease around initializer steps above 12° per sample, and increase in low-confidence spans. This policy was fixed before benchmark evaluation, but it did **not** preserve the real annotated gesture proxies within tolerance. Increment anchoring uses a single exponential-coordinate chart; rotation derivatives themselves use relative SO(3) logarithms. No claim of global optimization is made.

Missing targets are excluded rather than fabricated. Estimated missing body samples use the canonical existing `body_prior` provenance and confidence .05; original observed confidence remains unchanged. Entirely unobserved joint channels remain neutral/unobserved. Gaps are estimated by the objective, not cosmetic interpolation. Saved measurements list inferred sample counts. Low-confidence priors do not establish recovered truth.

Face/head/hand solvers and the existing compositor are reused. Face weights remain exactly identical to Stage 7. Head ownership remains face with body fallback; wrist compensation is recomputed against the new forearm, preserving the existing pipeline interface. Hand/finger missing-data behavior remains baseline behavior, including neutral returns and abrupt rotations. Contacts/events remain empty; contact/collision weights are explicitly zero. The AnimationBundle schema, writer, validator and Blender consumer are unchanged.

`--solver temporal` is available, with baseline default and optional SciPy dependency group. Stage 8 accepts one 2–4 second window. There is no stitching, contact solving, collision handling, new scheduling/cache infrastructure or Stage 9+ work.

## Frozen benchmark measurements

All clips use the same frozen 2.5-second windows, 60 frames at 24 fps, profile v2, observations, annotations, calibration and acceptance SHA256 `1c4346a7c90a7e0c9b1854caaebc1266097a95438fcc7ef76145bc4e01af221c`. The non_neutral source remains holdout; no parameters were changed after viewing its results. One numerical solve was run per clip with one fixed configuration.

| Metric (Stage 7 → Stage 8) | sing | dance | non_neutral (holdout) |
|---|---:|---:|---:|
| Reprojection median (height) | 0.04668 → 0.018054 | 0.041737 → 0.020112 | 0.084517 → 0.061202 |
| Reprojection p95 (height) | 0.097977 → 0.040995 | 0.10028 → 0.0529 | 0.1413 → 0.1116 |
| Angular velocity p95 (deg/s) | 268.61 → 163.85 | 820.22 → 519.06 | 369.14 → 141.17 |
| Angular acceleration p95 (deg/s²) | 9885.1 → 2058.1 | 20348 → 5754.5 | 15801 → 2258.6 |
| Observed rotation jump max (deg) | 74.438 → 74.438 | 177.22 → 173.1 | 81.664 → 81.15 |
| Root step max (m) | 0.016494 → 0.010461 | 0.063235 → 0.03167 | 0.035732 → 0.028133 |
| Root speed p95 (m/s) | 0.35357 → 0.21361 | 0.93506 → 0.72184 | 0.67479 → 0.31402 |
| Foot-bone drift p95 (m) | 0.33897 → 0.24846 | 0.21399 → 0.27014 | unavailable → unavailable |
| Full solve wall time (s) | 5.7451 → 6.8465 | 4.3288 → 5.4934 | 7.9263 → 8.8998 |
| Python peak RSS (MiB) | 40.703 → 97.641 | 40.469 → 98.266 | 40.219 → 92.391 |

Reprojection units are the frozen visible-skeletal-height proxy, not metric ground truth. Derivative metrics use the frozen evaluator, including its validity rules and head inclusion. Stage 8 infers some previously missing body samples, so its observed derivative coverage differs from baseline. Supplemental measurements report trajectory maxima across the same 11 optimized joints without validity filtering: sing 85.52° → 27.35°, dance 155.67° → 71.80°, holdout 71.75° → 16.84°. Dance still has a large body rotation transition.

Root path length falls from .484 to .292 m (sing), 1.151 to .871 m (dance), and .735 to .292 m (holdout). This can remove jitter but also attenuate motion: holdout vertical root range falls from .182 to .0579 m. Root RMS offsets from baseline are .0684/.0650/.0408 m. Reduced root steps are not proof of recovered physical movement.

## Regressions and acceptance

Reprojection median/p95 and the required ≥10% acceleration improvement pass on all three. Scalar-cap violation fraction stays zero. Root adjacent steps remain below .10 m. All three solves satisfy the 120 s/2 GiB Python budgets.

- **sing:** right upper-arm excursion loses 15.72°, lower arm loses 45.87° (maximum allowed 5°). Their peak timing shifts by 7 and 30 output frames (maximum 1). The observed full-channel 74.44° jump is an unchanged finger channel and fails the 45° gate.
- **dance:** right upper/lower arm excursion loses 19.76°/20.63°. Lower-arm peak shift is exactly 1 frame and passes; frame comparisons use integer sample indices to avoid floating-point false failures. Foot-bone drift median rises .20187 → .22869 m (+.02682); p95 rises .21399 → .27014 m (+.05615), both beyond the .01 m regression allowance. Maximum observed jump remains 173.10°, at the left wrist.
- **non_neutral holdout:** pelvis/head peak shifts are 7/2 frames. Full-channel jump remains 81.15°, at the left wrist. Foot stance drift is unavailable because feet are occluded; it is not counted as a pass.

The amplitude/timing measures are frozen baseline-relative quaternion proxies, not independently reconstructed source motion. Changes in reprojection fit, twist and compensation can change those proxies; that uncertainty does not justify weakening or overriding failed gates. This candidate cannot be advertised as preserving fast source movement.

## Videos and visual review

Each MP4 contains **original source | frozen Stage 7 Xandra | Stage 8 Xandra**, in that order. The Stage 7 bundle was newly rendered; Stage 6 video was not substituted. Both Xandra renders use the identical profile, assets, scene, main camera, 540×960 EEVEE settings, 16 samples, exposure and Blender adapter. The comparison is 1080×640, with aspect-preserving panels, exact recorded source PTS and identical output times. No interpolation, speed changes, camera reframing or motion-dependent scene changes were applied. All videos contain 60 decoded frames at 24 fps.

- [sing comparison](../../runs/stage8/sing/comparison.mp4), [review sheet](../../runs/stage8/sing/review.jpg)
- [dance comparison](../../runs/stage8/dance/comparison.mp4), [review sheet](../../runs/stage8/dance/review.jpg)
- [non_neutral holdout comparison](../../runs/stage8/non_neutral/comparison.mp4), [review sheet](../../runs/stage8/non_neutral/review.jpg)
- [Playback report](../../runs/stage8/report.html), [measurements](../../runs/stage8/measurements.json), [verification](../../runs/stage8/verification.json)

Visual inspection used temporal frame sequences extracted from the actual encoded comparison MP4s (8 samples/s); it is not a human real-time playback sign-off. The complete videos are provided for direct review. Sing has a more coherent arm/knee progression, but wrist/finger changes and altered excursion remain. Dance retains the recognizable arm rise and alternating knee lifts, with fewer neutral body resets, but still has sudden hand changes and feet sliding relative to the floor. The holdout's upper-body progression is steadier, but the animation still interprets the source kneeling posture as a standing forward lean; reduced root bob does not correct that semantic error. All three remain visibly imperfect.

No overall visual acceptance is claimed. The failed gesture and drift gates reinforce the limits seen in the rendered sequences. The result is useful evidence for the body optimizer, not an accepted replacement for Stage 7.

## Diagnostics, verification and reproduction

Diagnostics retain objective terms before/after, termination message/status, optimality, evaluation counts, wall time and process peak RSS. All three terminate on `ftol` (17/31/27 evaluations for sing/dance/holdout). Convergence means relative cost-change termination, not small-gradient/global optimum; holdout optimality is approximately .0325. The maximum budget is 80 evaluations and an internal 115 s deadline; exhaustion is explicitly marked failed and the CLI does not accept it as successful. The benchmark preserves the experimental candidate as evidence and reports its termination gate.

18 focused unit tests passed, including synthetic known-motion recovery with Gaussian noise, an outlier and a missing span. The synthetic test verifies lower recovery error/jitter, missing-span recovery, ≤5° fast-peak error, missing-target independence and sparse Jacobian dependencies. No repeated successful solver checks or benchmark retuning were used. Vectorized candidate FK matched the independent canonical FK within 7e-16 m. All three candidates pass the independent AnimationBundle reader; face weights are unchanged. Six actual renders and three comparisons pass full decode, frame count, resolution and timing verification. Asset hashes recorded by each renderer match the registered assets. A frozen file inventory confirms profiles, baseline artifacts, observations, calibration, annotations, split manifest and thresholds stayed unchanged.

sing: Stage 7 render 50.66 s, Stage 8 render 49.96 s; maximum renderer peak footprint 4.37 GiB (budget 8 GiB).

dance: Stage 7 render 49.30 s, Stage 8 render 49.71 s; maximum renderer peak footprint 4.42 GiB (budget 8 GiB).

non_neutral: Stage 7 render 47.39 s, Stage 8 render 47.44 s; maximum renderer peak footprint 4.29 GiB (budget 8 GiB).

Small implementation artifacts are `src/xms/solve/{temporal_body,residuals,parameterization,temporal_full}.py`, `tests/test_temporal_body.py` and `benchmarks/run_temporal.py`; CLI/pipeline integration is minimal. Media and executable bundles are in ignored `runs/stage8/`. Per-clip manifests record inputs, code/runtime versions, parameters and output paths; manifests were assembled after measurement and describe the small later metadata/gate-arithmetic edits, which did not alter saved motion.

To reproduce a benchmark, use the existing `.venv` (SciPy 1.18.1), asset config `runs/stage6/config.json` and the frozen saved observations/calibration:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/run_temporal.py sing --phase solve
PYTHONPATH=src .venv/bin/python benchmarks/run_temporal.py sing --phase render
```

Use a fresh `runs/stage8/sing` directory; the harness deliberately rejects overwrite. Repeat for dance and non_neutral without changing parameters. Blender runs sequentially in one worker. Pipeline invocation supports `xms run ... --solver temporal`, but the frozen benchmark harness is the path used for this evidence; it avoids retracking.

## Remaining limitations and engineering judgment

The fixed monocular camera/depth prior, scalar limits, initializer-derived temporal weights and weak direction prior cannot fully distinguish genuine fast rotation from observation jitter or determine a physically correct crouch. The coupled fit can redistribute root/rotation motion and reduce noisy baseline-relative gesture amplitudes too much. Foot contact/skating, collisions, anatomical constraints, face/audio timing and hand temporal identity are not solved. Existing hand losses and neutral returns still dominate some abrupt-rotation metrics. There is no evidence for full-clip stitching or production animation acceptance.

A next hypothesis for a separately authorized experiment would be to constrain fast source-space segment excursions/timing within the objective and examine reprojection-versus-direction-prior ambiguity, while retaining frozen acceptance. It has not been implemented or tested. Adding contact/hand fixes here would exceed Stage 8 scope.

**Decision: Stage 8 remains experimental; the requested overall motion-quality gain is not proven. Stop here.**
