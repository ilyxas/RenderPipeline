# Stages 9–16 implementation record

Functional delivery is separate from motion-quality acceptance. Frozen benchmarks and thresholds remain unchanged. Baseline stays available. Stage 8 prerequisite preserved at `6cf56e3`.

| Stage / commit | Implementation / verification | Acceptance and remaining work |
|---|---|---|
| 9 / `4e3025c` | Bounded 3s windows, .75s overlap, fixed global calibration, boundary constraints, quaternion sign continuity, baseline fallback logs. Two focused tests; real saved dance observations plus 120-sample overlap smoke validated and serialized (two windows, no dropped samples). | Functional; experimental quality. Full-film performance/visual acceptance not run. No chunked writer needed for tested size. |
| 10 / `bc8422f` | Visibility/height/speed hysteresis, world anchors, flight events, coupled leg/root contact residuals and drift QA; stance/flight/missing fixture passes. Saved dance solve serialized successfully. | Partial physical acceptance: skeletal origins replace unverified sole points; stance can be unavailable. Frozen real contact/jump gates are not accepted. No unconditional floor clamp. |
| 11 / `8bcc0ae + fdd9ff4` | ROI sharpness/geometry proxies and one weak-frame full-image retry; bounded lower-envelope expression offsets, channel-specific filters, ≤150ms face gaps, separate head rotation with scale removed and head-local gaze. Two gap/scale fixtures pass. | Functional video policy; 90% benchmark coverage/event acceptance unmeasured. Existing bounded head solver retained; neck distribution deferred to avoid changing ownership. |
| 12 / `9088ea6` | Aligned mono envelope extraction, optional Rhubarb speech/Demucs separate-runtime adapters, confidence-normalized lip fusion and mutual constraints; video dominance/silence fixture passes. | Partial: neither optional backend is installed; Demucs compatibility/speed smoke unavailable, no dependency/weight download. No phoneme or singing accuracy acceptance, teeth exposure calibration deferred. Original mix is soundtrack. |
| 13 / `fa49d0e` | Body-wrist/temporal L/R association, one missing-hand ROI retry, ≤150ms quaternion gaps, fixed-rest fingers and local wrists; bilateral bounded arm/wrist palm correction requires opposing 3D normals and proximity. Two association tests and saved 60-frame full solve/bundle validation pass. | Functional bounded policy; metacarpal twist remains rest. Palm quality, prayer-pose and frozen fast-gesture acceptance not demonstrated. |
| 14 / `b7dbb15` | Version 3 profiles preserve v2 frozen assets; analytical capsule/plane primitives, coupled collision costs in temporal windows, 5°/3cm increment bounds, proxy QA. Analytical fixture and both new profile hash checks pass. | Partial: proxy radii are conservative experimental dimensions, not surface validated; only forearm/torso pairs and floor registration, furniture/cloth proxies absent. Unresolved penetrations need review. |
| 15 / `56e60e6` | Blender evaluated hand–top/shorts surfaces, BVH nearest distances, watertight ray-parity signs, bounded exact edge/triangle intersection checks; all or sampled coverage; optional libigl adapter and at-most-one 5°/3cm solver refinement. Two analytical fixtures plus actual three-frame mesh smoke pass (~1.03GB peak RSS). | Partial coverage: smoke covers 3/60 frames; no whole-model surface acceptance. Hair, finger–finger, other skin pairs excluded. Signed smoke reveals 88.36mm hand–shorts depth at sample 30 (129 triangle intersections). Refinement was attempted during Stage 16 but skipped because arm/wrist channels were unobserved. Defect remains needs_review. |
| 16 | One-command CLI/default registration, output and segment options, static whole-trajectory framing, preview/final candidate presets, optional face close-up, sampled/all surface policy, one-refinement guard and reports. Disk-backed RGB decoding and linear-memory time-map matching. Eleven focused framing/timeline/bundle/ownership/sparse-constraint checks pass; v3 constrained temporal solve converges without fallback. Fresh 60-frame/2.5s main and face MP4 fully decode, correct timing/audio, 100% skeletal framing; one final 1080×1920/64-sample frame verified. | Functionally delivered; artistic/flicker and broad motion acceptance pending. Final full-clip orchestration configured, but only preview end-to-end and one final frame executed. Lighting/material look is preserved rather than retuned. |

Stage 16 also fixes constraint residual/sparsity row ordering, includes the torso endpoint in FK dependencies, and prevents bounded corrections from activating unobserved channels or exceeding baseline scalar limits. The first Stage 11 fixture used exact equality and failed by floating-point roundoff; the separate correction commit passed both fixtures.

## Demonstration and usage

See [README commands](../../README.md#run-a-new-video). Local assets are registered in ignored `.xms/config.json`; no source assets were modified.

- [Main Xandra MP4](../../runs/stage16/xandra-demo.mp4)
- [Face MP4](../../runs/stage16/jobs/20261010-002428-48ddf527/outputs/face.mp4)
- [Source comparison](../../runs/stage16/jobs/20261010-002428-48ddf527/comparison/compare.mp4)
- [Bundle](../../runs/stage16/jobs/20261010-002428-48ddf527/animation/metadata.json), [run manifest](../../runs/stage16/jobs/20261010-002428-48ddf527/manifest.json), [HTML report](../../runs/stage16/jobs/20261010-002428-48ddf527/report/report.html)
- [Main review frames](../../runs/stage16/review.jpg), [face review frames](../../runs/stage16/face-review.jpg), [final preset frame](../../runs/stage16/final-frame/000030.png)

Demo took 101.1s including tracking, main render, optional close-up, surface diagnostics and comparison. It uses a new 2–4.5s interval of sing, baseline body with new video policies; no previous animation is reused. Five sampled video frames show recognizable arm gestures and face expression changes. Feet cross/slide, hand gaps remain, and hair/shadow artifacts are visible. Sampled surface QA covers 8/60 frames (max signed depth .90mm); this is not full-clip surface acceptance. No overall human visual sign-off or frozen quality acceptance is claimed.

## Known limits

- Single visible, mainly upright person; occlusion, kneeling and complex posture semantics remain limited. Monocular depth/camera/floor are priors.
- Contact geometry uses foot bone origins; uncertain support is unavailable. Proxy radii are experimental; only forearm/torso costs are coupled into temporal solving. Baseline/video bodies remain unfiltered.
- Long gaps use neutral/body prior, metacarpal twist stays rest, neck distribution is deferred. Palm-contact and correction quality are not accepted.
- Rhubarb/Demucs are optional and absent here. Demucs compatibility/speed, lip timing and singing accuracy were not verified. Mixed energy is low confidence; teeth exposure has no new calibration.
- Surface tests cover declared hand–garment pairs; hair, finger–finger and other skin pairs are excluded. Refinement cannot invent missing arm/wrist data and is limited to one pass.
- Final/flicker/artistic acceptance and full-film memory/runtime budgets are unmeasured. Long native-resolution RGB decoding uses temporary disk space; full clips render sequentially. No stages 17–21 infrastructure was added.
