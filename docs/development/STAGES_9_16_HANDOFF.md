# Stages 9–16 implementation record

Functional delivery is separate from motion-quality acceptance. Frozen benchmarks and thresholds remain unchanged. Baseline stays available. Stage 8 prerequisite preserved at `6cf56e3`.

| Stage | Implementation / verification | Acceptance and remaining work |
|---|---|---|
| 9 | Bounded 3s windows, .75s overlap, fixed global calibration, boundary constraints, quaternion sign continuity, baseline fallback logs. Two focused tests; real saved dance observations plus 120-sample overlap smoke validated and serialized (two windows, no dropped samples). | Functional; experimental quality. Full-film performance/visual acceptance not run. No chunked writer needed for tested size. |
| 10 | Visibility/height/speed hysteresis, world anchors, flight events, coupled leg/root contact residuals and drift QA; stance/flight/missing fixture passes. Saved dance solve serialized successfully. | Partial physical acceptance: skeletal origins replace unverified sole points; stance can be unavailable. Frozen real contact/jump gates are not accepted. No unconditional floor clamp. |
| 11 | ROI sharpness/geometry proxies and one weak-frame full-image retry; bounded lower-envelope expression offsets, channel-specific filters, ≤150ms face gaps, separate head rotation with scale removed and head-local gaze. Two gap/scale fixtures pass. | Functional video policy; 90% benchmark coverage/event acceptance unmeasured. Existing bounded head solver retained; neck distribution deferred to avoid changing ownership. |
