# Viggle external mocap vs `video` solver — base video set (2026-10-11)

Branch `exp/external-mocap-viggle`. Both runs use `--channels body --quality preview`
on the eight clips of `base video/`. Viggle extraction: MetaHuman-body rig, 24 fps,
cached by video hash (280 credits for the set). Evidence is under
`runs/viggle-compare/` (not in Git): `side_by_side/<clip>.mp4` (source | video | viggle),
`side_sheet_*.jpg` (source front, then video and viggle side views at frames 20, 72, 130),
`front_sheet.jpg`, `metrics.json`.

**Status: implemented and functionally verified; quality not accepted.** Visual
judgement belongs to Ilya; the notes below are preliminary.

## Retarget verification (before any video)

- Viggle skeleton posed in Xandra's rest → zero deltas (< 1e-5°).
- Right shoulder 45° forward → exactly 45°, only the right-arm subtree moves; visible in side view.
- Viggle's own rest (T-pose) → T-posed Xandra, legs unchanged (< 3°).
- Text-to-motion sample "raise right arm overhead" → arm vertical in side view.

## Per clip (preliminary)

Metrics from the published bundles, identical for both solvers: foot skate = p95
horizontal speed of a sole within 2 cm of the floor (cm/s); floor min / hover max =
lowest sole below / above Y=0 (cm); hand–spine = closest wrist to the spine segment (cm).

| Clip | Motion | Side view (prelim.) | Skate video / viggle | Floor min / hover, viggle | Hand–spine video / viggle |
|---|---|---|---|---|---|
| 061c1113 | raise right arm | video: arm ends tilted forward; viggle: vertical | 11.5 / 5.5 | **−13.1** / 0.5 | 37.7 / 35.9 |
| 066332e1 | cross arms | video: hands reach forward, not on chest; viggle: crossed on chest | 22.4 / 5.5 | −0.5 / 1.0 | 33.1 / **11.1** |
| 53130c00 | head turn | similar; needs front video review | 1.8 / 3.8 | −0.2 / 0.4 | 39.0 / 38.6 |
| 619b1b62 | knee raise | video: leg extends forward like a kick; viggle: bent knee raised | 44.1 / 10.6 | −1.7 / 0.5 | 42.3 / 40.9 |
| 690b4562 | arm sideways then up | video: arm goes forward; viggle: sideways, vertical | 28.3 / 3.6 | −1.9 / **11.2** | 31.0 / 26.3 |
| 85a9fbd2 | arms out, hands to face | video: hands stay in front of the face; viggle: hands at the face | 1.3 / 4.3 | −0.7 / 0.3 | 35.8 / **18.0** |
| 9645beb2 | head (second clip) | similar; needs front video review | 2.1 / 3.9 | −0.2 / 0.3 | 42.3 / 40.1 |
| df1e6bc3 | forward kick | both reach a high kick; viggle less foot sliding | 51.1 / 6.5 | −1.2 / 1.8 | 34.6 / 28.5 |

Torso lean (median) is 3–9° for `video` and 0.3–4° for viggle on every clip. No
clip triggered the motion gate for either solver.

## Findings

- Depth and self-contact, the known weakness of `video`, look better with Viggle in
  side view on the arm, crossing, hands-to-face and knee clips.
- **Viggle root height is wrong on both arm-overhead clips**: its pelvis drops 12–13 cm
  while the subject stands still. With one floor shift per clip Xandra sinks into the
  floor (061c1113, −13.1 cm) or hovers before the drop (690b4562, +11.2 cm). Fixing this
  needs per-frame foot–floor support, not a clip constant.
- Viggle's world origin is ~1.6 m behind the subject; horizontal root travel is taken
  relative to the first sample.
- Viggle has no arm twist bones; arm roll is distributed by `xms.solve.twist`.
  Arm limits would not have triggered on any clip (reported, not applied).
- Head-yaw clips cannot be judged from stills; front video review is needed.
