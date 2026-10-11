# Render settings: samples and shadows (2026-10-11)

Apple M4 (Metal), Blender 4.5 EEVEE Next, Black Lace asset (v4), bedroom scene,
720×1280, same bundle and camera plan. The asset scenes ship `shadow_ray_count=4`,
`shadow_step_count=16`; Blender's defaults are 1 and 6.

| Variant | Samples | Shadow rays / steps | Seconds per frame | Mean abs diff vs A (0–255) |
|---|---|---|---|---|
| A (previous final) | 64 | 4 / 16 | 4.0 | — |
| B | 64 | 1 / 6 | 2.75 | 0.35 (0.2 % of pixels > 10) |
| C (adopted) | 32 | 1 / 6 | 1.4 | 0.6 (0.6 % of pixels > 10) |

A full 15 s clip rendered as C in 7 min 14 s (A: about 27 min plus the preview
preflight). Ilya compared A and C side by side in motion — lace, stockings, skin and
shadows — and accepted C: same look, no visible flicker. Rendering time is GPU
sampling: at 540×960/16 samples a frame renders in 0.88 s and frames arrive every
0.94 s, so per-frame overhead is small.

Adopted in `xms.render.camera.PRESETS`: preview 16 and final 32 samples, shadow
rays 1 and steps 6 for both; `frames/manifest.json` records the values used.
Not changed: final still renders a whole-clip 540×960 preflight first (≈15–20 % of
final time).
