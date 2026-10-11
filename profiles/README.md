# Profiles

Character and scene profiles are immutable, versioned registrations bound to the SHA-256
of their asset; a changed asset needs a new version, never an edit.

- `characters/xandra/v1`, `v2` — historical; `v2` is used by the frozen benchmarks.
- `characters/xandra/v3` — `blender/Xandra_Animations.blend` (blue top, black shorts).
- `characters/xandra/v4` — `Xandra_BlackLace_Fitting_v01.blend` (Black Lace fitting); same
  skeleton as v3 (rest transforms verified identical), `look.json` names the garments
  used by surface QA.
- `characters/xandra/arm_limits.v1.json`, `viggle_metahuman_body.v1.json` — bound to the
  skeleton (`rig.npz` hash), so they apply to every look on that skeleton.
- `scenes/bedroom/v1`, `v3` — the bedroom render scene; `v3` is current.
