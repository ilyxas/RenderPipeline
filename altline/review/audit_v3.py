"""Reviewer audit of a v3 run: audit.audit() + v2_vs_v3 / v1_vs_v3 + depth diagnostics + key frames.

    python3 -m altline.review.audit_v3 --clip 0663 [--clip ...]
"""

from __future__ import annotations

import argparse
import json

from altline.io_utils import REPORTS_ROOT, run_dir, write_json
from altline.review import audit, compare3
from altline.review.audit_v2 import KEYFRAMES


def keyframes(clip: str, new: str = "v3", old: str = "v2") -> list[str]:
    import cv2
    import numpy as np
    frs = KEYFRAMES[clip]
    out_dir = REPORTS_ROOT / f"{new}_{clip}" / "plots"

    def grab(run):
        cap = cv2.VideoCapture(str(run_dir(run) / "preview" / "preview.mp4"))
        got, t = {}, 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if t in frs:
                got[t] = img
            t += 1
        return got
    a_, b_ = grab(f"{new}_{clip}"), grab(f"{old}_{clip}")
    rows, paths = [], []
    for f in frs:
        a, b = a_[f], b_[f]
        w = a.shape[1]
        lab = np.full((44, w, 3), 255, np.uint8)
        cv2.putText(lab, f"{clip} frame {f}: top row {new}, bottom row {old} (source|front|side|top)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        rows += [lab, a, cv2.resize(b, (w, a.shape[0]))]
        p = out_dir / f"{new}_frame_{f:03d}.png"
        cv2.imwrite(str(p), a)
        paths.append(f"plots/{p.name}")
    img = np.vstack(rows)
    name = f"keyframes_{old}_{new}.png"
    cv2.imwrite(str(out_dir / name), cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6)), interpolation=cv2.INTER_AREA))
    return [f"plots/{name}"] + paths


def run(clip: str) -> dict:
    rid = f"v3_{clip}"
    rep = audit.audit(rid)
    cmp, _ = compare3.run(clip, ["v1", "v2", "v3"], ablate=True)
    rep.update({k: v for k, v in cmp.items() if k not in ("clip",)})
    rep["depth_rebuild_diagnostics(unfiltered re-run of current depth.reconstruct)"] = compare3.depth_diagnostics(rid)
    rep.setdefault("keyframe_images", {})["v2_vs_v3"] = keyframes(clip)
    write_json(REPORTS_ROOT / rid / "dev_audit.json", json.loads(json.dumps(rep, default=float)))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True)
    a = ap.parse_args(argv)
    for c in a.clip:
        run(c)
        print("ok", c)


if __name__ == "__main__":
    main()
