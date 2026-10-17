"""Reviewer audit of a v2 run: audit.audit() + v1_vs_v2 comparison + depth-rebuild diagnostics.

    python3 -m altline.review.audit_v2 --clip 0663 [--clip ...]

Writes reports/v2_<clip>/dev_audit.json (manual findings from reports/v2_<clip>/manual.json are
merged by audit.audit) and plots (acc_*.png from audit, cmp_*.png from compare, keyframes).
"""

from __future__ import annotations

import argparse
import json

from altline.io_utils import REPORTS_ROOT, run_dir, write_json
from altline.review import audit, compare

KEYFRAMES = {"061c": [22, 79, 101, 140], "0663": [18, 30, 44, 100], "5313": [11, 40, 58, 120], "619b": [6, 60, 89, 118],
             "690b": [0, 17, 39, 118], "85a9": [9, 16, 51, 100], "9645": [14, 72, 110, 144], "df1e": [42, 86, 95, 120]}


def keyframes(clip: str) -> list[str]:
    """v2 preview frames (source|front|side|top) at the key frames, v1 preview row below each."""
    import cv2
    import numpy as np
    frs = KEYFRAMES[clip]
    out_dir = REPORTS_ROOT / f"v2_{clip}" / "plots"

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
    v2, v1 = grab(f"v2_{clip}"), grab(f"v1_{clip}")
    rows, paths = [], []
    for f in frs:
        a, b = v2[f], v1[f]
        h, w = a.shape[:2]
        b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
        lab = np.full((44, w, 3), 255, np.uint8)
        cv2.putText(lab, f"{clip} frame {f}: top row v2 (source|front|side|top), bottom row v1 (source|front|side)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        bb = np.full((h, w, 3), 255, np.uint8)
        bb[:, :min(b.shape[1], w)] = b[:, :w]
        rows += [lab, a, bb]
        p = out_dir / f"v2_frame_{f:03d}.png"
        cv2.imwrite(str(p), a)
        paths.append(f"plots/{p.name}")
    img = np.vstack(rows)
    cv2.imwrite(str(out_dir / "keyframes_v1_v2.png"), cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6)),
                                                                   interpolation=cv2.INTER_AREA))
    return ["plots/keyframes_v1_v2.png"] + paths


def run(clip: str) -> dict:
    rid = f"v2_{clip}"
    rep = audit.audit(rid)
    rep["v1_vs_v2"] = compare.compare(f"v1_{clip}", rid, ablate=True)
    rep["depth_rebuild_diagnostics(unfiltered re-run of depth.reconstruct)"] = compare.depth_diagnostics(rid)
    rep.setdefault("keyframe_images", {})["v1_vs_v2"] = keyframes(clip)
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
