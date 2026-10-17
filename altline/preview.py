"""Visual check: the motion bundle on the reference rig next to the source video.

Output: <run>/preview/preview.mp4 — per frame: source | front | side | top view.
Rig bones (FK of the bundle, constant lengths) are coloured lines: left side blue,
right side orange, trunk dark grey; the red arrow is where the head faces. The calibrated tracker landmarks (scaled to the rig
and placed at the pelvis) are light grey dots, so depth or twist disagreements between
the tracker and the solved rig are visible at a glance.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .animation import load_bundle
from .io_utils import read_json, run_dir
from .quat import quat_to_mat
from .skeleton import Skeleton, load_skeleton

DOTS = ["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_index", "right_index", "left_pinky", "right_pinky", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"]


def fk_sequence(skel: Skeleton, quats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    worlds, heads, tails = [], [], []
    for qs in quats:
        basis = np.array([quat_to_mat(q) for q in qs])
        w, h, t = skel.fk(basis)
        worlds.append(w)
        heads.append(h)
        tails.append(t)
    return np.array(worlds), np.array(heads), np.array(tails)


def render(run: Path, skel: Skeleton, out: Path) -> Path:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bones, times, quats, fps = load_bundle(run / "animation" / "motion_bundle.json")
    worlds, heads, tails = fk_sequence(skel, quats)
    hb = skel.index["head"]
    cal_path = run / "calibration" / "guarded_pose.json"
    cal = read_json(cal_path if cal_path.exists() else run / "calibration" / "calibrated_pose.json")
    profile = read_json(run / "calibration" / "actor_profile.json")
    scale = profile["global_scale"]
    pelvis = skel.heads[skel.index["pelvis"]]
    video = cal["meta"]["video"]
    cap = cv2.VideoCapture(video)

    colors = ["#3b6fd8" if n.endswith("_l") else "#e8892b" if n.endswith("_r") else "#333333" for n in bones]
    tmp = Path(tempfile.mkdtemp(prefix="altline_preview_"))
    try:
        for t in range(len(times)):
            ok, bgr = cap.read()
            fig, axes = plt.subplots(1, 4, figsize=(14.0, 6.4), dpi=100,
                                     gridspec_kw={"width_ratios": [0.75, 1, 1, 1]})
            if ok:
                axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"source  frame {t}  t={times[t]:.2f}s", fontsize=9)
            axes[0].axis("off")
            f = cal["frames"][t]
            dots = None
            if f["valid"]:
                hip = 0.5 * (np.array(f["joints"]["left_hip"][:3]) + np.array(f["joints"]["right_hip"][:3]))
                dots = np.array([(np.array(f["joints"][k][:3]) - hip) * scale + pelvis for k in DOTS])
            views = [(0, 1, "front (x: actor left ->)"), (2, 1, "side (z: actor forward ->)"),
                     (0, 2, "top (x: left ->, y: forward ^)")]
            for ax, (h_axis, v_axis, label) in zip(axes[1:], views):
                for b in range(len(bones)):
                    par = skel.parents[b]
                    if par >= 0 and np.linalg.norm(heads[t, b] - tails[t, par]) > 1e-6:  # e.g. hip joints
                        ax.plot([tails[t, par, h_axis], heads[t, b, h_axis]], [tails[t, par, v_axis], heads[t, b, v_axis]],
                                color="#999999", lw=1, ls="--")
                    ax.plot([heads[t, b, h_axis], tails[t, b, h_axis]], [heads[t, b, v_axis], tails[t, b, v_axis]],
                            color=colors[b], lw=2.5, solid_capstyle="round")
                # head facing: arrow along the head's forward (local +Z) axis
                c = 0.5 * (heads[t, hb] + tails[t, hb])
                tip = c + 0.15 * worlds[t, hb][:, 2]
                ax.plot([c[h_axis], tip[h_axis]], [c[v_axis], tip[v_axis]], color="#d62728", lw=2)
                if dots is not None:
                    ax.scatter(dots[:, h_axis], dots[:, v_axis], s=8, color="#9a9a9a", zorder=3)
                ax.set_xlim(-1.0, 1.0)
                if v_axis == 1:
                    ax.axhline(0, color="#bbbbbb", lw=0.8)
                    ax.set_ylim(-0.1, 2.1)
                else:
                    ax.set_ylim(-0.8, 0.8)
                ax.set_aspect("equal")
                ax.set_title(label, fontsize=9)
                ax.tick_params(labelsize=7)
            fig.tight_layout()
            fig.savefig(tmp / f"{t:05d}.png")
            plt.close(fig)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        str(out)], check=True)
    finally:
        cap.release()
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="altline preview video")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rig", type=Path)
    args = ap.parse_args(argv)
    rd = run_dir(args.run_id)
    print(render(rd, load_skeleton(args.rig), rd / "preview" / "preview.mp4"))


if __name__ == "__main__":
    main()
