"""Run the whole altline pipeline on one video. Each stage hands off through files only.

    python -m altline.run "base video/clip.mp4" --run-id my_run [--rig rig.npz] [--no-preview]
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from . import animation, calibration, limits, observations, preview, solver
from .config import Config
from .io_utils import read_json, run_dir, write_json
from .skeleton import load_skeleton


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(video: Path, run_id: str, rig: Path | None = None, make_preview: bool = True,
        cfg: Config | None = None) -> Path:
    cfg = cfg or Config()
    rd = run_dir(run_id)
    skel = load_skeleton(rig)
    stages = {}

    def stage(name, fn):
        t0 = time.time()
        result = fn()
        stages[name] = {"seconds": round(time.time() - t0, 2)}
        return result

    raw = rd / "observations" / "raw_pose.json"
    if not raw.exists():
        stage("observations", lambda: observations.extract(video, raw))
    stage("calibration", lambda: calibration.calibrate(raw, rd / "calibration", skel, cfg.calibration))
    stage("solver", lambda: solver.solve(rd / "calibration" / "calibrated_pose.json",
                                         rd / "calibration" / "actor_profile.json",
                                         rd / "solver" / "raw_rotations.json", skel, cfg.solver))

    def _limits():
        clean, report = limits.apply_limits(read_json(rd / "solver" / "raw_rotations.json"), cfg.limits)
        write_json(rd / "limits" / "clean_rotations.json", clean)
        write_json(rd / "limits" / "limits_report.json", report)
    stage("limits", _limits)
    stage("animation", lambda: write_json(rd / "animation" / "motion_bundle.json",
                                          animation.build_bundle(read_json(rd / "limits" / "clean_rotations.json"), skel)))
    if make_preview:
        stage("preview", lambda: preview.render(rd, skel, rd / "preview" / "preview.mp4"))

    write_json(rd / "manifest.json", {
        "run_id": run_id,
        "video": str(video),
        "video_sha256": file_sha256(video),
        "rig": str(rig) if rig else "altline reference rig (skeleton.reference_skeleton)",
        "config": cfg.to_dict(),
        "stages": stages,
    })
    return rd


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="altline: video -> motion_bundle.json")
    ap.add_argument("video", type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rig", type=Path)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)
    print(run(args.video, args.run_id, args.rig, not args.no_preview))


if __name__ == "__main__":
    main()
