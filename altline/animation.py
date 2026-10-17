"""Task 5: animation package serialization.

Input: <run>/damper/damped_rotations.json (or any rotations file of that format) and the target rig.
Output: <run>/animation/motion_bundle.json

    {"format": "altline.motion_bundle.v1",
     "fps": float,
     "bones": [bone names, in channel order],
     "rotation_mode": "QUATERNION_WXYZ_LOCAL_BASIS",
     "frames": [{"t": seconds, "q": [[w, x, y, z] per bone]}]}

Each quaternion is the bone's local basis rotation (Blender's pose_bone.rotation_quaternion)
expressed in the rig's own bone frames: for every frame and bone, set
    pose.bones[name].rotation_quaternion = q
and key it at t * fps. No other processing is needed. Root translation is not part of v1
(MediaPipe world landmarks are hip-centred, so global travel is not observed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io_utils import read_json, run_dir, write_json
from .quat import mat_to_quat, quat_to_mat
from .skeleton import Skeleton, load_skeleton

FORMAT = "altline.motion_bundle.v1"


def to_rig_frames(q: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """anatomical basis -> rig basis: O^T B O (see skeleton.Skeleton.rig_offset)."""
    if np.allclose(offset, np.eye(3)):
        return q
    return mat_to_quat(offset.T @ quat_to_mat(q) @ offset)


def build_bundle(clean: dict, skel: Skeleton) -> dict:
    bones = clean["meta"]["bones"]
    if bones != skel.names:
        raise ValueError("clean rotations were solved for a different bone list than the target rig")
    frames = []
    for f in clean["frames"]:
        qs = []
        for b, q in enumerate(f["rotations"]):
            q = np.asarray(q, dtype=float)
            norm = np.linalg.norm(q)
            if abs(norm - 1.0) > 1e-6:
                raise ValueError(f"non-unit quaternion at frame {f['frame_index']} bone {bones[b]}: |q|={norm}")
            qs.append([round(float(v), 7) for v in to_rig_frames(q, skel.rig_offset[b])])
        frames.append({"t": round(float(f["timestamp_s"]), 6), "q": qs})
    return {
        "format": FORMAT,
        "fps": clean["meta"]["fps"],
        "bones": bones,
        "rotation_mode": "QUATERNION_WXYZ_LOCAL_BASIS",
        "frames": frames,
    }


def load_bundle(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Return (bones, times (T,), quats (T, N, 4), fps)."""
    b = read_json(path)
    if b.get("format") != FORMAT:
        raise ValueError(f"unexpected bundle format {b.get('format')}")
    times = np.array([f["t"] for f in b["frames"]])
    quats = np.array([f["q"] for f in b["frames"]], dtype=float)
    return b["bones"], times, quats, float(b["fps"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="altline motion bundle writer")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rig", type=Path)
    args = ap.parse_args(argv)
    rd = run_dir(args.run_id)
    bundle = build_bundle(read_json(rd / "damper" / "damped_rotations.json"), load_skeleton(args.rig))
    print(write_json(rd / "animation" / "motion_bundle.json", bundle))


if __name__ == "__main__":
    main()
