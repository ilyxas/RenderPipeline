"""Task 1: 2D/3D feature extraction with MediaPipe Pose Landmarker.

Input: an MP4 with one person and a static camera.
Output: <run>/observations/raw_pose.json

    {"meta": {...},
     "frames": [{"frame_index": int, "timestamp_s": float, "detected": bool,
                 "joints": {name: [x, y, z, visibility]},        # world space, meters
                 "image_joints": {name: [x, y, z, visibility]}}]} # normalized image plane

`joints` are MediaPipe world landmarks (origin at the hip midpoint, x = image right,
y = image down, z = away from the camera). They are stored raw; the conversion to the
canonical character space happens in calibration.py. `image_joints` are normalized
image coordinates (x, y in [0, 1], z = relative depth on the x scale).
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from .io_utils import run_dir, write_json

JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)


def default_model_path(variant: str = "full") -> Path:
    env = os.environ.get("ALTLINE_POSE_MODEL")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "altline" / f"pose_landmarker_{variant}.task"


def ensure_model(path: Path, variant: str = "full") -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(variant=variant)
    print(f"downloading MediaPipe model {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def _landmarks_to_dict(landmarks) -> dict[str, list[float]]:
    return {
        name: [float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)]
        for name, lm in zip(JOINT_NAMES, landmarks)
    }


def extract(video: Path, out_path: Path, model: Path | None = None, variant: str = "full") -> Path:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    model = ensure_model(model or default_model_path(variant), variant)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frames = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(index * 1000.0 / fps))
            result = landmarker.detect_for_video(image, timestamp_ms)
            frame = {"frame_index": index, "timestamp_s": index / fps, "detected": False,
                     "joints": {}, "image_joints": {}}
            if result.pose_world_landmarks:
                frame["detected"] = True
                frame["joints"] = _landmarks_to_dict(result.pose_world_landmarks[0])
                frame["image_joints"] = _landmarks_to_dict(result.pose_landmarks[0])
            frames.append(frame)
            index += 1
    cap.release()

    meta = {
        "video": str(video),
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": len(frames),
        "n_detected": sum(f["detected"] for f in frames),
        "model": model.name,
        "joint_names": JOINT_NAMES,
        "world_axes": "mediapipe: origin hip midpoint, +x image right, +y image down, +z away from camera, meters",
    }
    return write_json(out_path, {"meta": meta, "frames": frames})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--variant", default="full", choices=["lite", "full", "heavy"])
    args = ap.parse_args(argv)
    out = extract(args.video, run_dir(args.run_id) / "observations" / "raw_pose.json", args.model, args.variant)
    print(out)


if __name__ == "__main__":
    main()
