"""Stage 0 only: read-only assets, one-frame inference, two-key Blender action.

Uses the invoking Python runtime; never discovers or activates another venv.
Outputs are diagnostic evidence, not observations or AnimationBundles.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def blender_worker(out, scene):
    import bpy
    from mathutils import Quaternion
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    evidence = {
        "version": bpy.app.version_string,
        "objects": {o.name: {"type": o.type, "matrix_world": [list(r) for r in o.matrix_world]} for o in bpy.data.objects},
        "armature": rig.name,
        "bones": {b.name: {"parent": b.parent.name if b.parent else None,
                              "inherit_scale": b.inherit_scale,
                              "use_inherit_rotation": b.use_inherit_rotation,
                              "matrix_local": [list(r) for r in b.matrix_local]} for b in rig.data.bones},
        "shape_keys": {o.name: [k.name for k in o.data.shape_keys.key_blocks]
                       for o in bpy.data.objects if o.type == "MESH" and o.data.shape_keys},
    }
    with bpy.data.libraries.load(str(scene), link=False) as (source, target):
        evidence["scene_collections"] = list(source.collections)
    rig.animation_data_create()
    rig.animation_data.action = None
    for track in rig.animation_data.nla_tracks:
        track.mute = True
    bone = next(b for b in rig.pose.bones if b.bone.use_deform and not b.constraints)
    bone.rotation_mode = "QUATERNION"
    action = bpy.data.actions.new("XMS_Stage0_TwoKeys")
    rig.animation_data.action = action
    bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
    bone.keyframe_insert("rotation_quaternion", frame=1)
    bone.rotation_quaternion = Quaternion((0, 0, 1), 0.35)
    bone.keyframe_insert("rotation_quaternion", frame=2)
    matrices = []
    for frame in (1, 2):
        bpy.context.scene.frame_set(frame)
        evaluated = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
        matrices.append([list(row) for row in evaluated.pose.bones[bone.name].matrix])
    delta = max(abs(a-b) for row_a, row_b in zip(*matrices) for a, b in zip(row_a, row_b))
    evidence.update(diagnostic_bone=bone.name, evaluated_matrices=matrices,
                    evaluated_max_delta=delta,
                    action_api={"slots": hasattr(action, "slots"), "layers": hasattr(action, "layers"),
                                "legacy_fcurves": hasattr(action, "fcurves")})
    Path(out).write_text(json.dumps(evidence, indent=2))
    if delta <= 1e-6:
        raise RuntimeError("Two-key action did not change evaluated bone transform")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("character", "scene", "video", "pose-model", "blender", "ffmpeg", "ffprobe", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"stage": 0, "status": "running", "python": sys.version,
              "python_executable": sys.executable, "parameters": {k: str(v) for k, v in vars(args).items()},
              "checks": {}, "blockers": [], "assets": {}}

    def run(name, command):
        started = time.monotonic()
        result = subprocess.run([str(v) for v in command], capture_output=True, text=True, timeout=180)
        (out / (name + ".log")).write_text(result.stdout + result.stderr)
        report["checks"][name] = {"exit_status": result.returncode, "seconds": time.monotonic()-started,
                                   "log": str(out / (name + ".log"))}
        if result.returncode:
            report["blockers"].append(name + " failed; see log")
        return result

    try:
        for name in ("character", "scene", "video", "pose_model"):
            path = getattr(args, name).resolve()
            if path.is_file():
                report["assets"][name] = {"path": str(path), "sha256_before": sha256(path)}
            else:
                report["blockers"].append("Missing asset: " + str(path))
        modules = {}
        for name in ("numpy", "mediapipe"):
            try:
                modules[name] = importlib.import_module(name)
                report["checks"][name] = {"version": modules[name].__version__, "exit_status": 0}
            except ImportError as error:
                report["checks"][name] = {"exit_status": 1, "error": str(error)}
                report["blockers"].append("Missing runtime dependency: " + name)
        for name in ("ffmpeg", "ffprobe", "blender"):
            run(name + "_version", [getattr(args, name), "--version" if name == "blender" else "-version"])
        probe = run("probe", [args.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", args.video])
        if probe.returncode == 0:
            report["media"] = json.loads(probe.stdout)
            stream = next(s for s in report["media"]["streams"] if s["codec_type"] == "video")
            frame = out / "frame.rgb"
            decoded = run("decode", [args.ffmpeg, "-v", "error", "-i", args.video, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", frame])
            if decoded.returncode == 0:
                expected = stream["width"] * stream["height"] * 3
                if frame.stat().st_size != expected:
                    raise RuntimeError("Decoded RGB shape mismatch (including possible rotation)")
                if len(modules) == 2 and args.pose_model.is_file():
                    np, mp = modules["numpy"], modules["mediapipe"]
                    rgb = np.fromfile(frame, dtype=np.uint8).reshape(stream["height"], stream["width"], 3)
                    options = mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.pose_model)))
                    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
                        pose = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    if not pose.pose_landmarks or len(pose.pose_landmarks[0]) != 33 or len(pose.pose_world_landmarks[0]) != 33:
                        raise RuntimeError("Pose output missing or not 33 image/world landmarks")
                    (out / "pose.json").write_text(json.dumps({"image": [vars(p) for p in pose.pose_landmarks[0]], "world": [vars(p) for p in pose.pose_world_landmarks[0]]}, indent=2))
                    report["checks"]["pose"] = {"exit_status": 0, "shape": [1, 33, 3]}
            run("encode_preflight", [args.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=size=540x960:rate=24", "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", out / "encode_preflight.mp4"])
        if args.character.is_file() and args.scene.is_file():
            run("blender_action", [args.blender, "-b", args.character, "--python-exit-code", "1", "--python", Path(__file__).resolve(), "--", "--blender-worker", out / "blender.json", args.scene])
    except Exception as error:
        report["blockers"].append(type(error).__name__ + ": " + str(error))
    finally:
        for asset in report["assets"].values():
            asset["sha256_after"] = sha256(asset["path"])
            asset["unchanged"] = asset["sha256_before"] == asset["sha256_after"]
            if not asset["unchanged"]:
                report["blockers"].append("Source asset changed: " + asset["path"])
        report["status"] = "blocked" if report["blockers"] else "passed"
        (out / "manifest.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"status": report["status"], "manifest": str(out / "manifest.json"), "blockers": report["blockers"]}, indent=2))
    return 1 if report["blockers"] else 0


if __name__ == "__main__":
    if "--blender-worker" in sys.argv:
        index = sys.argv.index("--blender-worker")
        blender_worker(sys.argv[index+1], sys.argv[index+2])
    else:
        sys.exit(main())
