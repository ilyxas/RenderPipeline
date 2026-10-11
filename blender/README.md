# Blender entry points

Blender-side scripts load registered assets, apply a validated `AnimationBundle`, evaluate the rig, render frames, export the animated GLB (`export_glb.py`) and inspect surfaces. They must not perform tracking, solve IK, fuse face/audio signals, or mutate canonical assets (export sessions block saving).
