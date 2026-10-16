"""Target armature reference for altline.

The real Xandra rig (rig.npz from blender/) is off-limits for this workspace, so altline
ships its own simplified 20-bone reference hierarchy with constant bone lengths. It is
stored in the same format a real rig export must use, so swapping it is a data change,
not a code change (see altline/README.md, "Replacing the reference rig").

Space conventions ("canonical space", used by every stage after calibration):
    +X = character's left, +Y = up, +Z = character's forward. Right-handed, meters.

Each bone has a rest matrix whose columns are the bone's local axes in canonical space:
    Y = along the bone (head -> tail),
    Z = the "anatomical front" of the bone: a hint vector made perpendicular to Y
        (forward for trunk, arms and legs; up for feet),
    X = Y x Z.
With this convention, elbow flexion is a positive rotation about the forearm's local X,
and knee flexion is a negative rotation about the shin's local X.

Pose model (Blender's): for bone b with parent p
    pose_world[b] = pose_world[p] @ rest_local[b] @ basis[b],   rest_local[b] = rest[p]^T rest[b]
so `basis` is exactly what Blender stores in pose_bone.rotation_quaternion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .quat import normalize

FWD = (0.0, 0.0, 1.0)
UP = (0.0, 1.0, 0.0)

# name, parent, head, tail, roll hint. Right side is mirrored from left (x -> -x).
_A_POSE_DEG = 40.0  # arm angle from straight-down, in the frontal plane


def _arm_dir() -> tuple[float, float, float]:
    a = np.radians(_A_POSE_DEG)
    return (float(np.sin(a)), float(-np.cos(a)), 0.0)


def _reference_bones() -> list[tuple[str, str | None, tuple, tuple, tuple]]:
    d = np.array(_arm_dir())
    sh = np.array([0.17, 1.36, 0.0])  # clavicles are horizontal at rest
    el = sh + 0.28 * d
    wr = el + 0.25 * d
    ha = wr + 0.08 * d
    left = [
        ("clavicle_l", "chest", (0.0, 1.36, 0.0), tuple(sh), FWD),
        ("upper_arm_l", "clavicle_l", tuple(sh), tuple(el), FWD),
        ("forearm_l", "upper_arm_l", tuple(el), tuple(wr), FWD),
        ("hand_l", "forearm_l", tuple(wr), tuple(ha), FWD),
        ("thigh_l", "pelvis", (0.09, 0.95, 0.0), (0.09, 0.52, 0.0), FWD),
        ("shin_l", "thigh_l", (0.09, 0.52, 0.0), (0.09, 0.10, 0.0), FWD),
        ("foot_l", "shin_l", (0.09, 0.10, 0.0), (0.09, 0.02, 0.15), UP),
    ]
    right = []
    for name, parent, head, tail, hint in left:
        mirror = lambda p: (-p[0], p[1], p[2])  # noqa: E731
        right.append((name[:-2] + "_r", parent.replace("_l", "_r"), mirror(head), mirror(tail), hint))
    trunk = [
        ("pelvis", None, (0.0, 0.95, 0.0), (0.0, 1.05, 0.0), FWD),
        ("spine", "pelvis", (0.0, 1.05, 0.0), (0.0, 1.20, 0.0), FWD),
        ("chest", "spine", (0.0, 1.20, 0.0), (0.0, 1.36, 0.0), FWD),
        ("neck", "chest", (0.0, 1.36, 0.0), (0.0, 1.48, 0.0), FWD),
        ("head", "neck", (0.0, 1.48, 0.0), (0.0, 1.68, 0.0), FWD),
    ]
    return trunk + left + right


def frame_from_hint(y_dir: np.ndarray, z_hint: np.ndarray) -> np.ndarray:
    y = normalize(np.asarray(y_dir, dtype=float))
    z = np.asarray(z_hint, dtype=float)
    z = normalize(z - np.dot(z, y) * y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


@dataclass
class Skeleton:
    names: list[str]
    parents: np.ndarray            # (N,) int, -1 for root; parents precede children
    heads: np.ndarray              # (N, 3) rest heads, canonical space
    tails: np.ndarray              # (N, 3) rest tails, canonical space
    rest: np.ndarray               # (N, 3, 3) rest frames (anatomical convention above)
    lengths: np.ndarray = field(init=False)
    index: dict[str, int] = field(init=False)
    # Per-bone conjugation from the anatomical frame to the rig's own bone frame. Identity for
    # the reference rig; for a real rig with its own bone rolls, rig_basis = O^T anat_basis O.
    rig_offset: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.lengths = np.linalg.norm(self.tails - self.heads, axis=1)
        self.index = {n: i for i, n in enumerate(self.names)}
        for i, p in enumerate(self.parents):
            if p >= i:
                raise ValueError(f"bone {self.names[i]} appears before its parent")
        if self.rig_offset is None:
            self.rig_offset = np.repeat(np.eye(3)[None], len(self.names), axis=0)

    def rest_local(self, i: int) -> np.ndarray:
        p = self.parents[i]
        return self.rest[i] if p < 0 else self.rest[p].T @ self.rest[i]

    def fk(self, basis: np.ndarray, root_pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward kinematics from (N,3,3) local basis matrices.

        Returns (world_rotations (N,3,3), heads (N,3), tails (N,3)). Bone lengths are the
        rest lengths by construction; only rotations change.
        """
        n = len(self.names)
        world = np.zeros((n, 3, 3))
        heads = np.zeros((n, 3))
        tails = np.zeros((n, 3))
        for i in range(n):
            p = self.parents[i]
            if p < 0:
                world[i] = self.rest[i] @ basis[i]
                heads[i] = self.heads[i] if root_pos is None else root_pos
            else:
                world[i] = world[p] @ self.rest_local(i) @ basis[i]
                offset = self.rest[p].T @ (self.heads[i] - self.heads[p])
                heads[i] = heads[p] + world[p] @ offset
            tails[i] = heads[i] + world[i][:, 1] * self.lengths[i]
        return world, heads, tails

    def save_npz(self, path: Path) -> None:
        np.savez(
            path,
            names=np.array(self.names),
            parents=self.parents,
            heads=self.heads,
            tails=self.tails,
            rest=self.rest,
            rig_offset=self.rig_offset,
        )


def reference_skeleton() -> Skeleton:
    bones = _reference_bones()
    names = [b[0] for b in bones]
    idx = {n: i for i, n in enumerate(names)}
    parents = np.array([-1 if b[1] is None else idx[b[1]] for b in bones])
    heads = np.array([b[2] for b in bones], dtype=float)
    tails = np.array([b[3] for b in bones], dtype=float)
    rest = np.array([frame_from_hint(t - h, hint) for (_, _, h, t, hint), h, t in zip(bones, heads, tails)])
    return Skeleton(names, parents, heads, tails, rest)


ROLL_HINTS = {name: hint for name, _, _, _, hint in _reference_bones()}


def load_skeleton(path: Path | None) -> Skeleton:
    """Load a rig export, or the built-in reference rig when path is None.

    A rig export needs names/parents/heads/tails in canonical space for the driven bones.
    Anatomical frames are rebuilt from heads/tails with the same roll hints as the reference
    rig; if the export carries `rest` (the rig's own bone frames), the per-bone offset
    O = anat_rest^T rig_rest is kept so the exported basis matches the rig's bone rolls.
    """
    if path is None:
        return reference_skeleton()
    data = np.load(path, allow_pickle=False)
    names = [str(n) for n in data["names"]]
    heads = np.asarray(data["heads"], dtype=float)
    tails = np.asarray(data["tails"], dtype=float)
    anat = np.array([frame_from_hint(t - h, ROLL_HINTS.get(n, FWD)) for n, h, t in zip(names, heads, tails)])
    offset = None
    if "rest" in data.files:
        offset = np.einsum("nji,njk->nik", anat, np.asarray(data["rest"], dtype=float))
    return Skeleton(names, np.asarray(data["parents"], dtype=int), heads, tails, anat, rig_offset=offset)
