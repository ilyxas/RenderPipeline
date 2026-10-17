"""Rotation math for altline: unit quaternions [w, x, y, z] and 3x3 rotation matrices.

Everything here is gimbal-free: no Euler angles are used anywhere in the pipeline.
Matrices store a frame's axes as columns (column 0 = X, 1 = Y, 2 = Z).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float) / np.linalg.norm(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=float))
    h = 0.5 * angle
    return np.concatenate([[np.cos(h)], np.sin(h) * axis])


def quat_angle(q: np.ndarray) -> float:
    """Rotation angle of q in radians, in [0, pi]."""
    return 2.0 * float(np.arccos(np.clip(abs(q[0]), 0.0, 1.0)))


def quat_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle in radians between two orientations (sign-invariant)."""
    rel = quat_mul(quat_conj(a), b)
    return 2.0 * float(np.arctan2(np.linalg.norm(rel[1:]), abs(rel[0])))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns a unit quaternion with w >= 0."""
    m = np.asarray(m, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = quat_normalize(np.array(q))
    return q if q[0] >= 0 else -q


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    return quat_to_mat(quat_from_axis_angle(axis, angle))


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal (swing-only) rotation matrix taking direction a onto direction b."""
    a = normalize(a)
    b = normalize(b)
    c = float(np.dot(a, b))
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degrees: any axis perpendicular to a
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return rot_axis_angle(perp, np.pi)
    return rot_axis_angle(axis / s, float(np.arctan2(s, c)))


def signed_angle_about(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle (radians) rotating u to v about axis, using only components perpendicular to axis."""
    axis = normalize(axis)
    up = u - np.dot(u, axis) * axis
    vp = v - np.dot(v, axis) * axis
    return float(np.arctan2(np.dot(np.cross(up, vp), axis), np.dot(up, vp)))


def swing_twist(q: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose q = swing * twist, twist being a rotation about `axis` (in q's local frame)."""
    axis = normalize(np.asarray(axis, dtype=float))
    proj = np.dot(q[1:], axis) * axis
    twist = np.concatenate([[q[0]], proj])
    n = np.linalg.norm(twist)
    if n < EPS:  # 180-degree swing: twist is undefined, take identity
        twist = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        twist = twist / n
    swing = quat_mul(q, quat_conj(twist))
    return swing, twist


def twist_angle(twist: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle in (-pi, pi] of a pure twist quaternion about axis."""
    s = float(np.dot(twist[1:], normalize(axis)))
    return 2.0 * float(np.arctan2(s, twist[0]))


def quat_log(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of q, shortest arc."""
    q = q if q[0] >= 0 else -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < EPS:
        return 2.0 * v
    return v / s * 2.0 * np.arctan2(s, q[0])


def quat_exp(rv: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rv))
    if angle < EPS:
        return quat_normalize(np.concatenate([[1.0], 0.5 * rv]))
    return quat_from_axis_angle(rv / angle, angle)


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    if np.dot(a, b) < 0:
        b = -b
    rel = quat_mul(quat_conj(a), b)
    return quat_normalize(quat_mul(a, quat_exp(t * quat_log(rel))))


def is_rotation(m: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.allclose(m.T @ m, np.eye(3), atol=tol) and abs(np.linalg.det(m) - 1.0) < tol)
