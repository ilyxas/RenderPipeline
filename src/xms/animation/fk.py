"""Right-handed, metre, Y-up FK; XYZW quaternions; rest @ delta."""
import numpy as np


def quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quat(r):
    # Symmetric eigen formulation is stable at 180 degrees.
    r = np.asarray(r)
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
                  [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
                  [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    q = np.linalg.eigh(k)[1][:, -1]
    return q if q[3] >= 0 else -q


def trs(t, q, s):
    m = np.eye(4)
    m[:3,:3] = quat_matrix(q) @ np.diag(s)
    m[:3,3] = t
    return m


def decompose(m):
    m = np.asarray(m, dtype=float)
    s = np.linalg.norm(m[:3,:3], axis=0)
    r = m[:3,:3] / s
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or np.linalg.det(r) < 0:
        raise ValueError('Unsupported shear/reflection in rest TRS')
    return m[:3,3], matrix_quat(r), s


def evaluate(bundle, sample_index):
    a, m = bundle.arrays, bundle.metadata
    result = np.empty((len(m['joint_names']), 4, 4))
    affected = []
    for j, p in enumerate(a['parent_indices']):
        local = trs(a['rest_translation'][j], a['rest_rotation'][j], a['rest_scale'][j])
        local[:3,:3] = (quat_matrix(a['rest_rotation'][j])
                         @ quat_matrix(a['local_rotation_delta'][sample_index,j])
                         @ np.diag(a['rest_scale'][j]))
        result[j] = (a['armature_transform'] if p < 0 else result[p]) @ local
        affected.append(j == m['root_carrier_index'] or (p >= 0 and affected[p]))
    # Root is a world-space metre offset, applied once to the carrier subtree.
    result[np.asarray(affected), :3, 3] += a['root_translation'][sample_index]
    return result
