"""Optional large-mesh signed distance backend. No implicit install."""
import numpy as np


def signed_distance(points,vertices,triangles):
    try:import igl
    except ImportError as error:raise RuntimeError('libigl unavailable; use Blender BVH diagnostics') from error
    return igl.signed_distance(np.asarray(points,float),np.asarray(vertices,float),np.asarray(triangles,int))[0]
