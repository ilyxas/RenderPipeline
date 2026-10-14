"""Pure helpers for the GLB export session: mask selection and skin-influence accounting."""
import numpy as np

# The glTF exporter ignores weights at or below this (io_scene_gltf2 primitive_extract.py).
MIN_INFLUENCE=1e-4


def masked(weights,threshold=0.0,invert=False):
    """Vertices a Blender MASK modifier (vertex-group mode) removes: kept = weight > threshold, flipped by invert."""
    inside=np.asarray(weights,dtype=float)>threshold
    return inside if invert else ~inside


def influences(vertex_weights,limit):
    """vertex_weights: per vertex, the bone weights. Returns counts against the exported limit.

    The exporter keeps the `limit` largest weights and renormalises; vertices with more
    influences lose the rest (truncated)."""
    counts=np.array([sum(1 for w in ws if w>MIN_INFLUENCE) for ws in vertex_weights],dtype=int)
    lost=[sum(sorted((w for w in ws if w>MIN_INFLUENCE),reverse=True)[limit:]) for ws in vertex_weights]
    return {'max_influences_in_asset':int(counts.max()) if len(counts) else 0,'exported_influences':int(limit),
            'truncated_vertices':int((counts>limit).sum()),'vertices':int(len(counts)),
            'max_truncated_weight':round(float(max(lost)) if lost else 0.0,6)}
