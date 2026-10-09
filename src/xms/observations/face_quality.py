"""Image-quality proxies, kept distinct from expression strength."""
import numpy as np


def assess(crop):
    if crop.size==0 or min(crop.shape[:2])<32:return 0.
    gray=crop.astype(float).mean(axis=2)/255
    sharpness=np.mean(abs(np.diff(gray,axis=0)))+np.mean(abs(np.diff(gray,axis=1)))
    return float(min(1,min(crop.shape[:2])/128)*min(1,sharpness/.012))


def geometry_score(matrix):
    singular=np.linalg.svd(matrix[:3,:3],compute_uv=False)
    return float(np.clip(singular[-1]/max(singular[0],1e-8),0,1))
