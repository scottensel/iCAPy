import numpy as np
from functions.TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full
try:
    import cupy as cp  # Import CuPy if available
except ImportError:
    cp = None  # Set to None if CuPy is not available

def evaluate_3D_TV(y, use_cuda=False):

    # Choose backend
    xp = cp if use_cuda and cp is not None else np

    dx, dy, dz = gradient3D_full(y, use_cuda=use_cuda)

    amplitude = xp.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    total_variation = xp.sum(amplitude)

    return total_variation
