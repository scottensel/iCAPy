import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full


def evaluate_3D_TV(y, grad_bufs=None, use_numba=False):
    """
    Computes the 3D L1-norm total variation of a 3D or 4D volume.
    If y is 4D, a 3D algorithm is applied to the first 3 components
    in parallel (4th dimension is time).

    Inputs:
        y         - 3D or 4D array (numpy or cupy)
        grad_bufs - optional tuple (dx, dy, dz) of pre-allocated arrays
                    with the same shape as y; passed to gradient3D_full
                    to avoid allocation in the MyProx hot loop
        use_numba - bool, passed through to gradient3D_full

    Outputs:
        total_variation - scalar, total variation norm of y

    Implemented by Younes Farouj
    """
    # Detect CuPy — use the same module for amplitude computation
    try:
        import cupy as cp
        xp = cp if isinstance(y, cp.ndarray) else np
    except ImportError:
        xp = np

    # Compute the spatial gradient, reusing pre-allocated buffers if provided
    dx, dy, dz = gradient3D_full(y, out=grad_bufs, use_numba=use_numba)

    # Amplitude computed in-place to avoid allocating a separate array
    amplitude = dx ** 2
    amplitude += dy ** 2
    amplitude += dz ** 2
    xp.sqrt(amplitude, out=amplitude)

    # Total variation is the sum of gradient amplitudes over all voxels
    return float(xp.sum(amplitude))
