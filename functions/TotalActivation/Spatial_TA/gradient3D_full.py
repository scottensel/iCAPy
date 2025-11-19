import numpy as np
try:
    import cupy as cp  # Import CuPy if available
except ImportError:
    cp = None  # Set to None if CuPy is not available

def gradient3D_full(V, wx=None, wy=None, wz=None, use_cuda=False):
    """
    Compute the 3D gradient of a 3D or 4D volume along each axis, with optional weights.

    Parameters:
    - V: 3D or 4D numpy array; if 4D, gradients are computed along the first three dimensions.
    - wx, wy, wz: Optional weight matrices for the x, y, and z gradients.

    Returns:
    - dx, dy, dz: Gradients along x, y, and z axes, weighted if weights are provided.
    """
    # Choose backend
    xp = cp if use_cuda and cp is not None else np

    # Convert input array to the backend
    V = xp.array(V)

    # Check if V is 3D or 4D and apply differences accordingly
    if V.ndim == 3:
        dx = xp.pad(V[1:, :, :] - V[:-1, :, :],
                    ((0, 1), (0, 0), (0, 0)), mode='constant')
        dy = xp.pad(V[:, 1:, :] - V[:, :-1, :],
                    ((0, 0), (0, 1), (0, 0)), mode='constant')
        dz = xp.pad(V[:, :, 1:] - V[:, :, :-1],
                    ((0, 0), (0, 0), (0, 1)), mode='constant')

    elif V.ndim == 4:
        dx = xp.pad(V[1:, :, :, :] - V[:-1, :, :, :],
                    ((0, 1), (0, 0), (0, 0), (0, 0)), mode='constant')
        dy = xp.pad(V[:, 1:, :, :] - V[:, :-1, :, :],
                    ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant')
        dz = xp.pad(V[:, :, 1:, :] - V[:, :, :-1, :],
                    ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')

    else:
        raise ValueError("Input array V must be 3D or 4D.")

    # Apply weights if provided
    if wx is not None:
        dx = dx * wx
    if wy is not None:
        dy = dy * wy
    if wz is not None:
        dz = dz * wz

    return dx, dy, dz