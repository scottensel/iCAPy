import numpy as np
try:
    import cupy as cp  # Import CuPy if available
except ImportError:
    cp = None  # Set to None if CuPy is not available
def div3D_full(dx, dy, dz, wx=None, wy=None, wz=None, use_cuda=False):
    """
    Compute the divergence of a 3D or 4D gradient field.

    Parameters:
    - dx, dy, dz: 3D or 4D numpy arrays representing the gradient components.
    - wx, wy, wz: Optional weights for the x, y, and z components.

    Returns:
    - V: The divergence of the input gradient field.
    """
    # Choose backend
    xp = cp if use_cuda and cp is not None else np

    # Apply weights if provided
    if wx is not None:
        dx *= xp.array(wx)
    if wy is not None:
        dy *= xp.array(wy)
    if wz is not None:
        dz *= xp.array(wz)

    # Calculate divergence for 3D or 4D input
    if dx.ndim == 3:
        div_x = xp.pad(dx[1:, :, :] - dx[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='constant')
        div_y = xp.pad(dy[:, 1:, :] - dy[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='constant')
        div_z = xp.pad(dz[:, :, 1:] - dz[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='constant')
    elif dx.ndim == 4:
        div_x = xp.pad(dx[1:, :, :, :] - dx[:-1, :, :, :], ((1, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        div_y = xp.pad(dy[:, 1:, :, :] - dy[:, :-1, :, :], ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant')
        div_z = xp.pad(dz[:, :, 1:, :] - dz[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0)), mode='constant')
    else:
        raise ValueError("Input arrays dx, dy, and dz must be 3D or 4D.")

    # Sum components to get the divergence
    V = div_x + div_y + div_z

    return V