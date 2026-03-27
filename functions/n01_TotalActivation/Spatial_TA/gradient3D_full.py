import numpy as np


def gradient3D_full(V, wx=None, wy=None, wz=None):
    """
    Computes the 3D gradient of a 3D or 4D volume along each axis.
    Particularly suitable for fMRI data where the 4th dimension refers
    to time; a 3D algorithm is applied to the first 3 components.

    Inputs:
        V        - 3D or 4D array; if 4D, gradients are computed along
                   the first three spatial dimensions only
        wx, wy, wz - optional weight matrices along the X, Y and Z
                   directions; only applied if provided

    Outputs:
        dx, dy, dz - gradient arrays along X, Y and Z axes respectively;
                     zero-padded at the end of each dimension to preserve
                     the input shape

    Implemented by Younes Farouj, 12.03.2016
    Weight option added by Younes Farouj, 28.04.2016
    """
    V = np.array(V)

    # Computes the difference between neighbouring elements of the volume
    # and pads with a zero at the end of each spatial dimension.
    # dx is the gradient along X, dy along Y, dz along Z.
    if V.ndim == 3:
        dx = np.pad(V[1:, :, :] - V[:-1, :, :],
                    ((0, 1), (0, 0), (0, 0)), mode='constant')
        dy = np.pad(V[:, 1:, :] - V[:, :-1, :],
                    ((0, 0), (0, 1), (0, 0)), mode='constant')
        dz = np.pad(V[:, :, 1:] - V[:, :, :-1],
                    ((0, 0), (0, 0), (0, 1)), mode='constant')

    elif V.ndim == 4:
        dx = np.pad(V[1:, :, :, :] - V[:-1, :, :, :],
                    ((0, 1), (0, 0), (0, 0), (0, 0)), mode='constant')
        dy = np.pad(V[:, 1:, :, :] - V[:, :-1, :, :],
                    ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant')
        dz = np.pad(V[:, :, 1:, :] - V[:, :, :-1, :],
                    ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')

    else:
        raise ValueError("Input array V must be 3D or 4D.")

    # Only used if weight matrices are provided (not the case by default)
    if wx is not None:
        dx = dx * wx
    if wy is not None:
        dy = dy * wy
    if wz is not None:
        dz = dz * wz

    return dx, dy, dz
