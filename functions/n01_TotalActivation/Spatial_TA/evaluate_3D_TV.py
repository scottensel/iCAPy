import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full


def evaluate_3D_TV(y):
    """
    Computes the 3D L1-norm total variation of a 3D or 4D volume.
    If y is 4D, a 3D algorithm is applied to the first 3 components
    in parallel (4th dimension is time).

    Inputs:
        y - 3D or 4D array

    Outputs:
        total_variation - scalar, total variation norm of y

    Implemented by Younes Farouj
    """
    # Compute the spatial gradient along all three dimensions
    dx, dy, dz = gradient3D_full(y)

    # Amplitude of the gradient at each voxel (L2 norm of the gradient vector)
    amplitude = np.sqrt(abs(dx) ** 2 + abs(dy) ** 2 + abs(dz) ** 2)

    # Total variation is the sum of gradient amplitudes over all voxels
    total_variation = np.sum(amplitude)

    return total_variation
