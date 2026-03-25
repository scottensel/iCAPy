from functions.TotalActivation.Temporal_TA.filter_boundary import filter_boundary
import time
import numpy as np

def ta_temporal_onetimecourse(y, idx_vox, ParametersIn):
    """
    Perform temporal regularization on one voxel time course.

    Parameters:
    - y: 1D numpy array, noisy signal (one voxel time course).
    - idx_vox: int, index of the voxel.
    - ParametersIn: dict, containing necessary parameters for temporal regularization.

    Returns:
    - x: 1D numpy array, result of denoising.
    - ParametersOut: dict, containing updated noise estimates and lambda values.
    """


    # Extract relevant parameters
    y = np.array(y)
    n = ParametersIn['filter_analyze']['num']
    d = ParametersIn['filter_analyze']['den']
    maxeig = ParametersIn['maxeig']
    N = ParametersIn['Dimension'][3]
    Nit = ParametersIn['NitTemp']

    # Determine the initial lambda
    if 'NoiseEstimateFin' in ParametersIn and len(ParametersIn['NoiseEstimateFin'])-1 >= idx_vox: # has to be like this since the field doesnt exist so we must skip over this for the start
        lambda_ = ParametersIn['NoiseEstimateFin'][idx_vox]
    else:
        lambda_ = ParametersIn['LambdaTemp'][idx_vox]

    noise_estimate = np.array(ParametersIn['LambdaTemp'][idx_vox], dtype=np.float64)

    nv = np.zeros(Nit)

    Lambda = np.zeros(Nit)

    precision = noise_estimate / 100000

    # Dual variable initialization
    z = np.zeros(N)
    s = np.zeros(N)

    t = np.array(1)

    filtered_input = filter_boundary(n, d, y, 'normal')
    for k in range(Nit):
            z_prev = z.copy()

            # Dual variable update using filter_boundary
            filtered_transpose = filter_boundary(n, d, s, 'transpose')
            filtered_s = filter_boundary(n, d, filtered_transpose, 'normal')

            z = (1 / (lambda_ * maxeig)) * filtered_input + s - filtered_s / (maxeig)
            z = np.clip(z, -1, 1)

            t_prev = t
            t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            s = z + ((t_prev - 1) / t) * (z - z_prev)

            nv[k] = np.sqrt(np.mean((lambda_ * filter_boundary(n, d, z, 'transpose')) ** 2))

            if np.abs(nv[k] - noise_estimate) > precision:
                lambda_ *= noise_estimate / (nv[k])

            Lambda[k] = lambda_


    # Primal variable estimation after convergence
    x = y - lambda_ * filter_boundary(n, d, z, 'transpose')

    # Ensure that CuPy is available before checking if `nv` and `Lambda` are `CuPy` arrays
    ParametersOut = {
        'NoiseEstimateFin': nv[-1],
        'LambdasTempFin': Lambda[-1]
    }

    # Return result and parameters
    return x, ParametersOut
