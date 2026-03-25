from functions.TotalActivation.Temporal_TA.filter_boundary import filter_boundary
import time
import numpy as np
try:
    import cupy as cp  # Import CuPy if available
except ImportError:
    cp = None  # Set to None if CuPy is not available

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
    use_cuda = ParametersIn['use_cuda']

    # Select backend: CuPy if `use_cuda` is True, otherwise NumPy
    xp = cp if use_cuda and cp is not None else np

    # Extract relevant parameters
    y = xp.array(y)
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

    # Initialize parameters for the optimization
    # Initialize lambda as float64
    # lambda_ = xp.array(ParametersIn['LambdaTemp'][idx_vox], dtype=xp.float64) \
    #     if 'NoiseEstimateFin' not in ParametersIn else xp.array(ParametersIn['NoiseEstimateFin'][idx_vox],
    #                                                             dtype=xp.float64)

    noise_estimate = xp.array(ParametersIn['LambdaTemp'][idx_vox], dtype=xp.float64)
    nv = xp.zeros(Nit)
    Lambda = xp.zeros(Nit)
    precision = noise_estimate / 100000

    # Dual variable initialization
    z = xp.zeros(N)
    s = xp.zeros(N)

    t = xp.array(1)

    for k in range(Nit):
            z_prev = z.copy()

            # Dual variable update using filter_boundary
            filtered_input = filter_boundary(n, d, y, 'normal')
            filtered_transpose = filter_boundary(n, d, s, 'transpose')
            filtered_s = filter_boundary(n, d, filtered_transpose, 'normal')

            z = (1 / (lambda_ * maxeig)) * filtered_input + s - filtered_s / (maxeig)
            z = xp.clip(z, -1, 1)

            t_prev = t
            t = (1 + xp.sqrt(1 + 4 * t ** 2)) / 2
            s = z + ((t_prev - 1) / t) * (z - z_prev)

            nv[k] = xp.sqrt(xp.mean((lambda_ * filter_boundary(n, d, z, 'transpose')) ** 2))

            if xp.abs(nv[k] - noise_estimate) > precision:
                lambda_ *= noise_estimate / (nv[k])

            Lambda[k] = lambda_


    # Primal variable estimation after convergence
    x = y - lambda_ * filter_boundary(n, d, z, 'transpose')

    # Ensure that CuPy is available before checking if `nv` and `Lambda` are `CuPy` arrays
    ParametersOut = {
        'NoiseEstimateFin': nv[-1].get() if use_cuda and cp is not None and isinstance(nv, cp.ndarray) else nv[-1],
        'LambdasTempFin': Lambda[-1].get() if use_cuda and cp is not None and isinstance(Lambda, cp.ndarray) else
        Lambda[-1]
    }

    # Check if CuPy is available and x is a CuPy array before calling .get()
    x = x.get() if use_cuda and cp is not None and isinstance(x, cp.ndarray) else x

    # Return result and parameters
    return x, ParametersOut
