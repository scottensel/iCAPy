import numpy as np
from scipy.signal import convolve
from functions.TotalActivation.Temporal_TA.TA_Temporal_OneTimeCourse import ta_temporal_onetimecourse
import time
from tqdm import tqdm


def ta_temporal(TCN, param):
    # Initialize output and lambda arrays
    TC_OUT = np.zeros((param['Dimension'][3], param['NbrVoxels']))
    param['LambdaTemp'] = np.zeros(param['NbrVoxels'])

    # Define high-pass Daubechies wavelet filter for noise estimation
    g = np.array([0, -0.12941, -0.22414, 0.83652, -0.48296])

    # start_time = time.time()

   # Iterate over each voxel for temporal regularization
    for i in tqdm(range(param['NbrVoxels']), desc="Processing temporal step", ncols=80):
    # for i in range(param['NbrVoxels']):
        # Step 1: Wavelet decomposition for the current voxel time course

        # coef = convolve(TCN[:, i], g, mode='same')
        coef = cconv(TCN[:, i], g)

        # Step 2: Estimate noise level using median absolute deviation
        param['LambdaTemp'][i] = np.median(np.abs(coef - np.median(coef))) * param['LambdaTempCoef']

        # Step 3: Perform temporal regularization for the voxel
        TC_OUT[:, i], param_out = ta_temporal_onetimecourse(TCN[:, i], i, param)

        # Store the final regularization and noise estimates
        if 'NoiseEstimateFin' not in param and 'LambdaTempFin' not in param:
            param['LambdaTempFin'] = [param_out['LambdasTempFin']]
            param['NoiseEstimateFin'] = [param_out['NoiseEstimateFin']]
        elif len(param['NoiseEstimateFin'])-1 >= i:
            param['LambdaTempFin'][i] = param_out['LambdasTempFin']
            param['NoiseEstimateFin'][i] = param_out['NoiseEstimateFin']
        else:
            param['LambdaTempFin'].append(param_out['LambdasTempFin'])
            param['NoiseEstimateFin'].append(param_out['NoiseEstimateFin'])

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.6f} seconds ta_temporal")

    return TC_OUT, param


def cconv(a, b, N=None):
    """
    Circular convolution (mod-N) of 1D vectors a and b.

    Parameters
    ----------
    a, b : array_like
        Input sequences (real or complex). Will be flattened.
    N : int, optional
        Output length. If None, defaults to len(a) + len(b) - 1
        (same default as MATLAB's cconv).

    Returns
    -------
    c : ndarray
        Circular convolution of length N.
    """
    # a = np.asarray(a).ravel()
    # b = np.asarray(b).ravel()

    if N is None:
        N = a.size + b.size - 1

    # FFT-based circular convolution of length N
    c = np.fft.ifft(np.fft.fft(a, N) * np.fft.fft(b, N))

    # If both inputs are real, return real result (like MATLAB)
    if np.isrealobj(a) and np.isrealobj(b):
        c = c.real

    return c