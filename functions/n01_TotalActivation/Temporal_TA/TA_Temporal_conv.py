import numpy as np
from functions.n01_TotalActivation.Temporal_TA.TA_Temporal_OneTimeCourse import ta_temporal_onetimecourse


def ta_temporal(TCN, param):
    """
    Performs the temporal regularization part of total activation,
    voxel after voxel.

    Inputs:
        TCN   - (n_time_points x n_ret_voxels) 2D matrix of data input
                to the regularization
        param - dict containing all TA-relevant parameters:
            'Dimension'       - 4-element list/array of X, Y, Z, T sizes;
                                T (index 3) gives the number of time points
            'NbrVoxels'       - number of voxels to consider for
                                regularization
            'LambdaTempCoef'  - coefficient used to scale the noise
                                estimate when computing the initial
                                regularization parameter per voxel

    Outputs:
        TC_OUT - (n_time_points x n_ret_voxels) 2D matrix of outputs
                 from the regularization step
        param  - updated dict with the following fields added:
            'LambdaTemp'      - (n_voxels,) initial regularization
                                parameter per voxel (MAD-based estimate)
            'LambdaTempFin'   - (n_voxels,) final regularization estimate
                                for each voxel after convergence
            'NoiseEstimateFin'- (n_voxels,) final noise estimate for each
                                voxel after convergence

    Implemented by Isik Karahanoglu, 28.11.2011
    """

    # The output from the algorithm (time x voxels) is initialised as
    # a matrix of zeros
    TC_OUT = np.zeros((param['Dimension'][3], param['NbrVoxels']))

    # LambdaTemp contains the values of regularization parameters for each
    # voxel; also set to zero for now
    param['LambdaTemp'] = np.zeros(param['NbrVoxels'])

    # The noise estimation procedure uses a single-scale wavelet
    # decomposition. Here Daubechies wavelets with 4 vanishing moments are
    # used. The corresponding high-pass filter is given by:
    g = np.array([0, -0.12941, -0.22414, 0.83652, -0.48296])

    # We iterate through all voxels to solve the problem
    for i in range(param['NbrVoxels']):

        # Initialisation of the regularization parameter for the
        # considered voxel, in two steps:

        # Step 1: Wavelet decomposition of the time course of the voxel
        # of interest using circular convolution with the high-pass filter
        coef = cconv(TCN[:, i], g)

        # Step 2: Median absolute deviation (sum of absolute valued
        # distances of coefficients from the median).
        # From the MATLAB page on wavelet denoising:
        # "The median absolute deviation of the coefficients is a robust
        # estimate of noise."
        # This is thus our estimate of the noise level for the considered
        # voxel time course.
        param['LambdaTemp'][i] = np.median(np.abs(coef - np.median(coef))) * param['LambdaTempCoef']

        # Now that we have estimated our initial lambda (regularization
        # parameter), ta_temporal_onetimecourse performs the computations
        # for the considered time course TCN[:, i] of voxel i
        TC_OUT[:, i], param_out = ta_temporal_onetimecourse(TCN[:, i], i, param)

        # Takes the final estimates of regularization parameter and
        # effective noise for voxel i, and stores them in the param dict
        # for warm-starting the next TA iteration
        if 'NoiseEstimateFin' not in param and 'LambdaTempFin' not in param:
            param['LambdaTempFin']    = [param_out['LambdasTempFin']]
            param['NoiseEstimateFin'] = [param_out['NoiseEstimateFin']]
        elif len(param['NoiseEstimateFin']) - 1 >= i:
            param['LambdaTempFin'][i]    = param_out['LambdasTempFin']
            param['NoiseEstimateFin'][i] = param_out['NoiseEstimateFin']
        else:
            param['LambdaTempFin'].append(param_out['LambdasTempFin'])
            param['NoiseEstimateFin'].append(param_out['NoiseEstimateFin'])

    return TC_OUT, param


def cconv(a, b, N=None):
    """
    Circular convolution (mod-N) of two 1D vectors a and b.

    Equivalent to MATLAB's cconv(a, b, N). Used here to apply the
    Daubechies high-pass wavelet filter to each voxel time course for
    noise estimation via MAD of wavelet coefficients.

    Inputs:
        a, b - 1D array_like, input sequences (real or complex)
        N    - int, optional; output length. If None, defaults to
               len(a) + len(b) - 1, matching MATLAB's cconv default

    Outputs:
        c - (N,) ndarray, circular convolution result; real-valued if
            both inputs are real
    """
    if N is None:
        N = a.size + b.size - 1

    # FFT-based circular convolution of length N
    c = np.fft.ifft(np.fft.fft(a, N) * np.fft.fft(b, N))

    # If both inputs are real, discard negligible imaginary components
    if np.isrealobj(a) and np.isrealobj(b):
        c = c.real

    return c