import numpy as np
from functions.n00_Utilities.WriteInformation import write_information
import os

"""
Generates surrogate data on which to apply total activation for the
thresholding process (selecting relevant innovations). The phase of
the surrogate data has been scrambled.

Inputs:
    TC    - (n_ret_voxels x n_time_points) 2D matrix of data to scramble
    path  - path to the subject's data directory
    param - dict containing TA-relevant parameters:
        'Dimension'  - 4-element list/array of X, Y, Z and T sizes
        'NbrVoxels'  - number of retained voxels
        'title'      - title/date when TA was launched for this trial
    fid   - log file handle

Outputs:
    Surrogate - (n_ret_voxels x n_time_points) 2D matrix of surrogate
                data with scrambled phase
"""

def generate_surrogate(TC, path, param, fid=None, seed=None):


    TC = np.asarray(TC)
    V, T = TC.shape
    assert T == param["Dimension"][3]
    assert V == param["NbrVoxels"]

    rng = np.random.default_rng(seed)  # set seed for repeatability

    # output matrix
    Surrogate = np.empty_like(TC, dtype=float)

    for v in range(V):

        # phase_signal is a time x 1 vector filled with random phase
        # information (in rad, from -pi to pi)
        rand_signal = np.fft.fft(rng.random(T), n=T)
        phase_signal = np.angle(rand_signal)

        # We multiply the magnitude of the original data with random phase
        # information to generate surrogate data
        mag = np.abs(np.fft.fft(TC[v, :], n=T))
        Surrogate[v, :] = np.fft.ifft(np.exp(1j * phase_signal) * mag, n=T).real

    # Log surrogate data generation details
    if fid:
        write_information(fid, f"Surrogate data generated and saved at: {os.path.join(path, 'TA_results', param['title'], 'Surrogate')}...")

    return Surrogate
