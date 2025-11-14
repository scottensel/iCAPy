import numpy as np
from functions.Utilities.WriteInformation import write_information
import os

def generate_surrogate(TC, path, param, fid=None):
    # Initialize surrogate data matrix
    Surrogate = np.zeros_like(TC)
    n_time_points = param['Dimension'][3]

    # Generate surrogate data by random phase scrambling
    for iter_tc in range(param['NbrVoxels']):
        rand_signal = np.fft.fft(np.random.rand(n_time_points))
        phase_signal = np.angle(rand_signal)
        Surrogate[iter_tc, :] = np.real(np.fft.ifft(np.exp(1j * phase_signal) * np.abs(np.fft.fft(TC[iter_tc, :]))))

    # Log surrogate data generation details
    if fid:
        write_information(fid, f"Surrogate data generated and saved at: {os.path.join(path, 'TA_results', param['title'], 'Surrogate')}...")

    return Surrogate
