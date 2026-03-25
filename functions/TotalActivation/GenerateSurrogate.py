import numpy as np
from functions.Utilities.WriteInformation import write_information
import os

# def generate_surrogate(TC, path, param, fid=None):
#     # Initialize surrogate data matrix
#     Surrogate = np.zeros_like(TC)
#     n_time_points = param['Dimension'][3]
#
#     # Generate surrogate data by random phase scrambling
#     for iter_tc in range(param['NbrVoxels']):
#         rand_signal = np.fft.fft(np.random.rand(n_time_points))
#         phase_signal = np.angle(rand_signal)
#         Surrogate[iter_tc, :] = np.real(
#             np.fft.ifft(
#                 np.exp(1j * phase_signal) *
#                 np.abs(np.fft.fft(TC[iter_tc, :], n=n_time_points)),
#                 n=n_time_points
#             )
#         )
#     # Log surrogate data generation details
#     if fid:
#         write_information(fid, f"Surrogate data generated and saved at: {os.path.join(path, 'TA_results', param['title'], 'Surrogate')}...")
#
#     return Surrogate
#
# import numpy as np

def generate_surrogate(TC, path, param, fid=None, seed=None):
    # TC: (NbrVoxels, T)
    TC = np.asarray(TC)
    V, T = TC.shape
    assert T == param["Dimension"][3]
    assert V == param["NbrVoxels"]

    rng = np.random.default_rng(seed)  # set seed for repeatability

    Surrogate = np.empty_like(TC, dtype=float)

    for v in range(V):
        # MATLAB: rand_signal = fft(rand(T,1), T); phase_signal = angle(rand_signal)
        rand_signal = np.fft.fft(rng.random(T), n=T)
        phase_signal = np.angle(rand_signal)

        mag = np.abs(np.fft.fft(TC[v, :], n=T))
        Surrogate[v, :] = np.fft.ifft(np.exp(1j * phase_signal) * mag, n=T).real

    # Log surrogate data generation details
    if fid:
        write_information(fid, f"Surrogate data generated and saved at: {os.path.join(path, 'TA_results', param['title'], 'Surrogate')}...")

    return Surrogate
