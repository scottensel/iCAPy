# from scipy.signal import lfilter
#
# import numpy as np
# try:
#     import cupy as cp  # Import CuPy if available
# except ImportError:
#     cp = None  # Set to None if CuPy is not available
#
# def filter_boundary(fil_num, fil_den, input_signal, condition, use_cuda=False):
#     """
#     Apply a boundary filter with zero boundary conditions.
#
#     Parameters:
#     - fil_num: Numerator coefficients for the filter (1D array).
#     - fil_den: List containing causal and non-causal filter coefficients.
#     - input_signal: Input signal to be filtered (1D array).
#     - condition: 'normal' or 'transpose' (to apply the filter in reverse).
#
#     Returns:
#     - out: Filtered signal with boundary conditions applied.
#     """
#     # Select backend: CuPy if `use_cuda` is True, otherwise NumPy
#     xp = cp if use_cuda and cp is not None else np
#
#     # Check if `fil_num` or `input_signal` are empty and raise an informative error
#     if fil_num is None or len(fil_num) == 0:
#         raise ValueError("Filter numerator coefficients (fil_num) cannot be empty.")
#     if input_signal is None or len(input_signal) == 0:
#         raise ValueError("Input signal cannot be empty.")
#
#     # Ensure all arrays are on the GPU if use_cuda is True
#     input_signal = xp.array(input_signal)
#     fil_num = xp.array(fil_num)
#     if isinstance(fil_den, list):
#         fil_den = [xp.array(f) for f in fil_den]
#
#     # Apply numerator filter
#     if condition.lower() == 'transpose':
#         out = xp.flipud(xp.convolve(fil_num, xp.flipud(input_signal), mode='full')[:input_signal.size])
#     else:
#         out = xp.convolve(fil_num, input_signal, mode='full')[:input_signal.size]
#
#     # Apply causal and non-causal filters if provided
#     if len(fil_den) == 2:
#         causal = fil_den[0]
#         non_causal = fil_den[1]
#
#         if len(causal) > 1 or len(non_causal) > 1:
#             shiftnc = len(non_causal) - 1
#             out = xp.concatenate([xp.zeros(shiftnc), out, xp.zeros(shiftnc)])  # zero boundary padding
#
#             if condition.lower() == 'normal':
#                 out = xp.convolve(causal, out, mode='full')[:out.size]
#                 out = xp.flipud(xp.convolve(non_causal, xp.flipud(out), mode='full')[:out.size]) * non_causal[-1]
#                 out = out[2 * shiftnc:]
#             elif condition.lower() == 'transpose':
#                 out = xp.flipud(xp.convolve(causal, xp.flipud(out), mode='full')[:out.size])
#                 out = xp.convolve(non_causal, out, mode='full')[:out.size] * non_causal[-1]
#                 if shiftnc != 0:
#                    out = out[: -2 * shiftnc]  # Slice as intended when `shiftnc` is non-zero
#
#
#     return out.get() if use_cuda and cp is not None else out

import numpy as np
from scipy.signal import lfilter

def filter_boundary(fil_num, fil_den, input_signal, condition='normal'):
    """
    Python equivalent of MATLAB filter_boundary.m
    (Implements zero-boundary conditions)

    Parameters
    ----------
    fil_num : 1D array-like
        Numerator coefficients (from hrf_filters)
    fil_den : list of two 1D arrays
        Denominator parts: [causal, non_causal]
    input_signal : 1D array-like
        Input signal (like 'in' in MATLAB)
    condition : str
        'normal' or 'transpose'

    Returns
    -------
    out : 1D numpy array
        Filtered signal
    """

    # Convert to 1D arrays
    fil_num = np.asarray(fil_num).reshape(-1)
    input_signal = np.asarray(input_signal).reshape(-1)
    out = None  # initialize

    # --- FIR filter stage ---
    if condition.lower() == 'transpose':
        # out = flipud(filter(fil_num, 1, flipud(in)))
        out = lfilter(fil_num, [1], input_signal[::-1])[::-1]
    else:
        # out = filter(fil_num, 1, in)
        out = lfilter(fil_num, [1], input_signal)

    # --- IIR filter stage (denominator) ---
    if isinstance(fil_den, (list, tuple)) and len(fil_den) == 2:
        causal = np.atleast_1d(fil_den[0]).reshape(-1)
        non_causal = np.atleast_1d(fil_den[1]).reshape(-1)

        if (len(causal) + len(non_causal)) > 2:
            shiftnc = len(non_causal) - 1

            # Zero boundary padding
            if shiftnc > 0:
                out = np.pad(out, (shiftnc, shiftnc), mode='constant')

            if condition.lower() == 'normal':
                # out = filter(1, causal, out)
                out = lfilter([1], causal, out)
                # out = flipud(filter(1, non_causal, flipud(out))) * non_causal(end)
                out = lfilter([1], non_causal, out[::-1])[::-1] * non_causal[-1]
                # out = out(2*shiftnc+1:end)
                if shiftnc > 0:
                    out = out[2 * shiftnc :]

            elif condition.lower() == 'transpose':
                # out = flipud(filter(1, causal, flipud(out)))
                out = lfilter([1], causal, out[::-1])[::-1]
                # out = filter(1, non_causal, out) * non_causal(end)
                out = lfilter([1], non_causal, out) * non_causal[-1]
                # out = out(1:end-2*shiftnc)
                if shiftnc > 0:
                    out = out[: -2 * shiftnc]

    return out
