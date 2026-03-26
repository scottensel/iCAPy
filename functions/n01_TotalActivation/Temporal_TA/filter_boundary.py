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
