import numpy as np
from scipy.signal import lfilter


def filter_boundary(fil_num, fil_den, input_signal, condition='normal'):
    """
    Filters the input with the filter constructed for the differential
    operator L, using zero boundary conditions.

    Inputs:
        fil_num      - 1D array, numerator coefficients (FIR part) obtained
                       from hrf_filters; finite-length filtering
        fil_den      - list of two 1D arrays [causal, non_causal] obtained
                       from hrf_filters; denominator (IIR) part:
                         causal     - first element, causal IIR filter
                         non_causal - second element, non-causal IIR filter
                       Pass 1 (integer) if there is no denominator (FIR only)
        input_signal - 1D array, the input signal to filter
        condition    - str, 'normal' or 'transpose':
                         'normal'    - apply the filter directly
                         'transpose' - apply the conjugate transpose of the
                                       filter (i.e. filter the time-reversed
                                       signal); corresponds to x(-t)

    Outputs:
        out - 1D array, result of the filtering operation

    REMINDERS (from MATLAB implementation):
        - causal filter    : first column of fil_den
        - non-causal filter: second column of fil_den
        - No need to flip the signal in 'transpose' condition with the FIR
          filter because the filter coefficients are already flipped by
          default, i.e. filter_boundary(fliplr(n), d, s, 'transpose')

    Implemented by Isik Karahanoglu
    """

    # Ensure inputs are flat 1D arrays
    fil_num      = np.asarray(fil_num).reshape(-1)
    input_signal = np.asarray(input_signal).reshape(-1)

    # --- FIR filter stage (numerator only) ---
    # 'transpose' condition: filter the time-reversed signal, then reverse
    # the output back — equivalent to the conjugate transpose operation
    if condition.lower() == 'transpose':
        # MATLAB: out = flipud(filter(fil_num, 1, flipud(in)))
        out = lfilter(fil_num, [1], input_signal[::-1])[::-1]
    else:
        # MATLAB: out = filter(fil_num, 1, in)
        out = lfilter(fil_num, [1], input_signal)

    # --- IIR filter stage (denominator) ---
    # Only applied if fil_den has two parts (causal + non-causal)
    if isinstance(fil_den, (list, tuple)) and len(fil_den) == 2:
        causal     = np.atleast_1d(fil_den[0]).reshape(-1)
        non_causal = np.atleast_1d(fil_den[1]).reshape(-1)

        # Only filter if at least one part has more than one coefficient;
        # if both are length 1, the combined filter is trivial
        if (len(causal) + len(non_causal)) > 2:

            # shiftnc determines the zero-padding length needed at each end
            # to implement zero boundary conditions for the non-causal filter
            shiftnc = len(non_causal) - 1

            # Zero boundary padding: extend the signal with zeros at both ends
            if shiftnc > 0:
                out = np.pad(out, (shiftnc, shiftnc), mode='constant')

            if condition.lower() == 'normal':
                # MATLAB: out = filter(1, causal, out)   — causal IIR pass
                out = lfilter([1], causal, out)
                # MATLAB: out = flipud(filter(1, non_causal, flipud(out))) * non_causal(end)
                # non-causal IIR pass: filter the time-reversed signal, then
                # reverse back and scale by the last non-causal coefficient
                out = lfilter([1], non_causal, out[::-1])[::-1] * non_causal[-1]
                # MATLAB: out = out(2*shiftnc+1:end)  — remove leading padding
                if shiftnc > 0:
                    out = out[2 * shiftnc:]

            elif condition.lower() == 'transpose':
                # MATLAB: out = flipud(filter(1, causal, flipud(out)))
                # transpose of causal IIR = anti-causal filtering
                out = lfilter([1], causal, out[::-1])[::-1]
                # MATLAB: out = filter(1, non_causal, out) * non_causal(end)
                # transpose of non-causal IIR = causal filtering
                out = lfilter([1], non_causal, out) * non_causal[-1]
                # MATLAB: out = out(1:end-2*shiftnc)  — remove trailing padding
                if shiftnc > 0:
                    out = out[:-2 * shiftnc]

    return out