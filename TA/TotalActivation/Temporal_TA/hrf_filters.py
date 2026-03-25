import numpy as np
from scipy.signal import freqz
from functions.TotalActivation.Temporal_TA.cons_filter import cons_filter


def hrf_filters(param):
    """
    Generates filters linked to the hemodynamic response function.

    Parameters:
    - param: Dictionary containing the fields:
        - 'HRF': Type of HRF ('bold', 'spmhrf', or 'mion').
        - 'TR': Repetition time of the data.

    Returns:
    - param: Updated dictionary with added filter configurations and max eigenvalue.
    """

    TR = param['TR']  # Time repetition

    # Configure parameters based on the HRF type
    if param['HRF'] == 'bold':
        # Parameters for 'bold' HRF
        eps = 0.54
        ts = 1.54
        tf = 2.46
        t0 = 0.98
        alpha = 0.33
        E0 = 0.34
        V0 = 1
        k1 = 7 * E0
        k2 = 2
        k3 = 2 * E0 - 0.2

        c = (1 + (1 - E0) * np.log(1 - E0) / E0) / t0

        # Define zeros and pole
        a1 = -1 / t0
        a2 = -1 / (alpha * t0)
        a3 = -(1 + 1j * np.sqrt(4 * ts ** 2 / tf - 1)) / (2 * ts)
        a4 = -(1 - 1j * np.sqrt(4 * ts ** 2 / tf - 1)) / (2 * ts)
        aZeros = [a1, a2, a3, a4]

        psi = -((k1 + k2) * ((1 - alpha) / alpha / t0 - c / alpha) - (k3 - k2) / t0) / (-(k1 + k2) * c * t0 - k3 + k2)
        cons = 1

    elif param['HRF'] == 'spmhrf':
        # Parameters for 'spmhrf' HRF
        a1 = -0.27
        a2 = -0.27
        a3 = -0.4347 - 1j * 0.3497
        a4 = -0.4347 + 1j * 0.3497
        psi = -0.1336
        aZeros = [a1, a2, a3, a4]
        cons = 1

    elif param['HRF'] == 'mion':
        # Parameters for 'mion' HRF
        a1 = -209.4112
        a2 = -0.1443
        psi = [a1, a2]
        aZeros = [-1 / 1.5, -1 / 4.5, -1 / 13.5]
        cons = 1 / 0.00487366

    else:
        raise ValueError("Unknown filter")

    # Scale zeros and poles by TR to convert to time domain
    FilZeros = np.array(aZeros) * TR
    FilPoles = np.asarray(psi) * TR
    # FilPoles = np.array(psi if isinstance(psi, (list, np.ndarray)) else [psi]) * TR

    # Build the discrete filters in the time domain
    hnum = cons_filter(FilZeros) * cons
    hden = cons_filter(FilPoles)

    # Separate causal and non-causal parts of the poles
    causal = FilPoles[FilPoles.real < 0]
    n_causal = FilPoles[FilPoles.real > 0]

    # Causal and non-causal filters
    h_dc = cons_filter(causal)
    h_dnc = cons_filter(n_causal)

    # Update param with reconstruction and analysis filters
    param['filter_reconstruct'] = {'num': hnum, 'den': [h_dc, h_dnc]}

    # For analysis, add one more zero to the reconstruction filter
    FilZeros2 = np.append(FilZeros, 0)
    hnum2 = cons_filter(FilZeros2) * cons

    # Frequency response and max eigenvalue calculation
    d1 = freqz(hnum2, hden, worN=1024)[1]
    param['maxeig'] = np.max(np.abs(d1) ** 2)

    # Analysis filter
    param['filter_analyze'] = {'num': hnum2, 'den': [h_dc, h_dnc]}

    return param