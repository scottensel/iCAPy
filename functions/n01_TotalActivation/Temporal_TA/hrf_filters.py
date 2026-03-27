import numpy as np
from scipy.signal import freqz
from functions.n01_TotalActivation.Temporal_TA.cons_filter import cons_filter


def hrf_filters(param):
    """
    Generates the filters linked to the hemodynamic response function.

    The reconstruction filter converts neural activation to BOLD signal,
    while the analysis filter converts a BOLD signal into neural activity
    ('SPIKE') or into the innovation signal ('BLOCK', which combines the
    deconvolution and derivation steps).

    Inputs:
        param - dict containing TA-relevant parameters:
            'HRF' - type of hemodynamic response function to use:
                      'bold'   - Friston et al. biophysical model
                      'spmhrf' - SPM canonical HRF parameterisation
                      'mion'   - MION contrast agent model
            'TR'  - repetition time of the fMRI data (in seconds)

    Outputs:
        param - updated dict with the following fields added:
            'filter_reconstruct' - dict with keys:
                'num' - numerator (zeros) of the reconstruction filter
                'den' - list [h_dc, h_dnc], causal and non-causal parts
                        of the denominator (poles)
            'filter_analyze'     - dict with keys:
                'num' - numerator with one extra zero compared to the
                        reconstruction filter (for the derivation step)
                'den' - same [h_dc, h_dnc] as filter_reconstruct
            'maxeig'             - maximum eigenvalue of the analysis
                                   filter operator, used to determine the
                                   step size for convergence in TA

    Implemented by Isik Karahanoglu, 20.12.2010
    """

    TR = param['TR']

    # ------------------------------------------------------------------ #
    # Configure zeros and pole(s) based on the chosen HRF type            #
    # ------------------------------------------------------------------ #
    if param['HRF'] == 'bold':

        # Biophysical 'bold' model parameters
        eps   = 0.54
        ts    = 1.54
        tf    = 2.46
        t0    = 0.98
        alpha = 0.33
        E0    = 0.34
        V0    = 1
        k1    = 7 * E0
        k2    = 2
        k3    = 2 * E0 - 0.2

        c = (1 + (1 - E0) * np.log(1 - E0) / E0) / t0

        # Zeros of the linear differential operator
        a1 = -1 / t0
        a2 = -1 / (alpha * t0)
        a3 = -(1 + 1j * np.sqrt(4 * ts ** 2 / tf - 1)) / (2 * ts)
        a4 = -(1 - 1j * np.sqrt(4 * ts ** 2 / tf - 1)) / (2 * ts)
        aZeros = [a1, a2, a3, a4]

        # Pole of the linear differential operator
        psi = -((k1 + k2) * ((1 - alpha) / alpha / t0 - c / alpha) -
                (k3 - k2) / t0) / (-(k1 + k2) * c * t0 - k3 + k2)
        cons = 1

    elif param['HRF'] == 'spmhrf':

        # SPM canonical HRF parameterisation
        a1 = -0.27
        a2 = -0.27
        a3 = -0.4347 - 1j * 0.3497
        a4 = -0.4347 + 1j * 0.3497
        psi = -0.1336
        aZeros = [a1, a2, a3, a4]
        cons = 1

    elif param['HRF'] == 'mion':

        # MION contrast agent model
        a1   = -209.4112
        a2   = -0.1443
        psi  = [a1, a2]
        aZeros = [-1 / 1.5, -1 / 4.5, -1 / 13.5]
        cons = 1 / 0.00487366

    else:
        raise ValueError("Unknown filter")

    # Convert zeros and poles from continuous to discrete time by
    # scaling with TR (see Karahanoglu et al. 2011, p.5267)
    FilZeros = np.array(aZeros) * TR
    FilPoles = np.asarray(psi)  * TR

    # Build the discrete filters in the time domain according to
    # Karahanoglu et al. 2011 (p.5267).
    # hnum is the filter with the zeros of the linear differential
    # operator; hden is built from its poles.
    hnum = cons_filter(FilZeros) * cons
    hden = cons_filter(FilPoles)

    # Separate the causal (real part < 0) and non-causal (real part > 0)
    # parts of the poles; each part is filtered independently
    causal   = FilPoles[FilPoles.real < 0]
    n_causal = FilPoles[FilPoles.real > 0]

    # Shortest filter, 1st-order approximation for each part
    h_dc  = cons_filter(causal)    # causal part
    h_dnc = cons_filter(n_causal)  # non-causal part

    # Both causal and non-causal parts are stored as a two-element list,
    # matching MATLAB's h_d{1} and h_d{2}
    h_d = [h_dc, h_dnc]

    # Reconstruction filter: converts neural activity back to BOLD
    param['filter_reconstruct'] = {'num': hnum, 'den': h_d}

    # In the 'BLOCK' case, the analysis filter has one extra zero compared
    # to the reconstruction filter — this accounts for the derivation step
    FilZeros2 = np.append(FilZeros, 0)
    hnum2     = cons_filter(FilZeros2) * cons

    # 1024-element frequency response of the analysis filter numerator,
    # used to compute the maximal eigenvalue of the operator
    d1 = freqz(hnum2, hden, worN=1024)[1]

    # Maximum eigenvalue of the operator — used to determine the step size
    # of the forward-backward algorithm for convergence (see TA_Temporal)
    param['maxeig'] = np.max(np.abs(d1) ** 2)

    # Analysis filter: converts BOLD to innovation signal (deconv + deriv)
    param['filter_analyze'] = {'num': hnum2, 'den': h_d}

    return param