import numpy as np


def sol_dct(in_data, TR, TS, covariates=None):
    """
    Regresses out low-frequency components, and possible additional
    covariates, from the data. Low-frequency components are obtained
    from a DCT basis constructed by spm_dctmtx.

    Inputs:
        in_data    - (n_tp,) 1D array, time course data to detrend
        TR         - float, repetition time of the data in seconds
        TS         - float, cut-off period for the DCT basis in seconds
        covariates - (n_tp x n_cov) array of additional covariates to
                     include in the regression; None or [] if unwanted
                     (accepted for API compatibility with MATLAB but not
                     used in the regression — same as the MATLAB version)

    Outputs:
        dct_sol - (n_tp,) 1D array, detrended time course
        c_dct   - (k,) 1D array, DCT coefficients
    """
    n = len(in_data)

    # Number of DCT basis functions to use: matches MATLAB's round(2*n*TR/TS + 1)
    k = round(2 * n * TR / TS + 1)

    # Construct the DCT basis matrix (n x k)
    dct = spm_dctmtx(n, k)

    # Compute DCT coefficients by solving the normal equations:
    # c_dct = (D'D) \ (D'y)  — matches MATLAB exactly
    try:
        c_dct = np.linalg.solve(dct.T @ dct, dct.T @ in_data)
    except np.linalg.LinAlgError:
        # Fallback for rank-deficient / ill-conditioned cases
        c_dct = np.linalg.lstsq(dct, in_data, rcond=None)[0]

    # Remove the low-frequency components from the signal
    dct_sol = in_data - dct @ c_dct

    return dct_sol, c_dct


def spm_dctmtx(N, K, n=None, f=None):
    """
    Constructs a Discrete Cosine Transform (DCT) matrix, equivalent to
    MATLAB's spm_dctmtx used in SPM for detrending fMRI time series.

    Inputs:
        N - int, length of the time series (number of rows)
        K - int, number of DCT basis functions to generate (number of cols)
        n - (N,) array or None; time point indices; defaults to 0..N-1
        f - str or None; if 'diff', returns first derivative of the DCT
            basis at n; if 'diff2', returns the second derivative;
            None returns the standard DCT basis (default)

    Outputs:
        C - (N x K) 2D array, DCT matrix (or its derivative)
    """
    d = 0   # derivative order; 0 = standard DCT

    if K is None:
        K = N
    if n is None:
        n = np.arange(N)
    n = np.asarray(n).reshape(-1, 1)   # column vector (N x 1)

    if f is not None:
        if f == 'diff':
            d = 1
        elif f == 'diff2':
            d = 2
        else:
            raise ValueError("f must be None, 'diff', or 'diff2'.")

    C     = np.zeros((N, K))
    k_row = np.arange(1, K)[None, :]   # row vector 1..(K-1)

    if d == 0:
        # Standard DCT basis: first column is constant 1/sqrt(N),
        # remaining columns are cosine functions
        C[:, 0] = 1.0 / np.sqrt(N)
        if K > 1:
            C[:, 1:] = (
                np.sqrt(2.0 / N)
                * np.cos(np.pi * (2 * n + 1) * k_row / (2 * N))
            )

    elif d == 1:
        # First derivative of the DCT basis (first column is zero)
        if K > 1:
            C[:, 1:] = (
                -np.sqrt(2.0) * (1.0 / np.sqrt(N))
                * np.sin(0.5 * np.pi * (2 * n * k_row - n + k_row) / N)
                * np.pi * (k_row - 1) / N
            )

    elif d == 2:
        # Second derivative of the DCT basis (first column is zero)
        if K > 1:
            C[:, 1:] = (
                -np.sqrt(2.0) * (1.0 / np.sqrt(N))
                * np.cos(0.5 * np.pi * (2 * n + 1) * k_row / N)
                * (np.pi ** 2) * (k_row ** 2) / (N ** 2)
            )

    else:
        raise ValueError("Incorrect usage.")

    return C
