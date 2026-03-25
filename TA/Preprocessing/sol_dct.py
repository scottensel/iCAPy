import numpy as np

def sol_dct(in_data, TR, TS):
    """
    Regresses out low-frequency components from the input data using a DCT basis.

    Parameters:
    - in_data: 1D numpy array, time course data of length n_tp.
    - TR: float, repetition time of the data.
    - TS: float, sampling rate.

    Returns:
    - dct_sol: 1D numpy array, detrended time course.
    - c_dct: 1D numpy array, DCT coefficients.
    """
    n = len(in_data)
    k = round(2 * n * TR / TS + 1)

    # Construct the DCT matrix
    dct = spm_dctmtx(n, k)
    # Linear trend component
    # linear_trend = np.arange(1, n + 1) / np.sqrt(np.sum(np.arange(1, n + 1) ** 2))

    # dct = np.column_stack([dct, linear_trend])
    # Calculate DCT coefficients and detrend the data
    # c_dct = np.linalg.solve(dct.T @ dct, dct.T @ in_data)  # if (dct.T @ dct) is square & well-conditioned
    c_dct = np.linalg.pinv(dct).dot(in_data)

    dct_sol = in_data - dct.dot(c_dct)

    return dct_sol, c_dct

def spm_dctmtx(N, K, n=None, f=None):
    """
    Constructs a Discrete Cosine Transform (DCT) matrix.

    Parameters:
    - N: int, length of the time series.
    - K: int, number of DCT basis functions to generate.
    - n: 1D array or None, time points.
    - f : {'diff','diff2'}, optional
        If 'diff', return first derivative of the DCT basis at n.
        If 'diff2', return second derivative.

    Returns:
    - C: 2D numpy array, DCT matrix of shape (N, K).
    """
    d = 0  # No derivative by default

    if K is None:
        K = N
    if n is None:
        n = np.arange(N)
    n = np.asarray(n).reshape(-1, 1)  # column vector

    # Handle derivative flags
    if f is not None:
        if f == 'diff':
            d = 1
        elif f == 'diff2':
            d = 2
        else:
            raise ValueError("Incorrect usage: f must be None, 'diff', or 'diff2'.")

    # Initialize DCT matrix
    C = np.zeros((N, K))
    # Ensure column/row shapes (n: Mx1, k_row: 1x(K-1))
    k_row = np.arange(1, K)[None, :]

    if d == 0:
        # DCT basis
        C[:, 0] = 1 / np.sqrt(N)
        if K > 1:
            C[:, 1:] = (
                    np.sqrt(2 / N)
                    * np.cos(np.pi * (2 * n + 1) * k_row / (2 * N))
            )

    elif d == 1:
        # First derivative (first column remains zeros)
        if K > 1:
            C[:, 1:] = (
                    - np.sqrt(2) * (1 / np.sqrt(N))
                    * np.sin(0.5 * np.pi * (2 * n * k_row - 1 * n + k_row) / N)
                    * np.pi * (k_row - 1) / N
            )

    elif d == 2:
        # Second derivative (first column remains zeros)
        if K > 1:
            C[:, 1:] = (
                    - np.sqrt(2) * (1 / np.sqrt(N))
                    * np.cos(0.5 * np.pi * (2 * n + 1) * k_row / N)
                    * (np.pi ** 2) * (k_row ** 2) / (N ** 2)
            )

    else:
        raise ValueError("Incorrect usage.")

    return C
