import numpy as np


def jUpperTriMatToVec(m, offset=1):
    """
    Converts the upper-triangular part of a matrix into a vector.

    Parameters
    ----------
    m : ndarray, shape (n, n)
        Input square matrix.
    offset : int, optional
        Offset above main diagonal (as in MATLAB triu).

    Returns
    -------
    v : ndarray
        Vector containing the upper-triangular values of m.
    """
    m = np.asarray(m)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("jUpperTriMatToVec: input must be a square matrix")

    n = m.shape[0]
    rows, cols = np.triu_indices(n, k=offset)
    v = m[rows, cols]
    return v
