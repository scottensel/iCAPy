import numpy as np
from .jUpperTriMatToVec import jUpperTriMatToVec


def ComputeClusteringQuality(Consensus, K):
    """
    MATLAB-faithful version of ComputeClusteringQuality.m

    Parameters
    ----------
    Consensus : ndarray, shape (n_items, n_items)
        Ordered consensus matrix for a single K.
    K : int
        Number of clusters (unused numerically, kept for API compatibility).

    Returns
    -------
    CDF : ndarray, shape (101,)
        Cumulative distribution function of consensus values.
    AUC : float
        Area under the CDF curve.
    """
    Consensus = np.asarray(Consensus)

    # MATLAB: c = 0:0.01:1
    c = np.linspace(0.0, 1.0, 101)

    # Extract upper triangle (exclude diagonal)
    cons_vals = jUpperTriMatToVec(Consensus, offset=1)
    cons_vals = np.sort(cons_vals)

    # Compute CDF
    CDF = np.zeros(c.size)
    for i, ci in enumerate(c):
        CDF[i] = np.count_nonzero(cons_vals <= ci)

    CDF /= float(cons_vals.size)

    # MATLAB: AUC = diff(c) * CDF(2:end)'
    AUC = np.diff(c) @ CDF[1:]

    return CDF, AUC
