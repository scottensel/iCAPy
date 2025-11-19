import numpy as np
from .jUpperTriMatToVec import jUpperTriMatToVec


def ComputeClusteringQuality(Consensus, K_range):
    """Python translation of ComputeClusteringQuality.m

    Computes CDF and AUC quality indices for different numbers of clusters.

    Parameters
    ----------
    Consensus : ndarray, shape (n_items, n_items, nK)
        Consensus matrices for each K.
    K_range : array_like, shape (nK,)
        Range of K values.

    Returns
    -------
    CDF : ndarray, shape (nK, 101)
        Cumulative distribution function of consensus values for each K.
    AUC : ndarray, shape (nK,)
        Area under the CDF curve for each K.
    """
    Consensus = np.asarray(Consensus)
    K_range = np.asarray(K_range)
    nK = K_range.size

    c = np.linspace(0.0, 1.0, 101)  # 0:0.01:1
    CDF = np.zeros((nK, c.size))
    AUC = np.zeros(nK)

    for k in range(nK):
        cons_k = np.asarray(Consensus[:, :, k])
        cons_vals = jUpperTriMatToVec(cons_k, offset=1)  # exclude diagonal
        cons_vals = np.sort(cons_vals)

        # Compute CDF
        for i, ci in enumerate(c):
            CDF[k, i] = np.count_nonzero(cons_vals <= ci)
        CDF[k, :] = CDF[k, :] / float(cons_vals.size)

        # Vectorized AUC as in MATLAB: AUC(k) = diff(c) * CDF(k,2:end)'
        AUC[k] = np.diff(c) @ CDF[k, 1:]

    return CDF, AUC
