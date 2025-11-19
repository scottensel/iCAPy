import numpy as np


def ZScore_iCAPs(iCAPs, I_sig=None, IDX=None):
    """Python translation of ZScore_iCAPs.m

    Performs spatial z-scoring of iCAP maps.

    Parameters
    ----------
    iCAPs : ndarray, shape (nClus, n_vox)
        iCAP spatial maps.
    I_sig : ignored (kept for API compatibility)
    IDX : ignored (kept for API compatibility)

    Returns
    -------
    iCAPs_z : ndarray, shape (nClus, n_vox)
        Z-scored iCAP maps (per cluster).
    """
    iCAPs = np.asarray(iCAPs)
    n_clus, n_vox = iCAPs.shape
    iCAPs_z = np.zeros_like(iCAPs, dtype=float)

    for i in range(n_clus):
        row = iCAPs[i, :]
        # Histogram-based mode / median-like center as in MATLAB
        hist_vals, bin_edges = np.histogram(row, bins=100)
        aind = np.where(hist_vals == hist_vals.max())[0]
        med = bin_edges[aind[0]]
        # Normalization: std-like denominator copied from MATLAB intent
        denom = np.sqrt(np.sum((row - med) ** 2) / float(row.size))
        if denom == 0:
            iCAPs_z[i, :] = 0.0
        else:
            iCAPs_z[i, :] = (row - med) / denom

    return iCAPs_z
