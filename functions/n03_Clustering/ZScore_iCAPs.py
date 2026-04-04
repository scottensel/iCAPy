import numpy as np


def zscore_icaps(iCAPs, I_sig=None, IDX=None):
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
        Z-scored iCAP maps (per cluster). Rows that are entirely NaN
        (subject had no frames in that cluster) are left as NaN.
    """
    iCAPs = np.asarray(iCAPs)
    n_clus, n_vox = iCAPs.shape
    iCAPs_z = np.full_like(iCAPs, np.nan, dtype=float)

    for i in range(n_clus):
        row = iCAPs[i, :]

        # Skip rows that are entirely NaN — subject had no frames in this cluster
        if np.all(np.isnan(row)):
            continue

        # Use only finite values for histogram (ignore any remaining NaNs)
        row_finite = row[np.isfinite(row)]
        if row_finite.size == 0:
            continue

        # Histogram-based mode as in MATLAB
        hist_vals, bin_edges = np.histogram(row_finite, bins=100)
        aind = np.where(hist_vals == hist_vals.max())[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        med = bin_centers[aind[0]]

        denom = np.sqrt(np.sum((row_finite - med) ** 2) / float(row_finite.size))
        if denom == 0:
            iCAPs_z[i, :] = 0.0
        else:
            iCAPs_z[i, :] = (row - med) / denom

    return iCAPs_z