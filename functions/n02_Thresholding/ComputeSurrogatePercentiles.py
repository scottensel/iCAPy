import numpy as np
from functions.n00_Utilities.WriteInformation import write_information


def ComputeSurrogatePercentiles(I, param, fid):
    """
    Python translation of ComputeSurrogatePercentiles.m

    Parameters
    ----------
    I : ndarray, shape (n_tp, n_ret_vox)
        Matrix of surrogate innovation signals (time x voxel).
    param : dict-like
        Must contain key 'alpha', a 2-element iterable with the lower and
        upper percentile levels (e.g. [1, 99]).
    fid : file handle or str or None
        Log file handle / path, passed to write_information. If None, no
        log will be written.

    Returns
    -------
    PC : ndarray, shape (2, n_ret_vox)
        PC[0, :] is the lower percentile, PC[1, :] the upper percentile
        for each voxel.
    """
    I = np.asarray(I)
    alpha = np.asarray(param['alpha'])

    # In MATLAB: PC = prctile(I, param.alpha);
    # which returns a (len(alpha) x n_ret_vox) matrix when I is (n_tp x n_ret_vox).
    PC = np.percentile(I, alpha, axis=0, method="hazen")

    # Ensure shape is (2, n_ret_vox) even if alpha is given as list-like
    if PC.ndim == 1:
        PC = PC.reshape(1, -1)

    # Logging, if requested
    if fid is not None:
        bottom_mean = float(np.mean(PC[0, :]))
        top_mean = float(np.mean(PC[-1, :]))
        msg = (
            f"Computed voxel-wise percentiles: average values are "
            f"{bottom_mean} for bottom, and {top_mean} for top percentiles..."
        )
        write_information(fid, msg)

    return PC
