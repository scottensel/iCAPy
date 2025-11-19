import numpy as np
from functions.Utilities.WriteInformation import write_information


def _knee_point(y, x):
    """Simple knee point detection: maximum distance to line between endpoints."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("_knee_point: x and y must be 1D arrays of same length")

    # line between first and last points
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    if denom == 0:
        return 0

    # distance of each point to line
    distances = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    idx = int(np.argmax(distances))
    return idx


def evaluateSoftClusterThres(TC_stats, param, fid=None):
    """Python approximation of evaluateSoftClusterThres.m.

    Parameters
    ----------
    TC_stats : list of dict
        One entry per soft-threshold value. Each dict must contain:
        - 'bic' : array-like of shape (nSub,)
        - 'aic' : array-like of shape (nSub,)
    param : dict-like
        Must contain:
        - 'softClusterThres' : array-like of threshold values (xi)
        - 'n_subjects' : number of subjects
    fid : file handle or str or None
        For logging.

    Returns
    -------
    best_id_bic : int
        Index (0-based) of the chosen threshold according to BIC knee point.
    """
    thresVals = np.asarray(param['softClusterThres'], dtype=float)
    nThres = thresVals.size
    nSub = int(param['n_subjects'])

    BICvals = np.zeros((nThres, nSub), dtype=float)
    AICvals = np.zeros((nThres, nSub), dtype=float)

    for iT in range(nThres):
        BICvals[iT, :] = np.asarray(TC_stats[iT]['bic']).ravel()
        AICvals[iT, :] = np.asarray(TC_stats[iT]['aic']).ravel()

    # BIC knee point on summed values
    knee_idx = _knee_point(BICvals.sum(axis=1), thresVals)
    if fid is not None:
        write_information(fid, f"The BIC knee point is at xi={thresVals[knee_idx]}")

    return knee_idx
