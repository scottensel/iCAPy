import os
import numpy as np
import matplotlib.pyplot as plt

from functions.Utilities.WriteInformation import write_information


def _knee_pt(y, x):
    """
    MATLAB-ish knee point: max perpendicular distance to the chord between endpoints.
    Returns (knee_x, knee_idx) where knee_idx is 0-based.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.size != x.size:
        raise ValueError("knee_pt: x and y must have the same length")
    if y.size < 3:
        return x[0], 0

    # line between endpoints
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    dx = x2 - x1
    dy = y2 - y1
    denom = np.sqrt(dx * dx + dy * dy)
    if denom == 0:
        return x[0], 0

    # perpendicular distance from each point to the line
    # distance = |dy*x - dx*y + (x2*y1 - y2*x1)| / sqrt(dx^2 + dy^2)
    numer = np.abs(dy * x - dx * y + (x2 * y1 - y2 * x1))
    dist = numer / denom

    knee_idx = int(np.argmax(dist))
    return x[knee_idx], knee_idx


def evaluate_soft_cluster_thres(TC_stats, param, fid=None):
    """
    Python equivalent of MATLAB evaluateSoftClusterThres.m

    Parameters
    ----------
    TC_stats : list
        Length nThres. Each entry must provide .bic and .aic per subject.
        Accepts either:
          - dict with keys 'bic' and 'aic', or
          - object with attributes bic and aic.
        Each bic/aic should be array-like of length n_subjects.
    param : dict
        Must contain:
          - 'softClusterThres' (list/array)
          - 'n_subjects' (int)
          - 'outDir_reg' (str)
    fid : file handle or None

    Returns
    -------
    best_id_bic : int
        0-based index of the knee point on sum(BIC) across subjects.
        (Use directly for Python indexing, e.g. TC_list[best_id_bic])
    """
    thresVals = np.asarray(param["softClusterThres"], dtype=float)
    nThres = int(len(thresVals))
    nSub = int(param["n_subjects"])

    # ---- Extract BIC/AIC into (nThres x nSub) matrices ----
    BICvals = np.zeros((nThres, nSub), dtype=float)
    AICvals = np.zeros((nThres, nSub), dtype=float)

    for iT in range(nThres):
        ts = TC_stats[iT]
        if isinstance(ts, dict):
            bic = ts["bic"]
            aic = ts["aic"]
        else:
            bic = getattr(ts, "bic")
            aic = getattr(ts, "aic")

        bic = np.asarray(bic, dtype=float).ravel()
        aic = np.asarray(aic, dtype=float).ravel()

        if bic.size != nSub or aic.size != nSub:
            raise ValueError(
                f"TC_stats[{iT}] has bic/aic length {bic.size}/{aic.size}, expected n_subjects={nSub}"
            )

        BICvals[iT, :] = bic
        AICvals[iT, :] = aic

    out_dir = param["outDir_reg"]
    os.makedirs(out_dir, exist_ok=True)

    # ---- BIC knee on sum across subjects ----
    bic_sum = np.sum(BICvals, axis=1)
    _, knee_idx_bic = _knee_pt(bic_sum, thresVals)

    if fid is not None:
        write_information(fid, f"The BIC knee point is at xi={thresVals[knee_idx_bic]}")
    best_id_bic = int(knee_idx_bic)  # 0-based for Python

    # Plot: Sum of BIC
    fig = plt.figure(figsize=(3.85, 3.3))  # roughly MATLAB's small figure
    ax = fig.add_subplot(111)
    ax.plot(thresVals, bic_sum, "*")
    ax.plot(thresVals[knee_idx_bic], bic_sum[knee_idx_bic], "*r")
    ax.set_title("Sum of BIC across subjects")
    fig.savefig(os.path.join(out_dir, "BICsum.eps"), format="eps", bbox_inches="tight")
    plt.close(fig)

    # Plot: BIC distribution across subjects (mean +/- std)
    fig = plt.figure(figsize=(3.85, 3.3))
    ax = fig.add_subplot(111)
    ax.errorbar(thresVals, np.mean(BICvals, axis=1), yerr=np.std(BICvals, axis=1), fmt="-")
    ax.plot(thresVals[knee_idx_bic], np.mean(BICvals[knee_idx_bic, :]), "*r")
    ax.set_title("BIC distribution across subjects")
    fig.savefig(os.path.join(out_dir, "BICdist.eps"), format="eps", bbox_inches="tight")
    plt.close(fig)

    # ---- AIC knee on sum across subjects (MATLAB computes but does not return it) ----
    aic_sum = np.sum(AICvals, axis=1)
    _, knee_idx_aic = _knee_pt(aic_sum, thresVals)

    if fid is not None:
        write_information(fid, f"The AIC knee point is at xi={thresVals[knee_idx_aic]}")

    # Plot: Sum of AIC
    fig = plt.figure(figsize=(3.85, 3.3))
    ax = fig.add_subplot(111)
    ax.plot(thresVals, aic_sum, "*")
    ax.plot(thresVals[knee_idx_aic], aic_sum[knee_idx_aic], "*r")
    ax.set_title("Sum of AIC across subjects")
    fig.savefig(os.path.join(out_dir, "AICsum.eps"), format="eps", bbox_inches="tight")
    plt.close(fig)

    # Plot: AIC distribution across subjects
    fig = plt.figure(figsize=(3.85, 3.3))
    ax = fig.add_subplot(111)
    ax.errorbar(thresVals, np.mean(AICvals, axis=1), yerr=np.std(AICvals, axis=1), fmt="-")
    ax.plot(thresVals[knee_idx_aic], np.mean(AICvals[knee_idx_aic, :]), "*r")
    ax.set_title("AIC distribution across subjects")
    fig.savefig(os.path.join(out_dir, "AICdist.eps"), format="eps", bbox_inches="tight")
    plt.close(fig)

    return best_id_bic
