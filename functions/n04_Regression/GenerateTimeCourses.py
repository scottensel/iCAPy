import numpy as np
from scipy.optimize import minimize
from functions.n00_Utilities.WriteInformation import write_information


def generate_time_courses(Activity_inducing, AI_subject_labels, iCAPs, param, fid=None):
    """
    Python equivalent of GenerateTimeCourses.m.

    For each timepoint, runs two bounded optimisations matching MATLAB's fmincon:
      - positive part: lb=0,    ub=Inf   (non-negative contributions)
      - negative part: lb=-Inf, ub=0     (non-positive contributions)
    T = t_pos + t_neg

    Uses scipy.optimize.minimize with method='SLSQP', which is the closest
    Python equivalent to MATLAB's interior-point fmincon for this problem.

    Parameters
    ----------
    Activity_inducing : ndarray, shape (n_vox, n_timepoints)
        Activity-inducing signals, voxels x timepoints.
        MATLAB passes AI' so this expects the transposed orientation.
        If shape is (n_timepoints, n_vox), it will be transposed automatically
        with a warning, matching MATLAB's behaviour.
    AI_subject_labels : ndarray, shape (n_timepoints,)
        Subject index (0-based) for each timepoint.
    iCAPs : ndarray, shape (nClus, n_vox)
        iCAP spatial maps.
    param : dict
        Must contain:
          - 'n_subjects' : int
          - 'K'          : int  (number of iCAPs)
    fid : file handle or None

    Returns
    -------
    T_array : list of ndarray, length nSub
        Each entry shape (nClus, nTP_sub) — time courses per subject.
    stats : dict
        Per-subject model statistics matching MATLAB's stats struct:
          - 'RSS' : (nSub,)  residual sum of squares
          - 'n'   : (nSub,)  number of observations (n_vox * n_vol * 2)
          - 'k'   : (nSub,)  number of regressors  (n_iCAPs * n_vol * 2)
          - 'bic' : (nSub,)
          - 'aic' : (nSub,)
    """
    Activity_inducing = np.asarray(Activity_inducing, dtype=float)
    AI_subject_labels = np.asarray(AI_subject_labels, dtype=int)
    iCAPs             = np.asarray(iCAPs, dtype=float)

    # MATLAB: if size(Activity_inducing,1) < size(Activity_inducing,2) -> transpose
    if Activity_inducing.shape[0] < Activity_inducing.shape[1]:
        if fid is not None:
            write_information(
                fid,
                "GenerateTimeCourses: inverting activity inducing signal for unconstrained regression"
            )
        Activity_inducing = Activity_inducing.T

    # Now Activity_inducing is (n_vox, n_timepoints)
    n_vox, n_timepoints = Activity_inducing.shape
    n_sub   = int(param["n_subjects"])
    n_iCAPs = int(param["K"])

    # Per-subject timepoint counts (0-based labels)
    n_vol = np.zeros(n_sub, dtype=int)
    for iS in range(n_sub):
        n_vol[iS] = int(np.sum(AI_subject_labels == iS))

    # Bounds for the two constrained optimisations
    bounds_pos = [(0.0, None)]  * n_iCAPs   # lb=0,    ub=Inf
    bounds_neg = [(None, 0.0)] * n_iCAPs    # lb=-Inf, ub=0
    x0         = np.zeros(n_iCAPs)
    options    = {"ftol": 1e-8, "disp": False, "maxiter": 1000}

    t_pos = np.full((n_iCAPs, n_timepoints), np.nan)
    t_neg = np.full((n_iCAPs, n_timepoints), np.nan)

    if fid is not None:
        write_information(fid, f"GenerateTimeCourses: running constrained optimisation for {n_timepoints} timepoints...")

    for t in range(n_timepoints):
        y = Activity_inducing[:, t]   # (n_vox,)

        def obj(x, y=y):
            return float(np.sum((iCAPs.T @ x - y) ** 2))

        res_pos = minimize(obj, x0, method="SLSQP", bounds=bounds_pos, options=options)
        res_neg = minimize(obj, x0, method="SLSQP", bounds=bounds_neg, options=options)

        t_pos[:, t] = res_pos.x
        t_neg[:, t] = res_neg.x

    # Combined time courses
    T = t_pos + t_neg   # (n_iCAPs, n_timepoints)

    # Split into per-subject cell arrays (matching MATLAB T_array{iS})
    T_array = []
    for iS in range(n_sub):
        mask = AI_subject_labels == iS
        T_array.append(T[:, mask])

    # Residuals and model statistics (matching MATLAB stats struct exactly)
    res_pos_mat = iCAPs.T @ t_pos - Activity_inducing   # (n_vox, n_timepoints)
    res_neg_mat = iCAPs.T @ t_neg - Activity_inducing   # (n_vox, n_timepoints)

    RSS = np.zeros(n_sub)
    n   = np.zeros(n_sub, dtype=float)
    k   = np.zeros(n_sub, dtype=float)
    bic = np.zeros(n_sub)
    aic = np.zeros(n_sub)

    for iS in range(n_sub):
        vols_iS = AI_subject_labels == iS

        # MATLAB: sum(res_pos(vols_iS).^2 + res_neg(vols_iS).^2)
        # res matrices are (n_vox, n_timepoints); vols_iS selects columns
        RSS[iS] = float(
            np.sum(res_pos_mat[:, vols_iS] ** 2) +
            np.sum(res_neg_mat[:, vols_iS] ** 2)
        )

        # MATLAB: n = n_vox * n_vol(iS) * 2  (two regressions)
        n[iS] = float(n_vox * n_vol[iS] * 2)

        # MATLAB: k = n_iCAPs * n_vol(iS) * 2
        k[iS] = float(n_iCAPs * n_vol[iS] * 2)

        bic[iS] = n[iS] * np.log(RSS[iS] / n[iS]) + k[iS] * np.log(n[iS])
        aic[iS] = n[iS] * np.log(RSS[iS] / n[iS]) + k[iS] * 2.0

    stats = {
        "RSS": RSS,
        "n":   n,
        "k":   k,
        "bic": bic,
        "aic": aic,
    }

    if fid is not None:
        write_information(fid, "GenerateTimeCourses: done.")

    return T_array, stats