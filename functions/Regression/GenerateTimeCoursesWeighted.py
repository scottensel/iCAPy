import numpy as np
from functions.Utilities.WriteInformation import write_information


def GenerateTimeCoursesWeighted(Activity_inducing, AI_subject_labels, iCAPs, weights, param, fid=None):
    """Simplified Python approximation of GenerateTimeCoursesWeighted.m.

    Same as GenerateTimeCourses, but applies per-frame weights (e.g., soft
    cluster assignment weights) when estimating time courses.

    Parameters
    ----------
    Activity_inducing : ndarray, shape (n_frames, n_vox)
    AI_subject_labels : ndarray, shape (n_frames,)
    iCAPs : ndarray, shape (nClus, n_vox)
    weights : ndarray, shape (n_frames,)
        Weight for each frame (e.g., number of clusters assigned).
    param : dict-like
    fid : file handle or str or None

    Returns
    -------
    T_array : list of ndarray
    stats : dict
    """
    Activity_inducing = np.asarray(Activity_inducing, dtype=float)
    AI_subject_labels = np.asarray(AI_subject_labels, dtype=int)
    weights = np.asarray(weights, dtype=float)
    iCAPs = np.asarray(iCAPs, dtype=float)

    nClus, n_vox = iCAPs.shape
    n_frames = Activity_inducing.shape[0]
    nSub = int(param.get('n_subjects', AI_subject_labels.max()))

    # weighted pseudo-inverse: approximate by scaling rows of Y
    T_array = []
    residuals = []

    for iS in range(1, nSub + 1):
        mask = AI_subject_labels == iS
        Y = Activity_inducing[mask, :]
        w = weights[mask]
        if Y.size == 0:
            T_array.append(np.zeros((nClus, 0)))
            residuals.append(0.0)
            continue
        W = np.sqrt(w)[:, None]
        Yw = Y * W
        # re-use iCAPs (not weighted), simple LSQ
        AtA_inv = np.linalg.pinv(iCAPs @ iCAPs.T)
        pseudo = AtA_inv @ iCAPs
        T_sub = pseudo @ Yw.T
        T_array.append(T_sub)
        Y_hat = (iCAPs.T @ T_sub).T / (W + 1e-8)
        residuals.append(float(np.linalg.norm(Y - Y_hat)))

    stats = dict(residuals=np.array(residuals))
    if fid is not None:
        write_information(fid, "GenerateTimeCoursesWeighted: computed weighted LSQ time courses per subject.")

    return T_array, stats
