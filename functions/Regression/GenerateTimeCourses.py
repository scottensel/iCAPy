import numpy as np
from functions.Utilities.WriteInformation import write_information


def generate_time_courses(Activity_inducing, AI_subject_labels, iCAPs, param, fid=None):
    """Simplified Python approximation of GenerateTimeCourses.m.

    The original MATLAB implementation uses constrained optimization
    (fmincon) to estimate, for each time point, positive and negative
    contributions of iCAPs that best reconstruct the innovation signal.

    This Python version instead uses unconstrained linear least-squares:
        min_T || Activity_inducing - T^T * iCAPs ||_2

    Parameters
    ----------
    Activity_inducing : ndarray, shape (n_frames, n_vox)
        Activity-inducing signals concatenated across subjects.
    AI_subject_labels : ndarray, shape (n_frames,)
        Subject index for each frame (1..nSub).
    iCAPs : ndarray, shape (nClus, n_vox)
        iCAP spatial maps.
    param : dict-like
        Should contain 'n_subjects'.
    fid : file handle or str or None
        For logging.

    Returns
    -------
    T_array : list of ndarray
        One entry per subject, each of shape (nClus, nTP_subject).
    stats : dict
        Contains basic least-squares residuals per subject.
    """
    Activity_inducing = np.asarray(Activity_inducing, dtype=float)
    AI_subject_labels = np.asarray(AI_subject_labels, dtype=int)
    iCAPs = np.asarray(iCAPs, dtype=float)

    nClus, n_vox = iCAPs.shape
    n_frames = Activity_inducing.shape[0]
    nSub = int(param.get('n_subjects', AI_subject_labels.max()))

    # Precompute pseudo-inverse of iCAPs
    # Solve for T (nClus x n_frames_sub) in least-squares sense:
    #   iCAPs.T @ T[:, t] ~ Activity_inducing[t, :].T
    AtA_inv = np.linalg.pinv(iCAPs @ iCAPs.T)
    pseudo = AtA_inv @ iCAPs  # (nClus x n_vox)

    T_array = []
    residuals = []

    for iS in range(1, nSub + 1):
        mask = AI_subject_labels == iS
        Y = Activity_inducing[mask, :]  # nTP_sub x n_vox
        if Y.size == 0:
            T_array.append(np.zeros((nClus, 0)))
            residuals.append(0.0)
            continue
        # T_sub (nClus x nTP_sub) = pseudo @ Y.T
        T_sub = pseudo @ Y.T
        T_array.append(T_sub)
        # residual
        Y_hat = (iCAPs.T @ T_sub).T
        residuals.append(float(np.linalg.norm(Y - Y_hat)))

    stats = dict(residuals=np.array(residuals))
    if fid is not None:
        write_information(fid, "GenerateTimeCourses: computed LSQ time courses per subject.")

    return T_array, stats
