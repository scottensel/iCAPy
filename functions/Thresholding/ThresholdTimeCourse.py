import numpy as np


def ThresholdTimeCourse(S, T):
    """
    Python translation of ThresholdTimeCourse.m

    Determines the moments of an innovation time course that are significant
    from comparison to threshold values.

    Parameters
    ----------
    S : ndarray, shape (n_tp, n_vox)
        Innovation time courses (time x voxel).
    T : ndarray, shape (2, n_vox)
        Thresholds for each voxel. T[0, :] is the lower threshold,
        T[1, :] is the upper threshold.

    Returns
    -------
    Out : ndarray, shape (n_tp, n_vox)
        Integer array with:
        -  1 where S >= upper threshold
        - -1 where S <= lower threshold
        -  0 elsewhere
    """
    S = np.asarray(S)
    T = np.asarray(T)

    n_tp, n_vox = S.shape

    # Check dimensions of T. In MATLAB the expected size is (2 x n_vox)
    if T.ndim == 1:
        # single pair of thresholds replicated across voxels
        if T.size != 2:
            raise ValueError("ThresholdTimeCourse: single threshold vector must have length 2")
        T = np.tile(T.reshape(2, 1), (1, n_vox))
    elif T.shape[0] == n_vox and T.shape[1] == 2:
        # transposed input (n_vox x 2); transpose to (2 x n_vox)
        T = T.T
    elif T.shape[0] != 2 or T.shape[1] != n_vox:
        raise ValueError(
            "ThresholdTimeCourse: wrong number/shape of threshold pairs – expected (2, n_vox)"
        )

    lower = T[0, :].reshape(1, -1)
    upper = T[1, :].reshape(1, -1)

    Out = np.zeros_like(S, dtype=int)
    Out[S <= lower] = -1
    Out[S >= upper] = 1

    return Out
