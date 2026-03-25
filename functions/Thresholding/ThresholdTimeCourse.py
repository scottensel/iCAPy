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
    if np.shape(T)[1] != n_vox:
        raise ValueError("Threshold Time Course: wrong number of threshold pairs, one threshold pre voxel required!")

    if np.shape(T)[0] != 2:
        raise ValueError("Threshold Time Course: wrong number of thresholds, one positive and one negative threshold required!")

    # lower = T[0, :].reshape(1, -1)
    # upper = T[1, :].reshape(1, -1)
    #
    # Out = np.zeros_like(S, dtype=int)
    # Out[S <= lower] = -1
    # Out[S >= upper] = 1

    # Initially filling the output with zeros. If we find a data point
    # lying above the thresholds, we change the related value
    Out = np.zeros_like(S, dtype=float)

    # negative threshold (T[0, :]) and positive threshold (T[1, :])
    neg_th = T[0, :][np.newaxis, :]  # shape (1, nVox), broadcasts over time
    pos_th = T[1, :][np.newaxis, :]

    Out[S <= neg_th] = -1
    Out[S >= pos_th] = 1

    return Out
