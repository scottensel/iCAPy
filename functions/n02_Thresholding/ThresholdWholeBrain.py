import numpy as np


def ThresholdWholeBrain(mask1, n_voxels):
    """
    Determines the time points that exhibit whole-brain significant
    excursions in terms of innovation signal.

    Parameters
    ----------
    mask1 : ndarray, shape (n_tp, n_ret_vox)
        Matrix containing -1 or 1 for significant negative/positive
        excursions at a given time for a given voxel, and 0 for no
        significant excursion.
    n_voxels : int
        Number of voxels that must show significant excursions at the
        same time to conclude significance.

    Returns
    -------
    mask2 : ndarray of bool, shape (n_tp,)
        Boolean vector depicting the moments with significant excursions.
    """
    mask1 = np.asarray(mask1)
    # count how many voxels (columns) have a non-zero excursion at each time point
    counts = np.sum(np.abs(mask1) > 0, axis=1)
    mask2 = counts >= int(n_voxels)
    return mask2
