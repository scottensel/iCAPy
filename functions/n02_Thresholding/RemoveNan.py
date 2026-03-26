import numpy as np
from functions.n00_Utilities.WriteInformation import write_information


def RemoveNan(Innovation, param, fid):
    """
    Python translation of RemoveNan.m

    This function removes NaN time courses from the innovation matrix and
    updates the voxel mask accordingly.

    Parameters
    ----------
    Innovation : ndarray
        Innovation time courses. Expected shape is (n_tp, n_ret_vox). If the
        number of columns is smaller than the number of rows, the matrix is
        transposed (to mimic the MATLAB behavior where time should be in rows).
    param : dict-like
        Must contain key 'mask', a boolean array of length n_vox_total indicating
        which voxels were included in the TA analysis.
    fid : file handle or str or None
        Passed to write_information. If None, logging is skipped.

    Returns
    -------
    Innovation2 : ndarray, shape (n_tp, n_ret_vox_2)
        Innovation matrix with the NaN traces removed.
    mask_out : ndarray (boolean), shape (n_vox_total,)
        Updated mask including only the voxels whose time courses are not NaN.
    """
    Innovation = np.asarray(Innovation)

    # In MATLAB the expected orientation is (n_tp x n_ret_vox). If we detect
    # that the opposite is true, we transpose and log a message.
    if Innovation.shape[1] < Innovation.shape[0]:
        if fid is not None:
            write_information(
                fid,
                "Time series probably laid in columns, should be rows; transposing..."
            )
        Innovation = Innovation.T

    # mask2 is True for columns (voxels) which have no NaNs
    mask2 = ~np.any(np.isnan(Innovation), axis=0)
    Innovation2 = Innovation[:, mask2]

    # Update voxel-wise mask: param['mask'] is for all voxels, and entries
    # where param['mask'] == True correspond to columns of Innovation
    mask = np.asarray(param['mask']).astype(bool).copy()
    # Only positions where mask is True are updated with mask2 values
    mask[mask] = mask2
    mask_out = mask

    if fid is not None:
        n_removed = int(np.sum(~mask2))
        write_information(
            fid,
            f"There are {n_removed} voxels that have been removed because of being NaN after TA..."
        )

    return Innovation2, mask_out, mask2
