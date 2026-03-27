import numpy as np
from functions.n00_Utilities.WriteInformation import write_information


def create_ta_data(fData, param, fid=None):
    """
    Creates a two-dimensional matrix containing the relevant data for
    total activation, from 4D input data and a mask determining the
    voxels to retain.

    Inputs:
        fData - (X x Y x Z x T) 4D array of functional data
        param - dict containing TA parameters; must include:
            'mask'      - 1D boolean array of length X*Y*Z with True for
                          voxels to retain and False for out-of-brain or
                          NaN voxels; populated by CreateTAMask
            'Dimension' - 4-element list/array [X, Y, Z, T]
        fid   - optional log file handle for write_information

    Outputs:
        fData_2D - (n_retained_voxels x T) 2D array with the time courses
                   of the retained voxels only
        param    - updated dict with the following fields added:
            'IND'       - 1D array of linear indices of the retained voxels
                          in the flattened (column-major) volume
            'VoxelIdx'  - (n_retained_voxels x 3) array with the 3D
                          coordinates [x, y, z] of each retained voxel
            'NbrVoxels' - int, number of retained voxels
    """
    num_voxels     = int(np.sum(param['mask']))
    num_timepoints = param['Dimension'][3]

    # Initialise output matrix with NaN (same as MATLAB's nan(...) call)
    fData_2D = np.full((num_voxels, num_timepoints), np.nan)

    # Extract the masked voxels at each time point using column-major
    # (Fortran) order to match MATLAB's memory layout
    for t in range(num_timepoints):
        tmp          = fData[:, :, :, t]
        tmp          = np.ravel(tmp, order='F')
        fData_2D[:, t] = tmp[param['mask']]

    # Find the linear indices of the retained voxels (1D, column-major)
    param['IND'] = np.where(param['mask'])[0]

    # Derive the 3D coordinates of those elements: VoxelIdx has shape
    # (n_retained_voxels x 3), matching MATLAB's ind2sub output
    param['VoxelIdx'] = np.column_stack(
        np.unravel_index(param['IND'], param['Dimension'][:3], order='F')
    ).astype(int)

    # Number of retained voxels
    param['NbrVoxels'] = fData_2D.shape[0]

    if fid:
        total_voxels = int(np.prod(param['Dimension'][:3]))
        write_information(
            fid,
            f"Keeping {num_voxels} out of {total_voxels} voxels for TA..."
        )

    return fData_2D, param
