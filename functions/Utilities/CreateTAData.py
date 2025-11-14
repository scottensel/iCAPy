import numpy as np
from functions.Utilities.WriteInformation import write_information

def create_ta_data(fData, param, fid=None):
    """
    Reshapes and filters fData based on param['mask'] to retain only
    within-brain voxels for total activation analysis.

    Parameters:
    - fData: 4D numpy array of shape (X, Y, Z, T) with functional data.
    - param: Dictionary containing TA parameters, including:
        - 'mask': 1D boolean array with shape (X*Y*Z,) representing within-brain voxels.
        - 'Dimension': 4-element tuple with dimensions of fData.
    - fid: Optional file object for logging.

    Returns:
    - fData_2D: 2D array with shape (retained_voxels, T) containing only selected voxel time courses.
    - param: Updated param dictionary with indices of retained voxels.
    """
    num_voxels = np.sum(param['mask'])
    num_timepoints = param['Dimension'][3]
    fData_2D = np.full((num_voxels, num_timepoints), np.nan)

    for t in range(param['Dimension'][3]):
        tmp = fData[:, :, :, t]  # 3D volume at time t
        # tmp = np.reshape(tmp , (-1, 1), order='F')
        tmp = np.ravel(tmp, order='F')
        fData_2D[:, t] = tmp[param['mask']]

    # fData_flat = fData.reshape(-1, num_timepoints)  # Shape: (X*Y*Z, T)
    # fData_2D = fData_flat[param['mask'], :]  # Apply mask to retain within-brain voxels only

    # Store indices of retained elements in param
    param['IND'] = np.where(param['mask'])[0]

    # param['VoxelIdx'] = np.column_stack(np.unravel_index(param['IND'], param['Dimension'][:3]))
    param['VoxelIdx'] = np.column_stack(np.unravel_index(param['IND'], param['Dimension'][:3], order='F')).astype(int)
    param['NbrVoxels'] = fData_2D.shape[0]

    # Log information if fid is provided
    if fid:
        retained_voxels = np.sum(param['mask'])
        total_voxels = np.prod(param['Dimension'][:3])
        # fid.write(f"Keeping {retained_voxels} out of {total_voxels} voxels for TA...\n")
        write_information(fid, f"Keeping {retained_voxels} out of {total_voxels} voxels for TA...")

    return fData_2D, param
