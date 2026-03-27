import numpy as np
from functions.n01_TotalActivation.Temporal_TA.filter_boundary import filter_boundary

"""
Generates the innovation and activity-inducing signals from the
activity-related signals (output from total activation).

Inputs:
    TC_OUT - (n_time_points x n_ret_voxels) 2D matrix of outputs from
             total activation
    param  - dict containing TA-relevant parameters:
        'Dimension'  - 4-element list/array of X, Y, Z and T sizes
        'NbrVoxels'  - number of voxels retained for TA
        'f_Recons'   - filter that converts the activity-related signal
                       into the activity-inducing signal
        'f_Analyze'  - filter with added derivation step

Outputs:
    Innovation         - (n_time_points x n_ret_voxels) 2D matrix of
                         innovation signals
    Activity_inducing  - (n_time_points x n_ret_voxels) 2D matrix of
                         activity-inducing signals
"""

def generate_innovations(TC_OUT, param):

    # Initialize output arrays
    Activity_inducing = np.zeros((param['Dimension'][3], param['NbrVoxels']))
    Innovation = np.zeros((param['Dimension'][3], param['NbrVoxels']))

    # Each voxel time course is deconvolved using the reconstruction filter
    # (solely deconvolution)
    for i in range(param['NbrVoxels']):

        # Applies the reconstruction filter (only deconvolution)
        Activity_inducing[:, i] = filter_boundary(param['filter_reconstruct']['num'], param['filter_reconstruct']['den'], TC_OUT[:, i], 'normal')

        # Applies the analysis filter (also encompasses the differentiation step)
        Innovation[:, i] = filter_boundary(param['filter_analyze']['num'], param['filter_analyze']['den'], TC_OUT[:, i], 'normal')

    return Innovation, Activity_inducing
