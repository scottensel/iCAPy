import numpy as np
from functions.n01_TotalActivation.Temporal_TA.filter_boundary import filter_boundary


def generate_innovations(TC_OUT, param):
    """
    Python equivalent of Generate_Innovations.m

    TC_OUT: array of shape (n_time_points, n_ret_voxels)
    param: dict with keys
        - 'Dimension' (list/tuple, time is index 3)
        - 'NbrVoxels'
        - 'filter_reconstruct' -> {'num', 'den'}
        - 'filter_analyze'     -> {'num', 'den'}
    """
    # Initialize output arrays
    Activity_inducing = np.zeros((param['Dimension'][3], param['NbrVoxels']))
    Innovation = np.zeros((param['Dimension'][3], param['NbrVoxels']))

    # Apply reconstruction and analysis filters for each voxel time course
    for i in range(param['NbrVoxels']):

        Activity_inducing[:, i] = filter_boundary(param['filter_reconstruct']['num'], param['filter_reconstruct']['den'], TC_OUT[:, i], 'normal')

        Innovation[:, i] = filter_boundary(param['filter_analyze']['num'], param['filter_analyze']['den'], TC_OUT[:, i], 'normal')

    return Innovation, Activity_inducing
