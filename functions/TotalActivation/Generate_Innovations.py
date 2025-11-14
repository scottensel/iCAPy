import numpy as np
from functions.TotalActivation.Temporal_TA.filter_boundary import filter_boundary
import time


def generate_innovations(TC_OUT, param):
    # Initialize output arrays
    Activity_inducing = np.zeros((param['Dimension'][3], param['NbrVoxels']))
    Innovation = np.zeros((param['Dimension'][3], param['NbrVoxels']))

    start_time = time.time()
    # Apply reconstruction and analysis filters for each voxel time course
    for i in range(param['NbrVoxels']):
        Activity_inducing[:, i] = filter_boundary(param['filter_reconstruct']['num'], param['filter_reconstruct']['den'], TC_OUT[:, i], 'normal')

        Innovation[:, i] = filter_boundary(param['filter_analyze']['num'], param['filter_analyze']['den'], TC_OUT[:, i], 'normal')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds. generate innovations")

    return Innovation, Activity_inducing
