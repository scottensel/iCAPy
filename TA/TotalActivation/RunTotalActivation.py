import numpy as np
from functions.TotalActivation.Temporal_TA.TA_Temporal_conv_parallel import ta_temporal
from functions.TotalActivation.Spatial_TA.TA_Spatial import ta_spatial
from functions.TotalActivation.Spatial_TA.get_decision_gradient import get_decision_gradient


def run_total_activation(TCN, param):
    # Initialize output arrays
    TC_OUT = np.zeros((param['Dimension'][3], param['NbrVoxels']))
    xT = np.zeros_like(TC_OUT)
    xS = np.zeros_like(TC_OUT)
    stepsize = 1

    # Initialize decision gradient weights
    param = get_decision_gradient(param)
    print('Computed weights, entering loop...')

    # Main loop for temporal and spatial regularization
    # for k in tqdm(range(1, param['Nit'] + 1), desc="Processing", ncols=80):
    for k in range(1, param['Nit'] + 1):
        print(f"Currently at iteration {k} of {param['Nit']}...")

        # Increment temporal regularization iterations
        param['NitTemp'] += 100

        # Temporal Regularization
        temp, param = ta_temporal(TC_OUT - xT + TCN, param)
        xT += stepsize * (temp - TC_OUT)
        print('Finished temporal step...')

        # Spatial Regularization if not the last iteration
        if k < param['Nit']:
            temp2 = ta_spatial(TC_OUT - xS + TCN, param)
            xS += (temp2 - TC_OUT)
        print('Finished spatial step...')

        # Weighted averaging
        TC_OUT = xT * param['weights'][0] + param['weights'][1] * xS
        print('Finished weighted averaging step...')

    return TC_OUT, param
