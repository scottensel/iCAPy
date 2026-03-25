import numpy as np
from functions.TotalActivation.Temporal_TA.TA_Temporal_conv import ta_temporal
from functions.TotalActivation.Temporal_TA.TA_Temporal_conv_parallel import ta_temporal_parallel
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


    # ---- NEW: fixed-size warm-start buffers (persist across k iterations) ----
    if param['runParallel']:
        V = param['NbrVoxels']
        if 'NoiseEstimateFin' not in param or param['NoiseEstimateFin'] is None:
            param['NoiseEstimateFin'] = np.full(V, np.nan, dtype=np.float64)  # NaN = not computed yet
        else:
            param['NoiseEstimateFin'] = np.asarray(param['NoiseEstimateFin'], dtype=np.float64)

        if 'LambdaTempFin' not in param or param['LambdaTempFin'] is None:
            param['LambdaTempFin'] = np.full(V, np.nan, dtype=np.float64)
        else:
            param['LambdaTempFin'] = np.asarray(param['LambdaTempFin'], dtype=np.float64)


    # Main loop for temporal and spatial regularization
    # for k in tqdm(range(1, param['Nit'] + 1), desc="Processing", ncols=80):
    for k in range(1, param['Nit'] + 1):
        print(f"Currently at iteration {k} of {param['Nit']}...")

        # Increment temporal regularization iterations
        param['NitTemp'] += 100

        # Temporal Regularization
        if param['runParallel']:
            temp, param = ta_temporal_parallel(TC_OUT - xT + TCN, param)
        else:
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
