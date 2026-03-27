import numpy as np
from functions.n01_TotalActivation.Temporal_TA.TA_Temporal_conv import ta_temporal
from functions.n01_TotalActivation.Temporal_TA.TA_Temporal_conv_parallel import ta_temporal_parallel
from functions.n01_TotalActivation.Spatial_TA.TA_Spatial import ta_spatial
from functions.n01_TotalActivation.Spatial_TA.get_decision_gradient import get_decision_gradient

"""
Core script for total activation, calling the spatial and temporal
subparts of the routine.

Inputs:
    TCN   - (n_time_points x n_ret_voxels) matrix of input data
    param - dict containing relevant TA parameters:
        'Dimension'  - 4-element list/array of X, Y, Z and T sizes
        'NbrVoxels'  - number of voxels entering TA
        'GM_map'     - 3D volume containing elements from the
                       probabilistic gray matter map
        'sigma'      - used in the computation of the spatial TA weights
        'NitTemp'    - number of iterations to run the temporal scheme
        'Nit'        - number of iterations to run the forward-backward
                       scheme
        'weights'    - weight given to temporal and spatial regularization
                       schemes in the final averaging process

Outputs:
    TC_OUT - (n_time_points x n_ret_voxels) 2D matrix of data containing
             the output time courses from the total activation process
"""

def run_total_activation(TCN, param):

    # Spatial regularization is called (note that temporal regularization is
    # called from within MySpatial); TCN contains the time courses that we
    # measure (y), atlas contains the atlas map, param contains all parameters
    # for the algorithm (as well as the gray matter map)
    # Important: if using TV_PPM as a spatial method, 'atlas' is NOT USED
    # anywhere, although it is an argument of MySpatial

    # Initialize output arrays
    TC_OUT = np.zeros((param['Dimension'][3], param['NbrVoxels']))

    # Initialization of temporal and spatial matrices (solutions to the
    # two regularization problems)
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


    ## Main loop for temporal and spatial regularization
    # To solve the problem, the temporal and spatial regularizations
    # are applied one after the other, and then the output from one
    # iteration is the weighted sum of the two outputs; this is known
    # as the 'generalized forward-backward' splitting scheme FISTA
    for k in range(1, param['Nit'] + 1):
        print(f"Currently at iteration {k} of {param['Nit']}...")

        # Increment temporal regularization iterations
        param['NitTemp'] += 100


        # 1. TEMPORAL REGULARIZATION
        #
        # TC_OUT is only made of zeros at iteration 1, and then is the
        # current solution. TCN contains our data. xT is made of zeros
        # at first iteration and then, contains the temporal
        # regularizer solution
        # Hence, at iteration k=1, we give our data (TCN) to My
        # Temporal. Then, we give (whole estimate - temporal estimate +
        # our data). See Karahanoglu et al. 2013 (NI), Algorithm 1

        # Temporal Regularization
        # this is python only, not in matlab version
        # used to speed up the temporal step using processes
        if param['runParallel']:
            temp, param = ta_temporal_parallel(TC_OUT - xT + TCN, param)
        else:
            temp, param = ta_temporal(TC_OUT - xT + TCN, param)

        xT += stepsize * (temp - TC_OUT)
        print('Finished temporal step...')

        # Exactly the same process is done for the spatial
        # regularization; the if condition forces the algorithm to stop
        # with a temporal regularization step (no spatial
        # regularization done at k=5)
        # Spatial Regularization if not the last iteration
        if k < param['Nit']:
            temp2 = ta_spatial(TC_OUT - xS + TCN, param)
            xS += (temp2 - TC_OUT)
        print('Finished spatial step...')

        # Weighted averaging
        TC_OUT = xT * param['weights'][0] + param['weights'][1] * xS
        print('Finished weighted averaging step...')

    return TC_OUT, param
