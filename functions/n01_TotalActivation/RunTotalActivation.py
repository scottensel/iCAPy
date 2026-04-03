import numpy as np

from functions.n00_Utilities.WriteInformation import write_information
from functions.n01_TotalActivation.Temporal_TA.TA_Temporal_conv import ta_temporal
from functions.n01_TotalActivation.Temporal_TA.TA_Temporal_conv_parallel import ta_temporal_parallel
from functions.n01_TotalActivation.Spatial_TA.TA_Spatial import ta_spatial
from functions.n01_TotalActivation.Spatial_TA.get_decision_gradient import get_decision_gradient
from functions.n00_Utilities.ta_checkpoint import (
    find_ta_checkpoint,
    save_ta_checkpoint,
    load_ta_checkpoint,
    delete_ta_checkpoint,
)

"""
Core script for total activation, calling the spatial and temporal
subparts of the routine.

Acceleration flags (set in Inputs_TA.py):
    param['runParallel']      = 1  — temporal step uses multiprocessing
    param['use_numba']        = 1  — spatial step uses Numba JIT (pip install numba)
    param['use_gpu']          = 1  — spatial step uses CuPy GPU (pip install cupy-cuda12x)
    param['gpu_chunk_size']   = N  — time points per GPU chunk
    param['numba_chunk_size'] = N  — time points per Numba chunk (None = no chunking)

Checkpointing (set in Inputs_TA.py):
    param['checkpoint_every'] = N  — save state every N outer iterations
                              = 0  — disable checkpointing
    param['outDir_TA']            — TA output folder (set by Run_TA.py);
                                    checkpoint saved here as ta_checkpoint.h5
                                    and deleted automatically on completion

Inputs:
    TCN   - (n_time_points x n_ret_voxels) matrix of input data
    param - dict containing relevant TA parameters:
        'Dimension'   - 4-element list/array of X, Y, Z and T sizes
        'NbrVoxels'   - number of voxels entering TA
        'GM_map'      - 3D volume containing elements from the
                        probabilistic gray matter map
        'sigma'       - used in the computation of the spatial TA weights
        'NitTemp'     - number of iterations to run the temporal scheme
        'Nit'         - number of iterations to run the forward-backward
                        scheme
        'weights'     - weight given to temporal and spatial regularization
                        schemes in the final averaging process
        'runParallel' - 0/1, use parallel temporal step
        'use_numba'   - 0/1, use Numba JIT for spatial step
        'use_gpu'     - 0/1, use CuPy GPU for spatial step
        'outDir_TA'   - str, subject TotalActivation output folder;
                        required for checkpointing

Outputs:
    TC_OUT - (n_time_points x n_ret_voxels) 2D matrix of data containing
             the output time courses from the total activation process
    param  - updated param dict with warm-start fields populated
"""


def run_total_activation(TCN, param, fid):
    """
    Runs the forward-backward splitting loop for total activation.

    At each outer iteration k:
        1. Temporal regularization on (current estimate - xT + data)
        2. Spatial regularization on  (current estimate - xS + data)
           (skipped on the final iteration)
        3. Weighted average of temporal and spatial outputs

    See Karahanoglu et al. 2013 (NI), Algorithm 1.

    Checkpoint save/load logic lives in:
        functions/n00_Utilities/ta_checkpoint.py
    """
    checkpoint_every = int(param.get('checkpoint_every', 1))
    do_checkpoint = (checkpoint_every > 0 and 'outDir_TA' in param)

    if checkpoint_every > 0 and 'outDir_TA' not in param:
        print("[WARNING] checkpoint_every > 0 but param['outDir_TA'] is not "
              "set — checkpointing disabled for this run.")

    # Compute decision gradient weights from the GM map (done once before loop)
    param = get_decision_gradient(param)
    print('Computed weights, entering loop...')

    # Initialise fixed-size warm-start buffers for the parallel temporal step.
    # NaN signals "not yet computed" — the first iteration falls back to the
    # MAD-based LambdaTemp estimate per voxel.
    if param.get('runParallel', 0):
        V = param['NbrVoxels']
        if 'NoiseEstimateFin' not in param or param['NoiseEstimateFin'] is None:
            param['NoiseEstimateFin'] = np.full(V, np.nan, dtype=np.float64)
        else:
            param['NoiseEstimateFin'] = np.asarray(param['NoiseEstimateFin'],
                                                    dtype=np.float64)
        if 'LambdaTempFin' not in param or param['LambdaTempFin'] is None:
            param['LambdaTempFin'] = np.full(V, np.nan, dtype=np.float64)
        else:
            param['LambdaTempFin'] = np.asarray(param['LambdaTempFin'],
                                                 dtype=np.float64)

    # Print active acceleration mode once at the start
    if param.get('use_gpu', 0):
        print('Spatial step:  GPU (CuPy)')
    elif param.get('use_numba', 0):
        print('Spatial step:  Numba JIT (CPU parallel)')
    else:
        print('Spatial step:  NumPy (CPU)')
    print('Temporal step: multiprocessing (parallel CPU)'
          if param.get('runParallel', 0) else 'Temporal step: sequential CPU')
    if do_checkpoint:
        print(f"Checkpointing: every {checkpoint_every} iteration(s) "
              f"→ {param['outDir_TA']}/ta_checkpoint.h5")
    else:
        print('Checkpointing: disabled')

    # ── Detect existing checkpoint and resume if found ────────────────────────
    start_k = 0
    if do_checkpoint and find_ta_checkpoint(param):
        print('\nCheckpoint detected — resuming interrupted run...')
        TC_OUT, xT, xS, start_k, param = load_ta_checkpoint(param)
        print(f"Resuming from iteration {start_k + 1} of {param['Nit']}\n")
    else:
        if do_checkpoint:
            print('No checkpoint found — starting from iteration 1\n')
        TC_OUT = np.zeros((param['Dimension'][3], param['NbrVoxels']))
        xT     = np.zeros_like(TC_OUT)
        xS     = np.zeros_like(TC_OUT)

    stepsize = 1

    # ── Main forward-backward splitting loop ─────────────────────────────────
    # At each outer iteration k:
    #   1. Temporal regularization applied to (current estimate - xT + data)
    #   2. Spatial regularization applied to (current estimate - xS + data)
    #      (skipped on the final iteration)
    #   3. Weighted average of temporal and spatial outputs
    # See Karahanoglu et al. 2013 (NI), Algorithm 1
    for k in range(start_k + 1, param['Nit'] + 1):

        # print(f"Currently at iteration {k} of {param['Nit']}...")
        write_information(fid, f"Currently at iteration {k} of {param['Nit']}...")

        # Increment temporal regularization iterations each outer loop
        param['NitTemp'] += 100

        # 1. TEMPORAL REGULARIZATION
        if param.get('runParallel', 0):
            temp, param = ta_temporal_parallel(TC_OUT - xT + TCN, param)
        else:
            temp, param = ta_temporal(TC_OUT - xT + TCN, param)

        xT += stepsize * (temp - TC_OUT)
        print('Finished temporal step...')

        # 2. SPATIAL REGULARIZATION (skipped on the last outer iteration)
        if k < param['Nit']:
            temp2 = ta_spatial(TC_OUT - xS + TCN, param)
            xS   += (temp2 - TC_OUT)
        print('Finished spatial step...')

        # 3. WEIGHTED AVERAGING
        TC_OUT = xT * param['weights'][0] + param['weights'][1] * xS
        print('Finished weighted averaging step...')

        # Save checkpoint if due — not on the final iteration since the
        # run is about to complete and final outputs are saved by the caller
        if do_checkpoint and (k % checkpoint_every == 0) and k < param['Nit']:
            save_ta_checkpoint(k, TC_OUT, xT, xS, param)

    # Delete checkpoint on successful completion — no leftover files
    if do_checkpoint:
        delete_ta_checkpoint(param)

    return TC_OUT, param
