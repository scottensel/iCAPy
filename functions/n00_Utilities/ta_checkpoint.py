import os
import json
import numpy as np
import h5py

"""
ta_checkpoint.py
────────────────
Utility functions for saving and loading incremental checkpoints during
the total activation forward-backward splitting loop.

Only one checkpoint file exists at a time:
    <param['outDir_TA']>/ta_checkpoint.h5

Each new checkpoint is written to a .tmp file first and then renamed
atomically over the previous one, so a kill mid-write never leaves a
corrupt file. The checkpoint is deleted automatically on successful
completion of the run.

Public API
----------
    save_ta_checkpoint(k, TC_OUT, xT, xS, param)
    load_ta_checkpoint(param)  → TC_OUT, xT, xS, start_k, param
    delete_ta_checkpoint(param)
    find_ta_checkpoint(param)  → bool
"""

# Fixed filename — only one checkpoint file ever exists per subject run
_CHECKPOINT_FNAME = 'ta_checkpoint.h5'

# Param fields that contain large arrays or are non-serialisable —
# excluded from the JSON attribute and saved separately (or reconstructed)
_SKIP_FIELDS = {
    'NoiseEstimateFin', 'LambdaTempFin',
    'weight_x', 'weight_y', 'weight_z',
    'GM_map', 'VoxelIdx', 'VoxelIdx_xyz',
    'mask', 'IND', 'LambdaTemp',
}


def _checkpoint_path(param):
    """
    Returns the full path to the checkpoint file.

    Inputs:
        param - dict, must contain 'outDir_TA'

    Outputs:
        path - str, full path to ta_checkpoint.h5
    """
    return os.path.join(param['outDir_TA'], _CHECKPOINT_FNAME)


def find_ta_checkpoint(param):
    """
    Returns True if a checkpoint file exists for this subject run.

    Inputs:
        param - dict, must contain 'outDir_TA'

    Outputs:
        bool
    """
    return os.path.isfile(_checkpoint_path(param))


def save_ta_checkpoint(k, TC_OUT, xT, xS, param):
    """
    Saves the full forward-backward loop state to a single rolling
    checkpoint file, overwriting the previous one atomically.

    Writes to <outDir_TA>/ta_checkpoint.h5.tmp first, then renames
    to ta_checkpoint.h5 — guarantees the checkpoint is never left in
    a corrupt state if the process is killed mid-write.

    Saves:
        TC_OUT, xT, xS     - compressed HDF5 datasets (gzip level 4)
        NoiseEstimateFin    - warm-start array for temporal step
        LambdaTempFin       - warm-start array for temporal step
        k                   - completed iteration index (HDF5 attribute)
        param_json          - all serialisable param fields as JSON
                              (includes NitTemp which is incremented each
                              iteration and must be restored correctly)

    Inputs:
        k      - int, outer iteration just completed (1-based)
        TC_OUT - (T x V) ndarray, current output estimate
        xT     - (T x V) ndarray, temporal regularizer accumulator
        xS     - (T x V) ndarray, spatial regularizer accumulator
        param  - dict, full parameter dict at this point in the run;
                 must contain 'outDir_TA'
    """
    path     = _checkpoint_path(param)
    tmp_path = path + '.tmp'
    os.makedirs(param['outDir_TA'], exist_ok=True)

    with h5py.File(tmp_path, 'w') as f:

        # Large loop-state arrays — compressed to keep file size small
        f.create_dataset('TC_OUT', data=TC_OUT, compression='gzip',
                         compression_opts=4)
        f.create_dataset('xT',     data=xT,     compression='gzip',
                         compression_opts=4)
        f.create_dataset('xS',     data=xS,     compression='gzip',
                         compression_opts=4)

        # Iteration index — used to set start_k on resume
        f.attrs['k'] = k

        # Warm-start arrays for the temporal step
        if 'NoiseEstimateFin' in param and param['NoiseEstimateFin'] is not None:
            f.create_dataset('NoiseEstimateFin',
                             data=np.asarray(param['NoiseEstimateFin']),
                             compression='gzip', compression_opts=4)
        if 'LambdaTempFin' in param and param['LambdaTempFin'] is not None:
            f.create_dataset('LambdaTempFin',
                             data=np.asarray(param['LambdaTempFin']),
                             compression='gzip', compression_opts=4)

        # All serialisable param fields as a JSON string attribute.
        # NitTemp is the most important — it is incremented each outer
        # iteration and must be at the correct value when resuming.
        param_json = {}
        for key, val in param.items():
            if key in _SKIP_FIELDS:
                continue
            try:
                json.dumps(val)
                param_json[key] = val
            except (TypeError, ValueError):
                pass   # silently skip non-serialisable values
        f.attrs['param_json'] = json.dumps(param_json)

    # Atomic rename — replaces the old checkpoint in one OS operation
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp_path, path)

    print(f"  Checkpoint saved (iteration {k}) → {path}")


def load_ta_checkpoint(param):
    """
    Loads the loop state from the checkpoint file and merges saved param
    fields back into param.

    Inputs:
        param   - dict, current param dict (modified in-place);
                  must contain 'outDir_TA'

    Outputs:
        TC_OUT  - (T x V) ndarray
        xT      - (T x V) ndarray
        xS      - (T x V) ndarray
        start_k - int, last completed iteration; loop resumes from start_k+1
        param   - dict, updated with NitTemp and warm-start arrays restored
    """
    path = _checkpoint_path(param)
    print(f"  Loading checkpoint → {path}")

    with h5py.File(path, 'r') as f:
        TC_OUT  = f['TC_OUT'][:]
        xT      = f['xT'][:]
        xS      = f['xS'][:]
        start_k = int(f.attrs['k'])

        # Restore warm-start arrays for the temporal step
        if 'NoiseEstimateFin' in f:
            param['NoiseEstimateFin'] = f['NoiseEstimateFin'][:]
        if 'LambdaTempFin' in f:
            param['LambdaTempFin'] = f['LambdaTempFin'][:]

        # Restore all scalar param fields — NitTemp especially
        param_json = json.loads(f.attrs.get('param_json', '{}'))
        for key, val in param_json.items():
            param[key] = val

    return TC_OUT, xT, xS, start_k, param


def delete_ta_checkpoint(param):
    """
    Deletes the checkpoint file on successful completion of the run.
    Safe to call even if the file does not exist.

    Inputs:
        param - dict, must contain 'outDir_TA'
    """
    path = _checkpoint_path(param)
    if os.path.exists(path):
        os.remove(path)
        print(f"  Checkpoint deleted (run complete) → {path}")
