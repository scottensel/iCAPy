import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full
from functions.n01_TotalActivation.Spatial_TA.div3D_full import div3D_full
from functions.n01_TotalActivation.Spatial_TA.evaluate_3D_TV import evaluate_3D_TV
from functions.n01_TotalActivation.Spatial_TA.MyProx import MyProx
from functions.n01_TotalActivation.Spatial_TA.xp_backend import get_backend


def _run_myprox_chunk(y_chunk, weight_x_chunk, weight_y_chunk, weight_z_chunk,
                      chunk_param, xp, use_numba):
    """
    Runs MyProx on a single time chunk (X x Y x Z x chunk_T).
    All arrays must already be on the correct device (GPU or CPU).
    Used by both the GPU and Numba chunked paths.
    """
    def Op(x_vol, dx=None, dy=None, dz=None):
        out = (dx, dy, dz) if dx is not None else None
        return gradient3D_full(
            x_vol, weight_x_chunk, weight_y_chunk, weight_z_chunk,
            out=out, use_numba=use_numba
        )

    def Adj_Op(u, v, w, out=None):
        result = div3D_full(
            u, v, w, weight_x_chunk, weight_y_chunk, weight_z_chunk,
            out=out, use_numba=use_numba
        )
        result *= -1.0
        return result

    def evaluate_norm(x_vol, grad_bufs=None, use_numba=use_numba):
        return evaluate_3D_TV(x_vol, grad_bufs=grad_bufs, use_numba=use_numba)

    return MyProx(y_chunk, Op, Adj_Op, evaluate_norm, chunk_param,
                  use_numba=use_numba)


def _run_chunked(y_vol, param, xp, use_numba, chunk_size, label):
    """
    Processes the 4D volume in time chunks of size chunk_size.
    Works for both GPU (xp=cupy) and CPU Numba (xp=numpy) paths.

    Parameters
    ----------
    y_vol      - (X x Y x Z x T) numpy array (always CPU on entry)
    param      - full param dict
    xp         - array module (cupy or numpy)
    use_numba  - bool, passed to gradient/div/TV functions
    chunk_size - int, number of time points per chunk
    label      - str, printed in progress messages ('GPU' or 'Numba')
    """
    dim = param['Dimension']
    T   = dim[3]

    x_vol_out = np.zeros_like(y_vol)
    n_chunks  = int(np.ceil(T / chunk_size))

    for c in range(n_chunks):
        t0 = c * chunk_size
        t1 = min(t0 + chunk_size, T)
        print(f"  {label} spatial: chunk {c+1}/{n_chunks} "
              f"(time points {t0}–{t1-1})...", flush=True)

        # Extract chunk — move to GPU if needed
        y_chunk  = xp.asarray(y_vol[:, :, :, t0:t1])
        wx_chunk = xp.asarray(param['weight_x'][:, :, :, t0:t1])
        wy_chunk = xp.asarray(param['weight_y'][:, :, :, t0:t1])
        wz_chunk = xp.asarray(param['weight_z'][:, :, :, t0:t1])

        # Build a param dict with the correct T for this chunk
        chunk_param        = {k: v for k, v in param.items()}
        chunk_dim          = list(dim)
        chunk_dim[3]       = t1 - t0
        chunk_param['Dimension'] = chunk_dim

        # Run MyProx on this chunk
        x_chunk = _run_myprox_chunk(
            y_chunk, wx_chunk, wy_chunk, wz_chunk,
            chunk_param, xp,
            use_numba=(use_numba and xp is np)  # Numba only on CPU
        )

        # Move back to CPU if needed and store
        x_vol_out[:, :, :, t0:t1] = (xp.asnumpy(x_chunk)
                                      if xp is not np else x_chunk)

        # Free memory explicitly before next chunk
        del y_chunk, wx_chunk, wy_chunk, wz_chunk, x_chunk
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()

    return x_vol_out


def ta_spatial(y, param):
    """
    Computes the TV regularization:
        F(x) = min ||y - x||^2 + lambda * ||TV{x}||_1
    using the 3D extension of FISTA.

    Execution path is selected from param flags set in Inputs_TA.py:

        use_gpu=1    — GPU via CuPy, chunked along T by gpu_chunk_size.
                       VRAM has no swap fallback — always set chunk_size.
        use_numba=1  — Numba JIT with prange over T, optionally chunked
                       along T by numba_chunk_size if RAM is limited.
        default      — in-place NumPy, no chunking needed (uses ~3.5GB).

    Chunking is mathematically identical to processing all T at once
    because spatial TV regularization operates on each time point
    independently — there is no temporal coupling in the spatial step.

    Inputs:
        y     - (n_time_points x n_ret_voxels) 2D array
        param - dict containing TA parameters including:
            'use_gpu'          - int flag (0/1)
            'use_numba'        - int flag (0/1)
            'gpu_chunk_size'   - int or None (see Inputs_TA.py for guidance)
            'numba_chunk_size' - int or None (see Inputs_TA.py for guidance)
            'NitSpat'          - default 400
            'LambdaSpat'       - default 2
            'Lip'              - default 12
            'tol'              - default 1e-6
            'VoxelIdx'         - (n_ret_vox x 3) 3D coordinates
            'Dimension'        - [X, Y, Z, T]
            'weight_x/y/z'     - spatial weight volumes

    Outputs:
        x_out - (n_time_points x n_ret_voxels) 2D numpy array

    Implemented by Younes Farouj, 10.03.2016
    """
    param.setdefault('NitSpat',           400)
    param.setdefault('LambdaSpat',        2)
    param.setdefault('Lip',               12)
    param.setdefault('tol',               1e-6)
    param.setdefault('gpu_chunk_size',    40)
    param.setdefault('numba_chunk_size',  None)

    xp, use_numba = get_backend(param)

    dim = param['Dimension']   # (X, Y, Z, T)
    T   = dim[3]

    param.setdefault('weight_x', np.ones(dim))
    param.setdefault('weight_y', np.ones(dim))
    param.setdefault('weight_z', np.ones(dim))

    x_out = np.zeros_like(y)

    # Cache voxel index arrays (computed once across all outer TA iterations)
    if 'VoxelIdx_xyz' not in param:
        vox = np.asarray(param['VoxelIdx'], dtype=np.intp)
        param['VoxelIdx_xyz'] = (vox[:, 0], vox[:, 1], vox[:, 2])
    xi, yi, zi = param['VoxelIdx_xyz']

    # Convert 2D input (T x V) to 4D volume (X x Y x Z x T)
    y_vol             = np.zeros(dim, dtype=y.dtype)
    y_vol[xi, yi, zi, :] = y.T

    # ── GPU chunked path ──────────────────────────────────────────────────────
    if xp is not np:
        chunk_size = param['gpu_chunk_size']
        if chunk_size is None:
            chunk_size = T   # only safe on A100-class GPUs (>63GB VRAM)

        x_vol = _run_chunked(y_vol, param, xp, use_numba=False,
                             chunk_size=chunk_size, label='GPU')

    # ── Numba chunked path ────────────────────────────────────────────────────
    elif use_numba:
        chunk_size = param['numba_chunk_size']
        if chunk_size is None:
            # No chunking — process all T at once.
            # Uses ~63GB RAM for a typical brain volume.
            x_vol = _run_myprox_chunk(
                y_vol,
                param['weight_x'], param['weight_y'], param['weight_z'],
                param, xp=np, use_numba=True
            )
        else:
            x_vol = _run_chunked(y_vol, param, xp=np, use_numba=True,
                                 chunk_size=chunk_size, label='Numba')

    # ── Plain NumPy path ──────────────────────────────────────────────────────
    else:
        # In-place NumPy — no chunking needed.
        # Working memory is ~3.5GB (one 4D volume); all 18 MyProx arrays
        # are pre-allocated once and reused across all 400 iterations.
        x_vol = _run_myprox_chunk(
            y_vol,
            param['weight_x'], param['weight_y'], param['weight_z'],
            param, xp=np, use_numba=False
        )

    x_out[:] = x_vol[xi, yi, zi, :].T
    return x_out
