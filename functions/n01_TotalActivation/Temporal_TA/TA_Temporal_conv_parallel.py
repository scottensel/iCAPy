import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

from functions.n01_TotalActivation.Temporal_TA.TA_Temporal_OneTimeCourse_parallel import ta_temporal_onetimecourse_parallel


def cconv(a, b, N=None):
    """
    Circular convolution (mod-N) of two 1D vectors a and b.

    Equivalent to MATLAB's cconv(a, b, N). Used here to apply the
    Daubechies high-pass wavelet filter to each voxel time course for
    noise estimation via MAD of wavelet coefficients.

    Inputs:
        a, b - 1D array_like, input sequences (real or complex)
        N    - int, optional; output length. If None, defaults to
               len(a) + len(b) - 1, matching MATLAB's cconv default

    Outputs:
        c - (N,) ndarray, circular convolution result; real-valued if
            both inputs are real
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if N is None:
        N = a.size + b.size - 1

    # FFT-based circular convolution of length N
    c = np.fft.ifft(np.fft.fft(a, N) * np.fft.fft(b, N))

    # If both inputs are real, discard negligible imaginary components
    if np.isrealobj(a) and np.isrealobj(b):
        c = c.real

    return c


def _worker_voxel_range(v0, v1,
                        shm_tcn_name, shm_out_name, shm_lam_name,
                        shm_lamfin_name, shm_noisefin_name,
                        tcn_shape, out_shape, dtype_str,
                        g, scalarsIn, lambda_temp_coef):
    """
    Worker function executed in each subprocess.

    Processes voxels in the range [v0, v1) using shared memory arrays so
    that no data is copied between processes. Each worker independently
    computes the MAD-based lambda estimate and runs the temporal
    regularization for its assigned voxels, writing results directly into
    the shared output array.

    Inputs:
        v0, v1           - int, start and end voxel indices for this worker
        shm_*_name       - str, names of shared memory blocks for:
                             TCN input, TC_OUT output, LambdaTemp,
                             LambdaTempFin, NoiseEstimateFin
        tcn_shape        - tuple, shape of the TCN array (n_time x n_vox)
        out_shape        - tuple, shape of the TC_OUT array
        dtype_str        - str, numpy dtype string (e.g. '<f8' for float64)
        g                - (5,) array, Daubechies high-pass wavelet filter
        scalarsIn        - list of pre-extracted TA scalars:
                             [n, d, maxeig, N, Nit]
        lambda_temp_coef - float, coefficient to scale MAD noise estimate
    """
    dtype = np.dtype(dtype_str)

    # Attach to shared memory blocks — no data is copied
    shm_tcn      = shared_memory.SharedMemory(name=shm_tcn_name)
    shm_out      = shared_memory.SharedMemory(name=shm_out_name)
    shm_lam      = shared_memory.SharedMemory(name=shm_lam_name)
    shm_lamfin   = shared_memory.SharedMemory(name=shm_lamfin_name)
    shm_noisefin = shared_memory.SharedMemory(name=shm_noisefin_name)

    try:
        TCN              = np.ndarray(tcn_shape,        dtype=dtype, buffer=shm_tcn.buf)
        TC_OUT           = np.ndarray(out_shape,        dtype=dtype, buffer=shm_out.buf)
        V                = tcn_shape[1]
        LambdaTemp       = np.ndarray((V,), dtype=dtype, buffer=shm_lam.buf)
        LambdaTempFin    = np.ndarray((V,), dtype=dtype, buffer=shm_lamfin.buf)
        NoiseEstimateFin = np.ndarray((V,), dtype=dtype, buffer=shm_noisefin.buf)

        for i in range(v0, v1):

            # Step 1: Wavelet decomposition of the voxel time course using
            # circular convolution with the Daubechies high-pass filter
            coef = cconv(TCN[:, i], g)

            # Step 2: MAD-based noise estimate — robust estimate of the
            # noise level for this voxel's time course
            LambdaTemp[i] = np.median(np.abs(coef - np.median(coef))) * lambda_temp_coef

            # Step 3: Run temporal regularization for this voxel;
            # results are written in-place into TC_OUT, LambdaTempFin,
            # and NoiseEstimateFin via the shared memory arrays
            x = ta_temporal_onetimecourse_parallel(
                TCN[:, i], i, scalarsIn, LambdaTemp, LambdaTempFin, NoiseEstimateFin
            )
            TC_OUT[:, i] = x

            if i % 10000 == 0:
                print(f"{mp.current_process().name}: voxel {i}", flush=True)

    finally:
        # Always close shared memory handles, even if an exception occurs
        shm_tcn.close()
        shm_out.close()
        shm_lam.close()
        shm_lamfin.close()
        shm_noisefin.close()


def ta_temporal_parallel(TCN, param):
    """
    Performs the temporal regularization part of total activation across
    all voxels in parallel using shared memory (Windows-compatible).

    Equivalent to ta_temporal (TA_Temporal_conv.py) but distributes voxels
    across multiple processes using Python's multiprocessing with shared
    memory so that no large arrays are pickled or copied between processes.

    Warm-start behaviour (matching the sequential version):
        - If param['NoiseEstimateFin'][i] is finite, use it as the initial
          lambda for voxel i to speed up convergence
        - Otherwise fall back to the MAD-based LambdaTemp[i]

    Inputs:
        TCN   - (n_time_points x n_ret_voxels) 2D array of data input to
                the regularization; will be cast to float64
        param - dict containing all TA-relevant parameters:
            'Dimension'        - 4-element list/array of X, Y, Z, T sizes
            'NbrVoxels'        - number of voxels to process
            'LambdaTempCoef'   - coefficient to scale the MAD noise
                                 estimate into the initial lambda
            'filter_analyze'   - dict with 'num' and 'den' filter arrays
            'maxeig'           - maximum eigenvalue of the filter operator
            'NitTemp'          - number of iterations per voxel
            ['NoiseEstimateFin'] - (n_voxels,) warm-start noise estimates
                                 from a previous TA iteration; NaN entries
                                 fall back to LambdaTemp
            ['LambdaTempFin']  - (n_voxels,) warm-start lambda values from
                                 a previous TA iteration

    Outputs:
        TC_OUT - (n_time_points x n_ret_voxels) 2D array of outputs from
                 the regularization step
        param  - updated dict with the following fields added/updated:
            'LambdaTemp'       - (n_voxels,) MAD-based initial lambda per
                                 voxel
            'LambdaTempFin'    - (n_voxels,) final lambda per voxel after
                                 convergence; persisted for warm-starting
            'NoiseEstimateFin' - (n_voxels,) final noise estimate per voxel
                                 after convergence; persisted for
                                 warm-starting
    """
    TCN    = np.asarray(TCN, dtype=np.float64, order="C")
    n_rows = param['Dimension'][3]
    n_cols = param['NbrVoxels']

    if TCN.shape != (n_rows, n_cols):
        raise ValueError(f"TCN shape {TCN.shape} does not match expected {(n_rows, n_cols)}")

    # Pre-extract scalars so workers receive plain Python objects,
    # not the full param dict (avoids pickling overhead)
    n      = param['filter_analyze']['num']
    d      = param['filter_analyze']['den']
    maxeig = float(param['maxeig'])
    Nit    = int(param['NitTemp'])
    scalarsIn = [n, d, maxeig, n_rows, Nit]

    lambda_temp_coef = float(param["LambdaTempCoef"])

    # Daubechies high-pass wavelet filter (4 vanishing moments)
    g = np.array([0, -0.12941, -0.22414, 0.83652, -0.48296], dtype=np.float64)

    # Ensure warm-start arrays exist with the correct dtype and shape.
    # NaN signals "no previous estimate available" for a given voxel.
    if 'NoiseEstimateFin' not in param or param['NoiseEstimateFin'] is None:
        param['NoiseEstimateFin'] = np.full(n_cols, np.nan, dtype=np.float64)
    else:
        param['NoiseEstimateFin'] = np.asarray(param['NoiseEstimateFin'], dtype=np.float64)

    if 'LambdaTempFin' not in param or param['LambdaTempFin'] is None:
        param['LambdaTempFin'] = np.full(n_cols, np.nan, dtype=np.float64)
    else:
        param['LambdaTempFin'] = np.asarray(param['LambdaTempFin'], dtype=np.float64)

    # Allocate shared memory blocks — one per array that workers read or write
    shm_tcn      = shared_memory.SharedMemory(create=True, size=TCN.nbytes)
    shm_out      = shared_memory.SharedMemory(create=True, size=(n_rows * n_cols * 8))
    shm_lam      = shared_memory.SharedMemory(create=True, size=(n_cols * 8))
    shm_lamfin   = shared_memory.SharedMemory(create=True, size=(n_cols * 8))
    shm_noisefin = shared_memory.SharedMemory(create=True, size=(n_cols * 8))

    try:
        # Create numpy views into the shared memory buffers
        shm_TCN      = np.ndarray((n_rows, n_cols), dtype=np.float64, buffer=shm_tcn.buf)
        shm_TCOUT    = np.ndarray((n_rows, n_cols), dtype=np.float64, buffer=shm_out.buf)
        shm_Lam      = np.ndarray((n_cols,),        dtype=np.float64, buffer=shm_lam.buf)
        shm_LamFin   = np.ndarray((n_cols,),        dtype=np.float64, buffer=shm_lamfin.buf)
        shm_NoiseFin = np.ndarray((n_cols,),        dtype=np.float64, buffer=shm_noisefin.buf)

        # Initialise shared contents
        shm_TCN[:]   = TCN
        shm_TCOUT[:] = 0.0
        shm_Lam[:]   = 0.0

        # Seed warm-start arrays from param — preserve previous estimates
        # so that re-entering TA does not restart lambda from scratch
        shm_LamFin[:]   = param['LambdaTempFin']
        shm_NoiseFin[:] = param['NoiseEstimateFin']

        # Spawn worker processes, each handling a contiguous chunk of voxels
        ctx    = mp.get_context("spawn")
        n_procs = ctx.cpu_count()
        print("Using", n_procs, "processes", flush=True)

        chunk = (n_cols + n_procs - 1) // n_procs
        procs = []

        for k in range(n_procs):
            v0 = k * chunk
            v1 = min((k + 1) * chunk, n_cols)
            if v0 >= v1:
                break

            p = ctx.Process(
                target=_worker_voxel_range,
                args=(v0, v1,
                      shm_tcn.name, shm_out.name, shm_lam.name,
                      shm_lamfin.name, shm_noisefin.name,
                      (n_rows, n_cols), (n_rows, n_cols),
                      np.dtype(np.float64).str,
                      g, scalarsIn, lambda_temp_coef)
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode not in (0, None):
                raise RuntimeError(
                    f"Worker {p.name} crashed with exit code {p.exitcode}"
                )

        # Copy results out of shared memory back into regular numpy arrays
        TC_OUT = np.array(shm_TCOUT, copy=True)
        param["LambdaTemp"]        = np.array(shm_Lam,      copy=True)

        # Persist warm-start arrays in param for the next TA iteration
        param["LambdaTempFin"]    = np.array(shm_LamFin,   copy=True)
        param["NoiseEstimateFin"] = np.array(shm_NoiseFin, copy=True)

        return TC_OUT, param

    finally:
        # Release and unlink all shared memory blocks regardless of outcome
        shm_tcn.close();      shm_tcn.unlink()
        shm_out.close();      shm_out.unlink()
        shm_lam.close();      shm_lam.unlink()
        shm_lamfin.close();   shm_lamfin.unlink()
        shm_noisefin.close(); shm_noisefin.unlink()