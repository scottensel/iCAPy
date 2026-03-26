import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

from functions.n01_TotalActivation.Temporal_TA.TA_Temporal_OneTimeCourse_parallel import ta_temporal_onetimecourse_parallel


def cconv(a, b, N=None):
    a = np.asarray(a)
    b = np.asarray(b)
    if N is None:
        N = a.size + b.size - 1
    c = np.fft.ifft(np.fft.fft(a, N) * np.fft.fft(b, N))
    if np.isrealobj(a) and np.isrealobj(b):
        c = c.real
    return c


def _worker_voxel_range(v0, v1,
                        shm_tcn_name, shm_out_name, shm_lam_name, shm_lamfin_name, shm_noisefin_name,
                        tcn_shape, out_shape, dtype_str,
                        g, scalarsIn, lambda_temp_coef):
    dtype = np.dtype(dtype_str)

    shm_tcn = shared_memory.SharedMemory(name=shm_tcn_name)
    shm_out = shared_memory.SharedMemory(name=shm_out_name)
    shm_lam = shared_memory.SharedMemory(name=shm_lam_name)
    shm_lamfin = shared_memory.SharedMemory(name=shm_lamfin_name)
    shm_noisefin = shared_memory.SharedMemory(name=shm_noisefin_name)

    try:
        TCN = np.ndarray(tcn_shape, dtype=dtype, buffer=shm_tcn.buf)
        TC_OUT = np.ndarray(out_shape, dtype=dtype, buffer=shm_out.buf)

        V = tcn_shape[1]
        LambdaTemp = np.ndarray((V,), dtype=dtype, buffer=shm_lam.buf)
        LambdaTempFin = np.ndarray((V,), dtype=dtype, buffer=shm_lamfin.buf)
        NoiseEstimateFin = np.ndarray((V,), dtype=dtype, buffer=shm_noisefin.buf)

        for i in range(v0, v1):
            coef = cconv(TCN[:, i], g)
            LambdaTemp[i] = np.median(np.abs(coef - np.median(coef))) * lambda_temp_coef

            x = ta_temporal_onetimecourse_parallel(
                TCN[:, i], i, scalarsIn, LambdaTemp, LambdaTempFin, NoiseEstimateFin
            )
            TC_OUT[:, i] = x

            if i % 10000 == 0:
                print(f"{mp.current_process().name}: voxel {i}", flush=True)

    finally:
        shm_tcn.close()
        shm_out.close()
        shm_lam.close()
        shm_lamfin.close()
        shm_noisefin.close()


def ta_temporal_parallel(TCN, param):
    """
    Parallel TA temporal step across voxels using shared memory (Windows-friendly).
    Warm-start behavior:
      - If param['NoiseEstimateFin'][i] is finite -> use it as initial lambda
      - Else -> use LambdaTemp[i]
    """
    TCN = np.asarray(TCN, dtype=np.float64, order="C")
    n_rows = param['Dimension'][3]
    n_cols = param['NbrVoxels']

    if TCN.shape != (n_rows, n_cols):
        raise ValueError(f"TCN shape {TCN.shape} does not match expected {(n_rows, n_cols)}")

    # Scalars for ta_temporal_onetimecourse
    n = param['filter_analyze']['num']
    d = param['filter_analyze']['den']
    maxeig = float(param['maxeig'])
    Nit = int(param['NitTemp'])
    scalarsIn = [n, d, maxeig, n_rows, Nit]

    lambda_temp_coef = float(param["LambdaTempCoef"])
    g = np.array([0, -0.12941, -0.22414, 0.83652, -0.48296], dtype=np.float64)

    # ---- Ensure warm-start arrays exist and are correct dtype/shape ----
    if 'NoiseEstimateFin' not in param or param['NoiseEstimateFin'] is None:
        param['NoiseEstimateFin'] = np.full(n_cols, np.nan, dtype=np.float64)
    else:
        param['NoiseEstimateFin'] = np.asarray(param['NoiseEstimateFin'], dtype=np.float64)

    if 'LambdaTempFin' not in param or param['LambdaTempFin'] is None:
        param['LambdaTempFin'] = np.full(n_cols, np.nan, dtype=np.float64)
    else:
        param['LambdaTempFin'] = np.asarray(param['LambdaTempFin'], dtype=np.float64)

    # ---- create shared memory blocks ----
    shm_tcn = shared_memory.SharedMemory(create=True, size=TCN.nbytes)
    shm_out = shared_memory.SharedMemory(create=True, size=(n_rows * n_cols * 8))
    shm_lam = shared_memory.SharedMemory(create=True, size=(n_cols * 8))
    shm_lamfin = shared_memory.SharedMemory(create=True, size=(n_cols * 8))
    shm_noisefin = shared_memory.SharedMemory(create=True, size=(n_cols * 8))

    try:
        shm_TCN = np.ndarray((n_rows, n_cols), dtype=np.float64, buffer=shm_tcn.buf)
        shm_TCOUT = np.ndarray((n_rows, n_cols), dtype=np.float64, buffer=shm_out.buf)
        shm_Lam = np.ndarray((n_cols,), dtype=np.float64, buffer=shm_lam.buf)
        shm_LamFin = np.ndarray((n_cols,), dtype=np.float64, buffer=shm_lamfin.buf)
        shm_NoiseFin = np.ndarray((n_cols,), dtype=np.float64, buffer=shm_noisefin.buf)

        # Initialize shared contents
        shm_TCN[:] = TCN
        shm_TCOUT[:] = 0.0
        shm_Lam[:] = 0.0

        # ---- CRITICAL CHANGE: seed warm-start arrays from param (do NOT reset to NaN each call) ----
        shm_LamFin[:] = param['LambdaTempFin']      # may contain NaNs or previous values
        shm_NoiseFin[:] = param['NoiseEstimateFin'] # may contain NaNs or previous values

        # ---- multiprocessing chunking ----
        ctx = mp.get_context("spawn")
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
                      shm_tcn.name, shm_out.name, shm_lam.name, shm_lamfin.name, shm_noisefin.name,
                      (n_rows, n_cols), (n_rows, n_cols), np.dtype(np.float64).str,
                      g, scalarsIn, lambda_temp_coef)
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode not in (0, None):
                raise RuntimeError(f"Worker {p.name} crashed with exit code {p.exitcode}")

        # Copy results back
        TC_OUT = np.array(shm_TCOUT, copy=True)
        param["LambdaTemp"] = np.array(shm_Lam, copy=True)

        # Persist warm-start arrays for next TA iteration
        param["LambdaTempFin"] = np.array(shm_LamFin, copy=True)
        param["NoiseEstimateFin"] = np.array(shm_NoiseFin, copy=True)

        return TC_OUT, param

    finally:
        shm_tcn.close(); shm_tcn.unlink()
        shm_out.close(); shm_out.unlink()
        shm_lam.close(); shm_lam.unlink()
        shm_lamfin.close(); shm_lamfin.unlink()
        shm_noisefin.close(); shm_noisefin.unlink()
