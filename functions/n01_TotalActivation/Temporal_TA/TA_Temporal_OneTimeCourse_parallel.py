import numpy as np
from functions.n01_TotalActivation.Temporal_TA.filter_boundary import filter_boundary


def ta_temporal_onetimecourse_parallel(y, idx_vox, scalarsIn,
                                       LambdaTemp, LambdaTempFin, NoiseEstimateFin):
    y = np.asarray(y)
    n, d, maxeig, N, Nit = scalarsIn

    # Warm-start rule:
    # - if NoiseEstimateFin already exists (finite), use it as initial lambda
    # - otherwise fall back to LambdaTemp
    prev_noise = float(NoiseEstimateFin[idx_vox])
    if np.isfinite(prev_noise):
        lambda_ = prev_noise
    else:
        lambda_ = float(LambdaTemp[idx_vox])

    noise_estimate = float(LambdaTemp[idx_vox])

    nv = np.zeros(Nit, dtype=np.float64)
    Lambda = np.zeros(Nit, dtype=np.float64)
    precision = noise_estimate / 100000.0

    z = np.zeros(N, dtype=np.float64)
    s = np.zeros(N, dtype=np.float64)
    t = 1.0

    filtered_input = filter_boundary(n, d, y, 'normal')

    for k in range(Nit):
        z_prev = z.copy()

        filtered_transpose = filter_boundary(n, d, s, 'transpose')
        filtered_s = filter_boundary(n, d, filtered_transpose, 'normal')

        z = (1.0 / (lambda_ * maxeig)) * filtered_input + s - filtered_s / maxeig
        np.clip(z, -1, 1, out=z)

        t_prev = t
        t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        s = z + ((t_prev - 1.0) / t) * (z - z_prev)

        At_z = filter_boundary(n, d, z, 'transpose')
        nv[k] = lambda_ * np.sqrt(np.mean(At_z * At_z))

        if abs(nv[k] - noise_estimate) > precision:
            lambda_ *= noise_estimate / nv[k]

        Lambda[k] = lambda_

    x = y - lambda_ * filter_boundary(n, d, z, 'transpose')

    # Save outputs for next warm-start:
    # LambdaTempFin holds lambda, NoiseEstimateFin holds noise estimate
    LambdaTempFin[idx_vox] = Lambda[-1]
    NoiseEstimateFin[idx_vox] = nv[-1]

    return x