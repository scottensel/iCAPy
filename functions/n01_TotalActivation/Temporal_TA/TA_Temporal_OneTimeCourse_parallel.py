import numpy as np
from functions.n01_TotalActivation.Temporal_TA.filter_boundary import filter_boundary


def ta_temporal_onetimecourse_parallel(y, idx_vox, scalarsIn,
                                       LambdaTemp, LambdaTempFin, NoiseEstimateFin):
    """
    Performs the temporal regularization process on one provided time course.
    Parallelism-friendly version: all parameters are passed as flat arrays
    and scalars rather than a parameter dict, avoiding shared-memory issues
    when called from a multiprocessing pool.

    Inputs:
        y                - (N,) array, noisy signal (one voxel time course)
        idx_vox          - int, index of the current voxel
        scalarsIn        - tuple of pre-extracted TA scalar parameters:
                             n        - numerator of the deconv + derivation
                                        operator
                             d        - denominator of the deconv + derivation
                                        operator
                             maxeig   - maximal eigenvalue of the operator,
                                        used to determine the step size for
                                        convergence
                             N        - number of time points
                             Nit      - number of iterations to run temporal
                                        regularization for
        LambdaTemp       - (n_voxels,) array, MAD of wavelet coefficients
                           used as the initial regularization parameter at
                           the first iteration
        LambdaTempFin    - (n_voxels,) shared array; stores the final lambda
                           value from the last iteration and is updated
                           in-place for warm-starting subsequent calls
        NoiseEstimateFin - (n_voxels,) shared array; stores the final noise
                           estimate from the last iteration and is updated
                           in-place for warm-starting subsequent calls

    Outputs:
        x - (N,) array, estimated activity-related signal after temporal
            regularization

    Side effects:
        LambdaTempFin[idx_vox]    is updated with Lambda[-1]
        NoiseEstimateFin[idx_vox] is updated with nv[-1]

    Implemented by Isik Karahanoglu, 28.11.2011
    """

    y = np.asarray(y)

    # Unpack pre-extracted scalars
    n, d, maxeig, N, Nit = scalarsIn

    # If we have already gone through the whole process, lambda has been
    # updated into a gradually more accurate estimate, and so we re-use the
    # final value obtained as a start lambda
    prev_noise = float(NoiseEstimateFin[idx_vox])
    if np.isfinite(prev_noise):
        lambda_ = prev_noise
    # If we are on the first iteration, take the MAD of wavelet coefficients
    # (computed before the call to Temporal_TA) as 'lambda'
    else:
        lambda_ = float(LambdaTemp[idx_vox])

    # noise_estimate contains the MAD of wavelet coefficients, i.e. the
    # initial value of the regularization parameter at iteration 1 of the
    # first forward-backward call
    noise_estimate = float(LambdaTemp[idx_vox])

    # nv contains our noise estimate made at each iteration of the algorithm
    # (how far is our solution from the recorded data)
    nv = np.zeros(Nit, dtype=np.float64)

    # Lambda will contain the regularization estimates at each iteration
    Lambda = np.zeros(Nit, dtype=np.float64)

    # Arbitrarily set precision threshold
    precision = noise_estimate / 100000.0

    # z will contain our dual variable
    z = np.zeros(N, dtype=np.float64)

    # t is an auxiliary variable used in the weight boosting process to
    # speed up convergence
    t = 1.0

    # s is the other auxiliary variable used in the boosting process (the
    # 'boosted' version of z)
    s = np.zeros(N, dtype=np.float64)

    # Pre-compute the filtered input once since y does not change
    filtered_input = filter_boundary(n, d, y, 'normal')

    # We run the optimisation algorithm for Nit iterations; note that there
    # is no other convergence control, only this pre-selected value
    for k in range(Nit):

        # z_prev contains the dual estimate from iteration k
        z_prev = z.copy()

        # Computation of the dual variable at iteration k+1; it involves
        # the regularization parameter lambda, the step size maxeig, and
        # the application of the filter (deconvolution + derivative) to the
        # recorded signal y or the kth boosted dual estimate s.
        # The final step clips the result to [-1, 1].
        filtered_transpose = filter_boundary(n, d, s, 'transpose')
        filtered_s = filter_boundary(n, d, filtered_transpose, 'normal')
        z = (1.0 / (lambda_ * maxeig)) * filtered_input + s - filtered_s / maxeig
        np.clip(z, -1, 1, out=z)

        # t_prev stores the kth value; t and s are updated as part of the
        # FISTA boosting scheme that speeds up convergence
        t_prev = t
        t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        s = z + ((t_prev - 1.0) / t) * (z - z_prev)

        # nv measures how far we are from the recorded data at this
        # iteration: it is related to y - x(k+1), our effective noise
        # quantification assuming the residual is pure noise
        At_z = filter_boundary(n, d, z, 'transpose')
        nv[k] = lambda_ * np.sqrt(np.mean(At_z * At_z))

        # If the effective noise estimate differs from the initial noise
        # estimate, update lambda accordingly:
        # - if nv[k] < noise_estimate, lambda increases (sparsify more)
        # - if nv[k] > noise_estimate, lambda decreases (stay closer to data)
        if abs(nv[k] - noise_estimate) > precision:
            lambda_ *= noise_estimate / nv[k]

        # Store the updated lambda value at this iteration
        Lambda[k] = lambda_

    # When the algorithm exits, use the converged dual variable to recover
    # the primal variable (the estimated activity-related signal x)
    x = y - lambda_ * filter_boundary(n, d, z, 'transpose')

    # Store the final noise estimate and regularization parameter so that
    # if Temporal_TA is called again, they are used directly as warm-start
    LambdaTempFin[idx_vox]    = Lambda[-1]
    NoiseEstimateFin[idx_vox] = nv[-1]

    return x