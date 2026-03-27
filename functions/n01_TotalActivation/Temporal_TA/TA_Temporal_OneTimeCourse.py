from functions.n01_TotalActivation.Temporal_TA.filter_boundary import filter_boundary
import numpy as np


def ta_temporal_onetimecourse(y, idx_vox, ParametersIn):
    """
    Performs the temporal regularization process on one provided time course.

    Inputs:
        y            - (N,) array, noisy signal (one voxel time course)
        idx_vox      - int, index of the current voxel
        ParametersIn - dict containing necessary parameters:
            'filter_analyze'   - dict with keys:
                'num'              - numerator of the deconv + derivation
                                     operator
                'den'              - denominator of the deconv + derivation
                                     operator
            'maxeig'           - maximal eigenvalue of the operator, used
                                 to determine the step size of the algorithm
                                 for convergence
            'Dimension'        - 4-element list/array of X, Y, Z, T sizes;
                                 T (index 3) gives the number of time points
            'NitTemp'          - number of iterations to run temporal
                                 regularization for
            'LambdaTemp'       - (n_voxels,) array, MAD of wavelet
                                 coefficients used as initial regularization
                                 parameter at the first iteration
            ['NoiseEstimateFin'] - (n_voxels,) array, final noise estimates
                                 from a previous call; if present and finite
                                 for this voxel, used as warm-start lambda
                                 to speed up convergence

    Outputs:
        x            - (N,) array, estimated activity-related signal after
                       temporal regularization
        ParametersOut - dict containing updated estimates for warm-starting:
            'NoiseEstimateFin'  - final noise estimate from the last
                                  iteration (nv[-1])
            'LambdasTempFin'    - final regularization parameter from the
                                  last iteration (Lambda[-1])

    Implemented by Isik Karahanoglu, 28.11.2011
    """

    y = np.array(y)

    # Numerator and denominator of the 'deconv + derivation' operator
    n = ParametersIn['filter_analyze']['num']
    d = ParametersIn['filter_analyze']['den']

    # Maximal eigenvalue of the operator (used to determine the step size
    # of the algorithm for convergence)
    maxeig = ParametersIn['maxeig']

    # Number of time points
    N = ParametersIn['Dimension'][3]

    # Current value of the number of iterations to run temporal
    # regularization for
    Nit = ParametersIn['NitTemp']

    # If we have already gone through the whole process, lambda has been
    # updated into a gradually more accurate estimate, and so we re-use the
    # final value obtained as a start lambda
    if 'NoiseEstimateFin' in ParametersIn and len(ParametersIn['NoiseEstimateFin']) - 1 >= idx_vox:
        lambda_ = ParametersIn['NoiseEstimateFin'][idx_vox]
    # If we are on the first iteration, take the MAD of wavelet coefficients
    # (computed before the call to Temporal_TA) as 'lambda'
    else:
        lambda_ = ParametersIn['LambdaTemp'][idx_vox]

    # noise_estimate contains the MAD of wavelet coefficients, i.e. the
    # initial value of the regularization parameter at iteration 1 of the
    # first forward-backward call
    noise_estimate = np.array(ParametersIn['LambdaTemp'][idx_vox], dtype=np.float64)

    # nv contains our noise estimate made at each iteration of the algorithm
    # (how far is our solution from the recorded data)
    nv = np.zeros(Nit)

    # Lambda will contain the regularization estimates at each iteration
    Lambda = np.zeros(Nit)

    # Arbitrarily set precision threshold
    precision = noise_estimate / 100000

    # z will contain our dual variable
    z = np.zeros(N)

    # t is an auxiliary variable used in the weight boosting process to
    # speed up convergence
    t = np.array(1)

    # s is the other auxiliary variable used in the boosting process (the
    # 'boosted' version of z)
    s = np.zeros(N)

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
        z = (1 / (lambda_ * maxeig)) * filtered_input + s - filtered_s / maxeig
        z = np.clip(z, -1, 1)

        # t_prev stores the kth value; t and s are updated as part of the
        # FISTA boosting scheme that speeds up convergence
        t_prev = t
        t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        s = z + ((t_prev - 1) / t) * (z - z_prev)

        # nv measures how far we are from the recorded data at this
        # iteration: it is related to y - x(k+1), our effective noise
        # quantification assuming the residual is pure noise
        nv[k] = np.sqrt(np.mean((lambda_ * filter_boundary(n, d, z, 'transpose')) ** 2))

        # If the effective noise estimate differs from the initial noise
        # estimate, update lambda accordingly:
        # - if nv[k] < noise_estimate, lambda increases (sparsify more)
        # - if nv[k] > noise_estimate, lambda decreases (stay closer to data)
        if np.abs(nv[k] - noise_estimate) > precision:
            lambda_ *= noise_estimate / nv[k]

        # Store the updated lambda value at this iteration
        Lambda[k] = lambda_

    # When the algorithm exits, use the converged dual variable to recover
    # the primal variable (the estimated activity-related signal x)
    x = y - lambda_ * filter_boundary(n, d, z, 'transpose')

    # Store the final noise estimate and regularization parameter so that
    # if Temporal_TA is called again, they are used directly as warm-start
    ParametersOut = {
        'NoiseEstimateFin': nv[-1],
        'LambdasTempFin':   Lambda[-1],
    }

    return x, ParametersOut