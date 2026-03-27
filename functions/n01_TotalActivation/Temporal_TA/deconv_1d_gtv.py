import numpy as np
from functions.n01_TotalActivation.Temporal_TA.filter_boundary import filter_boundary


def deconv_1d_gtv(y, n, lambda_val, Nit, maxeig, d=1, p_fig=0, noise_estimate=1e-4):
    """
    1-D deconvolution with generalised total variation regularization,
    using a forward-backward splitting algorithm with FISTA acceleration.

    This is the low-level single time course solver that underlies the
    temporal total activation step. See TA_Temporal_OneTimeCourse for the
    voxel-loop wrapper that calls this routine.

    Inputs:
        y              - (N,) array, observed noisy signal
        n              - numerator coefficients of the analysis filter
                         (deconvolution + derivation operator)
        lambda_val     - float, initial regularization parameter (lambda)
        Nit            - int, maximum number of iterations
        maxeig         - float, maximum eigenvalue of the filter operator;
                         used to set the step size for convergence
        d              - denominator coefficients of the analysis filter;
                         defaults to 1 (no IIR part, pure FIR)
        p_fig          - int, if 1 plots the original and filtered signal
                         at each iteration (for debugging); default 0
        noise_estimate - float, target noise level used to update lambda;
                         defaults to 1e-4

    Outputs:
        x      - (N,) array, estimated activity-related signal after
                 deconvolution and regularization
        nv     - (Nit,) array, effective noise estimate at each iteration
                 (sqrt of mean squared residual)
        J      - (Nit,) array, value of the cost function at each iteration
                 (data fidelity + regularization term)
        Lambda - (Nit,) array, value of the regularization parameter lambda
                 at each iteration
    """

    # Stopping criterion on the cost function change between iterations
    stop_cri = 1e-6

    N = len(y)

    # Dual variable z and boosted dual variable s, both initialised to zero
    z = np.zeros(N)
    s = np.zeros(N)

    # t is the auxiliary FISTA boosting variable
    t = 1

    # Storage arrays for cost, noise estimate, and lambda across iterations
    J      = np.zeros(Nit)
    nv     = np.zeros(Nit)
    Lambda = np.zeros(Nit)

    # Precision threshold for lambda update — same as TA_Temporal_OneTimeCourse
    precision = noise_estimate / 100000

    for k in range(Nit):

        # z_l contains the dual estimate from iteration k
        z_l = z.copy()

        # Dual variable update at iteration k+1:
        # involves lambda, the step size maxeig, and the analysis filter
        z = (1 / (lambda_val * maxeig) * filter_boundary(n, d, y, 'normal') +
             s - filter_boundary(n, d, filter_boundary(n, d, s, 'transpose'), 'normal') / maxeig)

        # Clip dual variable to the unit ball (projection step)
        z = np.clip(z, -1, 1)

        # FISTA boosting: update t and the boosted dual variable s
        t_l = t
        t   = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        s   = z + (t_l - 1) / t * (z - z_l)

        # Primal estimate at the current iteration
        x = y - lambda_val * filter_boundary(n, d, z, 'transpose')

        # Cost function: data fidelity term + weighted TV regularization term
        J[k]  = np.sum((x - y) ** 2) + lambda_val * np.sum(np.abs(filter_boundary(n, d, x, 'normal')))

        # Effective noise estimate: RMS of the residual (y - x)
        nv[k] = np.sqrt(np.sum((x - y) ** 2) / N)

        # Optional per-iteration plot for debugging
        if p_fig:
            import matplotlib.pyplot as plt
            plt.plot(y, '-g', label="Original")
            plt.plot(x, '-r', label="Filtered")
            plt.pause(0.2)
            plt.clf()

        # Early stopping: if the cost function has not changed meaningfully
        # in the last two iterations, the solution has converged
        if k > 2 and abs(J[k - 1] - J[k - 2]) < stop_cri:
            break

        # Lambda update: if the effective noise estimate differs from the
        # target, adjust lambda to push the solution toward the target level
        # - if nv[k] < noise_estimate: lambda increases (sparsify more)
        # - if nv[k] > noise_estimate: lambda decreases (stay closer to data)
        if abs(nv[k] - noise_estimate) > precision:
            lambda_val = lambda_val * noise_estimate / nv[k]

        # Store the updated lambda value at this iteration
        Lambda[k] = lambda_val

    # Final primal estimate using the converged dual variable
    x = y - lambda_val * filter_boundary(n, d, z, 'transpose')

    return x, nv, J, Lambda