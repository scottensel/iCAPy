import numpy as np


def MyProx(x, Op, Adj_Op, evaluate_norm, param):
    """
    Computes the following TV regularization:
        F(x) = min ||y - x||^2 + lambda * ||Op{x}||_1
    where Op is the gradient operator, using the 3D extension of FISTA:

        Beck, A., & Teboulle, M. (2009). A fast iterative
        shrinkage-thresholding algorithm for linear inverse problems.
        SIAM journal on imaging sciences, 2(1), 183-202.

    Inputs:
        x            - 3D or 4D array; if 4D, a 3D algorithm is applied
                       to the first 3 components in parallel (time is 4th)
        Op           - function handle, gradient operator:
                         dx, dy, dz = Op(x)
        Adj_Op       - function handle, adjoint of the gradient operator
                       (negative divergence):
                         out = Adj_Op(u, v, w)
        evaluate_norm - function handle, evaluates the TV norm of a volume:
                         total = evaluate_norm(x)
        param        - dict containing required spatial TA parameters:
            'Lip'        - float, Lipschitz constant of the gradient
                           operator (used as step size denominator)
            'tol'        - float, convergence tolerance; iterations stop
                           when the relative change in energy falls below
                           this threshold
            'LambdaSpat' - float, regularization coefficient for spatial
                           TV regularization
            'NitSpat'    - int, maximum number of FISTA iterations

    Outputs:
        x_out - denoised output array, same shape as x

    Implemented by Younes Farouj, 10.03.2016
    """
    L          = param['Lip']
    tol        = param['tol']
    lambda_val = param['LambdaSpat']
    max_iter   = param['NitSpat']

    # Initialise dual variables u, v, w at zero (same shape as gradient output)
    u, v, w         = Op(np.zeros_like(x))
    u_old, v_old, w_old = u.copy(), v.copy(), w.copy()

    # Initialise energy and convergence tracking
    NRJ_old  = 0
    residual = tol + 1   # ensure we enter the loop
    t_old    = 1

    for k in range(max_iter):

        # Check convergence: stop if relative energy change is below tolerance
        if residual < tol:
            break

        # Backward step: update the primal variable x_out using the adjoint
        # of the gradient operator applied to the current dual variables
        x_out = x + lambda_val * Adj_Op(u, v, w)

        # Evaluate the functional being minimized:
        # E = lambda * TV(x_out) + 0.5 * ||x - x_out||^2
        NRJ      = lambda_val * evaluate_norm(x_out) + 0.5 * np.linalg.norm(x - x_out) ** 2
        residual = abs(NRJ - NRJ_old) / NRJ
        NRJ_old  = NRJ

        # Gradient step on the dual variables (proximal operator argument)
        dx, dy, dz = Op(x_out)
        u -= (1 / (L * lambda_val)) * dx
        v -= (1 / (L * lambda_val)) * dy
        w -= (1 / (L * lambda_val)) * dz

        # Soft thresholding (projection onto the unit ball in the dual space):
        # clamp the amplitude of the dual variable vector to at most 1
        proj_amplitude  = np.maximum(1, np.sqrt(u ** 2 + v ** 2 + w ** 2))
        u_temp = u / proj_amplitude
        v_temp = v / proj_amplitude
        w_temp = w / proj_amplitude

        # FISTA acceleration: update the momentum variable t and compute
        # the over-relaxed (boosted) dual variables
        t    = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        u    = u_temp + (t_old - 1) / t * (u_temp - u_old)
        v    = v_temp + (t_old - 1) / t * (v_temp - v_old)
        w    = w_temp + (t_old - 1) / t * (w_temp - w_old)

        # Store current values for the next FISTA boosting step
        u_old, v_old, w_old, t_old = u_temp, v_temp, w_temp, t

    return x_out
