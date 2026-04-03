import numpy as np


def MyProx(x, Op, Adj_Op, evaluate_norm, param, use_numba=False):
    """
    Computes the following TV regularization:
        F(x) = min ||y - x||^2 + lambda * ||Op{x}||_1
    where Op is the gradient operator, using the 3D extension of FISTA:

        Beck, A., & Teboulle, M. (2009). A fast iterative
        shrinkage-thresholding algorithm for linear inverse problems.
        SIAM journal on imaging sciences, 2(1), 183-202.

    All large working arrays are allocated once before the loop and
    reused in-place at every iteration. Supports numpy, Numba JIT, and
    CuPy GPU execution — the path is selected automatically based on
    the type of x and the use_numba flag.

    Inputs:
        x             - 3D or 4D array (numpy or cupy)
        Op            - gradient operator: dx, dy, dz = Op(x, dx, dy, dz)
        Adj_Op        - adjoint (neg divergence): out = Adj_Op(u, v, w)
        evaluate_norm - TV norm: total = evaluate_norm(x, grad_bufs)
        param         - dict with 'Lip', 'tol', 'LambdaSpat', 'NitSpat'
        use_numba     - bool, passed through to Op and evaluate_norm

    Outputs:
        x_out - denoised output array, same shape and type as x

    Implemented by Younes Farouj, 10.03.2016
    """
    # Detect CuPy — all allocations use the same module as x
    try:
        import cupy as cp
        xp = cp if isinstance(x, cp.ndarray) else np
    except ImportError:
        xp = np

    L          = param['Lip']
    tol        = param['tol']
    lambda_val = param['LambdaSpat']
    max_iter   = param['NitSpat']
    step       = 1.0 / (L * lambda_val)   # pre-computed constant step size

    # Allocate all working arrays once — reused every iteration to avoid
    # thousands of large array allocations inside the hot loop
    u     = xp.zeros_like(x)
    v     = xp.zeros_like(x)
    w     = xp.zeros_like(x)
    u_old = xp.zeros_like(x)
    v_old = xp.zeros_like(x)
    w_old = xp.zeros_like(x)
    u_tmp = xp.zeros_like(x)
    v_tmp = xp.zeros_like(x)
    w_tmp = xp.zeros_like(x)

    x_out          = xp.zeros_like(x)
    proj_amplitude = xp.zeros_like(x)
    dx_buf         = xp.zeros_like(x)
    dy_buf         = xp.zeros_like(x)
    dz_buf         = xp.zeros_like(x)

    NRJ_old  = 0.0
    residual = tol + 1.0
    t_old    = 1.0

    for k in range(max_iter):

        # Check convergence
        if residual < tol:
            break

        # Backward step: x_out = x + lambda * Adj_Op(u, v, w)
        adj = Adj_Op(u, v, w)
        xp.multiply(lambda_val, adj, out=x_out)
        x_out += x

        # Energy: E = lambda * TV(x_out) + 0.5 * ||x - x_out||^2
        NRJ      = (lambda_val * evaluate_norm(x_out, (dx_buf, dy_buf, dz_buf),
                                               use_numba=use_numba)
                    + 0.5 * float(xp.dot((x - x_out).ravel(), (x - x_out).ravel())))
        residual = abs(NRJ - NRJ_old) / (NRJ if NRJ != 0.0 else 1.0)
        NRJ_old  = NRJ

        # Gradient step on dual variables, using pre-allocated buffers
        Op(x_out, dx_buf, dy_buf, dz_buf)
        u -= step * dx_buf
        v -= step * dy_buf
        w -= step * dz_buf

        # Soft thresholding: project onto unit ball, computed in-place
        xp.multiply(u, u, out=proj_amplitude)
        proj_amplitude += v * v
        proj_amplitude += w * w
        xp.sqrt(proj_amplitude, out=proj_amplitude)
        xp.maximum(1.0, proj_amplitude, out=proj_amplitude)

        xp.divide(u, proj_amplitude, out=u_tmp)
        xp.divide(v, proj_amplitude, out=v_tmp)
        xp.divide(w, proj_amplitude, out=w_tmp)

        # FISTA acceleration
        t     = (1.0 + xp.sqrt(1.0 + 4.0 * t_old ** 2)) / 2.0
        accel = (t_old - 1.0) / t

        xp.subtract(u_tmp, u_old, out=u); u *= accel; u += u_tmp
        xp.subtract(v_tmp, v_old, out=v); v *= accel; v += v_tmp
        xp.subtract(w_tmp, w_old, out=w); w *= accel; w += w_tmp

        # Store for next FISTA step
        xp.copyto(u_old, u_tmp)
        xp.copyto(v_old, v_tmp)
        xp.copyto(w_old, w_tmp)
        t_old = float(t)

    return x_out
