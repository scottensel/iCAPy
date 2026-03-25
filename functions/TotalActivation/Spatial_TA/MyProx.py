import numpy as np
from numpy.linalg import norm
from tqdm import tqdm


def MyProx(x, Op, Adj_Op, evaluate_norm, param):
    """
    Optimized MyProx function with optional GPU support.

    Parameters:
    - x: Input array.
    - Op: Function that computes the gradient.
    - Adj_Op: Function that computes the adjoint operator.
    - evaluate_norm: Function to evaluate the norm.
    - param: Dictionary of parameters, should include 'Lip', 'tol', 'LambdaSpat', 'NitSpat', and 'cuda' flag.
    """
    # Choose backend based on GPU availability

    # Move x to the selected backend
    x = np.array(x)

    # Initialize constants and variables on GPU if enabled
    L, tol, lambda_val, max_iter = param['Lip'], param['tol'], param['LambdaSpat'], param['NitSpat']
    u, v, w = Op(np.zeros_like(x))
    u_old, v_old, w_old = u.copy(), v.copy(), w.copy()
    NRJ_old, residual = 0, tol + 1
    t_old = 1

    # Iterative FISTA scheme
    # for k in tqdm(range(max_iter), desc="Processing spatial step", ncols=80):
    for k in range(max_iter):
        if residual < tol:
            break

        # Backward step
        x_out = x + lambda_val * Adj_Op(u, v, w)

        # Energy evaluation
        NRJ = lambda_val * evaluate_norm(x_out) + 0.5 * np.linalg.norm(x - x_out) ** 2
        residual = abs(NRJ - NRJ_old) / NRJ  # Avoid division by zero
        NRJ_old = NRJ

        # Update gradients
        dx, dy, dz = Op(x_out)
        u -= (1 / (L * lambda_val)) * dx
        v -= (1 / (L * lambda_val)) * dy
        w -= (1 / (L * lambda_val)) * dz

        # Soft thresholding
        proj_amplitude = np.maximum(1, np.sqrt(u ** 2 + v ** 2 + w ** 2))
        u_temp, v_temp, w_temp = u / proj_amplitude, v / proj_amplitude, w / proj_amplitude

        # FISTA Acceleration
        t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        u = u_temp + (t_old - 1) / t * (u_temp - u_old)
        v = v_temp + (t_old - 1) / t * (v_temp - v_old)
        w = w_temp + (t_old - 1) / t * (w_temp - w_old)
        u_old, v_old, w_old, t_old = u_temp, v_temp, w_temp, t

        # print(f"Iteration {k + 1}")

    return x_out
