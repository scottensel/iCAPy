import numpy as np

# ── Numba JIT variant ─────────────────────────────────────────────────────
# Compiled once on first call, cached for all subsequent calls.
# prange(T) parallelises over time points — each is independent.
try:
    from numba import njit, prange as _prange

    @njit(parallel=True, cache=True)
    def _gradient3D_numba(V, dx, dy, dz):
        """
        Numba-compiled 4D gradient (X x Y x Z x T).
        Parallelises over the T dimension; each time point is independent.
        """
        X = V.shape[0]
        Y = V.shape[1]
        Z = V.shape[2]
        T = V.shape[3]

        for t in _prange(T):
            # X gradient
            for x in range(X - 1):
                for y in range(Y):
                    for z in range(Z):
                        dx[x, y, z, t] = V[x + 1, y, z, t] - V[x, y, z, t]
            for y in range(Y):
                for z in range(Z):
                    dx[X - 1, y, z, t] = 0.0

            # Y gradient
            for x in range(X):
                for y in range(Y - 1):
                    for z in range(Z):
                        dy[x, y, z, t] = V[x, y + 1, z, t] - V[x, y, z, t]
            for x in range(X):
                for z in range(Z):
                    dy[x, Y - 1, z, t] = 0.0

            # Z gradient
            for x in range(X):
                for y in range(Y):
                    for z in range(Z - 1):
                        dz[x, y, z, t] = V[x, y, z + 1, t] - V[x, y, z, t]
            for x in range(X):
                for y in range(Y):
                    dz[x, y, Z - 1, t] = 0.0

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


def gradient3D_full(V, wx=None, wy=None, wz=None, out=None, use_numba=False):
    """
    Computes the 3D gradient of a 3D or 4D volume along each axis.
    Particularly suitable for fMRI data where the 4th dimension refers
    to time; a 3D algorithm is applied to the first 3 components.

    Three execution paths are available, selected by the caller:
        use_numba=True  — Numba JIT with prange over T (fastest on CPU)
        GPU array input — CuPy operations used automatically if V is a
                          CuPy array (caller sets this via xp_backend)
        default         — in-place NumPy (no np.pad, no extra allocations)

    Inputs:
        V          - 3D or 4D array (numpy or cupy); if 4D, gradients are
                     computed along the first three spatial dimensions only
        wx, wy, wz - optional weight matrices along X, Y and Z directions
        out        - optional tuple (dx, dy, dz) of pre-allocated arrays
                     with the same shape as V; results written in-place
        use_numba  - bool, if True and Numba is installed, use the JIT
                     path; ignored for GPU arrays

    Outputs:
        dx, dy, dz - gradient arrays along X, Y and Z axes respectively;
                     zero-padded at the end of each dimension

    Implemented by Younes Farouj, 12.03.2016
    Weight option added by Younes Farouj, 28.04.2016
    """
    # Detect CuPy arrays — use the same module for all operations
    try:
        import cupy as cp
        xp = cp if isinstance(V, cp.ndarray) else np
    except ImportError:
        xp = np

    V = xp.asarray(V)

    # Numba path: only for 4D numpy float64 arrays
    if (use_numba and _NUMBA_AVAILABLE
            and xp is np and V.ndim == 4):
        V64 = np.ascontiguousarray(V, dtype=np.float64)
        if out is not None:
            dx, dy, dz = out
        else:
            dx = np.empty_like(V64)
            dy = np.empty_like(V64)
            dz = np.empty_like(V64)
        _gradient3D_numba(V64, dx, dy, dz)
        if wx is not None: dx *= xp.asarray(wx)
        if wy is not None: dy *= xp.asarray(wy)
        if wz is not None: dz *= xp.asarray(wz)
        return dx, dy, dz

    # NumPy / CuPy in-place path (no np.pad)
    if out is not None:
        dx, dy, dz = out
    else:
        dx = xp.empty_like(V)
        dy = xp.empty_like(V)
        dz = xp.empty_like(V)

    if V.ndim == 3:
        dx[:-1, :, :] = V[1:, :, :] - V[:-1, :, :]
        dx[-1,  :, :] = 0.0
        dy[:, :-1, :] = V[:, 1:, :] - V[:, :-1, :]
        dy[:, -1,  :] = 0.0
        dz[:, :, :-1] = V[:, :, 1:] - V[:, :, :-1]
        dz[:, :, -1 ] = 0.0

    elif V.ndim == 4:
        dx[:-1, :, :, :] = V[1:, :, :, :] - V[:-1, :, :, :]
        dx[-1,  :, :, :] = 0.0
        dy[:, :-1, :, :] = V[:, 1:, :, :] - V[:, :-1, :, :]
        dy[:, -1,  :, :] = 0.0
        dz[:, :, :-1, :] = V[:, :, 1:, :] - V[:, :, :-1, :]
        dz[:, :, -1,  :] = 0.0

    else:
        raise ValueError("Input array V must be 3D or 4D.")

    if wx is not None: dx *= xp.asarray(wx)
    if wy is not None: dy *= xp.asarray(wy)
    if wz is not None: dz *= xp.asarray(wz)

    return dx, dy, dz
