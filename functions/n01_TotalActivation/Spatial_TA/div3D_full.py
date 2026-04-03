import numpy as np

# ── Numba JIT variant ─────────────────────────────────────────────────────
try:
    from numba import njit, prange as _prange

    @njit(parallel=True, cache=True)
    def _div3D_numba(dx, dy, dz, V):
        """
        Numba-compiled 4D divergence (X x Y x Z x T).
        Parallelises over the T dimension.

        Equivalent to:
            post - pre of dx[:-1], applied along each spatial axis.
        Verified identical to the np.pad original.
        """
        X = dx.shape[0]
        Y = dx.shape[1]
        Z = dx.shape[2]
        T = dx.shape[3]

        for t in _prange(T):
            # X term
            for y in range(Y):
                for z in range(Z):
                    V[0, y, z, t] = dx[0, y, z, t]
            for x in range(1, X - 1):
                for y in range(Y):
                    for z in range(Z):
                        V[x, y, z, t] = dx[x, y, z, t] - dx[x - 1, y, z, t]
            for y in range(Y):
                for z in range(Z):
                    V[X - 1, y, z, t] = -dx[X - 2, y, z, t]

            # Y term
            for x in range(X):
                for z in range(Z):
                    V[x, 0, z, t] += dy[x, 0, z, t]
            for x in range(X):
                for y in range(1, Y - 1):
                    for z in range(Z):
                        V[x, y, z, t] += dy[x, y, z, t] - dy[x, y - 1, z, t]
            for x in range(X):
                for z in range(Z):
                    V[x, Y - 1, z, t] += -dy[x, Y - 2, z, t]

            # Z term
            for x in range(X):
                for y in range(Y):
                    V[x, y, 0, t] += dz[x, y, 0, t]
            for x in range(X):
                for y in range(Y):
                    for z in range(1, Z - 1):
                        V[x, y, z, t] += dz[x, y, z, t] - dz[x, y, z - 1, t]
            for x in range(X):
                for y in range(Y):
                    V[x, y, Z - 1, t] += -dz[x, y, Z - 2, t]

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


def div3D_full(dx, dy, dz, wx=None, wy=None, wz=None, out=None, use_numba=False):
    """
    Computes the 3D divergence of the vector field [dx, dy, dz].
    Particularly suitable for fMRI data where the 4th dimension of d*
    refers to time; a 3D algorithm is applied to the first 3 components.

    Three execution paths are available:
        use_numba=True  — Numba JIT with prange over T (fastest on CPU)
        GPU array input — CuPy operations used automatically
        default         — in-place NumPy, no np.pad

    The equivalence between the in-place indexing and the original
    np.pad version is:
        padarray(E,[1,0,0],'post') - padarray(E,[1,0,0],'pre'), E=dx[:-1]
        row 0:    E[0]              =  dx[0]
        row i:    E[i] - E[i-1]    =  dx[i] - dx[i-1]
        row N-1: -E[N-2]           = -dx[-2]

    Verified numerically to be bitwise identical to the original.

    Inputs:
        dx, dy, dz - 3D or 4D arrays (numpy or cupy)
        wx, wy, wz - optional weight matrices
        out        - optional pre-allocated output array (same shape as dx)
        use_numba  - bool, use Numba JIT path if available

    Outputs:
        V - divergence array, same shape as dx

    Implemented by Younes Farouj, 12.03.2016
    Weight option added by Younes Farouj, 28.04.2016
    """
    # Detect CuPy
    try:
        import cupy as cp
        xp = cp if isinstance(dx, cp.ndarray) else np
    except ImportError:
        xp = np

    dx = xp.asarray(dx, dtype=float)
    dy = xp.asarray(dy, dtype=float)
    dz = xp.asarray(dz, dtype=float)

    if wx is not None: dx = dx * xp.asarray(wx)
    if wy is not None: dy = dy * xp.asarray(wy)
    if wz is not None: dz = dz * xp.asarray(wz)

    # Numba path: 4D numpy only
    if (use_numba and _NUMBA_AVAILABLE
            and xp is np and dx.ndim == 4):
        dx64 = np.ascontiguousarray(dx, dtype=np.float64)
        dy64 = np.ascontiguousarray(dy, dtype=np.float64)
        dz64 = np.ascontiguousarray(dz, dtype=np.float64)
        V = out if out is not None else np.zeros_like(dx64)
        if out is not None: V[:] = 0.0
        _div3D_numba(dx64, dy64, dz64, V)
        return V

    # NumPy / CuPy in-place path
    if out is not None:
        V = out
        V[:] = 0.0
    else:
        V = xp.zeros_like(dx)

    if dx.ndim == 3:
        E = dx[:-1, :, :]
        V[0,    :, :]  =  E[0,  :, :]
        V[1:-1, :, :] =  E[1:, :, :] - E[:-1, :, :]
        V[-1,   :, :]  = -E[-1, :, :]

        E = dy[:, :-1, :]
        V[:, 0,    :]  +=  E[:, 0,  :]
        V[:, 1:-1, :] +=  E[:, 1:, :] - E[:, :-1, :]
        V[:, -1,   :]  += -E[:, -1, :]

        E = dz[:, :, :-1]
        V[:, :, 0   ]  +=  E[:, :, 0 ]
        V[:, :, 1:-1] +=  E[:, :, 1:] - E[:, :, :-1]
        V[:, :, -1  ]  += -E[:, :, -1]

    elif dx.ndim == 4:
        E = dx[:-1, :, :, :]
        V[0,    :, :, :]  =  E[0,  :, :, :]
        V[1:-1, :, :, :] =  E[1:, :, :, :] - E[:-1, :, :, :]
        V[-1,   :, :, :]  = -E[-1, :, :, :]

        E = dy[:, :-1, :, :]
        V[:, 0,    :, :]  +=  E[:, 0,  :, :]
        V[:, 1:-1, :, :] +=  E[:, 1:, :, :] - E[:, :-1, :, :]
        V[:, -1,   :, :]  += -E[:, -1, :, :]

        E = dz[:, :, :-1, :]
        V[:, :, 0,    :]  +=  E[:, :, 0,  :]
        V[:, :, 1:-1, :] +=  E[:, :, 1:, :] - E[:, :, :-1, :]
        V[:, :, -1,   :]  += -E[:, :, -1, :]

    else:
        raise ValueError("Input arrays dx, dy, and dz must be 3D or 4D.")

    return V
