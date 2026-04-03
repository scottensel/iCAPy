"""
xp_backend.py
─────────────
Returns the array namespace (numpy, cupy, or numpy+numba) to use for
spatial TA operations, based on the flags set in param.

Usage
-----
    from functions.n01_TotalActivation.Spatial_TA.xp_backend import get_backend
    xp, USE_NUMBA = get_backend(param)

    # then use xp.zeros, xp.sqrt, etc. throughout the spatial functions
    # USE_NUMBA tells MyProx / gradient / div whether to call JIT variants
"""

_GPU_WARNED = False   # print the fallback warning only once per run


def get_backend(param):
    """
    Selects the array backend based on param flags.

    Priority:  use_gpu=1  >  use_numba=1  >  plain numpy

    Parameters
    ----------
    param : dict
        Must contain 'use_gpu' and 'use_numba' keys (both default to 0).

    Returns
    -------
    xp : module
        Either cupy or numpy.
    use_numba : bool
        True if Numba JIT variants should be used (only when xp is numpy
        and param['use_numba']==1).
    """
    global _GPU_WARNED

    use_gpu   = bool(param.get('use_gpu',   0))
    use_numba = bool(param.get('use_numba', 0))

    if use_gpu:
        try:
            import cupy as cp
            return cp, False   # GPU takes precedence; numba not used
        except ImportError:
            if not _GPU_WARNED:
                print(
                    "[iCAPy WARNING] use_gpu=1 but CuPy is not installed. "
                    "Falling back to NumPy. "
                    "Install CuPy with:  pip install cupy-cuda12x"
                )
                _GPU_WARNED = True
            # fall through to numba / numpy

    if use_numba:
        try:
            import numba  # noqa: F401 — just check it is importable
            return __import__('numpy'), True
        except ImportError:
            print(
                "[iCAPy WARNING] use_numba=1 but Numba is not installed. "
                "Falling back to NumPy. "
                "Install Numba with:  pip install numba"
            )

    return __import__('numpy'), False
