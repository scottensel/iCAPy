import numpy as np
from itertools import combinations


def cons_filter(root):
    """
    Constructs a filter given its zeros or its poles. The employed equations
    are derived in 'A Signal Processing Approach to Generalized 1-D Total
    Variation', p. 5267 (discrete versions of continuously defined filters).

    Inputs:
        root - scalar or array containing the zeros or poles for the filter;
               may be real or complex

    Outputs:
        fil - real-valued array of coefficients of the generated filter;
              imaginary parts cancel out by construction (symmetric roots)

    Initial version: 20.12.2010, Isik Karahanoglu
    """
    # Ensure root is always a 1D array so len() and combinations() work
    # correctly even when a single scalar pole/zero is passed
    root = np.atleast_1d(np.asarray(root))

    n = len(root)

    # Initialise filter coefficient array as complex to avoid silent casting;
    # first coefficient is always 1
    fil = np.zeros(n + 1, dtype=complex)
    fil[0] = 1.0

    # Build each subsequent coefficient from all size-i combinations of roots
    for i in range(1, n + 1):
        fil[i] = (-1) ** i * np.sum(
            np.exp([np.sum(c) for c in combinations(root, i)])
        )

    # Imaginary parts cancel by construction for conjugate-symmetric roots;
    # take real part and warn if any significant imaginary residual remains
    imag_max = np.max(np.abs(fil.imag))
    if imag_max > 1e-10:
        import warnings
        warnings.warn(
            f"cons_filter: imaginary residual {imag_max:.2e} — "
            "roots may not be conjugate-symmetric"
        )

    return fil.real