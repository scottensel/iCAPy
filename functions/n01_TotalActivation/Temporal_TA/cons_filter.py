import numpy as np
from itertools import combinations


def cons_filter(root):
    """
    Constructs a filter given its zeros or its poles. The employed equations
    are derived in 'A Signal Processing Approach to Generalized 1-D Total
    Variation', p. 5267 (discrete versions of continuously defined filters).

    Inputs:
        root - array containing the zeros or the poles for the filter

    Outputs:
        fil - array of coefficients of the generated filter

    Initial version: 20.12.2010, Isik Karahanoglu
    """
    n = len(root)

    # Initialise filter coefficient array; first coefficient is always 1
    fil = np.zeros(n + 1)
    fil[0] = 1.0

    # Build each subsequent coefficient from all size-i combinations of roots
    for i in range(1, n + 1):
        fil[i] = (-1) ** i * np.sum(np.exp([np.sum(c) for c in combinations(root, i)]))

    return fil