import numpy as np
from itertools import combinations

def cons_filter(root):
    """
    Constructs the filter based on given roots (zeros or poles).

    Parameters:
    - root: Array of zeros or poles.

    Returns:
    - fil: Array of filter coefficients.
    """
    n = len(root)
    fil = np.zeros(n + 1)
    fil[0] = 1.0

    for i in range(1, n + 1):
        fil[i] = (-1)**i * np.sum(np.exp([np.sum(c) for c in combinations(root, i)]))

    return fil