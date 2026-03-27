import numpy as np


def Build_Connectivity_Matrix(IDX, tmp_ss, type, n_items):
    """
    Constructs a connectivity matrix from cluster assignment indices.

    Parameters
    ----------
    IDX : array_like, shape (n_subsampled_items,)
        Cluster labels for the sampled items (0-based, 0..K-1).
    tmp_ss : array_like of int
        Indices (0-based) of the sampled items in the full set [0..n_items-1].
    type : {'items', 'subjects'}
        Subsampling type (treated identically, as in the MATLAB code).
    n_items : int
        Total number of items.

    Returns
    -------
    M : ndarray, shape (n_items, n_items)
        Connectivity matrix where M[i, j] = 1 if items i and j are in the
        same cluster; 0 otherwise (including unsampled items).
    """
    IDX    = np.asarray(IDX).astype(int)
    tmp_ss = np.asarray(tmp_ss).astype(int)
    n_items = int(n_items)

    # Use -1 as the unassigned sentinel so that cluster 0 is not skipped.
    # MATLAB used 0 as sentinel because its labels start at 1; here labels
    # start at 0 so we need a value outside the valid label range.
    IDX_full = np.full(n_items, -1, dtype=int)

    if type in ("items", "subjects"):
        IDX_full[tmp_ss] = IDX

    M = np.zeros((n_items, n_items), dtype=float)

    for i in range(n_items):
        if IDX_full[i] >= 0:               # -1 means unassigned
            j = IDX_full == IDX_full[i]
            M[i, j] = 1.0

    return M