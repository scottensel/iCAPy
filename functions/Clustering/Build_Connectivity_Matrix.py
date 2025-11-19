import numpy as np


def Build_Connectivity_Matrix(IDX, tmp_ss, type, n_items):
    """Python translation of Build_Connectivity_Matrix.m

    Constructs a connectivity matrix from cluster assignment indices.

    Parameters
    ----------
    IDX : array_like, shape (n_subsampled_items,)
        Cluster labels for the sampled items.
    tmp_ss : array_like of int
        Indices (0-based) of the sampled items in the full set [0..n_items-1].
    type : {'items', 'subjects'}
        Subsampling type (currently treated identically, as in the MATLAB code).
    n_items : int
        Total number of items.

    Returns
    -------
    M : ndarray, shape (n_items, n_items)
        Connectivity matrix where M[i, j] = 1 if items i and j are in the
        same nonzero cluster; 0 otherwise.
    """
    IDX = np.asarray(IDX).astype(int)
    tmp_ss = np.asarray(tmp_ss).astype(int)
    n_items = int(n_items)

    IDX_full = np.zeros(n_items, dtype=int)

    if type in ("items", "subjects"):
        IDX_full[tmp_ss] = IDX
    else:
        # 'dims' case not implemented in the MATLAB code either
        IDX_full[tmp_ss] = IDX

    M = np.zeros((n_items, n_items), dtype=float)

    for i in range(len(IDX_full)):
        if IDX_full[i] > 0:
            j = IDX_full == IDX_full[i]
            M[i, j] = 1.0

    return M
