import numpy as np


def getIDXmat(dist_to_centroid, softClusterThres):
    """Best-effort Python translation of getIDXmat.m

    Given a matrix of distances from each item to each centroid, constructs
    a soft cluster assignment matrix based on a threshold.

    Parameters
    ----------
    dist_to_centroid : ndarray, shape (n_items, nClus)
        Distances of each frame/item to each cluster centroid.
    softClusterThres : float
        Soft-clustering threshold factor (xi in the MATLAB comments).

    Returns
    -------
    IDX_mat : ndarray, shape (n_items, nClus)
        Binary matrix where IDX_mat[i, k] == 1 if item i is assigned to
        cluster k at the given soft threshold.
    dist_thres : ndarray, shape (n_items, nClus)
        Distance ratios dist_to_centroid / min_dist_per_item.
    clos_to_centroid : ndarray, shape (n_items, nClus)
        Inverse of dist_thres (larger means closer / stronger assignment).
    """
    D = np.asarray(dist_to_centroid, dtype=float)
    n_items, nClus = D.shape

    # minimum distance per item
    min_dist = D.min(axis=1, keepdims=True)
    # avoid division by zero
    min_dist[min_dist == 0] = 1.0

    # ratio of distance to minimum (>=1)
    dist_thres = D / min_dist

    # soft assignment: clusters whose distance is within factor softClusterThres
    IDX_mat = dist_thres <= float(softClusterThres)

    # closeness measure (inverse distance ratio)
    clos_to_centroid = 1.0 / dist_thres

    return IDX_mat.astype(int), dist_thres, clos_to_centroid
