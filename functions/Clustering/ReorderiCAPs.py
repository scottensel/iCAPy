import numpy as np
from .ZScore_iCAPs import zscore_icaps


def reorder_icaps(IDX, iCAPs, icaps_folds=None):
    """
    Python translation of ReorderiCAPs.m, but with Python-native 0-based labels.

    Reorders clusters according to the number of frames per cluster.

    Parameters
    ----------
    IDX : array_like, shape (n_frames,)
        Cluster labels (0..nClus-1).
    iCAPs : ndarray, shape (nClus, n_vox)
        iCAP spatial maps.
    icaps_folds : dict or list or None
        - If dict: contains per-fold clustering results with keys:
            'iCAPs', 'IDX', 'sum_dist', 'dist_to_centroid', 'iCAPs_z'
        - If [] (list): indicates replicate data not saved (MATLAB-style iCAPs_folds = [])
        - If None: no fold info.

    Returns
    -------
    IDX2 : ndarray, shape (n_frames,)
        Reordered cluster labels (0..nClus-1).
    iCAPs2 : ndarray, shape (nClus, n_vox)
        Reordered iCAP maps.
    icaps_folds2 : dict or list or None
        Reordered folds structure if dict provided; otherwise returns original icaps_folds.
    """
    IDX = np.asarray(IDX).astype(int)
    iCAPs = np.asarray(iCAPs)

    # number of clusters (safer from iCAPs than from IDX.max() when some clusters are empty)
    n_clusters = int(iCAPs.shape[0])

    # count frames per cluster
    n_frames = np.zeros(n_clusters, dtype=int)
    for iC in range(n_clusters):
        n_frames[iC] = int(np.sum(IDX == iC))

    # sort clusters by decreasing size
    isort = np.argsort(-n_frames)

    # relabel map: old_label -> new_label (both 0-based)
    relabel = np.empty(n_clusters, dtype=int)
    for new_label, old_label in enumerate(isort):
        relabel[old_label] = new_label

    # reorder labels and maps
    IDX2 = relabel[IDX]
    iCAPs2 = iCAPs[isort, :]

    # If folds are not a dict (None or []), do nothing further (MATLAB iCAPs_folds = [])
    if not isinstance(icaps_folds, dict):
        return IDX2, iCAPs2, icaps_folds

    # If dict is present but no fold data, do nothing further
    if ("iCAPs" not in icaps_folds) or (len(icaps_folds["iCAPs"]) == 0):
        return IDX2, iCAPs2, icaps_folds

    # reorder fold structures
    icaps_folds2 = {
        "iCAPs": [],
        "IDX": [],
        "sum_dist": [],
        "dist_to_centroid": [],
        "iCAPs_z": [],
    }

    n_folds = len(icaps_folds["iCAPs"])
    for iFold in range(n_folds):
        iCAPs_fold = np.asarray(icaps_folds["iCAPs"][iFold])
        IDX_fold = np.asarray(icaps_folds["IDX"][iFold]).astype(int)
        sum_dist_fold = np.asarray(icaps_folds["sum_dist"][iFold])
        dist_to_centroid_fold = np.asarray(icaps_folds["dist_to_centroid"][iFold])
        iCAPs_z_fold = np.asarray(icaps_folds["iCAPs_z"][iFold])

        # relabel fold IDX (0-based)
        IDX_new = relabel[IDX_fold]

        # reorder per-cluster fold outputs
        iCAPs_new = iCAPs_fold[isort, :]
        iCAPs_z_new = iCAPs_z_fold[isort, :]
        sum_dist_new = sum_dist_fold[isort]
        dist_to_centroid_new = dist_to_centroid_fold[:, isort]

        icaps_folds2["iCAPs"].append(iCAPs_new)
        icaps_folds2["IDX"].append(IDX_new)
        icaps_folds2["sum_dist"].append(sum_dist_new)
        icaps_folds2["dist_to_centroid"].append(dist_to_centroid_new)
        icaps_folds2["iCAPs_z"].append(iCAPs_z_new)

    return IDX2, iCAPs2, icaps_folds2