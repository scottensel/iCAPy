import numpy as np
from .ZScore_iCAPs import ZScore_iCAPs


def ReorderiCAPs(IDX, iCAPs, iCAPs_folds=None):
    """Python translation of ReorderiCAPs.m

    Reorders clusters according to the number of frames per cluster.

    Parameters
    ----------
    IDX : array_like, shape (n_frames,)
        Cluster labels (1..nClus).
    iCAPs : ndarray, shape (nClus, n_vox)
        iCAP spatial maps.
    iCAPs_folds : dict or None
        Optional structure containing per-fold clustering results:
        keys: 'iCAPs', 'IDX', 'sum_dist', 'dist_to_centroid', 'iCAPs_z'.

    Returns
    -------
    IDX2 : ndarray, shape (n_frames,)
        Reordered cluster labels.
    iCAPs2 : ndarray, shape (nClus, n_vox)
        Reordered iCAP maps.
    iCAPs_folds2 : dict or None
        Reordered folds structure (if provided), else None.
    """
    IDX = np.asarray(IDX).astype(int)
    iCAPs = np.asarray(iCAPs)
    n_clusters = int(IDX.max())

    n_frames = np.zeros(n_clusters, dtype=int)
    for iC in range(1, n_clusters + 1):
        n_frames[iC - 1] = np.sum(IDX == iC)

    # sort clusters by decreasing size
    isort = np.argsort(-n_frames)  # descending
    # new labels: 1..n_clusters in sorted order
    IDX2 = np.zeros_like(IDX)
    for new_label, old_idx in enumerate(isort, start=1):
        IDX2[IDX == (old_idx + 1)] = new_label

    # reorder maps
    iCAPs2 = iCAPs[isort, :]

    iCAPs_folds2 = None
    if iCAPs_folds is not None and len(iCAPs_folds.get('iCAPs', [])) > 0:
        iCAPs_folds2 = {
            'iCAPs': [],
            'IDX': [],
            'sum_dist': [],
            'dist_to_centroid': [],
            'iCAPs_z': [],
        }
        n_folds = len(iCAPs_folds['iCAPs'])
        for iFold in range(n_folds):
            iCAPs_fold = np.asarray(iCAPs_folds['iCAPs'][iFold])
            IDX_fold = np.asarray(iCAPs_folds['IDX'][iFold]).astype(int)
            sum_dist_fold = np.asarray(iCAPs_folds['sum_dist'][iFold])
            dist_to_centroid_fold = np.asarray(iCAPs_folds['dist_to_centroid'][iFold])
            iCAPs_z_fold = np.asarray(iCAPs_folds['iCAPs_z'][iFold])

            IDX_new = np.zeros_like(IDX_fold)
            for new_label, old_idx in enumerate(isort, start=1):
                IDX_new[IDX_fold == (old_idx + 1)] = new_label

            iCAPs_new = iCAPs_fold[isort, :]
            iCAPs_z_new = iCAPs_z_fold[isort, :]
            sum_dist_new = sum_dist_fold[isort]
            dist_to_centroid_new = dist_to_centroid_fold[:, isort]

            iCAPs_folds2['iCAPs'].append(iCAPs_new)
            iCAPs_folds2['IDX'].append(IDX_new)
            iCAPs_folds2['sum_dist'].append(sum_dist_new)
            iCAPs_folds2['dist_to_centroid'].append(dist_to_centroid_new)
            iCAPs_folds2['iCAPs_z'].append(iCAPs_z_new)
    return IDX2, iCAPs2, iCAPs_folds2
