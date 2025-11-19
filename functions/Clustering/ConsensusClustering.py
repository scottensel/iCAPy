import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from .Build_Connectivity_Matrix import Build_Connectivity_Matrix
from .ComputeClusteringQuality import ComputeClusteringQuality
from functions.Utilities.WriteInformation import write_information


def ConsensusClustering(X, K_range, Subsample_type, Subsample_fraction,
                        n_folds, DistType, subject_labels, outDir_cons, fid=None):
    """Python approximation of ConsensusClustering.m

    Runs consensus clustering for a range of cluster numbers K using k-means.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_dims)
        Data matrix (frames x voxels or similar).
    K_range : array_like
        Range of K values to test.
    Subsample_type : {'items', 'subjects'}
        Subsampling type.
    Subsample_fraction : float in (0,1]
        Fraction of items/subjects to keep in each fold.
    n_folds : int
        Number of folds.
    DistType : {'sqeuclidean', 'cosine'}
        Distance type for k-means and inter-fold matching.
    subject_labels : ndarray, shape (n_items,)
        Subject index for each item; used when Subsample_type == 'subjects'.
    outDir_cons : str
        Output directory for consensus results.
    fid : file handle or str or None
        For logging.

    Returns
    -------
    CDF : ndarray, shape (len(K_range), 101)
    AUC : ndarray, shape (len(K_range),)
        Quality indices from ComputeClusteringQuality.
    """
    os.makedirs(outDir_cons, exist_ok=True)
    X = np.asarray(X)
    K_range = np.asarray(K_range).astype(int)
    n_items, n_dims = X.shape

    if fid is not None:
        write_information(fid, f"n_items {n_items}, n_dims {n_dims}")

    Consensus_all = []

    for k_idx, K in enumerate(K_range):
        if fid is not None:
            write_information(fid, f"Running consensus clustering for K = {K}...")

        # Connectivity matrices for each fold
        M_all = np.zeros((n_items, n_items, n_folds), dtype=float)

        for iFold in range(n_folds):
            # Subsampling
            if Subsample_type == "subjects":
                subj_list = np.unique(subject_labels)
                n_subjects = subj_list.size
                n_subjects_ss = int(np.floor(Subsample_fraction * n_subjects))
                subsampled_subj = np.random.choice(subj_list, size=n_subjects_ss, replace=False)
                tmp_ss = np.isin(subject_labels, subsampled_subj)
            else:
                n_items_ss = int(np.floor(Subsample_fraction * n_items))
                tmp_idx = np.random.choice(n_items, size=n_items_ss, replace=False)
                tmp_ss = np.zeros(n_items, dtype=bool)
                tmp_ss[tmp_idx] = True

            X_ss = X[tmp_ss, :]

            # K-means clustering
            # Map DistType to sklearn metric
            # sklearn KMeans uses Euclidean; cosine distance could be approximated by normalizing X.
            if DistType == "cosine":
                # Normalize rows
                norm = np.linalg.norm(X_ss, axis=1, keepdims=True)
                norm[norm == 0] = 1.0
                X_km = X_ss / norm
            else:
                X_km = X_ss

            km = KMeans(n_clusters=K, n_init=10, random_state=None)
            IDX_ss = km.fit_predict(X_km) + 1  # 1-based labels to mimic MATLAB

            # Build connectivity matrix for this fold
            M = Build_Connectivity_Matrix(IDX_ss, np.where(tmp_ss)[0], Subsample_type, n_items)
            M_all[:, :, iFold] = M

        # Consensus matrix: average across folds
        Consensus = np.mean(M_all, axis=2)
        Consensus_all.append(Consensus)

        # Optionally save each Consensus matrix
        np.save(os.path.join(outDir_cons, f"Consensus_{K}.npy"), Consensus)

    # Stack into 3D array for quality computation
    Consensus_stack = np.stack(Consensus_all, axis=2)
    CDF, AUC = ComputeClusteringQuality(Consensus_stack, K_range)

    return CDF, AUC
