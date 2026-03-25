import os
import platform
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from functions.Utilities.WriteInformation import write_information
from .ZScore_iCAPs import zscore_icaps
from .Cosine_Kmeans import cosine_kmeans


def _run_one_fold(Frames, K, DistType, n_init, max_iter, limit_threads):
    """
    Run a single k-means replicate and return labels, centers, full
    distance matrix, and per-cluster sum of distances.
    """
    if DistType == "cosine":
        labels0, centers, _ = cosine_kmeans(
            Frames,
            n_clusters=K,
            n_init=n_init,
            max_iter=max_iter if max_iter is not None else 100,
            random_state=None,
        )
        # Build distance matrix: normalise both Frames and centers first,
        # matching MATLAB's distfun which renormalises centroids each call
        X_normed = Frames / np.maximum(
            np.linalg.norm(Frames, axis=1, keepdims=True), np.finfo(float).eps
        )
        C_normed = centers / np.maximum(
            np.linalg.norm(centers, axis=1, keepdims=True), np.finfo(float).eps
        )
        D = cdist(X_normed, C_normed, metric="cosine")

    else:
        if platform.system() == "Windows" and limit_threads is not None:
            with threadpool_limits(limits=limit_threads):
                km = KMeans(n_clusters=K, n_init=n_init, random_state=None) \
                     if max_iter is None else \
                     KMeans(n_clusters=K, n_init=n_init, max_iter=max_iter, random_state=None)
                labels0 = km.fit_predict(Frames).astype(np.int64)
        else:
            km = KMeans(n_clusters=K, n_init=n_init, random_state=None) \
                 if max_iter is None else \
                 KMeans(n_clusters=K, n_init=n_init, max_iter=max_iter, random_state=None)
            labels0 = km.fit_predict(Frames).astype(np.int64)

        centers = km.cluster_centers_
        De = cdist(Frames, centers, metric="euclidean")
        D  = De * De if DistType == "sqeuclidean" else De

    labels0  = labels0.astype(np.int64)
    assigned = D[np.arange(D.shape[0]), labels0]
    sum_dist = np.bincount(labels0, weights=assigned, minlength=K).astype(np.float64)

    return labels0, centers, D, sum_dist


def make_icaps(Frames, param, fid=None):
    """
    Python port of MakeiCAPs.m.  IDX is 0-based (0..K-1) throughout.

    Branch 1 (default): saveClusterReplicateData missing or false
        Single kmeans call with Replicates=n_folds.
        Returns iCAPs, IDX, dist_to_centroid; iCAPs_folds=[].

    Branch 2: saveClusterReplicateData true
        Runs n_folds replicates independently (Replicates=1 each).
        Hungarian-matches folds 2..n_folds to fold 1.
        Selects the best fold by minimum total sum of distances.
        Returns iCAPs_folds dict with all fold data.
    """
    out_dir = param.get("outDir_iCAPs", None)
    if out_dir is not None and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    Frames    = np.asarray(Frames, dtype=np.float64)
    K         = int(param["K"])
    DistType  = param.get("DistType", "sqeuclidean")
    n_folds   = int(param.get("n_folds", 1))
    limit_threads = param.get("limitThreads", None)
    max_iter  = int(param["MaxIter"]) if "MaxIter" in param else None
    save_reps = bool(param.get("saveClusterReplicateData", False))

    # ------------------------------------------------------------------
    # Branch 1: default  (MATLAB: single kmeans call, Replicates=n_folds)
    # ------------------------------------------------------------------
    if not save_reps:
        IDX, iCAPs, dist_to_centroid, _ = _run_one_fold(
            Frames, K, DistType,
            n_init=n_folds, max_iter=max_iter,
            limit_threads=limit_threads,
        )

        if fid is not None:
            write_information(
                fid,
                f"iCAPs computed for {K} clusters, {n_folds} folds "
                f"and with distance {DistType}..."
            )

        return iCAPs, IDX, dist_to_centroid, []

    # ------------------------------------------------------------------
    # Branch 2: saveClusterReplicateData=True  (loop, Replicates=1 each)
    # ------------------------------------------------------------------
    IDX_list              = [None] * n_folds
    iCAPs_list            = [None] * n_folds
    sum_dist_list         = [None] * n_folds
    dist_to_centroid_list = [None] * n_folds

    for iFold in range(n_folds):
        if fid is not None:
            write_information(fid, str(iFold + 1))

        labels0, centers, D, sum_dist = _run_one_fold(
            Frames, K, DistType,
            n_init=1, max_iter=max_iter,
            limit_threads=limit_threads,
        )

        IDX_list[iFold]              = labels0
        iCAPs_list[iFold]            = centers
        sum_dist_list[iFold]         = sum_dist
        dist_to_centroid_list[iFold] = D

    # Match all folds to fold 1 (Hungarian algorithm)
    for iFold in range(1, n_folds):
        if DistType == "cosine":
            ref_normed  = iCAPs_list[0] / np.maximum(
                np.linalg.norm(iCAPs_list[0], axis=1, keepdims=True), np.finfo(float).eps
            )
            fold_normed = iCAPs_list[iFold] / np.maximum(
                np.linalg.norm(iCAPs_list[iFold], axis=1, keepdims=True), np.finfo(float).eps
            )
            dist_between_folds = cdist(ref_normed, fold_normed, metric="cosine")
        else:
            dist_between_folds = cdist(iCAPs_list[0], iCAPs_list[iFold], metric="sqeuclidean")

        row_ind, col_ind = linear_sum_assignment(dist_between_folds)
        indexhun = np.empty(K, dtype=np.int64)
        indexhun[row_ind.astype(np.int64)] = col_ind.astype(np.int64)

        IDX_new              = np.empty_like(IDX_list[iFold])
        iCAPs_new            = np.empty_like(iCAPs_list[iFold])
        sum_dist_new         = np.empty_like(sum_dist_list[iFold])
        dist_to_centroid_new = np.empty_like(dist_to_centroid_list[iFold])

        for iC in range(K):
            matched                         = indexhun[iC]
            IDX_new[IDX_list[iFold] == matched] = iC
            iCAPs_new[iC, :]                = iCAPs_list[iFold][matched, :]
            sum_dist_new[iC]                = sum_dist_list[iFold][matched]
            dist_to_centroid_new[:, iC]     = dist_to_centroid_list[iFold][:, matched]

        IDX_list[iFold]              = IDX_new
        iCAPs_list[iFold]            = iCAPs_new
        sum_dist_list[iFold]         = sum_dist_new
        dist_to_centroid_list[iFold] = dist_to_centroid_new

    # Build iCAPs_folds dict
    iCAPs_folds = {
        "iCAPs":            iCAPs_list,
        "IDX":              IDX_list,
        "sum_dist":         sum_dist_list,
        "dist_to_centroid": dist_to_centroid_list,
    }

    # Select best fold and compute z-scored maps
    total_dist_sum = np.zeros(n_folds, dtype=np.float64)
    iCAPs_z_list   = [None] * n_folds

    for iFold in range(n_folds):
        total_dist_sum[iFold] = float(np.sum(iCAPs_folds["sum_dist"][iFold]))
        iCAPs_z_list[iFold]   = zscore_icaps(
            iCAPs_folds["iCAPs"][iFold], [], iCAPs_folds["IDX"][iFold]
        )

    iCAPs_folds["total_dist_sum"] = total_dist_sum
    iCAPs_folds["iCAPs_z"]        = iCAPs_z_list

    bestID           = int(np.argmin(total_dist_sum))
    iCAPs            = iCAPs_list[bestID]
    IDX              = IDX_list[bestID]
    dist_to_centroid = dist_to_centroid_list[bestID]

    if fid is not None:
        write_information(
            fid,
            f"iCAPs computed for {K} clusters, {n_folds} folds "
            f"and with distance {DistType}..."
        )

    return iCAPs, IDX, dist_to_centroid, iCAPs_folds