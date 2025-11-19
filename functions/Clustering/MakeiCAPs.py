import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from functions.Utilities.WriteInformation import write_information
from .ZScore_iCAPs import ZScore_iCAPs


def MakeiCAPs(Frames, param, fid=None):
    """Python approximation of MakeiCAPs.m

    Performs k-means clustering of selected frames into iCAPs, with multiple
    folds and Hungarian matching across folds.

    Parameters
    ----------
    Frames : ndarray, shape (n_frames, n_vox)
        Data matrix of frames to cluster.
    param : dict-like
        Must contain:
        - 'K' : number of clusters
        - 'DistType' : 'sqeuclidean' or 'cosine'
        - 'n_folds' : number of folds
        - 'save_cluster_dist' : bool, whether to keep per-fold distances
        - 'outDir_iCAPs' : output directory (not heavily used here)
    fid : file handle or str or None
        For logging.

    Returns
    -------
    iCAPs : ndarray, shape (K, n_vox)
        Final cluster centroids.
    IDX : ndarray, shape (n_frames,)
        Cluster labels (1..K).
    dist_to_centroid : ndarray or list
        Distances of each frame to each centroid.
    iCAPs_folds : dict
        Per-fold clustering results.
    """
    Frames = np.asarray(Frames)
    n_frames, n_vox = Frames.shape
    K = int(param['K'])
    DistType = param.get('DistType', 'sqeuclidean')
    n_folds = int(param.get('n_folds', 1))
    save_cluster_dist = bool(param.get('save_cluster_dist', False))

    # Pre-normalization for cosine distance
    if DistType == 'cosine':
        norm = np.linalg.norm(Frames, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        Frames_km = Frames / norm
    else:
        Frames_km = Frames

    all_iCAPs = []
    all_IDX = []
    all_sum_dist = []
    all_dist_to_centroid = []

    # Run k-means for each fold independently
    for iFold in range(n_folds):
        km = KMeans(n_clusters=K, n_init=10, random_state=None)
        labels = km.fit_predict(Frames_km) + 1  # 1..K
        centers = km.cluster_centers_

        # distances of each frame to its centroid
        dists = cdist(Frames_km, centers, metric='euclidean')
        min_dists = np.min(dists, axis=1)

        all_iCAPs.append(centers)
        all_IDX.append(labels)
        all_sum_dist.append(np.bincount(labels, weights=min_dists, minlength=K + 1)[1:])
        all_dist_to_centroid.append(dists)

    # Match clusters across folds to first fold using Hungarian algorithm
    iCAPs_ref = all_iCAPs[0]
    for iFold in range(1, n_folds):
        cost = cdist(iCAPs_ref, all_iCAPs[iFold],
                     metric='cosine' if DistType == 'cosine' else 'euclidean')
        row_ind, col_ind = linear_sum_assignment(cost)
        # Reorder fold iFold according to mapping
        perm = np.argsort(col_ind)
        all_iCAPs[iFold] = all_iCAPs[iFold][perm, :]
        all_sum_dist[iFold] = all_sum_dist[iFold][perm]
        all_dist_to_centroid[iFold] = all_dist_to_centroid[iFold][:, perm]

        # Remap labels
        labels_fold = all_IDX[iFold]
        new_labels = np.zeros_like(labels_fold)
        for new_k, old_k in enumerate(perm, start=1):
            new_labels[labels_fold == (old_k + 1)] = new_k
        all_IDX[iFold] = new_labels

    # Build iCAPs_folds structure
    iCAPs_folds = {
        'iCAPs': all_iCAPs,
        'IDX': all_IDX,
        'sum_dist': all_sum_dist,
        'dist_to_centroid': all_dist_to_centroid,
        'iCAPs_z': [ZScore_iCAPs(c) for c in all_iCAPs],
    }

    # Select best fold by minimal total distance
    total_dist = np.array([np.sum(sd) for sd in all_sum_dist])
    bestID = int(np.argmin(total_dist))
    iCAPs = all_iCAPs[bestID]
    IDX = all_IDX[bestID]
    dist_to_centroid = all_dist_to_centroid[bestID]

    if fid is not None:
        write_information(
            fid,
            f"iCAPs computed for {K} clusters, {n_folds} folds and with distance {DistType}..."
        )

    return iCAPs, IDX, dist_to_centroid, iCAPs_folds
