import numpy as np
import scipy.sparse as sp
from functions.n00_Utilities.WriteInformation import write_information
from functions.n04_Regression.getIDXmat import getIDXmat


"""
Computes the time courses of activity for all assessed iCAPs using
spatio-temporal regression. This is an alternative to the unconstrained
approach — it uses the information of significant innovations to construct
the design matrix.

Inputs:
    clusteringResults - dict containing the necessary input data:
        'iCAPs'           - iCAPs maps resulting from k-means clustering
                            (n_iCAPs x n_voxels)
        'AI'              - activity-inducing signals
                            (n_voxels x (n_subj * n_TP)) matrix of
                            concatenated subject activity-inducing data
        'IDX'             - clustering information of significant
                            innovation frames (n_significant_innov,)
        'time_labels'     - timing information of significant innovation
                            frames (n_significant_innov,)
        'subject_labels'  - subject information of significant innovation
                            frames (n_significant_innov,)
        ['dist_to_centroid'] - (n_significant_innov x n_iCAPs) matrix of
                            distances from all significant innovation
                            frames to all cluster centroids (iCAPs)

        For regression with a soft assignment factor, one of the following
        is required:
        ['dist_to_centroid'] - as described above
        ['iCAPs_folds']   - information on all clustering folds, used to
                            retrieve distances to cluster centers:
                              'sum_dist' - total sum of distances from all
                                  frames to their corresponding cluster
                                  centers
                              'dist_to_centroid' - matrix containing
                                  distances to all cluster centers for
                                  every frame

    param - dict containing necessary parameters:
        'n_subjects'         - number of subjects
        ['softClusterThres'] - soft assignment factor; should be larger
                               than 1 (1.1 to 1.25 are good values; for
                               parameter optimization run
                               evaluateSoftClusterThres). If not set or
                               empty, hard clustering will be used for
                               back-projection.
        ['excludeMotionFrames'] - select whether motion frames should be
                               excluded from the regression (if scrubbing)
        ['n_folds']          - required if distances should be taken from
                               clusteringResults['iCAPs_folds']

Outputs:
    TC    - list of length nSubjects; each element is a
            (nClus x nTP_subject) array of time courses
    stats - dict containing model fitting statistics per subject:
              'RSS' - residual sum of squares
              'n'   - number of observations
              'k'   - number of regressors
              'bic' - Bayesian information criterion
              'aic' - Akaike information criterion

v2.0 DZ 27.10.2017 - added compatibility with scrubbed data
v2.0 DZ 29.05.2018 - removed compatibility with scrubbed data and
                     updated for finalized toolbox
"""

def generate_time_courses_weighted(clusteringResults, param, fid=None):

    # --------------------------------------------------
    # Unpack inputs
    # --------------------------------------------------
    iCAPs             = np.asarray(clusteringResults["iCAPs"])
    AI                = np.asarray(clusteringResults["AI"]).T             # (nVox, nTP_all)
    subject_labels    = np.asarray(clusteringResults["subject_labels"]).astype(int)
    time_labels       = np.asarray(clusteringResults["time_labels"]).astype(int).copy()
    AI_subject_labels = np.asarray(clusteringResults["AI_subject_labels"]).astype(int)

    nSub        = int(param["n_subjects"])
    nClus, nVox = iCAPs.shape

    if fid is not None:
        write_information(fid, "GenerateTimeCoursesWeighted: starting spatio-temporal regression...")

    # --------------------------------------------------
    # Soft assignment matrix
    # --------------------------------------------------
    if "dist_to_centroid" in clusteringResults:
        dist_to_centroid = np.asarray(clusteringResults["dist_to_centroid"])
    else:
        raise RuntimeError(
            "GenerateTimeCoursesWeighted requires dist_to_centroid in clusteringResults."
        )

    IDX_mat, _, _ = getIDXmat(dist_to_centroid, param["softClusterThres"])

    # --------------------------------------------------
    # Per-subject timepoint counts (0-based subject labels)
    # --------------------------------------------------
    nTP_sub_arr = np.zeros(nSub, dtype=int)
    for iS in range(nSub):
        nTP_sub_arr[iS] = int(np.sum(AI_subject_labels == iS))

    # --------------------------------------------------
    # Fix time_labels: remove positive/negative frame doubling
    # --------------------------------------------------
    for iS in range(nSub):
        nTP  = nTP_sub_arr[iS]
        mask = (subject_labels == iS) & (time_labels > nTP)
        time_labels[mask] -= nTP

    # --------------------------------------------------
    # Regression per subject
    # --------------------------------------------------
    TC    = []
    stats = dict(RSS=[], n=[], k=[], bic=[], aic=[])

    for iS in range(nSub):
        nTP_sub = nTP_sub_arr[iS]

        vols_AI_iS      = AI_subject_labels == iS
        AI_sub          = AI[:, vols_AI_iS]                   # (nVox, nTP_sub)
        AI_concatenated = AI_sub.reshape(-1, order="F")        # (nTP_sub * nVox,)

        if fid is not None:
            write_information(fid, f"  subject {iS}: constructing design matrix...")

        # --------------------------------------------------
        # Build design matrix S (sparse, MATLAB-style).
        # col_meta tracks (cluster_id, segment_index) for every column so
        # that after zero-column removal the time boundaries stay in sync.
        # --------------------------------------------------
        sub_clus_time_labels = {}   # iC -> 1-based boundary array
        S_blocks             = []
        col_meta             = []   # list of (iC, seg_idx) per column

        for iC in range(nClus):
            t_innov = np.sort(
                time_labels[(subject_labels == iS) & (IDX_mat[:, iC] == 1)]
            )
            t_innov = np.unique(np.concatenate(([1], t_innov, [nTP_sub + 1])))
            sub_clus_time_labels[iC] = t_innov

            nInnov_Clus = len(t_innov)

            B_k = np.zeros((nTP_sub, nInnov_Clus - 1), dtype=float)
            for iFrame in range(nInnov_Clus - 1):
                firstID = t_innov[iFrame] - 1
                lastID  = t_innov[iFrame + 1] - 1
                B_k[firstID:lastID, iFrame] = 1.0

            iCAP_col = iCAPs[iC, :].reshape(-1, 1)
            S_k      = np.kron(B_k, iCAP_col)

            S_blocks.append(sp.csc_matrix(S_k))

            for seg in range(nInnov_Clus - 1):
                col_meta.append((iC, seg))

        S        = sp.hstack(S_blocks, format="csc")
        col_meta = np.array(col_meta, dtype=int)   # (total_cols, 2)

        # MATLAB: S(:, ~sum(S,1)) = []
        col_sums  = np.asarray(S.sum(axis=0)).ravel()
        keep_cols = col_sums != 0
        S         = S[:, keep_cols]
        col_meta  = col_meta[keep_cols]   # keep (iC, seg) pairs in sync with S

        nBeta = S.shape[1]

        if fid is not None:
            write_information(fid, f"  subject {iS}: solving OLS ({nBeta} regressors)...")

        # --------------------------------------------------
        # OLS solve — sparse normal equations matching MATLAB
        # --------------------------------------------------
        X1           = S.T @ S
        tmp          = S.T @ AI_concatenated
        innovWeights = np.linalg.solve(X1.toarray(), tmp)

        # --------------------------------------------------
        # Reconstruct time courses.
        # Use col_meta to map each surviving beta back to its cluster and
        # its original segment boundaries — avoids the count mismatch that
        # arose when zero columns were dropped from sub_clus_time_labels.
        # --------------------------------------------------
        TC_sub = np.zeros((nClus, nTP_sub))

        for iC in range(nClus):
            beta_mask    = col_meta[:, 0] == iC
            beta_indices = np.where(beta_mask)[0]
            seg_indices  = col_meta[beta_mask, 1]   # original segment indices

            innov_full = sub_clus_time_labels[iC]   # full boundary array (all segs)

            for beta_idx, seg_idx in zip(beta_indices, seg_indices):
                firstID = innov_full[seg_idx] - 1         # 0-based start (inclusive)
                lastID  = innov_full[seg_idx + 1] - 1     # 0-based end (exclusive)
                TC_sub[iC, firstID:lastID] = innovWeights[beta_idx]

        TC.append(TC_sub)

        # --------------------------------------------------
        # Stats — matches MATLAB exactly
        # --------------------------------------------------
        resid = S @ innovWeights - AI_concatenated
        RSS   = float(np.sum(resid ** 2))
        n_obs = int(S.shape[0])
        k_par = int(S.shape[1])

        stats["RSS"].append(RSS)
        stats["n"].append(n_obs)
        stats["k"].append(k_par)
        stats["bic"].append(n_obs * np.log(RSS / n_obs) + k_par * np.log(n_obs))
        stats["aic"].append(n_obs * np.log(RSS / n_obs) + 2.0 * k_par)

    for key in stats:
        stats[key] = np.asarray(stats[key])

    if fid is not None:
        write_information(fid, "GenerateTimeCoursesWeighted: regression completed.")

    return TC, stats