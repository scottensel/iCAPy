import numpy as np
from functions.Utilities.WriteInformation import write_information
from functions.Regression.getIDXmat import getIDXmat  # must exist


def generate_time_courses_weighted(clusteringResults, param, fid=None):
    """
    Faithful Python translation of GenerateTimeCoursesWeighted.m
    """

    # --------------------------------------------------
    # Unpack inputs (MATLAB struct equivalent)
    # --------------------------------------------------
    iCAPs = np.asarray(clusteringResults["iCAPs"])           # (nClus, nVox)
    AI = np.asarray(clusteringResults["AI"]).T               # (nVox, nTP_all)
    IDX = np.asarray(clusteringResults["IDX"]).astype(int)
    subject_labels = np.asarray(clusteringResults["subject_labels"]).astype(int)
    time_labels = np.asarray(clusteringResults["time_labels"]).astype(int)
    AI_subject_labels = np.asarray(clusteringResults["AI_subject_labels"]).astype(int)

    nSub = int(param["n_subjects"])
    nClus, nVox = iCAPs.shape

    # --------------------------------------------------
    # Distances → soft assignment
    # --------------------------------------------------
    if "dist_to_centroid" in clusteringResults:
        dist_to_centroid = clusteringResults["dist_to_centroid"]
    else:
        raise RuntimeError(
            "GenerateTimeCoursesWeighted requires dist_to_centroid for soft assignment"
        )

    IDX_mat, _, _ = getIDXmat(dist_to_centroid, param["softClusterThres"])

    # --------------------------------------------------
    # Fix time_labels (MATLAB logic)
    # --------------------------------------------------
    for iS in range(1, nSub + 1):
        nTP_sub = np.sum(AI_subject_labels == iS)
        mask = (subject_labels == iS) & (time_labels > nTP_sub)
        time_labels[mask] -= nTP_sub

    # --------------------------------------------------
    # Outputs
    # --------------------------------------------------
    TC = []
    stats = dict(RSS=[], n=[], k=[], bic=[], aic=[])

    # --------------------------------------------------
    # Regression per subject
    # --------------------------------------------------
    for iS in range(1, nSub + 1):

        vols_AI = AI_subject_labels == iS
        AI_sub = AI[:, vols_AI]
        nTP_sub = AI_sub.shape[1]

        y = AI_sub.reshape(-1, order="F")  # MATLAB-style vectorization

        S_blocks = []
        cluster_ids = []

        for iC in range(nClus):

            # innovation times
            t = np.sort(time_labels[(subject_labels == iS) & (IDX_mat[:, iC] == 1)])
            t = np.unique(np.concatenate(([1], t, [nTP_sub + 1])))

            B = np.zeros((nTP_sub, len(t) - 1))
            for k in range(len(t) - 1):
                B[t[k] - 1 : t[k + 1] - 1, k] = 1

            S_k = np.kron(B, iCAPs[iC, :][:, None])
            S_blocks.append(S_k)
            cluster_ids.extend([iC] * B.shape[1])

        S = np.hstack(S_blocks)

        # --------------------------------------------------
        # OLS solve
        # --------------------------------------------------
        beta = np.linalg.lstsq(S, y, rcond=None)[0]

        # --------------------------------------------------
        # Reconstruct time courses
        # --------------------------------------------------
        TC_sub = np.zeros((nClus, nTP_sub))
        ptr = 0

        for iC in range(nClus):
            nSeg = np.sum(np.array(cluster_ids) == iC)
            b = beta[ptr : ptr + nSeg]
            ptr += nSeg

            t = np.sort(time_labels[(subject_labels == iS) & (IDX_mat[:, iC] == 1)])
            t = np.unique(np.concatenate(([1], t, [nTP_sub + 1])))

            for k in range(len(b)):
                TC_sub[iC, t[k] - 1 : t[k + 1] - 1] = b[k]

        TC.append(TC_sub)

        # --------------------------------------------------
        # Stats
        # --------------------------------------------------
        resid = S @ beta - y
        RSS = np.sum(resid**2)
        n_obs = S.shape[0]
        k_par = S.shape[1]

        stats["RSS"].append(RSS)
        stats["n"].append(n_obs)
        stats["k"].append(k_par)
        stats["bic"].append(n_obs * np.log(RSS / n_obs) + k_par * np.log(n_obs))
        stats["aic"].append(n_obs * np.log(RSS / n_obs) + 2 * k_par)

    # convert stats lists to arrays
    for k in stats:
        stats[k] = np.asarray(stats[k])

    if fid is not None:
        write_information(fid, "GenerateTimeCoursesWeighted: spatio-temporal regression completed.")

    return TC, stats
