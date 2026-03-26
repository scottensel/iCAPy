import os
import numpy as np
import matplotlib.pyplot as plt

from functions.Utilities.WriteInformation import write_information
from functions.Clustering.ZScore_iCAPs import zscore_icaps
from .getIDXmat import getIDXmat


def evaluate_soft_cluster_thres_corrs(clustering_results, TC, param, fid=None):
    """
    MATLAB-faithful port of evaluateSoftClusterThres_corrs.m

    All subject/cluster indices are 0-based. Time indices (t, time_labels)
    are 1-based (frame 1 = first frame) and converted to 0-based only when
    indexing into arrays.
    """

    thresVals = np.asarray(param["softClusterThres"], dtype=float)
    nThres    = thresVals.size

    AI_subject_labels = np.asarray(clustering_results["AI_subject_labels"]).squeeze()
    subject_labels    = np.asarray(clustering_results["subject_labels"]).squeeze()
    time_labels       = np.asarray(clustering_results["time_labels"]).squeeze()
    dist_to_centroid  = np.asarray(clustering_results["dist_to_centroid"])
    iCAPs             = np.asarray(clustering_results["iCAPs"])

    nSub = int(param["n_subjects"])

    # MATLAB: AI = clusteringResults.AI'  => (nVox, nTP_all)
    AI = np.asarray(clustering_results["AI"])
    if AI.ndim != 2:
        raise ValueError("clustering_results['AI'] must be 2D.")
    if AI.shape[0] > AI.shape[1]:
        AI = AI.T   # ensure (nVox, nTP_all)

    iCAPs_z = zscore_icaps(iCAPs)

    nVox, nTP_all  = AI.shape
    nI             = int(subject_labels.shape[0])
    n_items, nClus = dist_to_centroid.shape

    if n_items != nI:
        raise ValueError(
            f"dist_to_centroid has {n_items} rows but subject_labels has {nI}."
        )

    # ------------------------------------------------------------------
    # Precompute soft assignments for every threshold
    # ------------------------------------------------------------------
    IDX_mat_all   = np.zeros((n_items, nClus, nThres), dtype=int)
    clus_weights  = np.zeros((n_items, nClus, nThres), dtype=float)
    nClus_mat_all = np.zeros((n_items, nThres), dtype=int)

    for iT, xi in enumerate(thresVals):
        IDX_mat, dist_thres, clos_to_centroid = getIDXmat(dist_to_centroid, xi)
        IDX_mat_all[:, :, iT]  = IDX_mat
        nClus_mat_all[:, iT]   = IDX_mat.sum(axis=1)
        clus_weights[:, :, iT] = clos_to_centroid

    # ------------------------------------------------------------------
    # Normalised innovations  (MATLAB: Activity_inducing_norm, innovations_norm)
    # Both are (nVox, nTP_all), 0-based subject labels
    # ------------------------------------------------------------------
    if fid is not None:
        write_information(fid, "Computing normalized innovations...")

    Activity_inducing_norm = np.zeros_like(AI, dtype=float)
    innovations_norm       = np.zeros_like(AI, dtype=float)
    nTP_sub                = np.zeros(nSub, dtype=int)

    for iS in range(nSub):
        vols_iS     = AI_subject_labels == iS          # 0-based
        nTP_sub[iS] = int(np.count_nonzero(vols_iS))   # fixed: was nTP_sub[iS-1]

        AI_iS  = AI[:, vols_iS]                        # (nVox, nTP_sub)
        denom  = np.sqrt(np.mean(AI_iS ** 2, axis=1, keepdims=True))
        denom[denom == 0] = 1.0

        AI_norm_iS = AI_iS / denom
        AI_norm_iS[np.isnan(AI_norm_iS)] = 0.0

        Activity_inducing_norm[:, vols_iS] = AI_norm_iS

        # MATLAB: [zeros(nVox,1), diff(AI_norm_iS, 1, 2)]
        innovations_norm[:, vols_iS] = np.concatenate(
            [np.zeros((nVox, 1)), np.diff(AI_norm_iS, axis=1)], axis=1
        )

    # ------------------------------------------------------------------
    # Innovation amplitudes vs. estimated TC changes
    # ------------------------------------------------------------------
    if fid is not None:
        write_information(fid, "Computing innovation amplitudes in the cluster maps...")

    z_thres = 1.5

    AI_change_measured  = np.zeros((nI, nThres), dtype=float)
    TC_change_estimated = np.zeros((nI, nThres), dtype=float)

    for iT in range(nThres):
        if fid is not None:
            write_information(fid, f"Soft cluster assignment factor: {thresVals[iT]}")

        for iI in range(nI):
            iS = int(subject_labels[iI])   # 0-based
            t  = int(time_labels[iI])      # 1-based time index (matches MATLAB)

            # MATLAB: if t > nTP_sub(iS); t = t - nTP_sub(iS); end
            if t > nTP_sub[iS]:
                t = t - nTP_sub[iS]

            # MATLAB: if t == 1; t = t + 1; end
            if t == 1:
                t = 2

            TC_iS   = np.asarray(TC[iT][iS])   # 0-based subject index
            if TC_iS.ndim != 2:
                continue
            nTP_TC = TC_iS.shape[1]
            if nTP_TC == 0 or t < 2 or t > nTP_TC:
                continue

            vols_iS  = AI_subject_labels == iS
            innovSub = innovations_norm[:, vols_iS]   # (nVox, nTP_sub)

            if (t - 1) >= innovSub.shape[1]:
                continue

            # MATLAB: TC{iT}{iS}(:,t) - TC{iT}{iS}(:,t-1)  (1-based t)
            # => 0-based: TC_iS[:, t-1] - TC_iS[:, t-2]
            TC_change_tmp = TC_iS[:, t - 1] - TC_iS[:, t - 2]

            cluster_ids    = np.where(IDX_mat_all[iI, :, iT].astype(bool))[0]
            clusterWeights = clus_weights[iI, :, iT]

            measured_sum  = 0.0
            estimated_sum = 0.0

            for iC in cluster_ids:
                # MATLAB: innovSub(:,t) is 1-based t => 0-based: innovSub[:, t-1]
                col      = innovSub[:, t - 1]
                vox_mask = (col != 0) & (iCAPs_z[iC, :] > z_thres)
                vals     = col[vox_mask]
                if vals.size > 0:
                    measured_sum += float(np.mean(vals)) * float(clusterWeights[iC])

                estimated_sum += float(TC_change_tmp[iC]) * float(clusterWeights[iC])

            AI_change_measured[iI, iT]  = measured_sum
            TC_change_estimated[iI, iT] = estimated_sum

    AI_change_measured[np.isnan(AI_change_measured)]   = 0.0
    TC_change_estimated[np.isnan(TC_change_estimated)] = 0.0

    # ------------------------------------------------------------------
    # MATLAB: corr_amps = diag(corr(AI_change_measured, TC_change_estimated))
    # ------------------------------------------------------------------
    corr_amps = np.zeros(nThres, dtype=float)
    for iT in range(nThres):
        x = AI_change_measured[:, iT]
        y = TC_change_estimated[:, iT]
        if np.std(x) == 0 or np.std(y) == 0:
            corr_amps[iT] = 0.0
        else:
            corr_amps[iT] = float(np.corrcoef(x, y)[0, 1])

    # MATLAB: [~, opt_id] = findpeaks(corr_amps); opt_id = opt_id(1)
    # findpeaks never picks index 0 or nThres-1
    peaks = [
        i for i in range(1, nThres - 1)
        if corr_amps[i] > corr_amps[i - 1] and corr_amps[i] > corr_amps[i + 1]
    ]
    opt_id = peaks[0] if peaks else int(np.argmax(corr_amps))

    if fid is not None:
        write_information(fid, f"The optimum correlation is at xi={thresVals[opt_id]}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    os.makedirs(param["outDir_reg"], exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.plot(thresVals, corr_amps, "*")
    ax.plot(thresVals[opt_id], corr_amps[opt_id], "*r")
    ax.set_title("correlation measured vs. estimated transient amplitudes")
    fig.savefig(
        os.path.join(param["outDir_reg"], "evalCorr.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "corr_amps":            corr_amps,
        "opt_id":               opt_id,
        "opt_xi":               float(thresVals[opt_id]),
        "AI_change_measured":   AI_change_measured,
        "TC_change_estimated":  TC_change_estimated,
    }