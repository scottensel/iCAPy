import os
import numpy as np
import matplotlib.pyplot as plt

from functions.Utilities.WriteInformation import write_information
from functions.Clustering.ZScore_iCAPs import zscore_icaps
from .getIDXmat import getIDXmat


def evaluate_soft_cluster_thres_corrs(clustering_results, TC, param, fid=None):
    """
    MATLAB-faithful port of evaluateSoftClusterThres_corrs.m

    MATLAB reference behavior:
      - thresVals = param.softClusterThres
      - AI = clusteringResults.AI'
      - compute soft assignments per threshold using getIDXmat(dist_to_centroid, xi)
      - compute normalized innovations per subject:
          Activity_inducing_norm(:,vols_iS)=AI(:,vols_iS)./sqrt(mean(AI(:,vols_iS).^2,2))
          innovations_norm(:,vols_iS)=[zeros(nVox,1), diff(Activity_inducing_norm(:,vols_iS),1,2)]
      - for each threshold and each innovation frame:
          t corrected if > nTP_sub(iS) then t=t-nTP_sub(iS); if t==1 then t=2
          compute measured innovation amplitude inside iCAP z-map mask (>1.5) and innov!=0
          compute estimated amplitude change from timecourse differences
      - corr_amps = diag(corr(AI_change_measured, TC_change_estimated))
      - opt_id = first peak of corr_amps (if none, choose global max)
      - save evalCorr.eps in param.outDir_reg
    """

    # -----------------------------
    # Pull inputs like MATLAB
    # -----------------------------
    thresVals = np.asarray(param["softClusterThres"], dtype=float)
    nThres = thresVals.size

    AI_subject_labels = np.asarray(clustering_results["AI_subject_labels"]).squeeze()
    subject_labels = np.asarray(clustering_results["subject_labels"]).squeeze()
    time_labels = np.asarray(clustering_results["time_labels"]).squeeze()
    dist_to_centroid = np.asarray(clustering_results["dist_to_centroid"])
    iCAPs = np.asarray(clustering_results["iCAPs"])

    nSub = int(param["n_subjects"])

    # MATLAB: AI = clusteringResults.AI'
    # We want AI as (nVox, nTP_all). If stored as (nTP_all, nVox), transpose.
    AI = np.asarray(clustering_results["AI"])
    if AI.ndim != 2:
        raise ValueError("clustering_results['AI'] must be 2D.")
    # Heuristic: if rows look like timepoints and columns like voxels, transpose.
    # MATLAB expects nVox x nTP_all.
    if AI.shape[0] > AI.shape[1]:
        AI = AI.T

    # MATLAB: iCAPs_z = ZScore_iCAPs(clusteringResults.iCAPs);
    iCAPs_z = zscore_icaps(iCAPs)

    nVox, nTP_all = AI.shape
    nI = int(subject_labels.shape[0])
    n_items, nClus = dist_to_centroid.shape

    if n_items != nI:
        raise ValueError(
            f"dist_to_centroid has {n_items} rows but subject_labels has {nI}. "
            "These must match (one row per innovation frame)."
        )

    # -----------------------------
    # Precompute soft assignments for each threshold (MATLAB for-loop)
    # -----------------------------
    IDX_mat_all = np.zeros((n_items, nClus, nThres), dtype=int)
    clus_weights = np.zeros((n_items, nClus, nThres), dtype=float)
    nClus_mat_all = np.zeros((n_items, nThres), dtype=int)

    for iT, xi in enumerate(thresVals):
        IDX_mat, dist_thres, clos_to_centroid = getIDXmat(dist_to_centroid, xi)
        IDX_mat_all[:, :, iT] = IDX_mat
        nClus_mat_all[:, iT] = IDX_mat.sum(axis=1)
        clus_weights[:, :, iT] = clos_to_centroid

    # -----------------------------
    # Compute normalized innovations (MATLAB block)
    # -----------------------------
    if fid is not None:
        write_information(fid, "Computing normalized innovations...")

    Activity_inducing_norm = np.zeros_like(AI, dtype=float)
    innovations_norm = np.zeros_like(AI, dtype=float)
    nTP_sub = np.zeros((nSub,), dtype=int)

    for iS in range(1, nSub + 1):
        vols_iS = (AI_subject_labels == iS)
        nTP_sub[iS - 1] = int(np.count_nonzero(vols_iS))

        AI_iS = AI[:, vols_iS]  # (nVox, nTP_sub)

        # denom = sqrt(mean(AI(:,vols_iS).^2,2))
        denom = np.sqrt(np.mean(AI_iS ** 2, axis=1, keepdims=True))
        denom[denom == 0] = 1.0

        AI_norm_iS = AI_iS / denom
        AI_norm_iS[np.isnan(AI_norm_iS)] = 0.0

        Activity_inducing_norm[:, vols_iS] = AI_norm_iS

        # innovations_norm(:,vols_iS)=[zeros(nVox,1), diff(AI_norm_iS,1,2)]
        diff_iS = np.diff(AI_norm_iS, axis=1)
        innovations_norm[:, vols_iS] = np.concatenate(
            [np.zeros((nVox, 1)), diff_iS], axis=1
        )

    # -----------------------------
    # Compute innovation amplitudes in the cluster maps
    # -----------------------------
    if fid is not None:
        write_information(fid, "Computing innovation amplitudes in the cluster maps...")

    z_thres = 1.5  # MATLAB: thres=1.5

    AI_change_measured = np.zeros((nI, nThres), dtype=float)
    TC_change_estimated = np.zeros((nI, nThres), dtype=float)

    for iT in range(nThres):
        if fid is not None:
            write_information(fid, f"Soft cluster assignment factor: {thresVals[iT]}")

        for iI in range(nI):
            # MATLAB progress print every 1000
            # if (iI+1) % 1000 == 0: print(iI+1, end=" ")

            iS = int(subject_labels[iI])  # MATLAB 1..nSub
            t = int(time_labels[iI])      # MATLAB 1-based time

            # MATLAB: if t>nTP_sub(iS); t=t-nTP_sub(iS); end
            if t > nTP_sub[iS - 1]:
                t = t - nTP_sub[iS - 1]

            # MATLAB: if t==1; t=t+1; end
            if t == 1:
                t = 2

            # Subject TC for this threshold
            # MATLAB: TC{iT}{iS} => Python: TC[iT][iS-1]
            TC_iS = np.asarray(TC[iT][iS - 1])
            if TC_iS.ndim != 2:
                # must be (nClus, nTP_sub)
                continue

            # ---- CRITICAL SAFETY CHECK (fixes your crash) ----
            # If TC has 0 timepoints or t is out of bounds, skip this innovation
            nTP_TC = TC_iS.shape[1]
            if nTP_TC == 0:
                continue
            if t < 2 or t > nTP_TC:
                continue

            # innovations for this subject
            vols_iS = (AI_subject_labels == iS)
            innovSub = innovations_norm[:, vols_iS]  # (nVox, nTP_sub)

            # Need innovSub time index (t-1 in 0-based)
            if (t - 1) >= innovSub.shape[1]:
                continue

            # MATLAB: TC_change_tmp = TC{iT}{iS}(:,t) - TC{iT}{iS}(:,t-1)
            TC_change_tmp = TC_iS[:, t - 1] - TC_iS[:, t - 2]  # 0-based

            # clusters assigned at this threshold for this innovation frame
            cluster_ids = np.where(IDX_mat_all[iI, :, iT].astype(bool))[0]
            clusterWeights = clus_weights[iI, :, iT]

            measured_sum = 0.0
            estimated_sum = 0.0

            for iC in cluster_ids:
                # MATLAB:
                # mean(innovSub(innovSub(:,t)~=0 & iCAPs_z(iC,:)'>thres, t)) * clusterWeights(iC)
                vox_mask = (innovSub[:, t - 1] != 0) & (iCAPs_z[iC, :] > z_thres)
                vals = innovSub[vox_mask, t - 1]
                if vals.size > 0:
                    measured_sum += float(np.mean(vals)) * float(clusterWeights[iC])

                # MATLAB: TC_change_tmp(iC)*clusterWeights(iC)
                estimated_sum += float(TC_change_tmp[iC]) * float(clusterWeights[iC])

            AI_change_measured[iI, iT] = measured_sum
            TC_change_estimated[iI, iT] = estimated_sum

    AI_change_measured[np.isnan(AI_change_measured)] = 0.0
    TC_change_estimated[np.isnan(TC_change_estimated)] = 0.0

    # -----------------------------
    # MATLAB: corr_amps = diag(corr(AI_change_measured, TC_change_estimated));
    # -----------------------------
    corr_amps = np.zeros((nThres,), dtype=float)
    for iT in range(nThres):
        x = AI_change_measured[:, iT]
        y = TC_change_estimated[:, iT]
        if np.std(x) == 0 or np.std(y) == 0:
            corr_amps[iT] = 0.0
        else:
            corr_amps[iT] = float(np.corrcoef(x, y)[0, 1])

    # MATLAB:
    # [~,opt_id]=findpeaks(corr_amps);
    # opt_id=opt_id(1);
    # If no peaks exist, MATLAB would error; here we fall back to argmax.
    peaks = []
    for i in range(1, nThres - 1):
        if corr_amps[i] > corr_amps[i - 1] and corr_amps[i] > corr_amps[i + 1]:
            peaks.append(i)
    opt_id = peaks[0] if len(peaks) > 0 else int(np.argmax(corr_amps))

    if fid is not None:
        write_information(fid, f"The optimum correlation is at xi={thresVals[opt_id]}")

    # -----------------------------
    # Plot + save EPS (MATLAB print evalCorr)
    # -----------------------------
    os.makedirs(param["outDir_reg"], exist_ok=True)

    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_subplot(111)
    ax.plot(thresVals, corr_amps, "*")
    ax.plot(thresVals[opt_id], corr_amps[opt_id], "*r")
    ax.set_title("correlation measured vs. estimated transient amplitudes")
    fig.savefig(
        os.path.join(param["outDir_reg"], "evalCorr.eps"),
        format="eps",
        bbox_inches="tight",
    )
    plt.close(fig)

    # MATLAB returns nothing; returning is handy for debugging
    return {
        "corr_amps": corr_amps,
        "opt_id": opt_id,
        "opt_xi": float(thresVals[opt_id]),
        "AI_change_measured": AI_change_measured,
        "TC_change_estimated": TC_change_estimated,
    }
