import numpy as np

"""
Computes temporal characteristics of iCAPs time courses.
Occurrences of only one frame will not be counted.

Input:
    TC                - list of length nSubjects; each element is a
                        (nClus x nTP) array of time courses per iCAP
    clusteringResults - dict with clustering results containing:
                          'AI_subject_labels'
                          'subject_labels'
                          'IDX'
                          ['scrub_labels'] - only needed if
                              param['excludeMotionFrames'] = 1
    param             - dict with necessary fields:
                          'activityThres' - threshold at which normalized
                              time courses will be considered "active",
                              default = 1

Output:
    tempChar - dict containing fields with temporal characteristics

      * Characteristics of significant innovations:
        'innov_counts_total'       - number of all significant innovations
                                     per subject
        'innov_counts_total_perc'  - percentage of significant innovations
                                     in all included time points
        'innov_counts'             - (nClus x nSub) number of significant
                                     innovations per iCAP per subject
        'innov_counts_percOfInnov' - (nClus x nSub) percentage of
                                     innovations in this iCAP in all
                                     significant innovations

      * Thresholded time courses and overall characteristics
        (lists of length nSubjects):
        'TC_norm_thes'         - normalized and thresholded time courses,
                                 with occurrences of only one frame removed
        'TC_active'            - activity information: 1 if positive
                                 activity, -1 if negative, 0 if none
        'coactiveiCAPs_total'  - (1 x nTP_subject) number of active iCAPs
                                 per timepoint
        'coactivation'         - (nClus x nClus x nTP_subject) coactivation
                                 between every two iCAPs

      * Temporal characteristics of activity blocks
        (nClus x nSub arrays):
        'occurrences'               - number of activity blocks in
                                      thresholded time courses per cluster
                                      per subject
        'occurrences_pos'           - number of positive activity blocks
        'occurrences_neg'           - number of negative activity blocks

        'duration_total_counts'     - number of active timepoints per
                                      cluster per subject
        'duration_total_pos_counts' - number of positively active timepoints
        'duration_total_neg_counts' - number of negatively active timepoints

        'duration_total_perc'       - total duration of iCAP as percentage
                                      of whole scan duration
        'duration_total_pos_perc'   - total positive duration as percentage
                                      of whole scan duration
        'duration_total_neg_perc'   - total negative duration as percentage
                                      of whole scan duration

        'duration_avg_counts'       - average duration of activity blocks
                                      (number of timepoints)
        'duration_avg_pos_counts'   - average duration of positive activity
                                      blocks (number of timepoints)
        'duration_avg_neg_counts'   - average duration of negative activity
                                      blocks (number of timepoints)

      * Co-activation characteristics (nClus x nClus x nSub arrays):
        'coupling_counts'           - coupling duration (number of
                                      timepoints)
        'coupling_jacc'             - coupling duration as percentage of
                                      total duration of both iCAPs
        'coupling_sameSign'         - same-signed coupling duration
                                      (number of timepoints)
        'coupling_diffSign'         - differently-signed coupling duration
                                      (number of timepoints)
        'coupling_sameSign_jacc'    - same-signed coupling duration as
                                      percentage of total duration of both
                                      iCAPs
        'coupling_diffSign_jacc'    - differently-signed coupling duration
                                      as percentage of total duration of
                                      both iCAPs
"""

def _bwconncomp_1d(tc_row):
    """1D analogue of MATLAB bwconncomp for a single time course row.

    Parameters
    ----------
    tc_row : ndarray, shape (nTP,)
        Time course values (thresholded).

    Returns
    -------
    activeComp : dict
        With keys:
        - 'NumObjects' : int
        - 'PixelIdxList' : list of 1D integer ndarrays
    """
    tc_row = np.asarray(tc_row)
    mask = tc_row != 0
    idx = np.nonzero(mask)[0]
    pixel_lists = []
    if idx.size > 0:
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
            else:
                pixel_lists.append(np.arange(start, prev + 1, dtype=int))
                start = k
                prev = k
        pixel_lists.append(np.arange(start, prev + 1, dtype=int))
    activeComp = {
        "NumObjects": len(pixel_lists),
        "PixelIdxList": pixel_lists,
    }
    return activeComp


def _splitPosNegComps(activeComp, tc_row):
    """Python version of splitPosNegComps(activeComp, TC).

    Splits connected components that include a sign change into pieces with
    constant sign.
    """
    tc_row = np.asarray(tc_row)
    new_pixel_lists = []
    for idx_array in activeComp["PixelIdxList"]:
        if len(idx_array) == 0:
            continue
        vals = tc_row[idx_array]
        signs = np.sign(vals)
        # build segments of constant sign
        current_sign = signs[0]
        current_segment = [idx_array[0]]
        for k, s in zip(idx_array[1:], signs[1:]):
            if s == current_sign:
                current_segment.append(k)
            else:
                new_pixel_lists.append(np.array(current_segment, dtype=int))
                current_sign = s
                current_segment = [k]
        new_pixel_lists.append(np.array(current_segment, dtype=int))
    activeComp["PixelIdxList"] = new_pixel_lists
    activeComp["NumObjects"] = len(new_pixel_lists)
    return activeComp


def _deleteOneFrameOcc(activeComp, tc_row):
    """Python version of deleteOneFrameOcc(activeComp, TC).

    Deletes occurrences (sets to 0) of components consisting of a single frame.
    """
    tc_row = np.asarray(tc_row).copy()
    for idx_array in activeComp["PixelIdxList"]:
        if len(idx_array) < 2:
            tc_row[idx_array] = 0
    return tc_row


def _getCompSign(activeComp, tc_row):
    """Python version of getCompSign(activeComp, TC).

    Ensures components have constant sign and returns component sign.
    """
    tc_row = np.asarray(tc_row)
    comp_sign = []
    # We allow for the possibility of sign changes and re-split if needed
    for idx_array in activeComp["PixelIdxList"]:
        if len(idx_array) == 0:
            comp_sign.append(0)
            continue
        vals = tc_row[idx_array]
        signs = np.sign(vals)
        # check for sign changes
        sign_diff = np.where(np.diff(signs) != 0)[0]
        if sign_diff.size > 0:
            # re-split and recurse
            activeComp2 = _splitPosNegComps(activeComp, tc_row)
            return _getCompSign(activeComp2, tc_row)
        else:
            comp_sign.append(int(signs[0]))
    activeComp["compSign"] = np.array(comp_sign, dtype=int)
    return activeComp


def compute_temporal_characteristics(TC, clusteringResults, param):

    # Helper to access fields from dict or object
    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    thres = param["activityThres"]
    AI_subject_labels = np.asarray(_get(clusteringResults, "AI_subject_labels")).astype(int)
    subject_labels = np.asarray(_get(clusteringResults, "subject_labels")).astype(int)
    IDX = np.asarray(_get(clusteringResults, "IDX")).astype(int)

    # constants
    # TC is expected to be a list-like of arrays
    if isinstance(TC, np.ndarray):
        # assume object array of shape (nSub,)
        TC_list = list(TC)
    else:
        TC_list = list(TC)

    nSub = len(TC_list)
    if nSub == 0:
        raise ValueError("computeTemporalCharacteristics: TC is empty")

    first_tc = np.asarray(TC_list[0])
    nClus = first_tc.shape[0]
    nTP_sub = np.zeros(nSub, dtype=int)
    for iS in range(nSub):
        nTP_sub[iS] = np.asarray(TC_list[iS]).shape[1]

    nTP_all = AI_subject_labels.size

    # scrub_labels handling
    scrub_labels_raw = _get(clusteringResults, "scrub_labels", None)
    exclude_motion = bool(param.get("excludeMotionFromTC", False))
    if scrub_labels_raw is None or (hasattr(scrub_labels_raw, "__len__") and len(scrub_labels_raw) == 0) or not exclude_motion:
        scrub_labels = np.ones(nTP_all, dtype=int)
    else:
        scrub_labels = np.asarray(scrub_labels_raw).astype(int).reshape(-1)
        if scrub_labels.size != nTP_all:
            raise ValueError("scrub_labels length does not match AI_subject_labels length")

    # Preallocate tempChar structure
    tempChar = {}
    tempChar["scrub_labels"] = [None] * nSub
    tempChar["nTP_sub"] = np.zeros(nSub, dtype=int)

    tempChar["innov_counts_total"] = np.zeros((1, nSub), dtype=float)
    tempChar["innov_counts_total_perc"] = np.zeros((1, nSub), dtype=float)
    tempChar["innov_counts"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["innov_counts_percOfInnov"] = np.zeros((nClus, nSub), dtype=float)

    tempChar["TC_norm_thes"] = [None] * nSub
    tempChar["TC_active"] = [None] * nSub
    tempChar["coactiveiCAPs_total"] = [None] * nSub

    tempChar["occurrences"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["occurrences_pos"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["occurrences_neg"] = np.zeros((nClus, nSub), dtype=float)

    tempChar["duration_total_counts"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["duration_total_pos_counts"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["duration_total_neg_counts"] = np.zeros((nClus, nSub), dtype=float)

    tempChar["duration_total_perc"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["duration_total_pos_perc"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["duration_total_neg_perc"] = np.zeros((nClus, nSub), dtype=float)

    tempChar["duration_avg_counts"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["duration_avg_pos_counts"] = np.zeros((nClus, nSub), dtype=float)
    tempChar["duration_avg_neg_counts"] = np.zeros((nClus, nSub), dtype=float)

    # coupling-related measures (nClus x nClus x nSub)
    shape_3d = (nClus, nClus, nSub)
    tempChar["coupling_counts"] = np.zeros(shape_3d, dtype=float)
    tempChar["coupling_jacc"] = np.zeros(shape_3d, dtype=float)
    tempChar["coupling_sameSign_counts"] = np.zeros(shape_3d, dtype=float)
    tempChar["coupling_diffSign_counts"] = np.zeros(shape_3d, dtype=float)
    tempChar["coupling_sameSign_jacc"] = np.zeros(shape_3d, dtype=float)
    tempChar["coupling_diffSign_jacc"] = np.zeros(shape_3d, dtype=float)
    tempChar["coupling_sameSign_perc"] = np.zeros(shape_3d, dtype=float)
    tempChar["coupling_diffSign_perc"] = np.zeros(shape_3d, dtype=float)
    # Fields that appear only in the diagonal NaN assignment in MATLAB
    tempChar["coupling_sameSign"] = np.zeros(shape_3d, dtype=float)
    tempChar["coupling_diffSign"] = np.zeros(shape_3d, dtype=float)

    tempChar["coupling"] = [None] * nSub
    tempChar["coupling_posPos_counts"] = [None] * nSub
    tempChar["coupling_posNeg_counts"] = [None] * nSub
    tempChar["coupling_negPos_counts"] = [None] * nSub
    tempChar["coupling_negNeg_counts"] = [None] * nSub

    # loop over all subjects
    for iS in range(nSub):
        subj_idx = iS  # MATLAB subject index
        vols_iS = AI_subject_labels == subj_idx
        subj_scrub = scrub_labels[vols_iS]
        tempChar["scrub_labels"][iS] = subj_scrub.astype(int)
        tempChar["nTP_sub"][iS] = int(np.count_nonzero(subj_scrub))

        # innovation counts per subject
        innov_counts_total = int(np.count_nonzero(subject_labels == subj_idx))
        tempChar["innov_counts_total"][0, iS] = innov_counts_total
        if tempChar["nTP_sub"][iS] > 0:
            tempChar["innov_counts_total_perc"][0, iS] = (
                innov_counts_total / float(tempChar["nTP_sub"][iS]) * 100.0
            )
        else:
            tempChar["innov_counts_total_perc"][0, iS] = 0.0

        # normalize time courses (global z-score)
        TC_iS = np.asarray(TC_list[iS], dtype=float)
        nClus_i, nTP_i = TC_iS.shape
        flat = TC_iS.reshape(-1, order="F")
        mean = flat.mean()
        std = flat.std(ddof=1)
        if std == 0:
            flat_z = np.zeros_like(flat)
        else:
            flat_z = (flat - mean) / std
        TC_norm_iS = flat_z.reshape(TC_iS.shape, order="F")

        # threshold
        TC_norm_thes_iS = TC_norm_iS.copy()
        TC_norm_thes_iS[np.abs(TC_norm_iS) < thres] = 0.0

        # remove occurrences of only one frame
        for iC in range(nClus):
            row = TC_norm_thes_iS[iC, :]
            activeComp = _bwconncomp_1d(row)
            activeComp = _splitPosNegComps(activeComp, row)
            row_clean = _deleteOneFrameOcc(activeComp, row)
            TC_norm_thes_iS[iC, :] = row_clean

        # number of co-active iCAPs per time point
        TC_active_iS = np.sign(TC_norm_thes_iS)
        TC_active_iS[TC_active_iS > 0] = 1
        TC_active_iS[TC_active_iS < 0] = -1

        coactive_total = np.sum(TC_active_iS != 0, axis=0).astype(float)
        # set not included time points to nan
        mask_keep = tempChar["scrub_labels"][iS] != 0
        coactive_total[~mask_keep] = np.nan

        tempChar["TC_norm_thes"][iS] = TC_norm_thes_iS
        tempChar["TC_active"][iS] = TC_active_iS
        tempChar["coactiveiCAPs_total"][iS] = coactive_total

        # allocate coupling arrays for this subject
        nTP_sub_iS = TC_norm_thes_iS.shape[1]
        coupling_iS = np.zeros((nClus, nClus, nTP_sub_iS), dtype=bool)
        posPos_iS = np.zeros_like(coupling_iS, dtype=bool)
        posNeg_iS = np.zeros_like(coupling_iS, dtype=bool)
        negPos_iS = np.zeros_like(coupling_iS, dtype=bool)
        negNeg_iS = np.zeros_like(coupling_iS, dtype=bool)

        # compute iCAP-wise characteristics
        for iC in range(nClus):
            row = TC_norm_thes_iS[iC, :]
            activeComp = _bwconncomp_1d(row)
            activeComp = _splitPosNegComps(activeComp, row)
            activeComp = _getCompSign(activeComp, row)

            # safety check: occurrences of only one frame should have been removed
            for idx_array in activeComp["PixelIdxList"]:
                if len(idx_array) < 2:
                    raise RuntimeError(
                        "Something went wrong while suppressing occurrences of only one frame"
                    )

            # compute occurrences and durations
            num_objects = activeComp["NumObjects"]
            comp_sign = activeComp.get("compSign", np.zeros(num_objects, dtype=int))

            tempChar["occurrences"][iC, iS] = num_objects
            tempChar["occurrences_pos"][iC, iS] = int(np.count_nonzero(comp_sign > 0))
            tempChar["occurrences_neg"][iC, iS] = int(np.count_nonzero(comp_sign < 0))

            total_counts = int(np.count_nonzero(row))
            pos_counts = int(np.count_nonzero(row > 0))
            neg_counts = int(np.count_nonzero(row < 0))
            tempChar["duration_total_counts"][iC, iS] = total_counts
            tempChar["duration_total_pos_counts"][iC, iS] = pos_counts
            tempChar["duration_total_neg_counts"][iC, iS] = neg_counts

            if tempChar["nTP_sub"][iS] > 0:
                tempChar["duration_total_perc"][iC, iS] = (
                    total_counts / float(tempChar["nTP_sub"][iS]) * 100.0
                )
                tempChar["duration_total_pos_perc"][iC, iS] = (
                    pos_counts / float(tempChar["nTP_sub"][iS]) * 100.0
                )
                tempChar["duration_total_neg_perc"][iC, iS] = (
                    neg_counts / float(tempChar["nTP_sub"][iS]) * 100.0
                )

            if num_objects > 0:
                tempChar["duration_avg_counts"][iC, iS] = total_counts / float(num_objects)
            if tempChar["occurrences_pos"][iC, iS] > 0:
                tempChar["duration_avg_pos_counts"][iC, iS] = (
                    pos_counts / float(tempChar["occurrences_pos"][iC, iS])
                )
            if tempChar["occurrences_neg"][iC, iS] > 0:
                tempChar["duration_avg_neg_counts"][iC, iS] = (
                    neg_counts / float(tempChar["occurrences_neg"][iC, iS])
                )

            # innovation counts for this iCAP / subject
            mask_ic = (IDX == iC) & (subject_labels == subj_idx)
            innov_counts_ic = int(np.count_nonzero(mask_ic))
            tempChar["innov_counts"][iC, iS] = innov_counts_ic
            if innov_counts_total > 0:
                tempChar["innov_counts_percOfInnov"][iC, iS] = (
                    innov_counts_ic / float(innov_counts_total) * 100.0
                )

        # compute signed co-activations with all other iCAPs
        for iC in range(nClus):
            row_i = TC_norm_thes_iS[iC, :]
            nonzero_i = row_i != 0
            pos_i = row_i > 0
            neg_i = row_i < 0

            for iC2 in range(nClus):
                row_j = TC_norm_thes_iS[iC2, :]
                nonzero_j = row_j != 0
                pos_j = row_j > 0
                neg_j = row_j < 0

                # time points of co-activation of iCAP iC and iCAP iC2
                coupling_vec = nonzero_i & nonzero_j
                coupling_iS[iC, iC2, :] = coupling_vec

                # Jaccard-like measures
                union_vec = (nonzero_i | nonzero_j)
                union_count = int(np.count_nonzero(union_vec))
                coupling_count = int(np.count_nonzero(coupling_vec))

                tempChar["coupling_counts"][iC, iC2, iS] = coupling_count
                if union_count > 0:
                    tempChar["coupling_jacc"][iC, iC2, iS] = coupling_count / float(
                        union_count
                    )
                else:
                    tempChar["coupling_jacc"][iC, iC2, iS] = np.nan

                # signed co-activation
                posPos_vec = pos_i & pos_j
                posNeg_vec = pos_i & neg_j
                negPos_vec = neg_i & pos_j
                negNeg_vec = neg_i & neg_j

                posPos_iS[iC, iC2, :] = posPos_vec
                posNeg_iS[iC, iC2, :] = posNeg_vec
                negPos_iS[iC, iC2, :] = negPos_vec
                negNeg_iS[iC, iC2, :] = negNeg_vec

                sameSign_count = int(np.count_nonzero(posPos_vec) + np.count_nonzero(negNeg_vec))
                diffSign_count = int(np.count_nonzero(posNeg_vec) + np.count_nonzero(negPos_vec))

                tempChar["coupling_sameSign_counts"][iC, iC2, iS] = sameSign_count
                tempChar["coupling_diffSign_counts"][iC, iC2, iS] = diffSign_count

                if union_count > 0:
                    tempChar["coupling_sameSign_jacc"][iC, iC2, iS] = sameSign_count / float(
                        union_count
                    )
                    tempChar["coupling_diffSign_jacc"][iC, iC2, iS] = diffSign_count / float(
                        union_count
                    )
                else:
                    tempChar["coupling_sameSign_jacc"][iC, iC2, iS] = np.nan
                    tempChar["coupling_diffSign_jacc"][iC, iC2, iS] = np.nan

                # percentage of signed co-activation with iCAP iC2, with
                # respect to total positive or negative activation of both iCAPs
                if coupling_count > 0:
                    tempChar["coupling_sameSign_perc"][iC, iC2, iS] = (
                        sameSign_count / float(coupling_count) * 100.0
                    )
                    tempChar["coupling_diffSign_perc"][iC, iC2, iS] = (
                        diffSign_count / float(coupling_count) * 100.0
                    )
                else:
                    tempChar["coupling_sameSign_perc"][iC, iC2, iS] = np.nan
                    tempChar["coupling_diffSign_perc"][iC, iC2, iS] = np.nan

                if iC == iC2:
                    tempChar["coupling_counts"][iC, iC2, iS] = np.nan
                    tempChar["coupling_jacc"][iC, iC2, iS] = np.nan

                    tempChar["coupling_sameSign"][iC, iC2, iS] = np.nan
                    tempChar["coupling_diffSign"][iC, iC2, iS] = np.nan

                    tempChar["coupling_sameSign_jacc"][iC, iC2, iS] = np.nan
                    tempChar["coupling_diffSign_jacc"][iC, iC2, iS] = np.nan

                    tempChar["coupling_sameSign_perc"][iC, iC2, iS] = np.nan
                    tempChar["coupling_diffSign_perc"][iC, iC2, iS] = np.nan

        # store per-subject coupling arrays
        tempChar["coupling"][iS] = coupling_iS
        tempChar["coupling_posPos_counts"][iS] = posPos_iS
        tempChar["coupling_posNeg_counts"][iS] = posNeg_iS
        tempChar["coupling_negPos_counts"][iS] = negPos_iS
        tempChar["coupling_negNeg_counts"][iS] = negNeg_iS

    # remove NaNs in duration_avg_*
    for key in ["duration_avg_counts", "duration_avg_pos_counts", "duration_avg_neg_counts"]:
        arr = tempChar[key]
        arr[np.isnan(arr)] = 0.0
        tempChar[key] = arr

    return tempChar
