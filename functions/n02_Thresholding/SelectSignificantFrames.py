import numpy as np
from functions.n02_Thresholding.ThresholdTimeCourse import ThresholdTimeCourse
from functions.n02_Thresholding.ThresholdWholeBrain import ThresholdWholeBrain
from functions.n02_Thresholding.check_interconnectedness import check_interconnectedness
from functions.n00_Utilities.WriteInformation import write_information


def SelectSignificantFrames(Innovation, param, fid):
    """
    Python translation of SelectSignificantFrames.m

    Performs the double thresholding process:
    1) assess the moments of an innovation time course with significant
       excursions (voxel-wise, temporal thresholding);
    2) assess the time points with sufficient such excursions brain-wise
       (spatial thresholding).

    Parameters
    ----------
    Innovation : ndarray, shape (n_tp, n_ret_vox)
        Innovation time courses for the considered subject (real data).
    param : dict-like
        Contains TA / iCAPs parameters and will be updated with new keys:
        - 'PC'              : (2, n_ret_vox) percentile matrix
        - 'f_voxels'        : fraction of voxels that must be significant
        - 'mask_threshold1' : (n_tp, n_ret_vox) matrix (-1,0,1)
        - 'mask_threshold2pos', 'mask_threshold2neg' : boolean (n_tp,)
        - 'time_labels'     : 1D integer array of selected time indices
    fid : file handle or str or None
        Passed to write_information. If None, logging is skipped.

    Returns
    -------
    SignInnov : ndarray, shape (n_significant_excursions, n_ret_vox)
        Matrix containing the frames that showed significant innovations.
        First all positive-innovation frames, then all negative-innovation
        frames (with sign flipped to be positive).
    param : dict-like
        Updated parameter dictionary.
    """
    Innovation = np.asarray(Innovation)

    # 1) Temporal thresholding, voxel-wise
    # Will be a matrix specifying the time points with significant innovations
    # (-1 for negative excursion, 0 for no excursion, +1 for positive excursion)
    param['mask_threshold1'] = ThresholdTimeCourse(Innovation, np.asarray(param['PC']))

    # Optionally: the MATLAB code contains commented logging here; we skip it.

    # 2) Spatial thresholding: how many voxels must show significance at once
    n_tp, n_ret_vox = Innovation.shape
    f_voxels = float(param['f_voxels'])
    n_voxels = int(np.floor(n_ret_vox * f_voxels))

    if fid is not None:
        write_information(
            fid,
            f"Spatial thresholding: there are {n_voxels} voxels that must show significance together..."
        )

    # Masks for positive / negative innovations
    mask_threshold1 = np.asarray(param['mask_threshold1'])

    # Positive innovations: keep +1, set negatives to 0
    mask_threshold1pos = mask_threshold1.copy()
    mask_threshold1pos[mask_threshold1pos < 0] = 0

    # Negative innovations: keep -1, set positives to 0
    mask_threshold1neg = mask_threshold1.copy()
    mask_threshold1neg[mask_threshold1neg > 0] = 0

    # Remove too small 3D clusters (interconnectedness check)
    mask_threshold1bpos = check_interconnectedness(mask_threshold1pos, param)
    mask_threshold1bneg = check_interconnectedness(mask_threshold1neg, param)

    # 3) Whole-brain thresholding: select time points with enough significant voxels
    param['mask_threshold2pos'] = ThresholdWholeBrain(mask_threshold1bpos, n_voxels)
    param['mask_threshold2neg'] = ThresholdWholeBrain(mask_threshold1bneg, n_voxels)

    if fid is not None:
        n_pos_frames = int(np.sum(param['mask_threshold2pos']))
        n_neg_frames = int(np.sum(param['mask_threshold2neg']))

        write_information(
            fid,
            f"There are {n_pos_frames} out of {n_tp} frames selected for positive innovation..."
        )
        write_information(
            fid,
            f"There are {n_neg_frames} out of {n_tp} frames selected for negative innovation..."
        )

    # 4) Build the final matrix of significant innovations
    # Positive innovation frames: negative elements removed
    SignInnovPos = Innovation[param['mask_threshold2pos'], :].copy()
    SignInnovPos[SignInnovPos < 0] = 0

    # Negative innovation frames: sign flipped, positive elements removed
    SignInnovNeg = -Innovation[param['mask_threshold2neg'], :].copy()
    SignInnovNeg[SignInnovNeg < 0] = 0

    # Final data matrix: positive frames followed by negative frames
    if SignInnovPos.size == 0 and SignInnovNeg.size == 0:
        SignInnov = np.zeros((0, n_ret_vox), dtype=Innovation.dtype)
    elif SignInnovPos.size == 0:
        SignInnov = SignInnovNeg
    elif SignInnovNeg.size == 0:
        SignInnov = SignInnovPos
    else:
        SignInnov = np.vstack((SignInnovPos, SignInnovNeg))

    # Time labels: indices (0-based in Python) of the kept frames
    pos_indices = np.where(param['mask_threshold2pos'])[0]
    neg_indices = np.where(param['mask_threshold2neg'])[0]
    param['time_labels'] = np.concatenate((pos_indices, neg_indices))

    return SignInnov, param
