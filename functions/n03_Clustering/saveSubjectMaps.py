import os
import numpy as np
from functions.Utilities.save4Dnii import save4dnii
from .ZScore_iCAPs import zscore_icaps


def save_subject_maps(param, subject_labels, IDX, I_sig, final_mask):
    """Python translation of saveSubjectMaps.m

    Saves subject-specific iCAP z-scored maps as 4D NIfTI files.

    Parameters
    ----------
    param : dict
        Must contain 'outDir_iCAPs' and 'outDir_main'.
    subject_labels : array_like, shape (n_frames,)
        Subject index for each frame. 0-based in Python pipeline.
    IDX : array_like, shape (n_frames,)
        Cluster label for each frame. 0-based (0..K-1).
    I_sig : ndarray, shape (n_frames, n_vox)
        Significant innovation frames.
    final_mask : array_like, shape (n_vox,)
        Voxel mask (boolean or 0/1).
    """
    subject_labels = np.asarray(subject_labels).astype(int)
    IDX            = np.asarray(IDX).astype(int)
    I_sig          = np.asarray(I_sig)

    # 0-based: cluster labels are 0..K-1, subject labels are 0..nSub-1
    nClus = int(IDX.max()) + 1
    nSub  = int(subject_labels.max()) + 1

    outDir_iCAPs = param["outDir_iCAPs"]
    hdrFilename  = os.path.join(param["outDir_main"], "final_mask.nii")

    os.makedirs(os.path.join(outDir_iCAPs, "subjectMaps"), exist_ok=True)

    for iC in range(nClus):      # 0-based: 0..K-1
        iCAP_sub = np.full((nSub, I_sig.shape[1]), np.nan, dtype=float)

        for iS in range(nSub):
            mask = (IDX == iC) & (subject_labels == iS)
            if np.any(mask):
                iCAP_sub[iS, :] = np.nanmean(I_sig[mask, :], axis=0)

        iCAP_sub_z = zscore_icaps(iCAP_sub)   # (nSub, n_vox)

        # MATLAB passes iCAP_sub' which is (n_vox, nSub) to save4Dnii
        fname = f"iCAP_z_{iC + 1:d}"          # keep 1-based filenames to match MATLAB
        save4dnii(
            outDir_iCAPs, "subjectMaps", fname,
            iCAP_sub_z.T,                      # transpose to (n_vox, nSub) matching MATLAB
            hdrFilename,
            mask1D=np.asarray(final_mask).astype(bool),
        )