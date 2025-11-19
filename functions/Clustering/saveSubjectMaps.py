import os
import numpy as np
from functions.Utilities.save4Dnii import save4dnii
from .ZScore_iCAPs import ZScore_iCAPs


def saveSubjectMaps(param, subject_labels, IDX, I_sig, final_mask):
    """Python translation of saveSubjectMaps.m

    Saves subject-specific iCAP maps as 4D NIfTI files.

    Parameters
    ----------
    param : dict-like
        Must contain:
        - 'outDir_iCAPs': output directory for iCAPs
        - 'outDir_main': directory containing 'final_mask.nii'
    subject_labels : array_like, shape (n_frames,)
        Subject index (1..nSub) for each frame.
    IDX : array_like, shape (n_frames,)
        Cluster label (1..nClus) for each frame.
    I_sig : ndarray, shape (n_frames, n_vox)
        Significant innovation frames.
    final_mask : array_like, shape (n_vox,)
        Voxel mask (boolean or 0/1).
    """
    subject_labels = np.asarray(subject_labels).astype(int)
    IDX = np.asarray(IDX).astype(int)
    I_sig = np.asarray(I_sig)

    nClus = int(IDX.max())
    nSub = int(subject_labels.max())

    outDir_iCAPs = param['outDir_iCAPs']
    hdrFilename = os.path.join(param['outDir_main'], 'final_mask.nii')

    for iC in range(1, nClus + 1):
        # average map per subject
        iCAP_sub = np.full((nSub, I_sig.shape[1]), np.nan, dtype=float)
        for iS in range(1, nSub + 1):
            mask = (IDX == iC) & (subject_labels == iS)
            if np.any(mask):
                iCAP_sub[iS - 1, :] = np.nanmean(I_sig[mask, :], axis=0)
        iCAP_sub_z = ZScore_iCAPs(iCAP_sub)

        fname = f"iCAP_z_{iC:02d}_subjects"
        save4dnii(outDir_iCAPs, "subjectMaps", fname, iCAP_sub_z,
                  hdrFilename, mask1D=np.asarray(final_mask).astype(bool))
