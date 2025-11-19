import os
import numpy as np
from scipy.io import loadmat
from functions.Utilities.WriteInformation import write_information


def AggregateSubjectFrames(Paths, param, fid):
    """Python translation of AggregateSubjectFrames.m (simplified I/O).

    Combines frames from multiple subjects into a single matrix for clustering.

    Parameters
    ----------
    Paths : list of str
        Each entry is a path to a subject directory containing TA/thresholding
        results (SignInnov.mat, Activity_inducing.mat, param.mat).
    param : dict-like
        Global parameter structure; updated with 'final_mask'.
    fid : file handle or str or None
        For logging via write_information.

    Returns
    -------
    I_sig : ndarray, shape (n_frames, n_final_ret_vox)
        Concatenated significant innovation frames.
    AI : ndarray, shape (n_frames_AI, n_final_ret_vox)
        Concatenated activity-inducing signals for the same voxels.
    subject_labels : ndarray, shape (n_frames,)
        Subject index for each frame in I_sig.
    time_labels : ndarray, shape (n_frames,)
        Time index (within subject) for each frame in I_sig.
    AI_subject_labels : ndarray, shape (n_frames_AI,)
        Subject index for each frame in AI.
    final_mask : ndarray, shape (n_vox,)
        Boolean mask of voxels present in all subjects.
    """
    nSub = len(Paths)
    # These will be grown dynamically
    I_sig_list = []
    AI_list = []
    subject_labels_list = []
    time_labels_list = []
    AI_subject_labels_list = []

    final_mask = None

    for iS, subj_path in enumerate(Paths, start=1):
        threshDir = os.path.join(subj_path, "02_Thresholding")
        taDir = os.path.join(subj_path, "01_TotalActivation")

        # Load subject param and data
        p_mat = loadmat(os.path.join(threshDir, "param.mat"), squeeze_me=True, struct_as_record=False)
        subj_param = p_mat["param"]

        signinnov_mat = loadmat(os.path.join(threshDir, "SignInnov.mat"), squeeze_me=True)
        SignInnov = np.asarray(signinnov_mat["SignInnov"])

        ai_mat = loadmat(os.path.join(taDir, "Activity_inducing.mat"), squeeze_me=True)
        Activity_inducing = np.asarray(ai_mat["Activity_inducing"])

        # Orient time x vox
        if SignInnov.shape[0] > SignInnov.shape[1]:
            SignInnov = SignInnov.T
        if Activity_inducing.shape[0] > Activity_inducing.shape[1]:
            Activity_inducing = Activity_inducing.T

        nSignInnov, nVox = SignInnov.shape
        nTP = Activity_inducing.shape[0]

        # Initialize / update final_mask
        if final_mask is None:
            final_mask = np.asarray(subj_param.mask_nonan).astype(bool)
        else:
            final_mask = final_mask & np.asarray(subj_param.mask_nonan).astype(bool)

        # Append frames
        I_sig_list.append(SignInnov[:, final_mask])
        AI_list.append(Activity_inducing[:, np.asarray(subj_param.mask).astype(bool)])

        # labels
        subject_labels_list.append(np.full(nSignInnov, iS, dtype=int))
        time_labels_list.append(np.asarray(subj_param.time_labels, dtype=int))
        AI_subject_labels_list.append(np.full(nTP, iS, dtype=int))

    I_sig = np.vstack(I_sig_list)
    AI = np.vstack(AI_list)
    subject_labels = np.concatenate(subject_labels_list)
    time_labels = np.concatenate(time_labels_list)
    AI_subject_labels = np.concatenate(AI_subject_labels_list)

    param["final_mask"] = final_mask

    if fid is not None:
        write_information(fid, f"\niCAPs clustering: Loaded the data from {nSub} subjects for clustering...")
        write_information(fid, f"There are in total {I_sig.shape[1]} voxels kept for clustering...")
        write_information(fid, f"There are in total {I_sig.shape[0]} frames kept for clustering...")

    return I_sig, AI, subject_labels, time_labels, AI_subject_labels, final_mask
