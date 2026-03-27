import os
import numpy as np
import pickle
from functions.n00_Utilities.WriteInformation import write_information


def aggregate_subject_frames(Paths, param, fid):
    """
    Combines frames from multiple subjects into a single matrix for clustering.

    Parameters
    ----------
    Paths : list of str
        Each entry is a path to a subject directory containing TA/thresholding
        results (SignInnov.pkl, Activity_inducing.pkl, param.pkl).
    param : dict-like
        Global parameter structure; updated with 'final_mask'.
    fid : file handle or str or None
        For logging via write_information.

    Returns
    -------
    I_sig : ndarray, shape (n_frames, n_final_ret_vox)
        Concatenated significant innovation frames (masked by final_mask).
    AI : ndarray, shape (n_frames_AI, n_final_ret_vox)
        Concatenated activity-inducing signals (masked by final_mask).
    subject_labels : ndarray, shape (n_frames,)
        Subject index (0-based) for each frame in I_sig.
    time_labels : ndarray, shape (n_frames,)
        Time index (within subject) for each frame in I_sig.
    AI_subject_labels : ndarray, shape (n_frames_AI,)
        Subject index (0-based) for each frame in AI.
    final_mask : ndarray, shape (n_vox,)
        Boolean mask of voxels present in all subjects (intersection of mask_nonan).

    12.6.2018 - YF/DZ modified for different frame numbers in subjects
    """
    nSub = len(Paths)

    # These will hold FULL-length (nVox) arrays; final_mask applied at the end
    I_sig_full_list = []
    AI_full_list = []
    subject_labels_list = []
    time_labels_list = []
    AI_subject_labels_list = []

    final_mask = None

    for iS, subj_path in enumerate(Paths):
        print(iS)

        threshDir = os.path.join(subj_path, "Thresholding", param["thresh_title"])
        taDir = os.path.join(subj_path, "TotalActivation")

        # Check existence of required files
        param_path = os.path.join(threshDir, "param.pkl")
        signinnov_path = os.path.join(threshDir, "SignInnov.pkl")
        activity_path = os.path.join(taDir, "Activity_inducing.pkl")

        if not (os.path.isfile(param_path) and os.path.isfile(signinnov_path) and os.path.isfile(activity_path)):
            raise FileNotFoundError(
                f"Cannot access data from subject {subj_path}; clustering process aborted..."
            )

        # ---- Load subject param and data from .pkl ----
        with open(param_path, "rb") as f:
            subj_param = pickle.load(f)

        with open(signinnov_path, "rb") as f:
            SignInnov = np.asarray(pickle.load(f))

        with open(activity_path, "rb") as f:
            Activity_inducing = np.asarray(pickle.load(f))

        # Orient time x vox (MATLAB checks and transposes similarly)
        if SignInnov.shape[0] > SignInnov.shape[1]:
            # warning-equivalent, but just do it silently unless you want prints
            SignInnov = SignInnov.T
        if Activity_inducing.shape[0] > Activity_inducing.shape[1]:
            Activity_inducing = Activity_inducing.T

        # Subject-specific sizes
        nTP = len(subj_param["mask_threshold2pos"])      # number of time points
        nSignInnov = len(subj_param["time_labels"])      # number of significant innovation frames
        nVox = len(subj_param["mask"])                   # number of voxels

        # MATLAB: final_mask = logical(final_mask & param.mask_nonan)
        mask_nonan = np.asarray(subj_param["mask_nonan"], dtype=bool).ravel()
        if final_mask is None:
            final_mask = mask_nonan.copy()
        else:
            final_mask &= mask_nonan

        # Build full nVox matrices like MATLAB:
        #   I_sig_full[:, mask_nonan] = SignInnov
        #   AI_full[:, mask] = Activity_inducing
        I_sig_full = np.zeros((nSignInnov, nVox), dtype=SignInnov.dtype)
        I_sig_full[:, mask_nonan] = SignInnov

        mask = np.asarray(subj_param["mask"], dtype=bool).ravel()
        AI_full = np.zeros((nTP, nVox), dtype=Activity_inducing.dtype)
        AI_full[:, mask] = Activity_inducing

        I_sig_full_list.append(I_sig_full)
        AI_full_list.append(AI_full)

        # Labels (1-based subject index like MATLAB)
        subject_labels_list.append(np.full(nSignInnov, iS, dtype=int))
        # time_labels_list.append(np.asarray(subj_param["time_labels"], dtype=int))
        time_labels_list.append(np.asarray(subj_param["time_labels"], dtype=int).ravel())
        AI_subject_labels_list.append(np.full(nTP, iS, dtype=int))

        del subj_param, SignInnov, Activity_inducing, I_sig_full, AI_full

    # Stack all subjects (still full nVox)
    I_sig_full = np.vstack(I_sig_full_list)
    AI_full = np.vstack(AI_full_list)
    subject_labels = np.concatenate(subject_labels_list)
    time_labels = np.concatenate(time_labels_list)
    AI_subject_labels = np.concatenate(AI_subject_labels_list)

    # Apply final_mask to get final voxel set (like MATLAB's final step)
    I_sig = I_sig_full[:, final_mask]
    AI = AI_full[:, final_mask]

    param["final_mask"] = final_mask

    if fid is not None:
        write_information(
            fid,
            f"iCAPs clustering: Loaded the data from {nSub} subjects for clustering...",
        )
        write_information(
            fid,
            f"There are in total {I_sig.shape[1]} voxels kept for clustering...",
        )
        write_information(
            fid,
            f"There are in total {I_sig.shape[0]} frames kept for clustering...",
        )

    return I_sig, AI, subject_labels, time_labels, AI_subject_labels, final_mask
