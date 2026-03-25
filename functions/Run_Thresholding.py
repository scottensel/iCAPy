import os
import copy
import pickle
from datetime import datetime

from functions.Utilities.WriteInformation import write_information
from functions.Utilities.check_ta_files import check_ta_files
from functions.Thresholding.RemoveNan import RemoveNan
from functions.Thresholding.ComputeSurrogatePercentiles import ComputeSurrogatePercentiles
from functions.Thresholding.SelectSignificantFrames import SelectSignificantFrames
from functions.Utilities.save4Dnii import save4dnii

import h5py
import numpy as np
# %% This function runs thresholding on innovations, according to the
# % parameters specified by the user
# %
# % Input:
# %   param - struct containing all necessary parameters to run TA
# %       .PathData - path to data
# %       .Subjects - cell array with list of subdirectories where each
# %           subject's fMRI data is stored; must contain one entry per
# %           subject to analyze.
# %           This is where the TA folder will be created (or looked for)
# %           for each subject.
# %       .n_subjects - number of subject to analyze
# %       [.title] - possibility to define a title for the current
# %           project (usefull if TA should be run for different
# %           parameters), default: current date
# %       [.force_Thresholding] - if set to 1, Thresholding will be forced to
# %           run, even if already has been done
# %       .alpha - Alpha-level at which to look for significance of
# %           innovation signal frames (first element is the percentile of
# %           the lower threshold - negative innovations - and second element
# %           the upper threshold one - positive innovations)
# %       .f_voxels - Fraction of voxels from the ones entering total
# %           activation for a given subject that should show an innovation
# %           at the same time point, so that the corresponding frame is
# %           retained for iCAPs clustering
# %       .thresh_title - Title used to create the folder where thresholding
# %           data will be saved
# %       .threshold_minclussize - Number of neighbours that must also show
# %           an innovation for a voxel to be retained
# %       .threshold_interconnectivity - Number of neighbours to consider in
# %           the process
# %
# % Output:
# %       Cycles through the subjects and thresholds their data:
# %       1) temporal threshold: takes the innovation signals generated from
# %       the surrogate data, builds a distribution from them, samples the
# %       X-th percentiles from the distribution at each voxel mask temporal
# %       frames from the real data with this
# %       2) spatial threshold: mark frames where X percent of significant
# %       voxels

def run_thresholding(param):
    """
    Python equivalent of Run_Thresholding.m.

    Parameters
    ----------
    param : dict
        See original MATLAB header for full description.
    """

    # Date and time when the routines are called
    param["date"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if not param.get("title"):
        param["title"] = param["date"]

    # Parameters initially entered by the user (restored inside the loop)
    param_CI = copy.deepcopy(param)

    # Log file
    ta_logs_dir = os.path.join(param["PathData"], "TAlogs")
    os.makedirs(ta_logs_dir, exist_ok=True)
    log_path = os.path.join(ta_logs_dir, f"log_Thresholding_{param['title']}.txt")
    fid = open(log_path, "a+", encoding="utf-8")

    write_information(
        fid,
        f"\nStarting the total activation/iCAPs tools for project entitled {param['title']}",
    )

    # Check data path
    if not os.path.isdir(param["PathData"]):
        write_information(fid, "Incorrect path towards the data: execution stopped")
        fid.close()
        raise FileNotFoundError(
            "The data folder that you specified does not exist! Please check and restart running..."
        )

    write_information(fid, "Entering the thresholding process...")

    # Loop over all subjects
    for i_s in range(param["n_subjects"]):
        subj_name = param["Subjects"][i_s]
        subj_path = os.path.join(param["PathData"], subj_name)
        write_information(fid, f"Analyzing subject {subj_path}...")

        if not os.path.isdir(subj_path):
            write_information(fid, f"Incorrect subject path {subj_path}: ignored subject")
            del param
            param = copy.deepcopy(param_CI)
            continue

        # TA results path for this subject
        results_path = os.path.join(subj_path, "TA_results", param["title"])

        # Check TA + thresholding flags
        ta_real_done, ta_surrogate_done, thresholding_done = check_ta_files(
            results_path
        )
        if (not ta_real_done) or (ta_surrogate_done != 1):
            write_information(
                fid,
                "No total activation results, run TA routine first! skipping...\n",
            )
            del param
            param = copy.deepcopy(param_CI)
            continue

        if param.get("force_Thresholding"):
            thresholding_done = 0

        if not thresholding_done:
            # ---- Load data -------------------------------------------------
            write_information(fid, "Loading total activation data...")

            # Real innovations
            with open(
                os.path.join(results_path, "TotalActivation", "Innovation.pkl"), "rb"
            ) as f:
                innovation = pickle.load(f)

            # TA param
            with open(
                os.path.join(results_path, "TotalActivation", "param.pkl"), "rb"
            ) as f:
                param_ta = pickle.load(f)

            # Surrogate innovations
            with open(
                os.path.join(results_path, "Surrogate", "innovation_surrogate.pkl"), "rb"
            ) as f:
                innovation_surrogate = pickle.load(f)

            # Copy TA fields into current param
            param["fHeader"] = param_ta["fHeader"]
            param["mask"] = param_ta["mask"]
            param["Dimension"] = param_ta["Dimension"]
            if "TemporalMask" in param_ta:
                param["TemporalMask"] = param_ta["TemporalMask"]
            del param_ta

            # ---- Remove NaNs -----------------------------------------------
            write_information(fid, "Removing NaNs from innovations...")
            innovation_surrogate, param["mask_nonan"], param["mask2_nonan"] = RemoveNan(
                innovation_surrogate, param, fid
            )

            # ---- Percentiles from surrogate --------------------------------
            write_information(fid, "Computing percentiles...")
            param["PC"] = ComputeSurrogatePercentiles(innovation_surrogate, param, fid)

            # with h5py.File(os.path.join("F:/iCAP/Data/Matlab/Ketamine/Khali/", param['Subjects'][i_s],
            #                             'TA_results/test3/TotalActivation/Innovation.mat'), "r") as f:
            #     innovation = np.array(f["Innovation"])
            # innovation = innovation.T

            # ---- Select significant innovation frames ----------------------
            write_information(fid, "Selecting significant innovation frames...")
            # MATLAB: Innovation(:, param.mask2_nonan)
            sign_innov, param = SelectSignificantFrames(
                innovation[:, param["mask2_nonan"]], param, fid
            )

            # ---- Saving ----------------------------------------------------
            write_information(fid, "Saving...")

            thresh_dir = os.path.join(results_path, "Thresholding", param["thresh_title"])
            os.makedirs(thresh_dir, exist_ok=True)

            # NIfTI outputs (same folder layout as MATLAB)
            subfolder = os.path.join("Thresholding", param["thresh_title"])
            save4dnii(
                results_path,
                subfolder,
                "SignInnov",
                sign_innov.T,
                param["fHeader"].fname,
                param["mask"],
                param["Dimension"],
            )

            save4dnii(
                results_path,
                subfolder,
                "mask_nonan",
                param["mask_nonan"],
                param["fHeader"].fname,
                param["mask"],
                param["Dimension"],
            )

            # Save each object as a .pkl file
            with open(os.path.join(thresh_dir, 'SignInnov.pkl'), 'wb') as f:
                pickle.dump(sign_innov, f)

            with open(os.path.join(thresh_dir, 'param.pkl'), 'wb') as f:
                pickle.dump(param, f)

            # make a dir just to save the thresholding data seperately so we
            # can download this faster to look at
            thresholdPath = os.path.join(param["PathData"], "Thresholding", param["title"], subj_name, param["title"], param["thresh_title"])
            os.makedirs(thresholdPath, exist_ok=True)

            with open(os.path.join(thresholdPath, 'param.pkl'), 'wb') as f:
                pickle.dump(param, f)

            write_information(
                fid, f"Finished running thresholding for subject {subj_path}...\n"
            )

        else:
            write_information(fid, "Thresholding already done, skipping...\n")

        # Reset param to original
        del param
        param = copy.deepcopy(param_CI)

    fid.close()
