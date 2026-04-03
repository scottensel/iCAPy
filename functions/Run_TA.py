import numpy as np
from datetime import datetime
from functions.n00_Utilities.WriteInformation import write_information
from functions.n00_Utilities.check_ta_files import check_ta_files
from functions.n00_Utilities.ReadTAData import read_ta_data
from functions.n00_Utilities.CreateTAMask import create_ta_mask
from functions.n00_Utilities.CreateTAData import create_ta_data
from functions.n00_Preprocessing.AssessMotion import assess_motion
from functions.n00_Preprocessing.InterpolateTimeCourses import interpolate_time_courses
from functions.n00_Preprocessing.DetrendTimeCourses import detrend_time_courses
from functions.n01_TotalActivation.Temporal_TA.hrf_filters import hrf_filters
from functions.n01_TotalActivation.RunTotalActivation import run_total_activation
from functions.n01_TotalActivation.Generate_Innovations import generate_innovations
from functions.n01_TotalActivation.GenerateSurrogate import generate_surrogate
from functions.n00_Utilities.save4Dnii import save4dnii
import time
import os
import pickle
import copy
import h5py

# This function runs total activation, according to the
# parameters specified by the user
#
# Input:
#   param - dict containing all necessary parameters to run TA
#       'PathData'        - path to data
#       'TR'              - TR of the fMRI data
#       'Subjects'        - list of subdirectories where each subject's
#                           fMRI data is stored; must contain one entry per
#                           subject to analyze.
#                           This is where the TA folder will be created
#                           (or looked for) for each subject.
#       'n_subjects'      - number of subjects to analyze
#       ['title']         - possibility to define a title for the current
#                           project (useful if TA should be run for
#                           different parameters), default: current date
#       ['force_TA_on_real']      - if set to 1, TA will be forced to run,
#                                   even if already has been done
#       ['force_TA_on_surrogate'] - if set to 1, TA on surrogate data will
#                                   be forced to run, even if already done
#       'Folder_functional'  - name of the functional folder; None if
#                              directly lying in PathData
#       'TA_func_prefix'     - string with the prefix for functional data
#                              to read
#       'Folder_GM'          - name of the folder with the probabilistic
#                              gray matter map
#       'TA_GM_prefix'       - string with the prefix of the probabilistic
#                              map to read
#       'T_gm'               - threshold probability for creating GM mask
#       ['is_morpho']        - if 1, morphological operations (opening and
#                              closure) will be run on the GM mask to
#                              remove holes
#       ['n_morpho_voxels']  - required if 'is_morpho' is set, size of
#                              morphological operators
#       'skipped_scans'      - number of fMRI scans to skip at the
#                              beginning
#       'doDetrend'          - select if detrending should be done
#       ['DCT_TS']           - required if 'doDetrend' is set, cut-off
#                              period for the DCT basis
#       ['Covariates']       - required if 'doDetrend' is set, covariates
#                              to add
#       'doScrubbing'        - select if motion censoring should be done
#       ['Folder_motion']    - required if 'doScrubbing' is set, folder
#                              with motion data from SPM realignment
#       ['TA_mot_prefix']    - required if 'doScrubbing' is set, prefix of
#                              motion data text file
#       ['skipped_scans_motionfile'] - number of lines to ignore at the
#                              beginning of the motion file; if empty or
#                              not set, defaults to param['skipped_scans']
#       ['FD_method']        - required if 'doScrubbing' is set; for now
#                              only 'Power' is implemented
#       ['FD_threshold']     - required if 'doScrubbing' is set, scrubbing
#                              threshold in mm
#       ['interType']        - interpolation method (see also interp1),
#                              default is 'spline'
#
# Output:
#   Creates a folder TA_results/<title> in each subject's folder and
#   saves results from total activation routine in subfolders:
#       - inputData:       data after preprocessing
#       - TotalActivation: results after running total activation on real
#                          data (checkpoint: TotalActivation/ta_checkpoint.h5)
#       - Surrogate:       results after running total activation on
#                          surrogate data (checkpoint: Surrogate/ta_checkpoint.h5)
#
# Checkpointing:
#   Both the real and surrogate TA runs save a rolling checkpoint file
#   (ta_checkpoint.h5) in their respective output subfolders. Each new
#   checkpoint overwrites the previous one. The file is deleted automatically
#   on successful completion. On restart, the pipeline detects the checkpoint
#   and resumes from the next iteration — no manual steps needed.
#   Checkpoint frequency is controlled by param['checkpoint_every'] in
#   Inputs_TA.py (1 = after every iteration, 0 = disabled).


def run_ta(param):

    # Set date and title for the project
    param['date'] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if 'title' not in param or not param['title']:
        param['title'] = param['date']

    # Will contain the parameters initially entered by the user, prior to any
    # change within the loop
    param_CI = copy.deepcopy(param)

    # Create log directory if it doesn't exist
    log_path = os.path.join(param['PathData'], 'TAlogs')
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, f'log_TA_{param["title"]}.txt')
    with open(log_file, 'a+') as fid:
        write_information(fid, f'\nStarting the total activation/iCAPs tools for project entitled {param["title"]}')

        # checks if path to data is correct
        if not os.path.isdir(param['PathData']):
            write_information(fid, 'Incorrect path towards the data: execution stopped')
            raise ValueError('The data folder that you specified does not exist!')

        write_information(fid, 'Entering the total activation part of the routines...')

        for i_TA in range(param['n_subjects']):

            # try except block so if one fails the rest can still go
            try:
                # Set up paths for subject-specific data
                subj_path_ta = os.path.join(param['PathData'], param['Subjects'][i_TA])
                write_information(fid, f'Analyzing subject {subj_path_ta}...')
                if not os.path.isdir(subj_path_ta):
                    write_information(fid, f'Incorrect subject path {subj_path_ta}: ignored subject')
                    continue

                # creates the path to the data
                results_path = os.path.join(subj_path_ta, 'TA_results', param['title'])
                os.makedirs(results_path, exist_ok=True)

                # Check if TA has already been executed
                ta_real_done, ta_surrogate_done, thresholding_done = check_ta_files(results_path)
                if param.get('force_TA_on_real', False):
                    ta_real_done = False
                if param.get('force_TA_on_surrogate', False):
                    ta_surrogate_done = False

                if not ta_real_done or ta_surrogate_done != 1:
                    # Read and preprocess subject data
                    fData, pData, fHeader, _ = read_ta_data(subj_path_ta, i_TA, param, fid)
                    param['GM_map'] = pData

                    fData = fData[..., param['skipped_scans']:]  # Discard initial volumes
                    write_information(fid, f'Discarding the first {param["skipped_scans"]} volumes from the time courses...')

                    # Dimension of data (X x Y x Z x T)
                    param['Dimension'] = fData.shape
                    param['fHeader'] = fHeader

                    #  Creates the mask that will be used for the analysis: we want to keep
                    #  only the brain information
                    param['mask'], param['mask_3D'] = create_ta_mask(param, fid)
                    write_information(fid, f'Saving fMRI 4D input (fData)...')
                    save4dnii(results_path, 'inputData', 'fData', fData, param['fHeader'].fname, param['mask'], param['Dimension'])

                    # Create 2D time-course data
                    TC, param = create_ta_data(fData, param, fid)
                    write_information(fid, f'Saving fMRI 2D input (TC)...')
                    save4dnii(results_path, 'inputData', 'TC1', TC, param['fHeader'].fname, param['mask'], param['Dimension'])

                    # Motion Analysis (if enabled)
                    #######################################
                    if param.get('doScrubbing', False):
                        param['TemporalMask'] = assess_motion(subj_path_ta, i_TA, param, fid)
                        if not np.all(param['TemporalMask']):
                            TC, param['TemporalMask'] = interpolate_time_courses(TC, param['TemporalMask'], param, fid)
                        else:
                            write_information(fid, f'No interpolation done for {subj_path_ta}...')

                    # Detrending (if enabled)
                    #######################################
                    if param.get('doDetrend', False):
                        TC = detrend_time_courses(TC, param, fid)

                    # Convert param['Dimension'] to a list to allow modification
                    dimension_list = list(param['Dimension'])
                    dimension_list[3] = TC.shape[1]  # Modify the element as needed

                    # Reassign the modified list back to param['Dimension'] as a tuple
                    param['Dimension'] = tuple(dimension_list)

                    # Save preprocessed input data
                    write_information(fid, f'Saving preprocessed fMRI 4D input (TC)...')
                    save4dnii(results_path, 'inputData', 'TC2', TC, param['fHeader'].fname, param['mask'],
                               param['Dimension'])

                    # Total Activation Start
                    ##################################################
                    # Creates the filters required: the one that 'deconvolves the BOLD
                    # signal and derivates it' (analyze), and the one that 'deconvolves
                    # the BOLD-like signal into neural activity' (reconstruct).
                    param = hrf_filters(param)

                    # Save the param state before TA runs so the surrogate gets exactly
                    # the same inputs (filters, dimensions, mask, etc.) as the real run
                    param_tmp = copy.deepcopy(param)

                    # ── Total Activation for Real Data ────────────────────────────
                    if not ta_real_done:
                        # Set outDir_TA so run_total_activation knows where to save
                        # the rolling checkpoint (TotalActivation/ta_checkpoint.h5).
                        # The checkpoint is deleted inside run_total_activation on
                        # successful completion — it never conflicts with the final
                        # output files (Activity_related.pkl etc.) saved below.
                        param['outDir_TA'] = os.path.join(results_path, 'TotalActivation')
                        os.makedirs(param['outDir_TA'], exist_ok=True)

                        start_time = time.perf_counter()
                        activity_related, param = run_total_activation(TC.T, param, fid)
                        elapsed_time = time.perf_counter() - start_time
                        write_information(fid, f'It took {elapsed_time:.2f} seconds to run total activation on real data...')





                        innovation, activity_inducing = generate_innovations(activity_related, param)

                        # Save results for real data
                        write_information(fid, 'Saving total activation results (mat and nifti)...')
                        save4dnii(results_path, 'TotalActivation', 'Activity_inducing', activity_inducing.T,
                                   param['fHeader'].fname, param['mask'], param['Dimension'])
                        save4dnii(results_path, 'TotalActivation', 'Activity_related', activity_related.T,
                                   param['fHeader'].fname, param['mask'], param['Dimension'])
                        save4dnii(results_path, 'TotalActivation', 'Innovation', innovation.T, param['fHeader'].fname,
                                   param['mask'], param['Dimension'])

                        with open(os.path.join(results_path, 'TotalActivation', 'Activity_inducing.pkl'), 'wb') as f:
                            pickle.dump(activity_inducing, f)
                        with open(os.path.join(results_path, 'TotalActivation', 'Activity_related.pkl'), 'wb') as f:
                            pickle.dump(activity_related, f)
                        with open(os.path.join(results_path, 'TotalActivation', 'Innovation.pkl'), 'wb') as f:
                            pickle.dump(innovation, f)
                        with open(os.path.join(results_path, 'TotalActivation', 'param.pkl'), 'wb') as f:
                            pickle.dump(param, f)

                        del innovation, activity_inducing, activity_related

                    elif ta_real_done:
                        write_information(fid, 'Total activation on real data already computed, skipping...')

                    # ── Total Activation for Surrogate Data ───────────────────────
                    if not ta_surrogate_done:
                        surrogate = generate_surrogate(TC, subj_path_ta, param, fid)

                        save4dnii(results_path, 'Surrogate', 'Surrogate', surrogate, param['fHeader'].fname,
                                  param['mask'], param['Dimension'])
                        surrogate = surrogate.T

                        # Set outDir_TA on param_tmp so run_total_activation saves
                        # the rolling checkpoint to Surrogate/ta_checkpoint.h5.
                        # This is a different subfolder from TotalActivation/ so
                        # there is no conflict with the real TA checkpoint or its
                        # final output files. The checkpoint is deleted inside
                        # run_total_activation on successful completion.
                        param_tmp['outDir_TA'] = os.path.join(results_path, 'Surrogate')
                        os.makedirs(param_tmp['outDir_TA'], exist_ok=True)

                        start_time = time.perf_counter()
                        activity_related_surrogate = run_total_activation(surrogate, param_tmp, fid)[0]
                        elapsed_time = time.perf_counter() - start_time
                        write_information(fid, f'It took {elapsed_time:.2f} seconds to run total activation on surrogate data...')

                        innovation_surrogate, activity_inducing_surrogate = generate_innovations(
                            activity_related_surrogate, param)

                        # Save results for surrogate data
                        write_information(fid, 'Saving total activation results of surrogate data (mat and nifti)...')
                        save4dnii(results_path, 'Surrogate', 'Activity_inducing_surrogate', activity_inducing_surrogate.T,
                                   param['fHeader'].fname, param['mask'], param['Dimension'])
                        save4dnii(results_path, 'Surrogate', 'Activity_related_surrogate', activity_related_surrogate.T,
                                   param['fHeader'].fname, param['mask'], param['Dimension'])
                        save4dnii(results_path, 'Surrogate', 'Innovation_surrogate', innovation_surrogate.T,
                                   param['fHeader'].fname, param['mask'], param['Dimension'])

                        with open(os.path.join(results_path, 'Surrogate', 'Activity_inducing_surrogate.pkl'), 'wb') as f:
                            pickle.dump(activity_inducing_surrogate, f)
                        with open(os.path.join(results_path, 'Surrogate', 'Activity_related_surrogate.pkl'), 'wb') as f:
                            pickle.dump(activity_related_surrogate, f)
                        with open(os.path.join(results_path, 'Surrogate', 'Innovation_surrogate.pkl'), 'wb') as f:
                            pickle.dump(innovation_surrogate, f)
                        with open(os.path.join(results_path, 'Surrogate', 'param.pkl'), 'wb') as f:
                            pickle.dump(param, f)

                        del innovation_surrogate, activity_inducing_surrogate, activity_related_surrogate, surrogate

                    elif ta_surrogate_done:
                        write_information(fid, 'Total activation on surrogate data already computed, skipping...')

                write_information(fid, f'Finished running total activation for subject {subj_path_ta}...')

            except Exception as e:

                write_information(fid, f'ERROR processing subject {param["Subjects"][i_TA]}: {type(e).__name__}: {e}')
                import traceback
                write_information(fid, traceback.format_exc())
                continue

            finally:
                # Always reset param for the next subject even if this one failed
                # Resets the parameters to what they were at the start of the loop
                # (before any subject-specific change could have been made)
                del param
                param = copy.deepcopy(param_CI)
