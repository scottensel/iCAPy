import os
import numpy as np
from datetime import datetime
from functions.Utilities.WriteInformation import write_information
from functions.Utilities.check_ta_files import check_ta_files
from functions.Utilities.ReadTAData import read_ta_data
from functions.Utilities.CreateTAMask import create_ta_mask
from functions.Utilities.CreateTAData import create_ta_data
from functions.Preprocessing.AssessMotion import assess_motion
from functions.Preprocessing.InterpolateTimeCourses import interpolate_time_courses
from functions.Preprocessing.DetrendTimeCourses import detrend_time_courses
from functions.TotalActivation.Temporal_TA.hrf_filters import hrf_filters
from functions.TotalActivation.RunTotalActivation import run_total_activation
from functions.TotalActivation.Generate_Innovations import generate_innovations
from functions.TotalActivation.GenerateSurrogate import generate_surrogate
from functions.Utilities.save4Dnii import save4dnii
import time
import os
import pickle
import copy
import h5py

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
        write_information(fid, f'Starting the total activation/iCAPs tools for project entitled {param["title"]}')

        if not os.path.isdir(param['PathData']):
            write_information(fid, 'Incorrect path towards the data: execution stopped')
            raise ValueError('The data folder that you specified does not exist!')

        write_information(fid, 'Entering the total activation part of the routines...')

        for i_TA in range(param['n_subjects']):
            # Set up paths for subject-specific data
            subj_path_ta = os.path.join(param['PathData'], param['Subjects'][i_TA])
            write_information(fid, f'Analyzing subject {subj_path_ta}...')
            if not os.path.isdir(subj_path_ta):
                write_information(fid, f'Incorrect subject path {subj_path_ta}: ignored subject')
                continue

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
                param['Dimension'] = fData.shape
                param['fHeader'] = fHeader

                # Generate mask
                param['mask'], param['mask_3D'] = create_ta_mask(param, fid)
                write_information(fid, f'Saving fMRI 4D input (fData)...')
                save4dnii(results_path, 'inputData', 'fData', fData, param['fHeader'].fname, param['mask'], param['Dimension'])

                # Create 2D time-course data
                TC, param = create_ta_data(fData, param, fid)
                write_information(fid, f'Saving fMRI 2D input (TC)...')
                save4dnii(results_path, 'inputData', 'TC1', TC, param['fHeader'].fname, param['mask'], param['Dimension'])

                # Motion Analysis (if enabled)
                if param.get('doScrubbing', False):
                    param['TemporalMask'] = assess_motion(subj_path_ta, i_TA, param, fid)
                    if not np.all(param['TemporalMask']):
                        TC, param['TemporalMask'] = interpolate_time_courses(TC, param['TemporalMask'], param, fid)
                    else:
                        write_information(fid, f'No interpolation done for {subj_path_ta}...')

                # Detrending (if enabled)
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
                # The 'analyze' and 'reconstruct' filter cases are
                # different: 'analyze' has an added zero, which means, an additional
                # derivative (probably reflects the fact that we are working with
                # sparsity imposed at the level of the innovation signal, the
                # derivative of the piece-wise constant neural activity)

                # Creates the filters required: the one that 'deconvolves the BOLD
                # signal and derivates it' (analyze), and the one that 'deconvolves
                # the BOLD-like signal into neural activity' (reconstruct). The
                # matlab variables contain the non-null values of the filter
                # coefficients, from sample f[n] = f[0]
                param = hrf_filters(param)

                # The param vecotr is updated within the total activation scheme; I want to give
                # exactly the saem input for surrogate and the real, si i save the state of the param
                # piror to TA
                param_tmp = copy.deepcopy(param)

                # Total Activation for Real Data
                if not ta_real_done:
                    start_time = time.perf_counter()

                    activity_related, param = run_total_activation(TC.T, param)

                    end_time = time.perf_counter()

                    elapsed_time = end_time - start_time

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

                    # Save each object as a .pkl file
                    with open(os.path.join(results_path, 'TotalActivation', 'Activity_inducing.pkl'), 'wb') as f:
                        pickle.dump(activity_inducing, f)

                    with open(os.path.join(results_path, 'TotalActivation', 'Activity_related.pkl'), 'wb') as f:
                        pickle.dump(activity_related, f)

                    with open(os.path.join(results_path, 'TotalActivation', 'Innovation.pkl'), 'wb') as f:
                        pickle.dump(innovation, f)

                    with open(os.path.join(results_path, 'TotalActivation', 'param.pkl'), 'wb') as f:
                        pickle.dump(param, f)

                    # free memory
                    del innovation, activity_inducing, activity_related


                elif ta_real_done:
                    write_information(fid, 'Total activation on real data already computed, skipping...')

                # Total Activation for Surrogate Data
                if not ta_surrogate_done:
                    # surrogate data generation
                    surrogate = generate_surrogate(TC, subj_path_ta, param, fid)

                    # with h5py.File(os.path.join("F:/iCAP/Data/Matlab/Ketamine/Khali/", param['Subjects'][i_TA], 'TA_results/test3/Surrogate/Surrogate.mat'), "r") as f:
                    #     surrogate = np.array(f["Surrogate"])

                    # save TA data
                    save4dnii(results_path, 'Surrogate', 'Surrogate', surrogate, param['fHeader'].fname, param['mask'],
                              param['Dimension'])
                    surrogate = surrogate.T


                    # run TA on surrogate
                    activity_related_surrogate = run_total_activation(surrogate, param_tmp)[0]
                    # activity_related_surrogate = activity_related_surrogate[0]
                    innovation_surrogate, activity_inducing_surrogate = generate_innovations(activity_related_surrogate,
                                                                                             param)

                    # Save results for surrogate data
                    write_information(fid, 'Saving total activation results of surrogate data (mat and nifti)...')
                    save4dnii(results_path, 'Surrogate', 'Activity_inducing_surrogate', activity_inducing_surrogate.T,
                               param['fHeader'].fname, param['mask'], param['Dimension'])
                    save4dnii(results_path, 'Surrogate', 'Activity_related_surrogate', activity_related_surrogate.T,
                               param['fHeader'].fname, param['mask'], param['Dimension'])
                    save4dnii(results_path, 'Surrogate', 'Innovation_surrogate', innovation_surrogate.T,
                               param['fHeader'].fname, param['mask'], param['Dimension'])

                    # Save each object as a .pkl file
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

            # Resets the parameters to what they were at the start of the loop
            # (before any subject-specific change could have been made
            del param
            param = copy.deepcopy(param_CI)