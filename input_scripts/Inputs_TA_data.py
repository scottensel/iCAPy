import os

def setup_data_params():
    """
    Returns
    -------
    param : dict
        Data- and thresholding-related parameters for TA.
    """
    param = {}

    # Path to datas C:\Users\scott\PycharmProjects\pythonProject\iCAPy\example data
    param['PathData'] = 'example data'

    # TR of the data
    param['TR'] = 2

    # List of subjects on which to run total activation
    # this is where the TA folder will be created (or looked for)
    param['Subjects'] = ['sub-10159','sub-10171']

    # number of subjects
    param['n_subjects'] = len(param['Subjects'])

    # Title for this analysis session
    # this will be the title used in later steps of the iCAP pipeline
    param['title'] = 'pyexampleToolbox_openfMRI'

    # Information about the folders where to retrieve functional and structural data
    #############################################################################

    # Name of folder containing the functional data
    param['Folder_functional'] = ['func/realigned']

    # common NAME of the functional data that each slice has (ex. NAME_0XX.nii)
    param['TA_func_prefix'] = ['s6rf']

    # Folder where the grey matter map can be found
    param['Folder_GM'] = ['anat/Segmented']

    # common NAME of the gray matter map
    param['TA_gm_prefix'] = ['c1']

    # Gray matter related information
    ###########################################################
    # Threshold of a probablistic gray matter map
    # (Values greater than this will be included in the mask)
    # must be between 0 and 1
    param['T_gm'] = 0.3

    # select if morphological operations (opening and closure) should be
    # run on the GM mask to remove wholes, and if yes, specify the size (in
    # voxels) for opening and closing operators
    param['is_morpho'] = 0
    param['n_morpho_voxels'] = 2

    # Functional processing related information
    ###########################################################
    # Number of scans to skip for equilibration effects
    param['skipped_scans'] = 5

    # select if detrending or not
    param['doDetrend'] = 1
    param['doNormalize'] = 1  # set to 1 for normalisation without DCT detrending

    # Detrending information: cut-off period for the DCT basis (for example,
    # 128 means a cutoff of 1/128 = 0.0078 [Hz], and covariates to add (should
    # be provided each as a column of 'Covariates')
    param['DCT_TS'] = 128
    param['Covariates'] = []

    # Select if scurbbing should be run on the data
    param['doScrubbing'] = 1

    # Folder where motion data from SPM realignment is stored, if motion data
    # is taken from another program than SPM, a text file with the 6 motion
    # parameters (3 translational in mm + 3 rotational in rad) should be set as
    # input here
    param['Folder_motion'] = param['Folder_functional']

    # Common name of motion file
    param['TA_mot_prefix'] = ['rp_f']

    # Number of lines to ignore at the beginning of the motion file
    # if none or empty will default to param.skipped_scans
    param['skipped_scans_motionfile'] = None

    # Motion information: type of method to use to quantify motion (choose
    # between 'Power' and ), and threshold of displacement to use for each
    # frame (in [mm])
    param['FD_method'] = 'Power'
    param['FD_threshold'] = 0.5

    # interpolation methods (spline or linear)
    # look at interp function to find all methods
    # default is spline
    param['interType'] = 'spline'

    return param
