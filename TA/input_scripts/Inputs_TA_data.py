import os

def setup_data_params():
    param = {}

    # Path to data and TR settings
    param['PathData'] = 'F:/iCAP/Data/Python/Ketamine/Khali'
    param['TR'] = 2.40

    # List of subjects
    param['Subjects'] = ['run5_20130209', 'run5_20130308', 'run7_20130322'] #'run5_20130209'; '/run5_20130308'; '/run7_20130322'
    param['n_subjects'] = len(param['Subjects'])

    # Title for this analysis session
    param['title'] = 'test2'

    # Data folder informatio
    param['Folder_functional'] = ['rs/realigned','rs/realigned','rs/realigned']
    param['TA_func_prefix'] = ['ketamine','ketamine','ketamine']
    param['Folder_GM'] = ['T1','T1','T1']
    param['TA_gm_prefix'] = ['rest_c2','rest_c2','rest_c2']

    # Gray matter mask thresholding and morphological operations
    param['T_gm'] = 0.3
    param['is_morpho'] = 0
    param['n_morpho_voxels'] = 3
    param['n_morpho_voxels2'] = 2

    # Functional preprocessing configurations
    param['skipped_scans'] = 10
    param['doDetrend'] = 1
    param['DCT_TS'] = 128
    param['Covariates'] = []

    # Motion scrubbing settings
    param['doScrubbing'] = 0
    param['Folder_motion'] = []
    param['TA_mot_prefix'] = []
    param['skipped_scans_motionfile'] = None
    param['FD_method'] = 'Power'
    param['FD_threshold'] = 0.5
    param['interType'] = 'spline'

    # CUDA/GPU setting
    param['use_cuda'] = 0  # Set to 1 if GPU acceleration is available

    # Specify your CUDA version and installation path
    cuda_version = '12.6'  # Change this to your actual CUDA version
    cuda_path = f'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{cuda_version}/bin'

    # Set the CUDA_PATH environment variable
    os.environ['CUDA_PATH'] = cuda_path

    return param
