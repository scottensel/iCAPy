import numpy as np
from functions.n00_Preprocessing.sol_dct import sol_dct
from functions.n00_Utilities.WriteInformation import write_information


def detrend_time_courses(TC, param, fid=None):
    """
    Detrends the time courses for total activation by regressing out
    low-frequency components using a DCT basis, then normalises each
    voxel time course to unit standard deviation.

    Inputs:
        TC    - (n_ret_voxels x n_time_points) 2D array of data to detrend
        param - dict containing relevant TA parameters; must include:
            'NbrVoxels'  - int, number of retained voxels
            'TR'         - float, repetition time of the data in seconds
            'DCT_TS'     - float, cut-off period for the DCT basis in seconds
            'Covariates' - array of covariates to include in the regression;
                           set to [] if none wished
        fid   - optional log file handle for write_information

    Outputs:
        TCN - (n_ret_voxels x n_time_points) 2D array of detrended and
              normalised time courses
    """
    # Initialise output array
    TCN    = np.zeros_like(TC)
    TC_tmp = TC.T    # transpose to (n_time_points x n_voxels) for sol_dct

    for i in range(param['NbrVoxels']):

        # Regress out low-frequency components using the DCT basis and
        # any additional covariates specified in param
        TCN[i, :], _ = sol_dct(TC_tmp[:, i], param['TR'], param['DCT_TS'],
                                param.get('Covariates', None))

        # Normalise to unit standard deviation (ddof=1 matches MATLAB's std)
        TCN[i, :] /= np.std(TCN[i, :], ddof=1)

    write_information(
        fid,
        f"Detrending and normalizing the data with DCT = "
        f"{param['DCT_TS']} [s] and {len(param['Covariates'])} covariate(s)"
    )

    return TCN
