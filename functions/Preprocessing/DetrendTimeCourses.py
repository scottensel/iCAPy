import numpy as np
from functions.Preprocessing.sol_dct import sol_dct
from functions.Utilities.WriteInformation import write_information

def detrend_time_courses(TC, param, fid=None):
    TCN = np.zeros_like(TC)

    TC_tmp = TC.T
    for i in range(param['NbrVoxels']):
        # Detrend using DCT and normalize (490,) shape
        TCN[i, :], c_dct = sol_dct(TC_tmp[:, i], param['TR'], param['DCT_TS'])
        TCN[i, :] /= np.std(TCN[i, :], ddof=1)

    # Log the detrending process
    # if fid:
    # fid.write(f"Detrending with DCT cutoff = {4param['DCT_TS']} s, {len(param['Covariates'])} covariates.\n")
    write_information(fid, f"Detrending with DCT cutoff = {param['DCT_TS']} s, {len(param['Covariates'])} covariates.")

    return TCN