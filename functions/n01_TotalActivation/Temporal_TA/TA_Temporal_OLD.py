import numpy as np
from pywt import wavedec
from tqdm import tqdm

from functions.n01_TotalActivation.Temporal_TA.TA_Temporal_OneTimeCourse_parallel import ta_temporal_onetimecourse


def ta_temporal(TCN, param):
    TC_OUT = np.zeros((param['Dimension'][3], param['NbrVoxels']))
    param['LambdaTemp'] = np.zeros(param['NbrVoxels'])


    # for i in range(param['NbrVoxels']):
    for i in tqdm(range(param['NbrVoxels']), desc="Processing temporal step", ncols=80):

        coef, len_ = wavedec(TCN[:, i], 'db3', level=1)
        coef[len_[0]:] = 0
        param['LambdaTemp'][i] = np.median(np.abs(coef)) * param['LambdaTempCoef']

        TC_OUT[:, i], param_out = ta_temporal_onetimecourse(TCN[:, i], i, param)
        param['LambdaTempFin'][i] = param_out['LambdasTempFin']
        param['NoiseEstimateFin'][i] = param_out['NoiseEstimateFin']


    return TC_OUT, param
