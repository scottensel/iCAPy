import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full


def get_decision_gradient(param):
    # Compute gradients
    dx, dy, dz = gradient3D_full(param['GM_map'])

    # Compute weights with broadcasting for time dimension
    wx_temp, wy_temp, wz_temp = 1 - np.abs(dx), 1 - np.abs(dy), 1 - np.abs(dz)
    time_dim = param['Dimension'][3]

    # Repeat gradients along time dimension
    param['weight_x'] = np.repeat(wx_temp[..., np.newaxis], time_dim, axis=-1)
    param['weight_y'] = np.repeat(wy_temp[..., np.newaxis], time_dim, axis=-1)
    param['weight_z'] = np.repeat(wz_temp[..., np.newaxis], time_dim, axis=-1)

    return param
