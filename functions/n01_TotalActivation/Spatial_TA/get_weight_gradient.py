import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full


def get_weight_gradient(param):
    # Compute gradients
    dx, dy, dz = gradient3D_full(param['GM_map'])

    # Calculate weights using exponential function
    wx_temp = np.exp(-np.abs(dx) / param['sigma'])
    wy_temp = np.exp(-np.abs(dy) / param['sigma'])
    wz_temp = np.exp(-np.abs(dz) / param['sigma'])
    time_dim = param['Dimension'][3]

    # Repeat gradients along time dimension
    param['weight_x'] = np.repeat(wx_temp[..., np.newaxis], time_dim, axis=-1)
    param['weight_y'] = np.repeat(wy_temp[..., np.newaxis], time_dim, axis=-1)
    param['weight_z'] = np.repeat(wz_temp[..., np.newaxis], time_dim, axis=-1)

    return param
