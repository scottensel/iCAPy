import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full


def get_weight_gradient(param):
    """
    Computes a soft weight matrix W for TV minimization using an
    exponential decay. If two neighbouring voxels are in different areas
    with respect to tissue type (e.g. GM vs WM+CSF) then their finite
    difference receives a lower weight in TV minimization.

    Inputs:
        param - dict containing:
            'GM_map'    - 3D probabilistic gray matter map volume
            'sigma'     - float, controls how sharply the weights decay
                          at tissue boundaries; larger sigma = softer
                          transition
            'Dimension' - 4-element list/array of X, Y, Z, T sizes;
                          T (index 3) gives the number of time points

    Outputs:
        param - updated dict with the following fields added:
            'weight_x'  - (X x Y x Z x T) weight volume along X
            'weight_y'  - weight volume along Y
            'weight_z'  - weight volume along Z

    Implemented by Younes Farouj, 29.04.2016
    """
    # dx, dy and dz are 3D matrices depicting the difference between two
    # elements of the GM map along the X, Y and Z directions respectively.
    # Zero-padding is done at the end of each dimension.
    dx, dy, dz = gradient3D_full(param['GM_map'])

    # If we transition from GM to another tissue type, dx is large and so
    # wx_temp becomes small (low weight — finite difference not penalised).
    # Conversely, between two GM voxels the gradient is near zero and the
    # weight is near 1 (finite difference fully penalised in TV).
    wx_temp = np.exp(-np.abs(dx) / param['sigma'])
    wy_temp = np.exp(-np.abs(dy) / param['sigma'])
    wz_temp = np.exp(-np.abs(dz) / param['sigma'])

    # Repeat the 3D weight volume T times along the time dimension so that
    # the same spatial weights are applied at every time point
    time_dim = param['Dimension'][3]
    param['weight_x'] = np.repeat(wx_temp[..., np.newaxis], time_dim, axis=-1)
    param['weight_y'] = np.repeat(wy_temp[..., np.newaxis], time_dim, axis=-1)
    param['weight_z'] = np.repeat(wz_temp[..., np.newaxis], time_dim, axis=-1)

    return param
