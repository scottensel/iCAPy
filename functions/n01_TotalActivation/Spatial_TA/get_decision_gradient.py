import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full


def get_decision_gradient(param):
    """
    Computes a decision weight matrix W for TV minimization.
    If two neighbouring voxels are in different areas with respect to
    tissue type (e.g. GM vs WM+CSF) then their finite difference is not
    considered in TV minimization — the weight between them is set low.

    Inputs:
        param - dict containing:
            'GM_map'    - 3D probabilistic gray matter map volume
            'Dimension' - 4-element list/array of X, Y, Z, T sizes;
                          T (index 3) gives the number of time points

    Outputs:
        param - updated dict with the following fields added:
            'weight_x'  - (X x Y x Z x T) weight volume along X;
                          close to 1 between same-tissue voxels,
                          close to 0 across tissue boundaries
            'weight_y'  - weight volume along Y
            'weight_z'  - weight volume along Z

    Implemented by Younes Farouj, 29.04.2016
    """
    # dx, dy and dz are 3D matrices depicting the difference between two
    # elements of the GM map along the X, Y and Z directions respectively.
    # Zero-padding is done at the end of each dimension.
    dx, dy, dz = gradient3D_full(param['GM_map'])

    # Weights are 1 where neighbours are in the same tissue type (dx ≈ 0)
    # and decrease toward 0 where there is a tissue boundary (dx ≈ 1)
    wx_temp = 1 - np.abs(dx)
    wy_temp = 1 - np.abs(dy)
    wz_temp = 1 - np.abs(dz)

    # Repeat the 3D weight volume T times along the time dimension so that
    # the same spatial weights are applied at every time point
    time_dim = param['Dimension'][3]
    param['weight_x'] = np.repeat(wx_temp[..., np.newaxis], time_dim, axis=-1)
    param['weight_y'] = np.repeat(wy_temp[..., np.newaxis], time_dim, axis=-1)
    param['weight_z'] = np.repeat(wz_temp[..., np.newaxis], time_dim, axis=-1)

    return param
