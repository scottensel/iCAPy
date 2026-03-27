import numpy as np
from scipy import ndimage

def check_interconnectedness(data2d, param):
    """
    Parameters
    ----------
    data2d : ndarray, shape (n_tp, n_ret_vox)
        2D array containing thresholded significant innovations for all voxels.
        Values are typically -1, 0 or +1.
    param : dict-like
        Dictionary of parameters. Required keys:
        - 'Dimension' : iterable of length 4, where Dimension[0:3] are the
          spatial dimensions (nx, ny, nz) and Dimension[3] is n_tp.
        - 'mask_nonan' : boolean or integer index array of length
          prod(Dimension[0:3]) specifying which voxels are valid.
        Optional keys (with defaults matching the MATLAB code):
        - 'threshold_interconnectivity' : int, connectivity (6, 18 or 26);
          default 26 for 3D data.
        - 'threshold_minclussize' : int, minimum cluster size; default 6.

    Returns
    -------
    out2d : ndarray, shape (n_tp, n_ret_vox)
        2D array (time x voxel) with small isolated clusters removed.
    """
    data2d = np.asarray(data2d)

    # check sizes
    n_tp, n_vox = data2d.shape
    if n_vox < n_tp:
        # in MATLAB this is a warning only; we mimic but do not alter data
        print(f"Warning: data probably inverted - number of voxels: {n_vox}, number of frames {n_tp}")

    dim = param['Dimension']
    nx, ny, nz = int(dim[0]), int(dim[1]), int(dim[2])
    n_tp_param = int(dim[3])

    if n_tp_param != n_tp:
        # We keep it permissive but warn – this mirrors the implicit
        # assumption in the MATLAB code that Dimension(4) = nTP
        print(f"Warning: Dimension[3] ({n_tp_param}) != number of time points ({n_tp})")

    n_vox_total = nx * ny * nz

    # Allocate full 4D volume (flattened spatial dimension, then time)
    data3d = np.zeros((n_vox_total, n_tp), dtype=data2d.dtype)

    mask_nonan = np.asarray(param['mask_nonan'])
    # We assume mask_nonan indexes into the flattened 3D grid
    data3d[mask_nonan, :] = data2d.T  # (n_ret_vox, n_tp) -> placed into full volume

    # Reshape into 4D volume: (nx, ny, nz, n_tp) using Fortran order to mimic MATLAB
    data3d = np.reshape(data3d, (nx, ny, nz, n_tp), order='F')

    # Get or set defaults for connectivity and minimum cluster size
    conn = param.get('threshold_interconnectivity', None)
    if conn is None:
        print("interconnectivity threshold not specified, defining default of 26")
        conn = 26
        param['threshold_interconnectivity'] = conn

    min_clust = param.get('threshold_minclussize', None)
    if min_clust is None:
        print("minimum cluster size not specified, defining default 6")
        min_clust = 6
        param['threshold_minclussize'] = min_clust

    # Build connectivity structure for ndimage.label
    if conn == 6:
        structure = ndimage.generate_binary_structure(3, 1)
    elif conn == 18:
        structure = ndimage.generate_binary_structure(3, 2)  # THIS is 18-connected in SciPy
    elif conn == 26:
        structure = np.ones((3, 3, 3), dtype=bool)
    else:
        raise ValueError("conn must be 6, 18, or 26")

    # Loop over time points and remove too small connected components
    for it in range(n_tp):
        vol = data3d[:, :, :, it]

        # Work on non-zero voxels only (MATLAB bwconncomp works on logical input)
        mask = vol != 0
        if not np.any(mask):
            continue

        labeled, num = ndimage.label(mask, structure=structure)
        if num == 0:
            continue

        # Size of each component (in number of voxels)
        component_sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
        # Components smaller than threshold_minclussize are removed
        small_components = np.where(component_sizes < min_clust)[0] + 1  # labels are 1..num

        if small_components.size > 0:
            # Set voxels belonging to small components to zero (remove cluster)
            small_mask = np.isin(labeled, small_components)
            vol[small_mask] = 0

        data3d[:, :, :, it] = vol

    # Back to 2D: (n_vox_total, n_tp) in Fortran order, then select non-NaN voxels and transpose
    out2d_full = np.reshape(data3d, (n_vox_total, n_tp), order='F')
    out2d = out2d_full[mask_nonan, :].T  # -> (n_tp, n_ret_vox)

    return out2d
