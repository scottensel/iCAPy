import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from functions.n00_Utilities.WriteInformation import write_information


def create_ta_mask(param, fid=None):
    """
    Creates a binary mask containing only within-brain voxels to retain
    for total activation. The mask is determined by thresholding the
    probabilistic gray matter map, and optionally refined using
    morphological operations to fill holes.

    Inputs:
        param - dict containing TA parameters; must include:
            'GM_map'          - 3D array with probabilistic GM values
            'T_gm'            - float, threshold between 0 and 1 past
                                which a voxel is considered gray matter
            ['is_morpho']     - bool, if True, applies morphological
                                closing followed by opening to fill holes
                                in the mask; default False
            ['n_morpho_voxels'] - int, size of the cubic structuring
                                element for morphological operations;
                                required if 'is_morpho' is True
        fid   - optional log file handle for write_information

    Outputs:
        mask    - (X*Y*Z,) 1D boolean array in column-major (Fortran)
                  order; True for retained voxels
        mask_3D - (X x Y x Z) 3D boolean array with the same mask
    """
    # Threshold the GM map: voxels above T_gm are considered gray matter
    mask_3D = np.where(param['GM_map'] > param['T_gm'], 1, 0).astype(bool)

    if param.get('is_morpho', False):
        # Perform morphological closing followed by opening to fill holes
        # and remove small isolated regions in the GM mask
        structuring_element = np.ones((param['n_morpho_voxels'],) * 3)
        mask_3D = binary_closing(mask_3D, structure=structuring_element)
        mask_3D = binary_opening(mask_3D, structure=structuring_element)

        write_information(
            fid,
            f"Created mask with a threshold of {param['T_gm']} and opening/closure..."
        )
    else:
        write_information(
            fid,
            f"Created mask with a threshold of {param['T_gm']}..."
        )

    # Flatten to 1D using column-major (Fortran) order to match MATLAB's
    # mask(:) operation which uses column-major indexing
    mask = np.ravel(mask_3D, order='F')

    return mask, mask_3D
