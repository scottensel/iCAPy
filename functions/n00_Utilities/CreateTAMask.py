import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from functions.n00_Utilities.WriteInformation import write_information


def create_ta_mask(param, fid=None):
    """
    Create a binary mask based on the probabilistic GM map.

    Parameters:
    - param: Dictionary containing TA parameters. It must include:
        - 'GM_map': 3D array of probabilistic GM values.
        - 'T_gm': Threshold for gray matter probability (between 0 and 1).
        - 'is_morpho' (optional): Whether to perform morphological operations.
        - 'n_morpho_voxels' (optional): Size of the structuring element for morphological operations.
    - fid: Optional file object for logging.

    Returns:
    - mask: 1D array indicating retained voxels (flattened binary mask).
    - mask_3D: 3D binary mask with the same shape as GM_map.
    """
    try:
        # Initialize 3D mask based on threshold in the GM map
        mask_3D = np.where(param['GM_map'] > param['T_gm'], 1, 0).astype(bool)

        if param.get('is_morpho', False):
            # Perform morphological closing followed by opening if specified
            structuring_element = np.ones((param['n_morpho_voxels'],) * 3)
            mask_3D = binary_closing(mask_3D, structure=structuring_element)
            mask_3D = binary_opening(mask_3D, structure=structuring_element)

            # if fid:
            #     fid.write(f"Created mask with threshold {param['T_gm']} and morphological closing/opening.\n")
            write_information(fid, f"Created mask with threshold {param['T_gm']} and morphological closing/opening.")

        else:
            # if fid:
            #     fid.write(f"Created mask with threshold {param['T_gm']}.\n")
            write_information(fid, f"Created mask with threshold {param['T_gm']}.")

        # Convert 3D mask to 1D for compatibility
        mask = np.ravel(mask_3D, order='F')  # <-- critical
        # mask = mask_3D.ravel()

        return mask, mask_3D

    except Exception as e:
        # If an error occurs, fall back to a simple threshold-based mask
        mask_3D = np.where(param['GM_map'] > param['T_gm'], 1, 0).astype(bool)
        mask = np.ravel(mask_3D, order='F')  # <-- critical
        # mask = mask_3D.ravel()

        # if fid:
        #     fid.write(f"Error encountered. Fallback mask created with threshold {param['T_gm']}.\n")
        write_information(fid, f"Error encountered. Fallback mask created with threshold {param['T_gm']}.")

        raise e  # Re-raise the exception for visibility if needed
