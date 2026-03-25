import os
import numpy as np
import nibabel as nib


def save4dnii(path, subFolder, fname, data, hdrFilename, mask1D=None, dim3D=None):
    """
    Save data in 4D NIFTI format, with handling for masked 2D data.

    Parameters:
    - path: str, base directory for output.
    - subFolder: str, subfolder under base directory.
    - fname: str, output filename (without extension).
    - data: 2D or 4D numpy array, data to be saved.
    - hdrFilename: str, path to an existing NIFTI file to use as a header template.
    - mask1D: 1D boolean array, optional mask applied to 2D data.
    - dim3D: tuple, optional dimensions for reshaping the mask (X, Y, Z).
    """
    # Load header from existing NIFTI file
    header_nifti = nib.load(hdrFilename)
    hdr = header_nifti.header.copy()

    # Use dimensions from header if dim3D is not provided
    if dim3D is None:
        dim3D = hdr.get_data_shape()[:3]

    # Initialize mask if not provided and data is 2D
    if mask1D is None and data.ndim == 2:
        mask1D = np.ones(data.shape[0], dtype=bool)

    # just for python if a mask comes in 1D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Handling 2D data case (masked within-brain voxels)
    if data.ndim == 2:
        if data.shape[0] < data.shape[1]:  # Likely inverted data
            print(f"Warning: Data appears inverted (voxels x frames mismatch), transposing data...")
            data = data.T  # Transpose to have (voxels, frames)

        if len(mask1D) != data.shape[0]:
            if np.count_nonzero(mask1D) != data.shape[0]:
                print("Error: Mask and data voxel counts do not match. Skipping NIFTI saving.")
                return
            # Insert data back into a full 3D+time array using the mask
            data_4D = np.zeros((len(mask1D), data.shape[1]))
            data_4D[mask1D, :] = data
        else:
            # Validate dimensions with the header
            if np.prod(hdr.get_data_shape()[:3]) != data.shape[0]:
                raise ValueError("save4Dnii: Dimensions mismatch with existing header.")
            data_4D = data

        # Reshape to 4D (X, Y, Z, T)
        data = data_4D.reshape([*dim3D[:3], data.shape[1]], order='F')
        del data_4D  # Free memory

    # Validate dimensions with the header
    if hdr.get_data_shape()[:3] != data.shape[:3]:
        raise ValueError("save4Dnii: Dimensions mismatch with existing header.")

    # Update header for the fourth dimension (time/frames)
    hdr['dim'][4] = data.shape[3]

    # Prepare output directory and file path
    outDir = os.path.join(path, subFolder)
    os.makedirs(outDir, exist_ok=True)
    outFile = os.path.join(outDir, f"{fname}.nii")

    # Save data as NIFTI with the modified header
    new_img = nib.Nifti1Image(data, header_nifti.affine, hdr)
    nib.save(new_img, outFile)
    print(f"Saved NIFTI file to {outFile}")

