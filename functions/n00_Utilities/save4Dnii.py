import os
import numpy as np
import nibabel as nib


def save4dnii(path, subFolder, fname, data, hdrFilename, mask1D=None, dim3D=None):
    """
    Saves data as a 4D NIFTI file, with handling for masked 2D input data.
    The header is taken from an existing NIFTI file and updated to match
    the output dimensions.

    Inputs:
        path        - str, base output directory
        subFolder   - str, subfolder under path where the file will be saved
        fname       - str, output filename without extension
        data        - 2D or 4D numpy array:
                        2D: (n_voxels x T) — within-mask voxel time courses
                        4D: (X x Y x Z x T) — ready to write directly
        hdrFilename - str, path to an existing NIFTI file used as the header
                      template (provides affine, voxel sizes, etc.)
        mask1D      - 1D boolean array, optional; required when data is 2D
                      and contains only the within-mask voxels (not the full
                      flattened volume). If data is 2D and mask1D is None,
                      all voxels are assumed present.
        dim3D       - tuple/list [X, Y, Z], optional; spatial dimensions for
                      reshaping the 2D data; taken from the header if not
                      provided

    Outputs:
        Saves a NIFTI file at path/subFolder/fname.nii and prints the
        saved file path to the terminal.
    """
    # Load the reference header from an existing NIFTI file
    header_nifti = nib.load(hdrFilename)
    hdr          = header_nifti.header.copy()

    # Use spatial dimensions from the reference header if not explicitly provided
    if dim3D is None:
        dim3D = hdr.get_data_shape()[:3]

    # Default mask: all voxels present (for the case where 2D data is the
    # full flattened volume without any masking)
    if mask1D is None and data.ndim == 2:
        mask1D = np.ones(data.shape[0], dtype=bool)

    # Handle 1D input by promoting it to (n_voxels x 1) for consistent
    # downstream processing
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # ---- Handle 2D data (n_voxels x T) ----
    if data.ndim == 2:
        # Warn and transpose if data appears to have fewer voxels than frames,
        # which suggests the array was passed in the wrong orientation
        if data.shape[0] < data.shape[1]:
            print("Warning: Data appears inverted (voxels x frames mismatch), "
                  "transposing data...")
            data = data.T

        if len(mask1D) != data.shape[0]:
            # data contains only within-mask voxels; insert them back into
            # the full volume before reshaping
            if np.count_nonzero(mask1D) != data.shape[0]:
                print("Error: Mask and data voxel counts do not match. "
                      "Skipping NIFTI saving.")
                return
            data_4D            = np.zeros((len(mask1D), data.shape[1]))
            data_4D[mask1D, :] = data
        else:
            # data already covers the full volume
            if np.prod(hdr.get_data_shape()[:3]) != data.shape[0]:
                raise ValueError("save4Dnii: Dimensions mismatch with existing header.")
            data_4D = data

        # Reshape to 4D (X x Y x Z x T) using column-major order to match
        # MATLAB's reshape convention
        data = data_4D.reshape([*dim3D[:3], data.shape[1]], order='F')
        del data_4D

    # Validate that the spatial dimensions of the data match the header
    if hdr.get_data_shape()[:3] != data.shape[:3]:
        raise ValueError("save4Dnii: Dimensions mismatch with existing header.")

    # Update the header's fourth dimension to match the number of volumes
    hdr['dim'][4] = data.shape[3]

    # Create output directory if needed and write the NIFTI file
    outDir  = os.path.join(path, subFolder)
    os.makedirs(outDir, exist_ok=True)
    outFile = os.path.join(outDir, f"{fname}.nii")

    new_img = nib.Nifti1Image(data, header_nifti.affine, hdr)
    nib.save(new_img, outFile)
    print(f"Saved NIFTI file to {outFile}")
