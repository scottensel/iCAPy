import os
import re
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
from functions.n00_Utilities.WriteInformation import write_information


def _reslice_to_functional(pData, pHeader, fHeader):
    """
    Resamples pData from GM (structural) space into functional space using
    the affine transforms from pHeader and fHeader.

    Equivalent to MATLAB's mapVTV(pData, pHeader, fHeader) call in
    ReadTAData.m, but vectorised via scipy.ndimage.affine_transform rather
    than looping voxel by voxel.

    The transform maps each output (functional) voxel coordinate to the
    corresponding input (structural) voxel coordinate:
        input_vox = inv(p_affine) @ f_affine @ output_vox

    Inputs:
        pData   - (X x Y x Z) 3D array of GM map in structural resolution
        pHeader - nibabel image object for the GM map
        fHeader - nibabel image object for the functional data

    Outputs:
        pData_resliced - (X' x Y' x Z') 3D array resampled to functional
                         resolution
    """
    p_affine = pHeader.affine
    f_affine = fHeader.affine
    out_shape = fHeader.shape[:3]

    # Voxel-to-voxel transform: functional vox -> structural vox
    M = np.linalg.inv(p_affine) @ f_affine

    return affine_transform(
        pData,
        M[:3, :3],
        offset=M[:3, 3],
        output_shape=out_shape,
        order=1,       # linear interpolation
        mode='nearest' # clamp out-of-bounds to edge (matches mapVTV clamping)
    )


def read_ta_data(path, sidx, param, fid=None):
    """
    Reads the input data for total activation; supports reading from a
    4D NIFTI, multiple 3D NIFTI files, or IMG/HDR file pairs.
    Also reads the probabilistic gray matter map and remaps it to
    functional resolution if the dimensions differ.

    Inputs:
        path  - str, full path towards the subject's data folder
        sidx  - int, 0-based subject index; used when param fields are
                lists with one entry per subject
        param - dict containing relevant TA parameters:
            'Folder_functional' - name of the functional data subfolder;
                                  [] / None if directly in path; can be a
                                  list of strings (one per subject)
            'TA_func_prefix'    - prefix string for functional NIFTI files;
                                  can be a list (one per subject)
            'Folder_GM'         - name of the probabilistic GM map subfolder;
                                  [] / None if directly in path; can be a list
            'TA_gm_prefix'      - prefix string for GM map NIFTI files;
                                  can be a list (one per subject). If a file
                                  starting with 'f' is not found, the prefix
                                  is tried without the leading 'f' (structural
                                  GM file in functional space)
        fid   - optional log file handle for write_information

    Outputs:
        fData   - (X x Y x Z x T) 4D array of functional data
        pData   - (X x Y x Z) 3D array of probabilistic GM map values,
                  resampled to functional resolution if needed
        fHeader - nibabel image object for the functional data (first volume)
        pHeader - nibabel image object for the GM map

    17.01.2017 (DZ): removed dependence on subject index input; added
                     option for c1 masks not in functional resolution —
                     maps to functional space and saves the result
    """
    fData, pData, fHeader, pHeader = None, None, None, None

    def _resolve(val, idx):
        """
        Returns the subject-specific value from val.
        - If val is not a list: use it as-is for all subjects.
        - If val is a list with one element: use that element for all subjects.
        - If val is a list with n_subjects elements: use val[idx].
        Matches MATLAB behaviour where a single string is shared across subjects.
        """
        if not isinstance(val, list):
            return val
        if len(val) == 1:
            return val[0]
        return val[idx]

    # Resolve per-subject fields
    folder_functional = _resolve(param['Folder_functional'], sidx)
    folder_gm         = _resolve(param['Folder_GM'], sidx)
    ta_func_prefix    = _resolve(param['TA_func_prefix'], sidx)
    ta_gm_prefix      = _resolve(param['TA_gm_prefix'], sidx)

    # Construct full paths and verify they exist
    FPath = os.path.join(path, folder_functional)
    PPath = os.path.join(path, folder_gm)

    if not os.path.isdir(FPath):
        raise FileNotFoundError(f"Functional folder not found: {FPath}")
    if not os.path.isdir(PPath):
        raise FileNotFoundError(f"GM map folder not found: {PPath}")

    # ── Read functional data ──────────────────────────────────────────────────

    # List NIFTI files matching the functional prefix, sorted numerically
    func_files = sorted(
        [f for f in os.listdir(FPath)
         if f.startswith(ta_func_prefix) and f.endswith('.nii')],
        key=lambda f: (int(re.search(r'(\d+)(?=\.\w+$)', f).group(1))
                       if re.search(r'(\d+)(?=\.\w+$)', f) else 0)
    )

    if len(func_files) == 1:
        # Single 4D NIFTI file: read all volumes directly
        file_path = os.path.join(FPath, func_files[0])
        header    = nib.load(file_path)
        n_volumes = header.shape[-1] if header.ndim == 4 else 1

        if n_volumes == 1:
            write_information(fid, "Only one NIFTI volume provided; multiple volumes required.")
            raise ValueError("Only one NIFTI volume provided; multiple volumes required.")
        else:
            fHeader = header
            fData   = header.get_fdata()
            write_information(
                fid,
                f"Reading data for subject {path}: 1 multi-volume NIFTI file found: {func_files[0]}"
            )

    elif len(func_files) > 1:
        # Multiple 3D NIFTI files: stack into a 4D array
        fHeader = nib.load(os.path.join(FPath, func_files[0]))
        fHeader.fname = os.path.join(FPath, func_files[0])
        fData = np.stack(
            [nib.load(os.path.join(FPath, f)).get_fdata() for f in func_files],
            axis=-1
        )
        write_information(
            fid,
            f"Reading data for subject {path}: {len(func_files)} NIFTI files found."
        )

    else:
        # No NIFTI files found — try IMG/HDR pairs instead
        hdr_files = sorted(
            [f for f in os.listdir(FPath)
             if f.startswith(ta_func_prefix) and f.endswith('.hdr')],
            key=lambda f: (int(re.search(r'(\d+)(?=\.\w+$)', f).group(1))
                           if re.search(r'(\d+)(?=\.\w+$)', f) else 0)
        )
        img_files = sorted(
            [f for f in os.listdir(FPath)
             if f.startswith(ta_func_prefix) and f.endswith('.img')],
            key=lambda f: (int(re.search(r'(\d+)(?=\.\w+$)', f).group(1))
                           if re.search(r'(\d+)(?=\.\w+$)', f) else 0)
        )

        if len(hdr_files) == len(img_files) > 0:
            fHeader = nib.load(os.path.join(FPath, hdr_files[0]))
            fData   = np.stack(
                [nib.load(os.path.join(FPath, f)).get_fdata() for f in img_files],
                axis=-1
            )
            write_information(
                fid,
                f"Reading data for subject {path}: {len(hdr_files)} HDR/IMG files found."
            )
        else:
            write_information(fid, "No valid NIFTI or HDR/IMG functional files found.")
            raise FileNotFoundError("No valid NIFTI or HDR/IMG functional files found.")

    # ── Read probabilistic GM map ─────────────────────────────────────────────

    gm_files = sorted(
        [f for f in os.listdir(PPath)
         if f.startswith(ta_gm_prefix) and f.endswith('.nii')]
    )

    if len(gm_files) == 1:
        pHeader = nib.load(os.path.join(PPath, gm_files[0]))
        pData   = pHeader.get_fdata()
        write_information(
            fid,
            f"Reading probabilistic GM data for subject {path}: "
            f"1 NIFTI file found: {gm_files[0]}"
        )

    elif len(gm_files) == 0 and ta_gm_prefix.startswith('f'):
        # If no file found with prefix 'f...', try the structural GM file
        # without the leading 'f' (GM map not yet resliced to functional space)
        ta_gm_prefix = ta_gm_prefix[1:]
        gm_files     = sorted(
            [f for f in os.listdir(PPath)
             if f.startswith(ta_gm_prefix) and f.endswith('.nii')]
        )
        if len(gm_files) == 1:
            pHeader = nib.load(os.path.join(PPath, gm_files[0]))
            pData   = pHeader.get_fdata()
            write_information(
                fid,
                f"Reading GM data for subject {path}: 1 NIFTI file found "
                f"with modified prefix: {gm_files[0]}"
            )
        else:
            write_information(fid, "No GM NIFTI files found or multiple GM files present.")
            raise ValueError("No GM NIFTI files found or multiple GM files present.")
    else:
        write_information(fid, "Multiple GM NIFTI files found with the same prefix.")
        raise ValueError("Multiple GM NIFTI files found with the same prefix.")

    # ── Reslice GM map to functional resolution if needed ─────────────────────
    # Matches MATLAB: if ~isequal(pHeader.dim, fHeader.dim) -> mapVTV(...)
    # Uses scipy.ndimage.affine_transform (vectorised) instead of the
    # voxel-by-voxel mapVTV loop for speed.
    # The resliced map is saved to disk with an 'f' prefix (matching MATLAB's
    # spm_write_vol call) so that subsequent runs find it directly and skip
    # the reslicing step.
    if pData.shape[:3] != fData.shape[:3]:
        write_information(
            fid,
            'Different data dimensions for GM map and functional data: '
            'converting GM map to functional resolution...'
        )
        pData = _reslice_to_functional(pData, pHeader, fHeader)

        # Save resliced GM map with 'f' prefix so next run skips this step
        resliced_fname = os.path.join(PPath, 'f' + gm_files[0])
        nib.save(nib.Nifti1Image(pData, fHeader.affine), resliced_fname)
        write_information(
            fid,
            f'Resliced GM map saved to: {resliced_fname}'
        )

        # Update pHeader to reflect the new resolution
        pHeader = nib.load(resliced_fname)

    return fData, pData, fHeader, pHeader