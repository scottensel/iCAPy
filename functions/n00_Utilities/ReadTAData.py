import os
import re
import nibabel as nib
import numpy as np
from functions.n00_Utilities.WriteInformation import write_information


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
        pData   - (X x Y x Z) 3D array of probabilistic GM map values
        fHeader - nibabel image object for the functional data (first volume)
        pHeader - nibabel image object for the GM map

    17.01.2017 (DZ): removed dependence on subject index input; added
                     option for c1 masks not in functional resolution —
                     maps to functional space and saves the result
    """
    fData, pData, fHeader, pHeader = None, None, None, None

    # Resolve per-subject fields: use the subject-specific entry if a list
    # is provided, otherwise use the single shared value
    folder_functional = (param['Folder_functional'][sidx]
                         if isinstance(param['Folder_functional'], list)
                         else param['Folder_functional'])
    folder_gm         = (param['Folder_GM'][sidx]
                         if isinstance(param['Folder_GM'], list)
                         else param['Folder_GM'])
    ta_func_prefix    = (param['TA_func_prefix'][sidx]
                         if isinstance(param['TA_func_prefix'], list)
                         else param['TA_func_prefix'])
    ta_gm_prefix      = (param['TA_gm_prefix'][sidx]
                         if isinstance(param['TA_gm_prefix'], list)
                         else param['TA_gm_prefix'])

    # Construct full paths and verify they exist
    FPath = os.path.join(path, folder_functional)
    PPath = os.path.join(path, folder_gm)

    if not os.path.isdir(FPath):
        raise FileNotFoundError(f"Functional folder not found: {FPath}")
    if not os.path.isdir(PPath):
        raise FileNotFoundError(f"GM map folder not found: {PPath}")

    # ---- Read functional data ----

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

    # ---- Read probabilistic GM map ----

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

    return fData, pData, fHeader, pHeader
