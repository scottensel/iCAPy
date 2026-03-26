import os
import nibabel as nib
import numpy as np
import re
from functions.n00_Utilities.WriteInformation import write_information

def read_ta_data(path, sidx, param, fid=None):
    fData, pData, fHeader, pHeader = None, None, None, None

    # Access Folder_functional and Folder_GM based on subject index
    folder_functional = param['Folder_functional'][sidx] if isinstance(param['Folder_functional'], list) else param['Folder_functional']
    folder_gm = param['Folder_GM'][sidx] if isinstance(param['Folder_GM'], list) else param['Folder_GM']
    ta_func_prefix = param['TA_func_prefix'][sidx] if isinstance(param['TA_func_prefix'], list) else param['TA_func_prefix']
    ta_gm_prefix = param['TA_gm_prefix'][sidx] if isinstance(param['TA_gm_prefix'], list) else param['TA_gm_prefix']

    # Construct paths for functional and GM data folders
    FPath = os.path.join(path, folder_functional)
    PPath = os.path.join(path, folder_gm)

    if not os.path.isdir(FPath):
        raise FileNotFoundError(f"Functional folder not found: {FPath}")
    if not os.path.isdir(PPath):
        raise FileNotFoundError(f"GM map folder not found: {PPath}")

    # List all NIFTI files with the functional prefix
    # func_files = sorted([f for f in os.listdir(FPath) if f.startswith(ta_func_prefix) and f.endswith('.nii')])
    func_files = sorted(
        [f for f in os.listdir(FPath) if f.startswith(ta_func_prefix) and f.endswith('.nii')],
        key=lambda f: int(re.search(r'(\d+)(?=\.\w+$)', f).group(1)) if re.search(r'(\d+)(?=\.\w+$)', f) else 0
    )

    # Case 1: Only one NIFTI file is found
    if len(func_files) == 1:
        file_path = os.path.join(FPath, func_files[0])
        header = nib.load(file_path)
        n_volumes = header.shape[-1] if header.ndim == 4 else 1

        if n_volumes == 1:
            error_msg = "Only one NIFTI volume provided; multiple volumes required."
            # if fid:
            #     fid.write(error_msg + "\n")
            write_information(fid, f"Only one NIFTI volume provided; multiple volumes required.")

            raise ValueError(error_msg)
        else:
            # Read multiple volumes from a single 4D NIFTI file
            fHeader = header
            fData = header.get_fdata()
            # if fid:
            write_information(fid, f"Reading data for subject {path}: 1 multi-volume NIFTI file found: {func_files[0]}")

    # Case 2: Multiple NIFTI files
    elif len(func_files) > 1:
        fData = []
        fHeader = nib.load(os.path.join(FPath, func_files[0]))  # Use the first file as the header
        fHeader.fname = os.path.join(FPath, func_files[0])
        for func_file in func_files:
            fData.append(nib.load(os.path.join(FPath, func_file)).get_fdata())
        fData = np.stack(fData, axis=-1)
        # if fid:
            # fid.write(f"Reading data for subject {path}: {len(func_files)} NIFTI files found.\n")
        write_information(fid, f"Reading data for subject {path}: {len(func_files)} NIFTI files found.")

    # Case 3: No NIFTI files found, attempt to load IMG/HDR pairs
    else:
        # hdr_files = sorted([f for f in os.listdir(FPath) if f.startswith(ta_func_prefix) and f.endswith('.hdr')])
        hdr_files = sorted(
            [f for f in os.listdir(FPath) if f.startswith(ta_func_prefix) and f.endswith('.hdr')],
            key=lambda f: int(re.search(r'(\d+)(?=\.\w+$)', f).group(1)) if re.search(r'(\d+)(?=\.\w+$)', f) else 0
        )
        # img_files = sorted([f for f in os.listdir(FPath) if f.startswith(ta_func_prefix) and f.endswith('.img')])
        img_files = sorted(
            [f for f in os.listdir(FPath) if f.startswith(ta_func_prefix) and f.endswith('.img')],
            key=lambda f: int(re.search(r'(\d+)(?=\.\w+$)', f).group(1)) if re.search(r'(\d+)(?=\.\w+$)', f) else 0
        )

        if len(hdr_files) == len(img_files) > 0:
            fData = []
            fHeader = nib.load(os.path.join(FPath, hdr_files[0]))  # Use the first file as the header
            for hdr_file, img_file in zip(hdr_files, img_files):
                img_path = os.path.join(FPath, img_file)
                fData.append(nib.load(img_path).get_fdata())
            fData = np.stack(fData, axis=-1)
            # if fid:
            #     fid.write(f"Reading data for subject {path}: {len(hdr_files)} HDR/IMG files found.\n")
            write_information(fid, f"Reading data for subject {path}: {len(hdr_files)} HDR/IMG files found.")

        else:
            error_msg = "No valid NIFTI or HDR/IMG functional files found."
            # if fid:
            #     fid.write(error_msg + "\n")
            write_information(fid, f"No valid NIFTI or HDR/IMG functional files found.")

            raise FileNotFoundError(error_msg)

    # Load the probabailtic GM map
    gm_files = sorted([f for f in os.listdir(PPath) if f.startswith(ta_gm_prefix) and f.endswith('.nii')])
    if len(gm_files) == 1:
        pHeader = nib.load(os.path.join(PPath, gm_files[0]))
        pData = pHeader.get_fdata()
        # if fid:
        #     fid.write(f"Reading GM data for subject {path}: 1 NIFTI file found: {gm_files[0]}\n")
        write_information(fid, f"Reading probabilistic GM data for subject {path}: 1 NIFTI file found: {gm_files[0]}")


    elif len(gm_files) == 0 and ta_gm_prefix.startswith('f'):
        # Try GM prefix without 'f'
        ta_gm_prefix = ta_gm_prefix[1:]
        gm_files = sorted([f for f in os.listdir(PPath) if f.startswith(ta_gm_prefix) and f.endswith('.nii')])
        if len(gm_files) == 1:
            pHeader = nib.load(os.path.join(PPath, gm_files[0]))
            pData = pHeader.get_fdata()
            # if fid:
            #     fid.write(f"Reading GM data for subject {path}: 1 NIFTI file found with modified prefix: {gm_files[0]}\n")
            write_information(fid, f"Reading GM data for subject {path}: 1 NIFTI file found with modified prefix: {gm_files[0]}")

        else:
            error_msg = "No GM NIFTI files found or multiple GM files present."
            # if fid:
            #     fid.write(error_msg + "\n")
            write_information(fid, f"No GM NIFTI files found or multiple GM files present.")
            raise ValueError(error_msg)
    else:
        error_msg = "Multiple GM NIFTI files found with the same prefix."
        # if fid:
        #     fid.write(error_msg + "\n")
        write_information(fid, f"Multiple GM NIFTI files found with the same prefix.")
        raise ValueError(error_msg)

    return fData, pData, fHeader, pHeader
