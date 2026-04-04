"""
example_normalization_OpenfMRI.py
──────────────────────────────────
Python equivalent of example_normalization_OpenfMRI.m

Applies SPM deformation fields (y_*.nii) to warp subject-space NIfTI files
into MNI space, then updates the param .pkl files for the clustering step.
No SPM or MATLAB required — uses nibabel and scipy only.

SPM y_* deformation fields are (X x Y x Z x 1 x 3) NIfTI volumes where
each voxel contains the absolute world-space MNI coordinate (in mm) that
the corresponding subject-space voxel maps to. This script reads those
coordinates, inverts the MNI affine to get voxel indices, and resamples
using scipy.ndimage.map_coordinates.

Usage:
    python example_normalization_OpenfMRI.py

Inputs (edit the configuration block below):
    PATH_DATA    - path to the root data directory
    SUBJECTS     - list of subject subfolder names
    THRESH_TITLE - thresholding subfolder name
    TITLE        - TA results folder name

Outputs (written to TA_results/<title>_MNI/ for each subject):
    Thresholding/<thresh_title>/SignInnov.nii          - warped to MNI
    Thresholding/<thresh_title>/mask_nonan.nii         - warped to MNI
    Thresholding/<thresh_title>/param.pkl              - updated for MNI space
    Thresholding/<thresh_title>/SignInnov.pkl           - 2D matrix in MNI space
    TotalActivation/Activity_inducing.nii              - warped to MNI
    TotalActivation/mask.nii                           - warped to MNI
    TotalActivation/param.pkl                          - updated for MNI space
    TotalActivation/Activity_inducing.pkl               - 2D matrix in MNI space
"""

import os
import glob
import pickle
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


# ── Configuration ─────────────────────────────────────────────────────────────

PATH_DATA = 'example data'
SUBJECTS = ['sub-10159', 'sub-10171']
THRESH_TITLE = 'Alpha_5_95_Fraction_0DOT05'
TITLE = 'pyexampleToolbox_openfMRI'

# MNI output bounding box and voxel size — matches MATLAB defaults
MNI_BB = np.array([[-78, -112, -70],
                    [ 78,   76,  85]])   # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
MNI_VOX = np.array([3.0, 3.0, 3.0])     # voxel size in mm


# ── Helper functions ──────────────────────────────────────────────────────────

def build_mni_affine(bb, vox):
    """
    Builds a NIfTI affine matrix for an MNI output grid defined by a
    bounding box and voxel size. Matches SPM's woptions.bb and woptions.vox.

    Inputs:
        bb  - (2 x 3) array [[xmin,ymin,zmin],[xmax,ymax,zmax]] in mm
        vox - (3,) array of voxel sizes in mm

    Outputs:
        affine - (4 x 4) affine mapping voxel indices to world mm
        shape  - (3,) output grid dimensions
    """
    origin = bb[0]
    shape = np.round((bb[1] - bb[0]) / vox).astype(int) + 1
    affine = np.diag(list(vox) + [1.0]).astype(float)
    affine[:3, 3] = origin
    return affine, shape


def apply_deformation(source_img, deform_img, out_affine, out_shape,
                      interpolation_order=1):
    """
    Warps source_img into MNI space using an SPM y_* deformation field.

    SPM y_* fields are defined in structural space and store absolute
    world-space MNI coordinates (mm) at each structural voxel —
    i.e. deform[x,y,z] = [mx, my, mz] tells you where structural voxel
    (x,y,z) maps to in MNI world space.

    The source image (functional) may have different dimensions from the
    deformation field (structural). The correct approach is:
        1. For each output MNI voxel, find its world-space mm position.
        2. Use the inverse deformation (structural -> MNI is known, so we
           interpolate the deformation field to get the structural voxel
           that maps nearest to our desired MNI position).

    Practical pull-warp approach:
        1. Convert each output MNI voxel to world mm via out_affine.
        2. Convert those world mm to structural voxel coords via
           inv(deform_affine) — this gives approximate structural coords.
        3. Interpolate the deformation field at those structural coords to
           get the actual MNI world position each structural voxel maps to.
        4. Convert structural voxel coords to functional voxel coords via
           inv(source_affine) @ deform_affine.
        5. Resample the source data at those functional coords.

    Inputs:
        source_img          - nibabel image in subject functional space
        deform_img          - nibabel y_* deformation field (X,Y,Z,1,3)
                              defined in structural space
        out_affine          - (4x4) affine for MNI output grid
        out_shape           - (3,) shape of MNI output grid
        interpolation_order - map_coordinates order (1=linear, 0=nearest)

    Outputs:
        out_data - ndarray of shape (*out_shape) or (*out_shape, T)
    """
    source_data = np.asarray(source_img.dataobj, dtype=np.float32)
    source_affine = source_img.affine

    is_4d = source_data.ndim == 4
    n_vols = source_data.shape[3] if is_4d else 1
    src_shape = source_data.shape[:3]

    # Load deformation field — squeeze (X,Y,Z,1,3) -> (X,Y,Z,3)
    deform_data = np.squeeze(np.asarray(deform_img.dataobj, dtype=np.float64))
    deform_affine = deform_img.affine
    deform_shape = deform_data.shape[:3]

    # Build output MNI grid: for each output voxel compute world mm position
    ox, oy, oz = np.meshgrid(
        np.arange(out_shape[0]),
        np.arange(out_shape[1]),
        np.arange(out_shape[2]),
        indexing='ij'
    )
    out_vox_flat = np.column_stack([ox.ravel(), oy.ravel(), oz.ravel()])
    ones = np.ones((out_vox_flat.shape[0], 1))
    out_world = (out_affine @ np.hstack([out_vox_flat, ones]).T)[:3].T  # (N_out, 3)

    # Convert output MNI world mm -> structural voxel coords
    # (approximate — deformation field lives in structural space)
    inv_deform_affine = np.linalg.inv(deform_affine)
    struct_vox = (inv_deform_affine @ np.hstack([out_world, ones]).T)[:3].T  # (N_out, 3)

    # Clip structural coords to deformation field bounds
    struct_coords = np.clip(
        struct_vox.T,
        [[0], [0], [0]],
        [[deform_shape[0] - 1], [deform_shape[1] - 1], [deform_shape[2] - 1]]
    )

    # Interpolate the deformation field at these structural positions
    # to get the MNI world coordinates each structural location maps to
    mni_world_interp = np.zeros((3, out_vox_flat.shape[0]))
    for dim in range(3):
        mni_world_interp[dim] = map_coordinates(
            deform_data[:, :, :, dim],
            struct_coords,
            order=1,
            mode='nearest'
        )
    mni_world_interp = mni_world_interp.T  # (N_out, 3)

    # Convert interpolated MNI world mm -> functional source voxel coords
    inv_src_affine = np.linalg.inv(source_affine)
    src_vox = (inv_src_affine @ np.hstack([mni_world_interp, ones]).T)[:3]  # (3, N_out)

    # Clip to functional source bounds
    src_coords = np.clip(
        src_vox,
        [[0], [0], [0]],
        [[src_shape[0] - 1], [src_shape[1] - 1], [src_shape[2] - 1]]
    )

    # Resample source data at computed functional voxel coordinates
    if is_4d:
        out_data = np.zeros((*out_shape, n_vols), dtype=np.float32)
        for t in range(n_vols):
            vol = source_data[:, :, :, t].astype(np.float64)
            warped = map_coordinates(vol, src_coords,
                                     order=interpolation_order,
                                     mode='constant', cval=0.0)
            out_data[:, :, :, t] = warped.reshape(out_shape)
    else:
        vol = source_data.astype(np.float64)
        warped = map_coordinates(vol, src_coords,
                                 order=interpolation_order,
                                 mode='constant', cval=0.0)
        out_data = warped.reshape(out_shape).astype(np.float32)

    return out_data


def warp_and_save(source_path, deform_path, out_path,
                  out_affine, out_shape, interp_order=4):
    """
    Warps a NIfTI file to MNI space and saves the result.

    Inputs:
        source_path  - str, path to input NIfTI
        deform_path  - str, path to y_* deformation field NIfTI
        out_path     - str, path for output NIfTI
        out_affine   - (4x4) MNI output affine
        out_shape    - (3,) MNI output grid shape
        interp_order - interpolation order (4=cubic, 1=linear, 0=nearest)
    """
    source_img = nib.load(source_path)
    deform_img = nib.load(deform_path)
    out_data = apply_deformation(source_img, deform_img,
                                 out_affine, out_shape, interp_order)
    nib.save(nib.Nifti1Image(out_data, out_affine), out_path)
    print(f"  Saved: {out_path}")


# ── Main normalization loop ───────────────────────────────────────────────────

def main():
    out_affine, out_shape = build_mni_affine(MNI_BB, MNI_VOX)
    print(f"MNI output grid: {out_shape}, voxel size: {MNI_VOX} mm")

    for subj in SUBJECTS:
        print(f"\nNormalizing subject: {subj}")

        subj_path = os.path.join(PATH_DATA, subj)
        thres_path = os.path.join(subj_path, 'TA_results', TITLE,
                                   'Thresholding', THRESH_TITLE)
        ta_path = os.path.join(subj_path, 'TA_results', TITLE,
                                'TotalActivation')
        input_path = os.path.join(subj_path, 'TA_results', TITLE,
                                   'inputData')
        out_thres = os.path.join(subj_path, 'TA_results', TITLE + '_MNI',
                                  'Thresholding', THRESH_TITLE)
        out_ta = os.path.join(subj_path, 'TA_results', TITLE + '_MNI',
                               'TotalActivation')
        os.makedirs(out_thres, exist_ok=True)
        os.makedirs(out_ta, exist_ok=True)

        # Find the SPM y_* deformation field
        deform_candidates = glob.glob(
            os.path.join(subj_path, 'anat', 'Segmented', 'y_*.nii')
        )
        if not deform_candidates:
            raise FileNotFoundError(
                f"No y_* deformation field found for subject {subj} in "
                f"{os.path.join(subj_path, 'anat', 'Segmented')}"
            )
        deform_path = deform_candidates[0]
        print(f"  Deformation field: {deform_path}")

        # ── Warp images ───────────────────────────────────────────────────────

        warp_and_save(
            os.path.join(thres_path, 'SignInnov.nii'), deform_path,
            os.path.join(out_thres, 'SignInnov.nii'),
            out_affine, out_shape, interp_order=1
        )
        warp_and_save(
            os.path.join(ta_path, 'Activity_inducing.nii'), deform_path,
            os.path.join(out_ta, 'Activity_inducing.nii'),
            out_affine, out_shape, interp_order=1
        )
        warp_and_save(
            os.path.join(input_path, 'mask.nii'), deform_path,
            os.path.join(out_ta, 'mask.nii'),
            out_affine, out_shape, interp_order=1
        )
        warp_and_save(
            os.path.join(thres_path, 'mask_nonan.nii'), deform_path,
            os.path.join(out_thres, 'mask_nonan.nii'),
            out_affine, out_shape, interp_order=1
        )

        # ── Update Thresholding param and SignInnov ───────────────────────────
        print("  Updating Thresholding param and SignInnov...")

        sign_innov_4d = nib.load(os.path.join(out_thres, 'SignInnov.nii')).get_fdata()
        mask_nonan_3d = np.squeeze(nib.load(os.path.join(out_thres, 'mask_nonan.nii')).get_fdata())
        mask_3d = np.squeeze(nib.load(os.path.join(out_ta, 'mask.nii')).get_fdata())

        # Binarise masks — handle NaN and interpolation float bleed
        mask_nonan_3d = (~np.isnan(mask_nonan_3d)) & (mask_nonan_3d > 0.5)
        mask_3d_bin = (~np.isnan(mask_3d)) & (mask_3d > 0.5)
        mask_nonan_1d = mask_nonan_3d.ravel(order='F')
        mask_1d = mask_3d_bin.ravel(order='F')

        with open(os.path.join(thres_path, 'param.pkl'), 'rb') as f:
            param_thres = pickle.load(f)

        param_thres['mask'] = mask_1d
        param_thres['Dimension'] = (mask_3d_bin.shape[0],
                                    mask_3d_bin.shape[1],
                                    mask_3d_bin.shape[2],
                                    param_thres['Dimension'][3])
        param_thres['mask_nonan'] = mask_nonan_1d

        for field in ('PC', 'mask_threshold1'):
            param_thres.pop(field, None)

        # 2D SignInnov in MNI: (n_frames x n_retained_voxels)
        sign_innov_2d = sign_innov_4d.reshape(-1, sign_innov_4d.shape[3], order='F')
        sign_innov_2d = sign_innov_2d[mask_nonan_1d, :].T

        # Mask 4D volume — set non-brain voxels to NaN
        sign_innov_4d = np.where(mask_nonan_3d[:, :, :, np.newaxis], sign_innov_4d, np.nan)

        with open(os.path.join(out_thres, 'param.pkl'), 'wb') as f:
            pickle.dump(param_thres, f)
        with open(os.path.join(out_thres, 'SignInnov.pkl'), 'wb') as f:
            pickle.dump(sign_innov_2d, f)
        nib.save(nib.Nifti1Image(sign_innov_4d.astype(np.float32), out_affine),
                 os.path.join(out_thres, 'SignInnov.nii'))

        # ── Update TotalActivation param and Activity_inducing ────────────────
        print("  Updating TotalActivation param and Activity_inducing...")

        ai_4d = nib.load(os.path.join(out_ta, 'Activity_inducing.nii')).get_fdata()

        with open(os.path.join(ta_path, 'param.pkl'), 'rb') as f:
            param_ta = pickle.load(f)

        param_ta['mask'] = mask_1d
        param_ta['mask_3D'] = mask_3d_bin
        param_ta['Dimension'] = (mask_3d_bin.shape[0],
                                  mask_3d_bin.shape[1],
                                  mask_3d_bin.shape[2],
                                  param_ta['Dimension'][3])
        param_ta['IND'] = np.where(mask_1d)[0]
        param_ta['VoxelIdx'] = np.column_stack(
            np.unravel_index(param_ta['IND'], mask_3d_bin.shape, order='F')
        ).astype(int)
        param_ta['NbrVoxels'] = int(np.sum(mask_1d))

        for field in ('GM_map', 'fHeader', 'weight_x', 'weight_y', 'weight_z',
                      'LambdaTemp', 'LambdaTempFin', 'NoiseEstimateFin',
                      'VoxelIdx_xyz'):
            param_ta.pop(field, None)

        # 2D Activity_inducing in MNI: (n_frames x n_retained_voxels)
        ai_2d = ai_4d.reshape(-1, ai_4d.shape[3], order='F')
        ai_2d = ai_2d[mask_1d, :].T

        # Mask 4D volume
        ai_4d = np.where(mask_3d_bin[:, :, :, np.newaxis], ai_4d, np.nan)

        with open(os.path.join(out_ta, 'param.pkl'), 'wb') as f:
            pickle.dump(param_ta, f)
        with open(os.path.join(out_ta, 'Activity_inducing.pkl'), 'wb') as f:
            pickle.dump(ai_2d, f)
        nib.save(nib.Nifti1Image(ai_4d.astype(np.float32), out_affine),
                 os.path.join(out_ta, 'Activity_inducing.nii'))

        print(f"  Done: {subj}")

    print("\nNormalization complete.")


if __name__ == "__main__":
    main()