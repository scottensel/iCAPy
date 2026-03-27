import os
import numpy as np
import nibabel as nib
from scipy.io import loadmat


def save_region_tables(param, iCAPs_z, final_mask):
    """Python translation of saveRegionTables.m

    Writes a region-wise table for each iCAP based on an atlas and a codeBook.

    Parameters
    ----------
    param : dict-like
        Must contain:
        - 'regTab_thres': z-threshold for active voxels
        - 'regTab_codeBook': path to a .mat file containing 'codeBook'
        - 'regTab_atlasFile': path to atlas NIfTI file
        - 'outDir_iCAPs': output directory for region tables
    iCAPs_z : ndarray, shape (nClus, n_vox)
        Z-scored iCAP maps.
    final_mask : array_like, shape (n_vox,)
        Boolean or 0/1 mask defining voxels used in clustering.
    """
    if not all(k in param for k in ('regTab_thres', 'regTab_codeBook', 'regTab_atlasFile')):
        print("attempt to write regions table: no threshold, codeBook or atlas file defined, skipping!")
        return

    # Load codeBook
    mat = loadmat(param['regTab_codeBook'], squeeze_me=True, struct_as_record=False)
    if 'codeBook' not in mat:
        raise ValueError("saveRegionTable: 'codeBook' not found in codeBook file.")
    codeBook = mat['codeBook']

    nReg = int(codeBook.num)
    nClus = iCAPs_z.shape[0]

    # Read atlas
    atlas_img = nib.load(param['regTab_atlasFile'])
    atlas_vol = atlas_img.get_fdata()

    # Reshape atlas to 1D for masked voxels
    atlas_1d = atlas_vol.reshape(-1, order='F')
    final_mask = np.asarray(final_mask).astype(bool)
    atlas_masked = atlas_1d[final_mask]

    out_dir = os.path.join(param['outDir_iCAPs'], 'regTables')
    os.makedirs(out_dir, exist_ok=True)
    tab_path = os.path.join(out_dir, 'iCAPs_regions_table.txt')

    with open(tab_path, 'w', encoding='utf-8') as tab_f:
        # Header
        tab_f.write("RegID	RegNames	")
        for iC in range(1, nClus + 1):
            tab_f.write(f"iCAP{iC}_meanZ	iCAP{iC}_nVox	iCAP{iC}_percVox	")
        tab_f.write("\n")

        # For each region
        region_labels = np.asarray(codeBook.id).astype(int)
        for iR in range(nReg):
            reg_id = int(region_labels[iR])
            reg_mask = atlas_masked == reg_id
            if not np.any(reg_mask):
                continue

            # Write region name
            rname = str(codeBook.rname[iR])
            name = str(codeBook.name[iR])
            tab_f.write(f"{reg_id}	{rname}	{name}")

            # For each cluster, compute stats
            for iC in range(nClus):
                icap_vals = iCAPs_z[iC, :]
                vals_reg = icap_vals[reg_mask]
                nVox = vals_reg.size
                if nVox == 0:
                    mean_z = 0.0
                    perc = 0.0
                else:
                    mean_z = float(np.mean(vals_reg))
                    perc = 100.0 * nVox / float(np.sum(final_mask))
                tab_f.write(f"	{mean_z:.2f}	{nVox:d}	{perc:.2f}")
            tab_f.write("\n")
