import numpy as np
from scipy.interpolate import interp1d
from functions.n00_Utilities.WriteInformation import write_information


def interpolate_time_courses(fData, TemporalMask, param, fid=None):
    """
    Interpolates the fMRI volumes that are scrubbed out from the data,
    using spline interpolation (or another method specified in param).

    If the first or last frame is excluded, all consecutive excluded frames
    at that boundary are also removed from the mask entirely (not just set
    to False) because there are no valid frames on one side to interpolate
    from. A gap threshold of 2 is used — if only one valid frame remains
    between excluded frames at the boundary, it is also excluded.

    Inputs:
        fData        - (n_ret_vox x n_tp) 2D array of functional data
        TemporalMask - (n_tp,) boolean array; False = frame to be scrubbed
        param        - dict containing TA parameters:
            ['interType'] - str, interpolation method passed to scipy
                            interp1d; default 'spline' (mapped to 'cubic')
        fid          - optional log file handle for write_information

    Outputs:
        TC           - (n_ret_vox x n_tp) 2D array after interpolation
        TemporalMask - updated mask with boundary frames removed
    """
    # Default interpolation type matches MATLAB's interp1(...,'spline')
    param.setdefault('interType', 'spline')

    # scipy uses 'cubic' for what MATLAB calls 'spline' in interp1
    interp_kind = 'cubic' if param['interType'] == 'spline' else param['interType']

    nTP = len(TemporalMask)

    # Indices of excluded (scrubbed) frames
    indd = np.where(~TemporalMask)[0]

    if len(indd) > 0:
        # If the first frame is excluded, remove all leading excluded frames
        # up to the first gap > 2 in the excluded indices.
        # MATLAB: TemporalMask(excludeTP) = [] (deletes elements entirely)
        # Python equivalent: trim the mask and data to the valid range
        if indd[0] == 0:
            diff_mask  = np.diff(indd)
            first_gap  = np.where(diff_mask > 2)[0]
            if len(first_gap) > 0:
                exclude_end = indd[first_gap[0] + 1]   # first index to keep
            else:
                exclude_end = nTP                       # exclude all
            TemporalMask = TemporalMask[exclude_end:]
            fData        = fData[:, exclude_end:]
            write_information(
                fid,
                "Excluding first frames from interpolation because there "
                "are no non-motion frames before..."
            )
            # Recompute after trimming
            nTP  = len(TemporalMask)
            indd = np.where(~TemporalMask)[0]

        # If the last frame is excluded, remove all trailing excluded frames
        # from the last gap > 2 in the excluded indices onward
        if len(indd) > 0 and indd[-1] == nTP - 1:
            diff_mask = np.diff(indd)
            last_gap  = np.where(diff_mask > 2)[0]
            if len(last_gap) > 0:
                exclude_start = indd[last_gap[-1] + 1]  # first excluded index to trim
            else:
                exclude_start = indd[0]
            TemporalMask = TemporalMask[:exclude_start]
            fData        = fData[:, :exclude_start]
            write_information(
                fid,
                "Excluding last frames from interpolation because there "
                "are no non-motion frames afterwards..."
            )
            nTP = len(TemporalMask)

    # TCon contains the values of the data points that we know (not scrubbed)
    TCon = fData[:, TemporalMask]

    # tinter is all time points; torig is the subset that we know
    tinter = np.arange(nTP)
    torig  = tinter[TemporalMask]

    # Interpolate across all time points using the known data points
    # (equivalent to MATLAB's interp1(torig, TCon', tinter, param.interType)')
    interpolator = interp1d(
        torig, TCon, kind=interp_kind,
        fill_value="extrapolate", axis=1
    )
    TC = interpolator(tinter)

    write_information(fid, f"{param['interType']} interpolation finished successfully...")

    return TC, TemporalMask
