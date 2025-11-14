import numpy as np
from scipy.interpolate import interp1d

def interpolate_time_courses(fData, TemporalMask, param, fid=None):
    param.setdefault('interType', 'spline')
    nTP = len(TemporalMask)

    # Exclude first and last frames from interpolation if no valid frames are nearby
    indices = np.where(~TemporalMask)[0]
    if indices[0] == 0:
        cut_off = np.where(np.diff(indices) > 2)[0][0] + 1
        TemporalMask[:indices[cut_off]] = False
        if fid:
            fid.write("Excluding initial frames from interpolation.\n")
    if indices[-1] == nTP - 1:
        cut_off = np.where(np.diff(indices) > 2)[0][-1] + 1
        TemporalMask[indices[cut_off]:] = False
        if fid:
            fid.write("Excluding final frames from interpolation.\n")

    # Prepare data for interpolation
    known_data = fData[:, TemporalMask]
    known_times = np.arange(nTP)[TemporalMask]
    all_times = np.arange(nTP)

    # Interpolate using the specified method
    interpolator = interp1d(known_times, known_data, kind=param['interType'], fill_value="extrapolate", axis=1)
    TC = interpolator(all_times)

    # Log interpolation type
    if fid:
        fid.write(f"{param['interType']} interpolation completed.\n")

    return TC, TemporalMask
