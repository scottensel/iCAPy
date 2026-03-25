import numpy as np
import os


def assess_motion(path, soi, param, fid=None):
    # Handle optional parameters and compatibility checks
    param['Folder_motion'] = param['Folder_motion'][soi] if isinstance(param['Folder_motion'], list) else param[
        'Folder_motion']
    param['TA_mot_prefix'] = param['TA_mot_prefix'][soi] if isinstance(param['TA_mot_prefix'], list) else param[
        'TA_mot_prefix']
    param.setdefault('skipped_scans_motionfile', param['skipped_scans'])

    motion_folder = os.path.join(path, param['Folder_motion'])
    if not os.path.isdir(motion_folder):
        raise FileNotFoundError("Motion folder does not exist.")

    # Locate and load the motion file
    motion_file = next(
        (f for f in os.listdir(motion_folder) if f.startswith(param['TA_mot_prefix']) and f.endswith('.txt')), None)
    if not motion_file:
        raise FileNotFoundError("Motion file not found or multiple motion files present.")

    RP = np.loadtxt(os.path.join(motion_folder, motion_file))
    RP = RP[param['skipped_scans_motionfile']:]  # Remove initial scans

    # Convert rotational displacements from radians to mm
    RP[:, 3:6] *= (180 / np.pi)

    if param['FD_method'] == 'Power':
        # Compute frame-to-frame displacement and apply FD Power
        RPDiff = np.vstack([np.zeros((1, 6)), np.diff(RP, axis=0)])
        RPDiff[:, 3:6] *= (50 * np.pi / 180)  # Radius of 50mm sphere
        FD_Power = np.sum(np.abs(RPDiff), axis=1)

        # Generate temporal mask based on FD threshold
        TemporalMask = FD_Power <= param['FD_threshold']

        # Log average and percentage of scrubbed frames
        if fid:
            mean_fd_power = FD_Power.mean()
            percent_scrubbed = np.mean(~TemporalMask) * 100
            fid.write(f"Average motion: {mean_fd_power:.4f} mm\n")
            fid.write(f"{percent_scrubbed:.2f}% of frames scrubbed\n")

    else:
        raise ValueError("Only 'Power' method for FD is implemented.")

    return TemporalMask
