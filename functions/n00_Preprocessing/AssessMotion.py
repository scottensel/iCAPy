import os
import numpy as np
from functions.n00_Utilities.WriteInformation import write_information


def assess_motion(path, soi, param, fid=None):
    """
    Evaluates the extent of motion of the considered subject in the scanner.
    Different methods exist to quantify motion; the desired approach is
    selected via param['FD_method']. Currently only 'Power' is implemented.

    Inputs:
        path  - str, full path towards the subject folder containing the
                motion file subfolder
        soi   - int, 0-based subject index; used when param fields are
                lists with one entry per subject
        param - dict with all relevant parameters; must contain:
            'Folder_motion'           - name of the folder containing the
                                        motion file; None / [] if directly
                                        in path; can be a list (one per
                                        subject)
            'TA_mot_prefix'           - prefix of the motion text file to
                                        read; can be a list (one per subject)
            'skipped_scans'           - number of scans to skip at the
                                        beginning due to T1 equilibration
            ['skipped_scans_motionfile'] - number of lines to skip at the
                                        start of the motion file; defaults
                                        to 'skipped_scans' if not set
            'FD_method'               - scrubbing method; only 'Power' is
                                        currently implemented
            'FD_threshold'            - float, FD threshold in mm; frames
                                        with FD above this are scrubbed
        fid   - optional log file handle for write_information

    Outputs:
        TemporalMask - (n_frames,) boolean array; True if a frame is kept,
                       False if it should be scrubbed out due to excessive
                       motion

    Reference (FD Power method):
        Power et al. (2012). Spurious but systematic correlations in
        functional connectivity MRI networks arise from subject motion.
        NeuroImage 59, 2142-2154.
    """
    # Resolve per-subject fields — use subject-specific entry if a list
    # is provided, otherwise use the single shared value
    param['Folder_motion'] = (param['Folder_motion'][soi]
                              if isinstance(param['Folder_motion'], list)
                              else param['Folder_motion'])
    param['TA_mot_prefix'] = (param['TA_mot_prefix'][soi]
                              if isinstance(param['TA_mot_prefix'], list)
                              else param['TA_mot_prefix'])

    # If skipped_scans_motionfile is not set, default to skipped_scans
    param.setdefault('skipped_scans_motionfile', param['skipped_scans'])

    # Check that the motion folder exists
    motion_folder = os.path.join(path, param['Folder_motion'])
    if not os.path.isdir(motion_folder):
        raise FileNotFoundError("There is no motion folder...")

    # Locate and read the motion file (must be exactly one matching .txt file)
    motion_file = next(
        (f for f in os.listdir(motion_folder)
         if f.startswith(param['TA_mot_prefix']) and f.endswith('.txt')),
        None
    )
    if not motion_file:
        raise FileNotFoundError("More than 1 motion file (or no motion file)...")

    RP = np.loadtxt(os.path.join(motion_folder, motion_file))

    # Remove the first scans (T1 equilibration frames)
    if int(param['skipped_scans_motionfile']) > 0:
        RP = RP[int(param['skipped_scans_motionfile']):]

    # Convert rotational displacement values from radians to degrees
    # (SPM outputs rotations in radians; MATLAB multiplies by 180/pi)
    RP[:, 3:6] *= (180.0 / np.pi)

    if param['FD_method'] == 'Power':

        # Compute frame-to-frame change in motion (prepend a row of zeros
        # so that the first frame has FD = 0, matching MATLAB's convention)
        RPDiff = np.vstack([np.zeros((1, 6)), np.diff(RP, axis=0)])

        # Convert rotational differences to mm displacement on the surface
        # of a sphere with radius 50 mm:
        # arc length = theta_degrees * (pi/180) * r = theta_rad * 50
        # MATLAB: RPDiffSphere(:,4:6) = RPDiffSphere(:,4:6) * 50*pi/180
        RPDiff[:, 3:6] *= (50.0 * np.pi / 180.0)

        # FD Power is the sum of absolute displacements across all 6 parameters
        FD_Power = np.sum(np.abs(RPDiff), axis=1)

        # Average and percentage of scrubbed frames (for logging)
        MeanFD_Power    = float(FD_Power.mean())
        PercentFD_Power = float(np.sum(FD_Power > param['FD_threshold']) /
                                len(FD_Power) * 100.0)

        # Temporal mask: True = keep, False = scrub
        TemporalMask = FD_Power <= param['FD_threshold']

        write_information(fid, f"Average motion:  {MeanFD_Power:.4f} [mm]")
        write_information(fid, f"Motion: there are {PercentFD_Power:.2f} percent of data scrubbed")

    else:
        raise ValueError(
            "Scrubbing method not recognized — only FD Power is implemented so far."
        )

    return TemporalMask
