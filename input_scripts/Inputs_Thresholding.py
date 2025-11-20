# input_scripts/Inputs_Thresholding.py

def setup_thresholding_params():
    """
    Python equivalent of Inputs_Thresholding.m.

    Returns
    -------
    param : dict
        Thresholding-specific keys, to be merged into an existing param dict.
    """
    param = {}

    # Force Thresholding to run even if results already exist
    param["force_Thresholding"] = 1

    # Alpha-levels for lower/upper innovation thresholds (percentiles)
    # MATLAB: param.alpha = [5 95];
    param["alpha"] = [5, 95]

    # Fraction of voxels that must show innovation at the same time point
    # MATLAB: param.f_voxels = 5/100;
    param["f_voxels"] = 5.0 / 100.0

    # Title string used to label thresholding results
    # MATLAB roughly:
    # param.thresh_title = ['Alpha_',strrep(num2str(param.alpha(1)),'.','DOT'),'_',...]
    alpha_low = str(param["alpha"][0]).replace(".", "DOT")
    alpha_high = str(param["alpha"][1]).replace(".", "DOT")
    param["thresh_title"] = f"Alpha_{alpha_low}_{alpha_high}"

    # Parameters for checking spatially interconnected frames
    param["threshold_minclussize"] = 6
    param["threshold_interconnectivity"] = 26

    return param
