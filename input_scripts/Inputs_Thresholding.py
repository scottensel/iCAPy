# Thresholding related information
####################################################
def setup_thresholding_params():
    """
    Returns
    -------
    param : dict
        Thresholding-specific keys, to be merged into an existing param dict.
    """
    param = {}

    # Force Thresholding to run even if results already exist
    param["force_Thresholding"] = 1

    # Alpha-levels for lower/upper innovation thresholds (percentiles)
    # first element is percentile of lower threshold - negative innovations
    # second element is percentile of upper threshold - positive innovations
    param["alpha"] = [5, 95]

    # Fraction of voxels from the ones entering total activation for a given
    # subject that should show an innovation at the same time point, so that
    # the corresponding frame is retained for iCAPs clustering
    param["f_voxels"] = 5.0 / 100.0

    # Title string used to label thresholding results and folder where it will be saved
    param["thresh_title"] = (
            "Alpha_"
            + str(param["alpha"][0]).replace(".", "DOT")
            + "_"
            + str(param["alpha"][1]).replace(".", "DOT")
            + "_Fraction_"
            + str(param["f_voxels"]).replace(".", "DOT")
    )

    # Number of neighbours that must also show an innovation for a voxel to be retained
    param["threshold_minclussize"] = 6

    # Number of neighbors to consider in the process
    param["threshold_interconnectivity"] = 26

    return param
