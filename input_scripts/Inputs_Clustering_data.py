# Clustering related information
####################################################
def setup_clustering_data_params():
    """
    Returns
    -------
    param : dict
        Data- and thresholding-related parameters for clustering.
    """
    param = {}

    # General data information
    # ------------------------
    # path where data is stored
    param["PathData"] = 'F:/iCAP/Data/Python/Ketamine/Khali'

    # Links towards the data of all subjects to analyze
    # -------------------------------------------------
    param["Subjects"] = ["run5_20130209", "run5_20130308", "run7_20130322"]
    param["n_subjects"] = len(param["Subjects"])

    # Title for this run (must match the title used for TA/thresholding)
    # This is for this specific run of the scripts and it from prior steps
    # to now build on that information
    param["title"] = "test_spatial"

    # Thresholding-related information (used when selecting frames for iCAPs)
    # -----------------------------------------------------------------------
    # Alpha-levels for lower/upper innovation thresholds (percentiles)
    # first element is percentile of lower threshold - negative innovations
    # second element is percentile of upper threshold - positive innovations
    param["alpha"] = [5, 95]

    # Fraction of voxels that must show innovation at the same time point
    param["f_voxels"] = 5.0 / 100.0

    # Title for thresholding results folder
    alpha_low_str = str(param["alpha"][0]).replace(".", "DOT")
    alpha_high_str = str(param["alpha"][1]).replace(".", "DOT")
    f_voxels_str = str(param["f_voxels"]).replace(".", "DOT")

    # title used to create folder where thresholding is saved
    param["thresh_title"] = (
        "Alpha_"
        + alpha_low_str
        + "_"
        + alpha_high_str
        + "_Fraction_"
        + f_voxels_str
    )

    return param
