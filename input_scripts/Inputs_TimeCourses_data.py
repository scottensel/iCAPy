# Inputs_TimeCourses_Data_OpenfMRI.py

def setup_timecourses_data_params():
    """
    Python equivalent of Inputs_TimeCourses_Data_OpenfMRI.m

    Returns
    -------
    param : dict
        Data- and iCAP-related parameters for time-course regression (OpenfMRI example).
    """
    param = {}

    # General data information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Path where we have our data stored
    param["PathData"] = "example data"

    # Links towards the data of all subjects to analyze
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # List of subjects on which to run total activation
    # (cell array in MATLAB → list in Python)
    param["Subjects"] = ["sub-10159", "sub-10171"]

    # Number of subjects considered
    param["n_subjects"] = len(param["Subjects"])

    # Title that we wish to give to this specific run of the scripts for saving
    # data, or that was used previously for first steps and that we wish to
    # build on now
    param["title"] = "exampleToolbox_openfMRI_MNI"

    # name of the iCAPs output for this data
    # if only a subset of subjects should be included in the clustering, this
    # can be useful to save those different runs in different folders
    param["data_title"] = param["title"] + "_allSubjects"

    # information about which TA data should be used for regression:
    # thresholding information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Alpha-levels for lower/upper innovation thresholds (percentiles)
    param["alpha"] = [5, 95]

    # Fraction of voxels from the ones entering total activation for a given
    # subject that should show an innovation at the same time point, so that
    # the corresponding frame is retained for iCAPs clustering
    param["f_voxels"] = 5.0 / 100.0

    # Title used to create the folder where thresholding data will be saved
    alpha_low_str = str(param["alpha"][0]).replace(".", "DOT")
    alpha_high_str = str(param["alpha"][1]).replace(".", "DOT")
    f_voxels_str = str(param["f_voxels"]).replace(".", "DOT")
    param["thresh_title"] = (
        "Alpha_"
        + alpha_low_str
        + "_"
        + alpha_high_str
        + "_Fraction_"
        + f_voxels_str
    )

    # information about the iCAPs clustering for which regression should be done
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Number of iCAPs
    param["K"] = 20  # scalar in the MATLAB file

    # Distance Type for the K-means clustering (in MATLAB they choose between 'sqeuclidean' and 'cosine')
    param["DistType"] = "cosine"

    # Number of times the clustering process is run in a row to extract iCAPs
    param["n_folds"] = 10

    # Title used to create the folder where iCAPs data has been saved
    k_val = param["K"]
    dist = param["DistType"]
    n_folds = param["n_folds"]
    param["iCAPs_title"] = f"K_{k_val}_Dist_{dist}_Folds_{n_folds}"

    return param
