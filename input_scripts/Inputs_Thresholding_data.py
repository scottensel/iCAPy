# Thresholding related information
####################################################
def setup_thresholding_data_params():
    """
    Returns
    -------
    param : dict
        Data-related parameters for the thresholding step.
    """
    param = {}

    # 1. Parameters to be entered by the user
    # General data information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Path where we have our data stored
    param["PathData"] = 'F:/iCAP/Data/Python/Ketamine/Khali'

    # Links towards the data of all subjects to analyze
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # List of subjects on which you ran total activation on
    # This is where the TA folder will be created (or looked for) for each subject
    param["Subjects"] = ["run5_20130209", "run5_20130308", "run7_20130322"]

    # Number of subjects considered
    param["n_subjects"] = len(param["Subjects"])

    # Title that we wish to give to this specific run of the scripts for saving data,
    # or that was used previously for first steps and that we wish to build on now
    param["title"] = "test3"

    return param
