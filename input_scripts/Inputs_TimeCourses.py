# Regression related information
####################################################

def setup_timecourses_params():
    """
    Returns
    -------
    param : dict
        Regression / iCAPs time-course specific keys.
    """
    param = {}

    # Force regression to run even if results already exist
    param["force_Regression"] = 1

    # Type of regression:
    #   'unconstrained' or 'transient-informed' (recommended)
    param["regType"] = "transient-informed"

    # parameter for soft cluster assignment in transient-informed regression
    # MATLAB: param.softClusterThres = [1:0.2:2];
    # That is: 1, 1.2, ..., 2.0
    param["softClusterThres"] = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


    # Evaluate amplitude correlations as part of soft cluster factor evaluation
    param["evalAmplitudeCorrs"] = 0

    # Threshold above which a z-scored iCAP time course is considered "active"
    param["activityThres"] = 1.0

    return param
