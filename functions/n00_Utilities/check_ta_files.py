import os
import math


def check_ta_files(path, thresholding_title=None):
    """
    Checks which steps of the total activation pipeline have already been
    completed for a given subject path, by looking for the expected output
    files in each subfolder.

    Inputs:
        path               - str, path to the subject's TA results folder
                             (i.e. TA_results/<title> inside the subject dir)
        thresholding_title - str or None, name of the thresholding subfolder
                             to check; if None, thresholding_done is nan

    Outputs:
        tuple of three values (ta_on_real, ta_on_surrogate, thresholding_done):

        ta_on_real:
            1 - TA on real data completed (Activity_related, Activity_inducing,
                Innovation, and param pkl files all present)
            0 - TA on real data not completed

        ta_on_surrogate:
            1 - surrogate data computed AND TA run on it (Activity_related,
                Activity_inducing, and Innovation surrogate pkl files present)
            2 - surrogate data computed but TA not yet run
                (Surrogate.pkl present but TA outputs missing)
            0 - surrogate data not computed

        thresholding_done:
            1   - thresholding completed (SignInnov.pkl and param.pkl present)
            0   - thresholding not completed
            nan - thresholding_title was not provided
    """

    if thresholding_title is None:
        thresholding_title = ''
        thresholding_done  = math.nan
    else:
        thresholding_done = None   # will be determined below

    # Check for total activation on real data
    if (os.path.isfile(os.path.join(path, 'TotalActivation', 'Activity_related.pkl'))   and
            os.path.isfile(os.path.join(path, 'TotalActivation', 'Activity_inducing.pkl')) and
            os.path.isfile(os.path.join(path, 'TotalActivation', 'Innovation.pkl'))       and
            os.path.isfile(os.path.join(path, 'TotalActivation', 'param.pkl'))):
        ta_on_real = 1
    else:
        ta_on_real = 0

    # Check for total activation on surrogate data.
    # Returns 1 if TA has been run on the surrogate, 2 if the surrogate has
    # been generated but TA not yet run, and 0 if nothing exists.
    if (os.path.isfile(os.path.join(path, 'Surrogate', 'Activity_related_surrogate.pkl'))   and
            os.path.isfile(os.path.join(path, 'Surrogate', 'Activity_inducing_surrogate.pkl')) and
            os.path.isfile(os.path.join(path, 'Surrogate', 'Innovation_surrogate.pkl'))):
        ta_on_surrogate = 1
    elif os.path.isfile(os.path.join(path, 'Surrogate', 'Surrogate.pkl')):
        ta_on_surrogate = 2
    else:
        ta_on_surrogate = 0

    # Check whether thresholding has been completed for the given title
    if thresholding_title:
        if (os.path.isfile(os.path.join(path, 'Thresholding', thresholding_title, 'SignInnov.pkl')) and
                os.path.isfile(os.path.join(path, 'Thresholding', thresholding_title, 'param.pkl'))):
            thresholding_done = 1
        else:
            thresholding_done = 0

    return ta_on_real, ta_on_surrogate, thresholding_done
