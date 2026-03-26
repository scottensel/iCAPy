import os

def check_ta_files(path, thresholding_title=None):
    if thresholding_title is None:
        thresholding_title = ''
        thresholding_done = float('nan')
    else:
        thresholding_done = None  # Placeholder, to be determined below

    # Check for Total Activation on real data
    if (os.path.isfile(os.path.join(path, 'TotalActivation', 'Activity_related.pkl')) and
        os.path.isfile(os.path.join(path, 'TotalActivation', 'Activity_inducing.pkl')) and
        os.path.isfile(os.path.join(path, 'TotalActivation', 'Innovation.pkl')) and
        os.path.isfile(os.path.join(path, 'TotalActivation', 'param.pkl'))):
        ta_on_real = 1
    else:
        ta_on_real = 0

    # Check for Total Activation on surrogate data
    if (os.path.isfile(os.path.join(path, 'Surrogate', 'Activity_related_surrogate.pkl')) and
        os.path.isfile(os.path.join(path, 'Surrogate', 'Activity_inducing_surrogate.pkl')) and
        os.path.isfile(os.path.join(path, 'Surrogate', 'Innovation_surrogate.pkl'))):
        ta_on_surrogate = 1
    elif os.path.isfile(os.path.join(path, 'Surrogate', 'Surrogate.pkl')):
        ta_on_surrogate = 2
    else:
        ta_on_surrogate = 0

    # Check if thresholding files exist
    if thresholding_title:
        if (os.path.isfile(os.path.join(path, 'Thresholding', thresholding_title, 'SignInnov.pkl')) and
            os.path.isfile(os.path.join(path, 'Thresholding', thresholding_title, 'param.pkl'))):
            thresholding_done = 1
        else:
            thresholding_done = 0

    return ta_on_real, ta_on_surrogate, thresholding_done
