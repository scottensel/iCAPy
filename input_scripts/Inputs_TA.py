# Total activation related information
####################################################
def setup_ta_params():
    """
    Returns
    -------
    param : dict
        Only TA-specific keys. Meant to be merged into an
        existing param dict from setup_data_params().
    """
    # Total activation-related parameters
    param = {}

    # Force TA to run, even if files exist and overwrite
    param['force_TA_on_real'] = 1
    param['force_TA_on_surrogate'] = 1

    # Type of assumed hemodynamic response function
    # Options: 'bold', 'spmhrf', 'mion'
    param['HRF'] = 'mion'

    # Number of outer iterations for the forward-backward scheme (k_max in Farouj et al. 2016)
    param['Nit'] = 5

    # Number of iterations for temporal and spatial regularization schemes
    param['NitTemp'] = 500
    param['NitSpat'] = 400

    # Tolerance threshold for convergence of TA methods
    param['tol'] = 1e-6

    # Weighting parameters for temporal and spatial TA schemes:
    # because the temporal part is more elaborate and more extensively tested, Dr
    # Karahanoglu and Dr Farouj opted for a 3/4 to 1/4 trade-off =P
    param['weights'] = [0.75, 0.25]

    # Regularization weight multiplier for temporal regularization
    param['LambdaTempCoef'] = 1 / 0.8095

    # Regularization weight for spatial regularization
    param['LambdaSpat'] = 6

    # Weight matrix generation parameter for gray matter (GM) map
    # if the 'GM map difference in value' between two voxels is
    # equal to 0.5, then the corresponding weight will be
    # exp(-abs(0.5)/sigma) = exp(-1) = 0.3679. If the difference is
    # 1.0 (maximum possible), the weight will be exp(-2) = 0.1353
    param['sigma'] = 0.5

    return param
