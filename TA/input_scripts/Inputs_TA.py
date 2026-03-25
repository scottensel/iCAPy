def setup_ta_params():
    # Total activation-related parameters
    param = {}

    # Force TA to run, even if files exist
    param['force_TA_on_real'] = 1
    param['force_TA_on_surrogate'] = 1

    # Type of assumed hemodynamic response function
    # Options: 'bold', 'spmhrf', 'mion'
    param['HRF'] = 'mion'

    # Number of outer iterations for the forward-backward scheme (k_max in Farouj et al. 2016)
    param['Nit'] = 2 #5

    # Number of iterations for temporal and spatial regularization schemes
    param['NitTemp'] = 3 #500
    param['NitSpat'] = 3 #400

    # Tolerance threshold for convergence
    param['tol'] = 1e-6

    # Weighting parameters for temporal and spatial TA schemes
    param['weights'] = [0.75, 0.25]

    # Regularization weight multiplier for temporal regularization
    param['LambdaTempCoef'] = 1 / 0.8095

    # Regularization weight for spatial regularization
    param['LambdaSpat'] = 6

    # Weight matrix generation parameter for gray matter (GM) map
    param['sigma'] = 0.5

    return param
