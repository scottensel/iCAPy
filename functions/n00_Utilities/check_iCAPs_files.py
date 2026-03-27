import os
import math


def check_icaps_files(data_path=None, iCAPs_path=None, cons_path=None, reg_path=None):
    """
    Checks which steps of the iCAPs pipeline have already been completed
    for the given output directories, by looking for the expected output
    files in each folder. Returns nan for any path that is not provided.

    Inputs:
        data_path   - str or None, path to the aggregated data directory;
                      checked for AI, I_sig, final_mask, subject_labels,
                      time_labels pkl files
        iCAPs_path  - str or None, path to the clustering output directory;
                      checked for iCAPs and IDX pkl files
        cons_path   - str or None, path to the consensus clustering output
                      directory; checked for AUC and CDF pkl files
        reg_path    - str or None, path to the regression output directory;
                      checked for TC/TC_stats (transient-informed) and
                      TC_unc/TC_unc_stats (unconstrained) pkl files

    Outputs:
        tuple of five values (aggregating_done, clustering_done,
                               consensus_clustering_done,
                               regression_ti_done, regression_unc_done):
            1   - step has been completed (all expected files present)
            0   - step has not been completed
            nan - path was not provided; result is undefined
    """

    # Check whether significant innovations and activity-inducing signals
    # have been aggregated across subjects
    if data_path:
        if (os.path.isfile(os.path.join(data_path, 'AI.pkl'))            and
                os.path.isfile(os.path.join(data_path, 'I_sig.pkl'))         and
                os.path.isfile(os.path.join(data_path, 'final_mask.pkl'))    and
                os.path.isfile(os.path.join(data_path, 'subject_labels.pkl')) and
                os.path.isfile(os.path.join(data_path, 'time_labels.pkl'))):
            aggregating_done = 1
        else:
            aggregating_done = 0
    else:
        aggregating_done = math.nan

    # Check whether k-means clustering has been run and iCAPs saved
    if iCAPs_path:
        if (os.path.isfile(os.path.join(iCAPs_path, 'iCAPs.pkl')) and
                os.path.isfile(os.path.join(iCAPs_path, 'IDX.pkl'))):
            clustering_done = 1
        else:
            clustering_done = 0
    else:
        clustering_done = math.nan

    # Check whether consensus clustering has been run (AUC and CDF saved)
    if cons_path:
        if (os.path.isfile(os.path.join(cons_path, 'AUC.pkl')) and
                os.path.isfile(os.path.join(cons_path, 'CDF.pkl'))):
            consensus_clustering_done = 1
        else:
            consensus_clustering_done = 0
    else:
        consensus_clustering_done = math.nan

    # Check whether regression has been run:
    # - transient-informed regression: TC.pkl and TC_stats.pkl
    # - unconstrained regression:      TC_unc_.pkl and TC_unc_stats.pkl
    if reg_path:
        if (os.path.isfile(os.path.join(reg_path, 'TC.pkl')) and
                os.path.isfile(os.path.join(reg_path, 'TC_stats.pkl'))):
            regression_ti_done = 1
        else:
            regression_ti_done = 0

        if (os.path.isfile(os.path.join(reg_path, 'TC_unc_.pkl')) and
                os.path.isfile(os.path.join(reg_path, 'TC_unc_stats.pkl'))):
            regression_unc_done = 1
        else:
            regression_unc_done = 0
    else:
        regression_ti_done  = math.nan
        regression_unc_done = math.nan

    return (aggregating_done, clustering_done, consensus_clustering_done,
            regression_ti_done, regression_unc_done)
