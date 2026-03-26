import os


def check_icaps_files(data_path=None, iCAPs_path=None, cons_path=None, reg_path=None):
    # Aggregating_done check
    if data_path:
        if (os.path.isfile(os.path.join(data_path, 'AI.pkl')) and
                os.path.isfile(os.path.join(data_path, 'I_sig.pkl')) and
                os.path.isfile(os.path.join(data_path, 'final_mask.pkl')) and
                os.path.isfile(os.path.join(data_path, 'subject_labels.pkl')) and
                os.path.isfile(os.path.join(data_path, 'time_labels.pkl'))):
            aggregating_done = 1
        else:
            aggregating_done = 0
    else:
        aggregating_done = float('nan')

    # Clustering_done check
    if iCAPs_path:
        if (os.path.isfile(os.path.join(iCAPs_path, 'iCAPs.pkl')) and
                os.path.isfile(os.path.join(iCAPs_path, 'IDX.pkl'))):
            clustering_done = 1
        else:
            clustering_done = 0
    else:
        clustering_done = float('nan')

    # ConsensusClustering_done check
    if cons_path:
        if (os.path.isfile(os.path.join(cons_path, 'AUC.pkl')) and
                os.path.isfile(os.path.join(cons_path, 'CDF.pkl'))):
            consensus_clustering_done = 1
        else:
            consensus_clustering_done = 0
    else:
        consensus_clustering_done = float('nan')

    # Regression_ti_done and Regression_unc_done check
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
        regression_ti_done = float('nan')
        regression_unc_done = float('nan')

    return (aggregating_done, clustering_done, consensus_clustering_done, regression_ti_done, regression_unc_done)
