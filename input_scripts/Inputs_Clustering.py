# input_scripts/Inputs_Clustering.py

def setup_clustering_params():
    """
    Python equivalent of Inputs_Clustering.m.

    Returns
    -------
    param : dict
        Only clustering-specific keys. Meant to be merged into an
        existing param dict from setup_data_params().
    """
    param = {}

    # iCAPs-related information
    # -------------------------------------------------------
    # specify if clustering should be done (set to 0 to only run consensus)
    param["doClustering"] = 1

    # force recomputation even if outputs already exist
    param["force_Aggregating"] = 0
    param["force_Clustering"] = 0

    # external mask defining input voxels (if None, intersection of all GMs)
    param["common_mask_file"] = None

    # extra mask applied on top of common GM mask (e.g. to exclude cerebellum)
    param["extra_mask_file"] = "GM_mask_MNI333_AAL.nii"

    # Number of iCAPs K
    # MATLAB: param.K = 19:21;
    param["K"] = list(range(19, 22))  # [19, 20, 21]

    # Distance type for k-means: 'sqeuclidean' or 'cosine'
    param["DistType"] = "cosine"

    # Number of folds (repeats) of the clustering
    param["n_folds"] = 10

    # Save subject-specific maps & region tables
    param["saveClusterReplicateData"] = 0
    param["MaxIter"] = 300

    param["saveSubjectMaps"] = 1

    param["saveRegionTables"] = 0
    param["regTab_thres"] = 1.5  # z-score threshold
    param["regTab_codeBook"] = "AALcodeBook.mat"
    param["regTab_atlasFile"] = "AAL90_correctLR.nii"

    # Title(s) used to create the folder where iCAPs data will be saved
    # MATLAB:
    # for nK=1:length(param.K)
    #     param.iCAPs_title{nK} = ['K_',num2str(param.K(nK)),'_Dist_',...
    #             param.DistType,'_Folds_',num2str(param.n_folds)];
    # end
    # if length(param.K)==1
    #     param.iCAPs_title=param.iCAPs_title{1};
    # end
    k_list = param["K"]
    dist = param["DistType"]
    n_folds = param["n_folds"]

    iCAPs_title_list = [
        f"K_{k}_Dist_{dist}_Folds_{n_folds}" for k in k_list
    ]
    if len(iCAPs_title_list) == 1:
        param["iCAPs_title"] = iCAPs_title_list[0]
    else:
        param["iCAPs_title"] = iCAPs_title_list

    # Consensus clustering parameters
    # -------------------------------------------------------
    param["doConsensusClustering"] = 0
    param["force_ConsensusClustering"] = 0

    # 'items' to subsample frames regardless of subject boundaries
    param["Subsample_type"] = "items"
    param["Subsample_fraction"] = 0.8
    param["cons_n_folds"] = 20

    # MATLAB:
    # param.cons_title=[num2str(param.K(1)) 'to' num2str(param.K(end)) ...
    #   '_SubsampleType_' param.Subsample_type ...
    #   '_Fraction_' strrep(num2str(param.Subsample_fraction),'.','DOT') ...
    #   '_nFolds_' num2str(param.cons_n_folds) ...
    #   '_Dist_' param.DistType];
    k_min = k_list[0]
    k_max = k_list[-1]
    frac_str = str(param["Subsample_fraction"]).replace(".", "DOT")
    cons_title = (
        f"{k_min}to{k_max}"
        f"_SubsampleType_{param['Subsample_type']}"
        f"_Fraction_{frac_str}"
        f"_nFolds_{param['cons_n_folds']}"
        f"_Dist_{dist}"
    )
    param["cons_title"] = cons_title

    # flag to indicate that cluster consensus should be computed
    param["computeClusterStability"] = 1

    return param
