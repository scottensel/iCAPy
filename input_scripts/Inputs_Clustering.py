# Clustering related information
####################################################
def setup_clustering_params():
    """
    Returns
    -------
    param : dict
        Only clustering-specific keys. Meant to be merged into an
        existing param dict from setup_data_params().
    """
    param = {}

    # performance Information
    # only matters if using Windows
    # kmeans has a memory leak if you have too many threads
    # this is not a problem for low number of subjects but for large number this may be a problem
    param["limitThreads"] = 4

    # iCAPs-related information
    # -------------------------------------------------------
    # specify if clustering should be done (set to 0 to only run consensus)
    param["doClustering"] = 0

    # force recomputation even if outputs already exist
    param["force_Aggregating"] = 0
    param["force_Clustering"] = 0

    # external mask defining input voxels (if None, intersection of all GMs)
    param["common_mask_file"] = None

    # extra mask applied on top of common GM mask (e.g. to exclude cerebellum)
    param["extra_mask_file"] = "GM_mask_MNI333_AAL.nii"

    # Number of iCAPs K
    param["K"] = [x for x in range(2, 5)]

    if 0 in param["K"]:
        raise ValueError("The list of K contains 0")

    # Distance type for k-means: 'sqeuclidean' or 'cosine'
    param["DistType"] = "cosine"

    # Number of folds (repeats) of the clustering
    param["n_folds"] = 10

    # Specificy if the result of each replicate should be saved during clustering
    param["saveClusterReplicateData"] = 0

    # Maximum number of allowed iterations of the kmeans clustering
    # default is 100 and is sometimes not enough if many frames are included,
    param["MaxIter"] = 300

    # save subject specific maps
    param["saveSubjectMaps"] = 1
    param["force_saveSubjectMaps"] = 1

    # save iCAP region tables
    ###########################################################
    # this section not tested or updated so may not work properly
    ###########################################################
    param["saveRegionTables"] = 0
    param["regTab_thres"] = 1.5  # z-score threshold
    param["regTab_codeBook"] = "AALcodeBook.mat"
    param["regTab_atlasFile"] = "AAL90_correctLR.nii"

    # Title(s) used to create the folder where iCAPs data will be saved
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
    # if consesus clustering should be run or not
    param["doConsensusClustering"] = 1

    # Consensus clustering forced to run or not
    # it will not remake the Consenus matrix but will rerun the ordering
    # if you need a fresh Consensus delete the consesnus file and it will rerun
    param["force_ConsensusClustering"] = 1

    # subsample type:
    # 'subjects' to subsample all frames from a subject
    # 'items' to subsample frames without taking into account within- or between-subject information
    param["Subsample_type"] = "items"
    param["Subsample_fraction"] = 0.8
    param["cons_n_folds"] = 20

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
    # consensus clustering and clustering must be run for this to computation to run
    param["computeClusterStability"] = 1

    return param
