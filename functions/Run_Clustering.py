from datetime import datetime
import pickle  # using pickle for all saves/loads here
import h5py
import os

from functions.n00_Utilities.WriteInformation import write_information
from functions.n00_Utilities.save4Dnii import save4dnii
from functions.n00_Utilities.check_iCAPs_files import check_icaps_files
from functions.n03_Clustering.AggregateSubjectFrames import aggregate_subject_frames
from functions.n03_Clustering.ConsensusClustering import consensus_clustering
from functions.n03_Clustering.MakeiCAPs import make_icaps
from functions.n03_Clustering.ReorderiCAPs import reorder_icaps
from functions.n03_Clustering.ZScore_iCAPs import zscore_icaps
from functions.n03_Clustering.saveSubjectMaps import save_subject_maps
from functions.n03_Clustering.saveRegionTables import save_region_tables
from functions.n03_Clustering.getClusterConsensus import get_cluster_consensus

"""
# This function runs clustering on the significant innovations found
# previously by thresholding; results will be the iCAPs maps
#
# Input:
#   param - dict containing all necessary parameters to run TA
#
#     * Data reading and saving information:
#       'PathData'       - path to data
#       'Subjects'       - list of subdirectories where each subject's
#                          fMRI data is stored; must contain one entry per
#                          subject to analyze.
#                          This is where the TA folder will be created
#                          (or looked for) for each subject.
#       'n_subjects'     - number of subjects to analyze
#       'title'          - possibility to define a title for the current
#                          project (useful if TA should be run for
#                          different parameters), default: current date;
#                          should be the title of the project for which
#                          TA has already been run
#       ['force_Aggregating']  - if set to 1, significant innovations and
#                          activity-inducing signal will be read and
#                          aggregated, even if this has been done before
#       ['force_Clustering']   - if set to 1, clustering will be forced to
#                          run, even if already has been done
#       ['thresh_title']  - information to read thresholding data and save
#                          clustering results; if not specified, the fields
#                          'alpha' and 'f_voxels' need to exist in param
#       ['data_title']    - information for saving of aggregated clustering
#                          data
#       ['iCAPs_title']   - string or list of strings with all the
#                          subfolders to create for saving clustering
#                          results
#
#     * Clustering information:
#       ['doClustering']  - specify if clustering should be done (set to 0
#                          to only run consensus clustering), default = 1
#       ['saveClusterReplicateData'] - specify if the result of each
#                          replicate should be saved during clustering,
#                          default = 0
#       'n_folds'         - number of replicates of clustering
#       'K'               - number of clusters; can be a single int or a
#                          list of multiple K values
#       'DistType'        - type of distance to use for k-means clustering
#                          ('sqeuclidean' or 'cosine')
#       ['MaxIter']       - maximum number of allowed iterations of k-means
#                          clustering; the default of 100 is sometimes not
#                          enough if many frames are included, default = 100
#
#     * Consensus clustering information:
#       ['doConsensusClustering']    - specify if consensus clustering
#                          should be done on top of the clustering above
#       ['force_ConsensusClustering'] - if set to 1, consensus clustering
#                          will be forced to run, even if already done
#       'Subsample_type'  - subsampling type for consensus clustering
#                          (default: 'items')
#                          'subjects' to subsample all frames from a
#                              subject
#                          'items' to subsample frames without taking into
#                              account within- or between-subject
#                              information
#       'Subsample_fraction' - fraction of subsampled data per fold
#       'cons_n_folds'    - number of folds for consensus clustering
#                          (clustering will be run with 'n_folds'
#                          replicates in each consensus clustering fold)
#       ['cons_title']    - subfolder in which to save consensus results
#
# Output:
#   Creates iCAPs maps and consensus clustering results
"""
def run_clustering(param):

    # Date and time when the routines are called
    param["date"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if not param.get("title"):
        param["title"] = param["date"]

    # Creates and opens a log-file that will contain all information related to
    # what is done to the data
    ta_logs_dir = os.path.join(param["PathData"], "TAlogs")
    os.makedirs(ta_logs_dir, exist_ok=True)
    log_path = os.path.join(ta_logs_dir, f"log_Clustering_{param['title']}.txt")
    fid = open(log_path, "a+", encoding="utf-8")

    write_information(
        fid,
        f"Starting the total activation/iCAPs tools for project entitled {param['title']}",
    )

    # Check data path
    if not os.path.isdir(param["PathData"]):
        write_information(fid, "Incorrect path towards the data: execution stopped")
        fid.close()
        raise FileNotFoundError(
            "The data folder that you specified does not exist! Please check and restart running..."
        )

    # Data title
    if not param.get("data_title"):
        param["data_title"] = param["title"]

    # Thresholding title
    if not param.get("thresh_title"):
        alpha = param["alpha"]
        f_vox = param["f_voxels"]
        param["thresh_title"] = (
            "Alpha_"
            + str(alpha[0]).replace(".", "DOT")
            + "_"
            + str(alpha[1]).replace(".", "DOT")
            + "_Fraction_"
            + str(f_vox).replace(".", "DOT")
        )

    # Main output dir for aggregated frames
    param["outDir_main"] = os.path.join(
        param["PathData"],
        "iCAPs_results",
        f"{param['data_title']}_{param['thresh_title']}",
    )
    if not os.path.isdir(param["outDir_main"]):
        os.makedirs(param["outDir_main"])

    # iCAPs titles
    if not param.get("iCAPs_title"):
        k_vals = param["K"] if isinstance(param["K"], (list, tuple)) else [param["K"]]
        icaps_titles = [
            f"K_{k}_Dist_{param['DistType']}_Folds_{param['n_folds']}" for k in k_vals
        ]
        param["iCAPs_title"] = icaps_titles[0] if len(icaps_titles) == 1 else icaps_titles

    # Check aggregation flag
    result = check_icaps_files(param["outDir_main"])
    aggregating_done = result[0]

    if param.get("force_Aggregating"):
        aggregating_done = 0

    do_consensus = bool(param.get("doConsensusClustering"))
    do_clustering = not (param.get("doClustering") is False)

    # ----------------------------------------------------------------------
    # Aggregation (if needed)
    # ----------------------------------------------------------------------
    if do_consensus or do_clustering:
        if not aggregating_done:
            write_information(fid, "Aggregating Subject Frames...")

            subj_paths = [
                os.path.join(param["PathData"], s) for s in param["Subjects"]
            ]
            results_paths = [
                os.path.join(sp, "TA_results", param["title"]) for sp in subj_paths
            ]

            (
                I_sig,
                AI,
                subject_labels,
                time_labels,
                AI_subject_labels,
                final_mask,
            ) = aggregate_subject_frames(results_paths, param, fid)

            write_information(fid, "Saving aggregated data...")

            # Save each object as a .pkl file
            with open(os.path.join(param["outDir_main"], "I_sig.pkl"), "wb") as f:
                pickle.dump(I_sig, f)

            with open(os.path.join(param["outDir_main"], "AI.pkl"), "wb") as f:
                pickle.dump(AI, f)

            with open(os.path.join(param["outDir_main"], "AI_subject_labels.pkl"), "wb") as f:
                pickle.dump(AI_subject_labels, f)

            with open(os.path.join(param["outDir_main"], "subject_labels.pkl"), "wb") as f:
                pickle.dump(subject_labels, f)

            with open(os.path.join(param["outDir_main"], "time_labels.pkl"), "wb") as f:
                pickle.dump(time_labels, f)

            with open(os.path.join(param["outDir_main"], "final_mask.pkl"), "wb") as f:
                pickle.dump(final_mask, f)

            # final_mask.nii, using first subject's mask_nonan as reference
            ref_mask = os.path.join(
                results_paths[0],
                "Thresholding",
                param["thresh_title"],
                "mask_nonan.nii",
            )
            save4dnii(
                param["outDir_main"],
                "",
                "final_mask",
                final_mask.reshape(-1,1),
                ref_mask,
            )

        else:

            # if aggregration already done and not forced load data
            write_information(
                fid, "Aggregating already done, loading aggregated data..."
            )

            # Load from .pkl instead of .mat
            with open(os.path.join(param["outDir_main"], "I_sig.pkl"), "rb") as f:
                I_sig = pickle.load(f)

            with open(os.path.join(param["outDir_main"], "final_mask.pkl"), "rb") as f:
                final_mask = pickle.load(f)

            with open(os.path.join(param["outDir_main"], "subject_labels.pkl"), "rb") as f:
                subject_labels = pickle.load(f)

            with open(os.path.join(param["outDir_main"], "time_labels.pkl"), "rb") as f:
                time_labels = pickle.load(f)

            with open(os.path.join(param["outDir_main"], "AI.pkl"), "rb") as f:
                AI = pickle.load(f)

            with open(os.path.join(param["outDir_main"], "AI_subject_labels.pkl"), "rb") as f:
                AI_subject_labels = pickle.load(f)


    # Adapt K + iCAPs titles for multiple K
    if isinstance(param["iCAPs_title"], (list, tuple)):
        param["iCAPs_title_cell"] = list(param["iCAPs_title"])
    else:
        param["iCAPs_title_cell"] = [param["iCAPs_title"]]

    param["K_vect"] = (
        param["K"] if isinstance(param["K"], (list, tuple)) else [param["K"]]
    )

    # ----------------------------------------------------------------------
    # MAIN CLUSTERING PROCEDURES NOW
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Consensus clustering
    # ----------------------------------------------------------------------
    if do_consensus:
        if not param.get("cons_title"):

            # setting name of folder
            k_vec = param["K_vect"]
            param["cons_title"] = (
                f"{k_vec[0]}to{k_vec[-1]}"
                f"_SubsampleType_{param['Subsample_type']}"
                f"_Fraction_{str(param['Subsample_fraction']).replace('.', 'DOT')}"
                f"_nFolds_{param['cons_n_folds']}"
                f"_Dist_{param['DistType']}"
            )

        param["outDir_cons"] = os.path.join(
            param["PathData"],
            "iCAPs_results",
            f"{param['data_title']}_{param['thresh_title']}",
            param["cons_title"],
        )

        if not os.path.isdir(param["outDir_cons"]):
            os.makedirs(param["outDir_cons"])

        # Check if consensus is forced
        if param.get("force_ConsensusClustering"):
            consensus_done = 0
        else:
            # Check if consensus has been done if not forced
            result = check_icaps_files(
                None, None, param["outDir_cons"]
            )
            consensus_done = result[2]

        if not consensus_done:
            # run consenus clustering
            consensus_clustering(I_sig, subject_labels, param, fid)

            # Save param as pickle
            with open(os.path.join(param["outDir_cons"], "param.pkl"), "wb") as f:
                pickle.dump(param, f)


    # ----------------------------------------------------------------------
    # K-means clustering for each K
    # ----------------------------------------------------------------------
    if do_clustering:
        write_information(fid, "Entering the clustering process...")

        # clustering for every K
        for i_k, title_k in enumerate(param["iCAPs_title_cell"]):
            param["iCAPs_title"] = title_k
            param["K"] = param["K_vect"][i_k]

            write_information(
                fid, f"K = {param['K']}, iCAPs title: {param['iCAPs_title']}"
            )

            param["outDir_iCAPs"] = os.path.join(
                param["outDir_main"], param["iCAPs_title"]
            )

            if not os.path.isdir(param["outDir_iCAPs"]):
                os.makedirs(param["outDir_iCAPs"])

            # Check if clustering is forced
            if param.get("force_Clustering"):
                clustering_done = 0
            else:
                # Check if clustering has been done if not forced
                result = check_icaps_files(
                    param["outDir_main"], param["outDir_iCAPs"]
                )
                clustering_done = result[1]

            if not clustering_done:
                # Run clustering
                write_information(fid, "Running Clustering...")
                icaps, idx, dist_to_centroid, icaps_folds = make_icaps(
                    I_sig, param, fid
                )

                # Reorder iCAPs by innovation counts
                write_information(fid, "Rearranging Time Courses...")

                # Dani: I replaced this by a function which just re-orders
                # the iCAPs (i.e. iCAPs numbering according to innovation
                # frame counts), but without changing the order in the Data
                idx, icaps, icaps_folds = reorder_icaps(idx, icaps, icaps_folds)

                # z-score iCAPs
                write_information(fid, "z-scoring iCAPs...")
                icaps_z = zscore_icaps(icaps, I_sig, idx)

                if isinstance(icaps_folds, dict) and icaps_folds["iCAPs"]:
                    for i_fold in range(len(icaps_folds["iCAPs"])):
                        icaps_folds["iCAPs_z"][i_fold] = zscore_icaps(
                            icaps_folds["iCAPs"][i_fold],
                            I_sig,
                            icaps_folds["IDX"][i_fold],
                        )

                # Save clustering outputs (pickle)
                write_information(fid, "Saving clustering data...")

                with open(os.path.join(param["outDir_iCAPs"], "iCAPs.pkl"), "wb") as f:
                    pickle.dump(icaps, f)

                with open(os.path.join(param["outDir_iCAPs"], "iCAPs_z.pkl"), "wb") as f:
                    pickle.dump(icaps_z, f)

                with open(os.path.join(param["outDir_iCAPs"], "IDX.pkl"), "wb") as f:
                    pickle.dump(idx, f)

                with open(os.path.join(param["outDir_iCAPs"], "dist_to_centroid.pkl"), "wb") as f:
                    pickle.dump(dist_to_centroid, f)

                with open(os.path.join(param["outDir_iCAPs"], "param.pkl"), "wb") as f:
                    pickle.dump(param, f)

                if icaps_folds is not None and icaps_folds:
                    with open(os.path.join(param["outDir_iCAPs"], "iCAPs_folds.pkl"), "wb") as f:
                        pickle.dump(icaps_folds, f)
                else:
                    folds_path = os.path.join(param["outDir_iCAPs"], "iCAPs_folds.pkl")
                    if os.path.isfile(folds_path):
                        os.remove(folds_path)

                # Save iCAP maps
                mask_nii = os.path.join(param["outDir_main"], "final_mask.nii")
                save4dnii(
                    param["outDir_iCAPs"], "", "iCAPs", icaps.T, mask_nii, final_mask
                )
                save4dnii(
                    param["outDir_iCAPs"], "", "iCAPs_z", icaps_z.T, mask_nii, final_mask
                )

            else:
                write_information(fid, "Clustering already done, skipping...")

            # saving subject maps
            subj_maps_dir = os.path.join(param["outDir_iCAPs"], "subjectMaps")
            subj_maps_file = os.path.join(
                subj_maps_dir, f"iCAP_z_{param['K']}.nii"
            )
            subjects_saved = os.path.isfile(subj_maps_file)

            if param.get("force_saveSubjectMaps"):
                subjects_saved = 0
                param["saveSubjectMaps"] = 1

            if param.get("saveSubjectMaps") and not subjects_saved:
                # Load IDX from pickle
                with open(os.path.join(param["outDir_iCAPs"], "IDX.pkl"), "rb") as f:
                    idx = pickle.load(f)

                write_information(fid, "Saving subject maps...")
                save_subject_maps(param, subject_labels, idx, I_sig, final_mask)

            # Saving region tables
            reg_table_path = os.path.join(param["outDir_iCAPs"], "iCAP_z_regions.txt")
            reg_table_exist = os.path.isfile(reg_table_path)

            if param.get("saveRegionTables") and not reg_table_exist:
                write_information(fid, "Saving region tables...")

                # Load iCAPs_z from pickle
                with open(os.path.join(param["outDir_iCAPs"], "iCAPs_z.pkl"), "rb") as f:
                    icaps_z = pickle.load(f)

                save_region_tables(param, icaps_z, final_mask)

    # ----------------------------------------------------------------------
    # Cluster stability based on consensus clustering
    # ----------------------------------------------------------------------
    # compute cluster stability based on consensus clustering (clustering
    # and consensus clustering have to be done already)
    if param.get("computeClusterStability"):

        write_information(fid, "Getting cluster consensus...")

        # do it for every K listed
        for i_k, title_k in enumerate(param["iCAPs_title_cell"]):
            param["iCAPs_title"] = title_k
            param["K"] = param["K_vect"][i_k]
            write_information(fid, f"K={param['K']}")

            param["outDir_iCAPs"] = os.path.join(
                param["outDir_main"], param["iCAPs_title"]
            )
            param["outDir_cons"] = os.path.join(
                param["PathData"],
                "iCAPs_results",
                f"{param['data_title']}_{param['thresh_title']}",
                param["cons_title"],
            )

            result = check_icaps_files(
                param["outDir_main"], param["outDir_iCAPs"], param["outDir_cons"]
            )
            clustering_done = result[1]
            consensus_done = result[2]

            if not consensus_done or not clustering_done:
                write_information(
                    fid,
                    "Run consensus clustering and clustering first, stability not computed!",
                )
                continue

            # Load IDX from pickle
            with open(os.path.join(param["outDir_iCAPs"], "IDX.pkl"), "rb") as f:
                idx = pickle.load(f)

            # Load Consensus_K from pickle
            consensus_path = os.path.join(param["outDir_cons"], f"Consensus_{param['K']}.pkl")
            with open(consensus_path, "rb") as f:
                cons = pickle.load(f)

            icaps_consensus, icaps_nitems = get_cluster_consensus(idx, cons)

            for i_c in range(param["K"]):
                write_information(
                    fid,
                    f"iCAP {i_c} ({icaps_nitems[i_c]} frames) average consensus is {icaps_consensus[i_c]}",
                )

            # Save consensus results via pickle as well
            with open(os.path.join(param["outDir_iCAPs"], "iCAPs_consensus.pkl"), "wb") as f:
                pickle.dump(icaps_consensus, f)

            with open(os.path.join(param["outDir_iCAPs"], "iCAPs_nItems.pkl"), "wb") as f:
                pickle.dump(icaps_nitems, f)

    fid.close()
