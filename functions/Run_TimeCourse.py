import os
import copy
from datetime import datetime
import pickle

import numpy as np

from functions.n00_Utilities.WriteInformation import write_information
from functions.n00_Utilities.check_iCAPs_files import check_icaps_files
from functions.n04_Regression.GenerateTimeCourses import generate_time_courses
from functions.n04_Regression.GenerateTimeCoursesWeighted import generate_time_courses_weighted
from functions.n04_Regression.Load_ClusteringResults import load_clustering_results
from functions.n04_Regression.computeTemporalCharacteristics import compute_temporal_characteristics
from functions.n04_Regression.evaluateSoftClusterThres import evaluate_soft_cluster_thres
from functions.n04_Regression.evaluateSoftClusterThres_corrs import evaluate_soft_cluster_thres_corrs

def _fmt_xi(x: float) -> str:
    """
    Format softClusterThres values for folder names:
    - if it's an integer (e.g., 1.0) -> "1"
    - otherwise trim floating noise (e.g., 0.199999999999 -> "0.1" or "0.2" depending on rounding)
    """
    # Round to a reasonable precision to kill binary float artifacts
    x = float(np.round(x, 6))

    # If effectively integer, print as int
    if np.isclose(x, int(round(x))):
        return str(int(round(x)))

    # Otherwise format compactly (no scientific notation, no trailing zeros)
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s


def run_regression(param):
    """
    Python equivalent of Run_Regression.m, using pickle (.pkl) for all saving/loading.
    """

    # Date and time
    param["date"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if not param.get("title"):
        param["title"] = param["date"]

    # Log file
    ta_logs_dir = os.path.join(param["PathData"], "TAlogs")
    os.makedirs(ta_logs_dir, exist_ok=True)
    log_path = os.path.join(ta_logs_dir, f"log_Regression_{param['title']}.txt")
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
            "The data folder that you specified does not exist!"
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

    # Main output directory
    param["outDir_main"] = os.path.join(
        param["PathData"],
        "iCAPs_results",
        f"{param['data_title']}_{param['thresh_title']}",
    )

    # iCAPs titles
    if not param.get("iCAPs_title"):
        k_vals = param["K"] if isinstance(param["K"], (list, tuple)) else [param["K"]]
        param["iCAPs_title"] = [
            f"K_{k}_Dist_{param['DistType']}_Folds_{param['n_folds']}"
            for k in k_vals
        ]
        if len(param["iCAPs_title"]) == 1:
            param["iCAPs_title"] = param["iCAPs_title"][0]

    if isinstance(param["iCAPs_title"], (list, tuple)):
        param["iCAPs_title_cell"] = list(param["iCAPs_title"])
    else:
        param["iCAPs_title_cell"] = [param["iCAPs_title"]]

    param["K_vect"] = (
        param["K"] if isinstance(param["K"], (list, tuple)) else [param["K"]]
    )

    # ------------------------------------------------------------------
    # Main regression loop
    # ------------------------------------------------------------------
    if not (param.get("doRegression") is False):
        write_information(fid, "Entering the time course retrieval process...")

        for i_k, title_k in enumerate(param["iCAPs_title_cell"]):
            param["iCAPs_title"] = title_k
            param["K"] = param["K_vect"][i_k]

            write_information(fid, f"K = {param['K']}, iCAPs title: {title_k}")

            param["outDir_iCAPs"] = os.path.join(
                param["outDir_main"], title_k
            )

            result = check_icaps_files(param["outDir_main"], param["outDir_iCAPs"])
            aggregating_done = result[0]
            clustering_done = result[1]

            if not aggregating_done:
                fid.close()
                raise RuntimeError("Run clustering first (aggregation missing).")
            if not clustering_done:
                fid.close()
                raise RuntimeError("Run clustering first (clustering missing).")

            reg_type = param.get("regType", "transient-informed")

            # ==============================================================
            # Unconstrained regression
            # ==============================================================
            if reg_type == "unconstrained":
                param["outDir_reg"] = os.path.join(
                    param["outDir_iCAPs"], "TCs_unconstrained"
                )
                os.makedirs(param["outDir_reg"], exist_ok=True)

                result = check_icaps_files(None, None, None, param["outDir_reg"])
                regression_done = result[3]

                if param.get("force_Regression"):
                    regression_done = 0

                if not regression_done:
                    write_information(fid, "Loading aggregated data and iCAPs...")

                    with open(os.path.join(param["outDir_main"], "AI.pkl"), "rb") as f:
                        AI = pickle.load(f)
                    with open(os.path.join(param["outDir_main"], "AI_subject_labels.pkl"), "rb") as f:
                        AI_subject_labels = pickle.load(f)
                    with open(os.path.join(param["outDir_iCAPs"], "iCAPs.pkl"), "rb") as f:
                        icaps = pickle.load(f)

                    write_information(fid, "Generating iCAPs Time Courses...")
                    TC_unc, TC_unc_stats = generate_time_courses(
                        AI.T, AI_subject_labels, icaps, param
                    )

                    with open(os.path.join(param["outDir_reg"], "TC_unc.pkl"), "wb") as f:
                        pickle.dump(TC_unc, f)
                    with open(os.path.join(param["outDir_reg"], "TC_unc_stats.pkl"), "wb") as f:
                        pickle.dump(TC_unc_stats, f)
                    with open(os.path.join(param["outDir_reg"], "param.pkl"), "wb") as f:
                        pickle.dump(param, f)

                    with open(os.path.join(param["outDir_main"], "subject_labels.pkl"), "rb") as f:
                        subject_labels = pickle.load(f)
                    with open(os.path.join(param["outDir_iCAPs"], "IDX.pkl"), "rb") as f:
                        idx = pickle.load(f)

                    clustering_results = {
                        "AI_subject_labels": AI_subject_labels,
                        "subject_labels": subject_labels,
                        "IDX": idx,
                    }

                    tempChar_unc = compute_temporal_characteristics(
                        TC_unc, clustering_results, param
                    )

                    with open(os.path.join(param["outDir_reg"], "tempChar_unc.pkl"), "wb") as f:
                        pickle.dump(tempChar_unc, f)

                else:
                    write_information(fid, "Unconstrained regression already done, skipping...")

            # ==============================================================
            # Transient-informed regression
            # ==============================================================
            elif reg_type == "transient-informed":
                diffs = np.diff(param["softClusterThres"])
                mean_diff = float(np.mean(diffs))
                # label = (
                #     f"TCs_{param['softClusterThres'][0]}_"
                #     f"{mean_diff}_"
                #     f"{param['softClusterThres'][-1]}"
                # ).replace(".", "DOT")

                diffs = np.diff(param["softClusterThres"])
                mean_diff = float(np.mean(diffs))

                start = _fmt_xi(param["softClusterThres"][0])
                step = _fmt_xi(mean_diff)
                end = _fmt_xi(param["softClusterThres"][-1])

                label = f"TCs_{start}_{step}_{end}".replace(".", "DOT")

                # start = str(param["softClusterThres"][0])
                # step = str(np.mean(np.diff(param["softClusterThres"])))
                # end = str(param["softClusterThres"][-1])
                #
                # label = f"TCs_{start}_{step}_{end}".replace(".", "DOT")

                param["outDir_reg"] = os.path.join(param["outDir_iCAPs"], label)
                os.makedirs(param["outDir_reg"], exist_ok=True)

                result = check_icaps_files(None, None, None, param["outDir_reg"])
                regression_done = result[3]

                if param.get("force_Regression"):
                    regression_done = 0

                if not regression_done:
                    write_information(fid, "Loading aggregated data and iCAPs...")
                    clustering_results = load_clustering_results(param)

                    TC_list = []
                    TC_stats_list = []
                    tempChar_list = []

                    for soft_value in param["softClusterThres"]:
                        write_information(fid, f"Soft assignment factor: {soft_value}")

                        # param_tmp = dict(param)
                        param_tmp = copy.deepcopy(param)
                        param_tmp["softClusterThres"] = soft_value

                        TC_i, TC_stats_i = generate_time_courses_weighted(
                            clustering_results, param_tmp, fid
                        )
                        tempChar_i = compute_temporal_characteristics(
                            TC_i, clustering_results, param_tmp
                        )

                        TC_list.append(TC_i)
                        TC_stats_list.append(TC_stats_i)
                        tempChar_list.append(tempChar_i)

                    with open(os.path.join(param["outDir_reg"], "TC.pkl"), "wb") as f:
                        pickle.dump(TC_list, f)
                    with open(os.path.join(param["outDir_reg"], "TC_stats.pkl"), "wb") as f:
                        pickle.dump(TC_stats_list, f)
                    with open(os.path.join(param["outDir_reg"], "tempChar.pkl"), "wb") as f:
                        pickle.dump(tempChar_list, f)
                    with open(os.path.join(param["outDir_reg"], "param.pkl"), "wb") as f:
                        pickle.dump(param, f)

                    if param.get("evalAmplitudeCorrs"):
                        evaluate_soft_cluster_thres_corrs(
                            clustering_results, TC_list, param, fid
                        )

                    best_id = evaluate_soft_cluster_thres(
                        TC_stats_list, param, fid
                    )

                    best_val = str(param["softClusterThres"][best_id]).replace(".", "DOT")

                    with open(os.path.join(param["outDir_iCAPs"], f"TC_{best_val}.pkl"), "wb") as f:
                        pickle.dump(TC_list[best_id], f)
                    with open(os.path.join(param["outDir_iCAPs"], f"TC_stats_{best_val}.pkl"), "wb") as f:
                        pickle.dump(TC_stats_list[best_id], f)
                    with open(os.path.join(param["outDir_iCAPs"], f"tempChar_{best_val}.pkl"), "wb") as f:
                        pickle.dump(tempChar_list[best_id], f)

                else:
                    write_information(fid, "Transient-informed regression already done, skipping...")

            else:
                fid.close()
                raise ValueError(f"Unknown regType '{reg_type}'")

    fid.close()
