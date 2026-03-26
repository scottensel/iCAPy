import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list

from functions.n00_Utilities.WriteInformation import write_information
from functions.n03_Clustering.Build_Connectivity_Matrix import Build_Connectivity_Matrix
from functions.n03_Clustering.ComputeClusteringQuality import ComputeClusteringQuality
from functions.n03_Clustering.MakeiCAPs import _run_one_fold


def consensus_clustering(X, subject_labels, param, fid=None):
    """
    Faithful Python port of MATLAB ConsensusClustering.m
    """
    K_range            = np.asarray(param["K_vect"], dtype=int)
    n_folds            = int(param["cons_n_folds"])
    n_rep              = int(param["n_folds"])
    DistType           = param["DistType"]
    Subsample_fraction = float(param["Subsample_fraction"])
    Subsample_type     = param["Subsample_type"]
    MaxIter            = int(param.get("MaxIter", 100))
    outDir_cons        = param["outDir_cons"]
    force              = bool(param.get("force_ConsensusClustering", False))
    limit_threads      = param.get("limitThreads", None)

    os.makedirs(outDir_cons, exist_ok=True)

    X = np.asarray(X, dtype=np.float64)
    n_items, n_dims = X.shape
    print(f"n_items {n_items}...")
    print(f"n_dims {n_dims}...")

    CDF = np.zeros((len(K_range), 101))
    AUC = np.zeros(len(K_range))

    # ------------------------------------------------------------------ #
    # Loop over K values                                                   #
    # ------------------------------------------------------------------ #
    for k_idx, K in enumerate(K_range):
        print(f"Running consensus clustering for K = {K}...")
        if fid is not None:
            write_information(fid, f"Running consensus clustering for K = {K}...")

        # MATLAB skips the entire K if consensusResults_K.mat exists and
        # force=false.  We use the ordered pkl as the equivalent sentinel.
        ordered_path = os.path.join(outDir_cons, f"Consensus_ordered_{K}.pkl")
        if os.path.isfile(ordered_path) and not force:
            print("consensus clustering already done, skipping ...")
            if fid is not None:
                write_information(fid, "consensus clustering already done, skipping ...")
            # Still need CDF/AUC - load ordered and recompute
            with open(ordered_path, "rb") as f:
                Consensus_ordered = pickle.load(f)
            CDF[k_idx, :], AUC[k_idx] = ComputeClusteringQuality(Consensus_ordered, K)
            del Consensus_ordered
            continue

        # -------------------------------------------------------------- #
        # Build or load unordered Consensus matrix                        #
        # -------------------------------------------------------------- #
        consensus_path = os.path.join(outDir_cons, f"Consensus_{K}.pkl")

        if os.path.isfile(consensus_path):
            print(f"Loading existing Consensus_{K} ...")
            with open(consensus_path, "rb") as f:
                Consensus = pickle.load(f)

        else:
            M_sum = np.zeros((n_items, n_items))
            I_sum = np.zeros((n_items, n_items))

            for h in range(n_folds):
                print(f"Fold {h + 1}:")

                # -------------------------------------------------------- #
                # Subsampling                                               #
                # -------------------------------------------------------- #
                if Subsample_type == "items":
                    n_items_ss = int(np.floor(Subsample_fraction * n_items))
                    idx_ss     = np.random.choice(n_items, n_items_ss, replace=False)
                    I_vec      = np.zeros(n_items, dtype=int)
                    I_vec[idx_ss] = 1

                elif Subsample_type == "subjects":
                    if subject_labels is None or len(subject_labels) == 0:
                        raise ValueError("subject_labels required for subjects subsampling")
                    subjects       = np.unique(subject_labels)
                    n_subjects_ss  = int(np.floor(Subsample_fraction * len(subjects)))
                    subjects_ss    = np.random.choice(subjects, n_subjects_ss, replace=False)
                    I_vec          = np.zeros(n_items, dtype=int)
                    for s in subjects_ss:
                        I_vec[subject_labels == s] = 1
                else:
                    raise ValueError(f"Unsupported Subsample_type: {Subsample_type}")

                # Indicator matrix  (MATLAB: I_sum += I_vec * I_vec')
                I_sum += np.outer(I_vec, I_vec)
                X_ss   = X[I_vec > 0, :]

                # -------------------------------------------------------- #
                # Clustering                                                #
                # MATLAB: kmeans(X_ss, K, 'Distance', DistType, ...)       #
                # For cosine we use cosine_kmeans (validated against MATLAB)#
                # For sqeuclidean we use _run_one_fold which uses sklearn   #
                # -------------------------------------------------------- #
                IDX, _, _, _ = _run_one_fold(
                    X_ss, K, DistType,
                    n_init=n_rep, max_iter=MaxIter,
                    limit_threads=limit_threads,
                )

                # -------------------------------------------------------- #
                # Connectivity matrix                                       #
                # MATLAB: Build_Connectivity_Matrix(IDX, find(I_vec>0), ..)#
                # find(I_vec>0) returns 1-based indices in MATLAB;         #
                # we pass 0-based indices and handle that in the function   #
                # -------------------------------------------------------- #
                print("Building connectivity matrix M ...")
                M_sum += Build_Connectivity_Matrix(
                    IDX,
                    np.where(I_vec > 0)[0],
                    Subsample_type,
                    n_items,
                )

                del X_ss, IDX, I_vec

            # ------------------------------------------------------------ #
            # Consensus matrix  (MATLAB: Consensus = M_sum ./ I_sum)       #
            # ------------------------------------------------------------ #
            Consensus       = np.zeros_like(M_sum)
            valid           = I_sum > 0
            Consensus[valid] = M_sum[valid] / I_sum[valid]

            n_invalid = int(np.sum(~valid))
            if n_invalid > 0:
                msg = (f"{n_invalid} ({100*n_invalid/valid.size:.2f}%) pairs never "
                       f"co-sampled — increase n_folds!")
                print(f"Warning: {msg}")
                if fid is not None:
                    write_information(fid, f"Warning: {msg}")
                Consensus[~valid] = 0.0

            del M_sum, I_sum, valid

            print("Saving consensus results (not ordered) ...")
            with open(consensus_path, "wb") as f:
                pickle.dump(Consensus, f)

        # -------------------------------------------------------------- #
        # Ordering  (MATLAB: linkage + optimalleaforder)                  #
        # -------------------------------------------------------------- #
        print("Ordering consensus matrix ...")
        D      = 1.0 - Consensus
        np.fill_diagonal(D, 0.0)
        D_cond = squareform(D, checks=False)
        Z      = linkage(D_cond, method="average")
        Z_opt  = optimal_leaf_ordering(Z, D_cond)
        order  = leaves_list(Z_opt)

        Consensus_ordered = Consensus[np.ix_(order, order)]
        del Consensus, D, D_cond, Z, Z_opt, order

        # -------------------------------------------------------------- #
        # CDF / AUC                                                        #
        # -------------------------------------------------------------- #
        CDF[k_idx, :], AUC[k_idx] = ComputeClusteringQuality(Consensus_ordered, K)

        # -------------------------------------------------------------- #
        # Save ordered consensus + PNG                                     #
        # -------------------------------------------------------------- #
        print("Saving consensus results ...")
        with open(ordered_path, "wb") as f:
            pickle.dump(Consensus_ordered, f)

        fig, ax = plt.subplots()
        im = ax.imshow(Consensus_ordered, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"k= {K}")
        fig.savefig(
            os.path.join(outDir_cons, f"Consensus_ordered_{K}.png"),
            dpi=200, bbox_inches="tight",
        )
        plt.close(fig)

        del Consensus_ordered

    # ------------------------------------------------------------------ #
    # Save CDF / AUC + plots                                              #
    # ------------------------------------------------------------------ #
    with open(os.path.join(outDir_cons, "CDF.pkl"), "wb") as f:
        pickle.dump(CDF, f)
    with open(os.path.join(outDir_cons, "AUC.pkl"), "wb") as f:
        pickle.dump(AUC, f)

    x = np.linspace(0, 1, 101)
    fig, ax = plt.subplots()
    for k_idx, K in enumerate(K_range):
        ax.plot(x, CDF[k_idx, :], linewidth=2, label=str(K))
    ax.legend(loc="lower right")
    ax.set_title("CDF")
    fig.savefig(os.path.join(outDir_cons, "CDF.svg"), format="svg", bbox_inches="tight")
    fig.savefig(os.path.join(outDir_cons, "CDF.png"), format="png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(K_range, AUC, "-o", linewidth=2)
    ax.set_xlabel("K")
    ax.set_title("AUC")
    fig.savefig(os.path.join(outDir_cons, "AUC.svg"), format="svg", bbox_inches="tight")
    fig.savefig(os.path.join(outDir_cons, "AUC.png"), format="png", bbox_inches="tight")
    plt.close(fig)

    return CDF, AUC