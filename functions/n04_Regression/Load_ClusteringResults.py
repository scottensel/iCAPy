import os
import pickle


def load_clustering_results(param):
    """
    Python equivalent of Load_ClusteringResults.m
    (pickle-only version)

    Loads clustering-related results (iCAPs, AI, labels, distances) from disk.

    Parameters
    ----------
    param : dict-like
        Must contain:
        - 'outDir_main'
        - 'outDir_iCAPs'

    Returns
    -------
    out : dict
        Dictionary with keys:
        - 'iCAPs'
        - 'AI'
        - 'AI_subject_labels'
        - 'IDX'
        - 'dist_to_centroid'
        - 'subject_labels'
        - 'time_labels'
    """

    outDir_main = param["outDir_main"]
    outDir_iCAPs = param["outDir_iCAPs"]

    def load_pkl(base, name):
        path = os.path.join(base, f"{name}.pkl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required file: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    AI = load_pkl(outDir_main, "AI")
    AI_subject_labels = load_pkl(outDir_main, "AI_subject_labels")
    time_labels = load_pkl(outDir_main, "time_labels")
    subject_labels = load_pkl(outDir_main, "subject_labels")

    iCAPs = load_pkl(outDir_iCAPs, "iCAPs")
    dist_to_centroid = load_pkl(outDir_iCAPs, "dist_to_centroid")
    IDX = load_pkl(outDir_iCAPs, "IDX")

    out = dict(
        iCAPs=iCAPs,
        AI=AI,
        AI_subject_labels=AI_subject_labels,
        IDX=IDX,
        dist_to_centroid=dist_to_centroid,
        subject_labels=subject_labels,
        time_labels=time_labels,
    )

    return out
