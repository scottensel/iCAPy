import numpy as np
from .getIDXmat import getIDXmat
from functions.Utilities.WriteInformation import write_information


def evaluateSoftClusterThres_corrs(TC, clusteringResults, dist_to_centroid,
                                   thresVals, param, fid=None):
    """Best-effort Python approximation of evaluateSoftClusterThres_corrs.m.

    The original MATLAB version evaluates soft-clustering thresholds based on
    correlations between measured and estimated transient amplitudes.

    This Python version preserves the high-level structure:
    - For each threshold value xi in thresVals:
        * compute soft cluster assignments via getIDXmat
        * compute basic summary statistics of how many clusters each
          innovation belongs to.

    It does NOT reproduce the full correlation plots from the truncated
    MATLAB file, but provides a scaffold where such computations can be
    plugged in.

    Parameters
    ----------
    TC : list-like
        Time courses per subject (same format as for computeTemporalCharacteristics).
    clusteringResults : dict
        Clustering results including at least AI_subject_labels, subject_labels, IDX.
    dist_to_centroid : ndarray, shape (n_items, nClus)
        Distances of frames to centroids.
    thresVals : array-like
        List of soft assignment thresholds.
    param : dict-like
        Parameter structure; must contain 'n_subjects'.
    fid : file handle or str or None
        For logging.

    Returns
    -------
    stats : dict
        Contains minimal summary fields describing soft assignments.
    """
    thresVals = np.asarray(thresVals, dtype=float)
    nThres = thresVals.size
    n_items, nClus = dist_to_centroid.shape
    nSub = int(param['n_subjects'])

    IDX_mat_all = np.zeros((n_items, nClus, nThres), dtype=int)
    nClus_mat_all = np.zeros((n_items, nThres), dtype=int)

    for iT, xi in enumerate(thresVals):
        IDX_mat, dist_thres, clos_to_centroid = getIDXmat(dist_to_centroid, xi)
        IDX_mat_all[:, :, iT] = IDX_mat
        nClus_mat_all[:, iT] = IDX_mat.sum(axis=1)

    if fid is not None:
        write_information(fid, "evaluateSoftClusterThres_corrs: computed soft assignments for all thresholds.")

    stats = dict(
        IDX_mat_all=IDX_mat_all,
        nClus_mat_all=nClus_mat_all,
        thresVals=thresVals,
    )
    return stats
