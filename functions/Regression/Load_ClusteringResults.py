import os
import numpy as np
from scipy.io import loadmat


def _load_array(base, name):
    """Helper: load array from either .npy or .mat (variable named like file)."""
    npy_path = os.path.join(base, name + '.npy')
    mat_path = os.path.join(base, name + '.mat')
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True)
    if os.path.exists(mat_path):
        mat = loadmat(mat_path, squeeze_me=True)
        if name in mat:
            return mat[name]
        # fall back to any array-like value
        for k, v in mat.items():
            if not k.startswith('__'):
                return v
    raise FileNotFoundError(f"Could not find {name}.npy or {name}.mat in {base}")


def Load_ClusteringResults(param):
    """Python translation of Load_ClusteringResults.m (best-effort).

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
    outDir_main = param['outDir_main']
    outDir_iCAPs = param['outDir_iCAPs']

    AI = _load_array(outDir_main, 'AI')
    AI_subject_labels = _load_array(outDir_main, 'AI_subject_labels')
    time_labels = _load_array(outDir_main, 'time_labels')
    subject_labels = _load_array(outDir_main, 'subject_labels')

    iCAPs = _load_array(outDir_iCAPs, 'iCAPs')
    dist_to_centroid = _load_array(outDir_iCAPs, 'dist_to_centroid')
    IDX = _load_array(outDir_iCAPs, 'IDX')

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
