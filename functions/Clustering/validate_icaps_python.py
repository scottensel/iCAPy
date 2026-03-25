"""
validate_icaps_python.py

WHERE TO RUN THIS:
    Same directory as matlab_ground_truth.mat (which is inside param.outDir_iCAPs,
    e.g. F:/iCAP/Data/.../iCAPs_results/.../K_5_Dist_cosine_Folds_10/)

    Also needs Cosine_Kmeans.py accessible. Either copy it here or adjust
    the sys.path lines below to point to wherever Cosine_Kmeans.py lives
    in your project.

    Run from terminal:
        python validate_icaps_python.py

    Or set mat_path and cosine_kmeans_dir at the top of this file
    if you want to run it from a different working directory.
"""

import sys
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist

# ── Point this to wherever your Cosine_Kmeans.py lives ────────────────────
COSINE_KMEANS_DIR = "."   # "." means same folder as this script
sys.path.insert(0, COSINE_KMEANS_DIR)
from Cosine_Kmeans import cosine_kmeans_fixed_seed

# ── Point this to your matlab_ground_truth.mat ─────────────────────────────
MAT_PATH = "matlab_ground_truth.mat"


# ==========================================================================
# 1. Load MATLAB ground truth
# ==========================================================================
print("Loading", MAT_PATH, "...")
mat = loadmat(MAT_PATH)

I_sig        = mat["I_sig"].astype(np.float64)           # (n_frames, n_voxels)
matlab_IDX   = mat["IDX"].squeeze().astype(np.int64) - 1 # 1-based -> 0-based
matlab_iCAPs = mat["iCAPs"].astype(np.float64)           # (K, n_voxels)
matlab_sumD  = mat["sumD"].squeeze().astype(np.float64)  # (K,)
matlab_D     = mat["dist_to_centroid"].astype(np.float64)# (n_frames, K)
init_centers = mat["init_centers"].astype(np.float64)    # (K, n_voxels)
seed_rows    = mat["seed_rows"].squeeze().astype(np.int64) - 1  # 0-based
K            = int(mat["K"].squeeze())
DistType     = str(mat["DistType"][0])

n_frames, n_voxels = I_sig.shape
print(f"  I_sig:    {n_frames} frames x {n_voxels} voxels")
print(f"  K={K}, DistType={DistType}")
print(f"  seed_rows (0-based, first 5): {seed_rows[:5].tolist()}\n")


# ==========================================================================
# 2. Run Python from the exact same starting centroids MATLAB used
# ==========================================================================
print("Running cosine_kmeans_fixed_seed ...")
py_labels, py_centers, py_inertia = cosine_kmeans_fixed_seed(
    I_sig,
    n_clusters=K,
    seed_points_0based=seed_rows,
    max_iter=100,
    tol=1e-6,
)

# Rebuild full distance matrix the same way MakeiCAPs.py does
X_norms  = np.linalg.norm(I_sig, axis=1, keepdims=True)
X_norms  = np.maximum(X_norms, np.finfo(float).eps)
X_normed = I_sig / X_norms

c_norms  = np.linalg.norm(py_centers, axis=1, keepdims=True)
c_norms  = np.maximum(c_norms, np.finfo(float).eps)
C_normed = py_centers / c_norms

py_D = cdist(X_normed, C_normed, metric="cosine")   # (n_frames, K)

py_sumD = np.array([
    py_D[py_labels == k, k].sum() for k in range(K)
], dtype=np.float64)


# ==========================================================================
# 3. Report
# ==========================================================================
FLOAT_TOL = 1e-8

def report(name, diff_max, diff_rel=None):
    status = "PASS" if diff_max < FLOAT_TOL else "FAIL"
    line = f"  [{status}]  {name:40s}  max_abs = {diff_max:.3e}"
    if diff_rel is not None:
        line += f"  max_rel = {diff_rel:.3e}"
    print(line)

print("=" * 72)
print("VALIDATION REPORT")
print("=" * 72)

# Centroids
ctr_abs = np.max(np.abs(py_centers - matlab_iCAPs))
ctr_rel = np.max(np.abs(py_centers - matlab_iCAPs) / (np.abs(matlab_iCAPs) + 1e-12))
report("Centroids (iCAPs)", ctr_abs, ctr_rel)

# Labels
n_bad = int(np.sum(py_labels != matlab_IDX))
status = "PASS" if n_bad == 0 else "FAIL"
print(f"  [{status}]  {'Labels (IDX)':40s}  "
      f"disagreeing = {n_bad}/{n_frames} ({100*n_bad/n_frames:.1f}%)")

# Full distance matrix
D_abs = np.max(np.abs(py_D - matlab_D))
D_rel = np.max(np.abs(py_D - matlab_D) / (np.abs(matlab_D) + 1e-12))
report("dist_to_centroid  (n_frames x K)", D_abs, D_rel)

# Per-cluster sumD
sd_abs = np.max(np.abs(py_sumD - matlab_sumD))
report("Per-cluster sumD", sd_abs)

# Total inertia
tot_abs = abs(py_inertia - float(matlab_sumD.sum()))
report("Total inertia", tot_abs)

print("=" * 72)


# ==========================================================================
# 4. Diagnostics when something fails
# ==========================================================================
any_fail = (ctr_abs >= FLOAT_TOL or n_bad > 0 or
            D_abs >= FLOAT_TOL or sd_abs >= FLOAT_TOL)

if not any_fail:
    print("\nAll checks passed — Python cosine k-means matches MATLAB exactly.")
else:
    print("\nDIAGNOSTICS")

    if ctr_abs >= FLOAT_TOL:
        print("\n  Centroid mismatch:")
        worst_k = int(np.argmax(
            np.max(np.abs(py_centers - matlab_iCAPs), axis=1)
        ))
        print(f"    Worst cluster: {worst_k}")
        print(f"    Python  ||center||: {np.linalg.norm(py_centers[worst_k]):.8f}")
        print(f"    MATLAB  ||center||: {np.linalg.norm(matlab_iCAPs[worst_k]):.8f}")
        print("    If norms differ >> pre-normalisation of X is wrong in Python.")
        print("    If norms match but values differ >> centroid update formula differs.")

    if n_bad > 0:
        bad_idx = np.where(py_labels != matlab_IDX)[0]
        print(f"\n  Label mismatches (first 10 of {n_bad}):")
        print(f"    Frame indices : {bad_idx[:10].tolist()}")
        print(f"    Python labels : {py_labels[bad_idx[:10]].tolist()}")
        print(f"    MATLAB labels : {matlab_IDX[bad_idx[:10]].tolist()}")
        print("    Label-only mismatches with matching centroids = tiebreaking diff.")
        print("    Label + centroid mismatches = update formula diff.")

    if D_abs >= FLOAT_TOL:
        wf, wk = np.unravel_index(np.argmax(np.abs(py_D - matlab_D)), py_D.shape)
        print(f"\n  Distance matrix mismatch:")
        print(f"    Worst cell: frame {wf}, cluster {wk}")
        print(f"    Python: {py_D[wf, wk]:.12f}")
        print(f"    MATLAB: {matlab_D[wf, wk]:.12f}")
        print("    If centroids match but D differs >> centroid renorm before cdist is wrong.")
