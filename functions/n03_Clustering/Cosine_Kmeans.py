import numpy as np
from scipy.spatial.distance import cdist


def cosine_kmeans(X, n_clusters, n_init=10, max_iter=100, tol=1e-6, random_state=None):
    """
    K-means clustering with cosine distance, matching MATLAB's:
        kmeans(X, K, 'Distance', 'cosine', 'Replicates', n_init)

    MATLAB normalises X to unit L2 norm once before the algorithm runs,
    then computes centroids as the arithmetic mean of those normalised rows.
    Centroids are renormalised only transiently inside the distance function.
    The returned centroids are means of unit-norm rows (not renormalised),
    identical to MATLAB's returned C matrix.

    Parameters
    ----------
    X            : (n_samples, n_features)  original data
    n_clusters   : int
    n_init       : int    number of replicates  (MATLAB 'Replicates')
    max_iter     : int    per replicate         (MATLAB 'MaxIter', default 100)
    tol          : float  convergence tolerance on centroid shift
    random_state : int or None

    Returns
    -------
    labels  : (n_samples,) int64   0-based cluster assignments
    centers : (n_clusters, n_features)  centroids (means of unit-norm rows)
    inertia : float  total cosine distance sum for the best replicate
    """
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    # Normalise X to unit L2 norm once, matching MATLAB's pre-processing
    X_norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(X_norms <= np.finfo(float).eps * X_norms.max()):
        raise ValueError(
            "Some rows of X have near-zero norm. "
            "MATLAB raises ZeroDataForCos in this case."
        )
    X_normed = X / X_norms

    best_inertia = np.inf
    best_centers = None
    best_labels  = None

    for _ in range(n_init):
        centers_normed = _kmeanspp_init(X_normed, n_clusters, rng)
        labels = np.empty(n_samples, dtype=np.int64)

        for _iter in range(max_iter):
            # Renormalise centroids before computing distances (MATLAB distfun)
            c_norms  = np.linalg.norm(centers_normed, axis=1, keepdims=True)
            C_normed = centers_normed / np.maximum(c_norms, np.finfo(float).eps)
            D        = cdist(X_normed, C_normed, metric='cosine')  # (n_samples, K)

            new_labels = np.argmin(D, axis=1).astype(np.int64)

            # Update centroids as mean of normalised member rows (MATLAB gcentroids)
            new_centers = np.zeros((n_clusters, n_features), dtype=np.float64)
            for k in range(n_clusters):
                mask = new_labels == k
                if np.any(mask):
                    new_centers[k] = X_normed[mask].mean(axis=0)
                else:
                    # Empty cluster: steal the point furthest from its centroid
                    d_assigned   = D[np.arange(n_samples), new_labels]
                    new_centers[k] = X_normed[np.argmax(d_assigned)]

            shift          = np.linalg.norm(new_centers - centers_normed)
            centers_normed = new_centers
            labels         = new_labels

            if shift < tol:
                break

        # Compute inertia for this replicate
        c_norms       = np.linalg.norm(centers_normed, axis=1, keepdims=True)
        C_normed_final = centers_normed / np.maximum(c_norms, np.finfo(float).eps)
        D_final        = cdist(X_normed, C_normed_final, metric='cosine')
        inertia        = float(np.sum(D_final[np.arange(n_samples), labels]))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()
            best_centers = centers_normed.copy()

    return best_labels, best_centers, best_inertia


def _kmeanspp_init(X_normed, n_clusters, rng):
    """
    K-means++ initialisation on unit-normalised data using cosine distance,
    matching MATLAB's default Start='plus'.
    """
    n_samples, n_features = X_normed.shape
    centers   = np.empty((n_clusters, n_features), dtype=np.float64)
    centers[0] = X_normed[rng.integers(0, n_samples)]
    min_dist  = np.full(n_samples, np.inf)

    for i in range(1, n_clusters):
        c      = centers[i - 1]
        c_norm = np.linalg.norm(c)
        if c_norm > np.finfo(float).eps:
            c = c / c_norm
        d        = np.maximum(1.0 - X_normed @ c, 0.0)
        min_dist = np.minimum(min_dist, d)
        denom    = min_dist.sum()

        if denom == 0 or not np.isfinite(denom):
            centers[i:] = X_normed[rng.integers(0, n_samples, size=n_clusters - i)]
            break

        centers[i] = X_normed[rng.choice(n_samples, p=min_dist / denom)]

    return centers
