import numpy as np

def cosine_kmeans(X, n_clusters, n_init=10, max_iter=300, tol=1e-4, random_state=None):
    """
    K-means clustering with cosine distance, matching MATLAB's kmeans(..., 'Distance', 'cosine').

    This implementation:
    1. Does NOT normalize input data
    2. Assigns points by maximum cosine similarity (= minimum cosine distance)
    3. Updates centroids as arithmetic mean in original space
    4. Does NOT normalize centroids after update

    This exactly matches MATLAB's behavior.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data in ORIGINAL space (not normalized)
    n_clusters : int
        Number of clusters
    n_init : int
        Number of random initializations
    max_iter : int
        Maximum iterations per initialization
    tol : float
        Convergence tolerance
    random_state : int or None
        Random seed for reproducibility

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels (0-based)
    centers : ndarray, shape (n_clusters, n_features)
        Cluster centroids in original space (NOT normalized)
    inertia : float
        Sum of cosine distances from points to their assigned centroids
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    best_inertia = np.inf
    best_centers = None
    best_labels = None

    for init_idx in range(n_init):
        # Random initialization: pick n_clusters random points as initial centers
        idx = np.random.choice(n_samples, n_clusters, replace=False)
        centers = X[idx].copy()

        for iteration in range(max_iter):
            # Compute cosine similarity between all points and all centers
            # cos_sim[i, k] = X[i]·centers[k] / (||X[i]|| * ||centers[k]||)

            # Normalize for similarity computation
            X_norm = np.linalg.norm(X, axis=1, keepdims=True)
            X_norm = np.maximum(X_norm, 1e-10)  # Avoid division by zero
            X_normed = X / X_norm

            C_norm = np.linalg.norm(centers, axis=1, keepdims=True)
            C_norm = np.maximum(C_norm, 1e-10)
            C_normed = centers / C_norm

            # Cosine similarity matrix
            cos_sim = X_normed @ C_normed.T

            # Assign each point to cluster with MAXIMUM similarity
            # (equivalent to MINIMUM cosine distance, since d_cos = 1 - similarity)
            labels = np.argmax(cos_sim, axis=1)

            # Update centers as arithmetic mean of assigned points IN ORIGINAL SPACE
            # This is the key: MATLAB does not normalize the centroids
            new_centers = np.zeros((n_clusters, n_features), dtype=np.float64)
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    # Empty cluster: reinitialize with random point
                    new_centers[k] = X[np.random.randint(n_samples)]

            # Check convergence by centroid shift
            center_shift = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if center_shift < tol:
                break

        # Compute final inertia (sum of cosine distances)
        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = np.maximum(X_norm, 1e-10)
        X_normed = X / X_norm

        C_norm = np.linalg.norm(centers, axis=1, keepdims=True)
        C_norm = np.maximum(C_norm, 1e-10)
        C_normed = centers / C_norm

        cos_sim = X_normed @ C_normed.T
        # Get similarity for each point to its assigned center
        assigned_sim = cos_sim[np.arange(n_samples), labels]
        # Convert to distance: d = 1 - similarity
        cos_dist = 1.0 - assigned_sim
        inertia = np.sum(cos_dist)

        # Keep best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels

    return best_labels, best_centers, best_inertia



