import numpy as np

def get_cluster_consensus(IDX, Consensus):
    """
    Computes the average consensus within each cluster. (m(k) in Monti et al.,2003)

    Parameters
    ----------
    IDX : array_like, shape (n_items,)
        Cluster assignments, 0-based (0..nClus-1), matching the Python pipeline convention.
    Consensus : ndarray, shape (n_items, n_items)
        Consensus matrix.

    Returns
    -------
    iCAPs_consensus : ndarray, shape (nClus,)
        Mean consensus value for each cluster.
    iCAPs_nItems : ndarray, shape (nClus,)
        Number of items in each cluster.
    """
    IDX = np.asarray(IDX).astype(int)
    Consensus = np.asarray(Consensus)
    nClus = int(IDX.max()) + 1 # 0-based: max label is nClus-1

    iCAPs_consensus = np.zeros(nClus, dtype=float)
    iCAPs_nItems = np.zeros(nClus, dtype=int)

    for iC in range(nClus):
        clusID = np.where(IDX == iC)[0]
        if clusID.size == 0:
            continue
        # sub-consensus matrix for this cluster
        sub = Consensus[np.ix_(clusID, clusID)]
        iCAPs_consensus[iC] = np.mean(sub)
        iCAPs_nItems[iC] = clusID.size

    return iCAPs_consensus, iCAPs_nItems


