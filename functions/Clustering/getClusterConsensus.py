import numpy as np


def getClusterConsensus(IDX, Consensus):
    """Python translation of getClusterConsensus.m

    Computes the average consensus within each cluster.

    Parameters
    ----------
    IDX : array_like, shape (n_items,)
        Cluster assignments (1..nClus in MATLAB; here assumed >=1 ints).
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
    nClus = int(IDX.max())

    iCAPs_consensus = np.zeros(nClus, dtype=float)
    iCAPs_nItems = np.zeros(nClus, dtype=int)

    for iC in range(1, nClus + 1):
        clusID = np.where(IDX == iC)[0]
        if clusID.size == 0:
            continue
        # sub-consensus matrix for this cluster
        sub = Consensus[np.ix_(clusID, clusID)]
        iCAPs_consensus[iC - 1] = np.mean(sub)
        iCAPs_nItems[iC - 1] = clusID.size

    return iCAPs_consensus, iCAPs_nItems
