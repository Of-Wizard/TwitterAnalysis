import networkx as nx


def hits(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    """
    :param G: A NetworkX graph
    :param max_iter: Maximum number of iterations in power method.
    :param tol: Error tolerance used to check convergence in power method iteration.
    :param nstart: Starting value of each node for power method iteration.
    :param normalized: Normalize results by the sum of all of the values.
    :return: (hubs,authorities) : two-tuple of dictionaries. Two dictionaries keyed by node containing the hub and authority
       values.
    """
    import numpy as np
    import scipy as sp
    import scipy.sparse.linalg  # call as sp.sparse.linalg

    if len(G) == 0:
        return {}, {}
    A = nx.adjacency_matrix(G, nodelist=list(G), dtype=float)

    if nstart is None:
        _, _, vt = sp.sparse.linalg.svds(A, k=1, maxiter=max_iter, tol=tol)
    else:
        nstart = np.array(list(nstart.values()))
        _, _, vt = sp.sparse.linalg.svds(A, k=1, v0=nstart, maxiter=max_iter, tol=tol)

    a = vt.flatten().real
    h = A @ a
    if normalized:
        h /= h.sum()
        a /= a.sum()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return hubs, authorities
