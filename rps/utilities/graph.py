import numpy as np

def cycle_GL(N):
    """ Generates a graph Laplacian for a cycle graph

    N: int (number of agents)

    -> NxN numpy array (representing the graph Laplacian)
    """
    ones = np.ones(N-1)
    L = 2*np.identity(N) - np.diag(ones, 1) - np.diag(ones, -1)
    L[N-1, 0] = -1
    L[0, N-1] = -1

    return L

def topological_neighbors(L, agent):
    """ Returns the neighbors of a particular agent using the graph Laplacian

    L: NxN numpy array (representing the graph Laplacian)
    agent: int (agent: 1 - N)

    -> 1xM numpy array (with M neighbors)
    """
    row = L[agent, :]
    N = np.size(row)
    # Since L = D - A
    return np.where(row < 0)[0]
