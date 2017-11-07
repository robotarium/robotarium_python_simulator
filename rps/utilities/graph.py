import numpy as np

def cycle_GL(N):
    ones = np.ones(N-1)
    L = 2*np.identity(N) - np.diag(ones, 1) - np.diag(ones, -1)
    L[N-1, 0] = -1
    L[0, N-1] = -1

    return L

def topological_neighbors(L, agent):
    row = L[agent, :]
    N = np.size(row)
    # Since L = D - A
    return np.where(row < 0)[0]
