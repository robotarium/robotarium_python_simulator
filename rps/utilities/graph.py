import numpy as np

def completeGL(n):
    """
    Return the graph Laplacian of a complete graph on n nodes.

    In a complete graph every node is connected to every other node, so
    each node has degree n-1.  The Laplacian is L = n*I - 11^T.
    """
    assert isinstance(n, int) and n > 0, "n must be a positive integer."
    
    L = n * np.eye(n) - np.ones((n, n))
    return L


def cycleGL(n):
    """
    Return the graph Laplacian of a cycle graph on n nodes.
    
    In a cycle graph every node has exactly two neighbours, so every diagonal 
    entry is 2. The two wrap-around edges between nodes 0 and n-1 are added
    explicitly.
    """
    assert isinstance(n, int) and n >= 3, "n must be an integer >= 3."
    
    L = 2 * np.eye(n) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
    
    # Close the ring
    L[0, n-1] = -1
    L[n-1, 0] = -1
    return L


def lineGL(n):
    """
    Return the graph Laplacian of a path (line) graph on n nodes.
    
    In a path graph interior nodes have degree 2 and the two endpoint
    nodes have degree 1.
    """
    assert isinstance(n, int) and n >= 2, "n must be an integer >= 2."
    
    L = 2 * np.eye(n) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
    
    # Correct the two endpoint degrees from 2 to 1
    L[0, 0] = 1
    L[n-1, n-1] = 1
    return L


def random_connectedGL(v, e):
    """
    Return the graph Laplacian of a random connected graph.
    
    Generates a guaranteed-connected graph on v nodes using a random
    spanning tree (Prüfer-style), then adds up to e extra edges chosen 
    uniformly at random from the remaining candidate pairs.
    """
    assert isinstance(v, int) and v > 0, "v must be a positive integer."
    assert isinstance(e, int) and e >= 0, "e must be a non-negative integer."
    
    L = np.zeros((v, v))
    
    # ── Build a random spanning tree ──
    # Add each node i (1..v-1) with an edge to a uniformly random predecessor j < i.
    for i in range(1, v):
        j = np.random.randint(0, i)
        L[i, j] = -1
        L[j, i] = -1
        L[i, i] += 1
        L[j, j] += 1
        
    # ── Add extra random edges ──
    # Candidate pairs are strictly upper-triangular entries that are not yet edges
    candidates = np.argwhere(np.triu(L == 0, 1))
    
    numEdges = min(e, len(candidates))
    if numEdges > 0:
        np.random.shuffle(candidates)
        chosen = candidates[:numEdges]
        
        for idx in range(numEdges):
            i, j = chosen[idx]
            L[i, j] = -1
            L[j, i] = -1
            L[i, i] += 1
            L[j, j] += 1
            
    return L


def randomGL(v, e):
    """
    Return the graph Laplacian of a random graph on v nodes with at most e edges.
    
    Selects up to e edges uniformly at random from all possible undirected
    pairs. The resulting graph may be disconnected.
    """
    assert isinstance(v, int) and v > 0, "v must be a positive integer."
    assert isinstance(e, int) and e >= 0, "e must be a non-negative integer."
    
    L = np.zeros((v, v))
    
    # Enumerate all unique undirected pairs (strictly upper triangle)
    candidates = np.argwhere(np.triu(np.ones((v, v)), 1))
    
    numEdges = min(e, len(candidates))
    if numEdges > 0:
        np.random.shuffle(candidates)
        chosen = candidates[:numEdges]
        
        for idx in range(numEdges):
            i, j = chosen[idx]
            L[i, j] = -1
            L[j, i] = -1
            L[i, i] += 1
            L[j, j] += 1
            
    return L


def delta_disk_neighbors(poses, agent, delta):
    """
    Return the indices of all agents within a given distance of a specified agent.
    
    Inputs
    ------
    poses : 2xN (or larger) matrix of agent positions
    agent : index of the query agent (0-based integer)
    delta : neighbourhood radius (m)
    """
    N = poses.shape[1]
    assert 0 <= agent < N, f"agent ({agent}) must be between 0 and {N-1}."
    
    # Build index list of all agents except the query agent
    others = np.concatenate((np.arange(agent), np.arange(agent + 1, N)))
    
    if len(others) == 0:
        return np.array([], dtype=int)
        
    # Keep only those within the delta-disk
    dists = np.linalg.norm(poses[:2, others] - poses[:2, agent:agent+1], axis=0)
    in_range = dists <= delta
    
    return others[in_range]


def topological_neighbors(L, agent):
    """
    Return the indices of an agent's graph neighbours.
    
    Reads connectivity directly from the graph Laplacian.
    
    Inputs
    ------
    L     : NxN graph Laplacian
    agent : index of the query agent (0-based integer)
    """
    N = L.shape[0]
    assert 0 <= agent < N, f"agent ({agent}) must be between 0 and {N-1}."
    
    # Extract the agent's row and zero out the diagonal before searching
    row = np.copy(L[agent, :])
    row[agent] = 0
    
    neighbors = np.where(row != 0)[0]
    return neighbors