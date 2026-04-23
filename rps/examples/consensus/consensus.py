import numpy as np
import rps.robotarium as robotarium
from rps.utilities.graph import cycleGL, topological_neighbors
from rps.utilities.transformations import create_si_to_uni_mapping

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 12
iterations = 1000

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# =========================================================
# GRAPH TOPOLOGY & CONTROLLER SETUP
# =========================================================
# Generate a connected cyclic graph Laplacian
L = cycleGL(N)

# Get the SI/UNI mapping functions
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for k in range(iterations):
    # Retrieve current poses
    x = r.get_poses()
    xi = uni_to_si_states(x)

    # Initialize SI velocity vector
    dxi = np.zeros((2, N))

    # Consensus algorithm: Each agent sums relative positions of neighbors
    for i in range(N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            dxi[:, i] += (xi[:, j] - xi[:, i])

    # Convert to unicycle velocities
    dxu = si_to_uni_dyn(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

r.debug()