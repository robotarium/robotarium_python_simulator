import numpy as np
import rps.robotarium as robotarium
from rps.utilities.graph import cycleGL, topological_neighbors
from rps.utilities.transformations import create_si_to_uni_mapping, create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 12
iterations = 2000

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# =========================================================
# GRAPH TOPOLOGY & SAFETY SETUP
# =========================================================
L = cycleGL(N)

# Use more robust dynamics mapping for consensus
_, uni_to_si_states = create_si_to_uni_mapping()
si_to_uni_dyn = create_si_to_uni_dynamics()

# Safety barrier certificate (applied in unicycle space)
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for k in range(iterations):
    x = r.get_poses()
    xi = uni_to_si_states(x)

    dxi = np.zeros((2, N))

    # Compute consensus control
    for i in range(N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            dxi[:, i] += (xi[:, j] - xi[:, i])

    # ---------------------------------------------------------
    # Velocity Thresholding
    # Cap speeds at 3/4 of max linear velocity (0.2 * 0.75 = 0.15 m/s)
    # ---------------------------------------------------------
    norms = np.linalg.norm(dxi, axis=0)
    threshold = 0.75 * r.MAX_LINEAR_VELOCITY
    to_thresh = norms > threshold
    
    # Avoid division by zero with small norms
    if np.any(to_thresh):
        dxi[:, to_thresh] *= threshold / norms[to_thresh]

    # ---------------------------------------------------------
    # Mapping and Safety
    # ---------------------------------------------------------
    dxu = si_to_uni_dyn(dxi, x)
    dxu = uni_barrier_cert(dxu, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

r.debug()