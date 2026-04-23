import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.graph import completeGL, topological_neighbors
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_poses

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 4  # 1 leader + 3 followers
iterations = 5000 
formation_control_gain = 10
desired_distance = 0.2  # Matches MATLAB version

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
# Initializing with the same constraints as the MATLAB script
initial_positions = generate_random_poses(N, width=1, height=1, spacing=0.5)
r = robotarium.Robotarium(
    number_of_robots=N, 
    show_figure=True, 
    initial_conditions=initial_positions,
    sim_in_real_time=True
)

# =========================================================
# GRAPH LAPLACIAN AND FORMATION SETUP
# =========================================================
# MATLAB: L(2:N, 2:N) = followers; L(2, 2) = L(2, 2) + 1; L(2, 1) = -1;
# Python (0-indexed):
followers = -completeGL(N-1)
L = np.zeros((N, N))
L[1:N, 1:N] = followers
L[1, 1] += 1
L[1, 0] = -1

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
# SI-to-Uni mapping with specific MATLAB gain (0.8)
si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=0.8)

# Unicycle-space barrier certificate
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()

# Leader controller drives the robot toward waypoints
leader_controller = create_si_position_controller(
    x_velocity_gain=0.8, 
    y_velocity_gain=0.8, 
    velocity_magnitude_limit=0.1
)

# =========================================================
# WAYPOINT INITIALIZATION
# =========================================================
# Corner waypoints
waypoints = np.array([[-1, -1, 1, 1], [0.8, -0.8, -0.8, 0.8]])
close_enough = 0.05
state = 0 

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for t in range(iterations):
    x = r.get_poses()
    dxi = np.zeros((2, N))

    # 1. Formation control for followers (Indices 1 to N-1)
    for i in range(1, N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            # Potential field: gain * (||actual||^2 - desired^2) * relative_pos
            dist_sq = np.linalg.norm(x[:2, j] - x[:2, i])**2
            dxi[:, i] += formation_control_gain * (dist_sq - desired_distance**2) * (x[:2, j] - x[:2, i])

    # 2. Leader waypoint tracking
    waypoint = waypoints[:, state].reshape(2, 1)
    dxi[:, 0] = leader_controller(x[:2, 0].reshape(2, 1), waypoint).flatten()

    # Advance waypoint state if leader is close
    if np.linalg.norm(x[:2, 0].reshape(2, 1) - waypoint) < close_enough:
        state = (state + 1) % waypoints.shape[1]

    # 3. Velocity thresholding (3/4 of max linear velocity)
    # Applied only to followers (Indices 1 to N-1)
    norms = np.linalg.norm(dxi, axis=0)
    threshold = 0.75 * r.MAX_LINEAR_VELOCITY
    to_thresh = (norms > threshold)
    to_thresh[0] = False # Exclude leader
    
    if np.any(to_thresh):
        dxi[:, to_thresh] *= threshold / norms[to_thresh]

    # 4. Conversion and Safety
    # Convert SI velocities to unicycle commands
    dxu = si_to_uni_dyn(dxi, x)
    # Apply barrier certificate in unicycle space
    dxu = uni_barrier_cert(dxu, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

# Print final results
r.debug()