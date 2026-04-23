import numpy as np

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 10
iterations = 3000

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================

# Single-integrator position controller 
# Explicitly matching MATLAB defaults (Python defaults to 1.0)
position_control = create_si_position_controller(
    x_velocity_gain=0.8, 
    y_velocity_gain=0.8, 
    velocity_magnitude_limit=0.15
)

# Barrier certificate prevents collisions and boundary violations
# Explicitly matching MATLAB defaults
si_barrier_certificate = create_si_barrier_certificate_with_boundary(
    safety_radius=0.15, 
    barrier_gain=100.0, 
    magnitude_limit=0.2, 
    boundary_points=np.array([-1.7, 1.7, -1.1, 1.1])
)

# Maps single-integrator velocities to unicycle commands
# Explicitly matching MATLAB defaults
si_to_uni_dyn = create_si_to_uni_dynamics(
    linear_velocity_gain=1.0, 
    angular_velocity_limit=np.pi
)

# =========================================================
# GOAL INITIALIZATION
# =========================================================

# Distribute 2N points evenly around an ellipse. Robots alternate between
# the first N points and the antipodal N points.
xybound = np.array([-1.0, 1.0, -0.8, 0.8])

p_theta = 2 * np.pi * (np.arange(1, 2*N, 2) / (2*N))
p_circ = np.vstack([
    np.hstack([xybound[1] * np.cos(p_theta), xybound[1] * np.cos(p_theta + np.pi)]),
    np.hstack([xybound[3] * np.sin(p_theta), xybound[3] * np.sin(p_theta + np.pi)])
])

flag = 0
x_goal = p_circ[:, :N]

for i in range(iterations):

    # Retrieve the most recent poses from the Robotarium.
    x = r.get_poses()
    x_position = x[:2, :]

    # ---------------------------------------------------------
    # Check if all robots are close enough to their goals,
    # then flip the flag to swap to the antipodal goal set
    # ---------------------------------------------------------
    if np.linalg.norm(x_goal - x_position, 1) < 0.03:
        flag = 1 - flag

    if flag == 0:
        x_goal = p_circ[:, :N]
    else:
        x_goal = p_circ[:, N:]

    # ---------------------------------------------------------
    # Compute control inputs and apply safety
    # ---------------------------------------------------------

    # Drive robots toward their current goal positions
    dxi = position_control(x_position, x_goal)

    # Apply barrier certificate to prevent collisions and boundary violations
    dxi = si_barrier_certificate(dxi, x)

    # Convert single-integrator to unicycle velocities
    dxu = si_to_uni_dyn(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

# =========================================================
# DEBUG REPORT
# =========================================================
r.debug()