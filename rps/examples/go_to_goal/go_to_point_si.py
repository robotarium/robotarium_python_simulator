import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_poses, create_at_position

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 6

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
initial_positions = generate_random_poses(N, spacing=0.5)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_positions)

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
si_barrier_certificate = create_si_barrier_certificate_with_boundary()
si_position_controller = create_si_position_controller()
si_to_uni = create_si_to_uni_dynamics()

position_error = 0.05
at_position = create_at_position(position_error=position_error)

# =========================================================
# GOAL INITIALIZATION
# =========================================================
arena_width = r.BOUNDARIES[1] - r.BOUNDARIES[0] - 3*r.ROBOT_DIAMETER
arena_height = r.BOUNDARIES[3] - r.BOUNDARIES[2] - 3*r.ROBOT_DIAMETER

goal_points = generate_random_poses(N, width=arena_width, height=arena_height, spacing=0.5)
goal_points = goal_points[:2, :] # Position only

x = r.get_poses()
r.step()

# =========================================================
# MAIN LOOP
# =========================================================
already_reported = np.zeros(N, dtype=bool)

# at_position returns (all_done, per_robot_done_array)
while not at_position(x, goal_points)[0]:
    x = r.get_poses()

    dxi = si_position_controller(x[:2, :], goal_points)
    dxi = si_barrier_certificate(dxi, x)
    dxu = si_to_uni(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

    # Reporting logic
    _, converged = at_position(x, goal_points)
    newly_arrived = converged & ~already_reported
    for i in np.where(newly_arrived)[0]:
        print(f"Robot {i+1} has reached its goal position.")
    already_reported |= converged

r.debug()