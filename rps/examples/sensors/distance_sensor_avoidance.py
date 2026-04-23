#
# As an example of how to use the distance sensors, this example has a robot navigate through
# a maze of other robots and obstacles making sure to avoid collisions.
#

import numpy as np
from numpy.typing import NDArray
import matplotlib.patches as patches
import warnings

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import create_at_position
from rps.utilities.barrier_certificates import _solve_qp


# Try to avoid obstacles by 0.15m
AVOID_DISTANCE = 0.15


def obstacle_barrier_certificate(
    dxi: NDArray[np.floating],
    x: NDArray[np.floating],
    centers: NDArray[np.floating],
    magnitude_limit: float = 0.15,
) -> NDArray[np.floating]:
    """
    Simple barrier certificate to avoid collisions with obstacles.  If a robot is within
    AVOID_DISTANCE of an obstacle
    """
    num_obstacles = centers.shape[0]
    num_constraints = num_obstacles + 8
    A = np.zeros((num_constraints, 2))
    b = np.zeros(num_constraints)

    # Apply Obstacle Constraints
    for i in range(num_obstacles):
        diff = x[:2] - centers[i, :]
        h = np.dot(diff, diff) - AVOID_DISTANCE**2

        A[i, :] = -2 * diff
        b[i] = 0.1 * h

    constraint = num_obstacles
    # vx <= magnitude_limit
    A[constraint, :] = [1, 0]
    b[constraint] = magnitude_limit * np.cos(np.pi / 8)
    constraint += 1

    # 1/sqrt(2) * (vx + vy) <= magnitude_limit
    A[constraint, :] = [1/np.sqrt(2), 1/np.sqrt(2)]
    b[constraint] = magnitude_limit * np.cos(np.pi / 8)
    constraint += 1

    # vy <= magnitude_limit
    A[constraint, :] = [0, 1]
    b[constraint] = magnitude_limit * np.cos(np.pi / 8)
    constraint += 1

    # 1/sqrt(2) * (-vx + vy) <= magnitude_limit
    A[constraint, :] = [-1/np.sqrt(2), 1/np.sqrt(2)]
    b[constraint] = magnitude_limit * np.cos(np.pi / 8)
    constraint += 1

    # -vx <= magnitude_limit
    A[constraint, :] = [-1, 0]
    b[constraint] = magnitude_limit * np.cos(np.pi / 8)
    constraint += 1

    # 1/sqrt(2) * (-vx - vy) <= magnitude_limit
    A[constraint, :] = [-1/np.sqrt(2), -1/np.sqrt(2)]
    b[constraint] = magnitude_limit * np.cos(np.pi / 8)
    constraint += 1

    # -vy <= magnitude_limit
    A[constraint, :] = [0, -1]
    b[constraint] = magnitude_limit * np.cos(np.pi / 8)
    constraint += 1

    # 1/sqrt(2) * (vx - vy) <= magnitude_limit
    A[constraint, :] = [1/np.sqrt(2), -1/np.sqrt(2)]
    b[constraint] = magnitude_limit * np.cos(np.pi / 8)
    constraint += 1

    vnew = _solve_qp(1, dxi, A, b)
    if vnew is None:
        warnings.warn("obstacle_barrier_certificate: QP failed. Commanding zero velocity.")
        return np.zeros(2)
    return vnew.flatten()

# =========================================================
# OBSTACLE SETUP
# =========================================================

# Obstacles are defined as line segments for distance sensor simulation
# only — there is no collision simulation with these obstacles.
#
# Each obstacle is a 2x2 array of two endpoints:
#   [[x1, y1],
#    [x2, y2]]
# Multiple obstacles are stacked along the 3rd axis: shape (M, 2, 2).
#
# NOTE: The arena boundary is NOT treated as an obstacle for distance
# sensors by default. Add them manually if needed. Arena corners:
#   Bottom-left: (-1.6, -1)   Bottom-right: (1.6, -1)
#   Top-right:   ( 1.6,  1)   Top-left:    (-1.6,  1)

obstacles = np.array([
    [[-1.6, -1.0], [1.6, -1.0]],
    [[-1.6,  1.0], [1.6,  1.0]],
    [[-1.6, -1.0], [-1.6, 1.0]],
    [[1.6, -1.0], [1.6, 1.0]],
])


# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================

initial_conditions = np.array([
    [-1.2, -0.6, 0.0, 0.0, 0.6],
    [0.0, 0.0, 0.6, -0.6, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])

N = 5
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=False,
    initial_conditions=initial_conditions,
    obstacles=obstacles,
    use_distance_sensors=True,
    show_distance_endpoints=False,
    show_distance_rays=False,
    show_obstacles=True,
    skip_initialization=True,
)


# =========================================================
# CONTROLLER SETUP
# =========================================================

# Single-integrator position controller drives the robot to each waypoint
controller = create_si_position_controller()

# Maps single-integrator velocities to unicycle commands
si_to_uni_dynamics = create_si_to_uni_dynamics()

# Position-only convergence check (no heading requirement)
at_position = create_at_position(position_error=AVOID_DISTANCE)

# Initialize and plot the goals
goals = initial_conditions.copy()
goals[:2, 0] = [1.2, 0.0]
goal_patch = patches.Circle(goals[:2, 0], AVOID_DISTANCE-0.02, color='b', alpha=0.5)
r._axes_handle.add_patch(goal_patch)
goal_label = r._axes_handle.text(goals[0, 0], goals[1, 0], 'Goal', fontsize=12, ha='center', va='center')

# A hitpoint is created when a sensor detects an obstacle within 1m.
# If a hitpoint is detected 5 times, it is classified as an obstacle and plotted.
# This is a simple way to filter out sensor noise and avoid plotting every single sensor reading.
hitpoints = []
obstacle_locations = []
obstacle_patches = []

x = r.get_poses()
r.step()

while not at_position(x[:2, :], goals[:2, :])[0]:
    # Retrieve the most recent poses from the Robotarium
    x = r.get_poses()

    # Retrieve the most recent distance sensor readings from the Robotarium
    distances = r.get_distances()
    # Retrive the distance endpoints from the Robotarium
    endpoints = r.get_distance_endpoints()
    # Iterate through robot 0's distance sensor readings
    for (i, distance) in enumerate(distances[:, 0]):
        if distance > 0.0 and distance < 1.0:
            end_point = endpoints[:, i, 0]
            
            # Check if its a new obstacle
            new_obstacle = False
            new_hitpoint = True
            for hitpoint in hitpoints:
                if np.linalg.norm(end_point - hitpoint[0]) <= AVOID_DISTANCE:
                    new_hitpoint = False
                    hitpoint[1] += 1
                    if hitpoint[1] == 5:
                        new_obstacle = True
                        break
            if new_hitpoint:
                hitpoints.append([end_point, 1])

            # Plot the new obstacle (if it is new)
            if new_obstacle:
                obstacle_locations.append(end_point)
                obstacle_patch = patches.Circle(end_point, AVOID_DISTANCE-0.02, color='r', alpha=0.5)
                r._axes_handle.add_patch(obstacle_patch)
                obstacle_patches.append(obstacle_patch)

    # ---------------------------------------------------------
    # Compute and send control inputs
    # ---------------------------------------------------------
    dxi = controller(x[:2, :], goals[:2, :])
    dxi[:, 0] = obstacle_barrier_certificate(dxi[:, 0], x[:, 0], np.array(obstacle_locations))
    dxi = si_to_uni_dynamics(dxi, x)
    r.set_velocities(np.arange(N), dxi)
    r.step()

r.debug()