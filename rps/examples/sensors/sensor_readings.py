import numpy as np

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import create_at_position

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 1

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
    [[-1.0, -0.5], [-1.0, 0.5]],
    [[-0.6, -0.5], [-0.6, 0.5]],
    [[ 0.6, -0.5], [ 0.6, 0.5]],
    [[ 1.0, -0.5], [ 1.0, 0.5]],
    [[-1.4, -0.9], [-1.4, 0.9]],
    [[-1.4,  0.9], [ 1.4, 0.9]],
    [[ 1.4,  0.9], [ 1.4, -0.9]],
    [[ 1.4, -0.9], [-1.4, -0.9]],
])

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
initial_conditions = np.array([[-1.2], [-0.6], [0.0]])

r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=False,
    initial_conditions=initial_conditions,
    use_distance_sensors=True,
    obstacles=obstacles,
    show_distance_rays=True
)

# =========================================================
# CONTROLLER SETUP
# =========================================================

# Single-integrator position controller drives the robot to each waypoint
controller = create_si_position_controller()

# Maps single-integrator velocities to unicycle commands
si_to_uni_dynamics = create_si_to_uni_dynamics()

# Position-only convergence check (no heading requirement)
at_position = create_at_position(position_error=0.05)

# =========================================================
# GOAL GRID INITIALIZATION
# =========================================================

# Generate a serpentine grid of goal waypoints covering the arena.
# Even columns (1-indexed) are flipped vertically so the robot traces
# a continuous back-and-forth path rather than resetting each column.
resolution_x = 0.4
resolution_y = 0.4
x_points = np.arange(-1.2, 1.2 + resolution_x, resolution_x)
y_points = np.arange(-0.6, 0.6 + resolution_y, resolution_y)
x_goals, y_goals = np.meshgrid(x_points, y_points)
y_goals[:, 1::2] *= -1  # Serpentine: flip every even column (0-indexed odd = MATLAB even)

# =========================================================
# DATA STRUCTURE INITIALIZATION
# =========================================================

# Pre-allocate first column of each field; columns are appended as the
# robot reaches each waypoint — matches MATLAB's column-append convention.
data = {
    'poses':           np.zeros((3, 1)),
    'distances':       np.zeros((7, 1)),
    'accelerations':   np.zeros((3, 1)),
    'magnetic_fields': np.zeros((3, 1)),
    'orientations':    np.zeros((1, 1)),  # simulator outputs yaw only (degrees)
    'encoders':        np.zeros((2, 1)),
}

# =========================================================
# MAIN EXPERIMENT LOOP
# =========================================================

index = 0  # Current goal index into flattened x_goals / y_goals

while index < x_goals.size:

    # Retrieve the most recent poses from the Robotarium.
    x = r.get_poses()

    goal = np.array([[x_goals.T.flatten()[index]],
                     [y_goals.T.flatten()[index]]])  # (2, 1)

    # ---------------------------------------------------------
    # Record sensor data when the robot reaches the current goal
    # ---------------------------------------------------------
    if at_position(x[:2, :], goal)[0]:

        # --- Distance sensors ---
        # 7 sensors per robot. Max range 1.2 m. -1 = no detection.
        # Sensor orientations in robot frame [x; y; theta] (m, m, rad):
        #   [-0.04,  0.00,  0.04,  0.05,  0.04,  0.00, -0.04]
        #   [ 0.04,  0.06,  0.05,  0.00, -0.05, -0.06, -0.04]
        #   [  pi,  pi/2,  pi/4,  0.00, -pi/4, -pi/2,   -pi ]
        distances = r.get_distances()         # (7, N)

        # --- IMU: accelerometer ---
        # Acceleration in robot body frame (m/s^2):
        #   X: Forward,  Y: Left,  Z: Up
        accelerations = r.get_accelerations() # (3, N)

        # --- IMU: magnetometer ---
        # Magnetic field in robot body frame (uT).
        # Simulated from testbed data; X aligns with magnetic north at theta=0.
        magnetic_fields = r.get_magnetic_fields() # (3, N)

        # --- IMU: fused orientation ---
        # Sensor-fused yaw in degrees, wrapped to [0, 360).
        # 0 degrees is aligned with the positive x-axis of the robotarium.
        orientations = r.get_orientations()   # (N,)

        # --- Wheel encoders ---
        # Cumulative tick counts since experiment start [left; right].
        # 28 counts/rev x 100.37 gear ratio ~= 2810 ticks/wheel revolution.
        # Wheel radius: 0.016 m,  Wheelbase: 0.105 m.
        encoders = r.get_encoders()           # (2, N)

        # Store readings for this waypoint (column-append, matching MATLAB)
        data['poses']           = np.hstack((data['poses'],           x))
        data['distances']       = np.hstack((data['distances'],       distances))
        data['accelerations']   = np.hstack((data['accelerations'],   accelerations))
        data['magnetic_fields'] = np.hstack((data['magnetic_fields'], magnetic_fields))
        data['orientations']    = np.hstack((data['orientations'],    orientations.reshape(1, N)))
        data['encoders']        = np.hstack((data['encoders'],        encoders))

        index += 1

    # ---------------------------------------------------------
    # Compute and send control inputs
    # ---------------------------------------------------------
    dxi = controller(x[:2, :], goal)
    dxu = si_to_uni_dynamics(dxi, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

# =========================================================
# DEBUG REPORT
# =========================================================
r.debug()