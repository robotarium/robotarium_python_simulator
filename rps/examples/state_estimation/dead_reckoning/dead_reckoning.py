# DEAD_RECKONING_WITH_PLOTTING
# Drives N robots to randomized goal positions while estimating each
# robot's pose in real time using two dead reckoning methods:
#
#   1. Encoder-only  — unicycle model integrating wheel displacements
#   2. IMU + Encoder — encoder for translation, IMU yaw for heading
#
# Sensor data is saved to .npy files at the end of the experiment for
# offline analysis.
#
# Sean Wilson
# 03/2026

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping, create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import create_at_position, generate_random_positions

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N            = 3    # Number of robots
final_count  = 3    # Number of goal sets to complete before stopping
goal_spacing = 0.5  # Minimum spacing between goal points (metres)

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
r = robotarium.Robotarium(number_of_robots=N, show_figure=True)

# =========================================================
# ROBOT PHYSICAL CONSTANTS
# =========================================================
# Pull from class constants so these stay in sync with the simulator
wheel_radius         = r.WHEEL_RADIUS
base_length          = r.BASE_LENGTH
counts_per_wheel_rev = r.ENCODER_COUNTS_PER_REVOLUTION * r.MOTOR_GEAR_RATIO
dt                   = r.TIME_STEP

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
single_integrator_position_controller = create_si_position_controller()
si_barrier_cert = create_si_barrier_certificate_with_boundary()
_, uni_to_si_states = create_si_to_uni_mapping()
si_to_uni_dyn = create_si_to_uni_dynamics()

# Goal-set checker: loose tolerance, position only
all_reached_checker = create_at_position(position_error=0.10)

# Per-robot checker: tighter tolerance to zero individual velocities
at_goal_checker = create_at_position(position_error=0.08)

# =========================================================
# GOAL INITIALIZATION
# =========================================================
arena_width  = (r.BOUNDARIES[1] - r.BOUNDARIES[0]) - 3 * r.ROBOT_DIAMETER
arena_height = (r.BOUNDARIES[3] - r.BOUNDARIES[2]) - 3 * r.ROBOT_DIAMETER

# goal_points is (2, N) — positions only, no heading
goal_points = generate_random_positions(N, spacing=goal_spacing,
                                        width=arena_width, height=arena_height)

# =========================================================
# DEAD RECKONING STATE INITIALIZATION
# =========================================================
# Both estimators start from the true pose at t=0.
x_init = r.get_poses()    # (3, N)
r.step()

encoder_pose     = x_init.copy()                   # (3, N)
imu_encoder_pose = x_init.copy()                   # (3, N)
encoder_ref      = r.get_encoders().astype(float)  # (2, N) baseline ticks

# Align IMU yaw to Vicon heading at t=0.
# get_orientations() returns (N,) yaw in degrees, [0, 360).
raw_imu_init   = r.get_orientations()                      # (N,)
imu_yaw_offset = np.deg2rad(raw_imu_init) - x_init[2, :]  # (N,)

# =========================================================
# DATA STORAGE INITIALIZATION
# =========================================================
# Pre-populate timestep 0 from x_init and readings after the first step.
# Lists of (K, N) frames are stacked to (K, N, T) at save time.
vicon_list      = [x_init]
imu_orient_list = [r.get_orientations().reshape(1, N)]
encoder_list    = [r.get_encoders().astype(float)]
gyro_list       = [r.get_gyros()]
acc_list        = [r.get_accelerations()]
mag_list        = [r.get_magnetic_fields()]

# =========================================================
# MAIN EXPERIMENT LOOP
# =========================================================
count = 0

while True:

    x    = r.get_poses()        # (3, N)
    x_si = uni_to_si_states(x)  # (2, N)

    # ---------------------------------------------------------
    # Record sensor data
    # ---------------------------------------------------------
    vicon_list.append(x)
    imu_orient_list.append(r.get_orientations().reshape(1, N))
    encoder_list.append(r.get_encoders().astype(float))
    gyro_list.append(r.get_gyros())
    acc_list.append(r.get_accelerations())
    mag_list.append(r.get_magnetic_fields())

    # ---------------------------------------------------------
    # Dead reckoning update for each robot
    # ---------------------------------------------------------
    raw_imu = r.get_orientations()   # (N,) yaw in degrees

    T = len(encoder_list)
    for i in range(N):
        # Encoder tick delta since last step
        if T >= 2:
            d_counts = encoder_list[-1][:, i] - encoder_list[-2][:, i]
        else:
            d_counts = np.zeros(2)

        # Tick delta -> wheel displacement (metres)
        d_left   = (d_counts[0] / counts_per_wheel_rev) * 2 * np.pi * wheel_radius
        d_right  = (d_counts[1] / counts_per_wheel_rev) * 2 * np.pi * wheel_radius
        d_center = (d_right + d_left) / 2

        # --- Encoder-only unicycle model ---
        d_theta   = (d_right - d_left) / base_length
        theta_enc = encoder_pose[2, i]
        encoder_pose[0, i] += d_center * np.cos(theta_enc + d_theta / 2)
        encoder_pose[1, i] += d_center * np.sin(theta_enc + d_theta / 2)
        encoder_pose[2, i]  = theta_enc + d_theta

        # --- IMU + encoder model: encoder for translation, IMU for heading ---
        # Wrap to (-pi, pi] so a 360->0 degree crossing in the IMU output
        # does not cause a 2*pi jump in the estimated heading.
        raw_yaw_rad = np.deg2rad(raw_imu[i]) - imu_yaw_offset[i]
        imu_yaw_rad = np.arctan2(np.sin(raw_yaw_rad), np.cos(raw_yaw_rad))
        imu_encoder_pose[0, i] += d_center * np.cos(imu_yaw_rad)
        imu_encoder_pose[1, i] += d_center * np.sin(imu_yaw_rad)
        imu_encoder_pose[2, i]  = imu_yaw_rad

    # ---------------------------------------------------------
    # Goal-set completion check
    # ---------------------------------------------------------
    if all_reached_checker(x_si, goal_points)[0]:
        count += 1
        if count == final_count:
            break
        goal_points = generate_random_positions(N, spacing=goal_spacing,
                                                width=arena_width, height=arena_height)

    # ---------------------------------------------------------
    # Compute and apply control inputs
    # ---------------------------------------------------------
    # goal_points is (2, N) so it's passed directly to the SI controller
    dxi = single_integrator_position_controller(x_si, goal_points)

    for k in range(N):
        if at_goal_checker(x_si[:, k:k+1], goal_points[:, k:k+1])[0]:
            dxi[:, k] = 0.0

    dxi = si_barrier_cert(dxi, x_si)
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

# =========================================================
# SAVE DATA
# =========================================================
# Stack list of (K, N) frames into (K, N, T) arrays
np.save('vicon_pose_array.npy',      np.stack(vicon_list,      axis=2))
np.save('imu_orientation_array.npy', np.stack(imu_orient_list, axis=2))
np.save('encoder_array.npy',         np.stack(encoder_list,    axis=2))
np.save('gyro_array.npy',            np.stack(gyro_list,       axis=2))
np.save('acc_array.npy',             np.stack(acc_list,        axis=2))
np.save('mag_array.npy',             np.stack(mag_list,        axis=2))

# =========================================================
# DEBUG REPORT
# =========================================================
r.debug()