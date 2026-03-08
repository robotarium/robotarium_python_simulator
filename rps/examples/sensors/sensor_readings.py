import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np

# Instantiate Robotarium object
N = 1

# Create obstacles to be used for distance sensor simulation using line segments. Each obstacle 
# is defined by a set of two points (start and end) which are represented as columns in a 2x2 numpy array,
# and multiple obstacles are stacked along the first dimension, i.e., a (num_obstacles)x2x2 numpy array.
# Currently, there is no support for collision simulation with these obstacles; they are only used for 
# simulating distance sensor readings.
# NOTE: The rectangular boundary of the Robotarium arena is not treated as an obstacle for distance sensors
# by default. If you wish to include the boundary as obstacles, you can add them 
# (X-axis: [-1.6, 1.6], Y-axis: [-1.0, 1.0]) here.
obstacles = np.stack(([[-1.0, -1.0], [-0.5, 0.5]],
                       [[-0.6, -0.6], [-0.5, 0.5]],
                       [[0.2, 0.2], [-0.5, 0.5]],
                       [[1.0, 1.0], [-0.5, 0.5]],
                       [[-1.4, -1.4], [-0.9, 0.9]],
                       [[-1.4, 1.4], [0.9, 0.9]],
                       [[1.4, 1.4], [0.9, -0.9]],
                       [[1.4, -1.4], [-0.9, -0.9]]))

initial_conditions = np.array([[-1.2], [-0.6], [0]])

r = robotarium.Robotarium(number_of_robots=N, 
                          show_figure=True, 
                          sim_in_real_time=False, 
                          initial_conditions=initial_conditions, 
                          use_distance_sensors=True,
                          obstacles=obstacles)

# Generate goal points in a grid pattern
resolution_x = 0.4
resolution_y = 0.4
x_points = np.arange(-1.2, 1.2+resolution_x, resolution_x)
y_points = np.arange(-0.6, 0.6+resolution_y, resolution_y)
x_goals, y_goals = np.meshgrid(x_points, y_points)
y_goals[:, 1::2] *= -1  # Serpentine pattern

# Create single integrator position controller
single_integrator_position_controller = create_si_position_controller()

# Create barrier certificates to avoid collision
si_barrier_cert = create_single_integrator_barrier_certificate()

# Create safe boundary avoidance (i.e., avoid collisions with the Robotarium arena)
# si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Create mapping from unicycle states to single integrator states
_, uni_to_si_states = create_si_to_uni_mapping()

# Create mapping from single integrator velocity commands to unicycle velocity commands
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

# Initialize data storage
data = dict()
data['poses'] = np.zeros((N, 3, 1))
data['distances'] = np.zeros((N, 7, 1))
data['accelerations'] = np.zeros((N, 3, 1))
data['magnetic_fields'] = np.zeros((N, 3, 1))
data['orientations'] = np.zeros((N, 1, 1))
data['encoders'] = np.zeros((N, 2, 1))

# Main loop to navigate to each goal and record sensor data
index = 0
while index < x_goals.size:
    # Get poses of agents
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Update current goal
    goal = np.vstack((x_goals.T.flatten()[index], y_goals.T.flatten()[index]))
    
    # Check if the robot has reached the goal and record sensor data
    if np.size(at_pose(np.vstack((x_si,x[2,:])), np.vstack((goal.reshape(2,1), 0)), position_error=0.025, rotation_error=100)) == N:
        # The poses are already stored in x through r.get_poses()
        poses = x

        # r.get_distances() returns distances in shape (N_sensors, N_robots) where N_sensors=7.
        # Max range is 1.2 meters, and -1 indicates no obstacle detected within range.
        # The orientations of the sensors in robot frame are
        #          [[-0.04, 0.0,  0.04, 0.05, 0.04,   0.0,   -0.04],
        #           [ 0.04, 0.06, 0.05, 0.0,  -0.05, -0.06, -0.04],
        #           [ math.pi, math.pi/2, math.pi/4, 0.0,  -math.pi/4, -math.pi/2, -math.pi]])
        distances = r.get_distances()

        # r.get_accelerations() returns accelerations in shape (3, N_robots) in Robot frame.
        # r.get_magnetic_fields() returns magnetic field readings in shape (3, N_robots) in Robot frame.
        # The magnetic field reading is simulated based on the testbed x-axis being aligned with magnetic north, and the y-axis being aligned with magnetic east.
        # r.get_orientations() returns fused orientation (yaw) in shape (1, N_robots) in IMU frame.
        # The orientations are global to the robotarium where 0 degrees is aligned with the positive x-axis of the robotarium, and positive rotation is counterclockwise.
        #
        # All IMU measurements are simulated with additive Gaussian noise according to measured standard deviations
        # from the physical IMU sensors on the Robotarium testbed, and the axes are defined as follows:
        # 
        #   Accelerometer axes (in robot frame):
        #       X-axis: Forward
        #       Y-axis: Left
        #       Z-axis: Up
        #
        #   Gyrometer axes (in robot frame):
        #       X-axis: Roll
        #       Y-axis: Pitch
        #       Z-axis: Yaw
        #
        #   Magnetometer axes (in global frame):
        #       X-axis: X-axis of the robotarium, aligned with magnetic north
        #       Y-axis: Y-axis of the robotarium, aligned with magnetic east
        #       Z-axis: Perpendicular to the plane of the robotarium, pointing up
        accelerations = r.get_accelerations()
        magnetic_fields = r.get_magnetic_fields()
        orientations = r.get_orientations()

        # r.get_encoders() returns wheel encoder readings in shape (2, N_robots) as [left_wheel; right_wheel] in ticks.
        # There are 28 ticks per revolution, and the motor gear ratio is 100.37:1, resulting in approximately
        # 2810 ticks per wheel revolution.
        # Differential drive geometry parameters:
        # Wheel radius: 0.016 m, Wheelbase: 0.105 m
        encoders = r.get_encoders()

        # Record data
        data['poses'] = np.dstack((data['poses'], poses.T.reshape(N, 3, 1)))
        data['distances'] = np.dstack((data['distances'], distances.T.reshape(N, 7, 1)))
        data['accelerations'] = np.dstack((data['accelerations'], accelerations.T.reshape(N, 3, 1)))
        data['magnetic_fields'] = np.dstack((data['magnetic_fields'],magnetic_fields.T.reshape(N, 3, 1)))
        data['orientations'] = np.dstack((data['orientations'], orientations.T.reshape(N, 1, 1)))
        data['encoders'] = np.dstack((data['encoders'], encoders.T.reshape(N, 2, 1)))

        # Increment goal index
        index += 1
    
    # Create single-integrator control inputs
    dxi = single_integrator_position_controller(x_si, goal)

    # Create safe control inputs (i.e., no collisions with other robots)
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)

    # Iterate the simulation
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
