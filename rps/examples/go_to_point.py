import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Instantiate Robotarium object
N = 5
#initial_conditions = np.array(np.mat('1 0.5 -0.5; 0 0 0; 0 0 0'))
#r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, update_time=0.1)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=False)

# Define goal points by removing orientation from poses
goal_points = generate_initial_conditions(N)

# Create barrier certificates to avoid collision
si_barrier_cert = create_single_integrator_barrier_certificate(N)

# define x initially
x = r.get_poses()
r.step()

# While the number of robots at the required poses is less
# than N...
while (np.size(at_pose(x, goal_points, rotation_error=100)) != N):

    # Get poses of agents
    x = r.get_poses()
    x_si = x[:2, :]

    # Create single-integrator control inputs
    dxi = single_integrator_position_controller(x_si, goal_points[:2, :], magnitude_limit=0.15)

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), single_integrator_to_unicycle2(dxi, x))
    # Iterate the simulation
    r.step()
