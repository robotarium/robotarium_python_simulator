import rps.robotarium as robotarium
import rps.utilities.graph as graph
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Instantiate Robotarium object
N = 5
r = robotarium.Robotarium(number_of_agents=N, show_figure=True, save_data=True, update_time=1)

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
    dxi = single_integrator_position_controller(x_si, goal_points[:2, :], magnitude_limit=0.08)

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), single_integrator_to_unicycle2(dxi, x))
    # Iterate the simulation
    r.step()

# Always call this function at the end of your scripts!  It will accelerate the
# execution of your experiment
r.call_at_scripts_end()
