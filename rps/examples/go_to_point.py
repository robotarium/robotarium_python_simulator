import robotarium
import utilities.graph as graph
from utilities.transformations import *
from utilities.barrier_certificates import *
from utilities.misc import *
from utilities.controllers import *

import numpy as np
import time

# Instantiate Robotarium object
N = 5
r = robotarium.Robotarium(number_of_agents=N)

# Define goal points by removing orientation from poses
goal_points = generate_initial_conditions(N)

# Create barrier certificates to avoid collision
si_barrier_cert = create_single_integrator_barrier_certificate(N)

# define x initially
x = r.get_poses()
r.step()

# While the number of robots at the required poses is less
# than N...
while(np.size(at_pose(x, goal_points, rotation_error=100)) != N):

    # Get poses of agents
    x = r.get_poses()
    x_si = x[:2, :]

    # Single integrator velocities
    dxi = single_integrator_position_controller(x_si, goal_points[:2, :], magnitude_limit=0.08)

    # Create safe control inputs
    dxi = si_barrier_cert(dxi, x_si)

    r.set_velocities(np.arange(N), single_integrator_to_unicycle2(dxi, x))
    r.step()
