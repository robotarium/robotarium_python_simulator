import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np

# Instantiate Robotarium object
N = 5
initial_conditions = np.array(np.mat('1 0.5 -0.5 0 0.28; 0.8 -0.3 -0.75 0.1 0.34; 0 0 0 0 0'))
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)

# The robots will never reach their goal points so set iteration number
iterations = 3000

# Define goal points outside of the arena
goal_points = np.array(np.mat('5 -5 5 -5 5; 5 5 -5 -5 5; 0 0 0 0 0'))

# Create unicycle position controller
si_position_controller = create_si_position_controller()

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Create barrier certificates to avoid collision
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()


# define x initially
x = r.get_poses()
r.step()

# While the number of robots at the required poses is less
# than N...
for i in range(iterations):

    # Get poses of agents
    x = r.get_poses()
    xi = uni_to_si_states(x)

    # Create single-integrator control inputs
    dxi = si_position_controller(xi, goal_points[:2, :])

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, xi)

    # Map the single integrator back to unicycle dynamics
    dxu = si_to_uni_dyn(dxi,x)

    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)

    # Iterate the simulation
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
