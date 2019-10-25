import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np

# Instantiate Robotarium object
N = 12
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# How many iterations do we want (about N*0.033 seconds)
iterations = 1000

#Maximum linear speed of robot specified by motors
magnitude_limit = 0.15

# We're working in single-integrator dynamics, and we don't want the robots
# to collide or drive off the testbed.  Thus, we're going to use barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Generated a connected graph Laplacian (for a cylce graph).
L = cycle_GL(N)

for k in range(iterations):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))

    # For each robot...
    for i in range(N):
        # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
        j = topological_neighbors(L, i)
        # Compute the consensus algorithm
        si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1)


    #Keep single integrator control vectors under specified magnitude
    # Threshold control inputs
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

    # Use the barrier certificate to avoid collisions
    si_velocities = si_barrier_cert(si_velocities, x_si)

    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
