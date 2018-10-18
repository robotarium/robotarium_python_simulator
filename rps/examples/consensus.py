import rps.robotarium as robotarium
import rps.utilities.graph as graph
import rps.utilities.transformations as transformations
from rps.utilities.barrier_certificates import *

import numpy as np

# Instantiate Robotarium object
N = 10
r = robotarium.Robotarium(number_of_agents=N, show_figure=True, save_data=True, update_time=1)

# This consensus algorithm uses single-integrator dynamics, so we'll need these mappings.
si_to_uni_dyn, si_to_uni_states = transformations.create_single_integrator_to_unicycle()

# Generated a connected graph Laplacian (for a cylce graph).
L = graph.cycle_GL(N)

# Create a single-integrator barrier certificate to avoid collisions.
si_barrier_cert = create_single_integrator_barrier_certificate(N)

for k in range(1000):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = si_to_uni_states(x)

    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))

    # For each robot...
    for i in range(N):
        # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
        j = graph.topological_neighbors(L, i)
        # Compute the consensus algorithm
        si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1)

    # Use the barrier certificate to avoid collisions
    si_velocities = si_barrier_cert(si_velocities, x_si)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), si_to_uni_dyn(si_velocities, x))
    # Iterate the simulation
    r.step()

# Always call this function at the end of your script!!!!
r.call_at_scripts_end()
