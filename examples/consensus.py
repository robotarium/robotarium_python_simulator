import robotarium
import utilities.graph as graph
import utilities.transformations as transformations
from utilities.barrier_certificates import *

import numpy as np

# Instantiate Robotarium object
N = 10
r = robotarium.Robotarium(number_of_agents=N)

si_to_uni_dyn, si_to_uni_states = transformations.create_single_integrator_to_unicycle()

L = graph.cycle_GL(N)

si_barrier_cert = create_single_integrator_barrier_certificate(N)

for k in range(1000):

    # Do stuff
    x = r.get_poses()
    x_si = si_to_uni_states(x)
    si_velocities = np.zeros((2, N))

    for i in range(N):
        j = graph.topological_neighbors(L, i)
        si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1)

    si_velocities = si_barrier_cert(si_velocities, x_si)

    r.set_velocities(np.arange(N), si_to_uni_dyn(si_velocities, x))
    r.step()

r.call_at_scripts_end()
